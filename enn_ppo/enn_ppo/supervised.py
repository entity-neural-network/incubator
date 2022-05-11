import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Tuple

import hyperstate
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from entity_gym.env.vec_env import VecActionMask
from entity_gym.ragged_dict import RaggedActionDict, RaggedBatchDict
from entity_gym.serialization import Trace
from entity_gym.serialization.sample_loader import Episode, MergedSamples
from entity_gym.simple_trace import Tracer
from ragged_buffer import RaggedBufferBool, RaggedBufferF32, RaggedBufferI64
from torch.optim import AdamW

from rogue_net.rogue_net import RogueNet, RogueNetConfig


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters

    Attributes:
        lr: learning rate
        anneal_lr: anneal learning rate
        max_grad_norm: max gradient norm
        batch_size: batch size
    """

    lr: float = 1e-4
    anneal_lr: bool = True
    max_grad_norm: float = 100.0
    batch_size: int = 512


@dataclass
class WandbConfig:
    """W&B tracking settings.

    Args:
        track: whether to track metrics to W&B
        project_name: the wandb's project name
        entity: the entity (team) of wandb's project
    """

    track: bool = False
    project_name: str = "enn-bc"
    entity: str = "entity-neural-network"


@dataclass
class Config:
    """Supervised training configuration.

    Args:
        optim: optimizer configuration
        wandb: wandb tracking settings
        model: transformer network hyperparameters
        dataset_path: file to load training/test dataset from
        epochs: number of epochs to train for
        loss_fn: loss function ("kl" or "mse")
        log_interval: print out loss every log_interval steps
        fast_eval_interval: interval at which to evaluate with subset of test data
        fast_eval_samples: number of samples to use in fast evaluation
    """

    optim: OptimizerConfig
    wandb: WandbConfig
    model: RogueNetConfig
    dataset_path: str
    epochs: int = 10
    loss_fn: Literal["kl", "mse"] = "mse"
    log_interval: int = 10
    fast_eval_interval: int = 32768
    fast_eval_samples: int = 8192


@dataclass
class DataSet:
    entities: RaggedBatchDict[np.float32]
    visible: RaggedBatchDict[np.bool_]
    actions: RaggedBatchDict[np.int64]
    logprobs: RaggedBatchDict[np.float32]
    masks: RaggedActionDict
    logits: Optional[RaggedBatchDict[np.float32]]
    batch_size: int
    frames: int

    permutation: Optional[npt.NDArray[np.int64]] = None

    @classmethod
    def from_merged_samples(
        cls, merged_samples: MergedSamples, batch_size: int
    ) -> "DataSet":
        frames = (merged_samples.frames // batch_size) * batch_size
        if frames == 0:
            frames = merged_samples.frames
            batch_size = frames
        return cls(
            entities=merged_samples.entities,
            visible=merged_samples.visible,
            actions=merged_samples.actions,
            logprobs=merged_samples.logprobs,
            masks=merged_samples.masks,
            logits=merged_samples.logits,
            batch_size=batch_size,
            frames=frames,
        )

    @classmethod
    def from_episodes(cls, episodes: List[Episode], batch_size: int) -> "DataSet":
        entities = RaggedBatchDict(RaggedBufferF32)
        visible = RaggedBatchDict(RaggedBufferBool)
        actions = RaggedBatchDict(RaggedBufferI64)
        logprobs = RaggedBatchDict(RaggedBufferF32)
        logits = RaggedBatchDict(RaggedBufferF32)
        masks = RaggedActionDict()
        for e in episodes:
            entities.extend(e.entities)
            visible.extend(e.visible)
            actions.extend(e.actions)
            logprobs.extend(e.logprobs)
            masks.extend(e.masks)
            logits.extend(e.logits)

        return cls.from_merged_samples(
            MergedSamples(
                entities=entities,
                visible=visible,
                actions=actions,
                logprobs=logprobs,
                masks=masks,
                logits=logits,
                frames=next(iter(entities.buffers.values())).size0(),
            ),
            batch_size=batch_size,
        )

    @property
    def nbatch(self) -> int:
        return self.frames // self.batch_size

    def batch(
        self, n: int
    ) -> Tuple[
        Dict[str, RaggedBufferF32],
        Dict[str, RaggedBufferBool],
        Dict[str, RaggedBufferI64],
        Dict[str, RaggedBufferF32],
        Dict[str, VecActionMask],
        Optional[Dict[str, RaggedBufferF32]],
    ]:
        if self.permutation is None:
            indices = np.arange(n * self.batch_size, (n + 1) * self.batch_size)
        else:
            indices = self.permutation[n * self.batch_size : (n + 1) * self.batch_size]
        return (
            self.entities[indices],
            self.visible[indices],
            self.actions[indices],
            self.logprobs[indices],
            self.masks[indices],
            self.logits[indices] if self.logits is not None else None,
        )

    def shuffle(self) -> None:
        self.permutation = np.random.permutation(self.frames)

    def deterministic_shuffle(self) -> None:
        stepsize = int(math.sqrt(self.frames))
        perm = np.zeros(self.frames, dtype=np.int64)
        index = 0
        offset = 0
        for i in range(self.frames):
            perm[i] = index
            index += stepsize
            if index >= self.frames:
                offset += 1
                index = offset
        self.permutation = perm


def load_dataset(filepath: str, batch_size: int) -> Tuple[Trace, DataSet, DataSet]:
    trace = Trace.deserialize(open(filepath, "rb").read(), progress_bar=True)

    # episodes = trace.episodes(progress_bar=True)
    # test_episodes = max(len(episodes) // 20, 1) * 2
    # test = episodes[-test_episodes:]
    # train = episodes[:-test_episodes]
    # trainds = DataSet.from_episodes(train, batch_size=batch_size)
    # testds = DataSet.from_episodes(test, batch_size=batch_size)
    train, test = trace.train_test_split(test_frac=0.1, progress_bar=True)
    trainds = DataSet.from_merged_samples(train, batch_size=batch_size)
    testds = DataSet.from_merged_samples(test, batch_size=batch_size)
    print(f"{trainds.frames} training samples")
    print(f"{testds.frames} test samples")
    return trace, trainds, testds


def compute_loss(
    model: RogueNet,
    batch: int,
    ds: DataSet,
    loss_fn: Literal["kl", "mse"],
    tracer: Tracer,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    entities, visible, actions, logprobs, masks, logits = ds.batch(batch)
    _, newlogprob, entropy, _, aux, newlogits = model.get_action_and_auxiliary(
        entities=entities,
        visible=visible,
        action_masks=masks,
        prev_actions=actions,
        tracer=tracer,
    )
    loss = torch.tensor(0.0, device=device)
    for actname, target_logprob in logprobs.items():
        # Create normalized distributions
        if loss_fn == "kl":
            if logits is None:
                logprob = newlogprob[actname]
                dist = torch.cat([logprob, (1 - logprob.exp()).log()], dim=1)
                target = torch.tensor(target_logprob.as_array(), device=device)
                target_dist = torch.cat([target.exp(), 1 - target.exp()], dim=1)
                loss += F.kl_div(
                    dist,
                    target_dist,
                )
            else:
                dist = newlogits[actname]
                target_dist = torch.tensor(logits[actname].as_array(), device=device)
                loss += F.kl_div(
                    dist,
                    target_dist.exp(),
                )
        elif loss_fn == "mse":
            logprob = newlogprob[actname]
            target = torch.tensor(target_logprob.as_array().squeeze(-1), device=device)
            loss += F.mse_loss(
                logprob.masked_fill(mask=logprob == float("-inf"), value=0.0),
                target.masked_fill(mask=target == float("-inf"), value=0.0),
            )
    return loss, sum(e.mean().item() for _, e in entropy.items())


def train(
    cfg: Config,
    model: RogueNet,
    trainds: DataSet,
    testds: DataSet,
    device: torch.device,
) -> None:
    tracer = Tracer(cuda=device == "cuda")

    optimizer = AdamW(model.parameters(), lr=cfg.optim.lr)
    for epoch in range(cfg.epochs + 1):
        test_loss = 0.0
        for test_batch in range(testds.nbatch):
            loss, _ = compute_loss(
                model, test_batch, testds, cfg.loss_fn, tracer, device
            )
            test_loss += loss.item()
        test_loss /= testds.nbatch
        print(f"Test loss {test_loss:.4f}")
        if cfg.wandb.track:
            wandb.log(
                {
                    "test_loss": test_loss,
                    "epoch": epoch,
                    "frame": epoch * trainds.frames,
                }
            )
        if epoch == cfg.epochs:
            break

        trainds.shuffle()
        model.train()
        for batch in range(trainds.nbatch):
            frame = batch * trainds.batch_size + epoch * trainds.frames
            if frame % cfg.fast_eval_interval == 0:
                test_loss = 0.0
                for test_batch in range(cfg.fast_eval_samples // testds.batch_size):
                    loss, _ = compute_loss(
                        model,
                        test_batch,
                        testds,
                        cfg.loss_fn,
                        tracer,
                        device,
                    )
                    test_loss += loss.item()
                test_loss /= cfg.fast_eval_samples // testds.batch_size
                print(f"Fast test loss {test_loss:.4f}")
                if cfg.wandb.track:
                    wandb.log(
                        {
                            "fast_test_loss": test_loss,
                            "epoch": epoch,
                            "frame": frame,
                        }
                    )

            optimizer.zero_grad()
            if cfg.optim.anneal_lr:
                frac = 1.0 - frame / (cfg.epochs * trainds.frames)
                lrnow = frac * cfg.optim.lr
                optimizer.param_groups[0]["lr"] = lrnow
            else:
                lrnow = cfg.optim.lr

            loss, entropy = compute_loss(
                model, batch, trainds, cfg.loss_fn, tracer, device
            )
            loss.backward()
            gradnorm = nn.utils.clip_grad_norm_(
                model.parameters(), cfg.optim.max_grad_norm
            )
            optimizer.step()
            if batch % cfg.log_interval == 0:
                print(
                    f"Epoch {epoch}/{cfg.epochs}, Batch {batch}/{trainds.nbatch}, Loss {loss.item():.4f}, Entropy {entropy:.4f}"
                )
            if cfg.wandb.track:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_entropy": entropy,
                        "gradnorm": gradnorm,
                        "epoch": epoch,
                        "frame": frame,
                        "lr": lrnow,
                    },
                )


@hyperstate.command(Config)
def main(cfg: Config) -> None:
    """Trains a supervised model on samples recorded from an entity-gym environment."""
    trace, traindata, testdata = load_dataset(cfg.dataset_path, cfg.optim.batch_size)
    if testdata.frames < cfg.fast_eval_samples:
        print(
            f"WARNING: fast_eval_samples {cfg.fast_eval_samples} is larger than test dataset {testdata.frames}"
        )
        cfg.fast_eval_samples = testdata.frames
    testdata.deterministic_shuffle()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # TODO: compute input normalization once at the beginning and then freeze it
    model = RogueNet(
        cfg.model,
        obs_space=trace.obs_space,
        action_space=trace.action_space,
    ).to(device)

    if cfg.wandb.track:
        config = asdict(cfg)
        run_name = None
        if os.path.exists("/xprun/info/config.ron"):
            import xprun  # type: ignore

            xp_info = xprun.current_xp()
            config["name"] = xp_info.xp_def.name
            config["base_name"] = xp_info.xp_def.base_name
            config["id"] = xp_info.id
            run_name = xp_info.xp_def.name
        wandb.init(
            project=cfg.wandb.project_name,
            entity=cfg.wandb.entity,
            config=asdict(cfg),
            name=run_name,
        )
        wandb.watch(model)

    assert cfg.loss_fn == "kl" or cfg.loss_fn == "mse"

    train(
        cfg=cfg,
        model=model,
        trainds=traindata,
        testds=testdata,
        device=device,
    )


if __name__ == "__main__":
    main()
