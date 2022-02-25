from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Dict
from enn_ppo.simple_trace import Tracer
from entity_gym.environment.vec_env import VecActionMask
from entity_gym.serialization import Trace
from entity_gym.serialization.sample_loader import Episode
from ragged_buffer import RaggedBufferF32, RaggedBufferI64
from enn_ppo.train import RaggedActionDict, RaggedBatchDict
from rogue_net.actor import AutoActor
import click
from rogue_net.transformer import Transformer, TransformerConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import numpy.typing as npt
import wandb
import hyperstate


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters

    Args:
    :param lr: learning rate
    :param anneal_lr: anneal learning rate
    :param max_grad_norm: max gradient norm
    :param batch_size: batch size
    """

    lr: float = 1e-4
    anneal_lr: bool = True
    max_grad_norm: float = 100.0
    batch_size: int = 512


@dataclass
class WandbConfig:
    """W&B tracking settings.

    Args:
    :param track: whether to track metrics to W&B
    :param project_name: the wandb's project name
    :param entity: the entity (team) of wandb's project
    """

    track: bool = False
    project_name: str = "enn-bc"
    entity: str = "entity-neural-network"


@dataclass
class Config:
    """Supervised training configuration.

    Args:
    :param optim: optimizer configuration
    :wandb wandb: wandb tracking settings
    :model model: transformer network hyperparameters
    :param dataset_path: file to load training/test dataset from
    :param epochs: number of epochs to train for
    :param loss_fn: loss function ("kl" or "mse")
    :param log_interval: print out loss every log_interval steps
    :param fast_eval_interval: interval at which to evaluate with subset of test data
    :param fast_eval_samples: number of samples to use in fast evaluation
    """

    optim: OptimizerConfig
    wandb: WandbConfig
    model: TransformerConfig
    dataset_path: str
    epochs: int = 10
    loss_fn: Literal["kl", "mse"] = "mse"
    log_interval: int = 10
    fast_eval_interval: int = 32768
    fast_eval_samples: int = 8192


@dataclass
class DataSet:
    entities: RaggedBatchDict[np.float32]
    actions: RaggedBatchDict[np.int64]
    logprobs: RaggedBatchDict[np.float32]
    masks: RaggedActionDict
    logits: Optional[RaggedBatchDict[np.float32]]
    batch_size: int
    frames: int

    permutation: Optional[npt.NDArray[np.int64]] = None

    @classmethod
    def from_episodes(cls, episodes: List[Episode], batch_size: int) -> "DataSet":
        entities = RaggedBatchDict(RaggedBufferF32)
        actions = RaggedBatchDict(RaggedBufferI64)
        logprobs = RaggedBatchDict(RaggedBufferF32)
        logits = RaggedBatchDict(RaggedBufferF32)
        masks = RaggedActionDict()
        for e in episodes:
            entities.extend(e.entities)
            actions.extend(e.actions)
            logprobs.extend(e.logprobs)
            masks.extend(e.masks)
            logits.extend(e.logits)

        frames = (
            next(iter(entities.buffers.values())).size0() // batch_size
        ) * batch_size
        return DataSet(
            entities,
            actions,
            logprobs,
            masks,
            logits if len(logits.buffers) > 0 else None,
            batch_size=batch_size,
            frames=frames,
        )

    @property
    def nbatch(self) -> int:
        return self.frames // self.batch_size

    def batch(
        self, n: int
    ) -> Tuple[
        Dict[str, RaggedBufferF32],
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
            self.actions[indices],
            self.logprobs[indices],
            self.masks[indices],
            self.logits[indices] if self.logits is not None else None,
        )

    def shuffle(self) -> None:
        self.permutation = np.random.permutation(self.frames)

    def deterministic_shuffle(self) -> None:
        average_episode_length = self.frames // len(self.entities.buffers)
        perm = np.zeros(self.frames, dtype=np.int64)
        index = 0
        offset = 0
        for i in range(self.frames):
            perm[i] = index
            index += average_episode_length * 3
            if index >= self.frames:
                offset += 1
                index = offset
        self.permutation = perm


def load_dataset(filepath: str, batch_size: int) -> Tuple[Trace, DataSet, DataSet]:
    trace = Trace.deserialize(open(filepath, "rb").read(), progress_bar=True)
    episodes = trace.episodes(progress_bar=True)

    test_episodes = max(len(episodes) // 20, 1) * 2
    test = episodes[-test_episodes:]
    train = episodes[:-test_episodes]
    return (
        trace,
        DataSet.from_episodes(train, batch_size=batch_size),
        DataSet.from_episodes(test, batch_size=batch_size),
    )


def compute_loss(
    model: AutoActor,
    batch: int,
    ds: DataSet,
    loss_fn: Literal["kl", "mse"],
    tracer: Tracer,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    entities, actions, logprobs, masks, logits = ds.batch(batch)
    _, newlogprob, entropy, _, aux, newlogits = model.get_action_and_auxiliary(
        entities=entities,
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
    model: AutoActor,
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
            test_loss = loss.item()
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
    testdata.deterministic_shuffle()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # TODO: compute input normalization once at the beginning and then freeze it
    model = AutoActor(
        cfg.model,
        obs_space=trace.obs_space,
        action_space=trace.action_space,
    ).to(device)

    if cfg.wandb.track:
        config = asdict(cfg)
        run_name = None
        if os.path.exists("/xprun/info/config.ron"):
            import xprun

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
