from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Dict
from enn_ppo.simple_trace import Tracer
from entity_gym.environment.vec_env import VecActionMask
from entity_gym.serialization import Trace
from entity_gym.serialization.sample_loader import Episode
from ragged_buffer import RaggedBufferF32, RaggedBufferI64
from enn_ppo.train import RaggedActionDict, RaggedBatchDict
from rogue_net.actor import AutoActor
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import numpy.typing as npt
import wandb


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
            if torch.any(logprob == float("-inf")):
                __import__("ipdb").set_trace()
    return loss, sum(e.mean().item() for _, e in entropy.items())


def train(
    model: AutoActor,
    trainds: DataSet,
    testds: DataSet,
    epochs: int,
    lr: float,
    anneal_lr: bool,
    max_grad_norm: float,
    loss_fn: Literal["kl", "mse"],
    fast_eval_interval: int,
    fast_eval_samples: int,
    log_interval: int,
    track: bool,
    device: torch.device,
) -> None:
    tracer = Tracer(cuda=device == "cuda")

    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs + 1):
        test_loss = 0.0
        for test_batch in range(testds.nbatch):
            loss, _ = compute_loss(model, test_batch, testds, loss_fn, tracer, device)
            test_loss = loss.item()
        print(f"Test loss {test_loss:.4f}")
        if track:
            wandb.log(
                {
                    "test_loss": test_loss,
                    "epoch": epoch,
                    "frame": epoch * trainds.frames,
                }
            )
        if epoch == epochs:
            break

        trainds.shuffle()
        model.train()
        for batch in range(trainds.nbatch):
            frame = batch * trainds.batch_size + epoch * trainds.frames
            optimizer.zero_grad()
            if anneal_lr:
                frac = 1.0 - frame / (epochs * trainds.frames)
                lrnow = frac * lr
                optimizer.param_groups[0]["lr"] = lrnow
            else:
                lrnow = lr

            loss, entropy = compute_loss(model, batch, trainds, loss_fn, tracer, device)
            loss.backward()
            gradnorm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if batch % log_interval == 0:
                print(
                    f"Epoch {epoch}/{epochs}, Batch {batch}/{trainds.nbatch}, Loss {loss.item():.4f}, Entropy {entropy:.4f}"
                )
            if frame % fast_eval_interval == 0:
                test_loss = 0.0
                for test_batch in range(fast_eval_samples // testds.batch_size):
                    loss, _ = compute_loss(
                        model,
                        test_batch,
                        testds,
                        loss_fn,
                        tracer,
                        device,
                    )
                    test_loss += loss.item()
                test_loss /= fast_eval_samples // testds.batch_size
                print(f"Fast test loss {test_loss:.4f}")
                if track:
                    wandb.log(
                        {
                            "fast_test_loss": test_loss,
                            "epoch": epoch,
                            "frame": frame,
                        }
                    )
            if track:
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


@click.command()
@click.option("--epochs", default=10, help="Number of epochs to train for")
@click.option("--batch-size", default=512, help="Batch size")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option("--anneal-lr/--no-anneal-lr", default=True, help="Anneal learning rate")
@click.option("--max-grad-norm", default=100.0, help="Max gradient norm")
@click.option("--loss-fn", default="mse", help='Loss function ("kl" or "mse")')
@click.option(
    "--fast-eval-interval",
    default=32768,
    help="Interval at which to evaluate with subset of test data.",
)
@click.option(
    "--fast-eval-samples",
    default=8192,
    help="Number of samples to use in fast evaluation.",
)
@click.option("--log-interval", default=10, help="Log interval")
@click.option("--filepath", default="enhanced250m-b.blob", help="Filepath to load from")
@click.option("--track/--no-track", default=False, help="Whether to log metrics to W&B")
@click.option(
    "--wandb-project-name", type=str, default="enn-bc", help="the wandb's project name"
)
@click.option(
    "--wandb-entity",
    type=str,
    default="entity-neural-network",
    help="the entity (team) of wandb's project",
)
def main(
    epochs: int,
    batch_size: int,
    lr: float,
    anneal_lr: bool,
    max_grad_norm: float,
    loss_fn: str,
    fast_eval_interval: int,
    fast_eval_samples: int,
    log_interval: int,
    filepath: str,
    track: bool,
    wandb_project_name: str,
    wandb_entity: str,
) -> None:
    trace, traindata, testdata = load_dataset(filepath, batch_size)
    testdata.deterministic_shuffle()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # TODO: compute input normalization once at the beginning and then freeze it
    model = AutoActor(
        obs_space=trace.obs_space,
        action_space=trace.action_space,
        d_model=256,
        n_head=4,
        n_layer=2,
    ).to(device)

    if track:
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "anneal_lr": anneal_lr,
                "max_grad_norm": max_grad_norm,
                "loss_fn": loss_fn,
                "log_interval": log_interval,
                "filepath": filepath,
            },
        )
        wandb.watch(model)

    assert loss_fn == "kl" or loss_fn == "mse"

    train(
        model=model,
        trainds=traindata,
        testds=testdata,
        epochs=epochs,
        lr=lr,
        anneal_lr=anneal_lr,
        max_grad_norm=max_grad_norm,
        loss_fn=loss_fn,  # type: ignore
        fast_eval_interval=fast_eval_interval,
        fast_eval_samples=fast_eval_samples,
        log_interval=log_interval,
        track=track,
        device=device,
    )


if __name__ == "__main__":
    main()
