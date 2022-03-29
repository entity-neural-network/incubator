import os
from dataclasses import dataclass, field
from typing import Optional

import hyperstate

from rogue_net.rogue_net import RogueNetConfig


@dataclass
class EnvConfig:
    """Environment settings.

    Attributes:
        kwargs: JSON dictionary with keyword arguments for the environment
        id: the id of the environment
    """

    kwargs: str = "{}"
    id: str = "MoveToOrigin"


@dataclass
class RolloutConfig:
    """Settings for rollout phase of PPO.

    Attributes:
        steps: the number of steps to run in each environment per policy rollout
        num_envs: the number of parallel game environments
        processes: The number of processes to use to collect env data. The envs are split as equally as possible across the processes
    """

    steps: int = 128
    num_envs: int = 4
    processes: int = 1


@dataclass
class EvalConfig:
    """Evaluation settings

    Attributes:
        interval: number of global steps between evaluations

        capture_videos: if --eval-render-videos is set, videos will be recorded of the environments during evaluation
        capture_samples: if set, write the samples from evals to this file
        capture_logits: if --eval-capture-samples is set, record full logits of the agent
        capture_samples_subsample: only persist every nth sample, chosen randomly
        run_on_first_step: whether to run eval on step 0

        env: Settings for the eval environment. If not set, use same settings as rollouts.
        num_envs: The number of parallel game environments to use for evaluation. If not set, use same settings as rollouts.
        processes: The number of processes used to run the environment. If not set, use same settings as rollouts.

        opponent: Path to opponent policy to evaluate against.
        opponent_only: Don't evaluate the policy, but instead run the opponent against itself.

        codecraft_eval: if toggled, run evals with CodeCraft environment
        codecraft_eval_opponent: path to CodeCraft policy to evaluate against
        codecraft_only_opponent: run only the opponent, not the agent
    """

    steps: int
    interval: int

    num_envs: Optional[int] = None
    processes: Optional[int] = None
    env: Optional[EnvConfig] = None
    capture_videos: bool = False
    capture_samples: str = ""
    capture_logits: bool = True
    capture_samples_subsample: int = 1
    run_on_first_step: bool = True
    opponent: Optional[str] = None
    opponent_only: bool = False


@dataclass
class PPOConfig:
    """Proximal Policy Optimization settings.

    Attributes:
        gae: whether to use GAE for advantage computation
        gamma: the discount factor gamma
        gae_lambda: the lambda for the general advantage estimation
        norm_adv: whether to normalize advantages
        clip_coef: the surrogate clipping coefficient
        clip_vloss: whether or not to use a clipped loss for the value function, as per the paper
        ent_coef: coefficient of the entropy
        vf_coef: coefficient of the value function
        target_kl: the target KL divergence threshold
        anneal_entropy: whether to anneal the entropy coefficient
    """

    gae: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    target_kl: Optional[float] = None
    anneal_entropy: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer settings.

    Attributes:
        lr: the learning rate of the optimizer
        bs: the batch size of the optimizer
        micro_bs: if set, use gradient accumulation to split up batches into smaller microbatches
        weight_decay: the weight decay of the optimizer
        anneal_lr: whether to anneal the learning rate
        update_epochs: the K epochs to update the policy
        max_grad_norm: the maximum norm for the gradient clipping
    """

    lr: float = 2.5e-4
    bs: int = 128
    weight_decay: float = 0.0
    micro_bs: Optional[int] = None
    anneal_lr: bool = True
    update_epochs: int = 3
    max_grad_norm: float = 2.0


@dataclass
class TrainConfig(hyperstate.Versioned):
    """Experiment settings.

    Attributes:
        net: policy network configuration
        name: the name of the experiment
        seed: seed of the experiment
        total_timesteps: total timesteps of the experiments
        max_train_time: train for at most this many seconds
        torch_deterministic: if toggled, `torch.backends.cudnn.deterministic=False`
        vf_net: value function network configuration (if not set, policy and value function share the same network)
        cuda: if toggled, cuda will be enabled by default
        track: if toggled, this experiment will be tracked with Weights and Biases
        wandb_project_name: the wandb's project name
        wandb_entity: the entity (team) of wandb's project
        capture_samples: if set, write the samples to this file
        capture_logits: If --capture-samples is set, record full logits of the agent
        capture_samples_subsample: only persist every nth sample, chosen randomly
        trial: trial number of experiment spawned by hyperparameter tuner
        data_dir: Directory to save output from training and logging
        cuda_empty_cache: If toggled, empty the cuda cache after each optimizer step.
    """

    env: EnvConfig
    net: RogueNetConfig
    optim: OptimizerConfig
    ppo: PPOConfig
    rollout: RolloutConfig
    eval: Optional[EvalConfig] = None
    vf_net: Optional[RogueNetConfig] = None

    name: str = field(default_factory=lambda: os.path.basename(__file__).rstrip(".py"))
    seed: int = 1
    total_timesteps: int = 25000
    max_train_time: Optional[int] = None
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "enn-ppo"
    wandb_entity: str = "entity-neural-network"
    capture_samples: Optional[str] = None
    capture_logits: bool = False
    capture_samples_subsample: int = 1
    trial: Optional[int] = None
    data_dir: str = "."
    cuda_empty_cache: bool = False

    @classmethod
    def version(clz) -> int:
        return 0


if __name__ == "__main__":
    hyperstate.schema_evolution_cli(TrainConfig)
