# ENN-PPO

PPO implementation compatible with Entity Gym.

Example usage for training a policy in one of the entity-gym example environments:

```bash
poetry run python enn_ppo/enn_ppo/train.py
```

To get a list of available hyperparameters, run:

```python
poetry run python enn_ppo/enn_ppo/train.py --hps-info
```

To use ENN-PPO with a [custom entity gym environment](/entity_gym/TUTORIAL.md), you can use something like the following code:

```python
import hyperstate
from enn_ppo.config import TrainConfig
from enn_ppo.train import train
from custom_env import CustomEnv


@hyperstate.command(TrainConfig)
def main(cfg: TrainConfig) -> None:
    train(cfg=cfg, env_cls=CustomEnv)

if __name__ == "__main__":
    main()
```

To run behavioral cloning on recorded samples:

```bash
# Download data (261MB)
# Larger 5GB file with 1M samples: https://www.dropbox.com/s/o7jf4r7m0xtm80p/enhanced250m-1m-v2.blob?dl=1
wget 'https://www.dropbox.com/s/es84ml3wltxdmnh/enhanced250m-60k.blob?dl=1' -O enhanced250m-60k.blob
# Run training
poetry run python enn_ppo/enn_ppo/supervised.py dataset_path=enhanced250m-60k.blob optim.batch_size=256 fast_eval_samples=256
```
