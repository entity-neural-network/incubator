from enn_ppo import train
from enn_ppo.train import ExperimentConfig, OptimizerConfig, PPOConfig, EnvConfig
from rogue_net.relpos_encoding import RelposEncodingConfig
from rogue_net.transformer import TransformerConfig


def test_multi_armed_bandit() -> None:
    cfg = ExperimentConfig(
        total_timesteps=500,
        cuda=False,
        ppo=PPOConfig(ent_coef=0.0, gamma=0.5),
        env=EnvConfig(id="MultiArmedBandit", processes=2, num_steps=16),
        net=TransformerConfig(n_layer=0, d_model=16),
        optim=OptimizerConfig(lr=0.05, bs=16, update_epochs=4),
    )
    meanrew = train.train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew > 0.99 / 32


def test_minefield() -> None:
    cfg = ExperimentConfig(
        total_timesteps=256,
        cuda=False,
        net=TransformerConfig(d_model=16),
        env=EnvConfig(id="Minefield", num_steps=16),
        optim=OptimizerConfig(bs=64),
        ppo=PPOConfig(),
    )
    meanrew = train.train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.0


def test_multi_snake() -> None:
    cfg = ExperimentConfig(
        total_timesteps=256,
        cuda=False,
        net=TransformerConfig(d_model=16),
        env=EnvConfig(id="MultiSnake", num_steps=16),
        optim=OptimizerConfig(bs=64),
        ppo=PPOConfig(),
    )
    meanrew = train.train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.0


def test_not_hotdog() -> None:
    cfg = ExperimentConfig(
        total_timesteps=500,
        cuda=False,
        net=TransformerConfig(d_model=16, n_layer=1),
        env=EnvConfig(id="NotHotdog", num_steps=16),
        optim=OptimizerConfig(bs=16, lr=0.005),
        ppo=PPOConfig(ent_coef=0.0, gamma=0.5),
    )
    meanrew = train.train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.99


def test_masked_count() -> None:
    cfg = ExperimentConfig(
        total_timesteps=2000,
        cuda=False,
        net=TransformerConfig(d_model=16, n_layer=1),
        env=EnvConfig(
            id="Count", num_envs=16, num_steps=1, kwargs='{"masked_choices": 2}'
        ),
        optim=OptimizerConfig(bs=16, lr=0.01),
        ppo=PPOConfig(),
    )
    meanrw = train.train(cfg)
    print(f"Final mean reward: {meanrw}")
    assert meanrw >= 0.99


def test_pick_matching_balls() -> None:
    cfg = ExperimentConfig(
        total_timesteps=256,
        cuda=False,
        net=TransformerConfig(d_model=16),
        env=EnvConfig(id="PickMatchingBalls", num_steps=16),
        optim=OptimizerConfig(bs=64),
        ppo=PPOConfig(),
    )
    meanrew = train.train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.0


def test_cherry_pick() -> None:
    cfg = ExperimentConfig(
        total_timesteps=256,
        cuda=False,
        net=TransformerConfig(d_model=16),
        env=EnvConfig(id="CherryPick", num_steps=16),
        optim=OptimizerConfig(bs=64),
        ppo=PPOConfig(),
    )
    meanrew = train.train(cfg)
    print(f"Final mean reward: {meanrew}")


def test_relpos_encoding() -> None:
    cfg = ExperimentConfig(
        total_timesteps=10000,
        cuda=False,
        net=TransformerConfig(
            d_model=16,
            n_layer=2,
            relpos_encoding=RelposEncodingConfig(
                extent=[1, 1],
                position_features=["x", "y"],
                per_entity_values=True,
            ),
        ),
        env=EnvConfig(id="FloorIsLava", num_steps=2, num_envs=64),
        optim=OptimizerConfig(bs=32, lr=0.02),
        ppo=PPOConfig(
            ent_coef=0.4,
            anneal_entropy=True,
        ),
    )
    meanrew = train.train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.97

    cfg.net.relpos_encoding = None
    meanrew = train.train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew < 0.2


def test_asymmetric_relpos_encoding() -> None:
    cfg = ExperimentConfig(
        total_timesteps=3000,
        cuda=False,
        net=TransformerConfig(
            d_model=16,
            n_layer=2,
            relpos_encoding=RelposEncodingConfig(
                extent=[5, 1],
                position_features=["x", "y"],
                per_entity_values=True,
            ),
        ),
        env=EnvConfig(id="FloorIsLava", num_steps=1, num_envs=64),
        optim=OptimizerConfig(bs=16, lr=0.02),
        ppo=PPOConfig(
            ent_coef=0.1,
            anneal_entropy=True,
        ),
    )

    meanrew = train.train(cfg)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.15
