from enn_ppo import train


def test_multi_armed_bandit() -> None:
    args = train.parse_args(
        [
            "--total-timesteps=500",
            "--ent-coef=0",
            "--gym-id=MultiArmedBandit",
            "--num-steps=16",
            "--gamma=0.5",
            "--cuda=False",
            "--hidden-size=16",
            "--learning-rate=0.05",
        ]
    )
    meanrew = train.train(args)
    print(f"Final mean reward: {meanrew}")
    assert meanrew > 0.99
