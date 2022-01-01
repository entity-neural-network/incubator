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
            "--d-model=16",
            "--learning-rate=0.05",
            "--n-layer=0",
            "--processes=2",
        ]
    )
    meanrew = train.train(args)
    print(f"Final mean reward: {meanrew}")
    assert meanrew > 0.99


def test_minefield() -> None:
    args = train.parse_args(
        [
            "--gym-id=Minefield",
            "--total-timesteps=256",
            "--num-steps=16",
            "--cuda=False",
            "--d-model=16",
        ]
    )
    meanrew = train.train(args)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.0


def test_multi_snake() -> None:
    args = train.parse_args(
        [
            "--gym-id=MultiSnake",
            "--total-timesteps=256",
            "--num-steps=16",
            "--cuda=False",
            "--d-model=16",
        ]
    )
    meanrew = train.train(args)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.0


def test_not_hotdog() -> None:
    # --total-timesteps=500 --ent-coef=0 --gym-id=NotHotdog --num-steps=16 --gamma=0.5 --cuda=False --d-model=16 --learning-rate=0.05
    args = train.parse_args(
        [
            "--total-timesteps=500",
            "--ent-coef=0",
            "--gym-id=NotHotdog",
            "--num-steps=16",
            "--gamma=0.5",
            "--cuda=False",
            "--n-layer=1",
            "--d-model=16",
            "--learning-rate=0.005",
        ]
    )
    meanrew = train.train(args)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.99


def test_masked_count() -> None:
    # --total-timesteps=2000 --track --gym-id=Count --num-envs=16 --max-log-frequency=1000 --n-layer=1 --num-steps=1 --d-model=16 --learning-rate=0.01 --cuda=False
    args = train.parse_args(
        [
            "--total-timesteps=2000",
            "--gym-id=Count",
            "--num-envs=16",
            "--n-layer=1",
            "--num-steps=1",
            "--d-model=16",
            "--learning-rate=0.01",
            '--env-kwargs={"masked_choices": 2}',
            "--cuda=False",
        ]
    )
    meanrw = train.train(args)
    print(f"Final mean reward: {meanrw}")
    assert meanrw >= 0.99


def test_pick_matching_balls() -> None:
    args = train.parse_args(
        [
            "--gym-id=PickMatchingBalls",
            "--total-timesteps=256",
            "--num-steps=16",
            "--cuda=False",
            "--d-model=16",
        ]
    )
    meanrew = train.train(args)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.0


def test_cherry_pick() -> None:
    args = train.parse_args(
        [
            "--gym-id=CherryPick",
            "--total-timesteps=256",
            "--num-steps=16",
            "--cuda=False",
            "--d-model=16",
        ]
    )
    meanrew = train.train(args)
    print(f"Final mean reward: {meanrew}")
