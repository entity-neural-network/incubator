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


def test_relpos_encoding() -> None:
    # poetry run python enn_ppo/enn_ppo/train.py --gym-id=FloorIsLava --total-timesteps=5000 --num-envs=64 --processes=1 --d-model=16 --n-layer=2 --num-steps=2 --num-minibatches=4 --ent-coef=0.3 --anneal-entropy --cuda=False --relpos-encoding='{"extent": [1, 1], "position_features": ["x", "y"]}' --learning-rate=0.01
    args = train.parse_args(
        [
            "--gym-id=FloorIsLava",
            "--total-timesteps=10000",
            "--num-envs=64",
            "--processes=1",
            "--d-model=16",
            "--n-layer=2",
            "--num-steps=2",
            "--num-minibatches=4",
            "--ent-coef=0.4",
            "--anneal-entropy",
            "--cuda=False",
            "--learning-rate=0.02",
            '--relpos-encoding={"extent": [1, 1], "position_features": ["x", "y"], "per_entity_values": true}',
        ]
    )

    meanrew = train.train(args)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.99

    args.relpos_encoding = None
    meanrew = train.train(args)
    print(f"Final mean reward: {meanrew}")
    assert meanrew < 0.2


def test_asymetric_relpos_encoding() -> None:
    # poetry run python enn_ppo/enn_ppo/train.py --gym-id=FloorIsLava --total-timesteps=5000 --num-envs=64 --processes=1 --d-model=16 --n-layer=2 --num-steps=2 --num-minibatches=4 --ent-coef=0.3 --anneal-entropy --cuda=False --relpos-encoding='{"extent": [1, 1], "position_features": ["x", "y"]}' --learning-rate=0.01
    args = train.parse_args(
        [
            "--gym-id=FloorIsLava",
            "--total-timesteps=2000",
            "--num-envs=64",
            "--processes=1",
            "--d-model=16",
            "--n-layer=2",
            "--num-steps=1",
            "--num-minibatches=4",
            "--ent-coef=0.0",
            "--anneal-entropy",
            "--cuda=False",
            "--learning-rate=0.02",
            '--relpos-encoding={"extent": [5, 1], "position_features": ["x", "y"], "per_entity_values": true}',
        ]
    )

    meanrew = train.train(args)
    print(f"Final mean reward: {meanrew}")
    assert meanrew >= 0.15
