# Gym-microrts Environment

[Gym-microrts](https://github.com/vwxyzjn/gym-microrts) is The Reinforcement Learning environment for the popular Real-time Strategy game simulator μRTS.

## Get started

Prerequisites:
* Java 8.0+

Run an experiment locally

```bash
poetry run python enn_zoo/enn_zoo/train.py \
    env.id=GymMicrorts \
    rollout.num_envs=4 \
    total_timesteps=100000 \
    rollout.steps=256
```

Run a different map. See [here](https://github.com/vwxyzjn/microrts/tree/master/maps/16x16) for a full list of maps.
```bash
poetry run python enn_zoo/enn_zoo/train.py \
    env.id=GymMicrorts \
    rollout.num_envs=4 \
    total_timesteps=100000 \
    rollout.steps=256 \
    env.kwargs="{\"map_path\": \"maps/16x16/basesWorkers16x16.xml\"}"
```


Run a tracked experiment

```bash
poetry run python enn_zoo/enn_zoo/train.py \
    env.id=GymMicrorts \
    rollout.num_envs=24 \
    total_timesteps=100000 \
    rollout.steps=256 \
    track=true
```

Run a tracked experiment with video tracking

```
poetry run python enn_zoo/enn_zoo/train.py \
    env.id=GymMicrorts \
    rollout.num_envs=1 \
    total_timesteps=10000 \
    rollout.steps=256 \
    eval.capture_videos=true \
    eval.steps=500 \
    rollout.num_envs=1 \
    eval.interval=10000 \
    track=true
```
