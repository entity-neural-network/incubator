# Gym-microrts Environment

[Gym-microrts](https://github.com/vwxyzjn/gym-microrts) is The Reinforcement Learning environment for the popular Real-time Strategy game simulator μRTS.

## Get started

Prerequisites:
* Java 8.0+

Run an experiment locally

```bash
poetry run python enn_ppo/enn_ppo/train.py \
    env.id=GymMicrorts \
    env.num_envs=4 \
    total_timesteps=100000 \
    env.num_steps=256
```

Run a different map. See [here](https://github.com/vwxyzjn/microrts/tree/master/maps/16x16) for a full list of maps.
```bash
poetry run python enn_ppo/enn_ppo/train.py \
    env.id=GymMicrorts \
    env.num_envs=4 \
    total_timesteps=100000 \
    env.num_steps=256 \
    env.kwargs="{\"map_path\": \"maps/16x16/basesWorkers16x16.xml\"}"
```


Run a tracked experiment

```bash
poetry run python enn_ppo/enn_ppo/train.py \
    env.id=GymMicrorts \
    env.num_envs=24 \
    total_timesteps=100000 \
    env.num_steps=256 \
    track=true
poetry run python enn_ppo/enn_ppo/train.py \
    env.id=GymMicrorts \
    env.num_envs=4 \
    total_timesteps=1000000 \
    env.num_steps=256 \
    env.kwargs="{\"map_path\": \"maps/16x16/basesWorkers16x16.xml\"}" \
    track=true
```
