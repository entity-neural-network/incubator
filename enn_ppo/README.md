# ENN-PPO

PPO implementation compatible with Entity Gym.

Example usage:

```bash
poetry run python enn_ppo/enn_ppo/train.py
```

To run behavioral cloning on recorded samples:

```bash
# Download data (261MB)
# Larger 5GB file with 1M samples: https://www.dropbox.com/s/o7jf4r7m0xtm80p/enhanced250m-1m-v2.blob?dl=1
wget 'https://www.dropbox.com/s/es84ml3wltxdmnh/enhanced250m-60k.blob?dl=1' -O enhanced250m-60k.blob
# Run training
poetry run python enn_ppo/enn_ppo/supervised.py --batch-size=1024 --lr=0.001 --filepath=enhanced250m-1m-60m.blob --track
```