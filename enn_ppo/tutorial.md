
# Tutorial

In this tutorial, you will learn how to:
1. [Install](#installation) the enn_ppo package
2. [Train](#training) a neural network to solve an [entity gym](TODO-LINK) environment
3. [Accelerate](#positional-encoding) learning with [relative positional encoding](TODO-LINK)
4. [Create](#config-files) a hyperparameter config file
5. [Save and load](#checkpointing) a trained model and inspect its predictions

## Installation

To install the enn_ppo package, run the following command:

```bash
pip install enn_ppo
```

## Training

We are going to train a neural network to solve the [TreasureHunt environment](TODO-LINK).
Setting up a training script takes just a few lines of code:

```python
from enn_ppo.config import TrainConfig
from enn_ppo.train import State, initialize, train
from entity_gym.examples.xor import Xor
import hyperstate

@hyperstate.stateful_command(TrainConfig, State, initialize)
def main(state_manager: hyperstate.StateManager) -> None:
    train(state_manager=state_manager, env=Xor)

if __name__ == "__main__":
    main()
```

Assuming you have saved the above script in a file called `train.py`, you can now run training with the following command:

```bash
python train.py
```

You should see something like the following output:

```bash
  512/25000 | meanrew 1.17e-02 | explained_var  0.00 | entropy  1.39 | episodic_reward 2.50e-01 | episode_length 2.39e+01 | episodes 8 | fps 701
 1024/25000 | meanrew 7.81e-03 | explained_var  0.03 | entropy  1.38 | episodic_reward 2.00e+00 | episode_length 1.24e+02 | episodes 4 | fps 671
...
24064/25000 | meanrew 9.77e-03 | explained_var  0.24 | entropy  1.38 | episodic_reward 2.50e+00 | episode_length 4.96e+02 | episodes 2 | fps 720
24576/25000 | meanrew 7.81e-03 | explained_var  0.23 | entropy  1.37 | episodic_reward 1.11e+00 | episode_length 1.09e+02 | episodes 9 | fps 721
```

The "meanrew" column shows the average reward received by the agent on each step.
If training is successful, you should see this value going up.
The "episodic_reward" column shows the total reward received by the agent in each episode.

For this environment, the maximum reward that can achieved is 5.0.
As you can see, we're not reaching a very high reward yet.
We can speed up training and improve the performance of the model by changing some of the hyperparameters.
With the following command, you should be able to reach a reward of more than 3.0:

```bash
python train.py total_timesteps=100000 rollout.steps=32 rollout.num_envs=64 rollout.processes=4 optim.bs=512 optim.lr=0.005
```

TODO: link to hyperparameter reference + tuning guide

## Relative positional encoding

ENN-PPO supports a technique called [relative positional encoding](TODO-LINK).
The details of how this works is somewhat complicated, but the main thing to remember is that using relative positional encoding can greatly improve performance in environments where entities have some kind of spatial relationship.
We can enable relative positional encoding by setting two more hyperparameters:

```bash
python train.py total_timesteps=100000 rollout.steps=32 rollout.num_envs=64 rollout.processes=4 optim.bs=512 optim.lr=0.005 net.relpos_encoding.extent='[5,5]' net.relpos_encoding.position_features='["x_pos","y_pos"]'
```

With this configuration, you should be able to reach a reward of more than 4.5.

If you want to learn more about relative positional encoding, see the ["How to use relative positional encoding"](TODO-LINK) guide, which explains how to choose good settings for relative positional encoding.

## Config files

With the addition of relative positional encoding, our training command is starting to get a little bit unwieldy.
To simplify the command, make it easier to edit, and ensure we don't lose our hyperparameters, we can use a config file.
Create a file called `config.ron` with the following contents in [Rusty Object Notation](https://github.com/ron-rs/ron#rusty-object-notation).

```rust
Config(
    total_timesteps: 100000,
    rollout: (
        steps: 32,
        num_envs: 64,
        processes: 4,
    ),
    optim: (
        bs: 512,
        lr: 0.005,
    ),
    net: (
        relpos_encoding: (
            extent: [5,5],
            position_features: ["x_pos","y_pos"],
        )
    ),
)
```

We can now run the training script with the following command:

```bash
python train.py --config=config.ron
```

## Checkpointing

Now that know how to train a good model, we can save it for later use.
To this, we just need to supply the training script with a checkpoint directory using the `--checkpoint-dir` command line argument:


```bash
python train.py --config=config.ron --checkpoint-dir=checkpoints
```

If you run the above command, the training script should be saving a checkpoint to the `checkpoints` directory.
The checkpoint will be a directory containing 4 files:

```
checkpoints
└── latest-step000000098304
    ├── config.ron
    ├── state.agent.pickle
    ├── state.optimizer.pickle
    └── state.ron
```

We can now load this checkpoint.
Paste the following code into a Python console:

```python
from enn_ppo import load_checkpoint
from enn_ppo.agent import RogueNetAgent
from entity_gym.env import *
checkpoint = load_checkpoint("checkpoints/latest-step000000098304")
agent = RogueNetAgent(checkpoint.state.agent)
obs = Observation(
    global_features=[0, 0],
    features={
        "Trap": [[-5, 0], [-2, 0], [0, 3], [0, -4], [0, -3]],
        "Treasure": [[2, 0]],
    },
    done=True,
    reward=0.0,
    actions={"move": GlobalCategoricalActionMask()},
)
action, predicted_return = agent.act(obs)
```

In my case, the agent chose to move left towards treasure, and the probability of all other actions are much lower. Since training is a somewhat random process, you may get a different result:

```python
>>> action
{'move': GlobalCategoricalAction(index=3, choice='right', probs=array([[6.1148562e-04, 1.3557974e-04, 5.6600979e-06, 9.9924737e-01]],
      dtype=float32))}
```

My agent is also very confident that it will reach the treasure, predicting a return that is higher than what it could actually get from a single treasure:

```python
>>> predicted_return
1.1504299640655518
```

Let's what happens if we give the agent a somewhat more difficult task by placing some traps between the player and treasure:

```python
>>> obs.features["Treasure"] = [[-4, 0]]
>>> action, predicted_return = agent.act(obs)
>>> action
{'move': GlobalCategoricalAction(index=2, choice='left', probs=array([[7.1715019e-03, 1.7328954e-03, 9.9067581e-01, 4.1975660e-04]],
      dtype=float32))}
>>> predicted_return
0.9853743314743042
```

As we might have hoped, the agent now moves toward the left. It also assigns much higher probabiliy than before to moving up or down, which would allow it to avoid the trap.

Finally, let's use the `CliRunner` class to observe the agent in its natural environment:

```python
>>> from entity_gym.runner import CliRunner
>>> from entity_gym.examples.tutorial import TreasureHunt
>>> CliRunner(TreasureHunt(), agent).run()
```

The `CliRunner` executes the environment and shows the observations and agent predictions.
Simply press ENTER to accept the default action chosen by the agent and continue to the next step:

```
Environment: TreasureHunt
Global features: x_pos, y_pos
Entity Trap: x_pos, y_pos
Entity Treasure: x_pos, y_pos
Categorical move: up, down, left, right

Step 0
Reward: 0.0
Total: 0.0
Predicted return: 3.215e+00
Global features: x_pos=0, y_pos=0
Entities
0 Trap(x_pos=-3, y_pos=5)
1 Trap(x_pos=-1, y_pos=1)
2 Trap(x_pos=-5, y_pos=-4)
3 Trap(x_pos=3, y_pos=2)
4 Trap(x_pos=0, y_pos=3)
5 Treasure(x_pos=-5, y_pos=-1)
6 Treasure(x_pos=-3, y_pos=2)
7 Treasure(x_pos=3, y_pos=1)
8 Treasure(x_pos=-4, y_pos=3)
9 Treasure(x_pos=2, y_pos=5)
Choose move (0/up 11.3% 1/down 4.0% 2/left 16.4% 3/right 68.3%)
Step 1
Reward: 0.0
Total: 0.0
Predicted return: 3.132e+00
Global features: x_pos=1, y_pos=0
Entities
0 Trap(x_pos=-3, y_pos=5)
1 Trap(x_pos=-1, y_pos=1)
2 Trap(x_pos=-5, y_pos=-4)
3 Trap(x_pos=3, y_pos=2)
4 Trap(x_pos=0, y_pos=3)
5 Treasure(x_pos=-5, y_pos=-1)
6 Treasure(x_pos=-3, y_pos=2)
7 Treasure(x_pos=3, y_pos=1)
8 Treasure(x_pos=-4, y_pos=3)
9 Treasure(x_pos=2, y_pos=5)
Choose move (0/up 23.5% 1/down 1.0% 2/left 6.7% 3/right 68.8%)
```

This concludes the tutorial.
If you want to learn more about enn-ppo, you can check out TODO LINK TO DOCS.