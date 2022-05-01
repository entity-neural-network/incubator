# Quick Start Guide

This tutorial will guide you through the process of creating a simple entity gym environment.

## 1. Installation

First, install the entity_gym package:

```bash
pip install entity_gym
```

## 2. Creating an empty environment

Create a new file `treasure_hunt.py` with the following contents:

```python
from typing import Dict, Mapping
from entity_gym.environment import *

class TreasureHunt(Environment):
    def __init__(self) -> None:
        pass

    def obs_space(self) -> ObsSpace:
        return ObsSpace()

    def action_space(self) -> Dict[str, ActionSpace]:
        return {}

    def reset(self) -> Observation:
        return Observation.empty()

    def act(self, action: Mapping[ActionType, Action]) -> Observation:
        return Observation.empty()


if __name__ == "__main__":
    env = TreasureHunt()
    CliRunner(env).run()
```

The `Environment` class defines the interface that all entity gym environments must implement.
The `CliRunner` class is used to run the environment in a command line interface.
Try it out by running the following command:

```bash
python treasure_hunt.py
```

Since we haven't implemented any functionality for our environment, this won't do much yet. However, you should see something like the following output:

```
Environment: TreasureHunt

Step 0
Reward: 0
Total: 0
Entities
Press ENTER to continue, CTRL-C to exit
Step 1
Reward: 0
Total: 0
Entities
Press ENTER to continue, CTRL-C to exit^C
Exiting
```

## 2. Player position

Let's add some state to our environment to track the player's position:

```python
class TreasureHunt(Environment):
    def __init__(self) -> None:
        self.x_pos = 0
        self.y_pos = 0

    def obs_space(self) -> ObsSpace:
        return ObsSpace(global_features=["x_pos", "y_pos"])

    def reset(self) -> Observation:
        self.x_pos = 0
        self.y_pos = 0
        return self._observe()

    def _observe(self) -> Observation:
        return Observation(global_features=[self.x_pos, self.y_pos])

    def act(self, action: Mapping[ActionType, Action]) -> Observation:
        return self._observe()
 
    ...
```

If you run the environment again, you should now see it print out the player's position:

```bash
$ python treasure_hunt.py
x_pos: 0
y_pos: 0
```

## 3. Move action

Now that we have a player position, we can add an action that moves the player:

```python
class TreasureHunt(Environment):
    def action_space(self) -> Dict[str, ActionSpace]:
        return {
            "move": GlobalCategoricalActionSpace(
                categories=["up", "down", "left", "right"]
            )
        }

    def act(self, action: Mapping[ActionType, Action]) -> Observation:
        if action["move"].label == "up":
            self.y_pos += 1
        elif action["move"].label == "down":
            self.y_pos -= 1
        elif action["move"].label == "left":
            self.x_pos -= 1
        elif action["move"].label == "right":
            self.x_pos += 1
        return self._observe()
    
    ...
```

You can now move the player around:

```bash
$ python treasure_hunt.py
x_pos: 0
y_pos: 0
```



## 4. Traps and Treasure

TODO: text

```python
import random

class TreasureHunt(Environment):
    def __init__(self):
        self.x_pos = 0
        self.y_pos = 0
        self.traps = []
        self.treasure = [] 

    def _random_empty_pos(self) -> Tuple[int, int]:
        while True:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in self.traps and (x, y) not in self.treasure:
                return x, y

    def reset(self) -> Observation:
        self.x_pos = 0
        self.y_pos = 0
        self.traps = []
        for _ in range(5):
            self.traps.append(self._random_empty_pos())
        for _ in range(5):
            self.treasure.append(self._random_empty_pos())
        return self._observe()
```

TODO: text

```python
class TreasureHunt(Environment):
    def obs_space(self) -> ObsSpace:
        return ObsSpace(
            global_features=["x_pos", "y_pos"],
            entities={
                "Trap": Entity(features=["x_pos", "y_pos"]),
                "Treasure": Entity(features=["x_pos", "y_pos"]),
            }
        )

    def obs(self) -> Observation:
        return Observation(
            global_features=[self.x_pos, self.y_pos],
            entities=[
                Entity(features=[x, y], name="Trap")
                for x, y in self.traps
            ],
            entities=[
                Entity(features=[x, y], name="Treasure")
                for x, y in self.treasure
            ]
        )
```

