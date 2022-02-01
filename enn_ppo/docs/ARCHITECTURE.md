# Internals

This document follows an observation from the MineSweeper environment through all the internals of entity-gym, enn-ppo, and rogue-net in excruciating detail.


## High-level overview

- The [Environment](#environment--observation) (entity-gym) provides a high-level abstraction for an environment.
- The [VecEnv](#vecenv--observation) (entity-gym) combines multiple environments and exposes a more efficient and lower-level batched representation of observations/actions.
- The [PPO training loop](#trainpy) (enn-ppo) keeps a sample buffer that combines observations from multiple steps.
- The policy is implemented by [RogueNet](#RogueNet) (rogue-net), a ragged batch transformer that takes lists of entities as input and outputs corresponding lists of actions.

## Environment / Observation

### Step 0

> reset method called

> env state

> pics

> observation

<details>
  <summary>Observations (click to expand)</summary>

```python
Observation(
    features={
        "Mine": array([
            [1.0, 2.0],
            [0.0, 4.0],
            [2.0, 4.0],
            [1.0, 1.0],
            [5.0, 1.0],
        ], dtype=float32),
        "Robot": array([
            [4.0, 0.0],
            [2.0, 2.0],
        ], dtype=float32),
    },
    actions={
        "Move": CategoricalActionMask(
            actor_types=["Robot"],
        )
    },
    done=False,
    reward=0.0,
    ids={"Robot": [("Robot", 0), ("Robot", 1)]},
    end_of_episode_info=None,
)
Observation(
    features={
        "Mine": array(
            [[5.0, 0.0], [4.0, 0.0], [2.0, 1.0], [1.0, 0.0], [5.0, 1.0]],
            dtype=float32,
        ),
        "Robot": array([[0.0, 0.0], [4.0, 4.0]], dtype=float32),
    },
    actions={
        "Move": CategoricalActionMask(
            actor_ids=None, actor_types=["Robot"], mask=None
        )
    },
    done=False,
    reward=0.0,
    ids={"Robot": [("Robot", 0), ("Robot", 1)]},
    end_of_episode_info=None,
)
```

### Step 1

> act method called

> actions

> env state

> pics

> observation

### Step 2

> act method called
> actions
> env state
> pics
> observation


</details>

## VecEnv / VecObs

## train.py

## rogue_net