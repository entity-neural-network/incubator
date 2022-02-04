# Internals

This document follows an batch of observations from the MineSweeper environment through the internals of entity-gym, enn-ppo, and rogue-net in excruciating detail.


## High-level overview

- The [Environment](#environment--observation) (entity-gym) provides a high-level abstraction for an environment.
- The [VecEnv](#vecenv--observation) (entity-gym) combines multiple environments and exposes a more efficient and lower-level batched representation of observations/actions.
- The [PPO training loop](#trainpy) (enn-ppo) keeps a sample buffer that combines observations from multiple steps.
- The policy is implemented by [RogueNet](#RogueNet) (rogue-net), a ragged batch transformer that takes lists of entities as input and outputs corresponding lists of actions.

## MineSweeper / State

Initial state of the three environments:

![](https://user-images.githubusercontent.com/12845088/152281730-d42a9ffe-b844-48c5-b6ff-de1ceecdb2f8.png)

<details>
    <summary>Environment State (click to expand)</summary>

```python
# Environment 1
mines = [(0, 2), (0, 1), (2, 2), (0, 0), (1, 0)]
robots = [(1, 1)]
orbital_cannon_cooldown = 5
orbital_cannon = False
# Environment 2
mines = [(2, 1)]
robots = [(2, 0)]
orbital_cannon_cooldown = 0
orbital_cannon = True
# Environment 3
mines = [(1, 0), (0, 1), (2, 2)]
robots = [(0, 0), (2, 0)]
orbital_cannon_cooldown = 5
orbital_cannon = False
```
</details>

## Environment / Observation

The MineSweeper class implements `Environment`, which provides a high-level abstraction for an environment.
`Environment`s expose their state as an `Observation` object, which contains a dictionary with the `features` of each entity, a list of `ids` to make it possible to reference specific entities, and a dictionary of `actions` that determines which entities can perform which actions.

<details>
  <summary>Observation #1 (click to expand)</summary>

```python
Observation(
    features={
        "Mine": [[0, 2], [0, 1], [2, 2], [0, 0], [1, 0]],
        "Robot": [[1, 1]]
    },
    ids={
        "Mine": [("Mine", 0), ("Mine", 1), ("Mine", 2), ("Mine", 3), ("Mine", 4)],
        "Robot": [("Robot", 0)],
    },
    actions={
        "Move": CategoricalActionMask(
            actor_ids=None,
            actor_types=["Robot"],
            mask=[[True, True, True, True, True]],
        ),
        "Fire Orbital Cannon": SelectEntityActionMask(
            actor_ids=None,
            actor_types=[],
            actee_types=["Mine", "Robot"],
            actee_ids=None,
            mask=None,
        ),
    },
    done=False,
    reward=0.0,
    end_of_episode_info=None,
)
```
</details>

<details>
  <summary>Observation #2 (click to expand)</summary>

```python
Observation(
    features={
        "Mine": [[2, 1]],
        "Robot": [[2, 0]], "Orbital Cannon": [[0]]
    },
    actions={
        "Move": CategoricalActionMask(
            actor_ids=None,
            actor_types=["Robot"],
            mask=[[False, True, True, False, True]],
        ),
        "Fire Orbital Cannon": SelectEntityActionMask(
            actor_ids=None,
            actor_types=["Orbital Cannon"],
            actee_types=["Mine", "Robot"],
            actee_ids=None,
            mask=None,
        ),
    },
    done=False,
    reward=0.0,
    ids={
        "Mine": [("Mine", 0)],
        "Robot": [("Robot", 0)],
        "Orbital Cannon": [("Orbital Cannon", 0)],
    },
    end_of_episode_info=None,
)
```
</details>

<details>
  <summary>Observation #3 (click to expand)</summary>

```python
Observation(
    features={
        "Mine": [[1, 0], [0, 1], [2, 2]],
        "Robot": [[0, 0], [2, 0]]
    },
    actions={
        "Move": CategoricalActionMask(
            actor_ids=None,
            actor_types=["Robot"],
            mask=[
                [True, False, True, False, True],
                [False, True, True, False, True],
            ],
        ),
        "Fire Orbital Cannon": SelectEntityActionMask(
            actor_ids=None,
            actor_types=[],
            actee_types=["Mine", "Robot"],
            actee_ids=None,
            mask=None,
        ),
    },
    done=False,
    reward=0.0,
    ids={
        "Mine": [("Mine", 0), ("Mine", 1), ("Mine", 2)],
        "Robot": [("Robot", 0), ("Robot", 1)],
    },
    end_of_episode_info=None,
)
```
</details>

## VecEnv / VecObs

The `ListEnv` is an implementation of `VecEnv` that aggregates the observations from multiple environments into a more efficient and lower level batched representation:
- Features of each entity type from all environments are combined into a single `RaggedBufferF32`
- Action masks from each action type from all environments are combined into a single `RaggedBufferBool`
- Instead of specifying the `actors` and `actees` of each action using `EntityID`s, we use the corresponding integer indices instead. The index of an entity is defined as follows:
  - The `entities` field of the `ObsSpace` specified by an `Environment` defines an ordering of the entity types.
  - In this case, the entity types are ordered as `["Mine", "Robot", "Orbital Cannon"]`.
  - We now go through all entity types in this order and sequentially assign an index to each entity.
  - For example, if there are three entities with `ids = {"Robot": [("Robot", 0)], "Mine": [("Mine", 0), ("Mine", 1)]}`, then the index of `("Robot", 0)` is `0`, the index of `("Mine", 0)` is `1`, and the index of `("Mine", 1)` is `2`.

<details>
  <summary>VecObs (click to expand)</summary>

```python
VecObs(
    features={
        "Mine": RaggedBufferF32(
            [
                [[0, 2], [0, 1], [2, 2], [0, 0], [1, 0]],
                [[2, 1]],
                [[1, 0], [0, 1], [2, 2]],
            ]
        ),
        "Robot": RaggedBufferF32(
            [
                [[1, 1]],
                [[2, 0]],
                [[0, 0], [2, 0]],
            ]
        ),
        "Orbital Cannon": RaggedBuffer(
            [
                [],
                [[0.0]],
                [],
            ]
        ),
    },
    action_masks={
        "Move": VecCategoricalActionMask(
            actors=RaggedBufferI64(
                [
                    [[5]],
                    [[1]],
                    [[3], [4]],
                ]
            ),
            mask=RaggedBufferBool(
                [
                    [[true, true, true, true, true]],
                    [[false, true, true, false, true]],
                    [
                        [true, false, true, false, true],
                        [false, true, true, false, true],
                    ],
                ]
            ),
        ),
        "Fire Orbital Cannon": VecSelectEntityActionMask(
            actors=RaggedBufferI64(
                [
                    [],
                    [[2]],
                    [],
                ]
            ),
            actees=RaggedBufferI64(
                [
                    [],
                    [[0], [1]],
                    [],
                ]
            ),
        ),
    },
    reward=array([0.0, 0.0, 0.0], dtype=float32),
    done=array([False, False, False]),
    end_of_episode_info={},
)
```
</details>

## enn_ppo/train.py

The PPO implementation in `enn_ppo/train.py` accumulates the `VecObs` from multiple steps into sample buffers.
These are later shuffled and split up into minibatches during the optimization phase.
In this case, we are just looking at a single rollout step and the batch of observations is forwarded unmodified to the policy to sample actions.

## RogueNet

The core of the policy is `RogueNet`, a ragged batch transformer implementation that takes in a ragged batch of observations and actor/actee/masks for each action, and outputs a ragged batch of actions and log-probabilities.

### Embedding

The first step is to flatten apply a projection to the features of each entity type to yield embeddings of the same size.
All embeddings are then concatenated into a single tensor which is ordered first by environment and then by entity index:

<details>
  <summary>Embedding Tensor (click to expand)</summary>

```python
tensor([
        # Environment 1
        [ 1.5280, -0.7984,  0.8672, -0.7984, -0.7984], # Mine 0
        [ 0.6134, -0.7676,  1.6895, -0.7676, -0.7676], # Mine 1
        [ 0.1566, -0.8506,  1.8400, -0.2497, -0.8963], # Mine 2
        [-0.8081, -0.7904,  1.4962,  0.9104, -0.8081], # Mine 3
        [-0.9405, -0.5402,  1.2698,  1.1515, -0.9405], # Mine 4
        [ 1.8806,  0.1884, -0.6897, -0.6897, -0.6897], # Robot 4
        # Environment 2
        [-0.8848, -0.5453,  1.6356,  0.6792, -0.8848], # Mine 0
        [ 1.3690,  1.0691, -0.8127, -0.8127, -0.8127], # Robot 0
        [-0.8059,  1.5626, -0.7685, -0.8059,  0.8175], # Orbital Cannon 0
        # Environment 3
        [-0.9405, -0.5402,  1.2698,  1.1515, -0.9405], # Mine 0
        [ 0.6134, -0.7676,  1.6895, -0.7676, -0.7676], # Mine 1
        [ 0.1566, -0.8506,  1.8400, -0.2497, -0.8963], # Mine 3
        [ 1.4806,  0.9317, -0.8041, -0.8041, -0.8041], # Robot 0
        [ 1.3690,  1.0691, -0.8127, -0.8127, -0.8127], # Robot 1
    ], device='cuda:0')
```
</details>

### Attention

Most of the transformer layers are applied independently to each entity.
However, the attention operation is applied to sequences of entities from the same timestep/environment.
It is currently implemented by packing/padding the flattened embeddings into a (sequence, entity, feature) tensor that places all entities from the same timestep/environment into the same sequence.
To do this, we compute three tensors:
- the `index` determines which entity is placed at each position the packed tensor
- the `batch` tells us what timestep/environment each entity came from, and is used to construct a mask that prevents attention from going across seperate timmesteps/environments
- the `inverse_index` is used to reconstruct the original flattened embedding tensor from the packed tensor

<details>
  <summary>Packing/padding metadata (click to expand)</summary>

```python
index = [
    [ 0,  1,  2,  3,  4,  5],
    [ 6,  7,  8,  0,  0,  0],
    [ 9, 10, 11, 12, 13,  0],
]
batch = [
    [ 0.,  0.,  0.,  0.,  0.,  0.],
    [ 1.,  1.,  1., nan, nan, nan],
    [ 2.,  2.,  2.,  2.,  2., nan],
]
inverse_index = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16
]
```
</details>

![](https://user-images.githubusercontent.com/12845088/147727605-d904ffff-42b4-4c51-9088-7ab32f9d481a.png)


### Categorical Action Head

Once the embeddings have passed through all layers, we  can compute the action heads for each entity.
Recall that we have a ragged list of indices of each actor.
However, the indices are only unique per environment, and we still need to add a ragged buffer of offsets to get a set of indices that is sequential over all environments and corresponds to the flattened embedding tensor.
The corrected indices are then used to index into the flattened embedding tensor to get the embedding for each actor.
We project the resulting embeddings onto the number of choices for each action to get a tensor of logits, and finally sample from the logits to get the action.

<details>
  <summary>"Move" action actors, offsets, indices, actions (click to expand)</summary>

```python
actors = RaggedBufferI64([
    [[5]],
    [[1]],
    [[3], [4]],
])
offsets = RaggedBuffer([
    [[0]],
    [[6]],
    [[9]],
])
actors + offsets = RaggedBufferI64([
    [[5]],
    [[7]],
    [[12], [13]],
])
indices = tensor([5, 7, 12, 13], dtype=int64)
# TODO: logits?
actions = tensor([4, 1, 4, 2], dtype=int64)
ragged_actions = RaggedBufferI64([
    [[4]],
    [[1]],
    [[4], [2]],
])
```
</details>

### Select Entity Action Head

The "Fire Orbital Cannon" action is a little more tricky. It is a SelectEntityAction, which means that it does not have a fixed number of choices, but the number of choices instead depends on the number of selectable entities in each the environment.
But at the end, we again get a list of indices corresponding to the entity selected by each actor.


![](https://user-images.githubusercontent.com/12845088/145058088-ae42f5f5-2782-4247-bcf5-8270a14e3510.png)


## Actions

Now, the actions computed by the model travel back to the environments.
The `ListEnv` receives ragged buffers for each action which represent the chosen action in the case of categorical actions, or the selected entity in the case of select entity actions.

<details>
  <summary>Ragged Actions (click to expand)</summary>

```python
actions = {
    'Fire Orbital Cannon': RaggedBuffer([
        [],
        [[0]],
        [],
    ]),
    'Move': RaggedBuffer([
        [[4]],
        [[1]],
        [[4], [2]],
    ]),
}
```
</details>

The actions are split up along the environment axis, joined with the list of actors from the initial `Observation`s, and actor indices are replaced with the corresponding `EntityID`s.
The resulting `Action` objects are dispatched to the `act` methods of the individual environments.

<details>
    <summary>Actions (click to expand)</summary>

```python
# Environment 1
{
    'Fire Orbital Cannon': SelectEntityAction(
        actors=[],
        actees=[],
    ),
    'Move': CategoricalAction(
        actors=[('Robot', 0)],
        actions=array([4]),
    ),
}
# Environment 2
{
    'Fire Orbital Cannon': SelectEntityAction(
        actors=[('Orbital Cannon', 0)],
        actees=[('Mine', 0)],
    ),
    'Move': CategoricalAction(
        actors=[('Robot', 0)],
        actions=array([1]),
    ),
}
# Environment 3
{
    'Fire Orbital Cannon': SelectEntityAction(
        actors=[],
        actees=[],
    ),
    'Move': CategoricalAction(
        actors=[('Robot', 0), ('Robot', 1)],
        actions=array([4, 2]),
    ),
}
```
</details>