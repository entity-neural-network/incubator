# ENN-PPO

PPO implementation compatible with Entity Gym.





## WORK IN PROGRESS Implementation details

The `rewards`, `dones`, and `values` can be stored in fixed-size preallocated tensors
as usual, since there is only one value per timestep and environment.
Observations, actions, and logprobs don't have a fixed shape and therefore require
special handling that differs during rollout and optimization.

### Rollouts

On each rollout step, environments return a `List[Observation]` and expect a `List[Dict[str, Action]]`.

Each observation has two components:

- `entities: Dict[str, np.ndarray]` maps each entity type to a (num_entity, num_feats) array of features.
- `action_masks: Mapping[str, ActionMask]` maps each action type to a list of indices of entities that can perform that action.

- `List[]

Each observation has a `Dict[str, np.ndarray]` 
- shuffling
- batching

### Optimization



### Ragged batch tensors



### Issues and possible improvements 


