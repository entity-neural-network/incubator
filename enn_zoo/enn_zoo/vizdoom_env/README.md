# ViZDoom environments

_Visual ZDoom environments, without the "Visual"_

Originally, [ViZDoom](https://github.com/mwydmuch/ViZDoom) (original DOOM) players are meant to be controlled using vision, but ViZDoom also offers access to entities on the map. These environments skip visual part and only use the position/rotation/etc info of entities to train agents.

## Get started

Run an experiment locally, from the root directory of the project:

```bash
poetry run python enn_zoo/enn_zoo/train.py --config enn_zoo/enn_zoo/vizdoom_env/vizdoom_config.ron
```

Available environments:
* DoomBasic ("basic.cfg")
* DoomHealthGathering ("health_gathering.cfg)
* DoomHealthGatheringSupreme ("health_gathering_supreme.cfg", health gathering but more difficult)
* DoomDefendTheCenter ("defend_the_center.cfg)
```
