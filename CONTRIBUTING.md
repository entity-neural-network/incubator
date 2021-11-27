# Contributing to Entity Neural Networks

👍🎉 Thank you for taking the time to contribute! 🎉👍

Feel free to open an issue or pull request if you have any questions or suggestions.
You can also [join our Discord](https://discord.gg/rrwSkmCp) and ask questions there.
If you plan to work on an issue, let us know in the issue thread so we can avoid duplicate work.

## Dev Setup

```bash
pip -r requirements-dev.txt
pip install -e entity_gym
pip install -e enn_ppo  # If you didn't have torch installed already, the torch-scatter install will fail. Just run the install command again.
```

## Code Formatting

We use [Black](https://black.readthedocs.io/en/stable/) for code formatting.
You can run `black .` to format all files.
Make sure that your version of `black` matches the version in [`requirements-dev.txt`]().

## Running MyPy and Tests

```bash
dmypy run -- entity_gym enn_ppo
pytest .
```

