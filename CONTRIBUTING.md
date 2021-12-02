# Contributing to Entity Neural Networks

👍🎉 Thank you for taking the time to contribute! 🎉👍

Feel free to open an issue or pull request if you have any questions or suggestions.
You can also [join our Discord](https://discord.gg/rrwSkmCp) and ask questions there.
If you plan to work on an issue, let us know in the issue thread so we can avoid duplicate work.

## Dev Setup

```bash
poetry install # torch-scatter installation will fail, simply run again
poetry install # fix torch-scatter installation
poetry install -E griddly
```

Then you can run the scripts under the poetry environment in two ways: `poetry run` or `poetry shell`. 

* `poetry run`:
    By prefixing `poetry run`, your command will run in poetry's virtual environment. For example, try running
    ```bash
    poetry run python enn_ppo/enn_ppo/train.py
    ```
* `poetry shell`:
    First, activate the poetry's virtual environment by executing `poetry shell`. Then, the name of the poetry's
    virtual environment (e.g. `(incubator-EKBuw-J_-py3.9)`) should appear in the left side of your shell.
    Afterwards, you can directly run
    ```bash
    python enn_ppo/enn_ppo/train.py
    ```

See the following video as an example.

[![asciicast](https://asciinema.org/a/452597.svg)](https://asciinema.org/a/452597)

## Code Formatting

We use [Black](https://black.readthedocs.io/en/stable/) for code formatting.
You can run the following to format all files:

```bash
poetry run black .
```

## Running MyPy and Tests

```bash
poetry run dmypy run -- entity_gym enn_ppo rogue_net
poetry run pytest .
```


## Troubleshoot Build Problems

`poetry` sometimes does not play nicely with `conda`. So make sure you run a fresh shell that does not activate any conda environments. If you are running into any other build issues, try the following recommended instructions.


* Ubuntu/Debian/Mint:
```bash
sudo apt install python3-dev make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```
