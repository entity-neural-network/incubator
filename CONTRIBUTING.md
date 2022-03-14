# Contributing to Entity Neural Networks

👍🎉 Thank you for taking the time to contribute! 🎉👍

To get an overview of some of this project's motivation and goals, you can take a look at [Neural Network Architectures for Structured State](https://docs.google.com/document/d/1Q87zeY7Z4u9cU0oLoH-BPQZDBQd4tHLWiEkj5YDSGw4).
Feel free to open an issue or pull request if you have any questions or suggestions.
You can also [join our Discord](https://discord.gg/rrwSkmCp) and ask questions there.
If you plan to work on an issue, let us know in the issue thread so we can avoid duplicate work.

## Dev Setup

```bash
poetry install
poetry run pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
poetry run pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
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

### Common Build Problems

`poetry` sometimes does not play nicely with `conda`. So make sure you run a fresh shell that does not activate any conda environments. If you try to run `poetry install` while a conda environment is active, you might encounter something like the following error:

```
Cargo, the Rust package manager, is not installed or is not on PATH.
```

This can be resolved by running `conda deactivate` first.

If you are running into any other build issues, try the following recommended instructions.

* Ubuntu/Debian/Mint:
```bash
sudo apt install python3-dev make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

### Install Example

[![asciicast](https://asciinema.org/a/452597.svg)](https://asciinema.org/a/452597)

## Code Style

We use [Pre-commit](https://pre-commit.com/) to 
<!-- * sort dependencies
* remove unused variables and imports -->
* format code using black (via `black`)
* check word spelling (via `codespell`)
* check typing (via `mypy`)

You can run the following command to do these automatically:

```bash
poetry run pre-commit run --all-files
```

## Running Tests

```bash
poetry run pytest .
```
