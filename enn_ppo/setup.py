#!/usr/bin/env python

from distutils.core import setup

setup(
    name="enn-gym",
    version="0.1",
    description="PPO training loop compatible with EntityGym",
    author="Clemens Winter",
    author_email="clemenswinter1@gmail.com",
    packages=["enn_ppo"],
    package_data={"enn_ppo": ["py.typed"]},
    dependency_links=["https://data.pyg.org/whl/torch-1.10.0+cu102.html",],
    install_requires=[
        "numpy~=1.21",
        "torch==1.10.*",
        "tensorboard~=2.7",
        "msgpack~=1.0",
        "msgpack-numpy~=0.4.7",
        "tqdm~=4.62",
        "torch-scatter==2.0.9",
        "ragged-buffer==0.2.0",
    ],
)
