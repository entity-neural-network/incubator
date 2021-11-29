#!/usr/bin/env python

from distutils.core import setup

setup(
    name="rogue_net",
    version="0.1",
    description="Entity Networks compatible with EntityGym",
    author="Clemens Winter",
    author_email="clemenswinter1@gmail.com",
    packages=["rogue_net"],
    package_data={"rogue_net": ["py.typed"]},
    dependency_links=["https://data.pyg.org/whl/torch-1.10.0+cu102.html",],
    install_requires=["torch==1.10.*", "torch-scatter==2.0.9", "ragged-buffer==0.2.0"],
)
