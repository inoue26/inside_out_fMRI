#!/usr/bin/env python
import datetime
import os
import shutil
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install

readme = open("README.md").read()

VERSION = "0.1.0"

requirements = [
    "torch",
]

VERSION += "_" + datetime.datetime.now().strftime("%Y%m%d%H%M")
# print(VERSION)

setup(
    # Metadata
    name="voluntary-fixation",
    version=VERSION,
    author="Araya Inc.",
    author_email="inoue@araya.org",
    url="git@github.com:inoue26/studyforrest_voluntary_fixation.git",
    description="What you see is what you receive or want ? ",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    # Package info
    # packages=find_packages(exclude=('*test*',)),
    packages=find_packages(
        include=('voluntary_fixation'), exclude=('*test*',)
    ),
    #
    zip_safe=True,
    install_requires=requirements,
    # Classifiers
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)