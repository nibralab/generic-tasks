#!/usr/bin/env bash

# Create a source distribution
python setup.py sdist

# Create a wheel distribution
python setup.py bdist_wheel

# Remove build directory
rm -rf build
