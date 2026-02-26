# -*- coding: utf-8 -*-

"""
POTNet Package Initialization

This file exposes the key modules, classes, and functions of the POTNet package for easier imports.
"""

# Import the core classes
# Import utility functions
from .potnet import POTNet, dense_block, load_model

# Expose all imports for `from potnet import *`
__all__ = [
    "POTNet",
    "load_model",
]


# Define the package metadata
__author__ = "Wenhui Sophia Lu"
__email__ = "sophialu@stanford.edu"
__version__ = "1.0.0"
