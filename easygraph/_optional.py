"""Shared handling for optional machine-learning dependencies."""

import sys
import warnings

from contextlib import contextmanager


PYTORCH_INSTALL_URL = "https://pytorch.org/get-started/locally/"

try:
    import torch as _torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    _torch = None

HAS_TORCH = _torch is not None


def require_torch(feature="This feature"):
    if _torch is None:
        raise ImportError(
            "{} requires PyTorch. Install a build for your Python version and "
            "platform using {}, then restart Python.".format(
                feature, PYTORCH_INSTALL_URL
            )
        )
    return _torch


def warn_missing_torch():
    if not HAS_TORCH:
        warnings.warn(
            "EasyGraph: PyTorch is not installed for Python {}.{}. Core graph "
            "algorithms and the C++ backend remain available. For PyTorch-based "
            "features, install a compatible build using {} and restart Python.".format(
                sys.version_info.major, sys.version_info.minor, PYTORCH_INSTALL_URL
            ),
            UserWarning,
            stacklevel=2,
        )


@contextmanager
def optional_torch_modules():
    """Allow absent optional packages, while preserving unrelated import errors."""
    try:
        yield
    except ModuleNotFoundError as error:
        if error.name not in {
            "torch",
            "torch_geometric",
            "torch_scatter",
            "torch_sparse",
            "sklearn",
        }:
            raise
