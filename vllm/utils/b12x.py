# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy accessors for the optional ``b12x`` package."""

import functools
import importlib
import importlib.util
from types import ModuleType


@functools.cache
def has_b12x() -> bool:
    """Return whether the B12X package is installed."""
    return importlib.util.find_spec("b12x") is not None


@functools.cache
def get_b12x_blockscaled() -> ModuleType | None:
    """Load the B12X block-scaled GEMM module when it is available."""
    if not has_b12x():
        return None
    try:
        return importlib.import_module("b12x.gemm.blockscaled")
    except (ImportError, ModuleNotFoundError):
        return None
