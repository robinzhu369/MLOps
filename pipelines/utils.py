"""Utility functions for pipeline state management."""

import functools

import numpy as np


def sanitize_for_checkpoint(obj):
    """Recursively convert numpy types to native Python types for serialization.

    LangGraph's MemorySaver uses msgpack which doesn't handle numpy types.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_checkpoint(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_checkpoint(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def checkpoint_safe(func):
    """Decorator that sanitizes node return values for checkpoint serialization."""
    @functools.wraps(func)
    def wrapper(state):
        result = func(state)
        return sanitize_for_checkpoint(result)
    return wrapper
