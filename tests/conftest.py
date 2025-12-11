"""Pytest configuration and shared fixtures for tqd tests."""

from __future__ import annotations

import os

import pytest
import torch


def is_distributed_available() -> bool:
    """Check if distributed environment is properly set up."""
    return (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and torch.cuda.is_available()
        and torch.distributed.is_initialized()
    )


requires_distributed = pytest.mark.skipif(
    not is_distributed_available(),
    reason="Requires distributed environment (run with torchrun)",
)


@pytest.fixture
def rank() -> str:
    """Get the current process rank."""
    return os.environ.get("RANK", "0")


@pytest.fixture
def world_size() -> int:
    """Get the world size for distributed training."""
    return int(os.environ.get("WORLD_SIZE", "2"))
