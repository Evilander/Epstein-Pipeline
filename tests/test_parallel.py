"""Tests for parallel processing utilities."""

import logging
import time

import pytest

from epstein_pipeline.utils.parallel import ParallelExecutionError, run_parallel


def _slow_fn(x: int) -> int:
    """A simple function to test parallel execution."""
    time.sleep(0.01)
    return x * 2


def _failing_fn(x: int) -> int:
    """A function that fails on even inputs."""
    if x % 2 == 0:
        raise ValueError(f"Cannot process {x}")
    return x * 2


def test_run_parallel_basic():
    items = list(range(10))
    results = run_parallel(_slow_fn, items, max_workers=4, label="Test")
    assert len(results) == 10
    assert set(results) == {i * 2 for i in range(10)}


def test_run_parallel_single_worker():
    items = list(range(5))
    results = run_parallel(_slow_fn, items, max_workers=1, label="Sequential")
    assert len(results) == 5


def test_run_parallel_empty():
    results = run_parallel(_slow_fn, [], max_workers=4, label="Empty")
    assert results == []


def test_run_parallel_with_processes():
    items = list(range(5))
    results = run_parallel(_slow_fn, items, max_workers=2, label="Process", use_processes=True)
    assert len(results) == 5


def test_run_parallel_handles_errors():
    items = list(range(6))
    with pytest.raises(ParallelExecutionError) as exc_info:
        run_parallel(_failing_fn, items, max_workers=2, label="Errors")

    assert len(exc_info.value.failures) == 3
    assert {item for item, _ in exc_info.value.failures} == {0, 2, 4}


async def _async_fn(x: int) -> int:
    return x * 2


def test_run_parallel_rejects_awaitables(caplog):
    items = [1, 2]
    with caplog.at_level(logging.ERROR), pytest.raises(ParallelExecutionError) as exc_info:
        run_parallel(_async_fn, items, max_workers=2, label="Awaitables")

    assert len(exc_info.value.failures) == 2
    assert "returned an awaitable" in caplog.text
