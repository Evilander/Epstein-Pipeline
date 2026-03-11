"""Shared parallel processing utilities using concurrent.futures."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import TypeVar

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class ParallelExecutionError(RuntimeError):
    """Raised when one or more items fail during batch execution."""

    def __init__(self, failures: list[tuple[object, Exception]]) -> None:
        self.failures = failures
        count = len(failures)
        sample_items = ", ".join(repr(item) for item, _ in failures[:3])
        suffix = " ..." if count > 3 else ""
        super().__init__(
            f"{count} item(s) failed during parallel execution: {sample_items}{suffix}"
        )


def _finalize_result(item: T, result: R) -> R:
    """Reject awaitables so sync worker pools do not leak coroutines."""
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError(
            f"run_parallel only supports synchronous callables; {item!r} returned an awaitable"
        )
    return result


def run_parallel(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int = 4,
    label: str = "Processing",
    use_processes: bool = False,
) -> list[R]:
    """Execute *fn* over *items* in parallel with a Rich progress bar.

    Parameters
    ----------
    fn:
        A callable that takes a single item and returns a result.
        When *use_processes* is True, *fn* must be picklable (i.e. a
        module-level function, not a lambda or bound method).
    items:
        The items to process.
    max_workers:
        Maximum number of concurrent workers.
    label:
        Description shown in the progress bar.
    use_processes:
        If True, use ``ProcessPoolExecutor`` for CPU-bound work.
        If False (default), use ``ThreadPoolExecutor`` for I/O-bound work.

    Returns
    -------
    list[R]
        Results in completion order (NOT input order).
    """
    items_list = list(items)
    if not items_list:
        return []

    # Clamp workers to item count
    workers = min(max_workers, len(items_list))

    # Fall back to sequential for single worker
    if workers <= 1:
        return _run_sequential(fn, items_list, label)

    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    results: list[R] = []
    failures: list[tuple[object, Exception]] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task(label, total=len(items_list))

        with executor_cls(max_workers=workers) as executor:
            future_to_item: dict[Future[R], T] = {
                executor.submit(fn, item): item for item in items_list
            }

            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(_finalize_result(future_to_item[future], result))
                except Exception as exc:
                    item = future_to_item[future]
                    failures.append((item, exc))
                    logger.error("Failed processing %s: %s", item, exc)
                progress.advance(task)

    if failures:
        raise ParallelExecutionError(failures)

    return results


def _run_sequential(
    fn: Callable[[T], R],
    items: list[T],
    label: str,
) -> list[R]:
    """Sequential fallback with progress bar."""
    results: list[R] = []
    failures: list[tuple[object, Exception]] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task(label, total=len(items))
        for item in items:
            try:
                results.append(_finalize_result(item, fn(item)))
            except Exception as exc:
                failures.append((item, exc))
                logger.error("Failed processing %s: %s", item, exc)
            progress.advance(task)

    if failures:
        raise ParallelExecutionError(failures)

    return results
