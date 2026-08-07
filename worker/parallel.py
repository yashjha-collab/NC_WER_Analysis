"""Multiprocessing helpers for the WER pipeline.

Uses ProcessPoolExecutor so CPU-heavy FA/NC work can use multiple cores
despite the CPython GIL. Pair with asyncio in the orchestrator for job
scheduling and with async STT clients for network waits.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _mp_context() -> mp.context.BaseContext:
    """Prefer ``fork`` so children inherit the parent's ``sys.path``.

    macOS defaults to ``spawn``, which re-imports the FastAPI app (``app.main``)
    in every child before ``REPO_ROOT`` is on the path, breaking ``worker.*``
    imports. Linux defaults to ``fork`` already; this just makes local macOS
    runs behave like the deployment VM. Falls back to the platform default
    when ``fork`` is unavailable.
    """
    try:
        return mp.get_context("fork")
    except (ValueError, RuntimeError):
        return mp.get_context()


def configure_worker_threads(num_threads: int = 2) -> None:
    """Cap intra-op threads so N workers × threads ≈ machine vCPUs."""
    n = max(1, int(num_threads))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "OMP_THREAD_LIMIT",
    ):
        os.environ[key] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:
        pass


def _init_worker(cpu_threads: int) -> None:
    configure_worker_threads(cpu_threads)


def map_parallel(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int = 8,
    cpu_threads: int = 2,
) -> list[R]:
    """Run ``fn`` over items in a process pool; preserve input order."""
    item_list = list(items)
    if not item_list:
        return []
    workers = max(1, min(int(max_workers), len(item_list)))
    if workers == 1:
        configure_worker_threads(cpu_threads)
        return [fn(item) for item in item_list]

    results: list[R | None] = [None] * len(item_list)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=_mp_context(),
        initializer=_init_worker,
        initargs=(cpu_threads,),
    ) as pool:
        future_map = {
            pool.submit(fn, item): idx for idx, item in enumerate(item_list)
        }
        for fut in as_completed(future_map):
            idx = future_map[fut]
            results[idx] = fut.result()
    return results  # type: ignore[return-value]


async def amap_parallel(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int = 8,
    cpu_threads: int = 2,
) -> list[Any]:
    """Async wrapper: schedule process-pool work without blocking the event loop."""
    import asyncio

    item_list = list(items)
    if not item_list:
        return []

    workers = max(1, min(int(max_workers), len(item_list)))
    loop = asyncio.get_running_loop()

    if workers == 1:
        return [await asyncio.to_thread(fn, item) for item in item_list]

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=_mp_context(),
        initializer=_init_worker,
        initargs=(cpu_threads,),
    ) as pool:
        futures = [
            loop.run_in_executor(pool, fn, item) for item in item_list
        ]
        return await asyncio.gather(*futures, return_exceptions=True)
