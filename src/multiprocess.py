"""
High-performance computing utilities — AFML Chapter 20.

mp_pandas_obj distributes any pandas-returning function across CPU cores
using a molecule pattern: the full index is split into sub-indices
(molecules), each molecule is processed by one worker, and the results
are concatenated.  AFML Snippet 20.5.

Note on Windows
---------------
Python's multiprocessing defaults to the 'spawn' start method on Windows.
Worker functions must therefore be module-level (picklable) — no lambdas
or closures that capture non-picklable objects.  Set num_threads=1 to run
single-threaded if pickling fails.

Example
-------
>>> from src.multiprocess import mp_pandas_obj
>>> results = mp_pandas_obj(
...     func=apply_triple_barrier,
...     pd_obj=('molecule', events.index),
...     num_threads=4,
...     close=close,
...     events=events,
...     pt_sl=[1.0, 1.0],
... )
"""
import time
import multiprocessing as mp
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lin_parts(num_atoms: int, num_threads: int) -> np.ndarray:
    """Split num_atoms atoms as evenly as possible across num_threads."""
    parts = np.minimum(
        num_atoms,
        np.arange(0, num_threads + 1) * num_atoms // num_threads,
    )
    return parts.astype(int)


def _expand_call(kargs: dict):
    """Unpack a job dict and call the stored function."""
    kargs = kargs.copy()
    func = kargs.pop('func')
    return func(**kargs)


def _init_pool_paths(extra_paths: list) -> None:
    """Pool initializer: propagate the parent's sys.path into each worker (Windows spawn)."""
    import sys
    for p in reversed(extra_paths):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Job dispatch
# ---------------------------------------------------------------------------

def process_jobs(jobs: list, num_threads: int = 1) -> list:
    """
    Execute a list of job dicts (each must contain key 'func').

    Parameters
    ----------
    jobs        : list of dicts; each has 'func' plus keyword args.
    num_threads : 1 → serial; >1 → multiprocessing Pool.

    Returns
    -------
    list of per-job results in the same order as `jobs`.
    """
    import sys as _sys
    if num_threads <= 1 or len(jobs) == 1:
        return [_expand_call(job) for job in jobs]

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_threads,
                  initializer=_init_pool_paths,
                  initargs=(_sys.path[:],)) as pool:
        results = pool.map(_expand_call, jobs)
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def mp_pandas_obj(func, pd_obj, num_threads: int = 1,
                  mp_batches: int = 1, lin_mols: bool = True,
                  **kwargs) -> pd.DataFrame:
    """
    Distribute a pandas computation across CPU cores.  AFML Snippet 20.5.

    Parameters
    ----------
    func        : callable(molecule, **kwargs) → pd.Series | pd.DataFrame
                  `molecule` receives the sub-index assigned to this worker.
    pd_obj      : (arg_name: str, iterable: pd.Index | list)
                  Name of the 'molecule' argument and the full index to split.
    num_threads : number of parallel workers (1 = single-threaded)
    mp_batches  : batches per thread; increase for better load balancing
    lin_mols    : split index linearly (always True in this implementation)
    **kwargs    : additional keyword arguments forwarded verbatim to func

    Returns
    -------
    pd.concat of all molecule results, sorted by index.
    """
    arg_name, index = pd_obj
    num_atoms = len(index)
    if num_atoms == 0:
        return pd.DataFrame()

    num_workers = max(1, num_threads * mp_batches)
    parts = _lin_parts(num_atoms, num_workers)

    jobs = []
    for i in range(len(parts) - 1):
        molecule = index[parts[i]:parts[i + 1]]
        if len(molecule) == 0:
            continue
        job = {arg_name: molecule, 'func': func}
        job.update(kwargs)
        jobs.append(job)

    results = process_jobs(jobs, num_threads=num_threads)
    results = [r for r in results if r is not None and len(r) > 0]

    if not results:
        return pd.DataFrame()

    out = pd.concat(results)
    return out.sort_index()


# ---------------------------------------------------------------------------
# Module-level benchmark worker (must be at module level for Windows spawn)
# ---------------------------------------------------------------------------

def _vol_molecule_worker(molecule, close, span: int = 50):
    """Compute get_daily_vol for a molecule sub-index. Module-level so it is picklable on Windows."""
    from labeling import get_daily_vol
    return get_daily_vol(close, span=span).reindex(molecule)


# ---------------------------------------------------------------------------
# Benchmarking utility
# ---------------------------------------------------------------------------

def benchmark_mp(func, pd_obj, thread_counts: list = None,
                 **kwargs) -> pd.DataFrame:
    """
    Benchmark func at different thread counts and return a timing table.

    Parameters
    ----------
    func         : function to benchmark (same signature as mp_pandas_obj)
    pd_obj       : (arg_name, index) tuple
    thread_counts: list of integers to try (default: [1, 2, 4])
    **kwargs     : forwarded to mp_pandas_obj

    Returns
    -------
    pd.DataFrame with columns [num_threads, elapsed_s, speedup]
    """
    if thread_counts is None:
        max_cpu = mp.cpu_count()
        thread_counts = [1, min(2, max_cpu), min(4, max_cpu)]

    rows = []
    baseline = None
    for n in thread_counts:
        t0 = time.perf_counter()
        mp_pandas_obj(func, pd_obj, num_threads=n, **kwargs)
        elapsed = time.perf_counter() - t0
        if baseline is None:
            baseline = elapsed
        rows.append({
            'num_threads': n,
            'elapsed_s': round(elapsed, 4),
            'speedup': round(baseline / elapsed, 3) if elapsed > 0 else None,
        })

    return pd.DataFrame(rows)
