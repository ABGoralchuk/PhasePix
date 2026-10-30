"""
CPU threading guard for PhasePix.

PyTorch on Windows can deadlock for 10+ minutes on tiny images (e.g. 700x700)
when the CPU backend falls back and the MKL / OpenMP thread pools oversubscribe
every logical core. The fix is to cap the thread count *before* PyTorch and the
math runtimes allocate their worker pools:

    1. ``apply_cpu_thread_env()`` writes ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS``
       (plus the OpenBLAS / NumExpr equivalents). OpenMP and MKL read these once,
       when their runtime is initialised, so this MUST run before ``numpy`` (MKL)
       and ``torch`` are imported.
    2. ``apply_torch_thread_cap()`` calls ``torch.set_num_threads()`` right after
       ``torch`` is imported, because the intra-op pool is created by PyTorch
       itself and ignores the environment variables set by a later import.

Both helpers are idempotent, never raise, and honour a lower thread count that
the user (or the portable launcher) already exported.
"""

from typing import Optional
import logging
import os

# Setup module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# Safe ceiling: 4 threads keeps MKL/OpenMP well below the oversubscription point
# on typical Windows office machines while still beating single-threaded Cellpose.
DEFAULT_CPU_THREAD_CAP = 4

# Environment variables read at runtime-initialisation time by the BLAS / OpenMP
# libraries bundled with numpy, SciPy and PyTorch.
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

# Optional PhasePix-specific override (e.g. ``PHASEPIX_CPU_THREADS=2``).
_REQUESTED_THREADS_ENV = "PHASEPIX_CPU_THREADS"

# Cached value so that logging happens once per process.
_effective_cpu_threads: Optional[int] = None


def resolve_cpu_thread_cap(default: int = DEFAULT_CPU_THREAD_CAP) -> int:
    """
    Compute the CPU thread count PhasePix should use.

    A value that is already exported through ``PHASEPIX_CPU_THREADS`` or one of
    the OpenMP / MKL variables is respected when it is *smaller* than the safe
    ceiling (a user who wants 2 threads keeps 2), otherwise it is clamped down to
    ``default`` so an aggressive ``OMP_NUM_THREADS=32`` cannot reintroduce the
    Windows deadlock.

    Args:
        default: Safe ceiling applied when nothing (or something larger) is set.

    Returns:
        int: Thread count between 1 and ``default``.
    """
    ceiling = max(1, int(default))
    requested = ceiling
    for name in (_REQUESTED_THREADS_ENV, *_THREAD_ENV_VARS):
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Ignoring non-numeric %s=%r.", name, raw)
            continue
        if value > 0:
            requested = min(requested, value)
    return max(1, requested)


def apply_cpu_thread_env(threads: Optional[int] = None) -> int:
    """
    Export the capped thread count for OpenMP / MKL / OpenBLAS / NumExpr.

    Must be called before ``numpy`` and ``torch`` are imported: those runtimes
    read the variables only once, at initialisation.

    Args:
        threads: Explicit thread count; ``None`` derives the safe cap.

    Returns:
        int: The thread count that was exported.
    """
    global _effective_cpu_threads
    cap = resolve_cpu_thread_cap() if threads is None else max(1, int(threads))
    for name in _THREAD_ENV_VARS:
        os.environ[name] = str(cap)
    if _effective_cpu_threads != cap:
        logger.info("CPU threading capped to %d thread(s) to prevent MKL/OpenMP deadlock.", cap)
    _effective_cpu_threads = cap
    return cap


def apply_torch_thread_cap(threads: Optional[int] = None) -> Optional[int]:
    """
    Cap PyTorch's intra-op thread pool via ``torch.set_num_threads``.

    Call this immediately after ``import torch`` - the environment variables set
    by :func:`apply_cpu_thread_env` only cover the OpenMP/MKL runtimes, whereas
    the PyTorch intra-op pool is configured through this API.

    Args:
        threads: Explicit thread count; ``None`` reuses the exported safe cap.

    Returns:
        Optional[int]: The applied thread count, or ``None`` when PyTorch does
        not expose the API (never raises, so startup cannot be broken).
    """
    if threads is None:
        threads = _effective_cpu_threads or resolve_cpu_thread_cap()
    threads = max(1, int(threads))
    try:
        import torch  # local import: caller has normally imported it already

        torch.set_num_threads(threads)
        applied = int(torch.get_num_threads())
    except Exception as exc:  # pragma: no cover - defensive, torch builds vary
        logger.warning("Could not cap PyTorch CPU threads: %s", exc)
        return None
    if applied != threads:
        logger.info("PyTorch reported %d CPU thread(s) after requesting %d.", applied, threads)
    else:
        logger.info("PyTorch CPU thread pool capped at %d thread(s).", applied)
    return applied
