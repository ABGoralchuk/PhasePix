"""
CPU threading guard for PhasePix.

PyTorch on Windows can deadlock for 10+ minutes on tiny images (e.g. 700x700)
when the CPU backend falls back and the MKL / OpenMP thread pools oversubscribe
every logical core. The fix is to cap the thread count *before* PyTorch and the
math runtimes allocate their worker pools:

    1. ``apply_cpu_thread_env()`` writes ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS``
       (plus the OpenBLAS / NumExpr equivalents) and sets
       ``KMP_DUPLICATE_LIB_OK=TRUE``. OpenMP and MKL read these once, when their
       runtime is initialised, so this MUST run before ``numpy`` (MKL) and
       ``torch`` are imported.
    2. ``apply_opencv_thread_limits()`` forces ``cv2.setNumThreads(0)`` so OpenCV
       never spawns its own TBB/OpenMP pool that competes with the PyTorch pool
       while resizing images and computing Cellpose flow dynamics.
    3. ``apply_torch_thread_cap()`` calls ``torch.set_num_threads()`` right after
       ``torch`` is imported, because the intra-op pool is created by PyTorch
       itself and ignores the environment variables set by a later import.

Why every step is needed on Windows:

    * ``KMP_DUPLICATE_LIB_OK=TRUE`` stops the Intel OpenMP (``libiomp5md.dll``)
      vs. LLVM OpenMP (``libomp.dll``) conflict that ships with PyTorch + MKL +
      OpenCV. Without it the conflicting runtimes can hang ``model.eval()`` /
      the first ``torch`` parallel region instead of running it.
    * An unbounded OpenCV pool *plus* an unbounded PyTorch pool oversubscribes
      every core, so each pool waits for a worker that never gets scheduled --
      the classic "tiny 700x700 image takes 10+ minutes" freeze.

All helpers are idempotent, never raise, and honour a lower thread count that
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

# Intel OpenMP duplicate-runtime escape hatch. PyTorch bundles ``libiomp5md.dll``
# while MKL / scikit-learn / OpenCV can drag in a second OpenMP runtime; the two
# copies fight over the same process and hang or abort the first parallel region
# (typically inside ``model.eval()``). "TRUE" makes them coexist.
_KMP_DUPLICATE_LIB_OK_ENV = "KMP_DUPLICATE_LIB_OK"

# OpenCV thread count. ``0`` disables OpenCV's internal thread pool entirely
# (documented OpenCV semantics: "if threads == 0, OpenCV will disable threading
# optimizations and run all its functions sequentially"), which is exactly what
# we want while PyTorch owns the CPU cores for Cellpose dynamics.
OPENCV_THREADS = 0

# Cached values so that logging happens once per process.
_effective_cpu_threads: Optional[int] = None
_effective_opencv_threads: Optional[int] = None


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

    Also sets ``KMP_DUPLICATE_LIB_OK=TRUE`` so the Intel OpenMP runtime shipped
    with PyTorch can coexist with the second OpenMP runtime pulled in by MKL /
    OpenCV instead of hanging ``model.eval()``.

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
    # Allow duplicate Intel/LLVM OpenMP runtimes in one process (PyTorch + MKL +
    # OpenCV on Windows); without this the clash can hard-hang the first parallel
    # region inside ``model.eval()``.
    os.environ[_KMP_DUPLICATE_LIB_OK_ENV] = "TRUE"
    if _effective_cpu_threads != cap:
        logger.info("CPU threading capped to %d thread(s) to prevent MKL/OpenMP deadlock.", cap)
    _effective_cpu_threads = cap
    return cap


def apply_opencv_thread_limits(threads: Optional[int] = None) -> Optional[int]:
    """
    Force OpenCV to run sequentially (``cv2.setNumThreads(0)`` by default).

    OpenCV keeps its own parallel-for backend (TBB / OpenMP / pthreads). When it
    runs concurrently with the PyTorch intra-op pool - resizing micrographs and
    computing Cellpose flow dynamics at the same time - the two pools
    oversubscribe the CPU and deadlock. Deferring all parallelism to PyTorch is
    both deadlock-free and, on small SEM images, just as fast.

    ``cv2.setNumThreads(0)`` disables OpenCV's threading optimisations, so
    ``cv2.getNumThreads()`` afterwards reports ``1`` (sequential execution).
    ``0`` is the only value that does this: OpenCV 4.x/5.x report the parallel
    backend's own concurrency (e.g. the CPU count) for every *positive* value,
    including ``1``, so asking for "one thread" would not actually serialise it.
    If a build ignores ``0`` the guard logs a warning instead of pretending the
    pool was capped.

    Safe to call repeatedly and before or after ``import cv2``; failure is only
    logged so a broken OpenCV build can never block startup.

    Args:
        threads: Explicit value forwarded to ``cv2.setNumThreads``; ``0`` (the
            default) disables OpenCV's thread pool.

    Returns:
        Optional[int]: The pool size OpenCV reports after the call (``1`` when
        threading is disabled), or ``None`` when the API is unavailable.
    """
    global _effective_opencv_threads
    requested = OPENCV_THREADS if threads is None else int(threads)
    try:
        import cv2  # local import: keeps this module free of heavy dependencies

        cv2.setNumThreads(requested)
        applied = int(cv2.getNumThreads())
    except Exception as exc:  # pragma: no cover - defensive, OpenCV builds vary
        logger.warning("Could not limit OpenCV threads: %s", exc)
        return None
    if requested <= 0 and applied > 1:
        logger.warning(
            "OpenCV ignored setNumThreads(0) and still reports %d thread(s); "
            "the PyTorch/OpenCV pool clash may persist on this build.",
            applied,
        )
    elif _effective_opencv_threads != applied:
        if applied <= 1:
            logger.info(
                "OpenCV threading disabled (cv2.setNumThreads(%d) -> %d thread) to avoid a "
                "PyTorch/OpenCV pool clash.",
                requested,
                applied,
            )
        else:
            logger.info(
                "OpenCV thread pool set to %d (requested %d) to avoid a PyTorch/OpenCV clash.",
                applied,
                requested,
            )
    _effective_opencv_threads = applied
    return applied


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
