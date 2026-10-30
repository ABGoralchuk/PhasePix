"""
CPU threading guard for PhasePix (build v1.7.3 - native speed restoration).

History
-------
PyTorch on **Windows CPU-only** machines can deadlock for 10+ minutes on tiny
images (e.g. 700x700) when the CPU backend falls back and the MKL / OpenMP
thread pools oversubscribe every logical core. v1.7.1 / v1.7.2 fixed that by
capping every thread pool in the process *unconditionally* -- which also slowed
macOS (Apple Silicon / MPS) and NVIDIA CUDA machines down, because
``cv2.setNumThreads(0)`` forces ``bilateralFilter`` / ``medianBlur`` /
``findContours`` / morphology into single-threaded execution and a 4-thread
MKL/NumPy cap throttles the CPU post-processing of Cellpose flow dynamics.

v1.7.3 makes the guard platform- and hardware-aware. The helpers below are
complete no-ops unless the Windows CPU fallback is really in play:

    * ``sys.platform != "win32"`` (macOS ``darwin``, Linux): nothing is touched.
      No ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` /
      ``NUMEXPR_NUM_THREADS`` exports, no ``cv2.setNumThreads(0)`` call and no
      ``torch.set_num_threads`` cap -> native full-speed threading.
    * ``sys.platform == "win32"`` **with** ``torch.cuda.is_available()``: the
      NVIDIA GPU owns the heavy work, so the guard stays off as well. OpenCV is
      reset to its system default pool (``cv2.setNumThreads(-1)``) and PyTorch
      gets ``os.cpu_count()`` intra-op threads.
    * ``sys.platform == "win32"`` **without** CUDA (the CPU-only fallback): the
      strict v1.7.2 guard runs -- 4 threads exported to OMP / MKL / OpenBLAS /
      NumExpr, ``cv2.setNumThreads(0)`` and ``torch.set_num_threads(4)``.
    * ``PHASEPIX_FORCE_THREAD_CAP=1`` forces the strict guard on any platform and
      backend (diagnostic escape hatch for an oversubscribed macOS / Linux box).

Why the Windows CPU guard needs every step:

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
import shutil
import subprocess
import sys

# Setup module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Strict ceiling used ONLY by the Windows CPU-only fallback: 4 threads keeps
# MKL/OpenMP well below the oversubscription point on typical Windows office
# machines while still beating single-threaded Cellpose.
DEFAULT_CPU_THREAD_CAP = 4

# Thread count assumed when ``os.cpu_count()`` cannot report one (native mode).
NATIVE_THREADS_FALLBACK = 8

# The platform string Python reports on Windows. Every other value (``darwin``
# for macOS, ``linux`` for Linux) runs without any thread cap.
WINDOWS_PLATFORM = "win32"

# Environment variables read at runtime-initialisation time by the BLAS / OpenMP
# libraries bundled with numpy, SciPy and PyTorch.
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

# Optional PhasePix-specific override (e.g. ``PHASEPIX_CPU_THREADS=2``) that can
# only *lower* the Windows CPU-fallback ceiling.
_REQUESTED_THREADS_ENV = "PHASEPIX_CPU_THREADS"

# Diagnostic escape hatch: ``PHASEPIX_FORCE_THREAD_CAP=1`` re-engages the strict
# Windows CPU-fallback guard on any platform / backend combination.
_FORCE_THREAD_CAP_ENV = "PHASEPIX_FORCE_THREAD_CAP"
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on", "y", "t"})

# Intel OpenMP duplicate-runtime escape hatch. PyTorch bundles ``libiomp5md.dll``
# while MKL / scikit-learn / OpenCV can drag in a second OpenMP runtime; the two
# copies fight over the same process and hang or abort the first parallel region
# (typically inside ``model.eval()``). "TRUE" makes them coexist.
_KMP_DUPLICATE_LIB_OK_ENV = "KMP_DUPLICATE_LIB_OK"

# OpenCV thread count used by the guarded (Windows CPU-only) path. ``0`` disables
# OpenCV's internal thread pool entirely (documented OpenCV semantics: "if
# threads == 0, OpenCV will disable threading optimizations and run all its
# functions sequentially"), freeing the cores for the PyTorch pool.
OPENCV_THREADS = 0

# Any negative value restores OpenCV's system default pool (= CPU count). Used on
# the native path so a stale guard call cannot leave OpenCV single-threaded.
OPENCV_DEFAULT_THREADS = -1

# Upper bound for the ``nvidia-smi`` probe performed before ``torch`` is imported.
_NVIDIA_SMI_TIMEOUT_S = 5.0

# Cached values so that logging happens once per process.
_effective_cpu_threads: Optional[int] = None
_effective_opencv_threads: Optional[int] = None
_cuda_available_cache: Optional[bool] = None
_logged_messages: set = set()


# ---------------------------------------------------------------------------
# Platform / hardware detection
# ---------------------------------------------------------------------------


def is_windows() -> bool:
    """
    Return ``True`` when PhasePix runs on Windows (``sys.platform == "win32"``).

    Only Windows needs the CPU thread guard; macOS (``darwin``) and Linux keep
    native full-speed threading.
    """
    return sys.platform == WINDOWS_PLATFORM


def force_thread_cap_requested() -> bool:
    """
    Return ``True`` when the strict thread cap was explicitly requested.

    ``PHASEPIX_FORCE_THREAD_CAP=1`` re-engages the v1.7.2 behaviour on macOS and
    on CUDA machines (e.g. to diagnose an oversubscribed CI container or a
    virtualised Linux host).
    """
    return os.environ.get(_FORCE_THREAD_CAP_ENV, "").strip().lower() in _TRUTHY_VALUES


def native_thread_count() -> int:
    """
    Return the number of logical CPU cores Python can see (never below 1).

    Used as the PyTorch intra-op pool size on the native (macOS / CUDA) path.
    """
    try:
        count = os.cpu_count()
    except Exception:  # pragma: no cover - defensive, os.cpu_count does not raise
        count = None
    if not count:
        return NATIVE_THREADS_FALLBACK
    return max(1, int(count))


def _probe_nvidia_driver() -> bool:
    """
    Cheap NVIDIA-driver probe used while ``torch`` is still unimported.

    The guard has to run before ``import torch`` (the ``OMP_*`` variables are
    read once, at runtime initialisation), so ``torch.cuda.is_available()``
    cannot be consulted at that point. Querying ``nvidia-smi -L`` -- which ships
    with every NVIDIA driver -- answers the same question without dragging the
    whole PyTorch runtime into memory at startup.

    Returns:
        bool: ``True`` when at least one NVIDIA GPU is visible to this process.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in ("", "-1"):
        # The GPU was deliberately hidden from this process.
        return False
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    extra_kwargs = {}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):  # keep the portable build windowless
        extra_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(
            [executable, "-L"],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
            **extra_kwargs,
        )
    except Exception as exc:  # pragma: no cover - driver layouts vary
        logger.debug("nvidia-smi probe failed: %s", exc)
        return False
    output = (completed.stdout or "").strip()
    return completed.returncode == 0 and "GPU" in output


def detect_cuda_available() -> bool:
    """
    Return ``True`` when an NVIDIA CUDA GPU is usable by this process.

    ``torch.cuda.is_available()`` is authoritative and is always preferred once
    ``torch`` has been imported. Before that moment :func:`_probe_nvidia_driver`
    is used instead, so the guard can decide without an early ``import torch``.
    Never raises: an unreadable CUDA runtime simply means "no CUDA".
    """
    global _cuda_available_cache
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            return bool(torch.cuda.is_available())
        except Exception as exc:  # pragma: no cover - defensive, torch builds vary
            logger.warning("torch.cuda.is_available() failed (%s); assuming CPU-only.", exc)
            return False
    if _cuda_available_cache is None:
        _cuda_available_cache = _probe_nvidia_driver()
    return _cuda_available_cache


def should_apply_thread_cap() -> bool:
    """
    Decide whether the strict Windows CPU-fallback guard must run.

    Returns:
        bool: ``True`` only on Windows *without* an NVIDIA CUDA GPU, or when the
        strict guard was explicitly requested through
        ``PHASEPIX_FORCE_THREAD_CAP=1``. macOS (``darwin``), Linux and
        CUDA-enabled Windows machines always keep native full-speed threading.
    """
    if force_thread_cap_requested():
        return True
    if not is_windows():
        return False
    return not detect_cuda_available()


def _log_once(key: str, message: str, *args) -> None:
    """Emit ``message`` a single time per process for the given ``key``."""
    if key in _logged_messages:
        return
    _logged_messages.add(key)
    logger.info(message, *args)


def _export_thread_env(cap: int) -> None:
    """Write ``cap`` into the OMP / MKL / OpenBLAS / NumExpr variables."""
    for name in _THREAD_ENV_VARS:
        os.environ[name] = str(int(cap))


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------


def resolve_cpu_thread_cap(default: int = DEFAULT_CPU_THREAD_CAP) -> int:
    """
    Compute the CPU thread count the Windows CPU fallback should use.

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
    Export the thread count OpenMP / MKL / OpenBLAS / NumExpr should use.

    On macOS (``darwin``), Linux and CUDA-enabled Windows machines this is a
    complete no-op: ``os.environ`` keeps its native configuration, so every
    runtime starts with full multi-core speed (v1.7.3). Only the Windows
    CPU-only fallback exports the strict cap -- together with
    ``KMP_DUPLICATE_LIB_OK=TRUE``, which lets the Intel OpenMP runtime shipped
    with PyTorch coexist with the second OpenMP runtime pulled in by MKL /
    OpenCV instead of hanging ``model.eval()``.

    Must be called before ``numpy`` and ``torch`` are imported: those runtimes
    read the variables only once, at initialisation. That early position is why
    the CUDA check falls back to an ``nvidia-smi`` probe while ``torch`` is not
    imported yet (see :func:`detect_cuda_available`).

    Args:
        threads: Explicit ceiling for the guarded (Windows CPU-only) path;
            ``None`` derives the safe cap through :func:`resolve_cpu_thread_cap`.

    Returns:
        int: The effective CPU thread count -- the native logical CPU count when
        the guard is inactive, otherwise the exported cap.
    """
    global _effective_cpu_threads
    if is_windows():
        # Harmless on every Windows backend, essential on the CPU-only fallback
        # (libiomp5md.dll vs. libomp.dll clash).
        os.environ[_KMP_DUPLICATE_LIB_OK_ENV] = "TRUE"

    if not should_apply_thread_cap():
        native = native_thread_count()
        _log_once(
            "env_native",
            "Native full-speed threading active (%s%s): OMP/MKL/OpenBLAS/NumExpr limits "
            "left untouched - no CPU thread cap applied.",
            sys.platform,
            ", NVIDIA CUDA backend" if is_windows() and detect_cuda_available() else "",
        )
        _effective_cpu_threads = native
        return native

    cap = max(1, int(threads)) if threads is not None else resolve_cpu_thread_cap()
    _export_thread_env(cap)
    _log_once(
        "env_guarded",
        "Windows CPU-fallback thread guard engaged: OMP/MKL/OpenBLAS/NumExpr capped to %d "
        "thread(s) to prevent the MKL/OpenMP deadlock.",
        cap,
    )
    _effective_cpu_threads = cap
    return cap


def _set_opencv_threads(requested: int) -> Optional[int]:
    """
    Apply ``cv2.setNumThreads(requested)`` and report the resulting pool size.

    Args:
        requested: ``0`` disables OpenCV threading, a negative value restores the
            system default pool, positive values request an explicit size.

    Returns:
        Optional[int]: The pool size OpenCV reports afterwards, or ``None`` when
        the API is unavailable (failure is only logged, never raised).
    """
    global _effective_opencv_threads
    requested = int(requested)
    try:
        import cv2  # local import: keeps this module free of heavy dependencies

        cv2.setNumThreads(requested)
        applied = int(cv2.getNumThreads())
    except Exception as exc:  # pragma: no cover - defensive, OpenCV builds vary
        logger.warning("Could not set OpenCV threads: %s", exc)
        return None
    if requested == 0 and applied > 1:
        logger.warning(
            "OpenCV ignored setNumThreads(0) and still reports %d thread(s); "
            "the PyTorch/OpenCV pool clash may persist on this build.",
            applied,
        )
    elif requested >= 0 and _effective_opencv_threads != applied:
        logger.info("OpenCV thread pool set to %d (requested %d).", applied, requested)
    _effective_opencv_threads = applied
    return applied


def apply_opencv_thread_limits(threads: Optional[int] = None) -> Optional[int]:
    """
    Keep OpenCV's own parallel-for pool at full speed unless the guard is needed.

    OpenCV runs its own TBB / OpenMP / pthreads pool. On the Windows CPU-only
    fallback that pool oversubscribes the cores alongside the PyTorch intra-op
    pool while Cellpose resizes micrographs and computes flow dynamics, so it is
    disabled with ``cv2.setNumThreads(0)``.

    On macOS, Linux and CUDA-enabled Windows machines (v1.7.3) the pool is left
    alone, or restored to the system default, because serialising
    ``bilateralFilter``, ``medianBlur``, ``findContours`` and the morphology
    kernels is what made v1.7.1 / v1.7.2 several times slower there. Outside
    Windows, ``cv2.setNumThreads`` is never called at all unless
    ``PHASEPIX_FORCE_THREAD_CAP=1`` is set.

    Safe to call repeatedly and before or after ``import cv2``; failure is only
    logged so a broken OpenCV build can never block startup.

    Args:
        threads: Explicit value forwarded to ``cv2.setNumThreads`` on the guarded
            path; ``0`` (the default) disables OpenCV's thread pool.

    Returns:
        Optional[int]: The pool size OpenCV reports after the call, or ``None``
        when OpenCV was not touched / the API is unavailable.
    """
    if not should_apply_thread_cap():
        if is_windows():
            # Windows + NVIDIA CUDA: restore the default pool so a stale guard
            # call can never leave the image pipeline single-threaded.
            applied = _set_opencv_threads(OPENCV_DEFAULT_THREADS)
            _log_once(
                "opencv_native_cuda",
                "Native full-speed threading active (win32, NVIDIA CUDA backend): OpenCV "
                "keeps its multi-threaded pool (%s thread(s)) - no cv2.setNumThreads(0) call.",
                applied if applied is not None else native_thread_count(),
            )
            return applied
        _log_once(
            "opencv_native_platform",
            "Native full-speed threading active (%s): OpenCV thread pool left untouched "
            "(no cv2.setNumThreads call) - multi-threaded image processing preserved.",
            sys.platform,
        )
        return None

    requested = OPENCV_THREADS if threads is None else int(threads)
    applied = _set_opencv_threads(requested)
    if requested <= 0:
        _log_once(
            "opencv_guarded",
            "Windows CPU-fallback thread guard engaged: OpenCV threading disabled "
            "(cv2.setNumThreads(%d) -> %s thread) to avoid a PyTorch/OpenCV pool clash.",
            requested,
            applied if applied is not None else "unknown",
        )
    return applied


def _set_torch_threads(threads: int) -> Optional[int]:
    """
    Apply ``torch.set_num_threads`` and report the resulting intra-op pool size.

    Args:
        threads: Requested intra-op thread count (clamped to at least 1).

    Returns:
        Optional[int]: The applied thread count, or ``None`` when PyTorch does
        not expose the API (failure is only logged, never raised).
    """
    threads = max(1, int(threads))
    try:
        import torch  # local import: caller has normally imported it already

        torch.set_num_threads(threads)
        applied = int(torch.get_num_threads())
    except Exception as exc:  # pragma: no cover - defensive, torch builds vary
        logger.warning("Could not set PyTorch CPU threads: %s", exc)
        return None
    if applied != threads:
        logger.info("PyTorch reported %d CPU thread(s) after requesting %d.", applied, threads)
    return applied


def _report_torch_threads() -> Optional[int]:
    """
    Report PyTorch's intra-op pool size without changing it (native path).

    Returns:
        Optional[int]: The current intra-op thread count, or ``None`` when
        PyTorch is unavailable.
    """
    try:
        import torch  # local import: caller has normally imported it already

        applied = int(torch.get_num_threads())
    except Exception as exc:  # pragma: no cover - defensive, torch builds vary
        logger.debug("Could not read the PyTorch CPU thread count: %s", exc)
        return None
    _log_once(
        "torch_native_platform",
        "Native full-speed threading active (%s): PyTorch intra-op pool left at its "
        "default (%d thread(s)) - no torch.set_num_threads cap applied.",
        sys.platform,
        applied,
    )
    return applied


def apply_torch_thread_cap(threads: Optional[int] = None) -> Optional[int]:
    """
    Configure PyTorch's intra-op thread pool via ``torch.set_num_threads``.

    Call this immediately after ``import torch`` - the environment variables set
    by :func:`apply_cpu_thread_env` only cover the OpenMP/MKL runtimes, whereas
    the PyTorch intra-op pool is configured through this API. Because ``torch``
    is imported at this point, the CUDA decision here is the authoritative
    ``torch.cuda.is_available()`` answer.

    Behaviour (v1.7.3):

        * macOS / Linux: nothing is changed; PyTorch keeps its native pool.
        * Windows + CUDA: ``torch.set_num_threads(os.cpu_count())`` so the CPU
          post-processing of the flow dynamics uses every core.
        * Windows CPU-only (or ``PHASEPIX_FORCE_THREAD_CAP=1``): the pool is
          capped at 4 threads to prevent the MKL/OpenMP deadlock.

    Args:
        threads: Explicit thread count; ``None`` reuses the exported safe cap on
            the guarded path (ignored on the native path).

    Returns:
        Optional[int]: The applied thread count, or ``None`` when PyTorch does
        not expose the API (never raises, so startup cannot be broken).
    """
    if not should_apply_thread_cap():
        if is_windows():
            target = native_thread_count()
            applied = _set_torch_threads(target)
            _log_once(
                "torch_native_cuda",
                "Native full-speed threading active (win32, NVIDIA CUDA backend): PyTorch "
                "intra-op pool set to %s thread(s) - CPU thread guard disabled.",
                applied if applied is not None else target,
            )
            return applied
        return _report_torch_threads()

    cap = max(1, int(threads)) if threads is not None else (
        _effective_cpu_threads or resolve_cpu_thread_cap()
    )
    applied = _set_torch_threads(cap)
    # Keep the environment in sync: when ``torch`` reports no CUDA although the
    # early ``nvidia-smi`` probe saw a GPU, this is the first moment the guard is
    # known to be needed, so export the cap for runtimes that initialise later.
    _export_thread_env(cap)
    _log_once(
        "torch_guarded",
        "Windows CPU-fallback thread guard engaged: PyTorch CPU thread pool capped at %s "
        "thread(s) (requested %d) to prevent the MKL/OpenMP deadlock.",
        applied if applied is not None else cap,
        cap,
    )
    return applied

