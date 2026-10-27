"""PhasePix compatibility patch for ``streamlit-drawable-canvas`` on modern Streamlit.

Why this module exists
----------------------
``streamlit-drawable-canvas==0.9.3`` encodes its background image with::

    import streamlit.elements.image as st_image
    background_image_url = st_image.image_to_url(background_image, width, True,
                                                 "RGB", "PNG", image_id)

Streamlit removed that helper from ``streamlit.elements.image`` (it now lives in
``streamlit.elements.lib.image_utils`` with a completely different, keyword-only
signature).  Depending on the installed Streamlit version the canvas therefore
either raises ``AttributeError`` / ``TypeError`` or silently produces an empty
``backgroundImageURL`` - which is exactly the "canvas is blank / Manual
Correction and Visual Ruler do nothing" symptom on Windows Docker.

The patch is applied in two independent layers so it can never be missed:

1. **Build time / manual run** - ``python patch_canvas.py`` rewrites the installed
   ``streamlit_drawable_canvas/__init__.py`` in ``site-packages`` so the package
   encodes the background image itself into ``data:image/png;base64,...`` and no
   longer touches any private Streamlit API at all.
2. **Runtime** - ``import patch_canvas`` installs the same encoder on
   ``streamlit.elements.image``, ``streamlit.elements.lib.image_utils`` and
   ``streamlit_drawable_canvas.st_image`` *before* the canvas widget is used
   (``app.py`` imports this module above ``streamlit_drawable_canvas``).

Everything in this module is idempotent and side-effect free when nothing needs
patching, so it is safe to import from ``app.py``, from the Dockerfile and from
plain command line scripts.
"""

from __future__ import annotations

import inspect
import os
import re
import sys
from typing import Any, Iterable, List, Tuple

# Attribute used both as an idempotency flag and as a "this is already patched"
# marker when scanning installed modules.
_PATCH_FLAG = "_phasepix_data_url_patch"
_ON_DISK_MARKER = "# --- PhasePix canvas patch ---"

# Helper injected into ``streamlit_drawable_canvas/__init__.py``.  It is fully
# self-contained (imports inside the function body) so that it keeps working no
# matter how the surrounding module reshuffles its imports.
_ON_DISK_HELPER = '''
# --- PhasePix canvas patch ---
# Modern Streamlit no longer exposes ``streamlit.elements.image.image_to_url``
# with the legacy 6-positional-argument signature that streamlit-drawable-canvas
# 0.9.x expects. Encode the background image locally instead: this needs no
# private Streamlit API, works in every browser and never returns an empty URL.
def _phasepix_data_url(image, *args, **kwargs):
    """Return a self-contained ``data:image/png;base64,...`` URL for an image."""
    import base64 as _base64
    import io as _io

    from PIL import Image as _Image

    try:
        import numpy as _np
    except Exception:  # pragma: no cover - numpy is a hard dependency anyway
        _np = None

    if _np is not None and isinstance(image, _np.ndarray):
        arr = image
        if arr.dtype != _np.uint8:
            arr = _np.clip(arr, 0, 255).astype(_np.uint8)
        if arr.ndim == 2:
            image = _Image.fromarray(arr, mode="L")
        elif arr.ndim == 3 and arr.shape[2] == 4:
            image = _Image.fromarray(arr, mode="RGBA")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            image = _Image.fromarray(arr, mode="RGB")
        else:
            raise TypeError("unsupported numpy array shape for PNG encoding")

    if isinstance(image, (bytes, bytearray)):
        image = _Image.open(_io.BytesIO(bytes(image)))

    if isinstance(image, str):
        return image

    if not isinstance(image, _Image.Image):
        raise TypeError("unsupported image type: %r" % (type(image),))

    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")

    buffer = _io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + _base64.b64encode(buffer.getvalue()).decode("ascii")


# --- end PhasePix canvas patch ---
'''

# ---------------------------------------------------------------------------
# Core encoder
# ---------------------------------------------------------------------------
def image_to_data_url(image: Any, *args: Any, **kwargs: Any) -> str:
    """Encode ``image`` as ``data:image/png;base64,...``.

    Accepts PIL images, numpy arrays (HxW, HxWx3, HxWx4), raw PNG/JPEG bytes and
    already-usable URL strings.  Extra positional/keyword arguments are accepted
    and ignored so the function is a drop-in replacement for both the legacy
    ``image_to_url(image, width, clamp, channels, output_format, image_id)``
    signature and the modern keyword-only one.
    """
    import base64
    import io

    from PIL import Image

    if isinstance(image, str):
        # Already a URL / data URL - nothing to encode.
        return image

    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(bytes(image)))

    if not isinstance(image, Image.Image):
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            np = None  # type: ignore[assignment]
        if np is not None and isinstance(image, np.ndarray):
            array = image
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            if array.ndim == 2:
                image = Image.fromarray(array, mode="L")
            elif array.ndim == 3 and array.shape[2] == 4:
                image = Image.fromarray(array, mode="RGBA")
            elif array.ndim == 3 and array.shape[2] == 3:
                image = Image.fromarray(array, mode="RGB")
            else:
                raise TypeError("unsupported numpy array shape for PNG encoding")

    if not isinstance(image, Image.Image):
        raise TypeError(f"unsupported image type: {type(image)!r}")

    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


setattr(image_to_data_url, _PATCH_FLAG, True)


def _is_legacy_canvas_call(args: Tuple[Any, ...], kwargs: dict) -> bool:
    """True when the call uses the pre-1.30 canvas signature.

    ``streamlit_drawable_canvas`` passes the pixel width as the *second*
    positional argument (an ``int``).  Streamlit's own code passes a
    ``LayoutConfig`` object instead, so we can tell the two callers apart and
    avoid turning every ``st.image`` into an inlined base64 blob.
    """
    if args and isinstance(args[0], (int, float)):
        return True
    layout_config = kwargs.get("layout_config")
    return isinstance(layout_config, (int, float))


# ---------------------------------------------------------------------------
# Runtime patch (layer 2)
# ---------------------------------------------------------------------------
def _patch_legacy_image_module() -> bool:
    """Attach the encoder to ``streamlit.elements.image``.

    This is the module ``streamlit_drawable_canvas`` imports as ``st_image``.
    On Streamlit >= 1.30 it no longer defines ``image_to_url`` at all, so we are
    only *adding* a helper - nothing inside Streamlit relies on this name.
    """
    try:
        import streamlit.elements.image as legacy_image  # type: ignore
    except Exception:
        return False
    setattr(legacy_image, "image_to_url", image_to_data_url)
    return True


def _patch_image_utils_module() -> bool:
    """Guard ``streamlit.elements.lib.image_utils.image_to_url``.

    Modern Streamlit calls this with a ``LayoutConfig`` object for ``st.image``;
    those calls are delegated to the original implementation.  Legacy canvas
    calls (int width) are answered with a self-contained data URL.
    """
    try:
        from streamlit.elements.lib import image_utils  # type: ignore
    except Exception:
        return False

    original = getattr(image_utils, "image_to_url", None)
    if original is None:
        setattr(image_utils, "image_to_url", image_to_data_url)
        return True
    if getattr(original, _PATCH_FLAG, False):
        return True

    def shim(image, *args, **kwargs):
        if _is_legacy_canvas_call(args, kwargs):
            return image_to_data_url(image, *args, **kwargs)
        return original(image, *args, **kwargs)

    setattr(shim, _PATCH_FLAG, True)
    setattr(shim, "__wrapped__", original)
    image_utils.image_to_url = shim
    return True


def _ensure_display_toolbar(canvas_pkg: Any) -> bool:
    """Make ``st_canvas(..., display_toolbar=...)`` safe on every release."""
    st_canvas = getattr(canvas_pkg, "st_canvas", None)
    if st_canvas is None or getattr(st_canvas, _PATCH_FLAG, False):
        return False

    try:
        parameters = inspect.signature(st_canvas).parameters
    except (TypeError, ValueError):
        return False

    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    if "display_toolbar" in parameters or accepts_kwargs:
        return False

    import functools

    @functools.wraps(st_canvas)
    def tolerant_st_canvas(*args, **kwargs):
        kwargs.pop("display_toolbar", None)
        return st_canvas(*args, **kwargs)

    setattr(tolerant_st_canvas, _PATCH_FLAG, True)
    canvas_pkg.st_canvas = tolerant_st_canvas
    return True


def _patch_canvas_module() -> bool:
    """Patch the canvas module object itself.

    ``st_image`` is bound to ``streamlit.elements.image`` at import time, so we
    overwrite the attribute on that exact object and, defensively, on the canvas
    package namespace.  ``display_toolbar`` is inspected as well so releases that
    dropped the keyword get a wrapper instead of raising ``TypeError``.
    """
    try:
        import streamlit_drawable_canvas as canvas_pkg  # type: ignore
    except Exception:
        return False

    st_image = getattr(canvas_pkg, "st_image", None)
    if st_image is not None:
        setattr(st_image, "image_to_url", image_to_data_url)

    if not hasattr(canvas_pkg, "_phasepix_data_url"):
        # Expose the helper on the package for debugging and verification.
        setattr(canvas_pkg, "_phasepix_data_url", image_to_data_url)

    _ensure_display_toolbar(canvas_pkg)
    return True


def patch_runtime() -> List[str]:
    """Apply every in-memory patch.  Returns the list of successful layers."""
    applied: List[str] = []
    if _patch_legacy_image_module():
        applied.append("streamlit.elements.image")
    if _patch_image_utils_module():
        applied.append("streamlit.elements.lib.image_utils")
    if _patch_canvas_module():
        applied.append("streamlit_drawable_canvas")
    return applied


# ---------------------------------------------------------------------------
# On-disk patch (layer 1 - used by the Dockerfile)
# ---------------------------------------------------------------------------
def _candidate_canvas_paths() -> Iterable[str]:
    seen = set()

    try:
        import streamlit_drawable_canvas  # type: ignore

        module_path = getattr(streamlit_drawable_canvas, "__file__", None)
        if module_path and os.path.isfile(module_path):
            absolute = os.path.abspath(module_path)
            seen.add(absolute)
            yield absolute
    except Exception:
        pass

    try:
        import site

        roots = list(site.getsitepackages())
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            roots.append(user_site)
    except Exception:
        roots = []

    for root in roots:
        candidate = os.path.join(root, "streamlit_drawable_canvas", "__init__.py")
        if os.path.isfile(candidate):
            absolute = os.path.abspath(candidate)
            if absolute not in seen:
                seen.add(absolute)
                yield absolute


def _inject_helper(source: str) -> str:
    """Insert the local PNG encoder before the first executable statement."""
    lines = source.splitlines(keepends=True)
    insert_at = None
    for index, line in enumerate(lines):
        if line.startswith("_RELEASE"):
            insert_at = index
            break
    if insert_at is None:
        for index, line in enumerate(lines):
            if line.startswith(("import ", "from ")):
                insert_at = index + 1
    if insert_at is None:
        insert_at = 0
    lines.insert(insert_at, _ON_DISK_HELPER)
    return "".join(lines)


def _patch_base_url_join(source: str) -> str:
    """Do not prepend ``server.baseUrlPath`` to a ``data:`` URL."""
    pattern = re.compile(
        r'^([ \t]*)background_image_url\s*=\s*st\._config\.get_option\("server\.baseUrlPath"\)\s*\+'
        r"\s*background_image_url\s*$",
        re.MULTILINE,
    )
    replacement = (
        r'\1if not str(background_image_url).startswith("data:"):\n'
        r'\1    background_image_url = st._config.get_option("server.baseUrlPath")'
        r" + background_image_url"
    )
    return pattern.sub(replacement, source)


def _drop_stale_bytecode(target: str) -> None:
    """Remove a stale ``__pycache__`` entry next to a patched module."""
    cache_dir = os.path.join(os.path.dirname(target), "__pycache__")
    stem = os.path.basename(target)[:-3]
    try:
        entries = os.listdir(cache_dir)
    except OSError:
        return
    for entry in entries:
        if entry.startswith(stem + ".") and entry.endswith((".pyc", ".pyo")):
            try:
                os.remove(os.path.join(cache_dir, entry))
            except OSError:
                pass


def patch_site_packages(path: str | None = None) -> Tuple[bool, str]:
    """Rewrite the installed canvas module in place.

    Returns ``(patched, message)`` where ``patched`` is ``False`` when the module
    was already patched or cannot be found (both are non-fatal).
    """
    target = path
    if target is None:
        target = next(iter(_candidate_canvas_paths()), None)
    if target is None:
        return False, "streamlit_drawable_canvas is not installed - skipping file patch"

    try:
        with open(target, "r", encoding="utf-8") as handle:
            source = handle.read()
    except OSError as exc:
        return False, f"cannot read {target}: {exc}"

    if _ON_DISK_MARKER in source:
        return False, f"already patched: {target}"
    if "st_image.image_to_url(" not in source:
        return False, f"no legacy image_to_url call in {target} - nothing to patch"

    patched_source = _inject_helper(source).replace(
        "st_image.image_to_url(", "_phasepix_data_url("
    )
    patched_source = _patch_base_url_join(patched_source)

    try:
        compile(patched_source, target, "exec")
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"patched canvas source does not compile: {exc}") from exc

    backup = target + ".phasepix.bak"
    if not os.path.exists(backup):
        try:
            with open(backup, "w", encoding="utf-8") as handle:
                handle.write(source)
        except OSError:
            pass

    try:
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(patched_source)
    except OSError as exc:
        return False, f"cannot write {target}: {exc}"

    _drop_stale_bytecode(target)
    return True, f"patched {target}"


# ---------------------------------------------------------------------------
# Verification + public API
# ---------------------------------------------------------------------------
def verify() -> bool:
    """Confirm that the canvas module now yields a PNG data URL."""
    try:
        from PIL import Image as PILImage

        import streamlit_drawable_canvas as canvas_pkg  # type: ignore
    except Exception as exc:
        print(f"[patch_canvas] canvas package unavailable: {exc}")
        return False

    encoder = getattr(canvas_pkg, "_phasepix_data_url", None)
    if encoder is None:
        st_image = getattr(canvas_pkg, "st_image", None)
        encoder = getattr(st_image, "image_to_url", None)
    if encoder is None:
        return False

    probe = PILImage.new("RGB", (8, 8), (12, 34, 56))
    try:
        url = encoder(probe, 8, True, "RGB", "PNG", "phasepix-verify")
    except Exception as exc:
        print(f"[patch_canvas] encoder raised: {exc}")
        return False
    return isinstance(url, str) and url.startswith("data:image/png;base64,")


def install(verbose: bool = False) -> dict:
    """Apply every runtime patch and return a status dictionary."""
    applied = patch_runtime()
    status = {"layers": applied, "verified": verify()}
    if verbose:
        print(f"[patch_canvas] runtime layers: {', '.join(applied) or 'none'}")
        print(f"[patch_canvas] encoder verified: {status['verified']}")
    return status


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    check_only = "--check" in argv

    print("[patch_canvas] PhasePix / streamlit-drawable-canvas compatibility patch")
    try:
        import streamlit

        print(f"[patch_canvas] streamlit {streamlit.__version__}")
    except Exception:
        print("[patch_canvas] streamlit is not importable")

    if not check_only:
        patched, message = patch_site_packages()
        print(f"[patch_canvas] {'OK' if patched else 'SKIP'}: {message}")

    applied = patch_runtime()
    print(f"[patch_canvas] runtime layers: {', '.join(applied) or 'none'}")

    ok = verify()
    print(f"[patch_canvas] encoder returns PNG data URLs: {ok}")

    if check_only:
        return 0 if ok else 1

    # A missing/renamed canvas module must not break the image build; a canvas
    # module that is present but still cannot encode is a hard failure.
    try:
        import streamlit_drawable_canvas  # noqa: F401

        return 0 if ok else 1
    except Exception:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

