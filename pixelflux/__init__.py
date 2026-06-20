"""pixelflux: X11/Wayland screen capture as a native CPython extension.

The X11 capture/encode core and the zero-copy result type live in the compiled
``pixelflux._capture`` module. The Wayland path is served by the optional Rust
``pixelflux_wayland`` backend. This package exposes a unified ``ScreenCapture``
plus a plain-Python settings holder and interpreter-exit safety.
"""

import atexit
import os
import weakref

from ._capture import (
    ScreenCapture as _ScreenCapture,
    StripeFrame,
    stripe_frame_from_buffer,
)

__all__ = ["ScreenCapture", "CaptureSettings", "StripeFrame", "stripe_frame_from_buffer"]


class CaptureSettings:
    """Capture/encode configuration. Defaults mirror the C++ defaults.

    Set fields as attributes, e.g. ``s = CaptureSettings(); s.capture_width = ...``.
    The native ``ScreenCapture.start_capture(settings, callback)`` reads these 32
    fields by attribute name. Field names match the former ctypes struct so
    existing callers keep working. (Not using __slots__: callers may stash extra
    attributes, e.g. ``recording_socket``, as they could on the old ctypes type.)
    """

    def __init__(self):
        self.capture_width = 0
        self.capture_height = 0
        self.scale = 1.0
        self.capture_x = 0
        self.capture_y = 0
        self.target_fps = 60.0
        self.jpeg_quality = 0
        self.paint_over_jpeg_quality = 0
        self.use_paint_over_quality = False
        self.paint_over_trigger_frames = 0
        self.damage_block_threshold = 0
        self.damage_block_duration = 0
        self.output_mode = 0
        self.h264_crf = 0
        self.h264_paintover_crf = 0
        self.h264_paintover_burst_frames = 0
        self.h264_fullcolor = False
        self.h264_fullframe = False
        self.h264_streaming_mode = False
        self.capture_cursor = False
        self.watermark_path = None
        self.watermark_location_enum = 0
        self.vaapi_render_node_index = 0
        self.use_cpu = False
        self.debug_logging = False
        self.h264_cbr_mode = False
        self.h264_bitrate_kbps = 0
        self.h264_vbv_buffer_size_kb = 0
        self.auto_adjust_screen_capture_size = False
        self.omit_stripe_headers = False
        self.deferred_free = False  # ignored by the C-API (it always takes ownership)
        self.vaapi_render_node_path = None


_GLOBAL_WAYLAND_BACKEND = None
# The ScreenCapture instance that currently owns the RUNNING process-wide Wayland
# capture (the latest one to call backend.start_capture()). Only the current owner may
# stop the backend, so a sibling instance (the input injector, which never starts
# capture, or an earlier display whose capture was taken over) can't tear down a
# capture still in use.
_WAYLAND_CAPTURE_OWNER = None
_WAYLAND_MODULE = None
if os.environ.get("PIXELFLUX_WAYLAND") == "true":
    try:
        from . import pixelflux_wayland as _WAYLAND_MODULE
        print(">> [PixelFlux] Rust Wayland Backend module loaded.")
    except ImportError as e:
        print(f">> [PixelFlux] Failed to load Wayland backend: {e}")
        _WAYLAND_MODULE = None


def _preload_wheel_libnvrtc():
    """The optional CUDA color-convert path (screen_capture_module.cpp) dlopen()s libnvrtc by
    soname. `pip install pixelflux[cuda]` (cu11/cu12/cu13) installs it under
    site-packages/nvidia/cuda_nvrtc/lib/, which is NOT on the default loader path, so the C++
    soname dlopen would miss it and silently fall back to libyuv. Preload it here (RTLD_GLOBAL)
    so the C++ soname dlopen (or its RTLD_DEFAULT fallback) resolves to this copy regardless of
    CUDA major. No-op when the wheel is absent (system libnvrtc / libyuv still apply)."""
    try:
        import ctypes, glob, importlib.util
        spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
        if not spec or not spec.submodule_search_locations:
            return
        libdir = os.path.join(list(spec.submodule_search_locations)[0], "lib")
        # Load the highest-versioned libnvrtc.so* present (handles cu11 .so.11.2 / cu12 .so.12 /
        # cu13 .so.13); rpath=$ORIGIN pulls its builtins dep. Load the first that opens.
        for so in sorted(glob.glob(os.path.join(libdir, "libnvrtc.so*")), reverse=True):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue
    except Exception:
        pass


_preload_wheel_libnvrtc()


def _construct_wayland_backend(w, h, node):
    """Construct the process-wide Wayland backend exactly once. The
    ``_GLOBAL_WAYLAND_BACKEND is None`` guard makes this idempotent, so callers
    racing to construct it (capture start vs. early input injection) build a
    single backend; the first StartCapture resizes it as needed."""
    global _GLOBAL_WAYLAND_BACKEND
    if _WAYLAND_MODULE is None:
        return None
    if _GLOBAL_WAYLAND_BACKEND is None:
        _GLOBAL_WAYLAND_BACKEND = _WAYLAND_MODULE.WaylandBackend(w, h, node)
        print(">> [PixelFlux] Rust Wayland Backend initialized "
              f"({w}x{h}, node='{node}').")
    return _GLOBAL_WAYLAND_BACKEND


def _ensure_wayland_backend(settings):
    """Construct the Wayland backend lazily, once, with the initial size + render
    node forwarded from selkies (so the device library no longer reads
    SELKIES_MANUAL_WIDTH/HEIGHT/MAX_RES/DRINODE itself). AUTO_GPU auto-detection
    stays in the Rust backend for the empty-node case."""
    if _WAYLAND_MODULE is None:
        return None
    node_b = getattr(settings, 'vaapi_render_node_path', None)
    # Field may now be a str (plain-Python settings) or bytes (legacy callers).
    if isinstance(node_b, bytes):
        node = node_b.decode('utf-8')
    elif node_b:
        node = str(node_b)
    else:
        node = ""
    w = int(getattr(settings, 'capture_width', 0) or 0)
    h = int(getattr(settings, 'capture_height', 0) or 0)
    return _construct_wayland_backend(w, h, node)


def _ensure_wayland_backend_for_input():
    """Ensure the backend exists for input injection / cursor / keymap access that
    may happen BEFORE any ScreenCapture.start_capture (selkies' input injector
    injects independently of capture). Build with safe defaults (0x0, empty render
    node -> backend AUTO_GPU-detects); the first StartCapture resizes it."""
    return _construct_wayland_backend(0, 0, "")


# __del__ isn't reliably run at interpreter shutdown, leaving the capture thread
# unjoined -> std::terminate. atexit runs while the interpreter is still alive, so it
# deterministically stops every live capture first. WeakSet => registration keeps
# no instance alive.
_live_captures = weakref.WeakSet()


def _stop_all_captures():
    for cap in list(_live_captures):
        try:
            cap.stop_capture()
        except Exception:
            pass


atexit.register(_stop_all_captures)


class ScreenCapture(_ScreenCapture):
    """Unified screen capture over the native X11 C-API or the Rust Wayland backend.

    ``start_capture(settings, callback)`` delivers a :class:`StripeFrame` to
    ``callback``; consume it with ``bytes(frame)`` / ``memoryview(frame)`` and read
    ``frame.data_type`` / ``frame.stripe_y_start`` / ``frame.stripe_height`` /
    ``frame.frame_id``.
    """

    # __new__ (C tp_new) creates the C++ module; __init__ sets up Python-side state.
    def __init__(self):
        super().__init__()
        self._is_capturing = False
        # Ownership of the process-wide Wayland capture is tracked module-side via
        # _WAYLAND_CAPTURE_OWNER (the latest start_capture() caller), so a sibling instance
        # (input injector, or a display whose capture was taken over) can't stop a capture
        # still in use. selkies runs several instances per process: capture + input injector.
        _live_captures.add(self)

    def start_capture(self, settings, callback):
        if self._is_capturing:
            raise ValueError("Capture already started.")
        if not callable(callback):
            raise TypeError("callback must be callable.")

        if _ensure_wayland_backend(settings):
            if settings.scale < 0.1:
                if settings.debug_logging:
                    print(f">> [PixelFlux] Warning: Scale {settings.scale} is invalid. Defaulting to 1.0")
                settings.scale = 1.0

            if settings.debug_logging:
                print(f">> [PixelFlux] Connecting to Rust Wayland Backend (Scale: {settings.scale})...")

            is_h264 = (settings.output_mode == 1)
            default_height = int(getattr(settings, 'capture_height', 0) or 0)
            omit_headers = bool(getattr(settings, 'omit_stripe_headers', False))

            def rust_bridge_callback(frame):
                # Headers omitted: payload is raw; Rust already set the metadata
                # (data_type/frame_id/stripe_y_start/stripe_height) as frame attrs.
                if omit_headers:
                    callback(frame)
                    return
                # Otherwise parse the header from a zero-copy memoryview (same byte
                # offsets as before), set the parsed metadata onto the frame's attrs,
                # then deliver it as-is (no copy).
                mv = memoryview(frame)
                size = len(mv)
                if is_h264:
                    data_type = 2
                    frame_id = int.from_bytes(mv[2:4], 'big') if size >= 4 else 0
                    y_start = int.from_bytes(mv[4:6], 'big') if size >= 6 else 0
                    # Real per-stripe height is in the packet header (bytes 8:10).
                    height = int.from_bytes(mv[8:10], 'big') if size >= 10 else default_height
                else:
                    data_type = 1
                    frame_id = int.from_bytes(mv[0:2], 'big') if size >= 2 else 0
                    y_start = int.from_bytes(mv[2:4], 'big') if size >= 4 else 0
                    height = 0
                frame.data_type = data_type
                frame.frame_id = frame_id
                frame.stripe_y_start = y_start
                frame.stripe_height = height
                callback(frame)

            _GLOBAL_WAYLAND_BACKEND.start_capture(rust_bridge_callback, settings)
            # This instance is now the latest to start the process-wide capture, so it
            # becomes the owner; only the current owner may stop the backend (so a sibling
            # capture or the input injector can't tear down a capture still in use).
            global _WAYLAND_CAPTURE_OWNER
            _WAYLAND_CAPTURE_OWNER = self
            self._is_capturing = True
            return

        # X11 path: the trampoline delivers a StripeFrame straight to `callback`.
        super().start_capture(settings, callback)
        self._is_capturing = True

    def stop_capture(self):
        if not self._is_capturing:
            return
        # Stop the native X11 capture (no-op on the Wayland path; its module never started).
        super().stop_capture()
        # Only stop the process-wide Wayland backend if THIS instance is the current
        # capture owner, so we don't tear down a backend other instances (selkies' input
        # injector, or a display that took capture over) still use.
        global _WAYLAND_CAPTURE_OWNER
        if _GLOBAL_WAYLAND_BACKEND and _WAYLAND_CAPTURE_OWNER is self:
            _GLOBAL_WAYLAND_BACKEND.stop_capture()
            _WAYLAND_CAPTURE_OWNER = None
        self._is_capturing = False

    # Input/cursor entry points lazily construct the Wayland backend so injection
    # works before the first start_capture (selkies injects independently). They
    # stay no-ops in X11 mode (no Wayland module -> helper returns None).
    def inject_key(self, scancode, state):
        backend = _ensure_wayland_backend_for_input()
        if backend:
            backend.inject_key(scancode, state)

    def inject_mouse_move(self, x, y):
        backend = _ensure_wayland_backend_for_input()
        if backend:
            backend.inject_mouse_move(float(x), float(y))

    def inject_relative_mouse_move(self, dx, dy):
        backend = _ensure_wayland_backend_for_input()
        if backend:
            backend.inject_relative_mouse_move(float(dx), float(dy))

    def inject_mouse_button(self, btn, state):
        backend = _ensure_wayland_backend_for_input()
        if backend:
            backend.inject_mouse_button(btn, state)

    def inject_mouse_scroll(self, x, y):
        backend = _ensure_wayland_backend_for_input()
        if backend:
            backend.inject_mouse_scroll(float(x), float(y))

    def set_cursor_rendering(self, enabled):
        backend = _ensure_wayland_backend_for_input()
        if backend:
            backend.set_cursor_rendering(bool(enabled))

    def set_cursor_callback(self, callback):
        backend = _ensure_wayland_backend_for_input()
        if backend:
            backend.set_cursor_callback(callback)

    def request_idr_frame(self):
        if not self._is_capturing:
            return
        # Route Wayland-first, matching start_capture's precedence (else a dual-active
        # config would misroute the IDR to the idle X11 module).
        if _GLOBAL_WAYLAND_BACKEND is not None:
            _wl_request_idr = getattr(_GLOBAL_WAYLAND_BACKEND, "request_idr_frame", None)
            if _wl_request_idr is not None:
                _wl_request_idr()
        else:
            super().request_idr_frame()

    def update_video_bitrate(self, bitrate):
        if self._is_capturing and _GLOBAL_WAYLAND_BACKEND is None:
            super().update_video_bitrate(bitrate)

    def update_framerate(self, fps):
        if self._is_capturing and _GLOBAL_WAYLAND_BACKEND is None:
            super().update_framerate(float(fps))

    def update_vbv_buf_size(self, buffer_size):
        if self._is_capturing and _GLOBAL_WAYLAND_BACKEND is None:
            super().update_vbv_buffer_size(buffer_size)
