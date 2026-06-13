import atexit
import ctypes
import os
import threading
import weakref

class CaptureSettings(ctypes.Structure):
    _fields_ = [
        ("capture_width", ctypes.c_int),
        ("capture_height", ctypes.c_int),
        ("scale", ctypes.c_double),
        ("capture_x", ctypes.c_int),
        ("capture_y", ctypes.c_int),
        ("target_fps", ctypes.c_double),
        ("jpeg_quality", ctypes.c_int),
        ("paint_over_jpeg_quality", ctypes.c_int),
        ("use_paint_over_quality", ctypes.c_bool),
        ("paint_over_trigger_frames", ctypes.c_int),
        ("damage_block_threshold", ctypes.c_int),
        ("damage_block_duration", ctypes.c_int),
        ("output_mode", ctypes.c_int),
        ("h264_crf", ctypes.c_int),
        ("h264_paintover_crf", ctypes.c_int),
        ("h264_paintover_burst_frames", ctypes.c_int),
        ("h264_fullcolor", ctypes.c_bool),
        ("h264_fullframe", ctypes.c_bool),
        ("h264_streaming_mode", ctypes.c_bool),
        ("capture_cursor", ctypes.c_bool),
        ("watermark_path", ctypes.c_char_p),
        ("watermark_location_enum", ctypes.c_int),
        ("vaapi_render_node_index", ctypes.c_int),
        ("use_cpu", ctypes.c_bool),
        ("debug_logging", ctypes.c_bool),
        ("h264_cbr_mode", ctypes.c_bool),
        ("h264_bitrate_kbps", ctypes.c_int),
        ("h264_vbv_buffer_size_kb", ctypes.c_int),
        ("auto_adjust_screen_capture_size", ctypes.c_bool),
        ("omit_stripe_headers", ctypes.c_bool),
        ("deferred_free", ctypes.c_bool),
        ("vaapi_render_node_path", ctypes.c_char_p),
    ]

class StripeEncodeResult(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("stripe_y_start", ctypes.c_int),
        ("stripe_height", ctypes.c_int),
        ("size", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("frame_id", ctypes.c_int),
    ]

StripeCallback = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(StripeEncodeResult), ctypes.c_void_p
)

# Rebound to the real C function below if the .so loads; the None default keeps
# OwnedFrame's free path a clean no-op instead of a NameError when it didn't.
free_stripe_encode_result_data = None


class OwnedFrame:
    """Refcount-owned, zero-copy view over a C-allocated encoded frame.

    In deferred-free mode C++ hands the encoded buffer's ownership to Python instead of
    freeing it in its callback; this frees it (via free_stripe_encode_result_data) exactly
    once, on finalization. memoryview() returns a view that ALIASES the C buffer (no copy)
    and pins this OwnedFrame alive until every consumer -- including a transport that
    retained a slice during a partial write -- releases it, so the view can go straight to
    send_bytes. This works on every Python version with no PEP 688.
    """
    # Pin contract: memoryview -> ctypes array -> self (via arr._pf_owner), so while any
    # view or slice lives, self can't be collected and __del__ (the single free) can't run.
    __slots__ = ("_data_ptr", "_ptr_value", "size", "frame_id", "_freed", "__weakref__")

    def __init__(self, data_ptr, size, frame_id):
        self._data_ptr = data_ptr
        # Integer address for from_address(); kept separate from the ctypes pointer so the
        # free path (which needs the typed pointer) and memoryview() don't share storage.
        self._ptr_value = ctypes.cast(data_ptr, ctypes.c_void_p).value
        self.size = size
        self.frame_id = frame_id
        self._freed = False

    def memoryview(self):
        if self._freed:
            raise ValueError("OwnedFrame already freed")
        arr = (ctypes.c_char * self.size).from_address(self._ptr_value)  # aliases C buffer
        arr._pf_owner = self   # pin: view -> arr -> self; __del__ (free) can't run under a live view
        return memoryview(arr)

    @staticmethod
    def take(result_ptr):
        """Take ownership of result_ptr.contents.data, returning an OwnedFrame (None if empty).

        Leak-safe entry point for deferred-free mode: in that mode pixelflux's callback
        does NOT free the buffer, so something must on every path -- if construction
        fails this frees it and re-raises. Call this BEFORE any step that can fail or
        early-return, so once it succeeds the OwnedFrame's refcount alone governs the free.
        """
        result = result_ptr.contents
        if not (result.data and result.size > 0):
            return None
        # Snapshot the address into an INDEPENDENT pointer: a pointer read from the ctypes
        # field aliases its storage, so nulling result.data below would also null our copy.
        data_ptr = ctypes.cast(
            ctypes.cast(result.data, ctypes.c_void_p).value, ctypes.POINTER(ctypes.c_ubyte)
        )
        try:
            frame = OwnedFrame(data_ptr, result.size, result.frame_id)
            # Null the field to transfer ownership: a still-set data pointer is the signal
            # to the callback's finally that nobody took the buffer and it must free it.
            result.data = ctypes.cast(None, ctypes.POINTER(ctypes.c_ubyte))
            return frame
        except BaseException:
            try:
                if free_stripe_encode_result_data is not None:
                    free_stripe_encode_result_data(result_ptr)
            except Exception:
                pass
            raise

    def _free(self):
        # The single guarded free, exactly once. Only reached from __del__: the pin means
        # this can't run while any view aliases the buffer, so the free is never a UAF.
        if not self._freed and self._data_ptr and free_stripe_encode_result_data is not None:
            self._freed = True
            r = StripeEncodeResult()
            r.data = self._data_ptr
            r.size = self.size
            free_stripe_encode_result_data(ctypes.byref(r))
            self._data_ptr = None

    def close(self):
        """No-op kept for API compatibility.

        Without PEP 688 we can't count live views, so close() must NOT free: a caller
        holding both this OwnedFrame and a still-live memoryview/slice would otherwise hit
        a use-after-free. The buffer is freed by __del__ once the pin's last view drops and
        this object is collected (prompt under CPython refcounting in the normal path where
        the consumer releases both the view and the owner).
        """
        pass

    def __del__(self):
        # Reached only after every pinning view is released; free the C buffer exactly once.
        # Swallow any error so finalization can't raise.
        try:
            self._free()
        except Exception:
            pass

lib_dir = os.path.dirname(__file__)
lib_path = os.path.join(lib_dir, 'screen_capture_module.so')

_legacy_lib = None
try:
    if os.path.exists(lib_path):
        _legacy_lib = ctypes.CDLL(lib_path)
except OSError:
    pass

# Bare-name fallback for system/LD_LIBRARY_PATH installs where the .so isn't packaged
# next to this module. The ABI guard below still runs on whatever loads, so a
# mismatched path-found library is rejected, not silently used.
if _legacy_lib is None:
    try:
        _legacy_lib = ctypes.CDLL('screen_capture_module.so')
    except OSError:
        pass

if _legacy_lib:
    create_module = _legacy_lib.create_screen_capture_module
    create_module.restype = ctypes.c_void_p
    destroy_module = _legacy_lib.destroy_screen_capture_module
    destroy_module.argtypes = [ctypes.c_void_p]
    start_capture_c = _legacy_lib.start_screen_capture
    start_capture_c.argtypes = [ctypes.c_void_p, CaptureSettings, StripeCallback, ctypes.c_void_p]
    stop_capture_c = _legacy_lib.stop_screen_capture
    stop_capture_c.argtypes = [ctypes.c_void_p]
    free_stripe_encode_result_data = _legacy_lib.free_stripe_encode_result_data
    free_stripe_encode_result_data.argtypes = [ctypes.POINTER(StripeEncodeResult)]
    request_idr = _legacy_lib.request_idr
    request_idr.argtypes = [ctypes.c_void_p]
    update_video_bitrate_c = _legacy_lib.update_video_bitrate
    update_video_bitrate_c.argtypes = [ctypes.c_void_p, ctypes.c_int]
    update_framerate_c = _legacy_lib.update_framerate
    update_framerate_c.argtypes = [ctypes.c_void_p, ctypes.c_double]
    update_vbv_buffer_size_c = _legacy_lib.update_vbv_buffer_size
    update_vbv_buffer_size_c.argtypes = [ctypes.c_void_p, ctypes.c_int]
    # Fail fast on any ctypes/C++ CaptureSettings ABI mismatch (field order/size).
    if hasattr(_legacy_lib, "pixelflux_capture_settings_size"):
        _legacy_lib.pixelflux_capture_settings_size.restype = ctypes.c_int
        _c_size = _legacy_lib.pixelflux_capture_settings_size()
        if _c_size != ctypes.sizeof(CaptureSettings):
            raise RuntimeError(
                f"CaptureSettings ABI mismatch: C++={_c_size} ctypes={ctypes.sizeof(CaptureSettings)}"
            )
    # Catch a same-size field reorder/rename the size check would miss. hasattr-guarded
    # so an older .so degrades to size-only. The hash mixes only (name, offset, size), so
    # it can't catch a same-name/same-size TYPE swap. MUST stay byte-for-byte identical to
    # the C++ side -- changing only this copy would make the guard mismatch on import.
    if hasattr(_legacy_lib, "pixelflux_capture_settings_layout_hash"):
        def _capture_settings_layout_hash(struct_cls):
            # FNV-1a-64 over each field's (name, '\0', offset<LE u32>, size<LE u32>),
            # byte-for-byte matching pixelflux_capture_settings_layout_hash() in C++.
            _FNV_OFFSET = 0xCBF29CE484222325
            _FNV_PRIME = 0x100000001B3
            _MASK = (1 << 64) - 1
            h = _FNV_OFFSET
            for _name, _ctype in struct_cls._fields_:
                _fld = getattr(struct_cls, _name)
                _stream = (
                    _name.encode("ascii") + b"\x00"
                    + (_fld.offset & 0xFFFFFFFF).to_bytes(4, "little")
                    + (_fld.size & 0xFFFFFFFF).to_bytes(4, "little")
                )
                for _b in _stream:
                    h ^= _b
                    h = (h * _FNV_PRIME) & _MASK
            return h

        _legacy_lib.pixelflux_capture_settings_layout_hash.restype = ctypes.c_uint64
        _c_layout = _legacy_lib.pixelflux_capture_settings_layout_hash()
        _py_layout = _capture_settings_layout_hash(CaptureSettings)
        if _c_layout != _py_layout:
            raise RuntimeError(
                "CaptureSettings layout hash mismatch (field reorder/rename?): "
                f"C++=0x{_c_layout:016x} ctypes=0x{_py_layout:016x}"
            )

_GLOBAL_WAYLAND_BACKEND = None
_WAYLAND_MODULE = None
if os.environ.get("PIXELFLUX_WAYLAND") == "true":
    try:
        from . import pixelflux_wayland as _WAYLAND_MODULE
        print(">> [PixelFlux] Rust Wayland Backend module loaded.")
    except ImportError as e:
        print(f">> [PixelFlux] Failed to load Wayland backend: {e}")
        _WAYLAND_MODULE = None


def _ensure_wayland_backend(settings):
    """Construct the Wayland backend lazily, once, with the initial size + render
    node forwarded from selkies (so the device library no longer reads
    SELKIES_MANUAL_WIDTH/HEIGHT/MAX_RES/DRINODE itself). AUTO_GPU auto-detection
    stays in the Rust backend for the empty-node case."""
    global _GLOBAL_WAYLAND_BACKEND
    if _WAYLAND_MODULE is None:
        return None
    if _GLOBAL_WAYLAND_BACKEND is None:
        node_b = getattr(settings, 'vaapi_render_node_path', None)
        node = node_b.decode('utf-8') if node_b else ""
        w = int(getattr(settings, 'capture_width', 0) or 0)
        h = int(getattr(settings, 'capture_height', 0) or 0)
        _GLOBAL_WAYLAND_BACKEND = _WAYLAND_MODULE.WaylandBackend(w, h, node)
        print(">> [PixelFlux] Rust Wayland Backend initialized "
              f"({w}x{h}, node='{node}').")
    return _GLOBAL_WAYLAND_BACKEND


# __del__ isn't reliably run at interpreter shutdown, leaving the capture thread
# unjoined -> std::terminate. atexit runs while the interpreter is still alive, so it
# deterministically stops every live capture first.
_live_captures = weakref.WeakSet()

def _stop_all_captures():
    for cap in list(_live_captures):
        try:
            if getattr(cap, '_is_capturing', False):
                cap.stop_capture()
        except Exception:
            pass

atexit.register(_stop_all_captures)


class ScreenCapture:
    """Python wrapper for screen capture module using ctypes."""

    def __init__(self):
        if _legacy_lib:
            self._module = create_module()
        else:
            self._module = None
        _live_captures.add(self)

        self._is_capturing = False
        self._python_stripe_callback = None
        self._c_callback = None
        # Introspection only: the free decision in _internal_c_callback is driven by
        # whether the C `data` pointer is still set, NOT this flag. Do NOT gate on it.
        self._deferred_free = False
        # True only on the instance whose start_capture() started the process-wide Wayland
        # backend, so its teardown doesn't stop a backend other instances still use
        # (selkies runs several per process: capture + input injector).
        self._owns_wayland_backend = False

    def __del__(self):
        try:
            if getattr(self, '_is_capturing', False):
                self.stop_capture()
            if getattr(self, '_module', None):
                destroy_module(self._module)
                self._module = None
        except Exception:
            pass

    def start_capture(self, settings: CaptureSettings, stripe_callback):
        if self._is_capturing:
            raise ValueError("Capture already started.")

        # Validate before storing so a non-callable doesn't leave a stale callback behind.
        if not callable(stripe_callback):
            raise TypeError("stripe_callback must be callable.")

        self._python_stripe_callback = stripe_callback

        if _ensure_wayland_backend(settings):
            # The Wayland bridge points result_struct.data into a transient Python
            # bytes object owned by CPython, so deferred-free (which would delete[] it
            # as if it were a C-allocated buffer) must never be active on this path.
            self._deferred_free = False
            if settings.scale < 0.1:
                if settings.debug_logging:
                    print(f">> [PixelFlux] Warning: Scale {settings.scale} is invalid. Defaulting to 1.0")
                settings.scale = 1.0

            if settings.debug_logging:
                print(f">> [PixelFlux] Connecting to Rust Wayland Backend (Scale: {settings.scale})...")
            
            is_h264 = (settings.output_mode == 1)

            def rust_bridge_callback(data_bytes):
                if not self._python_stripe_callback:
                    return
                size = len(data_bytes)
                # Point ctypes at the bytes buffer instead of copying it:
                # data_bytes is this function's argument, so it stays alive for
                # the whole synchronous callback below (its only consumer, which
                # copies before returning); result_struct.data holds only a raw
                # address into it.
                result_struct = StripeEncodeResult()
                result_struct.size = size
                result_struct.data = ctypes.cast(ctypes.c_char_p(data_bytes), ctypes.POINTER(ctypes.c_ubyte))
                if is_h264:
                    result_struct.type = 2
                    if size >= 4:
                        result_struct.frame_id = int.from_bytes(data_bytes[2:4], 'big')
                    else:
                        result_struct.frame_id = 0
                    if size >= 6:
                         result_struct.stripe_y_start = int.from_bytes(data_bytes[4:6], 'big')
                    else:
                         result_struct.stripe_y_start = 0
                    # Real per-stripe height is in the packet header (bytes 8:10).
                    if size >= 10:
                        result_struct.stripe_height = int.from_bytes(data_bytes[8:10], 'big')
                    else:
                        result_struct.stripe_height = settings.capture_height
                else:
                    result_struct.type = 1
                    if size >= 2:
                        result_struct.frame_id = int.from_bytes(data_bytes[0:2], 'big')
                    else:
                        result_struct.frame_id = 0
                    if size >= 4:
                        result_struct.stripe_y_start = int.from_bytes(data_bytes[2:4], 'big')
                    else:
                        result_struct.stripe_y_start = 0
                    result_struct.stripe_height = 0
                self._python_stripe_callback(ctypes.byref(result_struct), None)
            try:
                _GLOBAL_WAYLAND_BACKEND.start_capture(rust_bridge_callback, settings)
            except Exception:
                self._python_stripe_callback = None
                self._deferred_free = False
                raise
            # This instance started the process-wide backend, so only it may stop it.
            self._owns_wayland_backend = True
            self._is_capturing = True
            return

        if not self._module:
             raise OSError("Legacy screen_capture_module.so not found.")

        self._c_callback = StripeCallback(self._internal_c_callback)
        self._deferred_free = bool(getattr(settings, 'deferred_free', False))
        # Publish _is_capturing before the producer thread starts so its first
        # callbacks aren't dropped (and their buffers leaked).
        self._is_capturing = True
        try:
            start_capture_c(self._module, settings, self._c_callback, None)
        except Exception:
            self._is_capturing = False
            self._c_callback = None
            self._python_stripe_callback = None
            self._deferred_free = False
            raise

    def stop_capture(self):
        if not self._is_capturing:
            return
        
        if self._module and self._c_callback:
            stop_capture_c(self._module)
            self._c_callback = None

        # Only stop the process-wide Wayland backend if THIS instance started it, so we
        # don't tear down a backend other instances (selkies' input injector) still use.
        if _GLOBAL_WAYLAND_BACKEND and self._owns_wayland_backend:
             _GLOBAL_WAYLAND_BACKEND.stop_capture()
             self._owns_wayland_backend = False

        self._is_capturing = False
        self._python_stripe_callback = None

    def _internal_c_callback(self, result_ptr, user_data):
        try:
            if self._is_capturing and self._python_stripe_callback:
                self._python_stripe_callback(result_ptr, user_data)
        finally:
            # OwnedFrame.take() nulls data on a successful take, so a still-set data
            # pointer means nobody took the buffer and we must free it (non-deferred mode,
            # teardown, or a deferred callback that returned before take()). A nulled
            # pointer means the OwnedFrame owns the free, so freeing here would double-free.
            if result_ptr.contents.data:
                free_stripe_encode_result_data(result_ptr)

    def inject_key(self, scancode, state):
        if _GLOBAL_WAYLAND_BACKEND:
            _GLOBAL_WAYLAND_BACKEND.inject_key(scancode, state)

    def inject_mouse_move(self, x, y):
        if _GLOBAL_WAYLAND_BACKEND:
            _GLOBAL_WAYLAND_BACKEND.inject_mouse_move(float(x), float(y))

    def inject_relative_mouse_move(self, dx, dy):
        if _GLOBAL_WAYLAND_BACKEND:
            _GLOBAL_WAYLAND_BACKEND.inject_relative_mouse_move(float(dx), float(dy))

    def inject_mouse_button(self, btn, state):
        if _GLOBAL_WAYLAND_BACKEND:
            _GLOBAL_WAYLAND_BACKEND.inject_mouse_button(btn, state)

    def inject_mouse_scroll(self, x, y):
        if _GLOBAL_WAYLAND_BACKEND:
            _GLOBAL_WAYLAND_BACKEND.inject_mouse_scroll(float(x), float(y))

    def set_cursor_rendering(self, enabled):
        if _GLOBAL_WAYLAND_BACKEND:
            _GLOBAL_WAYLAND_BACKEND.set_cursor_rendering(bool(enabled))

    def set_cursor_callback(self, callback):
        if _GLOBAL_WAYLAND_BACKEND:
            _GLOBAL_WAYLAND_BACKEND.set_cursor_callback(callback)

    def request_idr_frame(self):
        if not self._is_capturing:
            return
        # Route Wayland-first, matching start_capture's precedence (else a dual-active
        # config would misroute the IDR to the idle legacy module).
        if _GLOBAL_WAYLAND_BACKEND is not None:
            # getattr-guarded so an older .so without the method is a clean no-op.
            _wl_request_idr = getattr(_GLOBAL_WAYLAND_BACKEND, "request_idr_frame", None)
            if _wl_request_idr is not None:
                _wl_request_idr()
        elif self._module:
            request_idr(self._module)

    def update_video_bitrate(self, bitrate):
        if self._is_capturing and self._module:
            update_video_bitrate_c(self._module, bitrate)
    
    def update_framerate(self, fps):
        if self._is_capturing and self._module:
            update_framerate_c(self._module, ctypes.c_double(fps))
    
    def update_vbv_buf_size(self, buffer_size):
        if self._is_capturing and self._module:
            update_vbv_buffer_size_c(self._module, buffer_size)
