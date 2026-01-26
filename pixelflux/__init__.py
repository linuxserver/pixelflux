import ctypes
import os
import threading
import sys

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

lib_dir = os.path.dirname(__file__)
lib_path = os.path.join(lib_dir, 'screen_capture_module.so')

_legacy_lib = None
try:
    if os.path.exists(lib_path):
        _legacy_lib = ctypes.CDLL(lib_path)
    else:
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
 
_GLOBAL_WAYLAND_BACKEND = None
if os.environ.get("PIXELFLUX_WAYLAND") == "true":
    try:
        from . import pixelflux_wayland
        _GLOBAL_WAYLAND_BACKEND = pixelflux_wayland.WaylandBackend()
        print(">> [PixelFlux] Rust Wayland Backend Initialized Globally.")
    except ImportError as e:
        print(f">> [PixelFlux] Failed to load Wayland backend: {e}")
        pass

class ScreenCapture:
    """Python wrapper for screen capture module using ctypes."""

    def __init__(self):
        if _legacy_lib:
            self._module = create_module()
        else:
            self._module = None
        
        self._is_capturing = False
        self._python_stripe_callback = None
        self._c_callback = None

    def __del__(self):
        if hasattr(self, '_module') and self._module:
            try:
                self.stop_capture()
                destroy_module(self._module)
            except:
                pass
            self._module = None

    def start_capture(self, settings: CaptureSettings, stripe_callback):
        if self._is_capturing:
            raise ValueError("Capture already started.")

        self._python_stripe_callback = stripe_callback
        mode = getattr(settings, 'mode', 'x11')

    def start_capture(self, settings: CaptureSettings, stripe_callback):
        if self._is_capturing:
            raise ValueError("Capture already started.")

        self._python_stripe_callback = stripe_callback
        
        if _GLOBAL_WAYLAND_BACKEND:
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
                c_buffer = (ctypes.c_ubyte * size).from_buffer_copy(data_bytes)
                result_struct = StripeEncodeResult()
                result_struct.size = size
                result_struct.data = ctypes.cast(c_buffer, ctypes.POINTER(ctypes.c_ubyte))
                if is_h264:
                    result_struct.type = 0
                    if size >= 4:
                        result_struct.frame_id = int.from_bytes(data_bytes[2:4], 'big')
                    else:
                        result_struct.frame_id = 0
                    if size >= 6:
                         result_struct.stripe_y_start = int.from_bytes(data_bytes[4:6], 'big')
                    else:
                         result_struct.stripe_y_start = 0
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
            _GLOBAL_WAYLAND_BACKEND.start_capture(rust_bridge_callback, settings)
            self._is_capturing = True
            return 

        if not self._module:
             raise OSError("Legacy screen_capture_module.so not found.")

        if not callable(stripe_callback):
            raise TypeError("stripe_callback must be callable.")
        
        self._c_callback = StripeCallback(self._internal_c_callback)
        start_capture_c(self._module, settings, self._c_callback, None)
        self._is_capturing = True

    def stop_capture(self):
        if not self._is_capturing:
            return
        
        if self._module and self._c_callback:
            stop_capture_c(self._module)
            self._c_callback = None
        
        if _GLOBAL_WAYLAND_BACKEND:
             _GLOBAL_WAYLAND_BACKEND.stop_capture()
            
        self._is_capturing = False
        self._python_stripe_callback = None

    def _internal_c_callback(self, result_ptr, user_data):
        if self._is_capturing and self._python_stripe_callback:
            try:
                self._python_stripe_callback(result_ptr, user_data)
            finally:
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
        if self._is_capturing and self._module:
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
