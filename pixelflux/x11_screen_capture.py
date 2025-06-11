import ctypes
import os

class CaptureSettings(ctypes.Structure):
    _fields_ = [
        ("capture_width", ctypes.c_int),
        ("capture_height", ctypes.c_int),
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
        ("h264_fullcolor", ctypes.c_bool),
        ("h264_fullframe", ctypes.c_bool),
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
try:
    lib = ctypes.CDLL(lib_path)
except OSError as e:
    raise OSError(
        f"Could not load shared library: {e}. Ensure "
        f"'screen_capture_module.so' is in the "
        f"'screen_capture' directory."
    ) from e

create_module = lib.create_screen_capture_module
create_module.restype = ctypes.c_void_p

destroy_module = lib.destroy_screen_capture_module
destroy_module.argtypes = [ctypes.c_void_p]

start_capture = lib.start_screen_capture
start_capture.argtypes = [
    ctypes.c_void_p,
    CaptureSettings,
    StripeCallback,
    ctypes.c_void_p
]

stop_capture = lib.stop_screen_capture
stop_capture.argtypes = [ctypes.c_void_p]

free_stripe_encode_result_data = lib.free_stripe_encode_result_data
free_stripe_encode_result_data.argtypes = [ctypes.POINTER(StripeEncodeResult)]


class ScreenCapture:
    """Python wrapper for screen capture module - DIRECT CALLBACK."""

    def __init__(self):
        self._module = create_module()
        if not self._module:
            raise Exception("Failed to create screen capture module.")
        self._is_capturing = False
        self._python_stripe_callback = None
        self._c_callback = None

    def __del__(self):
        if hasattr(self, '_module') and self._module:
            destroy_module(self._module)
            self._module = None

    def start_capture(self, settings: CaptureSettings, stripe_callback):
        if self._is_capturing:
            raise ValueError("Capture already started.")
        if not callable(stripe_callback):
            raise TypeError("stripe_callback must be callable.")

        self._python_stripe_callback = stripe_callback
        self._c_callback = StripeCallback(self._internal_c_callback)
        start_capture(self._module, settings, self._c_callback, None)
        self._is_capturing = True

    def stop_capture(self):
        if not self._is_capturing:
            raise ValueError("Capture not started.")
        stop_capture(self._module)
        self._is_capturing = False
        self._python_stripe_callback = None
        self._c_callback = None

    def _internal_c_callback(self, result_ptr, user_data):
        """Internal C callback, calls user's Python callback."""
        if self._is_capturing and self._python_stripe_callback:
            # The user's callback will receive result_ptr.
            # They can access result_ptr.contents.type to get the integer value of the data type.
            self._python_stripe_callback(result_ptr, user_data)
            free_stripe_encode_result_data(result_ptr)
