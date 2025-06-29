from . import screen_capture_module as scm

class ScreenCapture:
    """Python wrapper for screen capture module."""

    def __init__(self):
        self._module = scm.create_screen_capture_module()
        if not self._module:
            raise Exception("Failed to create screen capture module.")
        self._is_capturing = False
        self._python_stripe_callback = None

    def __del__(self):
        if hasattr(self, '_module') and self._module:
            scm.destroy_screen_capture_module(self._module)
            self._module = None

    def start_capture(self, settings: scm.CaptureSettings, stripe_callback):
        if self._is_capturing:
            raise ValueError("Capture already started.")
        if not callable(stripe_callback):
            raise TypeError("stripe_callback must be callable.")

        self._python_stripe_callback = stripe_callback
        scm.start_screen_capture(self._module, settings, self._internal_callback, None)
        self._is_capturing = True

    def stop_capture(self):
        if not self._is_capturing:
            raise ValueError("Capture not started.")
        scm.stop_screen_capture(self._module)
        self._is_capturing = False
        self._python_stripe_callback = None

    def _internal_callback(self, result: scm.StripeEncodeResult, user_data):
        """Internal callback, calls user's Python callback and frees data."""
        try:
            if self._is_capturing and self._python_stripe_callback:
                self._python_stripe_callback(result, user_data)
        finally:
            # Ensure the C++ allocated data is always freed
            if result:
                scm.free_stripe_encode_result_data(result)
