# pixelflux

[![PyPI version](https://badge.fury.io/py/pixelflux.svg)](https://badge.fury.io/py/pixelflux)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

**A performant web native pixel delivery pipeline for diverse sources, blending VNC-inspired parallel processing of pixel buffers with flexible modern encoding formats.**

This module provides a Python interface to a high-performance C++ capture library. It captures pixel data from a source (currently X11 screen regions), detects changes, and encodes modified stripes into JPEG or H.264, delivering them via a callback mechanism. This stripe-based, change-driven approach is designed for efficient streaming or processing of visual data.

## Installation

This module relies on a native C++ component.

1.  **Prerequisites (for the current X11 backend on Debian/Ubuntu):**
    Ensure you have CMake, a C++ compiler, and development files for X11, XShm (Xext), libjpeg-turbo, and libx264.

```bash
sudo apt-get update && \
sudo apt-get install -y \
  cmake \
  g++ \
  gcc \
  libjpeg-turbo8-dev \
  libx11-dev \
  libxext-dev \
  libx264-dev \
  make \
  python3-dev
```
**Note:** `libjpeg-turbo8-dev` might be `libjpeg62-turbo-dev` or similar on older systems.

2.  **Install via pip:**

```bash
pip install pixelflux
```

This command will:
*   Build the native `screen_capture_module.so` (or similar) library using CMake during the installation process.
*   Install the Python wrapper and the compiled shared library.

**Note:** The current backend is designed and tested for **Linux/X11** environments. Future development aims to support a wider range of framebuffers and platforms.

3. **Developer Install From Source**

```bash
sudo python3 setup.py install
```

## Usage

### Basic Capture

Here's a basic example demonstrating how to use the `pixelflux` module to start capturing and process encoded stripes.

```python
import ctypes
import time
# These components are provided by the pixelflux Python wrapper:
from pixelflux import CaptureSettings, ScreenCapture, StripeCallback, StripeEncodeResult

# Integer constants for OutputMode (as defined in C++ enum class OutputMode)
OUTPUT_MODE_JPEG = 0
OUTPUT_MODE_H264 = 1

# Integer constants for StripeDataType (as defined in C++ enum class StripeDataType)
STRIPE_TYPE_UNKNOWN = 0
STRIPE_TYPE_JPEG = 1
STRIPE_TYPE_H264 = 2

# Define your Python callback function.
# The StripeCallback type (a ctypes.CFUNCTYPE) expects this signature.
def my_python_callback(result_ptr: ctypes.POINTER(StripeEncodeResult), user_data_ptr: ctypes.c_void_p):
    """Callback function to process encoded stripes."""
    result = result_ptr.contents
    
    # Note: Based on the current Python wrapper's start_capture method (inferred from previous TypeError),
    # user_data_ptr might not be actively passed from Python. If user_data is needed,
    # the Python wrapper for start_capture would need to support passing it.
    # For this example, user_data_ptr is effectively ignored.

    if result.data and result.size > 0:
        type_str = "Unknown"
        if result.type == STRIPE_TYPE_JPEG:
            type_str = "JPEG"
        elif result.type == STRIPE_TYPE_H264:
            type_str = "H264"

        print(f"Received {type_str} stripe: frame_id={result.frame_id}, y_start={result.stripe_y_start}, height={result.stripe_height}, size={result.size} bytes")
    
    # Memory for result.data is managed by the C++ layer after this callback returns.
    # Do NOT free result.data here.

# Configure capture settings
capture_settings = CaptureSettings()
capture_settings.capture_width = 1280
capture_settings.capture_height = 720
capture_settings.capture_x = 0
capture_settings.capture_y = 0
capture_settings.target_fps = 30.0

# Set output mode to H.264
capture_settings.output_mode = OUTPUT_MODE_H264
capture_settings.h264_crf = 25 # H264 Constant Rate Factor (0-51, lower is better quality)

# Instantiate the ScreenCapture module
module = ScreenCapture()

# Create a C-callable function pointer from your Python callback.
# StripeCallback is the CFUNCTYPE provided by the pixelflux module.
c_callback_func_ptr = StripeCallback(my_python_callback)

try:
    # Call start_capture with settings and the callback pointer.
    module.start_capture(capture_settings, c_callback_func_ptr) 
    
    mode_str = "JPEG" if capture_settings.output_mode == OUTPUT_MODE_JPEG else "H264"
    print(f"Capture started (Mode: {mode_str}). Press Enter to stop...")
    input() # Keep capture running
finally:
    module.stop_capture()
    print("Capture stopped.")
```

### Capture Settings

The `CaptureSettings` class, exposed as a `ctypes.Structure` by the Python wrapper, allows configuration of various parameters:

```python
# Python representation of the C++ CaptureSettings struct,
# provided by the pixelflux Python wrapper.
class CaptureSettings(ctypes.Structure):
    _fields_ = [
        ("capture_width", ctypes.c_int),
        ("capture_height", ctypes.c_int),
        ("capture_x", ctypes.c_int),
        ("capture_y", ctypes.c_int),
        ("target_fps", ctypes.c_double),
        ("jpeg_quality", ctypes.c_int),             # (JPEG mode) Quality for changed stripes (0-100)
        ("paint_over_jpeg_quality", ctypes.c_int),  # (JPEG mode) Quality for static "paint-over" stripes (0-100)
        ("use_paint_over_quality", ctypes.c_bool),  # (JPEG/H264 mode) Enable paint-over with different quality
        ("paint_over_trigger_frames", ctypes.c_int),# Frames of no motion to trigger paint-over/IDR request
        ("damage_block_threshold", ctypes.c_int),   # Consecutive changes to trigger "damaged" state for a stripe
        ("damage_block_duration", ctypes.c_int),    # Frames a stripe stays "damaged" (affects paint-over logic)
        ("output_mode", ctypes.c_int),              # 0 for JPEG, 1 for H264
        ("h264_crf", ctypes.c_int),                 # (H264 mode) CRF value (0-51, lower is better quality/higher bitrate)
    ]
```

Adjust these settings to fine-tune capture performance and quality.

### Stripe Callback and Data Structure

The `start_capture` function requires a callback function, invoked by the native C++ module when an encoded stripe is ready. The Python wrapper provides the necessary `StripeCallback` type (a `ctypes.CFUNCTYPE`) and the `StripeEncodeResult` structure (a `ctypes.Structure`).

```python
# The pixelflux Python wrapper provides these definitions:

# StripeCallback is a ctypes.CFUNCTYPE defining the callback signature:
# StripeCallback = ctypes.CFUNCTYPE(
#    None, ctypes.POINTER(StripeEncodeResult), ctypes.c_void_p # Result pointer, User data pointer
# )

# StripeEncodeResult is a ctypes.Structure for the callback data:
class StripeEncodeResult(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),             # StripeDataType: 0 UNKNOWN, 1 JPEG, 2 H264
        ("stripe_y_start", ctypes.c_int),
        ("stripe_height", ctypes.c_int),
        ("size", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_ubyte)), # Pointer to the encoded data (includes custom header)
        ("frame_id", ctypes.c_int)              # Frame counter for this stripe
    ]
```

Your Python callback function must match the `StripeCallback` signature:
`def my_callback(result_ptr: ctypes.POINTER(StripeEncodeResult), user_data_ptr: ctypes.c_void_p):`

*   `result_ptr`: A ctypes pointer to a `StripeEncodeResult` structure. Access its fields via `result_ptr.contents`.
*   `user_data_ptr`: A `ctypes.c_void_p` representing the user data you passed to `start_capture`. If you passed `None`, this will be `None` or `0`. You are responsible for casting and interpreting this pointer if you use it.

Inside the callback, process `result_ptr.contents`. **The Python wrapper handles freeing the memory of `result.data` by calling the native `free_stripe_encode_result_data` function after your callback returns. Do not attempt to free it manually.**

## Features

*   **Efficient Pixel Capture:** Leverages a native C++ module using XShm for optimized X11 screen capture performance.
*   **Stripe-Based Encoding (JPEG & H.264):** Encodes captured frames into horizontal stripes, processed in parallel using a number of threads based on system core count. Each stripe is an independent data unit.
*   **Change Detection:** Encodes only stripes that have changed (based on XXH3 hash comparison) since the last frame, significantly reducing processing load and bandwidth. This approach is inspired by VNC.
*   **Configurable Capture Region:** Specify the exact X, Y, width, and height of the screen region to capture.
*   **Adjustable FPS, Quality, and Encoding Parameters:** Control frame rate, JPEG quality (0-100), and H.264 CRF (0-51).
*   **Dynamic Quality Optimizations:**
    *   **Paint-Over for Static Regions:** After a stripe remains static for `paint_over_trigger_frames`, it is resent. For JPEG, this uses `paint_over_jpeg_quality` if `use_paint_over_quality` is true. For H.264, this triggers a request for an IDR frame for that stripe, ensuring a full refresh.
    *   **Adaptive Behavior for Highly Active Stripes (Damage Throttling):**
        *   Identifies stripes that change very frequently (exceeding `damage_block_threshold` updates).
        *   For these "damaged" stripes, damage checks are done less frequently, saving resources on high motion.
        *   For JPEG output, the quality of these frequently changing stripes dynamically adjusts (reducing slightly on change) and resets to higher base/paint-over quality after a cooldown period of `damage_block_duration` frames. This manages resources effectively for volatile content.
*   **Direct Callback Mechanism:** Provides encoded stripe data, including a custom header, directly to your Python code for real-time processing or streaming.

## Example: Real-time H.264 Streaming with WebSockets

A comprehensive example, `screen_to_browser.py`, is located in the `examples` directory of this repository. This script demonstrates real-time screen capture, H.264 encoding, and streaming via WebSockets. It sets up:

*   A WebSocket server to stream encoded H.264 stripes.
*   An HTTP server to serve a client-side HTML page for viewing the stream.
*   The `pixelflux` module to perform the screen capture and encoding.

**To run this example:**

**Note:** This example assumes you are on a Linux host with a running X11 session and will only work from localhost unless https is added.

1.  Navigate to the `examples` directory within the repository:
    ```bash
    cd examples
    ```
2.  Execute the Python script:
    ```bash
    python3 screen_to_browser.py
    ```
3.  Open your web browser and go to the URL indicated by the script's output `http://localhost:9001` to view the live stream.

## License

This project is licensed under the **Mozilla Public License Version 2.0**.
A copy of the MPL 2.0 can be found at https://mozilla.org/MPL/2.0/.
