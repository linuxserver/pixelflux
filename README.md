# pixelflux

[![PyPI version](https://badge.fury.io/py/pixelflux.svg)](https://badge.fury.io/py/pixelflux)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

**A performant web native pixel delivery pipeline for diverse sources, blending VNC-inspired parallel processing of pixel buffers with flexible modern encoding formats.**

This module provides a Python interface to a high-performance C++ capture library. It captures pixel data from a source (currently X11 screen regions), detects changes, and encodes modified stripes into JPEG or H.264, delivering them via a callback mechanism. This stripe-based, change-driven approach is designed for efficient streaming or processing of visual data.

## Installation

This module relies on a native C++ extension that is compiled during installation using your system's C++ compiler.

1.  **Prerequisites (for the current X11 backend on Debian/Ubuntu):**
    Ensure you have a C++ compiler (`g++`) and development files for Python, X11, Xext (XShm), libjpeg-turbo, and libx264.

```bash
sudo apt-get update && \
sudo apt-get install -y \
  g++ \
  libjpeg-turbo8-dev \
  libx11-dev \
  libxfixes-dev \
  libxext-dev \
  libx264-dev \
  python3-dev
```

2.  **Install the Package:**
    You can install directly from PyPI or from a local source clone.

    **Option A: Install from PyPI**
    ```bash
    pip install pixelflux
    ```

    **Option B: Install from a local source directory**
    ```bash
    # From the root of the project repository
    pip install .
    ```

    This command will use `setuptools` to directly compile the C++ extension (`screen_capture_module.cpp`) and install it alongside the Python code into your environment.

    **Note:** The current backend is designed and tested for **Linux/X11** environments.

## Usage

### Basic Capture

Here is a basic example demonstrating how to use the `pixelflux` module to start capturing and process encoded stripes.

```python
import time
from pixelflux import CaptureSettings, ScreenCapture, StripeCallback

# Define your Python callback function.
# This function will be called from a background thread for each encoded stripe.
def my_python_callback(result, user_data):
    """
    Callback function to process encoded stripes.
    `result` is a StripeEncodeResult object with the stripe's data.
    `user_data` is whatever object you passed to start_capture (or None).
    """
    if result.data:
        # result.type will be 1 for JPEG, 2 for H.264
        type_str = "H264" if result.type == 2 else "JPEG"

        print(
            f"Received {type_str} stripe: "
            f"frame_id={result.frame_id}, "
            f"y_start={result.stripe_y_start}, "
            f"height={result.stripe_height}, "
            f"size={len(result.data)} bytes"
        )
    
    # Memory is managed automatically. No need to free anything.

# 1. Configure capture settings
settings = CaptureSettings()
settings.capture_width = 1280
settings.capture_height = 720
settings.capture_x = 0
settings.capture_y = 0
settings.target_fps = 30.0

# Set output mode to H.264 (1)
settings.output_mode = 1
settings.h264_crf = 25 # H264 Constant Rate Factor (0-51, lower is better quality)

# 2. Instantiate the ScreenCapture module
module = ScreenCapture()

# 3. Create a StripeCallback handler object
# This object simply holds your Python function.
callback_handler = StripeCallback(my_python_callback)

try:
    # 4. Start the capture, passing the settings and callback handler.
    # The third argument is optional user_data to be passed to your callback.
    module.start_capture(settings, callback_handler, None)
    
    print("Capture started. Press Enter to stop...")
    input() # Keep the main thread alive while capture runs in the background.

finally:
    # 5. Stop the capture. This will block until the background thread has exited.
    module.stop_capture()
    print("Capture stopped.")
```

### Capture Settings

The `CaptureSettings` class allows for detailed configuration of the capture process.

```python
# All attributes of the CaptureSettings object are standard Python properties.
settings = CaptureSettings()

# Core Capture
settings.capture_width = 1920
settings.capture_height = 1080
settings.capture_x = 0
settings.capture_y = 0
settings.target_fps = 60.0
settings.capture_cursor = True

# Encoding Mode (0 for JPEG, 1 for H.264)
settings.output_mode = 1

# JPEG Quality Settings
settings.jpeg_quality = 75              # Quality for changed stripes (0-100)
settings.paint_over_jpeg_quality = 90   # Quality for static "paint-over" stripes (0-100)

# H.264 Quality Settings
settings.h264_crf = 23                  # CRF value (0-51, lower is better quality/higher bitrate)
settings.h264_fullcolor = False         # Use I444 (full color) instead of I420
settings.h264_fullframe = False         # Encode full frames instead of just changed stripes

# Change Detection & Optimization
settings.use_paint_over_quality = True  # Enable paint-over/IDR requests for static regions
settings.paint_over_trigger_frames = 15 # Frames of no motion to trigger paint-over
settings.damage_block_threshold = 10    # Consecutive changes to trigger "damaged" state
settings.damage_block_duration = 30     # Frames a stripe stays "damaged"

# Watermarking
settings.watermark_path = b"/path/to/your/watermark.png" # Must be bytes
settings.watermark_location_enum = 4 # 0:None, 1:TL, 2:TR, 3:BL, 4:BR, 5:Middle, 6:Animated
```

### Stripe Callback and Data Structure

The `start_capture` function requires a `StripeCallback` object, which wraps your Python function. This function is invoked from a C++ background thread whenever an encoded stripe is ready.

Your callback function will receive two arguments:
1.  `result`: A `StripeEncodeResult` object containing the stripe data.
2.  `user_data`: The optional object you passed to `start_capture`.

The `StripeEncodeResult` object has the following read-only properties:

```python
class StripeEncodeResult:
    # This is illustrative. You do not define this class.
    # You receive an instance of it in your callback.

    @property
    def type(self) -> int: ... # StripeDataType: 1 for JPEG, 2 for H.264

    @property
    def stripe_y_start(self) -> int: ...

    @property
    def stripe_height(self) -> int: ...

    @property
    def size(self) -> int: ... # The size of the data in bytes

    @property
    def data(self) -> bytes: ... # The encoded stripe data as a Python bytes object

    @property
    def frame_id(self) -> int: ... # Frame counter for this stripe
```

**Memory Management:** The memory for the stripe data is managed automatically. When the `StripeEncodeResult` object received by your callback is garbage-collected by Python, its internal C++ destructor is called, which frees the underlying data buffer. **You do not need to do any manual memory management.**

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

A comprehensive example, `screen_to_browser.py`, is located in the `example` directory of this repository. This script demonstrates robust, real-time screen capture, H.264 encoding, and streaming via WebSockets. It sets up:

*   An `asyncio`-based WebSocket server to stream encoded H.264 stripes.
*   An HTTP server to serve a client-side HTML page for viewing the stream.
*   The `pixelflux` module to perform the screen capture and encoding, with proper asynchronous shutdown logic to prevent deadlocks.

**To run this example:**

**Note:** This example assumes you are on a Linux host with a running X11 session and will only work from localhost unless HTTPS is added.

1.  First, ensure you have the `websockets` library installed (it is a dependency for the example, not the library itself):
    ```bash
    pip install websockets
    ```

2.  Navigate to the `example` directory within the repository:
    ```bash
    cd example
    ```
3.  Execute the Python script:
    ```bash
    python3 screen_to_browser.py
    ```
4.  Open your web browser and go to the URL indicated by the script's output (usually `http://localhost:9001/index.html`) to view the live stream.

## License

This project is licensed under the **Mozilla Public License Version 2.0**.
A copy of the MPL 2.0 can be found at https://mozilla.org/MPL/2.0/.
