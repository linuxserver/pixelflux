# pixelflux

[![PyPI version](https://badge.fury.io/py/pixelflux.svg)](https://badge.fury.io/py/pixelflux)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

**A performant web native pixel delivery pipeline for diverse sources, blending VNC-inspired parallel processing of pixel buffers with flexible modern encoding formats.**

This module provides a Python interface to a high-performance C++ capture library. It captures pixel data from a source (currently X11 screen regions), detects changes, and encodes modified stripes into JPEG or H.264. It supports CPU-based encoding (libx264, libjpeg-turbo) as well as hardware-accelerated H.264 encoding via NVIDIA's NVENC and VA-API for Intel/AMD GPUs. The resulting data is delivered efficiently to your Python application via a callback mechanism.

## Installation

This module relies on a native C++ extension that is compiled during installation.

1.  **Prerequisites (for Debian/Ubuntu):**
    Ensure you have a C++ compiler (`g++`) and development files for Python, X11, Xfixes, XShm, libjpeg-turbo, and libx264. These are required for the base software encoding functionality.

    ```bash
    sudo apt-get update && \
    sudo apt-get install -y \
      g++ \
      libdrm-dev \
      libjpeg-turbo8-dev \
      libva-dev \
      libx11-dev \
      libx264-dev \
      libxext-dev \
      libxfixes-dev \
      libyuv-dev \
      python3-dev
    ```

2.  **Hardware Acceleration (Optional but Recommended):**
    *   **NVIDIA (NVENC):** No extra packages are needed at compile time. If you have the NVIDIA driver installed, the library will be detected and used automatically at runtime.
    *   **Intel/AMD (VA-API):** The `libva-dev` and `libdrm-dev` packages listed above are sufficient for compilation. Ensure you have the correct VA-API drivers for your hardware installed (e.g., `intel-media-va-driver-non-free` for Intel, `mesa-va-drivers` for AMD).

3.  **Install the Package:**
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

    **Note:** The current backend is designed and tested for **Linux/X11** environments.

## Usage

### Capture Settings

The `CaptureSettings` class allows for detailed configuration of the capture process.

```python
# All attributes of the CaptureSettings object are standard ctypes properties.
settings = CaptureSettings()

# --- Core Capture ---
settings.capture_width = 1920
settings.capture_height = 1080
settings.capture_x = 0
settings.capture_y = 0
settings.target_fps = 60.0
settings.capture_cursor = True

# --- Encoding Mode ---
# 0 for JPEG, 1 for H.264
settings.output_mode = 1

# --- JPEG Settings ---
settings.jpeg_quality = 75              # Quality for changed stripes (0-100)
settings.paint_over_jpeg_quality = 90   # Quality for static "paint-over" stripes (0-100)

# --- H.264 Settings ---
settings.h264_crf = 23                  # CRF value (0-51, lower is better quality/higher bitrate)
settings.h264_fullcolor = False         # Use I444 (full color) instead of I420 for software encoding
settings.h264_fullframe = True          # Encode full frames (required for HW accel) instead of just changed stripes

# --- Hardware Acceleration ---
# Set to >= 0 to enable VA-API on a specific /dev/dri/renderD* node.
# Set to -1 to disable VA-API and let the system try NVENC if available.
settings.vaapi_render_node_index = -1

# --- Change Detection & Optimization ---
settings.use_paint_over_quality = True  # Enable paint-over/IDR requests for static regions
settings.paint_over_trigger_frames = 15 # Frames of no motion to trigger paint-over
settings.damage_block_threshold = 10    # Consecutive changes to trigger "damaged" state
settings.damage_block_duration = 30     # Frames a stripe stays "damaged"

# --- Watermarking ---
# Must be a bytes object. The path to your PNG image.
settings.watermark_path = b"/path/to/your/watermark.png" 
# 0:None, 1:TopLeft, 2:TopRight, 3:BottomLeft, 4:BottomRight, 5:Middle, 6:Animated
settings.watermark_location_enum = 4 
```

### Stripe Callback and Data Structure

Your callback function receives a `ctypes.POINTER(StripeEncodeResult)`. You must access its fields via the `.contents` attribute.

The `StripeEncodeResult` struct has the following fields:

```python
# This is an illustrative Python representation of the C++ struct.
class StripeEncodeResult(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),             # 1 for JPEG, 2 for H.264
        ("stripe_y_start", ctypes.c_int),
        ("stripe_height", ctypes.c_int),
        ("size", ctypes.c_int),             # The size of the data in bytes
        ("data", ctypes.POINTER(ctypes.c_ubyte)), # Pointer to the encoded data
        ("frame_id", ctypes.c_int),         # Frame counter for this stripe
    ]
```

**Memory Management:** The data pointed to by `result.contents.data` is valid **only within the scope of your callback function**. The C++ library automatically frees this memory after your callback returns. To use the data, you must copy it, for example by using `ctypes.string_at(result.contents.data, result.contents.size)`.

## Features

*   **Efficient Pixel Capture:** Leverages a native C++ module using XShm for optimized X11 screen capture performance.
*   **Flexible Encoding Backends:**
    *   **Software:** libx264 (H.264) and libjpeg-turbo (JPEG).
    *   **Hardware:** NVIDIA NVENC and VA-API (Intel, AMD).
*   **Stripe-Based Processing:** For software encoding, can divide the screen into horizontal stripes and process them in parallel across CPU cores.
*   **Change Detection:** Encodes only stripes that have changed (based on an XXH3 hash comparison) since the last frame, significantly reducing processing load and bandwidth for software encoding modes.
*   **Dynamic Watermarking:** Overlay a PNG image on the captured video. The watermark can be pinned to a corner, centered, or animated to bounce around the screen.
*   **Dynamic Quality Optimizations:**
    *   **Paint-Over for Static Regions:** After a region remains static for `paint_over_trigger_frames`, it is resent at high quality (JPEG) or as a new IDR frame (H.264) to correct any compression artifacts.
    *   **Damage Throttling:** For highly active regions, the system can temporarily reduce the frequency of change detection to save CPU cycles.
*   **Direct Callback Mechanism:** Provides encoded stripe data directly to your Python code for real-time processing or streaming.

## Example: Real-time H.264 Streaming with WebSockets

A comprehensive example, `screen_to_browser.py`, is located in the `example` directory of this repository. This script demonstrates robust, real-time screen capture, H.264 encoding, and streaming via WebSockets. It sets up:

*   An `asyncio`-based WebSocket server to stream encoded H.264 frames.
*   An HTTP server to serve a client-side HTML page for viewing the stream.
*   The `pixelflux` module to perform the screen capture and encoding.

**To run this example:**

**Note:** This example assumes you are on a Linux host with a running X11 session.

1.  First, ensure you have the `websockets` library installed:
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
4.  Open your web browser and go to the URL indicated by the script's output (usually `http://localhost:9001`) to view the live stream.

## License

This project is licensed under the **Mozilla Public License Version 2.0**.
A copy of the MPL 2.0 can be found at https://mozilla.org/MPL/2.0/.
