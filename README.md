# pixelflux

[![PyPI version](https://badge.fury.io/py/pixelflux.svg)](https://badge.fury.io/py/pixelflux)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

**A performant web native pixel delivery pipeline for diverse sources, blending parallel processing of pixel buffers with flexible modern encoding formats.**

This module provides a Python interface to a high-performance capture library supporting both **X11** and **Wayland** environments. It captures pixel data, detects changes, and encodes modified stripes into JPEG or H.264.

It supports CPU-based encoding (libx264, libjpeg-turbo) as well as hardware-accelerated H.264 encoding via NVIDIA's NVENC and VA-API for Intel/AMD GPUs. The Wayland backend features a **zero-copy pipeline**, passing GPU buffers directly to the encoder to minimize latency and CPU usage.

## Installation

This module relies on native C++ (X11) and Rust (Wayland) extensions that are compiled during installation.

### 1. Prerequisites

Ensure you have a C++ compiler (`g++`), the Rust toolchain (`cargo`), and development files for Python and the underlying graphics libraries.

**Base Dependencies (Debian/Ubuntu):**
```bash
sudo apt-get update && \
sudo apt-get install -y \
  g++ \
  git \
  curl \
  python3-dev \
  libavcodec-dev \
  libavutil-dev \
  libjpeg-turbo8-dev \
  libx264-dev \
  libyuv-dev
```

**X11 Backend Dependencies:**
```bash
sudo apt-get install -y \
  libx11-dev \
  libxext-dev \
  libxfixes-dev
```

**Wayland Backend Dependencies:**
To build the Rust-based Wayland backend, you need the Rust toolchain and Wayland/DRM libraries:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Libraries
sudo apt-get install -y \
  libgbm-dev \
  libdrm-dev \
  libwayland-dev \
  libinput-dev \
  libxkbcommon-dev \
  libva-dev \
  libclang-dev
```

### 2. Hardware Acceleration (Optional but Recommended)
*   **NVIDIA (NVENC):** The library detects the NVIDIA driver at runtime. No extra compile-time packages are needed.
*   **Intel/AMD (VA-API):** Ensure `libva-dev` and `libdrm-dev` are installed. You must also have the correct drivers (e.g., `intel-media-va-driver-non-free` or `mesa-va-drivers`).

### 3. Install the Package

**Option A: Install from PyPI**
```bash
pip install pixelflux
```

**Option B: Install from local source**
```bash
# From the root of the project repository
pip install .
```

## Usage

### Backend Selection

By default, `pixelflux` loads the legacy X11 backend. To enable the **Wayland** backend (built on [Smithay](https://github.com/Smithay/smithay)), set the following environment variable before importing the module:

```bash
export PIXELFLUX_WAYLAND=true
```

To test launching programs into this backend simply add `WAYLAND_DISPLAY=wayland-1` before launching them: 

```bash
WAYLAND_DISPLAY=wayland-1 glmark2-es2-wayland -s 1920x1080
```

### Capture Settings

The `CaptureSettings` class configures both backends.

```python
from pixelflux import CaptureSettings, ScreenCapture

settings = CaptureSettings()

# --- Core Capture ---
settings.capture_width = 1920
settings.capture_height = 1080
settings.capture_x = 0
settings.capture_y = 0
settings.capture_cursor = True
settings.target_fps = 60.0
settings.scale = 1.0  # Fractional scaling (Wayland only)

# --- Encoding Mode ---
# 0 for JPEG, 1 for H.264
settings.output_mode = 1
# Force CPU encoding and ignore hardware encoders
capture_settings.use_cpu = False

# --- Debugging ---
settings.debug_logging = False # Enable/disable the continuous FPS and settings log to the console.

# --- JPEG Settings ---
settings.jpeg_quality = 75              # Quality for changed stripes (0-100)
settings.paint_over_jpeg_quality = 90   # Quality for static "paint-over" stripes (0-100)

# --- H.264 Settings ---
settings.h264_crf = 25                            # CRF value (0-51, lower is better quality/higher bitrate)
settings.h264_paintover_crf = 18                  # CRF for H.264 paintover on static content. Must be lower than h264_crf to activate.
settings.h264_paintover_burst_frames = 5          # Number of high-quality frames to send in a burst when a paintover is triggered.
settings.h264_fullcolor = False                   # Use I444 (full color) instead of I420 for software encoding
settings.h264_fullframe = True                    # Encode full frames (required for HW accel) instead of just changed stripes
settings.h264_streaming_mode = False              # Bypass all VNC logic and work like a normal video encoder, higher constant CPU usage for fullscreen gaming/videos
settings.h264_cbr_mode = False                    # Switches to CBR mode and ignores CRF value. Used in conjunction with h264_bitrate_kbps.
settings.h264_bitrate_kbps = 4000                 # Target bitrate for CBR mode. Required when h264_cbr_mode is enabled.
settings.h264_vbv_buffer_size_kb = 400            # Optional VBV buffer size in kilobits for custom buffer size.
settings.auto_adjust_screen_capture_size = True   # Allow pixelflux to adjust its capture width and height.

# --- Hardware Acceleration ---
# >= 0: Enable GPU Encoding on /dev/dri/renderD(128 + index)
# -1: Disable GPU Encoding (System will try NVENC if available when using the x11 backend, Wayland needs this set to a render node)
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

### Input Injection (Wayland Only)

In Wayland mode, `pixelflux` acts as the compositor. You cannot use external tools like `xdotool`. Instead, use the input injection methods provided by the `ScreenCapture` instance:

```python
capture = ScreenCapture()
capture.start_capture(settings, my_callback)

# Inject Mouse Motion (Absolute coordinates)
capture.inject_mouse_move(x=500.0, y=300.0)

# Inject Mouse Button (1=Left, 2=Middle, 3=Right, etc.)
# State: 1 = Pressed, 0 = Released
capture.inject_mouse_button(btn=1, state=1) 

# Inject Scroll (Vertical/Horizontal)
capture.inject_mouse_scroll(x=0.0, y=10.0)

# Inject Keyboard Key
# scancode: Linux raw keycode (e.g., 17 for 'w')
# state: 1 = Pressed, 0 = Released
capture.inject_key(scancode=17, state=1)
```

### Stripe Callback

Your callback receives a `ctypes.POINTER(StripeEncodeResult)`.

```python
def my_callback(result_ptr, user_data):
    result = result_ptr.contents
    
    # Access data
    # result.type (0=H264, 1=JPEG)
    # result.frame_id
    # result.stripe_y_start
    
    # Copy data to Python bytes
    encoded_data = ctypes.string_at(result.data, result.size)
    
    # Send encoded_data to client...
```

## Zero-Copy Pipeline (Wayland)

The Wayland backend implements a **Zero-Copy** architecture for hardware encoding.

1.  **Rendering:** The compositor renders the desktop to a GPU buffer (GBM).
2.  **Export:** This buffer is exported as a `Dmabuf` (file descriptor).
3.  **Encoding:** The `Dmabuf` is imported directly into the encoder context (NVENC or VA-API) without ever copying pixel data to system RAM (CPU).

**Performance Note:** Enabling **watermarking** or utilizing a render node different from the encoding node will force a "Readback" fallback, copying pixels to the CPU and breaking the zero-copy chain. This increases latency and CPU load.

## Features

*   **Hybrid Backend:**
    *   **X11 (C++):** Legacy support using XShm.
    *   **Wayland (Rust):** Modern, secure, headless compositor based on [Smithay](https://github.com/Smithay/smithay).
*   **Flexible Encoding:**
    *   **Software:** libx264 (H.264) and libjpeg-turbo (JPEG) with multi-threaded striping.
    *   **Hardware:** NVIDIA NVENC and VA-API (Intel/AMD) with Zero-Copy support.
*   **Smart Bandwidth Management:**
    *   **Change Detection:** Encodes only changed stripes (Software/JPEG mode).
    *   **Paint-Over:** Automatically improves quality for static regions.
    *   **Damage Throttling:** Limits processing during high-motion scenes.
*   **Input Handling:** Built-in input injection for mouse and keyboard (Wayland).
*   **Cursor Compositing:** Hardware cursor planes or software rendering options.
*   **Dynamic Watermarking:** Overlay PNGs with static positioning or DVD-screensaver style animation.

## License

This project is licensed under the **Mozilla Public License Version 2.0**.
A copy of the MPL 2.0 can be found at https://mozilla.org/MPL/2.0/.
