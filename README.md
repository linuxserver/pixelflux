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

### Automatic GPU Selection

Set `SELKIES_AUTO_GPU=true` (preferred, or the legacy `AUTO_GPU=true`) to let pixelflux pick a
render node automatically instead of supplying one. It enumerates `/sys/class/drm`, pairs each
`cardN` with its `renderD*` node by PCI device, and skips non-GPU cards (IPMI/VGA). Selection is
**driver-aware**: NVIDIA nodes are routed to NVENC, while Intel (`i915`) and AMD (`amdgpu`) nodes
take the VA-API path. Both the X11 and Wayland backends honor this.

```bash
export SELKIES_AUTO_GPU=true
```

When auto-selection is off and no node is supplied, an operator-set `DRINODE` (e.g.
`/dev/dri/renderD128`) is honored before falling back to the software renderer.

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
settings.h264_fullcolor = False                   # Use I444/full color (High 4:4:4) instead of I420. Supported by software encoding and NVENC.
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
# Explicit render node path (X11). Takes precedence over the positional index above and
# avoids the index ambiguity. Must be a bytes object, e.g. b"/dev/dri/renderD128".
settings.vaapi_render_node_path = None

# --- Wire Format / Zero-Copy (X11) ---
# False (default): prepend the per-stripe header to each packet (the WebSocket path).
# True: emit the raw encoded payload with no header (for a WebRTC path that frames itself).
settings.omit_stripe_headers = False
# Deprecated/ignored: the native frame handed to your callback always owns its buffer
# (zero-copy on every Python version, see below), so this flag no longer has any effect.
# Kept only for backward compatibility.
settings.deferred_free = False

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

Your callback receives a single **frame object** (`StripeFrame` on X11, `WaylandFrame` on
Wayland). Both support the buffer protocol — `bytes(frame)` / `memoryview(frame)` / `len(frame)`
— and expose the stripe metadata as attributes:

```python
def my_callback(frame):
    # frame.data_type      (0=Unknown, 1=JPEG, 2=H.264)
    # frame.frame_id
    # frame.stripe_y_start
    # frame.stripe_height
    encoded_data = bytes(frame)          # copy out, or use memoryview(frame) zero-copy (below)
    # Send encoded_data to the client...
```

### Zero-Copy Frames

`memoryview(frame)` aliases the native encoder buffer with **no copy**, on **every supported
Python version (3.9–3.14)**. The frame object owns its buffer and keeps it alive until every
consumer — including a transport that retained a slice during a partial write — has released its
view, so the hand-off is memory-safe. (The old `deferred_free` / `OwnedFrame` / PEP 688 /
Python-3.12-only path is gone; the native buffer protocol does this on all versions.) Hand the
view straight to an async socket; keep the frame referenced for the duration of the send.

```python
def my_callback(frame):
    if frame.data_type == 0 or len(frame) == 0:   # nothing to send
        return
    # Hand BOTH the view and the frame to your sender (e.g. an asyncio.Queue) so the buffer
    # outlives the send: the view pins the frame, which frees the buffer once the view drops.
    queue.put_nowait({"data": memoryview(frame), "owner": frame})
```

See `example/screen_to_browser.py` for a complete queue-based usage.

## Zero-Copy Pipeline (Wayland)

The Wayland backend implements a **Zero-Copy** architecture for hardware encoding.

1.  **Rendering:** The compositor renders the desktop to a GPU buffer (GBM).
2.  **Export:** This buffer is exported as a `Dmabuf` (file descriptor).
3.  **Encoding:** The `Dmabuf` is imported directly into the encoder context (NVENC or VA-API) without ever copying pixel data to system RAM (CPU).

**Performance Note:** Enabling **watermarking** or utilizing a render node different from the encoding node will force a "Readback" fallback, copying pixels to the CPU and breaking the zero-copy chain. This increases latency and CPU load.

## Recording Sink (Wayland)

The Wayland backend can output the raw H.264 video stream directly to a Unix domain socket for external recording.

*Note: This feature requires full-frame H.264 encoding (CPU, VA-API, or NVENC) and does not work with JPEG or striped H.264 modes.*

```python
# Enable the unix socket (forces IDR frames every 30 frames and on connect)
settings.recording_socket = "/tmp/pixelflux_record"
```

You can then capture the stream using `ffmpeg`:

```bash
# Raw copy
ffmpeg -f h264 -i unix:///tmp/pixelflux_record -c:v copy test.h264

# Re-encode for a clean MP4
ffmpeg -f h264 -framerate 60 -i unix:///tmp/pixelflux_record -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p test.mp4
```

## NVIDIA NVENC (X11)

*   **Multi-GPU containers:** When several GPUs are exposed to a container, NVENC is filtered
    in-process to the GPU you selected (no separate `LD_PRELOAD` shim is required). Verified on
    NVIDIA drivers 570–595.
*   **4:4:4 (High 4:4:4):** Set `h264_fullcolor = True` to encode full-chroma H.264 via NVENC
    (`h264_fullcolor` codec), in addition to the software path.
*   **Force a keyframe on demand:** `capture.request_idr_frame()` forces an IDR frame, e.g. when
    a client reconnects or its decoder is reset. It routes to whichever encoder is active
    (NVENC, VA-API, or software) and is a no-op while no capture is running.

### Optional CUDA Color Conversion (NVRTC)

NVENC encoding can optionally use a CUDA (NVRTC) kernel for ARGB→NV12 colorspace conversion,
loading `libnvrtc` at runtime. If `libnvrtc` is **absent or incompatible, pixelflux silently
falls back to the libyuv CPU conversion** — this is not a failure. Two environment kill-switches
let you disable the GPU paths explicitly:

```bash
export PIXELFLUX_NO_CUDA_CONVERT=1     # disable the CUDA conversion kernel (use libyuv CPU path)
export PIXELFLUX_NVENC_DEVICE_INPUT=0  # disable feeding NVENC a device buffer directly
```

#### Matching NVRTC to your NVIDIA driver

NVRTC emits PTX which the driver then JIT-compiles. A **newer** NVRTC can emit a PTX ISA version
the **older** driver cannot JIT, so the installed NVRTC must be **≤ the driver's CUDA version**.
(The kernel is device-cc-aware and targets as low as Kepler `sm_35`, so it is broadly
forward-compatible across GPUs; the constraint is purely the PTX ISA version the driver can JIT.)
Read your driver's CUDA version from `nvidia-smi` (the **"CUDA Version"** field, top-right) and
pin NVRTC to it.

**Recommended (version-agnostic):** install NVIDIA's `cuda-toolkit` meta-package and let its
`[nvrtc]` extra pull the correct nvrtc wheel for the version you pin — no need to know the
per-CUDA package name. These resolve from the public PyPI (no extra index required):

```bash
pip install "cuda-toolkit[nvrtc]==13.3.*"   # CUDA 13
pip install "cuda-toolkit[nvrtc]==12.9.*"   # CUDA 12
pip install "cuda-toolkit[nvrtc]==11.8.*"   # CUDA 11 (Kepler and older / driver <= 470)
```

`pip install pixelflux[cuda]` does the same (latest/CUDA 13); pin as above for older drivers.

> **Note:** plain `pip install pixelflux` does **not** auto-install NVRTC, because the right
> version depends on your driver. Use `pixelflux[cuda]` or one of the commands above. If no
> compatible `libnvrtc` is present, conversion just falls back to the libyuv CPU path with no
> failure.

## Features

*   **Hybrid Backend:**
    *   **X11 (C++):** Legacy support using XShm.
    *   **Wayland (Rust):** Modern, secure, headless compositor based on [Smithay](https://github.com/Smithay/smithay).
*   **Flexible Encoding:**
    *   **Software:** libx264 (H.264, incl. 4:4:4) and libjpeg-turbo (JPEG) with multi-threaded striping.
    *   **Hardware:** NVIDIA NVENC (incl. High 4:4:4, multi-GPU containers, optional CUDA conversion) and VA-API (Intel/AMD) with Zero-Copy support.
    *   **Driver-aware GPU auto-selection** via `SELKIES_AUTO_GPU`.
*   **Zero-Copy Frames (X11 & Wayland):** the native frame object (buffer protocol) hands the encoded buffer to Python with no copy, on every supported Python version (3.9–3.14).
*   **Smart Bandwidth Management:**
    *   **Change Detection:** Encodes only changed stripes (Software/JPEG mode).
    *   **Paint-Over:** Automatically improves quality for static regions.
    *   **Damage Throttling:** Limits processing during high-motion scenes.
    *   **On-demand keyframes:** `request_idr_frame()` forces an IDR for reconnecting clients.
*   **Input Handling:** Built-in input injection for mouse and keyboard (Wayland).
*   **Cursor Compositing:** Hardware cursor planes or software rendering options.
*   **Dynamic Watermarking:** Overlay PNGs with static positioning or DVD-screensaver style animation.
*   **Recording Sink:** Direct Unix socket output of full-frame H.264 streams for local capture.

## License

This project is licensed under the **Mozilla Public License Version 2.0**.
A copy of the MPL 2.0 can be found at https://mozilla.org/MPL/2.0/.
