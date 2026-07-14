# pixelflux

[![PyPI version](https://badge.fury.io/py/pixelflux.svg)](https://badge.fury.io/py/pixelflux)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://linuxserver.github.io/pixelflux/)

**A performant web native pixel delivery pipeline for diverse sources, blending parallel processing of pixel buffers with flexible modern encoding formats.**

This module provides a Python interface to a high-performance capture library supporting both **X11** and **Wayland** environments. It captures pixel data, detects changes, and encodes modified stripes into JPEG or H.264.

It supports CPU-based encoding (x264, JPEG) as well as hardware-accelerated H.264 encoding via NVIDIA's NVENC and VA-API for Intel/AMD GPUs. Both backends share a **zero-copy pipeline** that minimizes copies and latency end to end.

## Installation

pixelflux is a single self-contained **Rust** extension (no C/C++ sources) compiled during installation. Both the X11 and Wayland backends, all encoders, and the Python API live in it.

### 1. Prerequisites

Ensure you have the Rust toolchain (`cargo`), Python development files, and the development libraries below.

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build dependencies (Debian/Ubuntu)
sudo apt-get update && \
sudo apt-get install -y \
  git \
  curl \
  python3-dev \
  cmake \
  nasm \
  libclang-dev \
  libavcodec-dev \
  libavutil-dev \
  libx264-dev \
  libturbojpeg0-dev \
  libgbm-dev \
  libdrm-dev \
  libwayland-dev \
  libinput-dev \
  libxkbcommon-dev \
  libva-dev
```

> **Notes:** the FFmpeg bindings (`ffmpeg-next` 8.1) require **FFmpeg 8.1**; on distros shipping an older FFmpeg, install a newer build and point `PKG_CONFIG_PATH` at it. X11 capture uses pure-Rust XCB (no `libX11`/`libxcb`/`Xfixes` dev packages needed); colorspace conversion is pure-Rust and the NVENC/CUDA libraries are loaded at runtime (no compile-time NVIDIA packages).

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
settings.video_crf = 25                            # CRF value (0-51, lower is better quality/higher bitrate)
settings.video_paintover_crf = 18                  # CRF for H.264 paintover on static content. Must be lower than video_crf to activate.
settings.video_paintover_burst_frames = 5          # Number of high-quality frames to send in a burst when a paintover is triggered.
settings.video_fullcolor = False                   # Use I444/full color (High 4:4:4) instead of I420. Supported by software encoding and NVENC.
settings.video_fullframe = True                    # Encode full frames (required for HW accel) instead of just changed stripes
settings.video_streaming_mode = False              # Bypass all VNC logic and work like a normal video encoder, higher constant CPU usage for fullscreen gaming/videos
settings.video_cbr_mode = False                    # Switches to CBR mode and ignores CRF value. Used in conjunction with video_bitrate_kbps.
settings.video_bitrate_kbps = 4000                 # Target bitrate for CBR mode. Required when video_cbr_mode is enabled.
settings.video_vbv_multiplier = 1.5                # Optional CBR VBV size as a multiple of one frame's bit budget (0 = auto: 1.5, or 3 with periodic keyframes).
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

**Performance Note:** Software (Pixman) rendering, the absence of a hardware encoder, or utilizing a render node different from the encoding node will force a "Readback" fallback, copying pixels to the CPU and breaking the zero-copy chain (higher latency and CPU load). A watermark does **not** force readback — on the GPU path it is composited into the frame before encoding.

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

## Computer Use Interface (Wayland)

The Wayland backend implements the [Anthropic Computer Use specification](https://github.com/anthropics/claude-quickstarts/tree/main/computer-use-demo), providing an HTTP API for AI agents to control the desktop. Enable it by setting the `PIXELFLUX_CU` environment variable to the port the server should listen on:

```bash
export PIXELFLUX_CU=5000
```

When using Computer Use, call `ensure_wayland_display()` before starting a capture to bring the compositor socket up early — this lets apps launched alongside your script connect to `WAYLAND_DISPLAY` immediately. GPU auto-selection (`SELKIES_AUTO_GPU=true` / `auto_gpu` on `CaptureSettings`) works normally; the screenshot path forces a single-frame CPU readback when the GPU is in zero-copy mode.

The Computer Use server listens for `POST` requests on `/computer-use` and responds with JSON. Unless otherwise noted, successful actions return:

```json
{"result":"ok"}
```

Coordinates are specified in absolute framebuffer pixels. Any coordinates outside the framebuffer are automatically clamped to the nearest valid pixel.

### Actions

All actions are `POST` requests to `/computer-use` with a JSON body.

**`screenshot`** - Capture the current display as a base64-encoded PNG:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"screenshot"}' | jq -r '.data' | base64 -d > screen.png
```

**`mouse_move`** - Move the cursor to absolute pixel coordinates:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"mouse_move","coordinate":[500,300]}'
```

**`left_click`** / **`right_click`** / **`middle_click`** - Click a mouse button, optionally at a coordinate and/or while holding a keyboard modifier:

```bash
# Simple click
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"left_click"}'

# Right click at a specific position while holding Shift
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"right_click","coordinate":[800,600],"text":"shift"}'
```

**`double_click`** / **`triple_click`** - Perform multiple left mouse clicks, optionally while holding a modifier:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"double_click","coordinate":[400,300]}'

curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"triple_click","text":"ctrl"}'
```

**`left_click_drag`** - Press the left mouse button at `start_coordinate`, drag to `coordinate`, then release:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"left_click_drag","start_coordinate":[100,100],"coordinate":[500,300]}'
```

**`left_mouse_down`** / **`left_mouse_up`** - Press or release the left mouse button without moving the pointer:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"left_mouse_down"}'
```

**`type`** - Type a string of text:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"type","text":"Hello, world!"}'
```

**`key`** - Press a key or key combination. Key combinations are specified using `+` separators:

```bash
# Single key
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"key","text":"Return"}'

# Key combination
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"key","text":"ctrl+s"}'

curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"key","text":"ctrl+alt+Delete"}'
```

**`hold_key`** - Hold a key for the specified duration (seconds). Durations are capped at 100 seconds.

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"hold_key","text":"ctrl","duration":2.0}'
```

**`scroll`** - Scroll vertically or horizontally, optionally at a coordinate and/or while holding a keyboard modifier:

```bash
# Scroll down 3 clicks
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"scroll","scroll_direction":"down","scroll_amount":3}'

# Scroll at a position while holding Shift
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"scroll","coordinate":[500,400],"scroll_direction":"up","scroll_amount":5,"text":"shift"}'
```

**`cursor_position`** - Return the current cursor position:

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"cursor_position"}' | jq -r '.text'
# → X=500,Y=300
```

**`wait`** - Pause execution for the specified duration (seconds). Durations are capped at 100 seconds.

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"wait","duration":0.5}'
```

**`zoom`** - Capture and return a cropped base64-encoded PNG of the specified framebuffer region (`[left, top, right, bottom]`):

```bash
curl -s -X POST http://localhost:5000/computer-use \
  -H 'Content-Type: application/json' \
  -d '{"action":"zoom","region":[100,200,400,350]}' | jq -r '.data' | base64 -d > zoomed.png
```

## NVIDIA NVENC (X11)

*   **Multi-GPU containers:** When several GPUs are exposed to a container, NVENC is filtered
    in-process to the GPU you selected (no separate `LD_PRELOAD` shim is required). Verified on
    NVIDIA drivers 570–595.
*   **4:4:4 (High 4:4:4):** Set `video_fullcolor = True` to encode full-chroma H.264 via NVENC
    (`video_fullcolor` codec), in addition to the software path.
*   **Force a keyframe on demand:** `capture.request_idr_frame()` forces an IDR frame, e.g. when
    a client reconnects or its decoder is reset. It routes to whichever encoder is active
    (NVENC, VA-API, or software) and is a no-op while no capture is running.

### NVENC color conversion

NVENC encodes the captured ARGB directly (the driver's hardware does the ARGB→NV12 colorspace
conversion in BT.709), so there is **no CUDA Toolkit / NVRTC requirement** — only the NVIDIA
driver runtime (`libnvidia-encode`, `libcuda`), which is loaded at runtime. Nothing extra to
install at build or runtime beyond the driver.

## Features

*   **Dual Backend (one Rust extension):**
    *   **X11:** XShm capture via pure-Rust XCB, with XFixes cursor and watermark compositing.
    *   **Wayland:** Modern, secure, headless compositor based on [Smithay](https://github.com/Smithay/smithay).
*   **Flexible Encoding:**
    *   **Software:** x264 (H.264, incl. 4:4:4) and JPEG with multi-threaded striping.
    *   **Hardware:** NVIDIA NVENC (incl. High 4:4:4, ARGB-direct BT.709, multi-GPU containers, API-version negotiation) and VA-API (Intel/AMD, VA-VPP convert) with Zero-Copy support.
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
*   **AI Agent Control:** Computer Use API to dump screenshots and drive all facets of a desktop environment.

## License

This project is licensed under the **Mozilla Public License Version 2.0**.
A copy of the MPL 2.0 can be found at https://mozilla.org/MPL/2.0/.
