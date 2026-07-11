/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! X11 host capture: grab the root window into host memory as BGRA, composite the XFixes hardware
//! cursor and the watermark on the CPU, and feed each frame to [`X11Pipeline`], which owns
//! damage/stripe/encode. The grab goes through a shared-memory segment (XShm via x11rb) rather than
//! a plain `GetImage` for one reason: a full-screen frame is far too large to copy through the X
//! protocol socket every tick, so XShm has the server write the pixels straight into memory this
//! process already has mapped.
//!
//! [`run_capture`] splits the work across two threads because grabbing the next frame and encoding
//! the previous one have no reason to wait on each other: the caller's thread grabs frames and owns
//! the x11rb connection and the pool of shm surfaces, while a spawned encode thread owns the
//! [`X11Pipeline`] (and thus the encoder) and runs the delivery callback, so the two overlap for
//! throughput. Frames hand off through a bounded [`FramePool`] that carries only a raw pointer and
//! geometry — never an X object, none of which is safe to share — so nothing X-related ever crosses
//! the thread boundary and the encoder never has to touch X. Multi-instance safety for the encoders
//! is handled inside them (e.g. the libx264 open/close lock); each capture owns its own private xcb
//! connection, so there is no shared X state to serialize here.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use x11rb::connection::Connection;
use x11rb::protocol::shm::ConnectionExt as ShmExt;
use x11rb::protocol::xfixes::ConnectionExt as XfixesExt;
use x11rb::protocol::xproto::{ConnectionExt as XprotoExt, ImageFormat};
use x11rb::rust_connection::RustConnection;

use crate::encoders::overlay::WatermarkLocation;
use crate::encoders::software::EncodedStripe;
use crate::pipeline::X11Pipeline;
use crate::recording_sink::RecordingSink;
use crate::RustCaptureSettings;

/// @brief Cross-thread controls for a running capture: a bag of atomics (plus two mutex-guarded
/// payloads) the owning `ScreenCapture` pyclass flips from the Python thread and the capture thread
/// reads at the top of each iteration.
///
/// It exists so Python never has to reach into the pipeline. The `X11Pipeline` and its encoder are
/// only safe to touch from the capture and encode threads, so `request_idr` / rate / fps / region
/// changes are posted here as atomic flags and applied later on the thread that owns the pipeline,
/// rather than mutating encoder state across the thread boundary from Python.
///
/// 1. **Lifecycle**: `stop` ends the capture loop; `force_idr` requests an on-demand keyframe on the
///    next processed frame.
/// 2. **Rate control** (gated by `rate_dirty`): `bitrate_kbps`, `vbv_mult_milli` (the VBV frame-time
///    multiplier * 1000, held as an integer for atomics; `<= 0` selects the policy default), and
///    `fps_milli` (target fps * 1000, re-read every frame for dynamic pacing and rate control). These
///    atomics always hold the CURRENT values, so a pipeline rebuild can carry live rates forward.
/// 3. **Tunables** (gated by `tunables_dirty`): one `LiveTunables` struct behind `tunables`, set
///    rarely, carrying the per-frame quality knobs for the encode thread.
/// 4. **Capture geometry**: `capture_cursor` toggles the cursor overlay (read on every grab);
///    `region_dirty` guards `region` = `(x, y, w, h)` (w/h `<= 0` extends to the root edge), applied
///    by the capture thread with the same drain/recreate machinery as auto-adjust.
pub struct Controls {
    pub stop: AtomicBool,
    pub force_idr: AtomicBool,
    pub rate_dirty: AtomicBool,
    pub bitrate_kbps: AtomicI32,
    pub vbv_mult_milli: AtomicI32,
    pub fps_milli: AtomicU64,
    pub tunables_dirty: AtomicBool,
    pub tunables: Mutex<Option<crate::LiveTunables>>,
    pub capture_cursor: AtomicBool,
    pub region_dirty: AtomicBool,
    pub region: Mutex<(i32, i32, i32, i32)>,
}

impl Controls {
    /// @brief Seed the controls from the initial settings so the first loop iteration reads the
    /// configured bitrate / VBV / fps / region / cursor state rather than defaults.
    pub fn new(s: &RustCaptureSettings) -> Self {
        Self {
            stop: AtomicBool::new(false),
            force_idr: AtomicBool::new(false),
            rate_dirty: AtomicBool::new(false),
            bitrate_kbps: AtomicI32::new(s.video_bitrate_kbps),
            vbv_mult_milli: AtomicI32::new((s.video_vbv_multiplier * 1000.0).round() as i32),
            fps_milli: AtomicU64::new((s.target_fps.max(1.0) * 1000.0) as u64),
            tunables_dirty: AtomicBool::new(false),
            tunables: Mutex::new(None),
            capture_cursor: AtomicBool::new(s.capture_cursor),
            region_dirty: AtomicBool::new(false),
            region: Mutex::new((s.capture_x, s.capture_y, s.width, s.height)),
        }
    }
}

/// @brief A shared-memory image surface: a POSIX shm segment mapped into both this process and the
/// X server, into which `shm_get_image` writes one BGRA frame.
///
/// Both ends map the same physical pages, which is the whole point: a full-screen grab then costs no
/// per-frame copy across the X socket, because the server's blit lands directly in memory the
/// capture thread already reads. The surface owns the server-side segment id (`shmseg`), the local
/// mapping (`addr` / `size`), and the geometry (`width` / `height` / `stride`) needed to interpret
/// the bytes.
struct ShmSurface {
    shmseg: u32,
    addr: *mut u8,
    size: usize,
    width: u16,
    height: u16,
    stride: usize,
}

impl ShmSurface {
    /// @brief Allocate a shm segment of `width*height*4` bytes, attach it both locally and to the X
    /// server, and hand back a surface both ends can read.
    ///
    /// The segment is created with `shmget(IPC_PRIVATE)` and mapped locally via `shmat`, then
    /// attached on the server (`shm_attach`). That attach is confirmed with a round-trip `check()`
    /// BEFORE the segment is marked `IPC_RMID`: deleting only after both ends hold a reference means
    /// the kernel frees the segment once this process and the server have both detached, never while
    /// either still needs it. Every failure path unwinds the partial state (local detach and/or
    /// `IPC_RMID`) so a failed allocation leaks neither memory nor a server attachment; a zero-sized
    /// request is rejected up front.
    fn create(conn: &RustConnection, width: u16, height: u16) -> Result<Self, String> {
        let stride = width as usize * 4;
        let size = stride * height as usize;
        if size == 0 {
            return Err("zero-sized capture surface".into());
        }
        unsafe {
            let shmid = libc::shmget(libc::IPC_PRIVATE, size, libc::IPC_CREAT | 0o600);
            if shmid < 0 {
                return Err("shmget failed".into());
            }
            let addr = libc::shmat(shmid, std::ptr::null(), 0);
            if addr == (-1isize) as *mut libc::c_void {
                libc::shmctl(shmid, libc::IPC_RMID, std::ptr::null_mut());
                return Err("shmat failed".into());
            }
            let shmseg = match conn.generate_id() {
                Ok(id) => id,
                Err(e) => {
                    libc::shmdt(addr);
                    libc::shmctl(shmid, libc::IPC_RMID, std::ptr::null_mut());
                    return Err(format!("generate_id: {e}"));
                }
            };
            let attach = conn
                .shm_attach(shmseg, shmid as u32, false)
                .map_err(|e| format!("shm_attach: {e}"))
                .and_then(|c| c.check().map_err(|e| format!("shm_attach check: {e}")));
            libc::shmctl(shmid, libc::IPC_RMID, std::ptr::null_mut());
            if let Err(e) = attach {
                libc::shmdt(addr);
                return Err(e);
            }
            Ok(Self {
                shmseg,
                addr: addr as *mut u8,
                size,
                width,
                height,
                stride,
            })
        }
    }

    /// @brief Borrow the mapped segment as a mutable byte slice for the capture blit and overlays.
    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.addr, self.size) }
    }

    /// @brief Borrow the mapped segment read-only, for the stability comparison.
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.addr, self.size) }
    }

    /// @brief Detach the segment from the X server and then locally, clearing the mapped pointer.
    ///
    /// This is a manual method rather than a `Drop` impl because releasing the server-side
    /// attachment is an X protocol request that needs the live connection, which a `Drop` could not
    /// reach; the owner of the connection must call it before the surface is dropped. The order —
    /// server `shm_detach` first, then the local `shmdt` — completes the release of the segment that
    /// was marked `IPC_RMID` at creation, so the kernel reclaims it once neither end holds it.
    fn destroy(&mut self, conn: &RustConnection) {
        let _ = conn.shm_detach(self.shmseg);
        let _ = conn.flush();
        unsafe {
            libc::shmdt(self.addr as *mut libc::c_void);
        }
        self.addr = std::ptr::null_mut();
    }
}

/// @brief Resolve the capture dimensions `shm_get_image` will read, bounded so the region can never
/// run past the live root from the capture offset.
///
/// The bound is the reason this function exists: a grab that ran off the root edge would fail the
/// request outright, so the size is always clamped to what is actually there. With auto-adjust (or
/// an unset width/height `<= 0`) the capture tracks the full root minus the capture offset;
/// otherwise the requested size is clamped to what is available from that offset. `capture_x` /
/// `capture_y` are floored at `>= 0` exactly as `shm_get_image` consumes them, and saturating
/// subtraction plus a final `u16` clamp (minimum 2) keep pathological settings from overflowing or
/// collapsing to an unusable surface. H.264 (`output_mode == 1`) requires even dimensions, so both
/// are then rounded down to a multiple of two.
fn resolve_dims(root_w: u16, root_h: u16, s: &RustCaptureSettings) -> (u16, u16) {
    let cap_x = s.capture_x.max(0);
    let cap_y = s.capture_y.max(0);
    let avail_w = (root_w as i32).saturating_sub(cap_x).max(2);
    let avail_h = (root_h as i32).saturating_sub(cap_y).max(2);
    let mut w = if s.auto_adjust_screen_capture_size || s.width <= 0 {
        avail_w
    } else {
        s.width.min(avail_w)
    };
    let mut h = if s.auto_adjust_screen_capture_size || s.height <= 0 {
        avail_h
    } else {
        s.height.min(avail_h)
    };
    w = w.clamp(2, u16::MAX as i32);
    h = h.clamp(2, u16::MAX as i32);
    if s.output_mode == 1 {
        w &= !1;
        h &= !1;
    }
    (w as u16, h as u16)
}

/// Rows per stability-verification strip.
const VERIFY_STRIP_ROWS: usize = 2;
/// Number of verification strips spread over the frame height.
const VERIFY_STRIPS: usize = 16;

/// @brief Row origins for the stability-verification strips: up to `n` strips of
/// `rows` height spread evenly from the top edge to the bottom edge inclusive.
fn verify_strip_rows(height: usize, n: usize, rows: usize) -> Vec<usize> {
    if height <= rows * 2 {
        return vec![0];
    }
    let n = n.max(2);
    (0..n).map(|i| i * (height - rows) / (n - 1)).collect()
}

/// @brief Compare each verification strip against the same rows of the grabbed frame;
/// `true` when every sampled row is byte-identical, i.e. no client painted the sampled
/// rows between the two grabs.
fn verify_strips_match(frame: &[u8], stride: usize, verify: &[u8], ys: &[usize], rows: usize) -> bool {
    ys.iter().enumerate().all(|(i, &y)| {
        let f = &frame[y * stride..(y + rows) * stride];
        let v = &verify[i * rows * stride..(i + 1) * rows * stride];
        f == v
    })
}

/// @brief Copy the strip rows out of the grabbed frame into `dst` (strip-packed, the
/// same layout the verify segment uses), recording what was published for the next
/// frame's changed-content check.
fn pack_frame_strips(frame: &[u8], stride: usize, ys: &[usize], rows: usize, dst: &mut Vec<u8>) {
    dst.clear();
    for &y in ys {
        dst.extend_from_slice(&frame[y * stride..(y + rows) * stride]);
    }
}

/// @brief Outcome of [`grab_stable_frame`]: whether the content changed since the last accepted
/// frame, and the pacing `anchor` — the instant the grab was issued, so the caller paces one
/// frame period from here.
struct StableGrab {
    changed: bool,
    anchor: Instant,
}

/// @brief Grab one frame into `surface` with a single XShm round-trip and detect whether
/// the content changed since the last accepted frame.
///
/// A single atomic `shm_get_image` captures the full frame; the server never interleaves
/// another client inside one request, so the grab is always a coherent snapshot. Changed
/// content is detected by comparing sampled strip rows against the previous frame — when
/// every sampled strip matches, the frame is unchanged and can be skipped by the encoder.
#[allow(clippy::too_many_arguments)]
fn grab_stable_frame(
    conn: &RustConnection,
    root: u32,
    surface: &mut ShmSurface,
    prev_strips: &mut Vec<u8>,
    cap_x: i16,
    cap_y: i16,
    cap_w: u16,
    cap_h: u16,
) -> Result<StableGrab, String> {
    let grab_start = Instant::now();
    let stride = surface.stride;
    let ys = verify_strip_rows(cap_h as usize, VERIFY_STRIPS, VERIFY_STRIP_ROWS);

    conn.shm_get_image(
        root,
        cap_x,
        cap_y,
        cap_w,
        cap_h,
        !0u32,
        ImageFormat::Z_PIXMAP.into(),
        surface.shmseg,
        0,
    )
    .map_err(|e| format!("shm_get_image: {e}"))?
    .reply()
    .map_err(|e| format!("shm_get_image reply: {e}"))?;

    let strips_len = ys.len() * VERIFY_STRIP_ROWS * stride;
    let changed = prev_strips.len() != strips_len
        || !verify_strips_match(surface.as_slice(), stride, prev_strips, &ys, VERIFY_STRIP_ROWS);
    pack_frame_strips(surface.as_slice(), stride, &ys, VERIFY_STRIP_ROWS, prev_strips);
    Ok(StableGrab { changed, anchor: grab_start })
}

/// @brief Alpha-blend a source pixel (pre-split into r,g,b,a) over a BGRA destination pixel.
///
/// Cursor and watermark pixels are overwhelmingly either fully opaque or fully transparent, and this
/// runs per pixel per frame on the CPU, so the two extremes are special-cased to skip the blend
/// arithmetic entirely: an opaque source (`a == 255`) simply overwrites, a fully transparent source
/// (`a == 0`) is left as-is, and only genuine edge pixels pay for the integer source-over. In every
/// case only the B / G / R bytes are written; the destination's alpha byte is left as the capture
/// delivered it.
#[inline]
fn blend_pixel(dst: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
    if a == 255 {
        dst[0] = b;
        dst[1] = g;
        dst[2] = r;
    } else if a > 0 {
        let ia = 255 - a as u32;
        dst[0] = ((b as u32 * a as u32 + dst[0] as u32 * ia) / 255) as u8;
        dst[1] = ((g as u32 * a as u32 + dst[1] as u32 * ia) / 255) as u8;
        dst[2] = ((r as u32 * a as u32 + dst[2] as u32 * ia) / 255) as u8;
    }
}

/// @brief Frame-space top-left of the cursor image, given the XFixes hotspot position.
///
/// XFixes reports the cursor position at its HOTSPOT; X draws the image with its top-left at
/// `(pos - hot)`, and the capture origin `(cap_x, cap_y)` is subtracted to move it into frame space.
/// The result may go negative near the frame edges, which `overlay_cursor` clips per pixel to match
/// the server's own edge clipping.
#[inline]
fn cursor_image_origin(x: i16, y: i16, xhot: u16, yhot: u16, cap_x: i32, cap_y: i32) -> (i32, i32) {
    (x as i32 - xhot as i32 - cap_x, y as i32 - yhot as i32 - cap_y)
}

/// @brief Composite the XFixes cursor (ARGB `u32` per pixel) onto the BGRA frame with its top-left
/// at `(img_x, img_y)`, blending each pixel through `blend_pixel` with per-pixel bounds clipping so
/// an image straddling a frame edge writes only its in-frame portion.
#[allow(clippy::too_many_arguments)]
fn overlay_cursor(
    frame: &mut [u8],
    stride: usize,
    frame_w: i32,
    frame_h: i32,
    cur_w: i32,
    cur_h: i32,
    pixels: &[u32],
    img_x: i32,
    img_y: i32,
) {
    for y in 0..cur_h {
        let ty = img_y + y;
        if ty < 0 || ty >= frame_h {
            continue;
        }
        for x in 0..cur_w {
            let tx = img_x + x;
            if tx < 0 || tx >= frame_w {
                continue;
            }
            let px = pixels[(y * cur_w + x) as usize];
            let a = ((px >> 24) & 0xFF) as u8;
            let r = ((px >> 16) & 0xFF) as u8;
            let g = ((px >> 8) & 0xFF) as u8;
            let b = (px & 0xFF) as u8;
            let off = ty as usize * stride + tx as usize * 4;
            blend_pixel(&mut frame[off..off + 4], r, g, b, a);
        }
    }
}

/// @brief CPU watermark: the raw RGBA pixels (row-major, `w*h*4`) in host memory plus the
/// placement / animation state, blended directly into the captured BGRA frame.
///
/// The blend is on the CPU because the frame is already sitting in host shm memory before it reaches
/// any encoder, so stamping the overlay there is both the cheapest place to do it and the one place
/// it applies identically regardless of which encoder (hardware or software) runs downstream. The
/// pixels are kept as RGBA straight from `image`'s decode and converted per pixel as they are blended.
struct X11Watermark {
    pixels: Vec<u8>,
    w: i32,
    h: i32,
    pos_x: i32,
    pos_y: i32,
    sub_x: f64,
    sub_y: f64,
    vel_x: f64,
    vel_y: f64,
    loaded: bool,
}

impl X11Watermark {
    /// @brief Load the watermark image at `path` into host RGBA, or return an unloaded stub.
    ///
    /// An empty path or any decode failure yields `loaded == false`, which every method treats as a
    /// no-op, so a missing or broken watermark simply disables the overlay. The bouncing-animation
    /// velocities are seeded here.
    fn load(path: &str) -> Self {
        let mut wm = Self {
            pixels: Vec::new(),
            w: 0,
            h: 0,
            pos_x: 0,
            pos_y: 0,
            sub_x: 0.0,
            sub_y: 0.0,
            vel_x: 2.0,
            vel_y: 2.0,
            loaded: false,
        };
        if path.is_empty() {
            return wm;
        }
        if let Ok(img) = image::open(std::path::Path::new(path)) {
            let rgba = img.to_rgba8();
            wm.w = rgba.width() as i32;
            wm.h = rgba.height() as i32;
            wm.pixels = rgba.into_vec();
            wm.loaded = wm.w > 0 && wm.h > 0;
        }
        wm
    }

    /// @brief Set the watermark's top-left placement for this frame from the location enum; a stub
    /// (unloaded) watermark returns immediately.
    ///
    /// The fixed corners and center are direct arithmetic. The animated mode advances a bouncing
    /// position and reflects velocity off each frame edge, accumulating in the fractional `sub_x` /
    /// `sub_y` rather than the integer `pos_x` / `pos_y`: a sub-pixel-per-frame velocity has to
    /// survive between frames or the motion would either stall or snap by whole pixels, so the float
    /// carries the remainder and `pos_x` / `pos_y` take its floor for the actual blit.
    fn update_position(&mut self, frame_w: i32, frame_h: i32, loc_enum: i32) {
        if !self.loaded {
            return;
        }
        match WatermarkLocation::from(loc_enum) {
            WatermarkLocation::TL => {
                self.pos_x = 0;
                self.pos_y = 0;
            }
            WatermarkLocation::TR => {
                self.pos_x = frame_w - self.w;
                self.pos_y = 0;
            }
            WatermarkLocation::BL => {
                self.pos_x = 0;
                self.pos_y = frame_h - self.h;
            }
            WatermarkLocation::BR => {
                self.pos_x = frame_w - self.w;
                self.pos_y = frame_h - self.h;
            }
            WatermarkLocation::MI => {
                self.pos_x = (frame_w - self.w) / 2;
                self.pos_y = (frame_h - self.h) / 2;
            }
            WatermarkLocation::AN => {
                self.sub_x += self.vel_x;
                self.sub_y += self.vel_y;
                if self.sub_x <= 0.0 {
                    self.sub_x = 0.0;
                    self.vel_x = self.vel_x.abs();
                } else if self.sub_x + self.w as f64 >= frame_w as f64 {
                    self.sub_x = (frame_w - self.w) as f64;
                    self.vel_x = -self.vel_x.abs();
                }
                if self.sub_y <= 0.0 {
                    self.sub_y = 0.0;
                    self.vel_y = self.vel_y.abs();
                } else if self.sub_y + self.h as f64 >= frame_h as f64 {
                    self.sub_y = (frame_h - self.h) as f64;
                    self.vel_y = -self.vel_y.abs();
                }
                self.pos_x = self.sub_x as i32;
                self.pos_y = self.sub_y as i32;
            }
            WatermarkLocation::None => (),
        }
    }

    /// @brief Alpha-blend the loaded watermark into the BGRA frame at its current position; a no-op
    /// when unloaded. Clips per pixel at the frame bounds because the animated position — or a
    /// watermark larger than the capture — can leave part of the image off-frame, and only the
    /// in-frame portion may be written.
    fn blend_into(&self, frame: &mut [u8], stride: usize, frame_w: i32, frame_h: i32) {
        if !self.loaded {
            return;
        }
        for y in 0..self.h {
            let ty = self.pos_y + y;
            if ty < 0 || ty >= frame_h {
                continue;
            }
            for x in 0..self.w {
                let tx = self.pos_x + x;
                if tx < 0 || tx >= frame_w {
                    continue;
                }
                let src = ((y * self.w + x) * 4) as usize;
                let (r, g, b, a) = (
                    self.pixels[src],
                    self.pixels[src + 1],
                    self.pixels[src + 2],
                    self.pixels[src + 3],
                );
                let off = ty as usize * stride + tx as usize * 4;
                blend_pixel(&mut frame[off..off + 4], r, g, b, a);
            }
        }
    }
}

/// @brief A captured raw BGRA frame held in a pooled shm surface, ready to encode.
///
/// It carries the surface pointer plus geometry so the encode thread reads the pixels directly (no
/// copy) and can rebuild its pipeline when the capture size changed (auto-adjust). It also carries
/// the pool's surface `generation`, so a rebuild is still triggered when the surfaces were recreated
/// at the SAME size (a resize flap) and the new segments happen to reuse the old virtual addresses.
struct RawFrame {
    idx: usize,
    ptr: *mut u8,
    len: usize,
    width: u16,
    height: u16,
    stride: usize,
    generation: u64,
}
/// @brief `RawFrame` is `Send`: its raw pointer addresses a pooled shm surface that the pool
/// guarantees is not reused until the encode thread recycles this frame, so the handle is safe to
/// move across the capture -> encode thread boundary.
unsafe impl Send for RawFrame {}

/// @brief Mutex-guarded interior of [`FramePool`]: the free-surface index list and the single
/// capture -> encode handoff slot, under one lock so moving a surface between them — acquire,
/// publish, take, recycle — is always a single atomic step.
struct PoolInner {
    free: Vec<usize>,
    slot: Option<RawFrame>,
}

/// @brief Demand-driven capture -> encode handoff: a bounded single-slot channel over a fixed set of
/// pooled shm surfaces.
///
/// The capture thread writes into a pooled surface and `publish`es it into the single `slot`; the
/// encode thread `take`s it. Capture stays at most one frame ahead of encode because `acquire` and
/// `publish` BLOCK (bounded) until the encoder frees a surface / drains the slot, throttling capture
/// to the encode rate. Since X11 capture is pull-based (no backlog), throttling — rather than
/// capturing-then-dropping — means a full-resolution shm round-trip is never spent on a frame that
/// would be discarded, while the next capture still overlaps the current encode (the throughput win).
/// The encoder therefore only ever sees a contiguous frame stream, keeping the H.264 reference chain
/// valid.
///
/// `generation` is bumped by the capture thread each time the backing shm surfaces are destroyed and
/// recreated. Published frames carry it so the encode thread rebuilds its pipeline — dropping encoder
/// state keyed to surface base pointers (e.g. NVENC's pinned-host registrations) — even when a resize
/// flap lands back on the old dimensions and the recreated segments reuse the old virtual addresses.
struct FramePool {
    inner: Mutex<PoolInner>,
    cv: Condvar,
    stop: AtomicBool,
    generation: AtomicU64,
}

impl FramePool {
    /// @brief Create a pool with `n` free surfaces, an empty handoff slot, and generation 0.
    fn new(n: usize) -> Self {
        Self {
            inner: Mutex::new(PoolInner { free: (0..n).collect(), slot: None }),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
            generation: AtomicU64::new(0),
        }
    }

    /// @brief Capture: record that the surfaces were recreated by advancing the generation. Only
    /// called after `drain_for_resize` succeeded, so no frame from the previous generation is still
    /// in flight.
    fn bump_generation(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// @brief Read the current surface generation, stamped onto each published frame.
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// @brief Capture: claim a free surface to write the next frame into, or `None` on stop.
    ///
    /// Blocks (bounded, re-checking `stop` every 20ms) until a surface is free, which throttles
    /// capture to the encode rate so a full-resolution capture is never spent on a frame that would
    /// be dropped. The bounded re-check means a stop that races the wait cannot hang the thread.
    fn acquire(&self, stop: &AtomicBool) -> Option<usize> {
        let mut g = self.inner.lock().unwrap();
        loop {
            if let Some(idx) = g.free.pop() {
                return Some(idx);
            }
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let (gg, _) = self.cv.wait_timeout(g, Duration::from_millis(20)).unwrap();
            g = gg;
        }
    }

    /// @brief Capture: publish the just-captured frame into the single slot, or `false` (frame
    /// discarded) on stop.
    ///
    /// Blocks (bounded, re-checking `stop` every 20ms) until the encode thread has taken the previous
    /// frame, so capture stays at most one frame ahead and never drops under normal flow.
    fn publish(&self, frame: RawFrame, stop: &AtomicBool) -> bool {
        let mut g = self.inner.lock().unwrap();
        loop {
            if g.slot.is_none() {
                g.slot = Some(frame);
                drop(g);
                self.cv.notify_all();
                return true;
            }
            if stop.load(Ordering::Relaxed) {
                return false;
            }
            let (gg, _) = self.cv.wait_timeout(g, Duration::from_millis(20)).unwrap();
            g = gg;
        }
    }

    /// @brief Encode: block until a frame is available (`Some`) or stop is signalled (`None`).
    ///
    /// The wait is bounded (re-checking `stop` every 20ms) as defense-in-depth against a lost wakeup,
    /// so a stop that races the park can never leave the encode thread blocked forever.
    fn take(&self) -> Option<RawFrame> {
        let mut g = self.inner.lock().unwrap();
        loop {
            if let Some(f) = g.slot.take() {
                return Some(f);
            }
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            let (gg, _) = self.cv.wait_timeout(g, Duration::from_millis(20)).unwrap();
            g = gg;
        }
    }

    /// @brief Encode: return a surface to the free list after it has been encoded, waking any waiter
    /// parked in `acquire` / `publish` / `drain_for_resize`.
    fn recycle(&self, idx: usize) {
        self.inner.lock().unwrap().free.push(idx);
        self.cv.notify_all();
    }

    /// @brief Capture: before recreating surfaces (auto-adjust / region resize), reclaim the pending
    /// slot and wait until every surface is back in the free list, so no surface is destroyed while
    /// the encode thread is still reading it.
    ///
    /// Reclaiming the slot returns any un-taken frame to the free list; the loop then waits until
    /// `free.len() == n`, i.e. the encode thread has finished any in-flight frame. The wait is bounded
    /// and also breaks on stop — pool shutdown OR the external `stop` — so a panicked or dead encode
    /// thread that never recycles cannot wedge the capture thread here forever; a requested stop
    /// unblocks it. Returns `true` if it fully drained (safe to recreate surfaces), or `false` if it
    /// aborted on stop — in which case the caller tears down, which joins the encode thread before
    /// destroying surfaces, so the resize-safety guarantee still holds.
    fn drain_for_resize(&self, n: usize, stop: &AtomicBool) -> bool {
        let mut g = self.inner.lock().unwrap();
        if let Some(old) = g.slot.take() {
            g.free.push(old.idx);
        }
        while g.free.len() < n {
            if self.stop.load(Ordering::Acquire) || stop.load(Ordering::Relaxed) {
                return false;
            }
            let (gg, _) = self.cv.wait_timeout(g, Duration::from_millis(20)).unwrap();
            g = gg;
        }
        true
    }

    /// @brief Signal every waiter to stop and wake them, ending the encode thread's `take` loop.
    ///
    /// The inner mutex is acquired BEFORE storing `stop` and notifying, which closes the lost-wakeup
    /// window: with the lock held, `take` is either still before its `stop` check or already parked on
    /// the condvar (and will receive the notify), never in the gap between the two. The notify is
    /// issued after the guard is dropped so the woken thread does not immediately re-block on the lock
    /// this call holds.
    fn shutdown(&self) {
        let g = self.inner.lock().unwrap();
        self.stop.store(true, Ordering::Release);
        drop(g);
        self.cv.notify_all();
    }
}

/// @brief Encode thread body: consume captured frames from the pool, keep the [`X11Pipeline`] in
/// step with the capture size and cross-thread controls, encode, recycle, and deliver.
///
/// The loop runs until [`FramePool::take`] returns `None` (stop). For each frame:
///
/// 1. **(Re)build the pipeline** when it is missing, the frame size changed, or the surface
///    generation changed. A generation change alone forces the rebuild path because recreated shm
///    segments often reuse the old virtual base addresses, so encoder state keyed to base pointers
///    (NVENC's pinned-host cache) must be dropped even at identical dimensions. An in-place
///    [`X11Pipeline::reshape`] is preferred — NVENC reconfigures its live session and the striped
///    software path just re-derives stripe state — so a resize never stalls on a full encoder
///    re-init; encoders that cannot follow in place (VAAPI/OpenH264) report `false` and the pipeline
///    is rebuilt. On a rebuild the old pipeline (with its GPU session/surfaces) is dropped BEFORE the
///    new one is built, so an auto-adjust resize never holds two full encoder allocations at once
///    (transient 2x GPU memory). The rebuilt settings pull the CURRENT bitrate / VBV / fps from the
///    controls atomics, carrying live rate changes forward instead of reverting to the capture-start
///    values.
/// 2. **Apply cross-thread controls** here, on the thread that owns the pipeline: a pending
///    `force_idr` requests a keyframe; a `rate_dirty` swap (Acquire, pairing with the setters' Release
///    stores so the bitrate / VBV / fps payload is never seen half-applied) pushes a live rate change;
///    a `tunables_dirty` swap applies per-frame tunables into both the local settings copy and the
///    pipeline.
/// 3. **Encode and hand off**: read the pooled surface directly through its raw pointer — sound
///    because the pool guarantees the surface is not reused until it is recycled — run
///    [`X11Pipeline::process`], recycle the surface BEFORE delivering so a slow consumer never holds a
///    capture surface, then deliver any encoded stripes through `on_frame`.
///
/// The optional Unix-socket recording sink (parity with the Wayland path) is bound ONCE here and
/// owned outside the pipeline, so pipeline rebuilds on resize keep the socket listener and any
/// attached recorders alive. It can only carry a single full-frame H.264 stream, so configurations
/// that cannot produce one (JPEG output, or a striped CPU encoder) are warned about up front.
fn encode_loop<F>(pool: &FramePool, controls: &Controls, settings: &RustCaptureSettings, on_frame: &mut F)
where
    F: FnMut(Vec<EncodedStripe>),
{
    let mut psettings = settings.clone();
    let recording_sink = RecordingSink::try_bind(&settings.recording_socket);
    if recording_sink.is_some() {
        if settings.output_mode == 0 {
            eprintln!("[recording_sink] recording_socket set but output_mode is JPEG; no recordable H.264 stream.");
        } else if settings.use_cpu && !settings.use_openh264 && !settings.video_fullframe {
            eprintln!("[recording_sink] recording_socket set but the CPU encoder is striped; set video_fullframe=true for a recordable stream.");
        }
    }
    let mut pipeline: Option<X11Pipeline> = None;
    let (mut pw, mut ph) = (0i32, 0i32);
    let mut pgen = 0u64;
    let mut last_log_time = Instant::now();
    let mut frame_count: u64 = 0;
    let mut stripe_count: u64 = 0;

    while let Some(frame) = pool.take() {
        let (fw, fh) = (frame.width as i32, frame.height as i32);
        if pipeline.is_none() || pw != fw || ph != fh || pgen != frame.generation {
            let size_changed = pw != fw || ph != fh;
            psettings.width = fw;
            psettings.height = fh;
            psettings.video_bitrate_kbps = controls.bitrate_kbps.load(Ordering::Relaxed);
            psettings.video_vbv_multiplier =
                controls.vbv_mult_milli.load(Ordering::Relaxed) as f64 / 1000.0;
            psettings.target_fps =
                (controls.fps_milli.load(Ordering::Relaxed).max(1) as f64) / 1000.0;
            let reshaped = pipeline
                .as_mut()
                .is_some_and(|pl| pl.reshape(&psettings, size_changed));
            if !reshaped {
                drop(pipeline.take());
                pipeline = Some(X11Pipeline::new(psettings.clone(), recording_sink.clone()));
                if let Some(pl) = &pipeline {
                    let enc_name = pl.encoder_name();
                    let mut log_msg = format!(
                        "[x11] Stream settings active -> Res: {}x{} | FPS: {:.1} | Encoder: {}",
                        psettings.width, psettings.height, psettings.target_fps, enc_name
                    );
                    if psettings.output_mode == 0 {
                        log_msg.push_str(&format!(" | Mode: JPEG | Quality: {}", psettings.jpeg_quality));
                    } else {
                        log_msg.push_str(&format!(" | Mode: H264 | CRF: {}", psettings.video_crf));
                        if psettings.video_fullcolor {
                            log_msg.push_str(" | Colorspace: I444 (Full Range)");
                        } else {
                            log_msg.push_str(" | Colorspace: I420 (Limited Range)");
                        }
                    }
                    log_msg.push_str(&format!(
                        " | Damage Thresh: {}f | Damage Dur: {}f",
                        psettings.damage_block_threshold, psettings.damage_block_duration
                    ));
                    println!("{}", log_msg);
                }
            }
            pw = fw;
            ph = fh;
            pgen = frame.generation;
        }
        let pl = pipeline.as_mut().unwrap();

        if controls.force_idr.swap(false, Ordering::Relaxed) {
            pl.request_idr();
        }
        if controls.rate_dirty.swap(false, Ordering::Acquire) {
            let b = controls.bitrate_kbps.load(Ordering::Relaxed);
            let v = controls.vbv_mult_milli.load(Ordering::Relaxed) as f64 / 1000.0;
            let fps = (controls.fps_milli.load(Ordering::Relaxed).max(1) as f64) / 1000.0;
            pl.update_rate(b, v, fps);
        }
        if controls.tunables_dirty.swap(false, Ordering::Acquire) {
            if let Some(t) = controls.tunables.lock().unwrap().take() {
                t.apply_to(&mut psettings);
                pl.update_tunables(&t);
            }
        }

        let buf = unsafe { std::slice::from_raw_parts(frame.ptr, frame.len) };
        let stripes = pl.process(buf, frame.stride);
        pool.recycle(frame.idx);
        if !stripes.is_empty() {
            frame_count += 1;
            stripe_count += stripes.len() as u64;
            on_frame(stripes);
        }

        let now = Instant::now();
        let elapsed = now.duration_since(last_log_time).as_secs_f64();
        if elapsed >= 1.0 {
            if settings.debug_logging {
                let actual_fps = frame_count as f64 / elapsed;
                let stripes_per_sec = stripe_count as f64 / elapsed;
                println!(
                    "[x11] Res: {}x{} Encoder: {} EncFPS: {:.2} EncStripes/s: {:.2}",
                    psettings.width, psettings.height, pl.encoder_name(), actual_fps, stripes_per_sec
                );
            }
            frame_count = 0;
            stripe_count = 0;
            last_log_time = now;
        }
    }
}

/// @brief Run the X11 capture pipeline until `stop` is set, splitting capture and encode across two
/// threads that overlap for throughput.
///
/// This (the caller's) thread performs setup and then the capture loop; a spawned encode thread runs
/// [`encode_loop`] and invokes `on_frame(stripes)` once per encoded frame. The split lets capture and
/// encode overlap, and because frames are throttled/dropped as RAW frames before encode, the delivered
/// H.264 stays a valid contiguous reference chain.
///
/// 1. **Setup**: connect to X (a private connection) and require a 32-bpp BGRA root — the Z-pixmap
///    byte depth must be 4, which modern servers use for depth 24/32 — then negotiate XShm and
///    XFixes. XFixes is always negotiated (one round-trip) so the cursor overlay can be toggled on
///    live even when capture started without it. Initial dimensions and origin are resolved from the
///    settings and the live root geometry, and a [`FramePool`] of `POOL_N` = 3 shm surfaces is
///    allocated — the working set (one in-capture, one in-slot, one in-encode) that a one-frame-ahead
///    demand-driven capture needs, which also keeps the memory cost (`3 * W*H*4`, significant at 4K)
///    bounded.
/// 2. **Encode thread**: spawned with a raised scheduling priority; it reports its thread id back
///    through `encode_tid_tx` so the caller can detect a re-entrant stop issued from inside the
///    delivery callback.
/// 3. **Capture loop** (until `stop`): fps is re-read each iteration for live pacing — sleep to the
///    next frame deadline, or `yield_now` when already behind instead of busy-spinning.
///    - **Live region change** (`region_dirty`): re-target the grab origin immediately (an x/y pan
///      needs no surface work); a size change reuses the drain/recreate path below.
///    - **Auto-adjust**: on a root geometry change, drain in-flight frames then recreate the surfaces
///      and bump the generation (which the encode thread turns into a reshape or rebuild). The geometry
///      is re-resolved AFTER the drain, because a slow drain can outlast another geometry change (a
///      fast flap) and recreating at a stale size would make the next `shm_get_image` exceed the root;
///      a fully-reverted flap keeps its surfaces as-is. A stop that races the drain breaks out to
///      teardown.
///    - **Grab**: acquire a pooled surface (blocks until the encoder frees one), then `shm_get_image`
///      the region into it synchronously (`reply()` waits).
///    - **Overlays**: composite the XFixes cursor (drawn at its hotspot-offset origin; live-toggleable)
///      and the watermark onto the BGRA pixels on the CPU.
///    - **Publish**: hand the finished frame to the encode thread (blocks until the slot is free;
///      never drops). A stop observed while waiting in acquire/publish exits the loop.
/// 4. **Teardown**: stop and JOIN the encode thread BEFORE destroying the shm surfaces it may still be
///    reading, preserving the resize-safety guarantee.
///
/// Blocking; intended to run on a dedicated thread. The X connection and shm surfaces live on this
/// thread and the encoder lives on the encode thread — nothing X-related crosses the boundary.
pub fn run_capture<F>(
    settings: RustCaptureSettings,
    controls: Arc<Controls>,
    encode_tid_tx: Sender<thread::ThreadId>,
    on_frame: F,
) -> Result<(), String>
where
    F: FnMut(Vec<EncodedStripe>) + Send + 'static,
{
    let (conn, screen_num) =
        x11rb::connect(None).map_err(|e| format!("X11 connect failed: {e}"))?;
    let root = conn.setup().roots[screen_num].root;
    let root_depth = conn.setup().roots[screen_num].root_depth;

    let bpp = conn
        .setup()
        .pixmap_formats
        .iter()
        .find(|f| f.depth == root_depth)
        .map(|f| f.bits_per_pixel)
        .unwrap_or(32);
    if bpp != 32 {
        return Err(format!(
            "unsupported root depth {root_depth} ({bpp} bpp); only 32-bpp BGRA is supported"
        ));
    }

    conn.shm_query_version()
        .map_err(|e| format!("shm_query_version: {e}"))?
        .reply()
        .map_err(|e| format!("XShm unavailable: {e}"))?;
    conn.xfixes_query_version(5, 0)
        .map_err(|e| format!("xfixes_query_version: {e}"))?
        .reply()
        .map_err(|e| format!("XFixes unavailable: {e}"))?;

    let geo = conn
        .get_geometry(root)
        .map_err(|e| format!("get_geometry: {e}"))?
        .reply()
        .map_err(|e| format!("get_geometry reply: {e}"))?;

    let mut rsettings = settings.clone();
    let (mut cap_w, mut cap_h) = resolve_dims(geo.width, geo.height, &rsettings);
    let mut cap_x = rsettings.capture_x.max(0) as i16;
    let mut cap_y = rsettings.capture_y.max(0) as i16;

    const POOL_N: usize = 3;
    let mut surfaces: Vec<ShmSurface> = Vec::with_capacity(POOL_N);
    for _ in 0..POOL_N {
        surfaces.push(ShmSurface::create(&conn, cap_w, cap_h)?);
    }
    let pool = Arc::new(FramePool::new(POOL_N));

    let mut watermark = X11Watermark::load(&settings.watermark_path);

    let enc_pool = pool.clone();
    let enc_controls = controls.clone();
    let enc_settings = settings.clone();
    let encode_thread = thread::spawn(move || {
        crate::boost_thread_priority(-10);
        let _ = encode_tid_tx.send(thread::current().id());
        let mut on_frame = on_frame;
        encode_loop(&enc_pool, &enc_controls, &enc_settings, &mut on_frame);
    });

    let mut next_frame = Instant::now();
    // Frames between root-geometry polls (auto-adjust); ~0.5s at 60fps.
    const GEOMETRY_POLL_FRAMES: i32 = 30;
    let mut geometry_check = 0i32;
    let mut grab_failures = 0u32;
    // Strip rows of the last accepted frame (changed-content check) and the LCG state
    // behind the anti-phase-lock jitter.
    let mut prev_strips: Vec<u8> = Vec::new();
    let mut jitter_state: u64 = 0x9E37_79B9_7F4A_7C15;

    let result = (|| -> Result<(), String> {
        while !controls.stop.load(Ordering::Relaxed) {
            let fps = (controls.fps_milli.load(Ordering::Relaxed).max(1) as f64) / 1000.0;
            let frame_dur = Duration::from_secs_f64(1.0 / fps.max(1.0));
            let now = Instant::now();
            if now < next_frame {
                std::thread::sleep(next_frame - now);
            } else {
                std::thread::yield_now();
            }
            next_frame += frame_dur;
            let now = Instant::now();
            if next_frame < now {
                next_frame = now;
            }
            if controls.stop.load(Ordering::Relaxed) {
                break;
            }

            if controls.region_dirty.swap(false, Ordering::Acquire) {
                let (nx, ny, nw, nh) = *controls.region.lock().unwrap();
                rsettings.capture_x = nx;
                rsettings.capture_y = ny;
                rsettings.width = nw;
                rsettings.height = nh;
                cap_x = nx.max(0) as i16;
                cap_y = ny.max(0) as i16;
                if let Some(g) = conn.get_geometry(root).ok().and_then(|c| c.reply().ok()) {
                    let (fw, fh) = resolve_dims(g.width, g.height, &rsettings);
                    if fw != cap_w || fh != cap_h {
                        if !pool.drain_for_resize(POOL_N, &controls.stop) {
                            break;
                        }
                        for s in surfaces.iter_mut() {
                            s.destroy(&conn);
                        }
                        surfaces.clear();
                        cap_w = fw;
                        cap_h = fh;
                        for _ in 0..POOL_N {
                            surfaces.push(ShmSurface::create(&conn, cap_w, cap_h)?);
                        }
                        prev_strips.clear();
                        pool.bump_generation();
                    }
                }
            }

            // Root geometry is polled on a cadence, not per frame: the reply costs a
            // round-trip that would otherwise precede every grab, and external size
            // changes are rare (live resizes arrive through region_dirty instead).
            geometry_check -= 1;
            if settings.auto_adjust_screen_capture_size && geometry_check <= 0 {
                geometry_check = GEOMETRY_POLL_FRAMES;
                if let Some(g) = conn.get_geometry(root).ok().and_then(|c| c.reply().ok()) {
                    let (nw, nh) = resolve_dims(g.width, g.height, &rsettings);
                    if nw != cap_w || nh != cap_h {
                        if !pool.drain_for_resize(POOL_N, &controls.stop) {
                            break;
                        }
                        let (fw, fh) = conn
                            .get_geometry(root)
                            .ok()
                            .and_then(|c| c.reply().ok())
                            .map(|g| resolve_dims(g.width, g.height, &rsettings))
                            .unwrap_or((nw, nh));
                        if fw != cap_w || fh != cap_h {
                            for s in surfaces.iter_mut() {
                                s.destroy(&conn);
                            }
                            surfaces.clear();
                            cap_w = fw;
                            cap_h = fh;
                            for _ in 0..POOL_N {
                                surfaces.push(ShmSurface::create(&conn, cap_w, cap_h)?);
                            }
                            prev_strips.clear();
                            pool.bump_generation();
                        }
                    }
                }
            }

            let idx = match pool.acquire(&controls.stop) {
                Some(i) => i,
                None => break,
            };
            let surface = &mut surfaces[idx];
            let grab = match grab_stable_frame(
                &conn, root, surface, &mut prev_strips,
                cap_x, cap_y, cap_w, cap_h,
            ) {
                Ok(g) => {
                    grab_failures = 0;
                    g
                }
                Err(e) => {
                    // Most likely an external root resize between geometry polls made
                    // the grab run past the root edge: return the surface, re-poll
                    // geometry immediately, and only give up if it never recovers.
                    pool.recycle(idx);
                    grab_failures += 1;
                    geometry_check = 0;
                    if grab_failures > 120 {
                        return Err(e);
                    }
                    continue;
                }
            };
            // Lock the pacing to the accepted grab: the next tick lands one period
            // after the instant that (verifiably) captured a complete frame, so
            // capture converges to running just after the content finishes painting
            // — the phase-locked-content case that otherwise produces a standing
            // tear line becomes the best case instead of the worst.
            next_frame = grab.anchor + frame_dur;
            if grab.changed {
                // A zero-mean random phase walk on top of the lock: an alignment the
                // capture timer cannot lock onto (a painter pausing beyond the quiet
                // window) cannot then repeat frame after frame.
                jitter_state = jitter_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let jitter_us = (jitter_state >> 33) % 1500;
                next_frame += Duration::from_micros(jitter_us);
                next_frame = next_frame
                    .checked_sub(Duration::from_micros(750))
                    .unwrap_or(next_frame);
            }

            let frame_w = cap_w as i32;
            let frame_h = cap_h as i32;
            let stride = surface.stride;
            let buf = surface.as_mut_slice();

            if controls.capture_cursor.load(Ordering::Relaxed) {
                if let Some(c) = conn
                    .xfixes_get_cursor_image()
                    .ok()
                    .and_then(|c| c.reply().ok())
                {
                    if c.width > 0 && c.height > 0 {
                        let (img_x, img_y) = cursor_image_origin(
                            c.x,
                            c.y,
                            c.xhot,
                            c.yhot,
                            settings.capture_x,
                            settings.capture_y,
                        );
                        overlay_cursor(
                            buf,
                            stride,
                            frame_w,
                            frame_h,
                            c.width as i32,
                            c.height as i32,
                            &c.cursor_image,
                            img_x,
                            img_y,
                        );
                    }
                }
            }

            if watermark.loaded {
                watermark.update_position(frame_w, frame_h, settings.watermark_location_enum);
                watermark.blend_into(buf, stride, frame_w, frame_h);
            }

            let published = pool.publish(
                RawFrame {
                    idx,
                    ptr: surface.addr,
                    len: surface.size,
                    width: cap_w,
                    height: cap_h,
                    stride,
                    generation: pool.generation(),
                },
                &controls.stop,
            );
            if !published {
                break;
            }
        }
        Ok(())
    })();

    pool.shutdown();
    let _ = encode_thread.join();
    for s in surfaces.iter_mut() {
        s.destroy(&conn);
    }
    result
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    fn dummy(idx: usize) -> RawFrame {
        RawFrame {
            idx,
            ptr: std::ptr::null_mut(),
            len: 0,
            width: 0,
            height: 0,
            stride: 0,
            generation: 0,
        }
    }

    /// @brief A full acquire -> publish -> take -> recycle round-trip returns the same surface, and
    /// afterwards all three pool surfaces are acquirable again and distinct.
    #[test]
    fn roundtrip_then_recycle_returns_all_surfaces() {
        let p = FramePool::new(3);
        let stop = AtomicBool::new(false);
        let a = p.acquire(&stop).unwrap();
        assert!(p.publish(dummy(a), &stop));
        let f = p.take().unwrap();
        assert_eq!(f.idx, a);
        p.recycle(f.idx);
        let (x, y, z) = (
            p.acquire(&stop).unwrap(),
            p.acquire(&stop).unwrap(),
            p.acquire(&stop).unwrap(),
        );
        assert!(x != y && y != z && x != z);
    }

    /// @brief With every surface held and `stop` set, `acquire` returns `None` after its bounded wait
    /// rather than blocking forever.
    #[test]
    fn acquire_returns_none_when_exhausted_and_stopped() {
        let p = FramePool::new(2);
        let stop = AtomicBool::new(false);
        let _a = p.acquire(&stop).unwrap();
        let _b = p.acquire(&stop).unwrap();
        stop.store(true, Ordering::Relaxed);
        assert!(p.acquire(&stop).is_none());
    }

    /// @brief With the handoff slot already occupied and `stop` set, `publish` returns `false` (frame
    /// discarded) instead of blocking.
    #[test]
    fn publish_returns_false_when_slot_full_and_stopped() {
        let p = FramePool::new(3);
        let stop = AtomicBool::new(false);
        let a = p.acquire(&stop).unwrap();
        assert!(p.publish(dummy(a), &stop));
        let b = p.acquire(&stop).unwrap();
        stop.store(true, Ordering::Relaxed);
        assert!(!p.publish(dummy(b), &stop));
    }

    /// @brief `drain_for_resize` blocks until every held surface has been recycled (`free == n`), then
    /// reports a full drain and leaves the pool whole; a helper thread recycles the three held surfaces
    /// shortly after the drain begins.
    #[test]
    fn drain_for_resize_waits_until_all_free() {
        let p = Arc::new(FramePool::new(3));
        let stop = AtomicBool::new(false);
        let held = [
            p.acquire(&stop).unwrap(),
            p.acquire(&stop).unwrap(),
            p.acquire(&stop).unwrap(),
        ];
        let p2 = p.clone();
        let t = thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            for idx in held {
                p2.recycle(idx);
            }
        });
        assert!(p.drain_for_resize(3, &stop));
        t.join().unwrap();
        assert!(p.acquire(&stop).is_some());
    }

    /// @brief A surface is held and never recycled (as a dead encode thread would leave it): setting
    /// `stop` unblocks the bounded drain, which reports it did NOT fully drain instead of hanging.
    #[test]
    fn drain_for_resize_aborts_on_stop() {
        let p = FramePool::new(3);
        let stop = AtomicBool::new(false);
        let _held = p.acquire(&stop).unwrap();
        stop.store(true, Ordering::Relaxed);
        assert!(!p.drain_for_resize(3, &stop));
    }

    /// @brief A W1 -> W2 -> W1 resize flap where the encoder never takes the W2 frame:
    /// `drain_for_resize` reclaims the un-taken frame from the slot, the surfaces recreate twice, and
    /// the next taken frame lands back on the ORIGINAL dimensions — so the bumped generation is the
    /// only rebuild signal that invalidates encoder state keyed to the reused surface addresses.
    #[test]
    fn resize_flap_reclaim_changes_generation_at_identical_dims() {
        let p = FramePool::new(3);
        let stop = AtomicBool::new(false);
        let a = p.acquire(&stop).unwrap();
        assert!(p.publish(
            RawFrame { generation: p.generation(), width: 1920, height: 1080, ..dummy(a) },
            &stop
        ));
        let f = p.take().unwrap();
        let (last_gen, last_dims) = (f.generation, (f.width, f.height));
        p.recycle(f.idx);
        assert!(p.drain_for_resize(3, &stop));
        p.bump_generation();
        let b = p.acquire(&stop).unwrap();
        assert!(p.publish(
            RawFrame { generation: p.generation(), width: 2560, height: 1600, ..dummy(b) },
            &stop
        ));
        assert!(p.drain_for_resize(3, &stop));
        p.bump_generation();
        let c = p.acquire(&stop).unwrap();
        assert!(p.publish(
            RawFrame { generation: p.generation(), width: 1920, height: 1080, ..dummy(c) },
            &stop
        ));
        let g = p.take().unwrap();
        assert_eq!((g.width, g.height), last_dims, "flap lands on identical dimensions");
        assert_eq!(g.generation, 2);
        assert_ne!(g.generation, last_gen, "generation is the only rebuild signal");
    }

    /// @brief Recreating the surfaces bumps the generation, and frames published afterwards carry the
    /// new value — the signal the encode thread uses to trigger a rebuild at identical dimensions.
    #[test]
    fn generation_bumps_on_recreate_and_rides_published_frames() {
        let p = FramePool::new(3);
        let stop = AtomicBool::new(false);
        assert_eq!(p.generation(), 0);
        let a = p.acquire(&stop).unwrap();
        assert!(p.publish(RawFrame { generation: p.generation(), ..dummy(a) }, &stop));
        let f = p.take().unwrap();
        assert_eq!(f.generation, 0);
        p.recycle(f.idx);
        p.bump_generation();
        let b = p.acquire(&stop).unwrap();
        assert!(p.publish(RawFrame { generation: p.generation(), ..dummy(b) }, &stop));
        assert_eq!(p.take().unwrap().generation, 1);
    }
}

#[cfg(test)]
mod verify_tests {
    use super::*;

    /// @brief Strips span the full height (first at the top edge, last flush with the
    /// bottom edge), stay strictly increasing, and collapse to a single strip when the
    /// frame is too short to sample.
    #[test]
    fn strip_rows_span_top_to_bottom_within_bounds() {
        let ys = verify_strip_rows(720, 16, 2);
        assert_eq!(ys.len(), 16);
        assert_eq!(ys[0], 0);
        assert_eq!(*ys.last().unwrap(), 718);
        assert!(ys.windows(2).all(|w| w[1] > w[0]));
        assert_eq!(verify_strip_rows(3, 16, 2), vec![0]);
    }

    /// @brief Identical strips verify as stable; a single changed byte in any sampled
    /// strip (a paint front that advanced past it between the grabs) fails the check.
    #[test]
    fn strips_match_detects_inflight_paint() {
        let (w, h, rows) = (8usize, 64usize, 2usize);
        let stride = w * 4;
        let frame = vec![7u8; stride * h];
        let ys = verify_strip_rows(h, 4, rows);
        let mut verify = Vec::new();
        for &y in &ys {
            verify.extend_from_slice(&frame[y * stride..(y + rows) * stride]);
        }
        assert!(verify_strips_match(&frame, stride, &verify, &ys, rows));
        let idx = rows * stride + 5;
        verify[idx] ^= 0xFF;
        assert!(!verify_strips_match(&frame, stride, &verify, &ys, rows));
    }
}

#[cfg(test)]
mod cursor_tests {
    use super::*;

    /// @brief `cursor_image_origin` subtracts both the hotspot and the capture-region offset from the
    /// reported pointer position, and goes negative when the hotspot sits near the frame origin (the
    /// result is clipped when drawn).
    #[test]
    fn origin_subtracts_hotspot_and_capture_offset() {
        assert_eq!(cursor_image_origin(100, 80, 4, 6, 0, 0), (96, 74));
        assert_eq!(cursor_image_origin(100, 80, 4, 6, 10, 20), (86, 54));
        assert_eq!(cursor_image_origin(1, 1, 8, 8, 0, 0), (-7, -7));
    }

    /// @brief `overlay_cursor` blits at the hotspot-offset origin, not the raw pointer position: a
    /// 2x2 cursor with hotspot (1,1) at pointer (4,4) lands at (3,3)..(4,4), leaving the un-offset
    /// (hotspot) corner untouched.
    #[test]
    fn overlay_blits_at_hotspot_offset_origin() {
        let stride = 8 * 4;
        let mut frame = vec![0u8; stride * 8];
        let pixels = [0xFFFF_FFFFu32; 4];
        let (ox, oy) = cursor_image_origin(4, 4, 1, 1, 0, 0);
        overlay_cursor(&mut frame, stride, 8, 8, 2, 2, &pixels, ox, oy);
        let px = |x: usize, y: usize| frame[y * stride + x * 4];
        assert_eq!(px(3, 3), 255);
        assert_eq!(px(4, 4), 255);
        assert_eq!(px(2, 2), 0);
        assert_eq!(px(5, 5), 0, "no pixel at the un-offset (hotspot) corner");
    }

    /// @brief `overlay_cursor` clips a negative origin at the frame edge: with the origin at (-1,-1)
    /// only the in-frame quadrant is written, landing the cursor's bottom-right pixel at (0,0).
    #[test]
    fn overlay_clips_negative_origin_at_frame_edge() {
        let stride = 4 * 4;
        let mut frame = vec![0u8; stride * 4];
        let pixels = [0xFFFF_FFFFu32; 4];
        let (ox, oy) = cursor_image_origin(0, 0, 1, 1, 0, 0);
        overlay_cursor(&mut frame, stride, 4, 4, 2, 2, &pixels, ox, oy);
        assert_eq!(frame[0], 255);
        assert!(frame[4..].iter().all(|&b| b == 0), "no writes outside (0,0)");
    }
}
