/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! X11 host capture. Grabs the root window into a shared memory segment (XShm via x11rb) as
//! BGRA, composites the XFixes hardware cursor and the watermark on the CPU, and feeds each
//! frame to [`X11Pipeline`] which owns damage/stripe/encode.
//!
//! The whole pipeline runs on one thread (the caller's): the x11rb connection, the shm segment,
//! and the encoder all live for the duration of [`run_capture`]. Multi-instance safety for the
//! encoders is handled inside them (e.g. the libx264 open/close lock); each capture owns its own
//! private xcb connection, so there is no shared X state to serialize here.

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

/// Cross-thread controls for a running capture. The owner (the `ScreenCapture` pyclass) flips
/// these from the Python thread; the capture thread reads them at the top of each iteration and
/// applies them to its `X11Pipeline`. This keeps request_idr / rate / fps changes off the
/// pipeline's thread boundary: the pipeline and its encoder are only ever touched from the
/// capture thread.
pub struct Controls {
    pub stop: AtomicBool,
    pub force_idr: AtomicBool,
    pub rate_dirty: AtomicBool,
    pub bitrate_kbps: AtomicI32,
    /// VBV frame-time multiplier * 1000 (atomic-friendly; <= 0 = policy default).
    pub vbv_mult_milli: AtomicI32,
    /// target fps * 1000 (atomic-friendly; re-read each frame for dynamic pacing + rate control).
    pub fps_milli: AtomicU64,
    /// Pending per-frame tunables for the encode thread (one struct, set rarely).
    pub tunables_dirty: AtomicBool,
    pub tunables: Mutex<Option<crate::LiveTunables>>,
    /// Cursor overlay toggle, read by the capture thread each grab.
    pub capture_cursor: AtomicBool,
    /// Pending capture-region change (x, y, w, h; w/h <= 0 = to the root edge), applied by
    /// the capture thread with the same drain/recreate machinery as auto-adjust.
    pub region_dirty: AtomicBool,
    pub region: Mutex<(i32, i32, i32, i32)>,
}

impl Controls {
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

/// A shared-memory image surface: a POSIX shm segment attached to both this process and the X
/// server, into which `shm_get_image` writes one BGRA frame.
struct ShmSurface {
    shmseg: u32,
    addr: *mut u8,
    size: usize,
    width: u16,
    height: u16,
    stride: usize,
}

impl ShmSurface {
    /// Allocate a shm segment of `width*height*4` bytes, attach it locally and to the X server,
    /// then mark it `IPC_RMID` so the kernel frees it once both ends detach.
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
            // Attach on the server, confirm with a round-trip, THEN mark for deletion so the
            // segment survives until both this process and the server detach.
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

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.addr, self.size) }
    }

    /// Detach from the server (needs the connection) and locally. Call before drop.
    fn destroy(&mut self, conn: &RustConnection) {
        let _ = conn.shm_detach(self.shmseg);
        let _ = conn.flush();
        unsafe {
            libc::shmdt(self.addr as *mut libc::c_void);
        }
        self.addr = std::ptr::null_mut();
    }
}

/// Resolve the capture dimensions from settings + the live root geometry. With auto-adjust (or an
/// unset width/height) the capture tracks the full root; otherwise the requested size is clamped to
/// what's available from the capture offset. H.264 needs even dimensions.
fn resolve_dims(root_w: u16, root_h: u16, s: &RustCaptureSettings) -> (u16, u16) {
    // capture_x/y are clamped to >=0 the same way shm_get_image consumes them; saturating math
    // plus a final u16 clamp keep pathological settings from overflowing or truncating.
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

/// Alpha-blend a source pixel (already split into r,g,b,a) over a BGRA destination pixel:
/// opaque overwrites, partial alpha blends, fully transparent skips.
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

/// Frame-space top-left for the cursor image. XFixes reports the cursor position at its
/// HOTSPOT; X draws the image with its top-left at (pos - hot). May go negative near the
/// frame edges -- `overlay_cursor` clips per pixel, matching the server's own edge clipping.
#[inline]
fn cursor_image_origin(x: i16, y: i16, xhot: u16, yhot: u16, cap_x: i32, cap_y: i32) -> (i32, i32) {
    (x as i32 - xhot as i32 - cap_x, y as i32 - yhot as i32 - cap_y)
}

/// Composite the XFixes cursor (ARGB `u32` per pixel) onto the BGRA frame at `(img_x,img_y)`
/// (top-left), with per-pixel bounds clipping.
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

/// CPU watermark: holds the raw RGBA pixels in host memory plus the placement/animation state,
/// for blending directly into the captured BGRA frame on the CPU.
struct X11Watermark {
    pixels: Vec<u8>, // RGBA, row-major, w*h*4
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

    /// Update the watermark's top-left placement for this frame from the location setting;
    /// for the animated mode, advance the bouncing position and reflect off the frame edges.
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

/// A captured raw BGRA frame held in a pooled shm surface, ready to encode. Carries the surface
/// pointer + geometry so the encode thread reads it directly (no copy) and can rebuild its pipeline
/// if the capture size changed (auto-adjust), plus the pool's surface generation so a rebuild also
/// happens when the surfaces were recreated at the SAME size.
struct RawFrame {
    idx: usize,
    ptr: *mut u8,
    len: usize,
    width: u16,
    height: u16,
    stride: usize,
    generation: u64,
}
// The pointer addresses a pooled shm surface the pool guarantees is not reused until the encode
// thread recycles this frame, so the handle is safe to move across the thread boundary.
unsafe impl Send for RawFrame {}

struct PoolInner {
    free: Vec<usize>,
    slot: Option<RawFrame>,
}

/// Demand-driven capture->encode handoff. The capture thread writes into a pooled surface and
/// `publish`es it into a single slot; the encode thread `take`s it. Capture stays at most one frame
/// ahead of encode: `acquire`/`publish` BLOCK (bounded) until the encoder frees a surface / drains
/// the slot, so capture is throttled to the encode rate. Because X11 capture is pull-based (no
/// backlog), this throttling -- rather than capturing-then-dropping -- means capture never wastes a
/// full-resolution shm round-trip on a frame that would be discarded, while still overlapping the
/// next capture with the current encode (the throughput win). The encoder only ever sees a
/// contiguous frame stream, so the H.264 reference chain stays valid.
struct FramePool {
    inner: Mutex<PoolInner>,
    cv: Condvar,
    stop: AtomicBool,
    /// Surface generation: bumped by the capture thread each time the shm surfaces backing the
    /// pool are destroyed and recreated. Published frames carry it so the encode thread rebuilds
    /// its pipeline -- dropping encoder state keyed to surface base pointers (e.g. NVENC's
    /// pinned-host registrations) -- even when a resize flap lands back on the old dimensions and
    /// the recreated segments reuse the old virtual addresses.
    generation: AtomicU64,
}

impl FramePool {
    fn new(n: usize) -> Self {
        Self {
            inner: Mutex::new(PoolInner { free: (0..n).collect(), slot: None }),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
            generation: AtomicU64::new(0),
        }
    }

    /// Capture: record that the surfaces were recreated. Only called after `drain_for_resize`
    /// succeeded, so no frame from the previous generation is still in flight.
    fn bump_generation(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Current surface generation, stamped onto each published frame.
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Capture: get a free surface to write the next frame into, BLOCKING (bounded, re-checking
    /// `stop` every 20ms) until one is free. This throttles capture to the encode rate so a
    /// full-resolution capture is never spent on a frame that would be dropped. Returns None on stop.
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

    /// Capture: publish the just-captured frame into the single slot, BLOCKING (bounded) until the
    /// encode thread has taken the previous one -- capture stays at most one frame ahead, never
    /// dropping. Returns false (frame discarded) on stop.
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

    /// Encode: block until a frame is available (Some) or stop is signalled (None). The wait is
    /// bounded (re-checking `stop` every 20ms) as defense-in-depth against a lost wakeup, so a
    /// stop that races the park can never leave this thread blocked forever.
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

    /// Encode: return a surface to the free list after it has been encoded.
    fn recycle(&self, idx: usize) {
        self.inner.lock().unwrap().free.push(idx);
        self.cv.notify_all();
    }

    /// Capture: before recreating surfaces (auto-adjust resize), reclaim the pending slot and wait
    /// until every surface is back in the free list (the encode thread finished any in-flight
    /// frame), so no surface is destroyed while the encode thread is reading it. The wait is bounded
    /// and also breaks on stop (pool shutdown OR the external stop) -- with panic=abort gone, a
    /// panicked/dead encode thread that never recycles must not wedge the capture thread here
    /// forever; a requested stop unblocks it. Returns true if it fully drained (safe to recreate
    /// surfaces), false if it aborted on stop (the caller then tears down, which joins the encode
    /// thread before destroying surfaces, so the resize-safety guarantee still holds).
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

    fn shutdown(&self) {
        // Acquire the inner mutex BEFORE storing stop + notifying: this closes the lost-wakeup
        // window where take() has checked stop==false but not yet parked -- holding the lock
        // means take() is either still before its stop-check or already parked (and will get the
        // notify). Notify after dropping the guard so the woken thread doesn't immediately block
        // on the lock we hold.
        let g = self.inner.lock().unwrap();
        self.stop.store(true, Ordering::Release);
        drop(g);
        self.cv.notify_all();
    }
}

/// Encode thread body: pull the freshest captured frame, (re)build the pipeline if the size or
/// surface generation changed, apply cross-thread controls, encode, recycle the surface, then
/// deliver. Recycling before delivery means a slow consumer never holds a capture surface.
fn encode_loop<F>(pool: &FramePool, controls: &Controls, settings: &RustCaptureSettings, on_frame: &mut F)
where
    F: FnMut(Vec<EncodedStripe>),
{
    let mut psettings = settings.clone();
    // Optional Unix-socket H.264 fan-out (parity with the Wayland path). Bound ONCE per capture
    // and owned here, outside the pipeline: rebuilds on auto-adjust resizes must keep the socket
    // listener and any attached recorders alive. Full-frame H.264 only; warn on configurations
    // that can't produce a single recordable stream.
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

    while let Some(frame) = pool.take() {
        let (fw, fh) = (frame.width as i32, frame.height as i32);
        // Adapt on a size change OR a surface-generation change: recreated shm segments often
        // reuse the old virtual base addresses, so encoder state keyed to base pointers (NVENC's
        // pinned-host cache) must be dropped even when the dimensions are identical. Prefer an
        // in-place reshape (NVENC reconfigures its live session; the striped software path just
        // re-derives stripe state) so a resize never stalls on a full encoder re-init; encoders
        // that cannot follow (VAAPI/OpenH264) rebuild as before.
        if pipeline.is_none() || pw != fw || ph != fh || pgen != frame.generation {
            let size_changed = pw != fw || ph != fh;
            psettings.width = fw;
            psettings.height = fh;
            // The controls atomics always hold the CURRENT rate values (the update_* setters
            // store them there), so a rebuild carries live bitrate/VBV/fps changes forward
            // instead of reverting to the capture-start settings.
            psettings.video_bitrate_kbps = controls.bitrate_kbps.load(Ordering::Relaxed);
            psettings.video_vbv_multiplier =
                controls.vbv_mult_milli.load(Ordering::Relaxed) as f64 / 1000.0;
            psettings.target_fps =
                (controls.fps_milli.load(Ordering::Relaxed).max(1) as f64) / 1000.0;
            let reshaped = pipeline
                .as_mut()
                .is_some_and(|pl| pl.reshape(&psettings, size_changed));
            if !reshaped {
                // Drop the old pipeline (and its NVENC/VAAPI session + GPU surfaces) BEFORE
                // building the new one, so an auto-adjust resize never holds two full encoder
                // allocations at once (transient 2x GPU memory).
                drop(pipeline.take());
                pipeline = Some(X11Pipeline::new(psettings.clone(), recording_sink.clone()));
            }
            pw = fw;
            ph = fh;
            pgen = frame.generation;
        }
        let pl = pipeline.as_mut().unwrap();

        // Cross-thread controls are applied here, on the thread that owns the pipeline. The
        // Acquire on the rate_dirty swap pairs with the Release store on the update_* setters, so
        // the payload (bitrate/vbv/fps) is fully visible -- never seen half-applied.
        if controls.force_idr.swap(false, Ordering::Relaxed) {
            pl.request_idr();
        }
        if controls.rate_dirty.swap(false, Ordering::Acquire) {
            let b = controls.bitrate_kbps.load(Ordering::Relaxed);
            let v = controls.vbv_mult_milli.load(Ordering::Relaxed) as f64 / 1000.0;
            let fps = (controls.fps_milli.load(Ordering::Relaxed).max(1) as f64) / 1000.0;
            pl.update_rate(b, v, fps);
        }
        // Live tunables land in both settings copies; encoders re-read them per frame.
        if controls.tunables_dirty.swap(false, Ordering::Acquire) {
            if let Some(t) = controls.tunables.lock().unwrap().take() {
                t.apply_to(&mut psettings);
                pl.update_tunables(&t);
            }
        }

        // SAFETY: the pool guarantees this surface is not reused until we recycle it below.
        let buf = unsafe { std::slice::from_raw_parts(frame.ptr, frame.len) };
        let stripes = pl.process(buf, frame.stride);
        pool.recycle(frame.idx);
        if !stripes.is_empty() {
            on_frame(stripes);
        }
    }
}

/// Run the X11 capture pipeline until `stop` is set. Capture (this thread) grabs frames into a pool
/// of shm surfaces and hands the freshest to an internal encode+deliver thread; `on_frame(stripes)`
/// runs on that encode thread once per encoded frame. Splitting capture from encode lets the two
/// overlap (throughput); dropping happens on raw frames before encode, so the delivered H.264 stays
/// a valid contiguous reference chain. `encode_tid_tx` reports the encode thread's id (for the
/// caller's re-entrant-stop handling).
///
/// Blocking; intended to run on a dedicated thread. The X connection + shm surfaces live on this
/// thread; the encoder lives on the encode thread; nothing X-related crosses threads.
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

    // The Z-pixmap byte depth must be 4 (BGRA); modern servers use 32bpp for depth 24/32.
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
    // Always negotiate XFixes (one roundtrip) so the cursor overlay can be toggled on
    // live even when the capture started without it.
    conn.xfixes_query_version(5, 0)
        .map_err(|e| format!("xfixes_query_version: {e}"))?
        .reply()
        .map_err(|e| format!("XFixes unavailable: {e}"))?;

    let geo = conn
        .get_geometry(root)
        .map_err(|e| format!("get_geometry: {e}"))?
        .reply()
        .map_err(|e| format!("get_geometry reply: {e}"))?;

    // Geometry-resolution settings, mutable so live capture-region changes re-target the
    // grab (the encode side keys off the frame dimensions and follows by itself).
    let mut rsettings = settings.clone();
    let (mut cap_w, mut cap_h) = resolve_dims(geo.width, geo.height, &rsettings);
    let mut cap_x = rsettings.capture_x.max(0) as i16;
    let mut cap_y = rsettings.capture_y.max(0) as i16;

    // Pool of shm surfaces. 3 is the working set (one in-capture, one in-slot, one in-encode); a
    // demand-driven capture stays one frame ahead and blocks beyond that, so 3 suffices and keeps
    // the memory cost (3 * W*H*4) bounded -- it matters at 4K.
    const POOL_N: usize = 3;
    let mut surfaces: Vec<ShmSurface> = Vec::with_capacity(POOL_N);
    for _ in 0..POOL_N {
        surfaces.push(ShmSurface::create(&conn, cap_w, cap_h)?);
    }
    let pool = Arc::new(FramePool::new(POOL_N));

    let mut watermark = X11Watermark::load(&settings.watermark_path);

    // Encode + deliver thread: owns the pipeline + the Python callback, consumes raw frames from the
    // pool. It reports its thread id so the caller can detect a re-entrant stop from the callback.
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

    let result = (|| -> Result<(), String> {
        while !controls.stop.load(Ordering::Relaxed) {
            // Dynamic pacing: re-read fps each iteration so update_framerate takes effect live.
            let fps = (controls.fps_milli.load(Ordering::Relaxed).max(1) as f64) / 1000.0;
            let frame_dur = Duration::from_secs_f64(1.0 / fps.max(1.0));
            // Frame pacing: sleep until the next deadline; if already behind, yield so a
            // concurrent stop / other work can run instead of busy-spinning.
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

            // Live region change: re-target the grab origin immediately (an x/y pan needs no
            // surface work at all); a size change reuses the auto-adjust drain/recreate below
            // by re-resolving against the new region settings.
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
                        pool.bump_generation();
                    }
                }
            }

            // Auto-adjust: on a geometry change, drain in-flight frames then recreate the surfaces
            // (the encode thread reshapes or rebuilds its pipeline when it sees the new size or
            // generation -- the generation covers a flap back to the old size before the encoder
            // saw a frame).
            if settings.auto_adjust_screen_capture_size {
                if let Some(g) = conn.get_geometry(root).ok().and_then(|c| c.reply().ok()) {
                    let (nw, nh) = resolve_dims(g.width, g.height, &rsettings);
                    if nw != cap_w || nh != cap_h {
                        // If a stop races the drain, skip the recreate and fall through to teardown
                        // (which joins the encode thread before destroying surfaces).
                        if !pool.drain_for_resize(POOL_N, &controls.stop) {
                            break;
                        }
                        // The drain can outlast ANOTHER geometry change (a fast flap), so
                        // re-resolve against the current root before recreating: surfaces at a
                        // stale size would make the next shm_get_image exceed the root. If the
                        // flap fully reverted, the surfaces are still right -- keep them (and
                        // their generation: nothing the encoder holds went stale).
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
                            pool.bump_generation();
                        }
                    }
                }
            }

            // Acquire a pooled surface (blocks until the encoder frees one) and grab the region into
            // it (synchronous: reply() waits). None means stop was observed while waiting.
            let idx = match pool.acquire(&controls.stop) {
                Some(i) => i,
                None => break,
            };
            let surface = &mut surfaces[idx];
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

            let frame_w = cap_w as i32;
            let frame_h = cap_h as i32;
            let stride = surface.stride;
            let buf = surface.as_mut_slice();

            // Cursor overlay (XFixes reports the hotspot position; draw at pos - hot, offset
            // by the capture origin). Live-toggleable.
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

            // Watermark overlay.
            if watermark.loaded {
                watermark.update_position(frame_w, frame_h, settings.watermark_location_enum);
                watermark.blend_into(buf, stride, frame_w, frame_h);
            }

            // Hand the finished raw frame to the encode thread (blocks until the slot is free; never
            // drops). A false return means stop was observed while waiting -- exit the loop.
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

    // Teardown: stop + join the encode thread BEFORE destroying surfaces it may still be reading.
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

    #[test]
    fn roundtrip_then_recycle_returns_all_surfaces() {
        let p = FramePool::new(3);
        let stop = AtomicBool::new(false);
        let a = p.acquire(&stop).unwrap();
        assert!(p.publish(dummy(a), &stop));
        let f = p.take().unwrap();
        assert_eq!(f.idx, a);
        p.recycle(f.idx);
        // All three surfaces are acquirable again and distinct.
        let (x, y, z) = (
            p.acquire(&stop).unwrap(),
            p.acquire(&stop).unwrap(),
            p.acquire(&stop).unwrap(),
        );
        assert!(x != y && y != z && x != z);
    }

    #[test]
    fn acquire_returns_none_when_exhausted_and_stopped() {
        let p = FramePool::new(2);
        let stop = AtomicBool::new(false);
        let _a = p.acquire(&stop).unwrap();
        let _b = p.acquire(&stop).unwrap();
        stop.store(true, Ordering::Relaxed); // no free left + stop -> None (after the bounded wait)
        assert!(p.acquire(&stop).is_none());
    }

    #[test]
    fn publish_returns_false_when_slot_full_and_stopped() {
        let p = FramePool::new(3);
        let stop = AtomicBool::new(false);
        let a = p.acquire(&stop).unwrap();
        assert!(p.publish(dummy(a), &stop)); // slot now occupied
        let b = p.acquire(&stop).unwrap();
        stop.store(true, Ordering::Relaxed);
        assert!(!p.publish(dummy(b), &stop)); // slot full + stop -> false (frame discarded)
    }

    #[test]
    fn drain_for_resize_waits_until_all_free() {
        let p = Arc::new(FramePool::new(3));
        let stop = AtomicBool::new(false);
        let held = [
            p.acquire(&stop).unwrap(),
            p.acquire(&stop).unwrap(),
            p.acquire(&stop).unwrap(),
        ];
        // Another thread recycles everything shortly; drain must block until free == 3.
        let p2 = p.clone();
        let t = thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            for idx in held {
                p2.recycle(idx);
            }
        });
        assert!(p.drain_for_resize(3, &stop)); // fully drained
        t.join().unwrap();
        // Pool is whole again.
        assert!(p.acquire(&stop).is_some());
    }

    #[test]
    fn drain_for_resize_aborts_on_stop() {
        // A surface is held (never recycled, as a dead encode thread would leave it); drain must not
        // block forever -- setting stop unblocks it and it reports it did NOT fully drain.
        let p = FramePool::new(3);
        let stop = AtomicBool::new(false);
        let _held = p.acquire(&stop).unwrap();
        stop.store(true, Ordering::Relaxed);
        assert!(!p.drain_for_resize(3, &stop)); // aborted on stop (bounded, no hang)
    }

    #[test]
    fn resize_flap_reclaim_changes_generation_at_identical_dims() {
        // W1->W2->W1 flap where the encoder never takes the W2 frame: drain reclaims it
        // from the slot, the surfaces recreate twice, and the next taken frame lands back
        // on the ORIGINAL dimensions -- the generation change is then the only rebuild
        // signal (it invalidates encoder state keyed to the reused surface addresses).
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
        // Capture sees W2: drain, recreate (gen 1), publish a W2 frame.
        assert!(p.drain_for_resize(3, &stop));
        p.bump_generation();
        let b = p.acquire(&stop).unwrap();
        assert!(p.publish(
            RawFrame { generation: p.generation(), width: 2560, height: 1600, ..dummy(b) },
            &stop
        ));
        // W1 returns before the encoder takes: drain reclaims the W2 frame, recreate (gen 2).
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
        // Surfaces recreated (same dims): frames published afterwards must carry a NEW
        // generation -- the encode thread's rebuild trigger.
        p.bump_generation();
        let b = p.acquire(&stop).unwrap();
        assert!(p.publish(RawFrame { generation: p.generation(), ..dummy(b) }, &stop));
        assert_eq!(p.take().unwrap().generation, 1);
    }
}

#[cfg(test)]
mod cursor_tests {
    use super::*;

    #[test]
    fn origin_subtracts_hotspot_and_capture_offset() {
        // Pointer at (100,80), hotspot (4,6): the image top-left is (96,74).
        assert_eq!(cursor_image_origin(100, 80, 4, 6, 0, 0), (96, 74));
        // Capture region offset shifts it further.
        assert_eq!(cursor_image_origin(100, 80, 4, 6, 10, 20), (86, 54));
        // Hotspot near the frame origin pushes the top-left negative (clipped when drawn).
        assert_eq!(cursor_image_origin(1, 1, 8, 8, 0, 0), (-7, -7));
    }

    #[test]
    fn overlay_blits_at_hotspot_offset_origin() {
        // 8x8 BGRA frame; 2x2 opaque white cursor, hotspot (1,1), pointer at (4,4):
        // pixels must land at (3,3)..(4,4), not at the hotspot position (4,4)..(5,5).
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

    #[test]
    fn overlay_clips_negative_origin_at_frame_edge() {
        // Hotspot at the frame corner: origin (-1,-1); only the in-frame quadrant lands.
        let stride = 4 * 4;
        let mut frame = vec![0u8; stride * 4];
        let pixels = [0xFFFF_FFFFu32; 4];
        let (ox, oy) = cursor_image_origin(0, 0, 1, 1, 0, 0);
        overlay_cursor(&mut frame, stride, 4, 4, 2, 2, &pixels, ox, oy);
        assert_eq!(frame[0], 255); // (0,0) holds the cursor's bottom-right pixel
        assert!(frame[4..].iter().all(|&b| b == 0), "no writes outside (0,0)");
    }
}
