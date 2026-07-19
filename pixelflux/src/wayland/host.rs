//! Host-capture mode: pixelflux as a CLIENT of an external Wayland compositor
//! (e.g. labwc running `WLR_BACKENDS=headless`), inverting the nested topology.
//!
//! The host compositor owns the session — one seat, one selection, one screen
//! model — and pixelflux captures and injects as a privileged-protocol client:
//! frames via `zwlr_screencopy_v1` (v3 `copy_with_damage`, so idle screens cost
//! nothing), keyboard via a persistent `zwp_virtual_keyboard_v1` device carrying
//! selkies' own keymap text, pointer via `zwlr_virtual_pointer_v1`. Zero-copy is
//! preserved by allocating the capture buffers from pixelflux's OWN GBM device
//! (render node — no privileges): the compositor blits straight into the dmabufs
//! the encoder imports, so no CPU ever touches a frame. Without a GPU the same
//! loop degrades to wl_shm buffers feeding the CPU encode pool.
//!
//! One [`HostSession`] per backend: the capture loop runs on its own thread and
//! hands ready buffers to the calloop tick over a channel (slots are returned
//! after encode, bounding frames in flight); the input proxies are called from
//! the calloop thread directly (wayland-client proxies are thread-safe).

use std::fs::File;
use std::os::fd::AsFd;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};

use gbm::{BufferObjectFlags, Device as GbmDevice, Format as GbmFormat};
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::Buffer as _;
use smithay::utils::{Physical, Rectangle};
use wayland_client::protocol::{wl_output, wl_pointer, wl_registry, wl_seat, wl_shm, wl_shm_pool};
use wayland_client::{delegate_noop, Connection, Dispatch, Proxy, QueueHandle, WEnum};
use wayland_protocols::wp::linux_dmabuf::zv1::client::{
    zwp_linux_buffer_params_v1::{self, ZwpLinuxBufferParamsV1},
    zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1,
};
use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::{
    zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1,
    zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
};
use wayland_protocols_wlr::screencopy::v1::client::{
    zwlr_screencopy_frame_v1::{self, ZwlrScreencopyFrameV1},
    zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1,
};
use wayland_protocols_wlr::output_management::v1::client::{
    zwlr_output_configuration_head_v1::ZwlrOutputConfigurationHeadV1,
    zwlr_output_configuration_v1::{self, ZwlrOutputConfigurationV1},
    zwlr_output_head_v1::{self, ZwlrOutputHeadV1},
    zwlr_output_manager_v1::{self, ZwlrOutputManagerV1},
    zwlr_output_mode_v1::ZwlrOutputModeV1,
};
use wayland_protocols_wlr::virtual_pointer::v1::client::{
    zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1,
    zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1,
};

use crate::wayland::wlclient::{
    bounded_roundtrip, impl_sync_callback, memfd_with, socket_path, SyncState,
};

const KEYMAP_FORMAT_XKB_V1: u32 = 1;
/// Frames in flight between the capture thread and the encode tick. Two slots:
/// one being blitted by the compositor while the previous one is encoded.
const SLOTS: usize = 2;

/// A frame the compositor finished blitting, ready for the encoder.
pub struct HostFrame {
    slot: usize,
    /// GPU path: the filled dmabuf (Arc-backed; the buffer itself stays owned by
    /// the capture thread and is reused once the slot is released).
    pub dmabuf: Option<Dmabuf>,
    /// Software path: the shm mapping itself — the consumer converts straight
    /// out of it (one copy total); the slot is not reused until release.
    pub cpu: Option<HostCpuFrame>,
    pub width: i32,
    pub height: i32,
    pub damage: Vec<Rectangle<i32, Physical>>,
}

/// Borrow-by-Arc view of a software frame still sitting in its shm mapping.
pub struct HostCpuFrame {
    map: Arc<memmap2::MmapMut>,
    stride: usize,
    format: u32,
}

impl HostCpuFrame {
    /// Convert the frame into tight BGRA rows in `dst` (sized `w*4*h`); the
    /// announced shm format decides the per-pixel conversion (compositors pick
    /// their renderer's preferred read format, e.g. 24-bit BGR on NVIDIA GLES).
    pub fn write_bgra(&self, w: i32, h: i32, dst: &mut [u8]) {
        let row = (w * 4) as usize;
        let (src_bpp, swap_rb) = match self.format {
            f if f == wl_shm::Format::Xbgr8888 as u32
                || f == wl_shm::Format::Abgr8888 as u32 => (4usize, true),
            f if f == wl_shm::Format::Bgr888 as u32 => (3, false),
            _ => (4, false), // xrgb/argb: already BGRA byte order
        };
        for y in 0..h as usize {
            let src_start = y * self.stride;
            let src_end = (src_start + src_bpp * w as usize).min(self.map.len());
            if src_start >= src_end || (y + 1) * row > dst.len() {
                break;
            }
            let src_row = &self.map[src_start..src_end];
            let dst_row = &mut dst[y * row..(y + 1) * row];
            match (src_bpp, swap_rb) {
                (4, false) => {
                    let n = row.min(src_row.len());
                    dst_row[..n].copy_from_slice(&src_row[..n]);
                }
                (4, true) => {
                    for (d, s) in dst_row.chunks_exact_mut(4).zip(src_row.chunks_exact(4)) {
                        d[0] = s[2];
                        d[1] = s[1];
                        d[2] = s[0];
                        d[3] = s[3];
                    }
                }
                _ => {
                    for (d, s) in dst_row.chunks_exact_mut(4).zip(src_row.chunks_exact(3)) {
                        d[0] = s[0];
                        d[1] = s[1];
                        d[2] = s[2];
                        d[3] = 0xff;
                    }
                }
            }
        }
    }
}

enum ToHost {
    Start { width: i32, height: i32 },
    Release(usize),
    Stop,
}

struct HostState {
    shm: Option<wl_shm::WlShm>,
    dmabuf: Option<ZwpLinuxDmabufV1>,
    screencopy: Option<ZwlrScreencopyManagerV1>,
    seat: Option<wl_seat::WlSeat>,
    vk_mgr: Option<ZwpVirtualKeyboardManagerV1>,
    vptr_mgr: Option<ZwlrVirtualPointerManagerV1>,
    outputs: Vec<wl_output::WlOutput>,
    output_mgr: Option<ZwlrOutputManagerV1>,
    heads: Vec<ZwlrOutputHeadV1>,
    om_serial: Option<u32>,
    /// Outcome of the pending output configuration: Some(true) applied,
    /// Some(false) failed, retried internally on `cancelled`.
    cfg_result: Option<bool>,
    cfg_cancelled: bool,
    // Per-frame capture negotiation/results.
    announce_dmabuf: Option<(u32, i32, i32)>, // fourcc, w, h
    announce_shm: Option<(u32, i32, i32, i32)>, // format, w, h, stride
    buffer_done: bool,
    damage: Vec<Rectangle<i32, Physical>>,
    ready: bool,
    failed: bool,
    sync_done: bool,
}

impl Default for HostState {
    fn default() -> Self {
        Self {
            shm: None,
            dmabuf: None,
            screencopy: None,
            seat: None,
            vk_mgr: None,
            vptr_mgr: None,
            outputs: Vec::new(),
            output_mgr: None,
            heads: Vec::new(),
            om_serial: None,
            cfg_result: None,
            cfg_cancelled: false,
            announce_dmabuf: None,
            announce_shm: None,
            buffer_done: false,
            damage: Vec::new(),
            ready: false,
            failed: false,
            sync_done: false,
        }
    }
}

impl SyncState for HostState {
    fn sync_done_mut(&mut self) -> &mut bool {
        &mut self.sync_done
    }
}
impl_sync_callback!(HostState);

impl Dispatch<wl_registry::WlRegistry, ()> for HostState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { name, interface, version } = event {
            match interface.as_str() {
                "wl_shm" => state.shm = Some(registry.bind(name, 1, qh, ())),
                "zwp_linux_dmabuf_v1" if version >= 3 => {
                    state.dmabuf = Some(registry.bind(name, 3, qh, ()))
                }
                "zwlr_screencopy_manager_v1" if version >= 3 => {
                    state.screencopy = Some(registry.bind(name, 3, qh, ()))
                }
                "wl_seat" if state.seat.is_none() => {
                    state.seat = Some(registry.bind(name, 1, qh, ()))
                }
                "zwp_virtual_keyboard_manager_v1" => {
                    state.vk_mgr = Some(registry.bind(name, 1, qh, ()))
                }
                "zwlr_virtual_pointer_manager_v1" => {
                    state.vptr_mgr = Some(registry.bind(name, version.min(2), qh, ()))
                }
                "wl_output" => state.outputs.push(registry.bind(name, 1, qh, ())),
                "zwlr_output_manager_v1" => {
                    state.output_mgr = Some(registry.bind(name, 1, qh, ()))
                }
                _ => {}
            }
        }
    }
}

delegate_noop!(HostState: ignore wl_seat::WlSeat);
delegate_noop!(HostState: ignore wl_output::WlOutput);
delegate_noop!(HostState: ignore wl_shm::WlShm);
delegate_noop!(HostState: ignore wl_shm_pool::WlShmPool);
delegate_noop!(HostState: ignore wayland_client::protocol::wl_buffer::WlBuffer);
delegate_noop!(HostState: ZwpVirtualKeyboardManagerV1);
delegate_noop!(HostState: ZwpVirtualKeyboardV1);
delegate_noop!(HostState: ZwlrVirtualPointerManagerV1);
delegate_noop!(HostState: ZwlrVirtualPointerV1);
delegate_noop!(HostState: ZwlrScreencopyManagerV1);
delegate_noop!(HostState: ignore ZwlrOutputModeV1);
delegate_noop!(HostState: ZwlrOutputConfigurationHeadV1);

impl Dispatch<ZwlrOutputManagerV1, ()> for HostState {
    fn event(
        state: &mut Self,
        _: &ZwlrOutputManagerV1,
        event: zwlr_output_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_output_manager_v1::Event::Head { head } => state.heads.push(head),
            zwlr_output_manager_v1::Event::Done { serial } => state.om_serial = Some(serial),
            _ => {}
        }
    }

    wayland_client::event_created_child!(HostState, ZwlrOutputManagerV1, [
        zwlr_output_manager_v1::EVT_HEAD_OPCODE => (ZwlrOutputHeadV1, ()),
    ]);
}

impl Dispatch<ZwlrOutputHeadV1, ()> for HostState {
    fn event(
        state: &mut Self,
        head: &ZwlrOutputHeadV1,
        event: zwlr_output_head_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwlr_output_head_v1::Event::Finished = event {
            state.heads.retain(|h| h.id() != head.id());
        }
    }

    wayland_client::event_created_child!(HostState, ZwlrOutputHeadV1, [
        zwlr_output_head_v1::EVT_MODE_OPCODE => (ZwlrOutputModeV1, ()),
    ]);
}

impl Dispatch<ZwlrOutputConfigurationV1, ()> for HostState {
    fn event(
        state: &mut Self,
        _: &ZwlrOutputConfigurationV1,
        event: zwlr_output_configuration_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_output_configuration_v1::Event::Succeeded => state.cfg_result = Some(true),
            zwlr_output_configuration_v1::Event::Failed => state.cfg_result = Some(false),
            zwlr_output_configuration_v1::Event::Cancelled => state.cfg_cancelled = true,
            _ => {}
        }
    }
}

impl Dispatch<ZwpLinuxDmabufV1, ()> for HostState {
    fn event(
        _: &mut Self,
        _: &ZwpLinuxDmabufV1,
        _: <ZwpLinuxDmabufV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // format/modifier advertisements: the allocation follows the screencopy
        // frame's announcement instead.
    }
}

impl Dispatch<ZwpLinuxBufferParamsV1, ()> for HostState {
    fn event(
        _: &mut Self,
        _: &ZwpLinuxBufferParamsV1,
        _: zwp_linux_buffer_params_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // created/failed are only sent for non-immed creation.
    }
}

impl Dispatch<ZwlrScreencopyFrameV1, ()> for HostState {
    fn event(
        state: &mut Self,
        _: &ZwlrScreencopyFrameV1,
        event: zwlr_screencopy_frame_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_screencopy_frame_v1::Event::Buffer { format, width, height, stride } => {
                if let WEnum::Value(f) = format {
                    state.announce_shm = Some((f as u32, width as i32, height as i32, stride as i32));
                }
            }
            zwlr_screencopy_frame_v1::Event::LinuxDmabuf { format, width, height } => {
                state.announce_dmabuf = Some((format, width as i32, height as i32));
            }
            zwlr_screencopy_frame_v1::Event::BufferDone => state.buffer_done = true,
            zwlr_screencopy_frame_v1::Event::Damage { x, y, width, height } => {
                state.damage.push(Rectangle::new(
                    (x as i32, y as i32).into(),
                    (width as i32, height as i32).into(),
                ));
            }
            zwlr_screencopy_frame_v1::Event::Ready { .. } => state.ready = true,
            zwlr_screencopy_frame_v1::Event::Failed => state.failed = true,
            _ => {}
        }
    }
}

enum SlotBuffer {
    Gpu {
        _bo: gbm::BufferObject<()>,
        dmabuf: Dmabuf,
        wl: wayland_client::protocol::wl_buffer::WlBuffer,
    },
    Cpu {
        _pool: wl_shm_pool::WlShmPool,
        map: Arc<memmap2::MmapMut>,
        stride: i32,
        format: u32,
        wl: wayland_client::protocol::wl_buffer::WlBuffer,
    },
}

/// The calloop-side handle: input proxies plus the frame channel.
pub struct HostSession {
    conn: Connection,
    vk: Option<ZwpVirtualKeyboardV1>,
    vptr: Option<ZwlrVirtualPointerV1>,
    to_thread: Sender<ToHost>,
    frames: Receiver<HostFrame>,
    /// Newest frame, kept (slot and all) until replaced so an IDR request on a
    /// static screen can re-encode current content like compositor mode does.
    retained: Mutex<Option<HostFrame>>,
    pub display_size: Arc<Mutex<(i32, i32)>>,
    alive: Arc<AtomicBool>,
}

impl HostSession {
    /// Connect to `display` and bring up input devices + the capture thread
    /// (idle until [`start_capture`]). `gbm` enables the zero-copy path.
    pub fn connect(display: &str, gbm: Option<GbmDevice<File>>) -> Result<Self, String> {
        let path = socket_path(display).ok_or("XDG_RUNTIME_DIR is unset")?;
        let stream = UnixStream::connect(&path).map_err(|e| format!("connect {path}: {e}"))?;
        let conn = Connection::from_socket(stream).map_err(|e| format!("wayland setup: {e}"))?;
        let mut queue = conn.new_event_queue();
        let qh = queue.handle();
        let _registry = conn.display().get_registry(&qh, ());
        let mut state = HostState::default();
        bounded_roundtrip(&conn, &mut queue, &mut state)?;

        let seat = state.seat.clone().ok_or("host compositor advertises no wl_seat")?;
        state.screencopy.as_ref().ok_or("host compositor lacks zwlr_screencopy_manager_v1")?;

        let vk = match &state.vk_mgr {
            Some(mgr) => {
                let vk = mgr.create_virtual_keyboard(&seat, &qh, ());
                // A keymap must precede any key event; selkies replaces this with
                // its managed keymap through the ABI as soon as it starts.
                if let Some(text) = crate::wayland::vkclient::us_base_text() {
                    let mut data = text.as_bytes().to_vec();
                    data.push(0);
                    let fd = memfd_with(&data)?;
                    vk.keymap(KEYMAP_FORMAT_XKB_V1, fd.as_fd(), data.len() as u32);
                }
                Some(vk)
            }
            None => {
                eprintln!("[HostCapture] no zwp_virtual_keyboard_manager_v1: keyboard injection disabled.");
                None
            }
        };
        let vptr = match &state.vptr_mgr {
            Some(mgr) => Some(mgr.create_virtual_pointer(Some(&seat), &qh, ())),
            None => {
                eprintln!("[HostCapture] no zwlr_virtual_pointer_manager_v1: pointer injection disabled.");
                None
            }
        };
        bounded_roundtrip(&conn, &mut queue, &mut state)?;

        let (to_thread, from_main) = std::sync::mpsc::channel::<ToHost>();
        let (frame_tx, frames) = std::sync::mpsc::channel::<HostFrame>();
        let alive = Arc::new(AtomicBool::new(true));
        let display_size = Arc::new(Mutex::new((0, 0)));
        {
            let conn = conn.clone();
            let alive = alive.clone();
            let display_size = display_size.clone();
            std::thread::Builder::new()
                .name("pf-host-capture".into())
                .spawn(move || {
                    if let Err(e) =
                        capture_loop(conn, queue, state, gbm, from_main, frame_tx, display_size)
                    {
                        eprintln!("[HostCapture] capture loop ended: {e}");
                    }
                    alive.store(false, Ordering::Relaxed);
                })
                .map_err(|e| format!("spawn: {e}"))?;
        }
        Ok(Self {
            conn,
            vk,
            vptr,
            to_thread,
            frames,
            retained: Mutex::new(None),
            display_size,
            alive,
        })
    }

    pub fn start_capture(&self, width: i32, height: i32) {
        let _ = self.to_thread.send(ToHost::Start { width, height });
    }

    pub fn stop(&self) {
        let _ = self.to_thread.send(ToHost::Stop);
    }

    pub fn alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }

    /// Newest ready frame, releasing any staler ones straight back to the pool.
    pub fn try_take_frame(&self) -> Option<HostFrame> {
        let mut newest: Option<HostFrame> = None;
        loop {
            match self.frames.try_recv() {
                Ok(frame) => {
                    if let Some(stale) = newest.replace(frame) {
                        let _ = self.to_thread.send(ToHost::Release(stale.slot));
                    }
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
        newest
    }

    /// Return an encoded frame's slot for the host's next blit.
    pub fn release_frame(&self, frame: HostFrame) {
        let _ = self.to_thread.send(ToHost::Release(frame.slot));
    }

    /// Keep `frame` as the current content (releasing the one it replaces).
    pub fn retain_frame(&self, frame: HostFrame) {
        let old = self.retained.lock().unwrap().replace(frame);
        if let Some(old) = old {
            self.release_frame(old);
        }
    }

    /// Run `f` with the retained (current-content) frame, if any.
    pub fn with_retained<R>(&self, f: impl FnOnce(Option<&HostFrame>) -> R) -> R {
        f(self.retained.lock().unwrap().as_ref())
    }

    /// Upload selkies' managed keymap to the virtual keyboard verbatim.
    pub fn set_keymap(&self, text: &str) {
        let Some(vk) = &self.vk else { return };
        let mut data = text.as_bytes().to_vec();
        data.push(0);
        match memfd_with(&data) {
            Ok(fd) => {
                vk.keymap(KEYMAP_FORMAT_XKB_V1, fd.as_fd(), data.len() as u32);
                let _ = self.conn.flush();
            }
            Err(e) => eprintln!("[HostCapture] keymap upload failed: {e}"),
        }
    }

    /// Key event in xkb numbering (evdev + 8), matching the seat injectors.
    pub fn key(&self, xkb_keycode: u32, pressed: bool) {
        let Some(vk) = &self.vk else { return };
        if xkb_keycode < 8 {
            return;
        }
        vk.key(0, xkb_keycode - 8, if pressed { 1 } else { 0 });
        let _ = self.conn.flush();
    }

    pub fn pointer_motion_abs(&self, x: f64, y: f64) {
        let Some(vp) = &self.vptr else { return };
        let (w, h) = *self.display_size.lock().unwrap();
        if w <= 0 || h <= 0 {
            return;
        }
        let cx = x.clamp(0.0, (w - 1) as f64) as u32;
        let cy = y.clamp(0.0, (h - 1) as f64) as u32;
        vp.motion_absolute(0, cx, cy, w as u32, h as u32);
        vp.frame();
        let _ = self.conn.flush();
    }

    pub fn pointer_button(&self, btn: u32, pressed: bool) {
        let Some(vp) = &self.vptr else { return };
        vp.button(
            0,
            btn,
            if pressed { wl_pointer::ButtonState::Pressed } else { wl_pointer::ButtonState::Released },
        );
        vp.frame();
        let _ = self.conn.flush();
    }

    pub fn pointer_axis(&self, dx: f64, dy: f64) {
        let Some(vp) = &self.vptr else { return };
        if dy != 0.0 {
            vp.axis(0, wl_pointer::Axis::VerticalScroll, dy);
        }
        if dx != 0.0 {
            vp.axis(0, wl_pointer::Axis::HorizontalScroll, dx);
        }
        vp.frame();
        let _ = self.conn.flush();
    }
}

impl Drop for HostSession {
    fn drop(&mut self) {
        let _ = self.to_thread.send(ToHost::Stop);
    }
}

/// Ask the host (wlr-output-management) to set a custom mode on its first head.
/// Retries once across a `cancelled` (stale serial). False when the host lacks
/// the protocol or refused the mode.
fn resize_host_output(
    queue: &mut wayland_client::EventQueue<HostState>,
    state: &mut HostState,
    qh: &QueueHandle<HostState>,
    width: i32,
    height: i32,
) -> bool {
    let Some(mgr) = state.output_mgr.clone() else { return false };
    for _ in 0..2 {
        for _ in 0..50 {
            if state.om_serial.is_some() && !state.heads.is_empty() {
                break;
            }
            if queue.blocking_dispatch(state).is_err() {
                return false;
            }
        }
        let (Some(serial), Some(head)) = (state.om_serial, state.heads.first().cloned()) else {
            return false;
        };
        state.cfg_result = None;
        state.cfg_cancelled = false;
        let cfg = mgr.create_configuration(serial, qh, ());
        let cfg_head = cfg.enable_head(&head, qh, ());
        cfg_head.set_custom_mode(width, height, 0);
        cfg.apply();
        let _ = queue.flush();
        while state.cfg_result.is_none() && !state.cfg_cancelled {
            if queue.blocking_dispatch(state).is_err() {
                return false;
            }
        }
        cfg.destroy();
        if state.cfg_cancelled {
            // Stale serial: the compositor re-announces its state with a fresh one.
            state.om_serial = None;
            continue;
        }
        return state.cfg_result == Some(true);
    }
    false
}

fn fourcc_to_gbm(fourcc: u32) -> GbmFormat {
    // XR24 / AR24; anything else falls back to ARGB (the encoder reads BGRA bytes
    // and ignores alpha).
    match fourcc {
        0x34325258 => GbmFormat::Xrgb8888,
        _ => GbmFormat::Argb8888,
    }
}

fn capture_loop(
    _conn: Connection,
    mut queue: wayland_client::EventQueue<HostState>,
    mut state: HostState,
    gbm: Option<GbmDevice<File>>,
    from_main: Receiver<ToHost>,
    frame_tx: Sender<HostFrame>,
    display_size: Arc<Mutex<(i32, i32)>>,
) -> Result<(), String> {
    // Idle until capture starts.
    let (mut want_w, mut want_h) = loop {
        match from_main.recv() {
            Ok(ToHost::Start { width, height }) => break (width, height),
            Ok(ToHost::Release(..)) => continue,
            Ok(ToHost::Stop) | Err(_) => return Ok(()),
        }
    };
    let qh = queue.handle();
    let output = state.outputs.first().cloned().ok_or("host compositor has no wl_output")?;
    let screencopy = state.screencopy.clone().unwrap();
    if !resize_host_output(&mut queue, &mut state, &qh, want_w, want_h) {
        eprintln!(
            "[HostCapture] host output resize to {want_w}x{want_h} not applied; \
             capture follows the host's own size."
        );
    }

    let mut slots: Vec<Option<SlotBuffer>> = (0..SLOTS).map(|_| None).collect();
    let mut free: Vec<usize> = (0..SLOTS).collect();
    let mut announced: Option<(i32, i32)> = None;
    let mut warned_mismatch = false;
    let mut consecutive_failures = 0u32;

    loop {
        // Drain control messages; block only when out of buffers.
        loop {
            let msg = if free.is_empty() {
                match from_main.recv() {
                    Ok(m) => m,
                    Err(_) => return Ok(()),
                }
            } else {
                match from_main.try_recv() {
                    Ok(m) => m,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return Ok(()),
                }
            };
            match msg {
                ToHost::Release(slot) => free.push(slot),
                ToHost::Stop => return Ok(()),
                ToHost::Start { width, height } => {
                    // A capture reconfigure (client resolution change): resize the
                    // host output to follow, exactly like compositor mode does.
                    if (width, height) != (want_w, want_h) {
                        want_w = width;
                        want_h = height;
                        warned_mismatch = false;
                        if !resize_host_output(&mut queue, &mut state, &qh, want_w, want_h) {
                            eprintln!(
                                "[HostCapture] host output resize to {want_w}x{want_h} \
                                 not applied; capture follows the host's own size."
                            );
                        }
                    }
                }
            }
        }

        // One frame: negotiate (first time), attach our buffer, wait for damage.
        state.announce_dmabuf = None;
        state.announce_shm = None;
        state.buffer_done = false;
        state.damage.clear();
        state.ready = false;
        state.failed = false;
        let frame = screencopy.capture_output(1, &output, &qh, ());
        while !state.buffer_done && !state.failed {
            queue.blocking_dispatch(&mut state).map_err(|e| format!("dispatch: {e}"))?;
        }
        if state.failed {
            frame.destroy();
            consecutive_failures += 1;
            if consecutive_failures == 3 {
                eprintln!("[HostCapture] repeated screencopy failures; is the output alive?");
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }

        let (fw, fh) = state
            .announce_dmabuf
            .map(|(_, w, h)| (w, h))
            .or(state.announce_shm.map(|(_, w, h, _)| (w, h)))
            .ok_or("screencopy announced no buffer type")?;
        if announced != Some((fw, fh)) {
            announced = Some((fw, fh));
            *display_size.lock().unwrap() = (fw, fh);
            eprintln!(
                "[HostCapture] negotiated: {fw}x{fh} gbm={} dmabuf_global={} dmabuf_announce={:?} shm={:?}",
                gbm.is_some(),
                state.dmabuf.is_some(),
                state.announce_dmabuf,
                state.announce_shm,
            );
            // Size change: drop stale buffers.
            for s in slots.iter_mut() {
                *s = None;
            }
            free = (0..SLOTS).collect();
        }
        if (fw, fh) != (want_w, want_h) {
            // The encoder is configured for (want_w, want_h): hold frames until
            // the host applies the resize rather than encode mismatched buffers.
            frame.destroy();
            if !warned_mismatch {
                warned_mismatch = true;
                eprintln!(
                    "[HostCapture] waiting for host output {want_w}x{want_h} (currently {fw}x{fh})."
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(150));
            continue;
        }
        warned_mismatch = false;

        let slot_idx = *free.last().unwrap();
        if slots[slot_idx].is_none() {
            slots[slot_idx] = Some(match (&gbm, &state.dmabuf, state.announce_dmabuf) {
                (Some(dev), Some(dmabuf_global), Some((fourcc, w, h))) => {
                    let bo = dev
                        .create_buffer_object::<()>(
                            w as u32,
                            h as u32,
                            fourcc_to_gbm(fourcc),
                            BufferObjectFlags::RENDERING,
                        )
                        .map_err(|e| format!("GBM allocation {w}x{h}: {e:?}"))?;
                    let dmabuf = crate::create_dmabuf_from_bo(&bo);
                    let params = dmabuf_global.create_params(&qh, ());
                    let modifier: u64 = dmabuf.format().modifier.into();
                    for (i, handle) in dmabuf.handles().enumerate() {
                        params.add(
                            handle,
                            i as u32,
                            dmabuf.offsets().nth(i).unwrap_or(0),
                            dmabuf.strides().nth(i).unwrap_or(0),
                            (modifier >> 32) as u32,
                            (modifier & 0xffff_ffff) as u32,
                        );
                    }
                    let wl = params.create_immed(w, h, fourcc, zwp_linux_buffer_params_v1::Flags::empty(), &qh, ());
                    params.destroy();
                    SlotBuffer::Gpu { _bo: bo, dmabuf, wl }
                }
                _ => {
                    let (format, w, h, stride) =
                        state.announce_shm.ok_or("no shm fallback announced")?;
                    let shm = state.shm.clone().ok_or("host compositor lacks wl_shm")?;
                    let size = (stride * h) as usize;
                    let fd = memfd_with(&vec![0u8; size])?;
                    let pool = shm.create_pool(fd.as_fd(), size as i32, &qh, ());
                    let wl = pool.create_buffer(
                        0,
                        w,
                        h,
                        stride,
                        WEnum::<wl_shm::Format>::from(format).into_result().unwrap_or(wl_shm::Format::Xrgb8888),
                        &qh,
                        (),
                    );
                    let file = File::from(fd);
                    let map = unsafe { memmap2::MmapMut::map_mut(&file) }
                        .map_err(|e| format!("shm map: {e}"))?;
                    SlotBuffer::Cpu { _pool: pool, map: Arc::new(map), stride, format, wl }
                }
            });
        }
        free.pop();

        {
            let slot = slots[slot_idx].as_ref().unwrap();
            let wl = match slot {
                SlotBuffer::Gpu { wl, .. } => wl,
                SlotBuffer::Cpu { wl, .. } => wl,
            };
            frame.copy_with_damage(wl);
        }
        queue.flush().map_err(|e| format!("flush: {e}"))?;
        while !state.ready && !state.failed {
            queue.blocking_dispatch(&mut state).map_err(|e| format!("dispatch: {e}"))?;
        }
        frame.destroy();
        if state.failed {
            free.push(slot_idx);
            consecutive_failures += 1;
            if consecutive_failures == 3 {
                eprintln!("[HostCapture] repeated screencopy failures; is the output alive?");
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        }
        consecutive_failures = 0;

        let damage = std::mem::take(&mut state.damage);
        let out = match slots[slot_idx].as_ref().unwrap() {
            SlotBuffer::Gpu { dmabuf, .. } => HostFrame {
                slot: slot_idx,
                dmabuf: Some(dmabuf.clone()),
                cpu: None,
                width: fw,
                height: fh,
                damage,
            },
            SlotBuffer::Cpu { map, stride, format, .. } => HostFrame {
                slot: slot_idx,
                dmabuf: None,
                cpu: Some(HostCpuFrame {
                    map: map.clone(),
                    stride: *stride as usize,
                    format: *format,
                }),
                width: fw,
                height: fh,
                damage,
            },
        };
        if frame_tx.send(out).is_err() {
            return Ok(());
        }
    }
}
