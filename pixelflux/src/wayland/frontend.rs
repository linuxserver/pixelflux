/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Headless Smithay compositor that stands in for a real display server so ordinary Wayland
//! clients have somewhere to render — their composited output is exactly what the capture pipeline
//! reads back and H.264-encodes. There is no monitor, KMS, or libinput in this process, so
//! everything a desktop session normally receives from hardware — an output to map windows onto, a
//! seat to deliver input to, a clipboard to share — this frontend has to synthesize itself.
//!
//! This module owns `AppState`, the single context threaded through every Smithay protocol
//! handler, and implements those handlers: `wl_compositor` commit handling with the window
//! map/configure/focus state machine, seat/keyboard/pointer/touch routing through `FocusTarget`,
//! clipboard and primary-selection bridging to Python, and the xdg-shell / layer-shell /
//! xdg-activation / decoration / fractional-scale / dmabuf wiring. It also resolves cursor images
//! to PNG for the Python callback and provides the serial and monotonic-time helpers the input
//! path stamps onto events.

use std::borrow::Cow;
use std::fs::File;
use std::io::Cursor as IoCursor;
use std::time::Instant;

use gbm::{BufferObject, Device as RawGbmDevice};
use image::{ImageBuffer, ImageFormat, Rgba};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Mutex;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use smithay::backend::renderer::utils::RendererSurfaceState;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::{ Buffer, Fourcc};
use smithay::backend::renderer::{
    Bind, ExportMem, gles::GlesRenderer, pixman::PixmanRenderer, ImportDma,
};
use smithay::input::dnd::{DndFocus, Source};
use std::sync::Arc;
use crate::wayland::cursor::Cursor;
use smithay::wayland::viewporter::ViewporterState;
use smithay::delegate_viewporter;
use smithay::wayland::pointer_warp::{PointerWarpHandler, PointerWarpManager};
use smithay::reexports::wayland_server::protocol::wl_pointer::WlPointer;
use smithay::reexports::wayland_server::protocol::wl_shm;
use smithay::wayland::relative_pointer::RelativePointerManagerState;
use smithay::wayland::pointer_constraints::{PointerConstraintsHandler, PointerConstraintsState};
use smithay::input::pointer::PointerHandle;
use smithay::wayland::single_pixel_buffer::SinglePixelBufferState;
use smithay::delegate_single_pixel_buffer;
use smithay::desktop::{PopupKind, PopupManager};
use smithay::wayland::presentation::PresentationState;
use smithay::delegate_presentation;
use smithay::wayland::foreign_toplevel_list::{
    ForeignToplevelHandle, ForeignToplevelListHandler, ForeignToplevelListState,
};
use smithay::wayland::shell::xdg::decoration::{
    XdgDecorationHandler, XdgDecorationState,
};
use smithay::desktop::{layer_map_for_output, LayerSurface as DesktopLayerSurface};
use smithay::wayland::shell::wlr_layer::{
    WlrLayerShellHandler, WlrLayerShellState, Layer as WlrLayer, LayerSurface as WlrLayerSurface,
};
use smithay::delegate_layer_shell;
use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode;
use smithay::{delegate_foreign_toplevel_list, delegate_xdg_decoration};
use smithay::wayland::selection::wlr_data_control::{DataControlHandler, DataControlState};
use smithay::delegate_data_control;
use smithay::wayland::xdg_activation::{
    XdgActivationHandler, XdgActivationState, XdgActivationToken, XdgActivationTokenData,
};
use smithay::delegate_xdg_activation;
use smithay::wayland::selection::primary_selection::{
    set_primary_focus, PrimarySelectionHandler, PrimarySelectionState,
};
use smithay::delegate_primary_selection;

use smithay::{
    delegate_compositor, delegate_data_device, delegate_dmabuf, delegate_fractional_scale,
    delegate_output, delegate_seat, delegate_shm, delegate_virtual_keyboard_manager,
    delegate_xdg_shell, delegate_relative_pointer, delegate_pointer_warp, 
    delegate_pointer_constraints,
    desktop::{Space, Window},
    input::{
        keyboard::{KeyboardTarget, KeysymHandle, ModifiersState},
        pointer::{
            AxisFrame, ButtonEvent, CursorIcon, CursorImageAttributes, CursorImageStatus, GestureHoldBeginEvent,
            GestureHoldEndEvent, GesturePinchBeginEvent, GesturePinchEndEvent,
            GesturePinchUpdateEvent, GestureSwipeBeginEvent, GestureSwipeEndEvent,
            GestureSwipeUpdateEvent, MotionEvent, PointerTarget, RelativeMotionEvent,
        },
        touch::{DownEvent, OrientationEvent, ShapeEvent, TouchTarget, UpEvent},
        Seat, SeatHandler, SeatState,
    },
    output::Output,
    reexports::{
        wayland_protocols::xdg::shell::server::xdg_toplevel::State as XdgState,
        wayland_server::{
            backend::{ClientData, ClientId, DisconnectReason, ObjectId},
            protocol::{wl_buffer::WlBuffer, wl_surface::WlSurface},
            Client, DisplayHandle, Resource,
        },
    },
    utils::{Clock, IsAlive, Monotonic, Serial, Rectangle, Point, Logical},
    wayland::{
        buffer::BufferHandler,
        compositor::{
            with_states, BufferAssignment, CompositorClientState, CompositorHandler,
            CompositorState, SurfaceAttributes,
        },
        dmabuf::{DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier, get_dmabuf},
        fractional_scale::{FractionalScaleHandler, FractionalScaleManagerState},
        output::{OutputHandler, OutputManagerState},
        seat::WaylandFocus,
        selection::{
            data_device::{
                request_data_device_client_selection, DataDeviceHandler, DataDeviceState,
                WaylandDndGrabHandler,
            },
            SelectionHandler, SelectionSource, SelectionTarget,
        },
        shell::xdg::{
            PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
            XdgToplevelSurfaceData,
        },
        shm::{with_buffer_contents, ShmHandler, ShmState, BufferAccessError},
        virtual_keyboard::VirtualKeyboardManagerState,
    },
};

use crate::encoders::overlay::OverlayState;
use crate::encoders::vaapi::VaapiEncoder;
use crate::nvenc::NvencEncoder;
use crate::{RustCaptureSettings, StripeState};

use std::sync::atomic::{AtomicU32, Ordering};

static SERIAL_COUNTER: AtomicU32 = AtomicU32::new(1);

/// Hand out the next unique, monotonically increasing Wayland event serial.
///
/// Wayland tags each event with a serial so clients and Smithay can prove a request was caused
/// by a specific event (popup grab, selection change). Every injected input event draws a fresh
/// value from this process-wide atomic counter.
///
/// # Returns
///
/// A new [`Serial`] value.
pub fn next_serial() -> Serial {
    Serial::from(SERIAL_COUNTER.fetch_add(1, Ordering::SeqCst))
}

/// Millisecond timestamp for pointer / keyboard / touch events.
///
/// Samples `CLOCK_MONOTONIC` and wraps it to a `u32` millisecond count as required by the
/// Wayland protocol for input event timestamps.
///
/// # Returns
///
/// Monotonic time in milliseconds, wrapping at `u32::MAX`.
pub fn wayland_time() -> u32 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    (ts.tv_sec as u32).wrapping_mul(1000).wrapping_add((ts.tv_nsec as u32) / 1_000_000)
}

/// Microsecond timestamp for relative-pointer motion.
///
/// Samples `CLOCK_MONOTONIC` at microsecond resolution for the higher-resolution `u64` time
/// field used by the relative-pointer protocol.
///
/// # Returns
///
/// Monotonic time in microseconds, wrapping at `u64::MAX`.
pub fn wayland_utime() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    (ts.tv_sec as u64).wrapping_mul(1_000_000).wrapping_add((ts.tv_nsec as u64) / 1_000)
}

/// The one hardware H.264 encoder session backing a capture. Only a single GPU backend is
/// ever live for a given capture, and VA-API and NVENC expose entirely different session types, so
/// this enum is what lets the render and delivery code pass around "the hardware encoder" without
/// caring which vendor path actually produced the frames.
#[allow(clippy::large_enum_variant)]
pub enum GpuEncoder {
    Vaapi(VaapiEncoder),
    Nvenc(NvencEncoder),
}

/// Central context threaded through every Smithay handler; owns the Wayland globals, the
/// GBM/EGL (or pixman) renderer state, and the capture/encode pipeline state.
///
/// A single `AppState` lives for the whole compositor thread and is mutated in place by the
/// calloop event sources (client dispatch, input injection, the capture timer). The non-obvious
/// fields encode the pipeline's threading and buffer strategy:
///
/// 1. **Render / encode targets**:
///    - **`video_encoder`**: the zero-copy GPU session only (GLES render + same-GPU dmabuf
///      encode). Its EGL/dmabuf handles are calloop-affine, so this encoder runs inline on the
///      calloop thread; it is `None` whenever a readback path is active.
///    - **`encode_pool` / `encode_join`**: the readback capture||encode split. The calloop renders
///      and reads back into a pooled host buffer and publishes it, while a separate encode thread
///      owns the CPU / cross-GPU / pixman-HW encoders. `encode_join` returns that thread's hardware
///      session on shutdown so a restart can reconfigure it in place rather than rebuild it. Both
///      are `None` while a zero-copy GPU session runs.
///    - **`frame_buffer`**: scratch render target used only by the pixman memory-throttle path; the
///      normal readback path renders and reads back into the pooled buffers instead.
///    - **`pool_last_render` / `render_seq`**: buffer-age bookkeeping for the pixman path, which
///      renders directly into the pooled buffers (an age is "renders since this slot was last the
///      target"). The GLES path renders into one fixed offscreen buffer and never consults these.
///
/// 2. **Delivery** (`deliver_tx` / `deliver_join`): encoded frames go to a dedicated delivery
///    thread over a capacity-1 rendezvous channel, mirroring the X11 single-slot FramePool
///    (non-dropping, ordered, at most one frame of blocking backpressure), so a slow GIL-holding
///    Python callback can never stall input/control dispatch on the calloop thread.
///
/// 3. **Keyframes** (`pending_force_idr`): set by an IDR request (client reconnect / decoder reset)
///    and consumed once on the next captured frame to force an immediate keyframe.
///
/// 4. **Clipboard** (`pending_clipboard_read`): stages the mime chosen in `new_selection`; the loop
///    drains it only after the dispatch that stores the new client source, so the read targets the
///    new selection rather than the previous one.
///
/// 5. **GPU selection** (`auto_gpu_selected`): records that automatic (not explicit) selection
///    picked `render_node_path`, so `StartCapture` aims the encoder at that same node unless a
///    device was chosen explicitly.
pub struct AppState {
    pub compositor_state: CompositorState,
    pub fractional_scale_state: FractionalScaleManagerState,
    pub viewporter_state: ViewporterState,
    pub presentation_state: PresentationState,
    pub shm_state: ShmState,
    pub single_pixel_buffer: SinglePixelBufferState,
    pub dmabuf_state: DmabufState,
    pub dmabuf_global: Option<DmabufGlobal>,
    #[allow(dead_code)]
    pub output_state: OutputManagerState,
    pub seat_state: SeatState<AppState>,
    pub shell_state: XdgShellState,
    pub layer_shell_state: WlrLayerShellState,
    pub space: Space<Window>,
    pub data_device_state: DataDeviceState,
    pub data_control_state: DataControlState,
    pub dh: DisplayHandle,
    #[allow(dead_code)]
    pub seat: Seat<AppState>,
    pub outputs: Vec<Output>,
    pub pending_windows: Vec<Window>,

    pub foreign_toplevel_list: ForeignToplevelListState,
    pub xdg_decoration_state: XdgDecorationState,
    pub xdg_activation_state: XdgActivationState,
    pub primary_selection_state: PrimarySelectionState,
    pub popups: PopupManager,
    pub frame_buffer: Vec<u8>,

    pub gles_renderer: Option<GlesRenderer>,
    pub pixman_renderer: Option<PixmanRenderer>,

    pub gbm_device: Option<RawGbmDevice<File>>,
    pub offscreen_buffer: Option<(BufferObject<()>, Dmabuf)>,

    pub is_capturing: bool,
    pub settings: RustCaptureSettings,
    pub callback: Option<Py<PyAny>>,
    pub cursor_callback: Option<Py<PyAny>>,
    pub clipboard_callback: Option<Py<PyAny>>,
    pub pending_clipboard_read: Option<String>,

    pub last_log_time: Instant,
    pub start_time: Instant,
    pub clock: Clock<Monotonic>,

    pub frame_counter: u16,
    pub pending_force_idr: bool,
    pub needs_full_render: bool,
    pub use_gpu: bool,

    pub video_encoder: Option<GpuEncoder>,
    pub vaapi_state: StripeState,
    pub cursor_helper: Cursor,

    pub overlay_state: OverlayState,

    pub virtual_keyboard_state: VirtualKeyboardManagerState,

    pub current_cursor_icon: Option<CursorImageStatus>,
    pub cursor_buffer: Option<WlBuffer>,
    pub cursor_cache: std::collections::HashMap<u64, Vec<u8>>,
    pub render_cursor_on_framebuffer: bool,
    pub pointer_warp_state: PointerWarpManager,
    pub relative_pointer_state: RelativePointerManagerState,
    pub pointer_constraints_state: PointerConstraintsState,
    pub render_node_path: String,
    pub auto_gpu_selected: bool,
    pub recording_sink: Option<Arc<crate::recording_sink::RecordingSink>>,
    pub deliver_tx: Option<std::sync::mpsc::SyncSender<Vec<crate::encoders::software::EncodedStripe>>>,
    pub deliver_join: Option<std::thread::JoinHandle<()>>,
    /// Zero-copy frame parked after a full delivery channel. The hardware encode and
    /// its delivery both run on the calloop thread, and a blocking send there would
    /// freeze input/command/Wayland dispatch for as long as the Python consumer
    /// stalls. An encoded frame is part of the H.264 reference chain and can never
    /// be dropped, so it parks here instead, and new encodes pause until it leaves.
    pub pending_hw_delivery: Option<Vec<crate::encoders::software::EncodedStripe>>,
    /// Damage seen on ticks skipped while `pending_hw_delivery` was parked, folded
    /// into the next encode's change detection so a change that happened during the
    /// pause is never lost (each tick's damage list is otherwise discarded).
    pub pending_hw_damage: bool,
    pub encode_pool: Option<Arc<crate::WlFramePool>>,
    pub encode_join: Option<std::thread::JoinHandle<Option<GpuEncoder>>>,
    pub encode_controls: Arc<crate::WlEncodeControls>,
    pub encode_stats: Arc<crate::WlEncodeStats>,
    pub pool_last_render: Vec<u64>,
    pub render_seq: u64,
    pub pending_screenshot: Option<std::sync::mpsc::Sender<Vec<u8>>>,
}

/// Pointer-constraints protocol wiring. The headless capture path never enforces a lock or
/// confinement region, so activation and cursor-position hints are accepted as no-ops; the global
/// still exists so clients may bind it without error.
impl PointerConstraintsHandler for AppState {
    fn new_constraint(&mut self, _surface: &WlSurface, _pointer: &PointerHandle<Self>) {}

    fn cursor_position_hint(
        &mut self,
        _surface: &WlSurface,
        _pointer: &PointerHandle<Self>,
        _location: Point<f64, Logical>,
    ) {}
}

/// Foreign-toplevel-list protocol: exposes the managed state so Smithay can advertise each
/// toplevel (title / app-id) to listing clients such as taskbars.
impl ForeignToplevelListHandler for AppState {
    fn foreign_toplevel_list_state(&mut self) -> &mut ForeignToplevelListState {
        &mut self.foreign_toplevel_list
    }
}

/// xdg-activation protocol: hands out activation tokens and, on redemption, raises the
/// target window.
///
/// `token_created` accepts every token. `request_activation` honors a token only while it is fresh
/// (issued less than 10 seconds ago) so a stale or replayed token cannot steal focus, then raises
/// the window whose surface matches to the top of the space.
impl XdgActivationHandler for AppState {
    fn activation_state(&mut self) -> &mut XdgActivationState {
        &mut self.xdg_activation_state
    }

    fn token_created(&mut self, _token: XdgActivationToken, _data: XdgActivationTokenData) -> bool {
        true
    }

    fn request_activation(
        &mut self,
        _token: XdgActivationToken,
        token_data: XdgActivationTokenData,
        surface: WlSurface,
    ) {
        if token_data.timestamp.elapsed().as_secs() < 10 {
            let window = self.space.elements().find(|w| w.wl_surface().as_deref() == Some(&surface)).cloned();
            if let Some(window) = window {
                self.space.raise_element(&window, true);
            }
        }
    }
}

/// Carry the X11-style primary selection so middle-click paste works between clients.
/// Exposing the managed state lets Smithay record which client currently owns the primary selection;
/// `focus_changed` then keeps that ownership tracking keyboard focus, so a middle-click pastes from
/// whichever client the user is actually working in.
impl PrimarySelectionHandler for AppState {
    fn primary_selection_state(&mut self) -> &mut PrimarySelectionState {
        &mut self.primary_selection_state
    }
}

/// Default every toplevel to server-side decorations so clients don't bake their own title
/// bars and borders into the captured image.
///
/// The frontend shows fullscreen application content with no window-manager chrome, so a client
/// left to draw client-side decorations would paint a stray title bar straight into the encoded
/// frame. Pinning the negotiation to `Mode::ServerSide` (in `new_decoration`, and again when a
/// client unsets its preference in `unset_mode`) hands decoration duty to the compositor — which
/// draws none — leaving clean, borderless output; `request_mode` still grants a mode a client
/// explicitly insists on. Each path acknowledges with a configure.
impl XdgDecorationHandler for AppState {
    fn new_decoration(&mut self, toplevel: ToplevelSurface) {
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(Mode::ServerSide);
        });
        toplevel.send_configure();
    }

    fn request_mode(&mut self, toplevel: ToplevelSurface, mode: Mode) {
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(mode);
        });
        toplevel.send_configure();
    }

    fn unset_mode(&mut self, toplevel: ToplevelSurface) {
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(Mode::ServerSide);
        });
        toplevel.send_configure();
    }
}

/// wlr-layer-shell protocol: places panels / overlays / backgrounds (layer surfaces).
///
/// `new_layer_surface` resolves the requested output (or the first output), configures the surface
/// to that output's full pixel size, and maps it into the output's layer map so the render loop
/// composites it in the correct z-order. `layer_destroyed` needs no action here.
impl WlrLayerShellHandler for AppState {
    fn shell_state(&mut self) -> &mut WlrLayerShellState {
        &mut self.layer_shell_state
    }

    fn new_layer_surface(
        &mut self,
        surface: WlrLayerSurface,
        output: Option<smithay::reexports::wayland_server::protocol::wl_output::WlOutput>,
        _layer: WlrLayer,
        namespace: String,
    ) {
        let smithay_output = if let Some(wlo) = output.as_ref() {
            self.outputs.iter().find(|o| o.owns(wlo))
        } else {
            self.outputs.first()
        };

        if let Some(output) = smithay_output {
            let mode = output.current_mode().unwrap();
            
            surface.with_pending_state(|state| {
                state.size = Some(((mode.size.w as f64) as i32, (mode.size.h as f64) as i32).into());
            });
            surface.send_configure();

            let layer = DesktopLayerSurface::new(surface, namespace);
            let _ = layer_map_for_output(output).map_layer(&layer);
        }
    }

    fn layer_destroyed(&mut self, _surface: WlrLayerSurface) {}
}

/// Core `wl_compositor` protocol: per-surface commit handling plus the window
/// map/configure/focus state machine.
impl CompositorHandler for AppState {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }
    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<ClientState>().unwrap().compositor_state
    }

    /// Fold a client's `wl_surface.commit` into compositor state and — the reason the window
    /// logic lives in this handler — walk a brand-new toplevel through the xdg-shell handshake it
    /// must finish before it may be shown.
    ///
    /// A window cannot just appear on its first commit: xdg-shell requires the compositor to send an
    /// initial configure (telling the client the size and state to draw at) and the client to ack it
    /// and commit a buffer matching that size before the surface counts as mapped. Mapping earlier
    /// would flash an unconfigured, wrongly-sized window into the captured frame. So a pending
    /// toplevel is carried across two commits instead of one, while everything else here is
    /// per-commit housekeeping that must run whether or not a map is in flight. On each commit, in
    /// order:
    ///
    /// 1. **Buffer intake**: `on_commit_buffer_handler` ingests the newly attached buffer.
    /// 2. **Layer relayout**: if the surface is a mapped layer surface, re-arrange the output's
    ///    layer map so geometry tracks the new content.
    /// 3. **Cursor refresh**: if the surface backs the current cursor, re-send its image so a client
    ///    animating its own cursor surface is reflected downstream.
    /// 4. **Foreign-toplevel metadata**: push the current title / app-id to the foreign-toplevel
    ///    handle for taskbar-style clients.
    /// 5. **Window on-commit**: forward the commit to the matching mapped `Window`.
    /// 6. **Two-phase map of a pending toplevel**:
    ///    - **First commit — no initial configure sent yet**: this commit is the client announcing
    ///      it wants to be shown, so compute the logical size from the output mode/scale (falling
    ///      back to the settings resolution), send a fullscreen + activated configure, and re-queue
    ///      the window to wait for the client's acked commit — nothing is mapped yet.
    ///    - **Acked commit — the client has drawn to that configure**: map the element at the origin,
    ///      refresh its cached bounding box *before* reading geometry (so the drift check sees
    ///      current geometry and doesn't fire a redundant configure), enter the output, and, only if
    ///      the client's geometry still drifts more than a pixel from the expected fullscreen size,
    ///      send one corrective configure. Finally give the new window keyboard focus so input lands
    ///      on it at once.
    fn commit(&mut self, surface: &WlSurface) {
        smithay::backend::renderer::utils::on_commit_buffer_handler::<Self>(surface);

        if let Some(output) = self.outputs.first() {
            let mut layer_map = layer_map_for_output(output);
            let mut found = false;
            for layer in layer_map.layers() {
                if layer.wl_surface() == surface {
                    found = true;
                    break;
                }
            }
            if found {
                layer_map.arrange();
            }
        }

        if let Some(CursorImageStatus::Surface(ref cursor_surface)) = self.current_cursor_icon {
            if cursor_surface == surface {
                let status = CursorImageStatus::Surface(surface.clone());
                self.send_cursor_image(&status);
            }
        }

        if let Some(handle) = with_states(surface, |states| states.data_map.get::<ForeignToplevelHandle>().cloned()) {
             if let Some(window) = self.space.elements().find(|w| w.wl_surface().as_deref() == Some(surface)) {
                 if let Some(_toplevel) = window.toplevel() {
                     let (title, app_id) = with_states(surface, |states| {
                        let attributes = states.data_map.get::<XdgToplevelSurfaceData>().unwrap().lock().unwrap();
                        (attributes.title.clone(), attributes.app_id.clone())
                     });
                     
                     handle.send_title(&title.unwrap_or_default());
                     handle.send_app_id(&app_id.unwrap_or_default());
                     handle.send_done();
                 }
             }
        }

        if let Some(window) = self
            .space
            .elements()
            .find(|w| w.toplevel().map(|tl| tl.wl_surface() == surface).unwrap_or(false))
        {
            window.on_commit();
        }

        if let Some(idx) = self.pending_windows.iter().position(|w| {
            w.toplevel().map(|tl| tl.wl_surface() == surface).unwrap_or(false)
        }) {
            let window = self.pending_windows.remove(idx);
            let toplevel = window.toplevel().unwrap();

            let initial_configure_sent = with_states(surface, |states| {
                states
                    .data_map
                    .get::<XdgToplevelSurfaceData>()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .initial_configure_sent
            });

            if !initial_configure_sent {
                let (logical_width, logical_height) = if let Some(output) = self.outputs.first() {
                    let mode = output.current_mode().unwrap();
                    let scale = output.current_scale().fractional_scale();
                    (
                        (mode.size.w as f64 / scale).round() as i32,
                        (mode.size.h as f64 / scale).round() as i32,
                    )
                } else {
                    let scale = self.settings.scale.max(0.1);
                    (
                        (self.settings.width as f64 / scale).round() as i32,
                        (self.settings.height as f64 / scale).round() as i32,
                    )
                };

                toplevel.with_pending_state(|state| {
                    state.states.set(XdgState::Activated);
                    state.states.set(XdgState::Fullscreen);
                    state.size = Some((logical_width, logical_height).into());
                });
                toplevel.send_configure();

                self.pending_windows.push(window);
            } else {
                self.space.map_element(window.clone(), (0, 0), true);
                window.on_commit();

                if let Some(output) = self.outputs.first() {
                    output.enter(surface);

                    let mode = output.current_mode().unwrap();
                    let scale = output.current_scale().fractional_scale();
                    let (expected_w, expected_h) = (
                        (mode.size.w as f64 / scale).round() as i32,
                        (mode.size.h as f64 / scale).round() as i32,
                    );
                    
                    let geo = window.geometry();
                    if (geo.size.w - expected_w).abs() > 1 || (geo.size.h - expected_h).abs() > 1 {
                        toplevel.with_pending_state(|state| {
                            state.states.set(XdgState::Activated);
                            state.states.set(XdgState::Fullscreen);
                            state.size = Some((expected_w, expected_h).into());
                        });
                        toplevel.send_configure();
                    }
                }

                let serial = next_serial();
                let target = FocusTarget::Window(window.clone());
                if let Some(keyboard) = self.seat.get_keyboard() {
                    keyboard.set_focus(self, Some(target.clone()), serial);
                }
            }
        }
    }
}


impl AppState {
    /// Drain a clipboard read staged by `new_selection` and hand `(mime, bytes)` to the
    /// Python callback off-thread.
    ///
    /// Runs from the loop *after* the dispatch that stored the new client source, so the request
    /// targets the current selection rather than the previous one. It clones the callback, opens a
    /// pipe, and asks the owning client source to write the chosen mime into the pipe's writer. A
    /// spawned reader thread then reads the response — capped at 64 MiB so a hostile client cannot
    /// balloon memory, and bounded by a 10 s poll deadline so a client that takes the selection but
    /// never writes nor closes its fd cannot pin the thread forever (each clipboard change would
    /// otherwise leak one zombie thread + pipe) — and, unless the interpreter is finalizing,
    /// delivers the bytes to Python. The `PY_SHUTDOWN` checks keep this off a shutting-down
    /// interpreter.
    pub(crate) fn process_pending_clipboard_read(&mut self) {
        let Some(mime) = self.pending_clipboard_read.take() else { return };
        if crate::PY_SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let Some(cb) = self
            .clipboard_callback
            .as_ref()
            .map(|c| Python::attach(|py| c.clone_ref(py)))
        else {
            return;
        };
        let Ok((reader, writer)) = std::io::pipe() else { return };
        if request_data_device_client_selection::<AppState>(&self.seat, mime.clone(), writer.into())
            .is_err()
        {
            return;
        }
        std::thread::spawn(move || {
            use std::io::Read;
            use std::os::fd::AsRawFd;
            const CAP: usize = 64 * 1024 * 1024;
            const DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);
            let start = Instant::now();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 65536];
            loop {
                let Some(remaining) = DEADLINE.checked_sub(start.elapsed()) else { return };
                let mut pfd = libc::pollfd {
                    fd: reader.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                let timeout_ms = remaining.as_millis().min(i32::MAX as u128).max(1) as i32;
                let ready = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
                if ready < 0 {
                    if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    return;
                }
                if ready == 0 {
                    return;
                }
                match (&reader).read(&mut chunk) {
                    Ok(0) => break,
                    Ok(n) => {
                        let room = CAP - buf.len();
                        let take_n = n.min(room);
                        buf.extend_from_slice(&chunk[..take_n]);
                        if buf.len() == CAP {
                            break;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted
                        || e.kind() == std::io::ErrorKind::WouldBlock => continue,
                    Err(_) => return,
                }
            }
            if buf.is_empty() {
                return;
            }
            if crate::PY_SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            Python::attach(|py| {
                let bytes = PyBytes::new(py, &buf);
                let _ = cb.call1(py, (mime.as_str(), bytes));
            });
        });
    }


    /// Resolve a `CursorImageStatus` to PNG bytes plus hotspot and invoke the Python cursor
    /// callback. Also re-invoked from the calloop command handlers to replay the retained cursor
    /// when a callback re-registers or a capture restarts.
    ///
    /// Dispatches on the cursor status:
    ///
    /// 1. **Named**: look the themed cursor up by CSS name and emit its cached PNG (`"png"`), or
    ///    `"error"` if the theme lacks it.
    /// 2. **Hidden**: emit `"hide"` — the only message allowed through with empty data, since an
    ///    intentional pointer hide must blank the consumer's cursor.
    /// 3. **Surface** (a client-supplied cursor sprite): ignore any surface without the
    ///    `cursor_image` role, read the hotspot, then read the backing buffer by one of two paths:
    ///    - **SHM**: hash only the sprite's sub-region — width/height/stride/offset/format plus the
    ///      pixel span — because many sprites share one pool and differ only by `offset`, so hashing
    ///      the whole pool would collide. On a cache miss (and only when ≤128×128) convert the
    ///      BGRA/XRGB pixels to RGBA and PNG-encode; `Xrgb8888` has no alpha so byte 3 is forced
    ///      opaque, and stride/offset are clamped non-negative with checked arithmetic so a garbage
    ///      descriptor skips pixels instead of panicking.
    ///    - **dmabuf** (`NotManaged`): bind it to the GLES renderer, copy the framebuffer to
    ///      `Abgr8888`, map it back, and PNG-encode using the derived readback stride.
    ///    The PNG cache is bounded at 100 entries by arbitrary eviction; content-hashing means an
    ///    evicted sprite simply re-inserts on the next render.
    ///
    /// A surface whose buffer could not be read yields `("surface", empty)`, which the final gate
    /// suppresses — Python is called only for non-empty data or `"hide"` — so a transient read miss
    /// (common during a stop/start replay) preserves the consumer's last cursor instead of blanking
    /// it. `PY_SHUTDOWN` gates the whole call off a finalizing interpreter.
    pub(crate) fn send_cursor_image(&mut self, image: &CursorImageStatus) {
        if crate::PY_SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        if let Some(ref cb) = self.cursor_callback {
            let (msg_type, data, hot_x, hot_y) = match image {
                CursorImageStatus::Named(icon) => {
                    self.cursor_buffer = None;
                    let name = cursor_icon_to_str(icon);
                    if let Some((png_bytes, x, y)) = self.cursor_helper.get_png_data(name) {
                        ("png", png_bytes, x as i32, y as i32)
                    } else {
                        ("error", Vec::new(), 0, 0)
                    }
                }
                CursorImageStatus::Hidden => {
                    self.cursor_buffer = None;
                    ("hide", Vec::new(), 0, 0)
                },
                CursorImageStatus::Surface(ref surface) => {
                    let mut final_png = Vec::new();
                    let mut hot_x = 0;
                    let mut hot_y = 0;
                    let mut is_cursor_role = false;

                    with_states(surface, |states| {
                        if states.role == Some("cursor_image") {
                            is_cursor_role = true;
                        }
                        if let Some(attributes) = states.data_map.get::<Mutex<CursorImageAttributes>>() {
                            if let Ok(guard) = attributes.lock() {
                                hot_x = guard.hotspot.x;
                                hot_y = guard.hotspot.y;
                            }
                        }
                    });

                    if !is_cursor_role {
                        return;
                    }

                    let buffer_found = with_states(surface, |states| {
                        let mut attrs = states.cached_state.get::<SurfaceAttributes>();
                        
                        if let Some(BufferAssignment::NewBuffer(b)) = &attrs.current().buffer {
                            return Some(b.clone());
                        }

                        if let Some(mutex) = states.data_map.get::<Mutex<RendererSurfaceState>>() {
                            if let Ok(renderer_state) = mutex.try_lock() {
                                if let Some(b) = renderer_state.buffer() {
                                    let wl_buffer: &wayland_server::protocol::wl_buffer::WlBuffer = b;
                                    return Some(wl_buffer.clone());
                                }
                            }
                        }
                        None
                    });

                    if let Some(buffer) = buffer_found {
                        let shm_result = with_buffer_contents(&buffer, |ptr, len, spec| {
                            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                            let mut hasher = DefaultHasher::new();
                            spec.width.hash(&mut hasher);
                            spec.height.hash(&mut hasher);
                            spec.stride.hash(&mut hasher);
                            spec.offset.hash(&mut hasher);
                            spec.format.hash(&mut hasher);
                            let start = (spec.offset.max(0) as usize).min(len);
                            let span = (spec.stride.max(0) as usize)
                                .saturating_mul(spec.height.max(0) as usize);
                            let end = start.saturating_add(span).min(len);
                            slice[start..end].hash(&mut hasher);
                            let hash = hasher.finish();
                            (hash, spec.width, spec.height, spec.stride, spec.format, spec.offset, slice.to_vec())
                        });

                        match shm_result {
                            Ok((hash, width, height, stride, format, buf_offset, raw_bytes)) => {
                                if let Some(cached_png) = self.cursor_cache.get(&hash) {
                                    final_png = cached_png.clone();
                                } else {
                                    if width <= 128 && height <= 128 && !raw_bytes.is_empty() {
                                        let mut img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width as u32, height as u32);
                                        let stride_usize = stride.max(0) as usize;
                                        let base_offset = buf_offset.max(0) as usize;

                                        for y in 0..(height as u32) {
                                            for x in 0..(width as u32) {
                                                let offset = (y as usize)
                                                    .checked_mul(stride_usize)
                                                    .and_then(|row| base_offset.checked_add(row))
                                                    .and_then(|o| o.checked_add((x as usize) * 4));
                                                let offset = match offset {
                                                    Some(o) => o,
                                                    None => continue,
                                                };
                                                if offset.checked_add(4).is_some_and(|end| end <= raw_bytes.len()) {
                                                    let alpha = if format == wl_shm::Format::Xrgb8888 {
                                                        255
                                                    } else {
                                                        raw_bytes[offset + 3]
                                                    };
                                                    img_buf.put_pixel(x, y, Rgba([
                                                        raw_bytes[offset + 2],
                                                        raw_bytes[offset + 1],
                                                        raw_bytes[offset],
                                                        alpha
                                                    ]));
                                                }
                                            }
                                        }

                                        let mut bytes = Vec::new();
                                        if img_buf.write_to(&mut IoCursor::new(&mut bytes), ImageFormat::Png).is_ok() {
                                            self.cursor_cache.insert(hash, bytes.clone());
                                            final_png = bytes;
                                            if self.cursor_cache.len() > 100 {
                                                let evict = *self.cursor_cache.keys().next().unwrap();
                                                self.cursor_cache.remove(&evict);
                                            }
                                        }
                                    }
                                }
                            },
                            Err(BufferAccessError::NotManaged) => {
                                let mut gles_data: Option<(u64, i32, i32, Vec<u8>)> = None;

                                let dmabuf_opt = get_dmabuf(&buffer).ok().cloned();

                                if let Some(mut dmabuf) = dmabuf_opt {
                                    if let Some(renderer) = self.gles_renderer.as_mut() {
                                        let width = dmabuf.width() as i32;
                                        let height = dmabuf.height() as i32;

                                        match renderer.bind(&mut dmabuf) {
                                            Ok(frame) => {
                                                let rect = Rectangle::new((0, 0).into(), (width, height).into());
                                                
                                                match renderer.copy_framebuffer(&frame, rect, Fourcc::Abgr8888) {
                                                    Ok(mapping) => {
                                                        match renderer.map_texture(&mapping) {
                                                            Ok(data) => {
                                                                let mut hasher = DefaultHasher::new();
                                                                data.hash(&mut hasher);
                                                                let hash = hasher.finish();
                                                                gles_data = Some((hash, width, height, data.to_vec()));
                                                            },
                                                            Err(e) => eprintln!("Failed to map texture: {:?}", e)
                                                        }
                                                    },
                                                    Err(e) => eprintln!("Failed to copy framebuffer: {:?}", e)
                                                }
                                            },
                                            Err(e) => eprintln!("Failed to bind dmabuf to renderer: {:?}", e)
                                        }
                                    }
                                }

                                if let Some((hash, width, height, raw_bytes)) = gles_data {
                                     if let Some(cached_png) = self.cursor_cache.get(&hash) {
                                         final_png = cached_png.clone();
                                     } else {
                                         if width <= 128 && height <= 128 && !raw_bytes.is_empty() {
                                             let mut img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width as u32, height as u32);
                                             let stride_usize = rgba_readback_stride(raw_bytes.len(), height as usize, width as usize);
                                             
                                             for y in 0..(height as u32) {
                                                 for x in 0..(width as u32) {
                                                     let offset = (y as usize * stride_usize) + (x as usize * 4);
                                                     if offset + 4 <= raw_bytes.len() {
                                                         img_buf.put_pixel(x, y, Rgba([
                                                             raw_bytes[offset],
                                                             raw_bytes[offset + 1],
                                                             raw_bytes[offset + 2],
                                                             raw_bytes[offset + 3]
                                                         ]));
                                                     }
                                                 }
                                             }

                                             let mut bytes = Vec::new();
                                             if img_buf.write_to(&mut IoCursor::new(&mut bytes), ImageFormat::Png).is_ok() {
                                                 self.cursor_cache.insert(hash, bytes.clone());
                                                 final_png = bytes;
                                                 if self.cursor_cache.len() > 100 {
                                                     let evict = *self.cursor_cache.keys().next().unwrap();
                                                     self.cursor_cache.remove(&evict);
                                                 }
                                             }
                                         }
                                     }
                                }
                            },
                            Err(_) => {}
                        }
                        
                        self.cursor_buffer = Some(buffer);
                    }

                    if !final_png.is_empty() {
                        ("png", final_png, hot_x, hot_y)
                    } else {
                        ("surface", Vec::new(), 0, 0)
                    }
                }
            };

            if !data.is_empty() || msg_type == "hide" {
                Python::attach(|py| {
                    let py_bytes = PyBytes::new(py, &data);
                    let _ = cb.call1(py, (msg_type, py_bytes, hot_x, hot_y));
                });
            }
        }
    }
}

/// Clipboard mime types the bridge can hand to Python, most specific first; `new_selection`
/// picks the first of these that the client's source offers.
const CLIPBOARD_MIME_PREFERENCE: &[&str] = &[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/bmp",
    "text/plain;charset=utf-8",
    "UTF8_STRING",
    "text/plain",
    "STRING",
    "TEXT",
];

/// Selection (clipboard) bridge between Wayland clients and Python. `SelectionUserData` is
/// the Python-owned payload `(mime, bytes)` served to pasting clients when Python holds the
/// selection.
impl SelectionHandler for AppState {
    type SelectionUserData = std::sync::Arc<(String, Vec<u8>)>;

    /// A client took the clipboard: pick the best offered mime and stage it for the loop to
    /// read.
    ///
    /// Only client-owned clipboard (not primary) selections are relayed to Python. Among the
    /// source's offered mimes it chooses the most specific match from `CLIPBOARD_MIME_PREFERENCE`
    /// and records it in `pending_clipboard_read`. The read itself is deferred: the new source is
    /// stored only after this handler returns, so `process_pending_clipboard_read` runs
    /// post-dispatch and reads the new selection rather than the previous one.
    fn new_selection(
        &mut self,
        ty: SelectionTarget,
        source: Option<SelectionSource>,
        seat: Seat<Self>,
    ) {
        if ty != SelectionTarget::Clipboard {
            return;
        }
        if crate::PY_SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        if self.clipboard_callback.is_none() {
            return;
        }
        let Some(source) = source else { return };
        let mimes = source.mime_types();
        let Some(mime) = CLIPBOARD_MIME_PREFERENCE
            .iter()
            .find(|want| mimes.iter().any(|m| m == *want))
            .map(|s| s.to_string())
        else {
            return;
        };
        self.pending_clipboard_read = Some(mime);
        let _ = seat;
    }

    /// A client pastes the Python-owned selection: stream the stored bytes into the client's
    /// fd on a spawned thread, since the receiving pipe may backpressure.
    fn send_selection(
        &mut self,
        ty: SelectionTarget,
        _mime_type: String,
        fd: std::os::fd::OwnedFd,
        _seat: Seat<Self>,
        user_data: &Self::SelectionUserData,
    ) {
        if ty != SelectionTarget::Clipboard {
            return;
        }
        let payload = user_data.clone();
        std::thread::spawn(move || {
            use std::io::Write;
            let mut f = std::fs::File::from(fd);
            let _ = f.write_all(&payload.1);
        });
    }
}

/// `wl_data_device` protocol: exposes the managed state for drag-and-drop and clipboard
/// data transfers.
impl DataDeviceHandler for AppState {
    fn data_device_state(&mut self) -> &mut DataDeviceState {
        &mut self.data_device_state
    }
}
/// wlr-data-control protocol: exposes the managed state so privileged clients can read and
/// set selections.
impl DataControlHandler for AppState {
    fn data_control_state(&mut self) -> &mut DataControlState {
        &mut self.data_control_state
    }
}
/// Marker impl enabling Wayland drag-and-drop grabs with Smithay's default behavior.
impl WaylandDndGrabHandler for AppState {}
/// Buffer lifecycle hook: buffer destruction needs no bookkeeping here.
impl BufferHandler for AppState {
    fn buffer_destroyed(&mut self, _buffer: &WlBuffer) {}
}
/// `wl_shm` protocol: exposes the shared-memory buffer state.
impl ShmHandler for AppState {
    fn shm_state(&self) -> &ShmState {
        &self.shm_state
    }
}
/// Output protocol marker impl (no per-output callbacks are needed).
impl OutputHandler for AppState {}

/// Linux-dmabuf protocol: imports client dmabufs into the GLES renderer.
///
/// `dmabuf_imported` attempts the import into the GLES renderer and signals the client through the
/// notifier — success only when a GLES renderer exists and the import succeeds, otherwise failure
/// (the software / pixman path advertises no dmabuf global, so it always fails here).
impl DmabufHandler for AppState {
    fn dmabuf_state(&mut self) -> &mut DmabufState {
        &mut self.dmabuf_state
    }

    fn dmabuf_imported(
        &mut self,
        _global: &DmabufGlobal,
        dmabuf: Dmabuf,
        notifier: ImportNotifier,
    ) {
        if let Some(renderer) = self.gles_renderer.as_mut() {
            if renderer.import_dmabuf(&dmabuf, None).is_ok() {
                let _ = notifier.successful::<AppState>();
            } else {
                notifier.failed();
            }
        } else {
            notifier.failed();
        }
    }
}

/// Fractional-scale protocol: tells a newly-bound surface the output's current fractional
/// scale so it renders at the right pixel density.
impl FractionalScaleHandler for AppState {
    fn new_fractional_scale(
        &mut self,
        surface: smithay::reexports::wayland_server::protocol::wl_surface::WlSurface,
    ) {
        if let Some(output) = self.outputs.first() {
            let scale = output.current_scale().fractional_scale();
            with_states(&surface, |states| {
                smithay::wayland::fractional_scale::with_fractional_scale(states, |fs| {
                    fs.set_preferred_scale(scale);
                });
            });
        }
    }
}

/// Input-event target for Smithay's seat handlers. Smithay requires a concrete type as the
/// "target" of a keyboard / pointer / touch event; `FocusTarget` bridges that to the concrete
/// Wayland surface behind a window, popup, or layer surface.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum FocusTarget {
    Window(Window),
    Popup(PopupKind),
    LayerSurface(DesktopLayerSurface),
}

/// Wrap a `Window` as a focus target.
impl From<Window> for FocusTarget {
    fn from(w: Window) -> Self { FocusTarget::Window(w) }
}

/// Wrap a popup as a focus target.
impl From<PopupKind> for FocusTarget {
    fn from(p: PopupKind) -> Self { FocusTarget::Popup(p) }
}

/// Wrap a layer surface as a focus target.
impl From<DesktopLayerSurface> for FocusTarget {
    fn from(l: DesktopLayerSurface) -> Self { FocusTarget::LayerSurface(l) }
}

/// Liveness of a focus target: true while its underlying window / popup / layer surface is
/// still alive, so dead targets are dropped from focus.
impl IsAlive for FocusTarget {
    fn alive(&self) -> bool {
        match self {
            FocusTarget::Window(w) => w.alive(),
            FocusTarget::Popup(p) => p.alive(),
            FocusTarget::LayerSurface(l) => l.alive(),
        }
    }
}

/// Expose the wrapped target's underlying `wl_surface` and same-client checks so Smithay
/// can route focus and selection ownership by client.
impl WaylandFocus for FocusTarget {
    fn wl_surface(&self) -> Option<Cow<'_, WlSurface>> {
        match self {
            FocusTarget::Window(w) => w.wl_surface(),
            FocusTarget::Popup(p) => Some(Cow::Borrowed(p.wl_surface())),
            FocusTarget::LayerSurface(l) => Some(Cow::Borrowed(l.wl_surface())),
        }
    }
    fn same_client_as(&self, object_id: &ObjectId) -> bool {
        match self {
            FocusTarget::Window(w) => w.same_client_as(object_id),
            FocusTarget::Popup(p) => p.wl_surface().id().same_client_as(object_id),
            FocusTarget::LayerSurface(l) => l.wl_surface().id().same_client_as(object_id),
        }
    }
}

/// Forward every keyboard event (enter / leave / key / modifiers) to the wrapped target's
/// underlying `wl_surface`, which carries Smithay's real keyboard-target implementation.
impl KeyboardTarget<AppState> for FocusTarget {
    fn enter(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        keys: Vec<KeysymHandle<'_>>,
        serial: Serial,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::keyboard::KeyboardTarget::enter(
                surface.as_ref(),
                seat,
                data,
                keys,
                serial,
            );
        }
    }
    fn leave(&self, seat: &Seat<AppState>, data: &mut AppState, serial: Serial) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::keyboard::KeyboardTarget::leave(surface.as_ref(), seat, data, serial);
        }
    }
    fn key(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        key: KeysymHandle<'_>,
        state: smithay::backend::input::KeyState,
        serial: Serial,
        time: u32,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::keyboard::KeyboardTarget::key(
                surface.as_ref(),
                seat,
                data,
                key,
                state,
                serial,
                time,
            );
        }
    }
    fn modifiers(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        modifiers: ModifiersState,
        serial: Serial,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::keyboard::KeyboardTarget::modifiers(
                surface.as_ref(),
                seat,
                data,
                modifiers,
                serial,
            );
        }
    }
}

/// Forward drag-and-drop focus events (enter / motion / leave / drop) to the wrapped
/// target's underlying `wl_surface`, reusing the `WlSurface` offer-data type.
impl DndFocus<AppState> for FocusTarget {
    type OfferData<S: Source> = <WlSurface as DndFocus<AppState>>::OfferData<S>;

    fn enter<S: Source>(
        &self,
        data: &mut AppState,
        dh: &DisplayHandle,
        source: Arc<S>,
        seat: &Seat<AppState>,
        location: Point<f64, Logical>,
        serial: &Serial,
    ) -> Option<Self::OfferData<S>> {
        if let Some(surface) = self.wl_surface() {
            <WlSurface as DndFocus<AppState>>::enter(
                surface.as_ref(),
                data,
                dh,
                source,
                seat,
                location,
                serial,
            )
        } else {
            None
        }
    }

    fn motion<S: Source>(
        &self,
        data: &mut AppState,
        offer: Option<&mut Self::OfferData<S>>,
        seat: &Seat<AppState>,
        location: Point<f64, Logical>,
        time: u32,
    ) {
        if let Some(surface) = self.wl_surface() {
            <WlSurface as DndFocus<AppState>>::motion(
                surface.as_ref(),
                data,
                offer,
                seat,
                location,
                time,
            )
        }
    }

    fn leave<S: Source>(
        &self,
        data: &mut AppState,
        offer: Option<&mut Self::OfferData<S>>,
        seat: &Seat<AppState>,
    ) {
        if let Some(surface) = self.wl_surface() {
            <WlSurface as DndFocus<AppState>>::leave(surface.as_ref(), data, offer, seat)
        }
    }

    fn drop<S: Source>(
        &self,
        data: &mut AppState,
        offer: Option<&mut Self::OfferData<S>>,
        seat: &Seat<AppState>,
    ) {
        if let Some(surface) = self.wl_surface() {
            <WlSurface as DndFocus<AppState>>::drop(surface.as_ref(), data, offer, seat)
        }
    }
}

/// Forward every pointer event (motion, buttons, axis, and all swipe / pinch / hold gesture
/// phases) to the wrapped target's underlying `wl_surface`.
impl PointerTarget<AppState> for FocusTarget {
    fn enter(&self, seat: &Seat<AppState>, data: &mut AppState, event: &MotionEvent) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::enter(surface.as_ref(), seat, data, event);
        }
    }
    fn motion(&self, seat: &Seat<AppState>, data: &mut AppState, event: &MotionEvent) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::motion(surface.as_ref(), seat, data, event);
        }
    }
    fn relative_motion(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &RelativeMotionEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::relative_motion(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn button(&self, seat: &Seat<AppState>, data: &mut AppState, event: &ButtonEvent) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::button(surface.as_ref(), seat, data, event);
        }
    }
    fn axis(&self, seat: &Seat<AppState>, data: &mut AppState, frame: AxisFrame) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::axis(surface.as_ref(), seat, data, frame);
        }
    }
    fn frame(&self, seat: &Seat<AppState>, data: &mut AppState) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::frame(surface.as_ref(), seat, data);
        }
    }
    fn leave(&self, seat: &Seat<AppState>, data: &mut AppState, serial: Serial, time: u32) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::leave(
                surface.as_ref(),
                seat,
                data,
                serial,
                time,
            );
        }
    }
    fn gesture_swipe_begin(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GestureSwipeBeginEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_swipe_begin(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_swipe_update(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GestureSwipeUpdateEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_swipe_update(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_swipe_end(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GestureSwipeEndEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_swipe_end(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_pinch_begin(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GesturePinchBeginEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_pinch_begin(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_pinch_update(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GesturePinchUpdateEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_pinch_update(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_pinch_end(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GesturePinchEndEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_pinch_end(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_hold_begin(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GestureHoldBeginEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_hold_begin(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
    fn gesture_hold_end(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &GestureHoldEndEvent,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::pointer::PointerTarget::gesture_hold_end(
                surface.as_ref(),
                seat,
                data,
                event,
            );
        }
    }
}

/// Forward every touch event (down / up / motion / frame / cancel / shape / orientation) to
/// the wrapped target's underlying `wl_surface`.
impl TouchTarget<AppState> for FocusTarget {
    fn down(&self, seat: &Seat<AppState>, data: &mut AppState, event: &DownEvent, serial: Serial) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::down(surface.as_ref(), seat, data, event, serial);
        }
    }
    fn up(&self, seat: &Seat<AppState>, data: &mut AppState, event: &UpEvent, serial: Serial) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::up(surface.as_ref(), seat, data, event, serial);
        }
    }
    fn motion(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &smithay::input::touch::MotionEvent,
        serial: Serial,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::motion(
                surface.as_ref(),
                seat,
                data,
                event,
                serial,
            );
        }
    }
    fn frame(&self, seat: &Seat<AppState>, data: &mut AppState, serial: Serial) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::frame(surface.as_ref(), seat, data, serial);
        }
    }
    fn cancel(&self, seat: &Seat<AppState>, data: &mut AppState, serial: Serial) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::cancel(surface.as_ref(), seat, data, serial);
        }
    }
    fn shape(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &ShapeEvent,
        serial: Serial,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::shape(surface.as_ref(), seat, data, event, serial);
        }
    }
    fn orientation(
        &self,
        seat: &Seat<AppState>,
        data: &mut AppState,
        event: &OrientationEvent,
        serial: Serial,
    ) {
        if let Some(surface) = self.wl_surface() {
            smithay::input::touch::TouchTarget::orientation(
                surface.as_ref(),
                seat,
                data,
                event,
                serial,
            );
        }
    }
}

/// Map a Smithay `CursorIcon` to its CSS cursor-name string, used both for themed-cursor
/// lookup and for the name handed to the Python cursor callback; unknown icons fall back to
/// `"default"`.
pub fn cursor_icon_to_str(icon: &CursorIcon) -> &'static str {
    match icon {
        CursorIcon::Default => "default",
        CursorIcon::ContextMenu => "context-menu",
        CursorIcon::Help => "help",
        CursorIcon::Pointer => "pointer",
        CursorIcon::Progress => "progress",
        CursorIcon::Wait => "wait",
        CursorIcon::Cell => "cell",
        CursorIcon::Crosshair => "crosshair",
        CursorIcon::Text => "text",
        CursorIcon::VerticalText => "vertical-text",
        CursorIcon::Alias => "alias",
        CursorIcon::Copy => "copy",
        CursorIcon::Move => "move",
        CursorIcon::NoDrop => "no-drop",
        CursorIcon::NotAllowed => "not-allowed",
        CursorIcon::Grab => "grab",
        CursorIcon::Grabbing => "grabbing",
        CursorIcon::AllScroll => "all-scroll",
        CursorIcon::ColResize => "col-resize",
        CursorIcon::RowResize => "row-resize",
        CursorIcon::NResize => "n-resize",
        CursorIcon::EResize => "e-resize",
        CursorIcon::SResize => "s-resize",
        CursorIcon::WResize => "w-resize",
        CursorIcon::NeResize => "ne-resize",
        CursorIcon::NwResize => "nw-resize",
        CursorIcon::SeResize => "se-resize",
        CursorIcon::SwResize => "sw-resize",
        CursorIcon::EwResize => "ew-resize",
        CursorIcon::NsResize => "ns-resize",
        CursorIcon::NeswResize => "nesw-resize",
        CursorIcon::NwseResize => "nwse-resize",
        CursorIcon::ZoomIn => "zoom-in",
        CursorIcon::ZoomOut => "zoom-out",
        _ => "default",
    }
}

/// Seat wiring: declares the keyboard / pointer / touch focus types and reacts to cursor
/// changes and keyboard-focus moves.
impl SeatHandler for AppState {
    type KeyboardFocus = FocusTarget;
    type PointerFocus = FocusTarget;
    type TouchFocus = FocusTarget;
    fn seat_state(&mut self) -> &mut SeatState<AppState> {
        &mut self.seat_state
    }

    /// A client requested a cursor change (named, hidden, or surface-backed): retain it as
    /// the current icon and forward it to the Python cursor callback.
    fn cursor_image(&mut self, _seat: &Seat<AppState>, image: CursorImageStatus) {
        self.current_cursor_icon = Some(image.clone());
        self.send_cursor_image(&image);
    }

    /// Keep the primary selection's focus following keyboard focus, so middle-click paste
    /// targets the currently focused client (or clears it when nothing is focused).
    fn focus_changed(&mut self, seat: &Seat<AppState>, focus: Option<&Self::KeyboardFocus>) {
        if let Some(focus_target) = focus {
            let dh = &self.dh;
            let client = focus_target.wl_surface().and_then(|s| dh.get_client(s.id()).ok());
            set_primary_focus(dh, seat, client);
        } else {
            let dh = &self.dh;
            set_primary_focus(dh, seat, None);
        }
    }
}

/// Pointer-warp protocol: lets a client teleport the pointer to a surface-local position
/// (games / remote-desktop style warps).
///
/// `warp_pointer` locates the requesting surface's origin in the global space, adds the requested
/// surface-local offset to get a global position, recomputes what element lies under it, and emits
/// a synthetic motion event so focus and enter/leave follow the warp.
impl PointerWarpHandler for AppState {
    fn warp_pointer(
        &mut self,
        surface: WlSurface,
        _pointer: WlPointer,
        pos: Point<f64, Logical>,
        serial: Serial,
    ) {
        let surface_origin = self.space.elements().find_map(|window| {
            if window.wl_surface().as_deref() == Some(&surface) {
                self.space.element_location(window)
            } else {
                None
            }
        });

        if let Some(origin) = surface_origin {
            let global_pos = origin.to_f64() + pos;
            let time = wayland_time();

            if let Some(pointer) = self.seat.get_pointer() {
                let under = self.space.element_under(global_pos).map(|(w, loc)| {
                    (FocusTarget::Window(w.clone()), loc.to_f64())
                });
                
                pointer.motion(
                    self,
                    under,
                    &MotionEvent {
                        location: global_pos,
                        serial, 
                        time,
                    },
                );
            }
        }
    }
}

/// xdg-shell protocol: toplevel and popup lifecycle.
impl XdgShellHandler for AppState {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.shell_state
    }
    /// A new toplevel appears: wrap it in a `Window`, queue it for mapping, and register a
    /// foreign-toplevel handle (seeded with title / app-id) stored on the surface for later updates.
    fn new_toplevel(&mut self, surface: ToplevelSurface) {
        let window = Window::new_wayland_window(surface.clone());
        self.pending_windows.push(window);
        let (title, app_id) = with_states(surface.wl_surface(), |states| {
            let attributes = states.data_map.get::<XdgToplevelSurfaceData>().unwrap().lock().unwrap();
            (attributes.title.clone(), attributes.app_id.clone())
        });

        let handle = self.foreign_toplevel_list.new_toplevel::<AppState>(title.unwrap_or_default(), app_id.unwrap_or_default());
        
        with_states(surface.wl_surface(), |states| states.data_map.insert_if_missing(|| handle));
    }
    /// Register a new popup (menu, tooltip, combo-box list) with the `PopupManager` so it
    /// takes part in grab and dismissal handling, then send the initial configure xdg-shell requires
    /// before the client is allowed to draw it.
    fn new_popup(&mut self, surface: PopupSurface, _positioner: PositionerState) {
        if let Err(err) = self.popups.track_popup(PopupKind::Xdg(surface.clone())) {
            eprintln!("Failed to track popup: {:?}", err);
        }
        let _ = surface.send_configure();
    }
    /// Popup grab: find the popup's root surface and its window, then install a popup grab so
    /// dismissal and pointer routing behave correctly.
    fn grab(
        &mut self,
        surface: PopupSurface,
        _seat: smithay::reexports::wayland_server::protocol::wl_seat::WlSeat,
        serial: Serial,
    ) {
        let kind = PopupKind::Xdg(surface);
        if let Ok(root_surface) = smithay::desktop::find_popup_root_surface(&kind) {
            if let Some(window) = self.space.elements().find(|w| w.wl_surface().as_deref() == Some(&root_surface)).cloned() {
                let _ = self.popups.grab_popup(FocusTarget::Window(window), kind, &self.seat, serial);
            }
        }
    }
    /// Re-track a popup whose position changed (e.g. a submenu flipping sides to stay
    /// on-screen) so the `PopupManager` follows its new geometry, then echo the client's reposition
    /// token back to confirm the move took effect.
    fn reposition_request(
        &mut self,
        surface: PopupSurface,
        _positioner: PositionerState,
        token: u32,
    ) {
        if let Err(err) = self.popups.track_popup(PopupKind::Xdg(surface.clone())) {
            eprintln!("Failed to track popup: {:?}", err);
        }
        let _ = surface.send_repositioned(token);
    }
    /// A toplevel closed: drop it from the pending-window queue so a window that never
    /// finished mapping can't linger there, and remove its foreign-toplevel handle so taskbar-style
    /// clients stop listing a window that is gone.
    fn toplevel_destroyed(&mut self, surface: ToplevelSurface) {
        if let Some(idx) = self.pending_windows.iter().position(|w| w.toplevel().map(|t| *t == surface).unwrap_or(false)) {
            self.pending_windows.remove(idx);
        }
        if let Some(handle) = with_states(surface.wl_surface(), |states| states.data_map.get::<ForeignToplevelHandle>().cloned()) {
             self.foreign_toplevel_list.remove_toplevel(&handle);
        }
    }
}

/// Per-client data attached to every Wayland client connection; holds the compositor's
/// per-client surface state.
#[derive(Default)]
pub struct ClientState {
    pub compositor_state: CompositorClientState,
}
/// Client lifecycle hooks; connect and disconnect need no bookkeeping here.
impl ClientData for ClientState {
    fn initialized(&self, _client_id: ClientId) {}
    fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
}

delegate_compositor!(AppState);
delegate_shm!(AppState);
delegate_output!(AppState);
delegate_seat!(AppState);
delegate_xdg_shell!(AppState);
delegate_dmabuf!(AppState);
delegate_fractional_scale!(AppState);
delegate_virtual_keyboard_manager!(AppState);
delegate_data_device!(AppState);
delegate_data_control!(AppState);
delegate_pointer_warp!(AppState);
delegate_relative_pointer!(AppState);
delegate_pointer_constraints!(AppState);
delegate_foreign_toplevel_list!(AppState);
delegate_xdg_decoration!(AppState);
delegate_layer_shell!(AppState);
delegate_single_pixel_buffer!(AppState);
delegate_viewporter!(AppState);
delegate_presentation!(AppState);
delegate_xdg_activation!(AppState);
delegate_primary_selection!(AppState);

/// Row stride (bytes) of a tightly-mapped RGBA8 GPU readback, derived from the mapping
/// length rather than assuming `width*4`.
///
/// Dividing the buffer length by the height recovers a padded stride, so a padded readback cannot
/// skew the cursor image; the result never drops below one full `width*4` row, and a zero height
/// short-circuits to one row to avoid dividing by zero.
fn rgba_readback_stride(buf_len: usize, height: usize, width: usize) -> usize {
    let row = width.saturating_mul(4);
    if height == 0 {
        return row;
    }
    (buf_len / height).max(row)
}

#[cfg(test)]
mod stride_tests {
    use super::rgba_readback_stride;

    /// A tightly-packed readback yields exactly `width*4` stride.
    #[test]
    fn packed_readback_is_width_times_four() {
        assert_eq!(rgba_readback_stride(64 * 4 * 48, 48, 64), 64 * 4);
        assert_eq!(rgba_readback_stride(3 * 4 * 2, 2, 3), 12);
    }

    /// A padded readback (extra bytes per row) recovers the true padded stride from the
    /// buffer length.
    #[test]
    fn padded_readback_recovers_real_stride() {
        let (w, h, pad) = (3usize, 2usize, 4usize);
        let stride = w * 4 + pad;
        assert_eq!(rgba_readback_stride(stride * h, h, w), stride);
    }

    /// Extracting pixels with the recovered stride from a padded buffer reproduces the
    /// written values with no row-to-row skew.
    #[test]
    fn padded_extraction_has_no_skew() {
        let (w, h) = (3usize, 2usize);
        let stride = 16usize;
        let mut buf = vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                let o = y * stride + x * 4;
                buf[o] = x as u8 + 1;
                buf[o + 1] = y as u8 + 1;
            }
        }
        let s = rgba_readback_stride(buf.len(), h, w);
        assert_eq!(s, stride);
        for y in 0..h {
            for x in 0..w {
                let o = y * s + x * 4;
                assert_eq!(buf[o], x as u8 + 1);
                assert_eq!(buf[o + 1], y as u8 + 1);
            }
        }
    }

    /// Zero height returns one full row instead of dividing by zero.
    #[test]
    fn zero_height_no_divide_by_zero() {
        assert_eq!(rgba_readback_stride(0, 0, 10), 40);
    }

    /// A buffer shorter than a single row still reports a full `width*4` row.
    #[test]
    fn truncated_buffer_keeps_full_row() {
        assert_eq!(rgba_readback_stride(10, 5, 64), 64 * 4);
    }
}
