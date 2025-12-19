use std::borrow::Cow;
use std::fs::File;
use std::io::Cursor as IoCursor;
use std::time::Instant;

use gbm::{BufferObject, Device as RawGbmDevice};
use image::{ImageBuffer, ImageFormat, Rgba};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::renderer::{
    gles::GlesRenderer, pixman::PixmanRenderer, ImportDma,
};
use crate::wayland::cursor::Cursor;

use smithay::{
    delegate_compositor, delegate_data_device, delegate_dmabuf, delegate_fractional_scale,
    delegate_output, delegate_seat, delegate_shm, delegate_virtual_keyboard_manager,
    delegate_xdg_shell,
    desktop::{Space, Window},
    input::{
        keyboard::{KeyboardTarget, KeysymHandle, ModifiersState},
        pointer::{
            AxisFrame, ButtonEvent, CursorIcon, CursorImageStatus, GestureHoldBeginEvent,
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
            Client, DisplayHandle,
        },
    },
    utils::{Clock, IsAlive, Monotonic, Serial},
    wayland::{
        buffer::BufferHandler,
        compositor::{
            with_states, BufferAssignment, CompositorClientState, CompositorHandler,
            CompositorState, SurfaceAttributes,
        },
        dmabuf::{DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier},
        fractional_scale::{FractionalScaleHandler, FractionalScaleManagerState},
        output::{OutputHandler, OutputManagerState},
        seat::WaylandFocus,
        selection::{
            data_device::{
                ClientDndGrabHandler, DataDeviceHandler, DataDeviceState, ServerDndGrabHandler,
            },
            SelectionHandler,
        },
        shell::xdg::{
            PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
            XdgToplevelSurfaceData,
        },
        shm::{with_buffer_contents, ShmHandler, ShmState},
        virtual_keyboard::VirtualKeyboardManagerState,
    },
};

use crate::encoders::overlay::OverlayState;
use crate::encoders::vaapi::VaapiEncoder;
use crate::nvenc::NvencEncoder;
use crate::{RustCaptureSettings, StripeState};

/// @brief Enum wrapper for supported GPU hardware encoders.
pub enum GpuEncoder {
    Vaapi(VaapiEncoder),
    Nvenc(NvencEncoder),
}

/// @brief Global application state holding Wayland globals, renderer resources, and capture state.
///
/// This struct acts as the central context passed to all Smithay handlers. It manages
/// the lifecycle of the Wayland compositor, hardware acceleration contexts (GBM/EGL),
/// and the encoding pipeline state.
pub struct AppState {
    pub compositor_state: CompositorState,
    pub fractional_scale_state: FractionalScaleManagerState,
    pub shm_state: ShmState,
    pub dmabuf_state: DmabufState,
    pub dmabuf_global: Option<DmabufGlobal>,
    #[allow(dead_code)]
    pub output_state: OutputManagerState,
    pub seat_state: SeatState<AppState>,
    pub shell_state: XdgShellState,
    pub space: Space<Window>,
    pub data_device_state: DataDeviceState,
    pub dh: DisplayHandle,
    #[allow(dead_code)]
    pub seat: Seat<AppState>,
    pub outputs: Vec<Output>,
    pub pending_windows: Vec<Window>,

    pub frame_buffer: Vec<u8>,
    pub nv12_buffer: Vec<u8>,

    pub gles_renderer: Option<GlesRenderer>,
    pub pixman_renderer: Option<PixmanRenderer>,

    pub gbm_device: Option<RawGbmDevice<File>>,
    pub offscreen_buffer: Option<(BufferObject<()>, Dmabuf)>,

    pub is_capturing: bool,
    pub settings: RustCaptureSettings,
    pub callback: Option<Py<PyAny>>,
    pub cursor_callback: Option<Py<PyAny>>,
    pub stripes: Vec<StripeState>,

    pub last_log_time: Instant,
    pub encoded_frame_count: u32,
    pub total_stripes_encoded: u32,
    pub start_time: Instant,
    pub clock: Clock<Monotonic>,

    pub frame_counter: u16,
    pub use_gpu: bool,

    pub video_encoder: Option<GpuEncoder>,
    pub vaapi_state: StripeState,
    pub cursor_helper: Cursor,

    pub overlay_state: OverlayState,

    pub virtual_keyboard_state: VirtualKeyboardManagerState,

    pub current_cursor_icon: Option<CursorImageStatus>,
    pub render_cursor_on_framebuffer: bool,
}

/// @brief Handler for core compositor events like surface creation and commits.
impl CompositorHandler for AppState {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }
    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client.get_data::<ClientState>().unwrap().compositor_state
    }

    /// @brief Called when a client commits a buffer to a surface.
    ///
    /// This function is responsible for:
    /// 1. Triggering Smithay's internal buffer management.
    /// 2. Detecting if a new window (Toplevel) is ready to be mapped (shown).
    /// 3. Sending the initial configuration (resolution, state) to new windows.
    /// 4. Setting initial focus for keyboard/mouse when a window appears.
    fn commit(&mut self, surface: &WlSurface) {
        smithay::backend::renderer::utils::on_commit_buffer_handler::<Self>(surface);

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

                if let Some(output) = self.outputs.first() {
                    output.enter(surface);
                }

                let serial = Serial::from(self.clock.now().as_millis() as u32);
                let target = FocusTarget(window.clone());
                if let Some(keyboard) = self.seat.get_keyboard() {
                    keyboard.set_focus(self, Some(target.clone()), serial);
                }
            }
        }
    }
}

// --- Boilerplate implementations for various Wayland protocols ---

impl SelectionHandler for AppState {
    type SelectionUserData = ();
}
impl DataDeviceHandler for AppState {
    fn data_device_state(&self) -> &DataDeviceState {
        &self.data_device_state
    }
}
impl ClientDndGrabHandler for AppState {}
impl ServerDndGrabHandler for AppState {}
impl BufferHandler for AppState {
    fn buffer_destroyed(&mut self, _buffer: &WlBuffer) {}
}
impl ShmHandler for AppState {
    fn shm_state(&self) -> &ShmState {
        &self.shm_state
    }
}
impl OutputHandler for AppState {}

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

impl FractionalScaleHandler for AppState {
    fn new_fractional_scale(
        &mut self,
        _surface: smithay::reexports::wayland_server::protocol::wl_surface::WlSurface,
    ) {
    }
}

/// @brief A wrapper around a generic Window that implements input handling traits.
///
/// Smithay requires a specific struct to represent the "target" of an input event
/// (mouse, keyboard, touch). This struct bridges the gap between the abstract
/// input event and the concrete Wayland surface contained within a `Window`.
#[derive(Debug, Clone, PartialEq)]
pub struct FocusTarget(pub Window);

impl IsAlive for FocusTarget {
    fn alive(&self) -> bool {
        self.0.alive()
    }
}

impl WaylandFocus for FocusTarget {
    fn wl_surface(&self) -> Option<Cow<'_, WlSurface>> {
        self.0.wl_surface()
    }
    fn same_client_as(&self, object_id: &ObjectId) -> bool {
        self.0.same_client_as(object_id)
    }
}

/// @brief Routes keyboard events to the underlying Wayland surface.
///
/// When a key is pressed, this implementation ensures the event is serialized
/// into the Wayland protocol and sent to the client that owns the focused window.
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

/// @brief Routes pointer (mouse) events to the underlying Wayland surface.
///
/// This handles motion, clicks, scrolling (axis), and gestures. It delegates
/// the actual protocol generation to `smithay::input::pointer::PointerTarget`.
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

/// @brief Routes touch events (down, up, motion) to the underlying Wayland surface.
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

/// @brief Handles general seat operations, focusing primarily on cursor updates.
impl SeatHandler for AppState {
    type KeyboardFocus = FocusTarget;
    type PointerFocus = FocusTarget;
    type TouchFocus = FocusTarget;
    fn seat_state(&mut self) -> &mut SeatState<AppState> {
        &mut self.seat_state
    }

    /// @brief Called when the client requests a cursor change (e.g., hover over text).
    ///
    /// This method extracts the cursor image—either loading a system icon by name
    /// or converting a client-provided SHM buffer to PNG—and sends it to the
    /// Python layer to be forwarded to the web client.
    fn cursor_image(&mut self, _seat: &Seat<AppState>, image: CursorImageStatus) {
        self.current_cursor_icon = Some(image.clone());

        if let Some(ref cb) = self.cursor_callback {
            let (msg_type, data, hot_x, hot_y) = match image {
                CursorImageStatus::Named(icon) => {
                    let name = cursor_icon_to_str(&icon);
                    if let Some((png_bytes, x, y)) = self.cursor_helper.get_png_data(name) {
                        ("png", png_bytes, x, y)
                    } else {
                        ("error", Vec::new(), 0, 0)
                    }
                }
                CursorImageStatus::Hidden => ("hide", Vec::new(), 0, 0),
                CursorImageStatus::Surface(ref surface) => {
                    let mut png_data = Vec::new();
                    let hot_x = 0;
                    let hot_y = 0;

                    let buffer = with_states(surface, |states| {
                        states
                            .cached_state
                            .get::<SurfaceAttributes>()
                            .current()
                            .buffer
                            .as_ref()
                            .and_then(|b| match b {
                                BufferAssignment::NewBuffer(buffer) => Some(buffer.clone()),
                                _ => None,
                            })
                    });

                    if let Some(buffer) = buffer {
                        let _ = with_buffer_contents(&buffer, |ptr, len, spec| {
                            let width = spec.width as u32;
                            let height = spec.height as u32;
                            let stride = spec.stride as usize;

                            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

                            let mut img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width, height);

                            for y in 0..height {
                                for x in 0..width {
                                    let offset = (y as usize * stride) + (x as usize * 4);
                                    if offset + 4 <= slice.len() {
                                        let b = slice[offset];
                                        let g = slice[offset + 1];
                                        let r = slice[offset + 2];
                                        let a = slice[offset + 3];

                                        img_buf.put_pixel(x, y, Rgba([r, g, b, a]));
                                    }
                                }
                            }

                            let mut bytes = Vec::new();
                            let mut cursor = IoCursor::new(&mut bytes);
                            if img_buf.write_to(&mut cursor, ImageFormat::Png).is_ok() {
                                png_data = bytes;
                            }
                        });
                    }

                    if !png_data.is_empty() {
                        ("png", png_data, hot_x, hot_y)
                    } else {
                        ("surface", Vec::new(), 0, 0)
                    }
                }
            };

            if !data.is_empty() || msg_type == "hide" {
                #[allow(deprecated)]
                Python::with_gil(|py| {
                    let py_bytes = PyBytes::new(py, &data);
                    if let Err(e) = cb.call1(py, (msg_type, py_bytes, hot_x, hot_y)) {
                        eprintln!("Cursor Callback error: {:?}", e);
                    }
                });
            }
        }
    }

    fn focus_changed(&mut self, _seat: &Seat<AppState>, _focus: Option<&Self::KeyboardFocus>) {}
}

/// @brief Manages XDG Shell events (application windows).
impl XdgShellHandler for AppState {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.shell_state
    }
    /// @brief Called when a client creates a new top-level window.
    fn new_toplevel(&mut self, surface: ToplevelSurface) {
        let window = Window::new_wayland_window(surface.clone());
        self.pending_windows.push(window);
    }
    fn new_popup(&mut self, surface: PopupSurface, _positioner: PositionerState) {
        surface.send_configure().unwrap();
    }
    fn grab(
        &mut self,
        _surface: PopupSurface,
        _seat: smithay::reexports::wayland_server::protocol::wl_seat::WlSeat,
        _serial: Serial,
    ) {
    }
    fn reposition_request(
        &mut self,
        _surface: PopupSurface,
        _positioner: PositionerState,
        _token: u32,
    ) {
    }
}

#[derive(Default)]
pub struct ClientState {
    pub compositor_state: CompositorClientState,
}
impl ClientData for ClientState {
    fn initialized(&self, _client_id: ClientId) {}
    fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
}

// Delegate macros wire up Smithay's internal event dispatching to the AppState struct.
delegate_compositor!(AppState);
delegate_shm!(AppState);
delegate_output!(AppState);
delegate_seat!(AppState);
delegate_xdg_shell!(AppState);
delegate_dmabuf!(AppState);
delegate_fractional_scale!(AppState);
delegate_virtual_keyboard_manager!(AppState);
delegate_data_device!(AppState);
