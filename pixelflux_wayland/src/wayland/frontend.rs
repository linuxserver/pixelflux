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
                DataDeviceHandler, DataDeviceState, WaylandDndGrabHandler,
            },
            SelectionHandler,
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

pub fn next_serial() -> Serial {
    Serial::from(SERIAL_COUNTER.fetch_add(1, Ordering::SeqCst))
}

pub fn wayland_time() -> u32 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    (ts.tv_sec as u32).wrapping_mul(1000).wrapping_add((ts.tv_nsec as u32) / 1_000_000)
}

pub fn wayland_utime() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    (ts.tv_sec as u64).wrapping_mul(1_000_000).wrapping_add((ts.tv_nsec as u64) / 1_000)
}

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
    pub cursor_buffer: Option<WlBuffer>,
    pub cursor_cache: std::collections::HashMap<u64, Vec<u8>>,
    pub render_cursor_on_framebuffer: bool,
    pub pointer_warp_state: PointerWarpManager,
    pub relative_pointer_state: RelativePointerManagerState,
    pub pointer_constraints_state: PointerConstraintsState,
}

impl PointerConstraintsHandler for AppState {
    fn new_constraint(&mut self, _surface: &WlSurface, _pointer: &PointerHandle<Self>) {}

    fn cursor_position_hint(
        &mut self,
        _surface: &WlSurface,
        _pointer: &PointerHandle<Self>,
        _location: Point<f64, Logical>,
    ) {}
}

impl ForeignToplevelListHandler for AppState {
    fn foreign_toplevel_list_state(&mut self) -> &mut ForeignToplevelListState {
        &mut self.foreign_toplevel_list
    }
}

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

impl PrimarySelectionHandler for AppState {
    fn primary_selection_state(&mut self) -> &mut PrimarySelectionState {
        &mut self.primary_selection_state
    }
}

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

                if let Some(output) = self.outputs.first() {
                    output.enter(surface);
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


/// @brief Helper implementations for the global application state.
impl AppState {
    /// @brief Resolves the cursor state into image data and sends it to the Python layer.
    ///
    /// This method accepts a `CursorImageStatus` (Named, Hidden, or Surface), extracts
    /// the relevant pixel data (checking the hash cache for surfaces to avoid re-encoding),
    /// and outputs the final PNG bytes and hotspot coordinates to the registered Python callback.
    fn send_cursor_image(&mut self, image: &CursorImageStatus) {
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
                            slice.hash(&mut hasher);
                            let hash = hasher.finish();
                            (hash, spec.width, spec.height, spec.stride, slice.to_vec())
                        });

                        match shm_result {
                            Ok((hash, width, height, stride, raw_bytes)) => {
                                if let Some(cached_png) = self.cursor_cache.get(&hash) {
                                    final_png = cached_png.clone();
                                } else {
                                    if width <= 128 && height <= 128 && !raw_bytes.is_empty() {
                                        let mut img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width as u32, height as u32);
                                        let stride_usize = stride as usize;
                                        
                                        for y in 0..(height as u32) {
                                            for x in 0..(width as u32) {
                                                let offset = (y as usize * stride_usize) + (x as usize * 4);
                                                if offset + 4 <= raw_bytes.len() {
                                                    img_buf.put_pixel(x, y, Rgba([
                                                        raw_bytes[offset + 2], 
                                                        raw_bytes[offset + 1], 
                                                        raw_bytes[offset], 
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
                                                self.cursor_cache.clear();
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
                                            Ok(mut frame) => {
                                                let rect = Rectangle::new((0, 0).into(), (width, height).into());
                                                
                                                match renderer.copy_framebuffer(&mut frame, rect, Fourcc::Abgr8888) {
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
                                             let stride_usize = (width * 4) as usize;
                                             
                                             for y in 0..(height as u32) {
                                                 for x in 0..(width as u32) {
                                                     let offset = (y as usize * stride_usize) + (x as usize * 4);
                                                     if offset + 4 <= raw_bytes.len() {
                                                         img_buf.put_pixel(x, y, Rgba([
                                                             raw_bytes[offset],     // R
                                                             raw_bytes[offset + 1], // G
                                                             raw_bytes[offset + 2], // B
                                                             raw_bytes[offset + 3]  // A
                                                         ]));
                                                     }
                                                 }
                                             }

                                             let mut bytes = Vec::new();
                                             if img_buf.write_to(&mut IoCursor::new(&mut bytes), ImageFormat::Png).is_ok() {
                                                 self.cursor_cache.insert(hash, bytes.clone());
                                                 final_png = bytes;
                                                 if self.cursor_cache.len() > 100 {
                                                     self.cursor_cache.clear();
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

            if !data.is_empty() || msg_type == "hide" || msg_type == "surface" {
                #[allow(deprecated)]
                Python::with_gil(|py| {
                    let py_bytes = PyBytes::new(py, &data);
                    let _ = cb.call1(py, (msg_type, py_bytes, hot_x, hot_y));
                });
            }
        }
    }
}

impl SelectionHandler for AppState {
    type SelectionUserData = ();
}

impl DataDeviceHandler for AppState {
    fn data_device_state(&mut self) -> &mut DataDeviceState {
        &mut self.data_device_state
    }
}
impl DataControlHandler for AppState {
    fn data_control_state(&mut self) -> &mut DataControlState {
        &mut self.data_control_state
    }
}
impl WaylandDndGrabHandler for AppState {}
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
pub enum FocusTarget {
    Window(Window),
    Popup(PopupKind),
    LayerSurface(DesktopLayerSurface),
}

impl From<Window> for FocusTarget {
    fn from(w: Window) -> Self { FocusTarget::Window(w) }
}

impl From<PopupKind> for FocusTarget {
    fn from(p: PopupKind) -> Self { FocusTarget::Popup(p) }
}

impl From<DesktopLayerSurface> for FocusTarget {
    fn from(l: DesktopLayerSurface) -> Self { FocusTarget::LayerSurface(l) }
}

impl IsAlive for FocusTarget {
    fn alive(&self) -> bool {
        match self {
            FocusTarget::Window(w) => w.alive(),
            FocusTarget::Popup(p) => p.alive(),
            FocusTarget::LayerSurface(l) => l.alive(),
        }
    }
}

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

/// @brief Routes Drag'n'Drop events to the underlying Wayland surface.
///
/// This delegates DnD operations (enter, motion, leave, drop) to the specific
/// Wayland surface, allowing clients to negotiate data transfers (like file drops
/// or text copy/paste) via the Wayland protocol.
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
    fn cursor_image(&mut self, _seat: &Seat<AppState>, image: CursorImageStatus) {
        self.current_cursor_icon = Some(image.clone());
        self.send_cursor_image(&image);
    }

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

/// @brief Handler for pointer warp requests, enabling clients to reset the cursor position.
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

/// @brief Manages XDG Shell events (application windows).
impl XdgShellHandler for AppState {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.shell_state
    }
    /// @brief Called when a client creates a new top-level window.
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
    fn new_popup(&mut self, surface: PopupSurface, _positioner: PositionerState) {
        if let Err(err) = self.popups.track_popup(PopupKind::Xdg(surface.clone())) {
            eprintln!("Failed to track popup: {:?}", err);
        }
        let _ = surface.send_configure();
    }
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
    fn toplevel_destroyed(&mut self, surface: ToplevelSurface) {
        if let Some(idx) = self.pending_windows.iter().position(|w| w.toplevel().map(|t| *t == surface).unwrap_or(false)) {
            self.pending_windows.remove(idx);
        }
        if let Some(handle) = with_states(surface.wl_surface(), |states| states.data_map.get::<ForeignToplevelHandle>().cloned()) {
             self.foreign_toplevel_list.remove_toplevel(&handle);
        }
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
