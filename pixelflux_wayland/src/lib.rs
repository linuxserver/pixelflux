/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/*
  ▘    ▜ ▐▘▜     
▛▌▌▚▘█▌▐ ▜▘▐ ▌▌▚▘
▙▌▌▞▖▙▖▐▖▐ ▐▖▙▌▞▖
▌                
*/

#![allow(dead_code)]

use std::fs::File;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use gbm::{BufferObject, BufferObjectFlags, Device as RawGbmDevice, Format as GbmFormat};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use yuv::{
    BufferStoreMut, YuvBiPlanarImageMut, YuvConversionMode, YuvRange, YuvStandardMatrix,
};

use smithay::wayland::single_pixel_buffer::SinglePixelBufferState;
use smithay::wayland::viewporter::ViewporterState;
use smithay::wayland::presentation::PresentationState;
use smithay::wayland::selection::wlr_data_control::DataControlState;
use smithay::{
    backend::{
        allocator::{
            dmabuf::{Dmabuf, DmabufFlags},
            gbm::GbmDevice,
            Fourcc, Modifier,
        },
        drm::DrmNode,
        egl::{EGLContext, EGLDisplay},
        input::{Axis, AxisSource, KeyState, Keycode},
        renderer::{
            damage::OutputDamageTracker,
            element::{
                memory::MemoryRenderBufferRenderElement,
                surface::WaylandSurfaceRenderElement,
                AsRenderElements, Wrap,
            },
            gles::GlesRenderer,
            pixman::PixmanRenderer,
            Bind, ImportAll, ImportEgl, ImportMem,
        },
    },
    desktop::{space::SpaceRenderElements, Space},
    input::{
        keyboard::{FilterResult, XkbConfig},
        pointer::{AxisFrame, ButtonEvent, CursorImageStatus, MotionEvent, RelativeMotionEvent},
        SeatState,
    },
    output::{Mode as OutputMode, Output, PhysicalProperties, Scale as OutputScale, Subpixel},
    reexports::{
        calloop::{
            channel::Event as CalloopEvent, generic::Generic, timer::{TimeoutAction, Timer},
            EventLoop, Interest, Mode, PostAction,
        },
        pixman,
        wayland_server::{Display, DisplayHandle},
    },
    utils::{Clock, Physical, Point, Rectangle, Scale, Transform},
    wayland::{
        compositor::{with_states, CompositorState},
        dmabuf::{DmabufFeedbackBuilder, DmabufState},
        fractional_scale::FractionalScaleManagerState,
        output::OutputManagerState,
        selection::data_device::DataDeviceState,
        seat::WaylandFocus,
        shell::xdg::XdgShellState,
        shm::ShmState,
        socket::ListeningSocketSource,
        virtual_keyboard::VirtualKeyboardManagerState,
        pointer_warp::PointerWarpManager,
        relative_pointer::RelativePointerManagerState,
        pointer_constraints::PointerConstraintsState,
        foreign_toplevel_list::ForeignToplevelListState,
        shell::xdg::decoration::XdgDecorationState,
    },
    desktop::{layer_map_for_output, PopupManager},
    wayland::shell::wlr_layer::WlrLayerShellState,
    wayland::xdg_activation::XdgActivationState,
    wayland::selection::primary_selection::PrimarySelectionState,
};

pub mod encoders {
    pub mod nvenc;
    pub mod overlay;
    pub mod software;
    pub mod vaapi;
}

pub mod wayland;
pub mod recording_sink;

pub use encoders::nvenc;
pub use encoders::software::StripeState;
pub use encoders::vaapi;

fn get_process_rss_bytes() -> usize {
    if let Ok(contents) = std::fs::read_to_string("/proc/self/statm") {
        if let Some(rss_pages) = contents.split_whitespace().nth(1) {
            if let Ok(pages) = rss_pages.parse::<usize>() {
                return pages * 4096;
            }
        }
    }
    0
}

fn get_shm_usage_bytes() -> u64 {
    let mut total_size = 0;
    if let Ok(entries) = std::fs::read_dir("/dev/shm") {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                total_size += metadata.len();
            }
        }
    }
    total_size
}

fn calculate_memory_threshold(width: i32, height: i32) -> usize {
    let frame_size = (width.max(0) as usize)
        .saturating_mul(height.max(0) as usize)
        .saturating_mul(4);
    let base_app_memory: usize = 300 * 1024 * 1024;
    let buffer_allowance = frame_size.saturating_mul(20);
    let min_threshold: usize = 1536 * 1024 * 1024;
    base_app_memory
        .saturating_add(buffer_allowance)
        .max(min_threshold)
}

use encoders::nvenc::NvencEncoder;
use encoders::overlay::OverlayState;
use encoders::software::MAX_STRIPE_CAPACITY;
use encoders::vaapi::VaapiEncoder;

use wayland::cursor::Cursor;
use wayland::frontend::{AppState, ClientState, FocusTarget, GpuEncoder, next_serial, wayland_time, wayland_utime};

smithay::backend::renderer::element::render_elements! {
    pub CompositionElements<R, E> where R: ImportAll + ImportMem;
    Space=SpaceRenderElements<R, E>,
    Window=Wrap<E>,
    Cursor=MemoryRenderBufferRenderElement<R>,
    Surface=WaylandSurfaceRenderElement<R>,
}

/// @brief Helper to convert a GBM Buffer Object (GPU memory) into a DMABUF.
fn create_dmabuf_from_bo(bo: &BufferObject<()>) -> Dmabuf {
    let fd = bo.fd().expect("Failed to get FD from GBM BO");
    let modifier = bo.modifier().expect("Failed to get modifier");
    let stride = bo.stride().expect("Failed to get stride");
    let width = bo.width().expect("Failed to get width");
    let height = bo.height().expect("Failed to get height");

    let drm_modifier = Modifier::from(Into::<u64>::into(modifier));

    let mut builder = Dmabuf::builder(
        (width as i32, height as i32),
        Fourcc::Argb8888,
        drm_modifier,
        DmabufFlags::empty(),
    );

    builder.add_plane(fd, 0, 0, stride);
    builder.build().expect("Failed to build Dmabuf from GBM BO")
}

#[derive(Clone, Debug, PartialEq)]
pub struct RustCaptureSettings {
    pub width: i32,
    pub height: i32,
    pub scale: f64,
    pub capture_x: i32,
    pub capture_y: i32,
    pub target_fps: f64,
    pub jpeg_quality: i32,
    pub paint_over_jpeg_quality: i32,
    pub use_paint_over_quality: bool,
    pub paint_over_trigger_frames: u32,
    pub damage_block_threshold: u32,
    pub damage_block_duration: u32,
    pub output_mode: i32,
    pub h264_crf: i32,
    pub h264_paintover_crf: i32,
    pub h264_paintover_burst_frames: i32,
    pub h264_fullcolor: bool,
    pub h264_fullframe: bool,
    pub h264_streaming_mode: bool,
    pub capture_cursor: bool,
    pub watermark_path: String,
    pub watermark_location_enum: i32,
    pub vaapi_render_node_index: i32,
    pub use_cpu: bool,
    pub debug_logging: bool,
    pub recording_socket: String,
    // true => emit raw encoded payload without the per-stripe header (matches X11 omit_stripe_headers).
    pub omit_stripe_headers: bool,
}

impl Default for RustCaptureSettings {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 768,
            scale: 1.0,
            capture_x: 0,
            capture_y: 0,
            target_fps: 60.0,
            jpeg_quality: 75,
            paint_over_jpeg_quality: 95,
            use_paint_over_quality: true,
            paint_over_trigger_frames: 15,
            damage_block_threshold: 10,
            damage_block_duration: 30,
            output_mode: 0,
            h264_crf: 25,
            h264_paintover_crf: 18,
            h264_paintover_burst_frames: 5,
            h264_fullcolor: false,
            h264_fullframe: false,
            h264_streaming_mode: false,
            capture_cursor: false,
            watermark_path: String::new(),
            watermark_location_enum: 0,
            vaapi_render_node_index: -1,
            use_cpu: false,
            debug_logging: false,
            recording_socket: String::new(),
            omit_stripe_headers: false,
        }
    }
}

pub enum ThreadCommand {
    StartCapture(Py<PyAny>, RustCaptureSettings),
    StopCapture,
    SetCursorCallback(Py<PyAny>),
    KeyboardKey { scancode: u32, state: u32 },
    // Inject by X11/XKB keysym, resolved to a keycode (+ shift level) against our own
    // smithay xkb keymap. See the KeyboardKeysym handler.
    KeyboardKeysym { keysym: u32, state: u32 },
    // Reply with the smithay keyboard's keymap as an XKB_KEYMAP_FORMAT_TEXT_V1 string so a
    // consumer (selkies) can build its reverse keysym map from the IDENTICAL keymap.
    GetXkbKeymap { reply: std::sync::mpsc::Sender<String> },
    PointerMotion { x: f64, y: f64 },
    PointerRelativeMotion { dx: f64, dy: f64 },
    PointerButton { btn: u32, state: u32 },
    PointerAxis { x: f64, y: f64 },
    UpdateCursorConfig { render_on_framebuffer: bool },
    RequestIdr,
}

/// X11/XKB keycode = Linux evdev keycode + 8. `inject_key` works in evdev space so the
/// KeyboardKey handler ADDS this; `inject_keysym` resolves against xkb's already-X11 keycodes
/// (min..=max) and passes them straight to `KeyboardHandle::input`, so it never adds it.
const EVDEV_TO_XKB_KEYCODE_OFFSET: u32 = 8;

fn get_gpu_driver(card_index: i32) -> String {
    let path = format!("/sys/class/drm/renderD{}/device/driver", 128 + card_index);
    match std::fs::read_link(&path) {
        Ok(link_path) => link_path.to_string_lossy().to_lowercase(),
        Err(_) => String::new(),
    }
}

/// True when GPU auto-selection is requested, preferring SELKIES_AUTO_GPU and
/// only consulting AUTO_GPU when SELKIES_AUTO_GPU is unset.
fn auto_gpu_enabled() -> bool {
    std::env::var("SELKIES_AUTO_GPU")
        .or_else(|_| std::env::var("AUTO_GPU"))
        .unwrap_or_default()
        .to_lowercase()
        == "true"
}

/// Resolve a usable /dev/dri/renderD* node by walking /sys/class/drm cards in
/// numeric order. This skips cards with no render node (e.g. an IPMI/VGA card0)
/// and only returns a node that is actually present in this namespace, so it
/// behaves correctly inside containers where /dev/dri is filtered.
fn auto_select_render_node() -> Option<String> {
    // Don't `?`-return if /sys/class/drm is unreadable (e.g. a container that
    // bind-mounts /dev/dri without /sys): fall through to the /dev/dri scan below.
    let mut cards: Vec<(u32, std::path::PathBuf)> = std::fs::read_dir("/sys/class/drm")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let num = e.file_name().into_string().ok()?.strip_prefix("card")?.parse::<u32>().ok()?;
            Some((num, e.path()))
        })
        .collect();
    cards.sort_by_key(|(n, _)| *n);
    for (_, path) in &cards {
        if let Ok(drm_entries) = std::fs::read_dir(path.join("device/drm")) {
            for de in drm_entries.flatten() {
                let name = de.file_name().into_string().unwrap_or_default();
                if name.starts_with("renderD") {
                    let dev = format!("/dev/dri/{}", name);
                    if std::path::Path::new(&dev).exists() {
                        return Some(dev);
                    }
                }
            }
        }
    }
    // Fallback: lowest render node directly under /dev/dri.
    let mut nodes: Vec<String> = std::fs::read_dir("/dev/dri")
        .ok()?
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.starts_with("renderD"))
        .collect();
    nodes.sort();
    nodes.first().map(|n| format!("/dev/dri/{}", n))
}

/// Resolve a keysym to an X11 keycode (+ whether Shift is needed) against smithay's own xkb
/// keymap. Prefers the unshifted level-0 binding, else the shifted level-1 binding. Returns
/// None if no key in the active layout produces it. Read-only; never panics on the keysym.
fn resolve_keysym_to_keycode(
    keymap: &smithay::input::keyboard::xkb::Keymap,
    layout: smithay::input::keyboard::Layout,
    target_keysym: u32,
) -> Option<(u32, u8)> {
    use smithay::input::keyboard::xkb;
    let min_kc = keymap.min_keycode().raw();
    let max_kc = keymap.max_keycode().raw();
    if min_kc > max_kc {
        return None;
    }
    // Scan shift-levels in preference order (0 = unshifted, preferred; 1 = Shift; 2 =
    // AltGr / ISO_Level3_Shift; 3 = Shift+AltGr) so AltGr-only glyphs on non-US layouts
    // (e.g. @ € | \ ~ on many EU layouts) resolve instead of being dropped. Returns the
    // level so the caller can synthesize the matching modifier(s).
    for level in 0u32..4 {
        for raw_kc in min_kc..=max_kc {
            let kc = xkb::Keycode::new(raw_kc);
            let syms = keymap.key_get_syms_by_level(kc, layout.0, level);
            if syms.iter().any(|s| s.raw() == target_keysym) {
                return Some((raw_kc, level as u8));
            }
        }
    }
    None
}

/// @brief The main execution loop of the Wayland backend.
///
/// This function acts as the central nervous system of the application. It runs in its own thread
/// and manages the entire lifecycle of the Wayland compositor. Its specific responsibilities include:
///
/// 1. **Initialization**: Sets up the `calloop` event loop, initializes the Wayland display,
///    and establishes the rendering pipeline. It attempts to initialize hardware acceleration
///    via GBM/EGL (DRM render nodes) and falls back to software rendering (Pixman) if unavailable.
///
/// 2. **State Management**: Maintains the `AppState`, which holds Wayland globals (Compositor,
///    Seat, SHM, etc.), buffer pools, and window management logic.
///
/// 3. **Event Dispatch**:
///    - **Command Channel**: Listens for control messages from the Python thread (Start/Stop,
///      Input Injection, Configuration changes).
///    - **Wayland Socket**: Accepts connections from Wayland clients (applications) and handles
///      protocol events.
///
/// 4. **The Render Loop**:
///    A high-frequency timer triggers the frame generation process:
///    - **Compositing**: Renders all active windows onto a virtual output framebuffer.
///    - **Readback Logic**: Determines if the GPU buffer needs to be copied to CPU memory
///      (e.g., for software encoding, watermarking, or cross-GPU transfer).
///    - **Encoding**: Passes the frame to the active encoder. This handles the complex
///      "Zero-Copy" path (sharing DMABUFs directly with hardware encoders) vs the "Readback"
///      path (copying pixels for CPU-based processing/encoding).
///    - **Transmission**: Sends the encoded video packets back to the Python layer via callback.
fn run_wayland_thread(
    command_rx: smithay::reexports::calloop::channel::Channel<ThreadCommand>,
    initial_width: i32,
    initial_height: i32,
    explicit_dri_node: String,
) {
    // Initial framebuffer size comes from selkies (the server owns resolution policy
    // and forwards it via the WaylandBackend constructor); first StartCapture resizes.
    let width: i32 = if initial_width > 0 { initial_width } else { 1024 };
    let height: i32 = if initial_height > 0 { initial_height } else { 768 };

    let mut event_loop = EventLoop::<AppState>::try_new().expect("Unable to create event_loop");
    let display: Display<AppState> = Display::new().unwrap();
    let dh: DisplayHandle = display.handle();
    dh.set_default_max_buffer_size(10 * 1024 * 1024);

    // Explicit node from selkies (via the constructor); fall back to AUTO_GPU
    // hardware detection (which the device library owns) when none was given.
    let mut dri_node = explicit_dri_node;
    if dri_node.is_empty() && auto_gpu_enabled() {
        if let Some(node) = auto_select_render_node() {
            dri_node = node;
            println!("[Wayland] AUTO_GPU enabled. Selected: {}", dri_node);
        }
    }
    // With no explicit node and AUTO_GPU off, honor an operator-set DRINODE before
    // falling back to the software renderer.
    if dri_node.is_empty() && !auto_gpu_enabled() {
        if let Ok(node) = std::env::var("DRINODE") {
            if !node.is_empty() {
                dri_node = node;
                println!("[Wayland] Using DRINODE from environment: {}", dri_node);
            }
        }
    }

    let mut use_gpu = !dri_node.is_empty();
    let render_node_path = dri_node.clone();

    let mut gles_renderer = None;
    let mut pixman_renderer = None;
    let mut offscreen_buffer: Option<(BufferObject<()>, Dmabuf)> = None;
    let mut dmabuf_global = None;
    let mut gbm_device_raw = None;
    let mut dmabuf_state = DmabufState::new();

    let mut gpu_success = false;
    if use_gpu {
        println!("[Wayland] Initializing GL Renderer using device: {}", dri_node);
        let init_res: Result<(), String> = (|| {
            let device_path = std::path::Path::new(&dri_node);
            let file = File::options().read(true).write(true).open(device_path)
                .map_err(|e| format!("Failed to open render device: {}", e))?;
            let file_for_alloc = file.try_clone()
                .map_err(|e| format!("Failed to clone file for GBM Allocator: {}", e))?;
            let gbm_allocator = RawGbmDevice::new(file_for_alloc)
                .map_err(|_| "Failed to create Raw GBM Device")?;
            let gbm = GbmDevice::new(file)
                .map_err(|_| "Failed to create GBM device")?;
            let egl = unsafe { EGLDisplay::new(gbm) }
                .map_err(|_| "Failed to create EGL display")?;
            let context = EGLContext::new(&egl)
                .map_err(|_| "Failed to create EGL context")?;
            let mut renderer = unsafe { GlesRenderer::new(context) }
                .map_err(|_| "Failed to init GlesRenderer")?;
            
            if let Err(e) = renderer.bind_wl_display(&dh) {
                println!("[Wayland] Warning: Failed to bind EGL to Wayland Display (Optional): {:?}", e);
            }

            let formats = Bind::<Dmabuf>::supported_formats(&renderer)
                .ok_or("Failed to query formats")?
                .into_iter()
                .collect::<Vec<_>>();

            let node = DrmNode::from_path(device_path)
                .map_err(|_| "Failed to create DrmNode")?;
            let dmabuf_default_feedback = DmabufFeedbackBuilder::new(node.dev_id(), formats.clone()).build();

            dmabuf_global = Some(if let Ok(default_feedback) = dmabuf_default_feedback {
                dmabuf_state.create_global_with_default_feedback::<AppState>(&dh, &default_feedback)
            } else {
                dmabuf_state.create_global::<AppState>(&dh, formats)
            });

            let bo = gbm_allocator.create_buffer_object(
                width as u32, height as u32, GbmFormat::Argb8888, BufferObjectFlags::RENDERING
            ).map_err(|_| "Failed to allocate GBM buffer")?;

            let dmabuf = create_dmabuf_from_bo(&bo);
            offscreen_buffer = Some((bo, dmabuf));
            gbm_device_raw = Some(gbm_allocator);
            gles_renderer = Some(renderer);
            Ok(())
        })();

        match init_res {
            Ok(_) => gpu_success = true,
            Err(e) => {
                println!("[Wayland] GPU Initialization failed: {}. Falling back to Software Renderer (Pixman).", e);
                use_gpu = false;
            }
        }
    }

    if !gpu_success {
        if !dri_node.is_empty() && !use_gpu {
            // Pass
        } else {
            println!("[Wayland] DRINODE unset. Initializing Software Renderer (Pixman).");
        }
        pixman_renderer = Some(PixmanRenderer::new().expect("Failed to init PixmanRenderer"));
        use_gpu = false;
    }

    let compositor_state = CompositorState::new::<AppState>(&dh);
    let fractional_scale_state = FractionalScaleManagerState::new::<AppState>(&dh);
    let shm_state = ShmState::new::<AppState>(&dh, vec![]);
    let output_state = OutputManagerState::new_with_xdg_output::<AppState>(&dh);
    let mut seat_state = SeatState::new();
    let shell_state = XdgShellState::new::<AppState>(&dh);
    let space = Space::default();
    let layer_shell_state = WlrLayerShellState::new::<AppState>(&dh);
    let data_device_state = DataDeviceState::new::<AppState>(&dh);
    let data_control_state = DataControlState::new::<AppState, _>(&dh, None, |_| true);
    let virtual_keyboard_state = VirtualKeyboardManagerState::new::<AppState, _>(&dh, |_client| true);
    let pointer_warp_state = PointerWarpManager::new::<AppState>(&dh);
    let relative_pointer_state = RelativePointerManagerState::new::<AppState>(&dh);
    let pointer_constraints_state = PointerConstraintsState::new::<AppState>(&dh);

    let foreign_toplevel_list = ForeignToplevelListState::new::<AppState>(&dh);
    let xdg_decoration_state = XdgDecorationState::new::<AppState>(&dh);
    let single_pixel_buffer = SinglePixelBufferState::new::<AppState>(&dh);
    let viewporter_state = ViewporterState::new::<AppState>(&dh);
    let presentation_state = PresentationState::new::<AppState>(&dh, 1);
    let xdg_activation_state = XdgActivationState::new::<AppState>(&dh);
    let primary_selection_state = PrimarySelectionState::new::<AppState>(&dh);
    let popups = PopupManager::default();

    let mut seat = seat_state.new_wl_seat(&dh, "seat0");
    seat.add_keyboard(XkbConfig::default(), 200, 25)
        .expect("Failed to init keyboard");
    seat.add_pointer();

    let mut state = AppState {
        compositor_state,
        fractional_scale_state,
        viewporter_state,
        presentation_state,
        shm_state,
        single_pixel_buffer,
        dmabuf_state,
        dmabuf_global,
        output_state,
        seat_state,
        shell_state,
        layer_shell_state,
        space,
        data_device_state,
        data_control_state,
        dh: dh.clone(),
        seat,
        virtual_keyboard_state,
        pointer_warp_state,
        relative_pointer_state,
        pointer_constraints_state,
        outputs: Vec::new(),
        pending_windows: Vec::new(),
        foreign_toplevel_list,
        xdg_decoration_state,
        xdg_activation_state,
        primary_selection_state,
        popups,
        frame_buffer: vec![0u8; (width * height * 4) as usize],
        nv12_buffer: vec![0u8; (width * height * 3 / 2) as usize],
        gles_renderer,
        pixman_renderer,
        gbm_device: gbm_device_raw,
        offscreen_buffer,
        is_capturing: false,
        settings: RustCaptureSettings {
            width,
            height,
            ..RustCaptureSettings::default()
        },
        callback: None,
        cursor_callback: None,
        stripes: Vec::with_capacity(MAX_STRIPE_CAPACITY),
        last_log_time: Instant::now(),
        encoded_frame_count: 0,
        total_stripes_encoded: 0,
        start_time: Instant::now(),
        clock: Clock::new(),
        frame_counter: 0,
        pending_force_idr: false,
        synthetic_shift_keysyms: std::collections::HashMap::new(),
        synthetic_mod_refcounts: std::collections::HashMap::new(),
        use_gpu,
        video_encoder: None,
        vaapi_state: StripeState::default(),
        cursor_helper: Cursor::load(),
        overlay_state: OverlayState::default(),
        current_cursor_icon: None,
        cursor_buffer: None,
        cursor_cache: std::collections::HashMap::new(),
        render_cursor_on_framebuffer: false,
        render_node_path,
        recording_sink: None,
    };

    let output = Output::new(
        "HEADLESS-1".into(),
        PhysicalProperties {
            size: (width as i32, height as i32).into(),
            subpixel: Subpixel::Unknown,
            make: "Pixelflux".into(),
            model: "Virtual".into(),
            serial_number: "001".into(),
        },
    );
    output.change_current_state(
        Some(OutputMode {
            size: (width as i32, height as i32).into(),
            refresh: 60_000,
        }),
        Some(Transform::Normal),
        Some(OutputScale::Fractional(1.0)),
        Some((0, 0).into()),
    );
    output.set_preferred(OutputMode {
        size: (width as i32, height as i32).into(),
        refresh: 60_000,
    });
    state.space.map_output(&output, (0, 0));
    state.outputs.push(output.clone());
    let _global = output.create_global::<AppState>(&dh);
    let mut damage_tracker = OutputDamageTracker::from_output(&output);

    event_loop
        .handle()
        .insert_source(command_rx, move |event, _, state| {
            match event {
                CalloopEvent::Msg(ThreadCommand::StartCapture(cb, mut settings)) => {
                    let auto_gpu = auto_gpu_enabled();
                    if auto_gpu {
                        if let Some(idx_str) = state.render_node_path.strip_prefix("/dev/dri/renderD") {
                            if let Ok(idx) = idx_str.parse::<i32>() {
                                settings.vaapi_render_node_index = idx - 128;
                            }
                        }
                    }

                    // H.264 4:2:0 needs even dimensions.
                    if settings.output_mode == 1 {
                        settings.width &= !1;
                        settings.height &= !1;
                    }

                    state.recording_sink =
                        crate::recording_sink::RecordingSink::try_bind(&settings.recording_socket);

                    if let Some(output) = state.outputs.first() {
                        let current_mode = output.current_mode().unwrap();
                        let current_w = current_mode.size.w;
                        let current_h = current_mode.size.h;
                        let current_scale = output.current_scale().fractional_scale();
                        let current_refresh = current_mode.refresh;
                        let target_refresh = (settings.target_fps * 1000.0).round() as i32;

                        let scale = settings.scale.max(0.1);
                        let logical_width = (settings.width as f64 / scale).round() as i32;
                        let logical_height = (settings.height as f64 / scale).round() as i32;

                        if current_w != settings.width
                            || current_h != settings.height
                            || (current_scale - settings.scale).abs() > 0.001
                            || current_refresh != target_refresh
                        {
                            println!(
                                "[Wayland] Configuring Output: {}x{} @ {:.2} FPS (Scale {:.2})",
                                settings.width, settings.height, settings.target_fps, settings.scale
                            );
                            let new_mode = OutputMode {
                                size: (settings.width, settings.height).into(),
                                refresh: target_refresh,
                            };
                            output.change_current_state(
                                Some(new_mode),
                                Some(Transform::Normal),
                                Some(OutputScale::Fractional(settings.scale)),
                                Some((0, 0).into()),
                            );
                            output.set_preferred(new_mode);

                            let pixel_count = (settings.width * settings.height) as usize;
                            state.frame_buffer = vec![0u8; pixel_count * 4];

                            if state.use_gpu {
                                if let Some(gbm) = state.gbm_device.as_mut() {
                                    let bo = gbm
                                        .create_buffer_object(
                                            settings.width as u32,
                                            settings.height as u32,
                                            GbmFormat::Argb8888,
                                            BufferObjectFlags::RENDERING,
                                        )
                                        .expect("Failed to resize GBM buffer");

                                    let dmabuf = create_dmabuf_from_bo(&bo);
                                    state.offscreen_buffer = Some((bo, dmabuf));
                                }
                            }
                        }

                        // Size depends on fullcolor too, so (re)size unconditionally,
                        // not only on a resolution change.
                        let nv12_pixel_count = (settings.width * settings.height) as usize;
                        let nv12_needed = if settings.h264_fullcolor {
                            nv12_pixel_count * 3
                        } else {
                            nv12_pixel_count * 3 / 2
                        };
                        if state.nv12_buffer.len() != nv12_needed {
                            state.nv12_buffer = vec![0u8; nv12_needed];
                        }

                        for window in state.space.elements() {
                            if let Some(surface) = window.wl_surface() {
                                output.enter(&surface);
                                with_states(&surface, |states| {
                                    smithay::wayland::fractional_scale::with_fractional_scale(states, |fs| {
                                        fs.set_preferred_scale(scale);
                                    });
                                });
                            }
                            if let Some(toplevel) = window.toplevel() {
                                toplevel.with_pending_state(|state| {
                                    use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel::State;
                                    state.states.set(State::Fullscreen);
                                    state.states.set(State::Activated);
                                    state.size = Some((logical_width, logical_height).into());
                                });
                                toplevel.send_configure();
                            }
                        }
                    }

                    state.video_encoder = None;
                    let use_cpu_explicit = settings.use_cpu || settings.vaapi_render_node_index == -1;

                    if use_cpu_explicit {
                        println!("[Wayland] CPU encoding selected (use_cpu=true or vaapi_node=-1).");
                    } else {
                        let encode_driver = get_gpu_driver(settings.vaapi_render_node_index);
                        println!(
                            "[Wayland] Encode Node Index: {} | Driver: {}",
                            settings.vaapi_render_node_index, encode_driver
                        );

                        if encode_driver.contains("nvidia") {
                            println!("[Wayland] Nvidia Encoder detected. Initializing NVENC...");
                            let egl_display = if let Some(renderer) = state.gles_renderer.as_ref() {
                                renderer.egl_context().display().get_display_handle().handle
                            } else {
                                std::ptr::null()
                            };

                            match NvencEncoder::new(&settings, egl_display, state.recording_sink.clone()) {
                                Ok(encoder) => {
                                    state.video_encoder = Some(GpuEncoder::Nvenc(encoder));
                                    println!("[Wayland] NVENC Encoder initialized successfully.");
                                }
                                Err(e) => eprintln!(
                                    "[Wayland] Failed to init NVENC: {}. Falling back to CPU.",
                                    e
                                ),
                            }
                        } else {
                            println!("[Wayland] Initializing Unified VAAPI Encoder...");
                            if settings.h264_fullcolor {
                                println!("[Wayland] 4:4:4 Fullcolor requested. VAAPI does not support this profile reliably. Falling back to CPU.");
                            } else {
                                match VaapiEncoder::new(&settings, state.recording_sink.clone()) {
                                    Ok(encoder) => {
                                        state.video_encoder = Some(GpuEncoder::Vaapi(encoder));
                                        println!(
                                            "[Wayland] VAAPI Encoder initialized successfully."
                                        );
                                    }
                                    Err(e) => eprintln!(
                                        "[Wayland] Failed to init VAAPI: {}. Falling back to CPU.",
                                        e
                                    ),
                                }
                            }
                        }
                    }

                    let mut different_gpu = false;

                    if state.video_encoder.is_some() {
                        let encode_node_idx = settings.vaapi_render_node_index;
                        if !state.render_node_path.is_empty() && encode_node_idx >= 0 {
                            if !state.render_node_path.contains(&format!("renderD{}", 128 + encode_node_idx)) {
                                different_gpu = true;
                            }
                        }
                    }

                    if different_gpu {
                        println!("[Wayland] Decision: Rendering and Encoding GPUs differ -> Forcing Readback (CPU path for pixels).");
                    }
                    if state.video_encoder.is_none() {
                        println!("[Wayland] Decision: No GPU Encoder available -> Using CPU Software Encoding.");
                    } else if !different_gpu {
                        println!("[Wayland] Decision: Zero-Copy path active.");
                    }

                    let num_cores = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1);
                    let mut n_stripes = num_cores;

                    if settings.output_mode == 1 {
                        if state.video_encoder.is_some() || settings.h264_fullframe {
                            n_stripes = 1;
                        } else {
                            let min_h = 64;
                            if settings.height < min_h {
                                n_stripes = 1;
                            } else {
                                let max_stripes = (settings.height / min_h) as usize;
                                n_stripes = n_stripes.min(max_stripes).max(1);
                            }
                        }
                    } else {
                        n_stripes = n_stripes.min(settings.height as usize).max(1);
                    }

                    if state.recording_sink.is_some() {
                        if settings.output_mode == 0 {
                            eprintln!(
                                "[recording_sink] WARNING: recording_socket is set but output_mode is JPEG (0). \
                                 The recording sink requires a single H.264 stream. Please set output_mode=1 \
                                 on the Python CaptureSettings to produce a recordable output."
                            );
                        } else if state.video_encoder.is_none() && !settings.h264_fullframe {
                            eprintln!(
                                "[recording_sink] WARNING: recording_socket is set but the CPU encoder is running in \
                                 multi-stripe mode. This produces N independent sub-frame H.264 streams that \
                                 cannot be muxed together. Set h264_fullframe=true on the Python CaptureSettings \
                                 (or use a working GPU encoder) to produce a recordable single-stream output."
                            );
                        }
                    }

                    let mut log_msg = format!(
                        "Stream settings active -> Res: {}x{} | FPS: {:.1} | Stripes: {}",
                        settings.width, settings.height, settings.target_fps, n_stripes
                    );

                    if settings.output_mode == 0 {
                        log_msg.push_str(&format!(
                            " | Mode: JPEG | Quality: {}",
                            settings.jpeg_quality
                        ));
                        if settings.use_paint_over_quality {
                            log_msg.push_str(&format!(
                                " | PaintOver Q: {} (Trigger: {}f)",
                                settings.paint_over_jpeg_quality, settings.paint_over_trigger_frames
                            ));
                        }
                    } else {
                        let encoder_type = match &state.video_encoder {
                            Some(GpuEncoder::Nvenc(_)) => "NVENC",
                            Some(GpuEncoder::Vaapi(_)) => "VAAPI",
                            None => "CPU",
                        };
                        log_msg.push_str(&format!(" | Mode: H264 ({})", encoder_type));

                        if state.video_encoder.is_some() || settings.h264_fullframe {
                            log_msg.push_str(" FullFrame");
                        } else {
                            log_msg.push_str(" Striped");
                        }

                        if settings.h264_streaming_mode {
                            log_msg.push_str(" Streaming");
                        }

                        log_msg.push_str(&format!(" | CRF: {}", settings.h264_crf));

                        if settings.use_paint_over_quality {
                            log_msg.push_str(&format!(
                                " | PaintOver CRF: {} (Burst: {}f)",
                                settings.h264_paintover_crf, settings.h264_paintover_burst_frames
                            ));
                        }

                        let is_actually_444 = match &state.video_encoder {
                            Some(GpuEncoder::Nvenc(_)) => settings.h264_fullcolor,
                            Some(_) => false,
                            None => settings.h264_fullcolor,
                        };

                        let range_str = if is_actually_444 {
                            "I444 (Full Range)"
                        } else {
                            "I420 (Limited Range)"
                        };
                        log_msg.push_str(&format!(" | Colorspace: {}", range_str));
                    }

                    log_msg.push_str(&format!(
                        " | Damage Thresh: {}f | Damage Dur: {}f",
                        settings.damage_block_threshold, settings.damage_block_duration
                    ));

                    println!("{}", log_msg);

                    let watermark_output_scale = state
                        .outputs
                        .first()
                        .map(|o| o.current_scale().fractional_scale())
                        .unwrap_or(1.0);
                    state
                        .overlay_state
                        .load_watermark(&settings.watermark_path, watermark_output_scale);
                    state.callback = Some(cb);
                    state.is_capturing = true;
                    state.render_cursor_on_framebuffer = settings.capture_cursor; 
                    state.settings = settings.clone();
                    state.encoded_frame_count = 0;
                    state.total_stripes_encoded = 0;
                    state.last_log_time = Instant::now();
                    state.frame_counter = 0;
                    state.pending_force_idr = false;
                    state.stripes.clear();
                    state.vaapi_state = StripeState::default();
                }
                CalloopEvent::Msg(ThreadCommand::StopCapture) => {
                    println!("[Wayland] Capture loop stopped.");
                    state.is_capturing = false;
                    state.callback = None;
                    state.video_encoder = None;
                    state.recording_sink = None;
                }
                CalloopEvent::Msg(ThreadCommand::SetCursorCallback(cb)) => {
                    state.cursor_callback = Some(cb);
                }
                CalloopEvent::Msg(ThreadCommand::KeyboardKey { scancode, state: key_state_val }) => {
                    let key_state = if key_state_val > 0 { KeyState::Pressed } else { KeyState::Released };
                    let serial = next_serial();
                    let time = wayland_time();
                    if let Some(keyboard) = state.seat.get_keyboard() {
                        // scancode is an evdev keycode; xkb/smithay want X11 keycodes (see
                        // EVDEV_TO_XKB_KEYCODE_OFFSET).
                        keyboard.input(state, Keycode::new(scancode.saturating_add(EVDEV_TO_XKB_KEYCODE_OFFSET)), key_state, serial, time, |_, _, _| {
                            FilterResult::<()>::Forward
                        });
                    }
                }
                CalloopEvent::Msg(ThreadCommand::KeyboardKeysym { keysym, state: key_state_val }) => {
                    // Inject by keysym against our own live xkb keymap: resolve to an X11 keycode
                    // (+ Shift), then inject, synthesizing a Shift press/release for shifted keysyms.
                    let key_state = if key_state_val > 0 { KeyState::Pressed } else { KeyState::Released };
                    if let Some(keyboard) = state.seat.get_keyboard() {
                        let inject = |state: &mut AppState, x11_kc: u32, ks: KeyState| {
                            let serial = next_serial();
                            let time = wayland_time();
                            keyboard.input(state, Keycode::new(x11_kc), ks, serial, time, |_, _, _| {
                                FilterResult::<()>::Forward
                            });
                        };

                        match key_state {
                            KeyState::Pressed => {
                                // Phase 1: resolve against the live keymap (read-only) -> (target
                                // keycode, the modifier keycodes that select its shift-level). No
                                // `.unwrap()` on attacker-supplied data.
                                let resolved: Option<(u32, Vec<u32>)> =
                                    keyboard.with_xkb_state(state, |context| {
                                        let xkb_guard = match context.xkb().lock() {
                                            Ok(g) => g,
                                            Err(_) => return None,
                                        };
                                        let layout = xkb_guard.active_layout();
                                        // SAFETY: the &Keymap borrow stays within this guard's scope
                                        // and is only read; we never store it past the lock.
                                        let keymap = unsafe { xkb_guard.keymap() };
                                        let (kc, level) = resolve_keysym_to_keycode(keymap, layout, keysym)?;
                                        // Resolve a modifier keysym to its keycode, falling back to the
                                        // conventional X11 keycode when the layout lacks it.
                                        let resolve_mod = |ks: u32, fallback: u32| {
                                            resolve_keysym_to_keycode(keymap, layout, ks)
                                                .map(|(c, _)| c)
                                                .unwrap_or(fallback)
                                        };
                                        // Standard ISO convention: level 1 = Shift, level 2 = AltGr
                                        // (ISO_Level3_Shift), level 3 = Shift+AltGr.
                                        let mut mods: Vec<u32> = Vec::new();
                                        if level == 1 || level == 3 {
                                            // Shift_L (0xFFE1); fallback evdev KEY_LEFTSHIFT 42 + offset.
                                            mods.push(resolve_mod(0xFFE1, 42 + EVDEV_TO_XKB_KEYCODE_OFFSET));
                                        }
                                        if level == 2 || level == 3 {
                                            // ISO_Level3_Shift / AltGr (0xFE03); fallback evdev KEY_RIGHTALT 100 + offset.
                                            mods.push(resolve_mod(0xFE03, 100 + EVDEV_TO_XKB_KEYCODE_OFFSET));
                                        }
                                        Some((kc, mods))
                                    });

                                // Phase 2: inject and RECORD the injected keycodes so the matching
                                // key-up releases exactly these, regardless of later layout changes.
                                // Synthetic modifiers are ref-counted so two simultaneously-held
                                // shifted/AltGr keys don't release each other's modifier early.
                                if let Some((kc, mods)) = resolved {
                                    // Auto-repeat (~25Hz) re-presses the same keysym without an
                                    // intervening release. Only the FIRST press touches modifier
                                    // refcounts + the held-key map; otherwise the refcount would
                                    // climb while the single map entry (and its one release)
                                    // stayed at 1, leaving Shift/AltGr stuck held. Re-presses still
                                    // re-inject the key-down so auto-repeat key events flow through.
                                    if !state.synthetic_shift_keysyms.contains_key(&keysym) {
                                        for &m in &mods {
                                            let first = {
                                                let c = state.synthetic_mod_refcounts.entry(m).or_insert(0);
                                                *c += 1;
                                                *c == 1
                                            };
                                            if first {
                                                inject(state, m, KeyState::Pressed);
                                            }
                                        }
                                        state.synthetic_shift_keysyms.insert(keysym, (kc, mods));
                                    }
                                    inject(state, kc, KeyState::Pressed);
                                } else {
                                    eprintln!(
                                        "[Wayland] inject_keysym: keysym {:#06x} not found in active xkb layout; ignoring",
                                        keysym
                                    );
                                }
                            }
                            KeyState::Released => {
                                // Release the keycodes recorded at PRESS time; do NOT re-resolve
                                // (the layout may have changed mid-keystroke). A synthetic modifier
                                // is released only when its last holder is released (ref-counted).
                                if let Some((kc, mods)) =
                                    state.synthetic_shift_keysyms.remove(&keysym)
                                {
                                    inject(state, kc, KeyState::Released);
                                    for &m in &mods {
                                        let last = match state.synthetic_mod_refcounts.get_mut(&m) {
                                            Some(c) => {
                                                *c = c.saturating_sub(1);
                                                *c == 0
                                            }
                                            None => false,
                                        };
                                        if last {
                                            state.synthetic_mod_refcounts.remove(&m);
                                            inject(state, m, KeyState::Released);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                CalloopEvent::Msg(ThreadCommand::GetXkbKeymap { reply }) => {
                    // Hand back our keymap as an XKB_KEYMAP_FORMAT_TEXT_V1 string so a consumer can
                    // build its reverse keysym map from the IDENTICAL keymap.
                    let mut keymap_str = String::new();
                    if let Some(keyboard) = state.seat.get_keyboard() {
                        keymap_str = keyboard.with_xkb_state(state, |context| {
                            match context.xkb().lock() {
                                Ok(guard) => {
                                    // SAFETY: read-only use of the &Keymap within the guard scope.
                                    let keymap = unsafe { guard.keymap() };
                                    keymap.get_as_string(
                                        smithay::input::keyboard::xkb::KEYMAP_FORMAT_TEXT_V1,
                                    )
                                }
                                Err(_) => String::new(),
                            }
                        });
                    }
                    // Best-effort: the caller may have timed out and dropped the receiver.
                    let _ = reply.send(keymap_str);
                }
                CalloopEvent::Msg(ThreadCommand::PointerMotion { x, y }) => {
                    let serial = next_serial();
                    let time = wayland_time();
                    let scale = state.settings.scale;
                    let logical_w = (state.settings.width as f64 / scale).floor();
                    let logical_h = (state.settings.height as f64 / scale).floor();
                    let logical_x = (x / scale).max(0.0).min(logical_w - 1.0);
                    let logical_y = (y / scale).max(0.0).min(logical_h - 1.0);

                    if let Some(pointer) = state.seat.get_pointer() {
                        let p = Point::<f64, smithay::utils::Logical>::from((logical_x, logical_y));
                        let mut under = None;

                        if let Some(output) = state.outputs.first() {
                            let layer_map = layer_map_for_output(output);
                            let pointer_pos = p.to_i32_round();

                            for layer in layer_map.layers().rev() {
                                let state_layer = layer.layer();
                                if state_layer == smithay::wayland::shell::wlr_layer::Layer::Overlay || state_layer == smithay::wayland::shell::wlr_layer::Layer::Top {
                                    if let Some(bbox) = layer_map.layer_geometry(layer) {
                                        if bbox.contains(pointer_pos) {
                                            under = Some((FocusTarget::LayerSurface(layer.clone()), bbox.loc.to_f64()));
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        if under.is_none() {
                            under = state.space.element_under(p).map(|(window, loc)| {
                                (FocusTarget::Window(window.clone()), loc.to_f64())
                            });
                        }

                        if under.is_none() {
                            if let Some(output) = state.outputs.first() {
                                let layer_map = layer_map_for_output(output);
                                let pointer_pos = p.to_i32_round();

                                for layer in layer_map.layers().rev() {
                                    let state_layer = layer.layer();
                                    if state_layer == smithay::wayland::shell::wlr_layer::Layer::Bottom || state_layer == smithay::wayland::shell::wlr_layer::Layer::Background {
                                        if let Some(bbox) = layer_map.layer_geometry(layer) {
                                            if bbox.contains(pointer_pos) {
                                                under = Some((FocusTarget::LayerSurface(layer.clone()), bbox.loc.to_f64()));
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        pointer.motion(state, under, &MotionEvent { location: p, serial, time });
                        pointer.frame(state);
                    }
                }
                CalloopEvent::Msg(ThreadCommand::PointerRelativeMotion { dx, dy }) => {
                    let utime = wayland_utime();
                    let time = wayland_time();
                    let serial = next_serial();

                    if let Some(pointer) = state.seat.get_pointer() {
                        let current_pos = pointer.current_location();
                        
                        let scale = state.settings.scale;
                        let max_w = (state.settings.width as f64 / scale).floor() - 1.0;
                        let max_h = (state.settings.height as f64 / scale).floor() - 1.0;

                        let new_x = (current_pos.x + dx).max(0.0).min(max_w);
                        let new_y = (current_pos.y + dy).max(0.0).min(max_h);
                        
                        let new_pos = Point::<f64, smithay::utils::Logical>::from((new_x, new_y));

                        let under = state.space.element_under(new_pos).map(|(window, loc)| {
                            (FocusTarget::Window(window.clone()), loc.to_f64())
                        });

                        pointer.motion(
                            state, 
                            under.clone(), 
                            &MotionEvent { 
                                location: new_pos, 
                                serial, 
                                time 
                            }
                        );

                        let event = RelativeMotionEvent {
                            utime,
                            delta: (dx, dy).into(),
                            delta_unaccel: (dx, dy).into(),
                        };
                        pointer.relative_motion(state, under, &event);

                        pointer.frame(state);
                    }
                }
                CalloopEvent::Msg(ThreadCommand::PointerButton { btn, state: btn_state_val }) => {
                    let serial = next_serial();
                    let time = wayland_time();
                    let button_state = if btn_state_val > 0 { smithay::backend::input::ButtonState::Pressed } else { smithay::backend::input::ButtonState::Released };

                    if let Some(pointer) = state.seat.get_pointer() {
                        if button_state == smithay::backend::input::ButtonState::Pressed {
                            let pos = pointer.current_location();
                            let target_window = state.space.element_under(pos).map(|(w, _)| w.clone());

                            if let Some(window) = target_window {
                                state.space.raise_element(&window, true);
                                if let Some(keyboard) = state.seat.get_keyboard() {
                                    keyboard.set_focus(state, Some(FocusTarget::Window(window)), serial);
                                }
                            }
                        }
                        // `btn` is already an evdev BTN_ code by contract (the selkies
                        // consumer sends e.g. 272=BTN_LEFT/273=BTN_RIGHT/274=BTN_MIDDLE,
                        // and 0x113=BTN_SIDE/0x114=BTN_EXTRA for back/forward), so pass it
                        // straight through to smithay's pointer.
                        let button = btn;
                        pointer.button(state, &ButtonEvent { button, state: button_state, serial, time });
                        pointer.frame(state);
                    }
                }
                CalloopEvent::Msg(ThreadCommand::PointerAxis { x, y }) => {
                    let time = wayland_time();
                    
                    if let Some(pointer) = state.seat.get_pointer() {
                        const V120_MULTIPLIER: f64 = 12.0; 

                        let mut frame = AxisFrame::new(time).source(AxisSource::Wheel);

                        if x != 0.0 { 
                            frame = frame
                                .value(Axis::Horizontal, x)
                                .v120(Axis::Horizontal, (x * V120_MULTIPLIER) as i32);
                        }
                        
                        if y != 0.0 { 
                            frame = frame
                                .value(Axis::Vertical, y)
                                .v120(Axis::Vertical, (y * V120_MULTIPLIER) as i32);
                        }

                        if x != 0.0 || y != 0.0 {
                            pointer.axis(state, frame);
                            pointer.frame(state);
                        }
                    }
                }
                CalloopEvent::Msg(ThreadCommand::UpdateCursorConfig { render_on_framebuffer }) => {
                    state.render_cursor_on_framebuffer = render_on_framebuffer;
                }
                CalloopEvent::Msg(ThreadCommand::RequestIdr) => {
                    // On-demand keyframe (client reconnect / decoder reset). Consumed on
                    // the next captured frame; forces a send + IDR even on a static screen.
                    state.pending_force_idr = true;
                }
                CalloopEvent::Closed => {}
            }
        })
        .unwrap();

    let source = ListeningSocketSource::new_auto().unwrap();
    let socket_name = source.socket_name().to_string_lossy().into_owned();
    println!("[Wayland] Socket listening on: {:?}", socket_name);
    std::env::set_var("WAYLAND_DISPLAY", &socket_name);

    event_loop
        .handle()
        .insert_source(source, |client_stream, _, state| {
            if let Err(err) = state
                .dh
                .insert_client(client_stream, Arc::new(ClientState::default()))
            {
                eprintln!("Error adding wayland client: {:?}", err);
            }
        })
        .expect("Failed to init wayland socket source");

    let timer = Timer::immediate();
    let mut is_memory_throttling = false;
    event_loop
        .handle()
        .insert_source(timer, move |_, _, state| {
            let loop_start_time = Instant::now();
            state.space.refresh();

            let current_rss = get_process_rss_bytes();
            let shm_usage = get_shm_usage_bytes();
            let memory_threshold = calculate_memory_threshold(state.settings.width, state.settings.height);

            if current_rss > memory_threshold || shm_usage > (4 * 1024 * 1024 * 1024) {
                if !is_memory_throttling {
                    is_memory_throttling = true;
                }
            } else if is_memory_throttling {
                if current_rss < (memory_threshold as f64 * 0.75) as usize && shm_usage < (3 * 1024 * 1024 * 1024) {
                    is_memory_throttling = false;
                }
            }

            let now = Instant::now();
            let elapsed = now.duration_since(state.last_log_time).as_secs_f64();
            if elapsed >= 1.0 {
                if state.is_capturing && state.settings.debug_logging {
                    let actual_fps = if elapsed > 0.0 { state.encoded_frame_count as f64 / elapsed } else { 0.0 };
                    let stripes_per_sec = if elapsed > 0.0 { state.total_stripes_encoded as f64 / elapsed } else { 0.0 };

                    let mode_str = if state.settings.output_mode == 0 {
                        format!("JPEG Q:{}", state.settings.jpeg_quality)
                    } else {
                        let rendering_gpu = state.use_gpu;
                        let encoding_gpu_avail = state.video_encoder.is_some();
                        let mut different_gpu = false;

                        if state.video_encoder.is_some() {
                            // Use the node resolved at startup, not a fresh env read.
                            let dri_node = state.render_node_path.clone();
                            let encode_node_idx = state.settings.vaapi_render_node_index;
                            if !dri_node.is_empty() && encode_node_idx >= 0 {
                                if !dri_node.contains(&format!("renderD{}", 128 + encode_node_idx)) {
                                    different_gpu = true;
                                }
                            }
                        }

                        let is_readback = !rendering_gpu || !encoding_gpu_avail || different_gpu;
                        let copy_mode_str = if is_readback { "Readback" } else { "ZeroCopy" };

                        let backend = match &state.video_encoder {
                            Some(GpuEncoder::Nvenc(_)) => format!("NVENC ({})", copy_mode_str),
                            Some(GpuEncoder::Vaapi(_)) => format!("VAAPI ({})", copy_mode_str),
                            None => "CPU".to_string(),
                        };

                        let is_actually_444 = match &state.video_encoder {
                            Some(GpuEncoder::Nvenc(_)) => state.settings.h264_fullcolor,
                            Some(_) => false,
                            None => state.settings.h264_fullcolor,
                        };

                        let cs_str = if is_actually_444 { "CS_IN:I444" } else { "CS_IN:I420" };
                        let range_str = if is_actually_444 { "FR" } else { "LR" };
                        let frame_str = if state.video_encoder.is_some() || state.settings.h264_fullframe { "FF" } else { "Striped" };

                        format!("H264 ({}) {} {} {} CRF:{}", backend, cs_str, range_str, frame_str, state.settings.h264_crf)
                    };

                    let num_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
                    let mut n_stripes = num_cores;
                    if state.settings.output_mode == 1 {
                        if state.video_encoder.is_some() || state.settings.h264_fullframe {
                            n_stripes = 1;
                        } else {
                            let min_h = 64;
                            if state.settings.height < min_h { n_stripes = 1; }
                            else {
                                let max_stripes = (state.settings.height / min_h) as usize;
                                n_stripes = n_stripes.min(max_stripes).max(1);
                            }
                        }
                    } else {
                        n_stripes = n_stripes.min(state.settings.height as usize).max(1);
                    }

                    let rss_mb = current_rss / 1024 / 1024;
                    let shm_mb = shm_usage / 1024 / 1024;
                    let throttle_warn = if is_memory_throttling { " [THROTTLED]" } else { "" };

                    println!("Res: {}x{} Mode: {} Stripes: {} EncFPS: {:.2} EncStripes/s: {:.2} Mem: {}MB SHM: {}MB{}",
                        state.settings.width, state.settings.height, mode_str, n_stripes, actual_fps, stripes_per_sec, rss_mb, shm_mb, throttle_warn);
                }
                state.encoded_frame_count = 0;
                state.total_stripes_encoded = 0;
                state.last_log_time = now;
            }

            if !state.is_capturing {
                return TimeoutAction::ToDuration(Duration::from_millis(16));
            }

            // READ (don't take) the on-demand IDR request: it's cleared below only where an
            // encoder actually consumes it, so a request on a skipped frame isn't dropped.
            let requested_idr = state.pending_force_idr;

            state.overlay_state.update_position(
                state.settings.width,
                state.settings.height,
                state.settings.watermark_location_enum,
            );

            let mut render_success = false;
            let mut damage_rects: Vec<Rectangle<i32, Physical>> = Vec::new();
            let width = state.settings.width;
            let height = state.settings.height;

            if let Some(output) = state.outputs.first().cloned() {
                let render_age = if state.overlay_state.is_animated() { 0 } else { 1 };
                let rendering_gpu = state.use_gpu;
                let encoding_gpu_avail = state.video_encoder.is_some();

                let mut different_gpu = false;
                if state.video_encoder.is_some() {
                    let encode_node_idx = state.settings.vaapi_render_node_index;
                    if !state.render_node_path.is_empty() && encode_node_idx >= 0 {
                        if !state.render_node_path.contains(&format!("renderD{}", 128 + encode_node_idx)) {
                            different_gpu = true;
                        }
                    }
                }

                let needs_readback = !rendering_gpu || !encoding_gpu_avail || different_gpu;
                if rendering_gpu {
                    if let Some(renderer) = state.gles_renderer.as_mut() {
                        if let Some((_bo, dmabuf)) = state.offscreen_buffer.as_mut() {
                            match renderer.bind(dmabuf) {
                                Ok(mut frame) => {
                                    let mut elements: Vec<CompositionElements<GlesRenderer, WaylandSurfaceRenderElement<GlesRenderer>>> = Vec::new();

                                    if state.render_cursor_on_framebuffer {
                                        if let Some(pointer) = state.seat.get_pointer() {
                                            let pos = pointer.current_location();
                                            let output_scale_val = output.current_scale().fractional_scale();
                                            let scale = Scale::from(output_scale_val);

                                            if let Some(CursorImageStatus::Named(icon)) = &state.current_cursor_icon {
                                                let name = wayland::frontend::cursor_icon_to_str(icon);
                                                let time = Duration::from_millis(state.clock.now().as_millis() as u64);
                                                if let Some(image) = state.cursor_helper.get_image_by_name(name, output_scale_val.round() as u32, time) {
                                                    if let Some(elem) = state.overlay_state.get_cursor_element(renderer, image, pos.to_i32_round()) {
                                                        elements.push(CompositionElements::Cursor(elem));
                                                    }
                                                }
                                            } else if let Some(CursorImageStatus::Surface(surface)) = &state.current_cursor_icon {
                                                 let phys_pos = pos.to_physical(scale);
                                                 let elem_result = with_states(surface, |states| {
                                                     WaylandSurfaceRenderElement::from_surface(renderer, surface, states, phys_pos, 1.0, smithay::backend::renderer::element::Kind::Cursor)
                                                 });
                                                 if let Ok(Some(cursor_elem)) = elem_result {
                                                     elements.push(CompositionElements::Surface(cursor_elem));
                                                 }
                                            } else if state.current_cursor_icon.is_none() {
                                                let time = Duration::from_millis(state.clock.now().as_millis() as u64);
                                                let image = state.cursor_helper.get_image(output_scale_val.round() as u32, time);
                                                if let Some(elem) = state.overlay_state.get_cursor_element(renderer, image, pos.to_i32_round()) {
                                                    elements.push(CompositionElements::Cursor(elem));
                                                }
                                            }
                                        }
                                    }

                                    if let Some(elem) = state.overlay_state.get_watermark_element(renderer) {
                                        elements.push(CompositionElements::Cursor(elem));
                                    }

                                    {
                                        let layer_map = layer_map_for_output(&output);

                                        let draw_layer = |renderer: &mut GlesRenderer, elements: &mut Vec<CompositionElements<GlesRenderer, WaylandSurfaceRenderElement<GlesRenderer>>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
                                            for surface in layer_map.layers().rev() {
                                                let current_layer = surface.layer();
                                                if current_layer == target_layer {
                                                    if let Some(geo) = layer_map.layer_geometry(surface) {
                                                        let elem = smithay::wayland::compositor::with_states(surface.wl_surface(), |states| {
                                                            WaylandSurfaceRenderElement::from_surface(
                                                                renderer, surface.wl_surface(), states,
                                                                geo.loc.to_physical_precise_round(1), 1.0,
                                                                smithay::backend::renderer::element::Kind::Unspecified
                                                            )
                                                        });
                                                        if let Ok(Some(e)) = elem {
                                                            elements.push(CompositionElements::Surface(e));
                                                        }
                                                    }
                                                }
                                            }
                                        };

                                        draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Overlay);
                                        draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Top);
                                    }

                                    for window in state.space.elements_for_output(&output).collect::<Vec<_>>().into_iter().rev() {
                                        let window_loc = state.space.element_location(window).unwrap_or_default();

                                        if let Some(surface) = window.wl_surface() {
                                            let popups = PopupManager::popups_for_surface(&surface);
                                            for (popup, location) in popups {
                                                let popup_surface = popup.wl_surface();
                                                let popup_pos = window_loc + location;
                                                let elem = smithay::wayland::compositor::with_states(popup_surface, |states| {
                                                    WaylandSurfaceRenderElement::from_surface(
                                                        renderer,
                                                        popup_surface,
                                                        states,
                                                        popup_pos.to_physical_precise_round(1),
                                                        1.0,
                                                        smithay::backend::renderer::element::Kind::Unspecified
                                                    )
                                                });
                                                if let Ok(Some(e)) = elem {
                                                    elements.push(CompositionElements::Surface(e));
                                                }
                                            }
                                        }

                                        elements.extend(window.render_elements(renderer, window_loc.to_physical_precise_round(1), Scale::from(1.0), 1.0).into_iter().map(CompositionElements::Space));
                                    }

                                    {
                                        let layer_map = layer_map_for_output(&output);

                                        let draw_layer = |renderer: &mut GlesRenderer, elements: &mut Vec<CompositionElements<GlesRenderer, WaylandSurfaceRenderElement<GlesRenderer>>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
                                            for surface in layer_map.layers() {
                                                let current_layer = surface.layer();
                                                if current_layer == target_layer {
                                                    if let Some(geo) = layer_map.layer_geometry(surface) {
                                                        let elem = smithay::wayland::compositor::with_states(surface.wl_surface(), |states| {
                                                            WaylandSurfaceRenderElement::from_surface(
                                                                renderer, surface.wl_surface(), states,
                                                                geo.loc.to_physical_precise_round(1), 1.0,
                                                                smithay::backend::renderer::element::Kind::Unspecified
                                                            )
                                                        });
                                                        if let Ok(Some(e)) = elem {
                                                            elements.push(CompositionElements::Surface(e));
                                                        }
                                                    }
                                                }
                                            }
                                        };

                                        draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Bottom);
                                        draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Background);
                                    }
                                    match damage_tracker.render_output(renderer, &mut frame, render_age, &elements, [0.1, 0.1, 0.1, 1.0]) {
                                        Ok(result) => {
                                            render_success = true;
                                            if let Some(damage) = result.damage {
                                                damage_rects = damage.clone();
                                            }
                                        },
                                        Err(e) => eprintln!("Render error: {:?}", e)
                                    }
                                    if needs_readback {
                                        // Throttling skips readback entirely, so this is always full-size.
                                        let (read_w, read_h) = (width, height);

                                        if !is_memory_throttling {
                                            let _ = renderer.with_context(|gl| unsafe {
                                                gl.ReadPixels(
                                                    0, // x
                                                    0, // y
                                                    read_w,
                                                    read_h,
                                                    smithay::backend::renderer::gles::ffi::RGBA,
                                                    smithay::backend::renderer::gles::ffi::UNSIGNED_BYTE,
                                                    state.frame_buffer.as_mut_ptr() as *mut std::ffi::c_void,
                                                );
                                            });

                                            if state.video_encoder.is_some() {
                                                let w = width as u32;
                                                let h = height as u32;
                                                let src = &state.frame_buffer;

                                                if state.settings.h264_fullcolor {
                                                     let y_size = (w * h) as usize;
                                                     let (y_plane, rest) = state.nv12_buffer.split_at_mut(y_size);
                                                     let (u_plane, v_plane) = rest.split_at_mut(y_size);

                                                     let mut planar_image = yuv::YuvPlanarImageMut {
                                                         y_plane: BufferStoreMut::Borrowed(y_plane),
                                                         y_stride: w,
                                                         u_plane: BufferStoreMut::Borrowed(u_plane),
                                                         u_stride: w,
                                                         v_plane: BufferStoreMut::Borrowed(v_plane),
                                                         v_stride: w,
                                                         width: w,
                                                         height: h,
                                                     };

                                                     let _ = yuv::rgba_to_yuv444(
                                                         &mut planar_image,
                                                         src,
                                                         w * 4,
                                                         YuvRange::Full,
                                                         YuvStandardMatrix::Bt709,
                                                         YuvConversionMode::Balanced
                                                     );
                                                } else {
                                                    let y_size = (w * h) as usize;
                                                    let (y_plane, uv_plane) = state.nv12_buffer.split_at_mut(y_size);

                                                    let mut planar_image = YuvBiPlanarImageMut {
                                                        y_plane: BufferStoreMut::Borrowed(y_plane),
                                                        y_stride: w,
                                                        uv_plane: BufferStoreMut::Borrowed(uv_plane),
                                                        uv_stride: w,
                                                        width: w,
                                                        height: h,
                                                    };

                                                    let _ = yuv::rgba_to_yuv_nv12(
                                                        &mut planar_image,
                                                        src,
                                                        w * 4,
                                                        YuvRange::Limited,
                                                        YuvStandardMatrix::Bt709,
                                                        YuvConversionMode::Balanced
                                                    );
                                                }
                                            }
                                        }
                                    }
                                },
                                Err(e) => eprintln!("Failed to bind buffer: {:?}", e)
                            }
                        }
                    }
                } else {
                    if let Some(renderer) = state.pixman_renderer.as_mut() {
                        let ptr = state.frame_buffer.as_mut_ptr() as *mut u32;
                        let mut image = unsafe {
                            pixman::Image::from_raw_mut(pixman::FormatCode::A8R8G8B8, width as usize, height as usize, ptr, (width * 4) as usize, false).expect("Failed to create pixman image")
                        };
                                    match renderer.bind(&mut image) {
                                    Ok(mut frame) => {
                                        let mut elements: Vec<CompositionElements<PixmanRenderer, WaylandSurfaceRenderElement<PixmanRenderer>>> = Vec::new();

                                        if state.render_cursor_on_framebuffer {
                                            if let Some(pointer) = state.seat.get_pointer() {
                                                let pos = pointer.current_location();
                                                let output_scale_val = output.current_scale().fractional_scale();
                                                let scale = Scale::from(output_scale_val);

                                                if let Some(CursorImageStatus::Named(icon)) = &state.current_cursor_icon {
                                                    let name = wayland::frontend::cursor_icon_to_str(icon);
                                                    let time = Duration::from_millis(state.clock.now().as_millis() as u64);
                                                    if let Some(image) = state.cursor_helper.get_image_by_name(name, output_scale_val.round() as u32, time) {
                                                        if let Some(elem) = state.overlay_state.get_cursor_element(renderer, image, pos.to_i32_round()) {
                                                            elements.push(CompositionElements::Cursor(elem));
                                                        }
                                                    }
                                                } else if let Some(CursorImageStatus::Surface(surface)) = &state.current_cursor_icon {
                                                     let phys_pos = pos.to_physical(scale);
                                                     let elem_result = with_states(surface, |states| {
                                                         WaylandSurfaceRenderElement::from_surface(renderer, surface, states, phys_pos, 1.0, smithay::backend::renderer::element::Kind::Cursor)
                                                     });
                                                     if let Ok(Some(cursor_elem)) = elem_result {
                                                         elements.push(CompositionElements::Surface(cursor_elem));
                                                     }
                                                } else if state.current_cursor_icon.is_none() {
                                                    let time = Duration::from_millis(state.clock.now().as_millis() as u64);
                                                    let image = state.cursor_helper.get_image(output_scale_val.round() as u32, time);
                                                    if let Some(elem) = state.overlay_state.get_cursor_element(renderer, image, pos.to_i32_round()) {
                                                        elements.push(CompositionElements::Cursor(elem));
                                                    }
                                                }
                                            }
                                        }

                                        if let Some(elem) = state.overlay_state.get_watermark_element(renderer) {
                                            elements.push(CompositionElements::Cursor(elem));
                                        }

                                        {
                                            let layer_map = layer_map_for_output(&output);

                                            let draw_layer = |renderer: &mut PixmanRenderer, elements: &mut Vec<CompositionElements<PixmanRenderer, WaylandSurfaceRenderElement<PixmanRenderer>>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
                                                for surface in layer_map.layers() {
                                                    let current_layer = surface.layer();
                                                    if current_layer == target_layer {
                                                        if let Some(geo) = layer_map.layer_geometry(surface) {
                                                            let elem = smithay::wayland::compositor::with_states(surface.wl_surface(), |states| {
                                                                WaylandSurfaceRenderElement::from_surface(
                                                                    renderer, surface.wl_surface(), states,
                                                                    geo.loc.to_physical_precise_round(1), 1.0,
                                                                    smithay::backend::renderer::element::Kind::Unspecified
                                                                )
                                                            });
                                                            if let Ok(Some(e)) = elem {
                                                                elements.push(CompositionElements::Surface(e));
                                                            }
                                                        }
                                                    }
                                                }
                                            };

                                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Overlay);
                                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Top);
                                        }

                                        for window in state.space.elements_for_output(&output).collect::<Vec<_>>().into_iter().rev() {
                                            let loc = state.space.element_location(window).unwrap_or_default();
                                            
                                            if let Some(surface) = window.wl_surface() {
                                                let popups = PopupManager::popups_for_surface(&surface);
                                                for (popup, location) in popups {
                                                    let popup_surface = popup.wl_surface(); {
                                                        let popup_pos = loc + location;
                                                        let elem = smithay::wayland::compositor::with_states(popup_surface, |states| {
                                                            WaylandSurfaceRenderElement::from_surface(
                                                                renderer,
                                                                popup_surface,
                                                                states,
                                                                popup_pos.to_physical_precise_round(1),
                                                                1.0,
                                                                smithay::backend::renderer::element::Kind::Unspecified
                                                            )
                                                        });
                                                        if let Ok(Some(e)) = elem {
                                                            elements.push(CompositionElements::Surface(e));
                                                        }
                                                    }
                                                }
                                            }

                                            elements.extend(window.render_elements(renderer, loc.to_physical_precise_round(1), Scale::from(1.0), 1.0).into_iter().map(CompositionElements::Space));
                                        }

                                        {
                                            let layer_map = layer_map_for_output(&output);

                                            let draw_layer = |renderer: &mut PixmanRenderer, elements: &mut Vec<CompositionElements<PixmanRenderer, WaylandSurfaceRenderElement<PixmanRenderer>>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
                                                for surface in layer_map.layers() {
                                                    let current_layer = surface.layer();
                                                    if current_layer == target_layer {
                                                        if let Some(geo) = layer_map.layer_geometry(surface) {
                                                            let elem = smithay::wayland::compositor::with_states(surface.wl_surface(), |states| {
                                                                WaylandSurfaceRenderElement::from_surface(
                                                                    renderer, surface.wl_surface(), states,
                                                                    geo.loc.to_physical_precise_round(1), 1.0,
                                                                    smithay::backend::renderer::element::Kind::Unspecified
                                                                )
                                                            });
                                                            if let Ok(Some(e)) = elem {
                                                                elements.push(CompositionElements::Surface(e));
                                                            }
                                                        }
                                                    }
                                                }
                                            };

                                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Bottom);
                                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Background);
                                        }

                                let render_age = if state.overlay_state.is_animated() { 0 } else { 1 };
                                match damage_tracker.render_output(renderer, &mut frame, render_age, &elements, [0.1, 0.1, 0.1, 1.0]) {
                                    Ok(result) => {
                                        render_success = true;
                                        if let Some(damage) = result.damage { damage_rects = damage.clone(); }

                                        if state.video_encoder.is_some() && !is_memory_throttling {
                                            let w = width as u32;
                                            let h = height as u32;
                                            let src = &state.frame_buffer;

                                            if state.settings.h264_fullcolor {
                                                 let y_size = (w * h) as usize;
                                                 let (y_plane, rest) = state.nv12_buffer.split_at_mut(y_size);
                                                 let (u_plane, v_plane) = rest.split_at_mut(y_size);

                                                 let mut planar_image = yuv::YuvPlanarImageMut {
                                                     y_plane: BufferStoreMut::Borrowed(y_plane),
                                                     y_stride: w,
                                                     u_plane: BufferStoreMut::Borrowed(u_plane),
                                                     u_stride: w,
                                                     v_plane: BufferStoreMut::Borrowed(v_plane),
                                                     v_stride: w,
                                                     width: w,
                                                     height: h,
                                                 };

                                                 let _ = yuv::bgra_to_yuv444(
                                                     &mut planar_image,
                                                     src,
                                                     w * 4,
                                                     YuvRange::Full,
                                                     YuvStandardMatrix::Bt709,
                                                     YuvConversionMode::Balanced
                                                 );
                                            } else {
                                                let y_size = (w * h) as usize;
                                                let (y_plane, uv_plane) = state.nv12_buffer.split_at_mut(y_size);

                                                let mut planar_image = YuvBiPlanarImageMut {
                                                    y_plane: BufferStoreMut::Borrowed(y_plane),
                                                    y_stride: w,
                                                    uv_plane: BufferStoreMut::Borrowed(uv_plane),
                                                    uv_stride: w,
                                                    width: w,
                                                    height: h,
                                                };

                                                let _ = yuv::bgra_to_yuv_nv12(
                                                    &mut planar_image,
                                                    src,
                                                    w * 4,
                                                    YuvRange::Limited,
                                                    YuvStandardMatrix::Bt709,
                                                    YuvConversionMode::Balanced
                                                );
                                            }
                                        }
                                    },
                                    Err(e) => eprintln!("Render error: {:?}", e)
                                }
                            },
                            Err(e) => eprintln!("Failed to bind pixman image: {:?}", e)
                        }
                    }
                }

                if render_success {
                    let time = state.clock.now();
                    for window in state.space.elements() {
                        window.send_frame(&output, time, Some(Duration::ZERO), |_, _| Some(output.clone()));
                    }

                    if is_memory_throttling {
                    } else if let Some(ref mut encoder) = state.video_encoder {
                        let is_dirty = !damage_rects.is_empty();
                        let is_animated = state.overlay_state.is_animated();

                        let mut send_frame = false;
                        let mut force_idr = false;

                        let normal_qp = state.settings.h264_crf as u32;
                        let paint_qp = state.settings.h264_paintover_crf as u32;
                        let mut target_qp = normal_qp;

                        let trigger_frames = state.settings.paint_over_trigger_frames;
                        let use_paint_over = state.settings.use_paint_over_quality;
                        let burst = state.settings.h264_paintover_burst_frames;
                        let streaming = state.settings.h264_streaming_mode;

                        let st = &mut state.vaapi_state;

                        if st.h264_burst_frames_remaining > 0 {
                            send_frame = true;
                            target_qp = paint_qp;
                            st.h264_burst_frames_remaining -= 1;

                            if is_dirty {
                                st.h264_burst_frames_remaining = 0;
                                st.paint_over_sent = false;
                                target_qp = normal_qp;
                            }
                        }

                        if !send_frame && (streaming || is_animated) {
                            send_frame = true;
                        }

                        // Periodic recovery keyframe. The HW encoders use an effectively infinite
                        // GOP and the Wayland path has no request-IDR channel, so a (re)connecting
                        // client can only start decoding once an IDR is forced. Keep the author's
                        // intent of NOT emitting a keyframe every second on a static screen by
                        // spacing these ~2s apart (vs the pre-commit ~1s), which still bounds
                        // reconnect-recovery latency instead of leaving it at one IDR per u16 wrap
                        // (~18 min @ 60fps).
                        // Clamp fps before the cast: target_fps<=0 would make kf_interval 1 (an IDR
                        // every frame). Matches the frame-pacing clamp below.
                        let safe_fps = state.settings.target_fps.max(1.0);
                        let kf_interval = ((safe_fps * 2.0).round() as u64).max(1);
                        let periodic_idr = (state.frame_counter as u64 % kf_interval) == 0;
                        // A recovery keyframe is due either on real motion, the first frame, the
                        // periodic cadence, or an on-demand request.
                        let recovery_idr = state.frame_counter == 0 || periodic_idr || requested_idr;
                        if is_dirty {
                            // Real motion: full reset of paint-over bookkeeping (the screen changed).
                            send_frame = true;
                            force_idr = recovery_idr;
                            st.no_motion_frame_count = 0;
                            st.paint_over_sent = false;
                            st.h264_burst_frames_remaining = 0;
                            target_qp = normal_qp;
                        } else if recovery_idr {
                            // Recovery keyframe on a STATIC screen. Do NOT reset no_motion_frame_count
                            // / paint_over_sent here -- that restarts the paint-over countdown every
                            // ~2s and can starve it (those reset only on real motion above). Leave an
                            // in-flight burst untouched; override QP only if none is running.
                            send_frame = true;
                            force_idr = true;
                            if st.h264_burst_frames_remaining <= 0 {
                                target_qp = normal_qp;
                            }
                        } else if !send_frame {
                            st.no_motion_frame_count += 1;

                            if use_paint_over && st.no_motion_frame_count >= trigger_frames && !st.paint_over_sent && paint_qp < normal_qp {
                                send_frame = true;
                                st.paint_over_sent = true;
                                force_idr = true;
                                target_qp = paint_qp;
                                st.h264_burst_frames_remaining = burst - 1;
                            }
                        }

                        if send_frame {
                            let force_idr_for_recording = state
                                .recording_sink
                                .as_ref()
                                .map(|s| s.should_force_idr())
                                .unwrap_or(false);
                            let force_idr = force_idr || force_idr_for_recording;
                            let result = match encoder {
                                GpuEncoder::Nvenc(enc) => {
                                    if needs_readback {
                                        enc.encode_raw(&state.nv12_buffer, state.frame_counter as u64, target_qp, force_idr)
                                    } else {
                                        if let Some((_, ref dmabuf)) = state.offscreen_buffer {
                                            enc.encode(dmabuf, state.frame_counter as u64, target_qp, force_idr)
                                        } else {
                                            Err("NVENC ZeroCopy requires offscreen buffer (GPU context)".to_string())
                                        }
                                    }
                                },
                                GpuEncoder::Vaapi(enc) => {
                                    if needs_readback {
                                        enc.encode_raw(&state.nv12_buffer, state.frame_counter as u64, target_qp, force_idr)
                                    } else {
                                        if let Some((_, ref dmabuf)) = state.offscreen_buffer {
                                            enc.encode_dmabuf(dmabuf, state.frame_counter as u64, target_qp, force_idr)
                                        } else {
                                            Err("Vaapi ZeroCopy requires offscreen buffer (GPU context)".to_string())
                                        }
                                    }
                                }
                            };

                            if let Ok(data) = result {
                                if !data.is_empty() {
                                    state.encoded_frame_count += 1;
                                    state.total_stripes_encoded += 1;
                                    if let Some(ref cb) = state.callback {
                                        Python::attach(|py| {
                                            // MOVE the encoded buffer into the frame (no copy); full-frame
                                            // H.264, so metadata as attrs (y_start=0, height=full frame).
                                            match Py::new(py, WaylandFrame::new_owned_meta(
                                                data, 2, 0, height, state.frame_counter as i32,
                                            )) {
                                                Ok(py_frame) => {
                                                    if let Err(e) = cb.call1(py, (py_frame,)) { eprintln!("Callback error: {:?}", e); }
                                                }
                                                Err(e) => eprintln!("Frame alloc error: {:?}", e),
                                            }
                                        });
                                    }
                                }
                            } else if let Err(e) = result {
                                eprintln!("HW Encode Error: {}", e);
                            }
                        }
                    } else {
                        if state.overlay_state.is_animated() {
                             damage_rects.push(Rectangle::new((0,0).into(), (width, height).into()));
                        }

                        // Give the software H.264 path the same IDR triggers as the GPU path
                        // (request_idr + ~2s periodic recovery); without it CPU has no IDR channel.
                        // Clamp fps before the cast: target_fps<=0 would make kf_interval 1 (an IDR
                        // every frame). Matches the frame-pacing clamp below.
                        let safe_fps = state.settings.target_fps.max(1.0);
                        let kf_interval = ((safe_fps * 2.0).round() as u64).max(1);
                        let periodic_idr = (state.frame_counter as u64 % kf_interval) == 0;
                        // Only meaningful for H.264 (output_mode 1); encode_cpu ignores it for JPEG.
                        let force_idr_all = state.settings.output_mode == 1
                            && (state.frame_counter == 0 || periodic_idr || requested_idr);

                        let encoded_packets = encoders::software::encode_cpu(
                            &mut state.stripes,
                            &state.frame_buffer,
                            width,
                            height,
                            &damage_rects,
                            &state.settings,
                            state.frame_counter,
                            state.use_gpu,
                            state.recording_sink.as_ref(),
                            force_idr_all,
                        );

                        if !encoded_packets.is_empty() {
                            state.encoded_frame_count += 1;
                            state.total_stripes_encoded += encoded_packets.len() as u32;
                            if let Some(ref cb) = state.callback {
                                Python::attach(|py| {
                                    for packet in encoded_packets {
                                        // MOVE each packet into its frame (no copy); metadata as attrs.
                                        match Py::new(py, WaylandFrame::new_owned_meta(
                                            packet.data, packet.data_type, packet.stripe_y_start,
                                            packet.stripe_height, packet.frame_id,
                                        )) {
                                            Ok(py_frame) => {
                                                if let Err(e) = cb.call1(py, (py_frame,)) { eprintln!("Callback error: {:?}", e); }
                                            }
                                            Err(e) => eprintln!("Frame alloc error: {:?}", e),
                                        }
                                    }
                                });
                            }
                        }
                    }
                    // Consume the on-demand IDR request only now that an encode pass actually ran;
                    // if throttling, leave it set for the next frame rather than dropping it.
                    if !is_memory_throttling {
                        state.pending_force_idr = false;
                    }
                    state.frame_counter = state.frame_counter.wrapping_add(1);
                }
            }
            let work_elapsed = loop_start_time.elapsed();
            // Clamp to a positive, finite fps; Duration::from_secs_f64 panics on inf/negative.
            let fps = (if is_memory_throttling { 5.0 } else { state.settings.target_fps }).max(1.0);
            let target_frame_duration = Duration::from_secs_f64(1.0 / fps);
            let wait_duration = target_frame_duration.saturating_sub(work_elapsed);
            let final_wait = if wait_duration.as_millis() < 1 { Duration::from_millis(1) } else { wait_duration };
            TimeoutAction::ToDuration(final_wait)
        })
        .expect("Failed to init capture timer");

    event_loop
        .handle()
        .insert_source(Generic::new(display, Interest::READ, Mode::Level), |_, display, _state| {
            unsafe { display.get_mut().dispatch_clients(_state).unwrap(); }
            Ok(PostAction::Continue)
        })
        .unwrap();

    event_loop.run(None, &mut state, |state| { state.dh.flush_clients().unwrap(); }).unwrap();
}

/// Zero-copy encoded-frame handoff to Python. Owns the encoded `Vec<u8>` and
/// exposes it read-only via the buffer protocol, so `bytes(frame)` /
/// `memoryview(frame)` alias the Rust buffer instead of copying. Mirrors the
/// X11 C-API `StripeFrame` (same 4 int attrs).
#[pyclass]
struct WaylandFrame {
    data: Vec<u8>,
    #[pyo3(get, set)]
    data_type: i32,
    #[pyo3(get, set)]
    stripe_y_start: i32,
    #[pyo3(get, set)]
    stripe_height: i32,
    #[pyo3(get, set)]
    frame_id: i32,
}

impl WaylandFrame {
    /// Hot-path constructor: MOVES the encoded buffer in (no copy) and carries stripe
    /// metadata as attributes, so the consumer can read it without parsing a header
    /// (required for omit_stripe_headers).
    fn new_owned_meta(data: Vec<u8>, data_type: i32, stripe_y_start: i32, stripe_height: i32, frame_id: i32) -> Self {
        Self { data, data_type, stripe_y_start, stripe_height, frame_id }
    }
}

#[pymethods]
impl WaylandFrame {
    // Symmetry/testability: copies the bytes-like into the owned Vec. The hot
    // path uses `new_owned_meta` (a move) instead.
    #[new]
    #[pyo3(signature = (data, data_type = 0, stripe_y_start = 0, stripe_height = 0, frame_id = 0))]
    fn new(data: Vec<u8>, data_type: i32, stripe_y_start: i32, stripe_height: i32, frame_id: i32) -> Self {
        Self { data, data_type, stripe_y_start, stripe_height, frame_id }
    }

    fn __len__(&self) -> usize {
        self.data.len()
    }

    // PyBuffer_FillInfo INCREFs `slf` into view->obj, pinning the Vec until every
    // view is released, so memoryviews can outlive the Python `frame` handle.
    unsafe fn __getbuffer__(
        slf: PyRefMut<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        let r = pyo3::ffi::PyBuffer_FillInfo(
            view,
            slf.as_ptr(),
            slf.data.as_ptr() as *mut std::os::raw::c_void,
            slf.data.len() as pyo3::ffi::Py_ssize_t,
            1, // readonly
            flags,
        );
        if r != 0 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut pyo3::ffi::Py_buffer) {}
}

/// @brief Python interface class.
///
/// This class is exposed to Python and spawns the Wayland thread upon instantiation.
/// It provides methods to control the capture session and inject input events.
#[pyclass]
struct WaylandBackend {
    tx: smithay::reexports::calloop::channel::Sender<ThreadCommand>,
}

#[pymethods]
impl WaylandBackend {
    #[new]
    fn new(width: i32, height: i32, dri_node: String) -> Self {
        let (tx, rx) = smithay::reexports::calloop::channel::channel();
        thread::spawn(move || {
            run_wayland_thread(rx, width, height, dri_node);
        });
        WaylandBackend { tx }
    }

    fn start_capture(&self, callback: Py<PyAny>, settings: &Bound<'_, PyAny>) -> PyResult<()> {
        let watermark_path_obj = settings.getattr("watermark_path")?;
        let watermark_path = if let Ok(s) = watermark_path_obj.extract::<String>() {
            s
        } else if let Ok(b) = watermark_path_obj.extract::<Vec<u8>>() {
            String::from_utf8_lossy(&b).into_owned()
        } else {
            String::new()
        };

        let scale = settings.getattr("scale").ok().and_then(|x| x.extract().ok()).unwrap_or(1.0);

        let rust_settings = RustCaptureSettings {
            width: settings.getattr("capture_width")?.extract()?,
            height: settings.getattr("capture_height")?.extract()?,
            scale,
            capture_x: settings.getattr("capture_x")?.extract()?,
            capture_y: settings.getattr("capture_y")?.extract()?,
            target_fps: settings.getattr("target_fps")?.extract()?,
            jpeg_quality: settings.getattr("jpeg_quality")?.extract()?,
            paint_over_jpeg_quality: settings.getattr("paint_over_jpeg_quality")?.extract()?,
            use_paint_over_quality: settings.getattr("use_paint_over_quality")?.extract()?,
            paint_over_trigger_frames: settings.getattr("paint_over_trigger_frames")?.extract()?,
            damage_block_threshold: settings.getattr("damage_block_threshold")?.extract()?,
            damage_block_duration: settings.getattr("damage_block_duration")?.extract()?,
            output_mode: settings.getattr("output_mode")?.extract()?,
            h264_crf: settings.getattr("h264_crf")?.extract()?,
            h264_paintover_crf: settings.getattr("h264_paintover_crf")?.extract()?,
            h264_paintover_burst_frames: settings.getattr("h264_paintover_burst_frames")?.extract()?,
            h264_fullcolor: settings.getattr("h264_fullcolor")?.extract()?,
            h264_fullframe: settings.getattr("h264_fullframe")?.extract()?,
            h264_streaming_mode: settings.getattr("h264_streaming_mode")?.extract()?,
            capture_cursor: settings.getattr("capture_cursor")?.extract()?,
            watermark_path,
            watermark_location_enum: settings.getattr("watermark_location_enum")?.extract()?,
            vaapi_render_node_index: settings.getattr("vaapi_render_node_index")?.extract()?,
            use_cpu: settings.getattr("use_cpu")?.extract()?,
            debug_logging: settings.getattr("debug_logging")?.extract()?,
            recording_socket: settings
                .getattr("recording_socket")
                .ok()
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_default(),
            // When true, encoders emit the raw payload without the per-stripe header (X11 parity).
            // Stripe metadata is still exposed on WaylandFrame attributes, so the consumer must read
            // it from there rather than parsing header bytes when this is set.
            omit_stripe_headers: settings
                .getattr("omit_stripe_headers")
                .ok()
                .and_then(|v| v.extract::<bool>().ok())
                .unwrap_or(false),
        };

        self.tx
            .send(ThreadCommand::StartCapture(callback, rust_settings))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to send start command: {}", e)))?;
        Ok(())
    }

    fn stop_capture(&self) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::StopCapture)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to send stop command: {}", e)))?;
        Ok(())
    }

    fn set_cursor_callback(&self, callback: Py<PyAny>) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::SetCursorCallback(callback))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set cursor callback: {}", e)))?;
        Ok(())
    }

    fn inject_key(&self, scancode: u32, state: u32) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::KeyboardKey { scancode, state })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject key: {}", e)))?;
        Ok(())
    }

    /// Inject a key by X11/XKB keysym (e.g. 0x41 'A', 0xFF0D Return), resolved against our own
    /// xkb keymap. Prefer over `inject_key` when you have a keysym. A shifted keysym gets a
    /// synthetic Shift press/release. `state`: 1 = press, 0 = release.
    fn inject_keysym(&self, keysym: u32, state: u32) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::KeyboardKeysym { keysym, state })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject keysym: {}", e)))?;
        Ok(())
    }

    /// Return the active xkb keymap as an XKB_KEYMAP_FORMAT_TEXT_V1 string so a consumer can build
    /// a reverse keysym->keycode map from the identical keymap. Empty string if it can't be read.
    fn get_xkb_keymap_string(&self, py: Python<'_>) -> PyResult<String> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel::<String>();
        self.tx
            .send(ThreadCommand::GetXkbKeymap { reply: reply_tx })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to request keymap: {}", e)))?;
        // Release the GIL while waiting (the wayland thread can call back into Python -> deadlock);
        // move the owned Receiver in (it's Send) and bound the wait so a stall can't hang us.
        let result = py.detach(move || reply_rx.recv_timeout(Duration::from_secs(2)));
        match result {
            Ok(s) => Ok(s),
            Err(_) => Ok(String::new()),
        }
    }

    fn inject_mouse_move(&self, x: f64, y: f64) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::PointerMotion { x, y })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject motion: {}", e)))?;
        Ok(())
    }

    fn inject_relative_mouse_move(&self, dx: f64, dy: f64) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::PointerRelativeMotion { dx, dy })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject relative motion: {}", e)))?;
        Ok(())
    }

    fn inject_mouse_button(&self, btn: u32, state: u32) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::PointerButton { btn, state })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject button: {}", e)))?;
        Ok(())
    }

    fn inject_mouse_scroll(&self, x: f64, y: f64) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::PointerAxis { x, y })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject axis: {}", e)))?;
        Ok(())
    }

    fn set_cursor_rendering(&self, enabled: bool) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::UpdateCursorConfig { render_on_framebuffer: enabled })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set cursor config: {}", e)))?;
        Ok(())
    }

    /// Forces an IDR/keyframe on the next captured frame so a (re)connecting client
    /// or a decoder reset can resume immediately instead of waiting for the periodic
    /// recovery keyframe. No-op cost on the JPEG/software path (keyframes are N/A).
    fn request_idr_frame(&self) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::RequestIdr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to request IDR: {}", e)))?;
        Ok(())
    }
}

#[pymodule]
fn pixelflux_wayland(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WaylandBackend>()?;
    m.add_class::<WaylandFrame>()?;
    Ok(())
}

#[cfg(test)]
mod keysym_release_replay_tests {
    //! Focused tests for fix #5-rust: `inject_keysym` records the keycodes injected at
    //! PRESS time into `synthetic_shift_keysyms` (a `HashMap<u32, (u32, u32)>`) so the
    //! matching key-up releases the SAME physical keycodes, even if the active xkb layout
    //! changed mid-keystroke. The release path must read the recorded mapping and must NOT
    //! re-resolve the keysym against the (possibly different) live layout.
    //!
    //! The real record/replay lives inside the calloop loop in `run_wayland_thread`
    //! (src/lib.rs ~963-1022) and resolves against a live `xkb::Keymap`, which needs a
    //! connected keyboard. These tests model the identical state machine with the same map
    //! type and a deterministic 2-layout resolver so the invariant is provable in isolation.

    use std::collections::HashMap;

    /// Simulated key-up actions emitted by the release path, in order.
    /// Mirrors the real `inject(state, kc, KeyState::Released)` calls.
    #[derive(Debug, PartialEq, Eq)]
    enum Release {
        Key(u32),
        Shift(u32),
    }

    /// Deterministic stand-in for `resolve_keysym_to_keycode`, parameterized by layout.
    /// Returns (keycode, needs_shift). Two layouts deliberately map the same keysym to
    /// DIFFERENT keycodes / shift-requirements to model a layout switch mid-keystroke.
    ///
    /// Layout 0 ("us"):    keysym 0x41 ('A') -> kc 38, needs_shift=true ; Shift_L -> kc 50
    /// Layout 1 ("de"):    keysym 0x41 ('A') -> kc 24, needs_shift=false (different key!)
    fn resolve(layout: u32, keysym: u32) -> Option<(u32, bool)> {
        match (layout, keysym) {
            (0, 0x41) => Some((38, true)),
            (0, 0xFFE1) => Some((50, false)), // Shift_L on layout 0
            (1, 0x41) => Some((24, false)),
            (1, 0xFFE1) => Some((62, false)), // Shift_L on layout 0 -> different kc on layout 1
            (_, 0xFF0D) => Some((36, false)), // Return: same on both, never shifted
            _ => None,
        }
    }

    const SHIFT_L: u32 = 0xFFE1;

    /// Models the PRESS branch (src/lib.rs ~964-1008): resolve against the *current* layout,
    /// then RECORD (kc, shift_kc_or_0) keyed by keysym. Returns false if unresolved.
    fn press(map: &mut HashMap<u32, (u32, u32)>, current_layout: u32, keysym: u32) -> bool {
        let Some((kc, needs_shift)) = resolve(current_layout, keysym) else {
            return false;
        };
        let shift_kc = if needs_shift {
            resolve(current_layout, SHIFT_L).map(|(kc, _)| kc).unwrap_or(50)
        } else {
            0
        };
        map.insert(keysym, (kc, if needs_shift { shift_kc } else { 0 }));
        true
    }

    /// Models the FIXED RELEASE branch (src/lib.rs ~1010-1021): read the recorded keycodes
    /// and release exactly those; key first, then synthetic Shift if shift_kc != 0.
    fn release_fixed(map: &mut HashMap<u32, (u32, u32)>, keysym: u32) -> Vec<Release> {
        let mut out = Vec::new();
        if let Some((kc, shift_kc)) = map.remove(&keysym) {
            out.push(Release::Key(kc));
            if shift_kc != 0 {
                out.push(Release::Shift(shift_kc));
            }
        }
        out
    }

    /// Models the BUGGY alternative the fix replaces: ignore the record and RE-RESOLVE
    /// against the live (possibly changed) layout. Kept only to prove the fix diverges from
    /// it exactly when the layout changes mid-keystroke.
    fn release_reresolve(current_layout: u32, keysym: u32) -> Vec<Release> {
        let mut out = Vec::new();
        if let Some((kc, needs_shift)) = resolve(current_layout, keysym) {
            out.push(Release::Key(kc));
            if needs_shift {
                let shift_kc = resolve(current_layout, SHIFT_L).map(|(kc, _)| kc).unwrap_or(50);
                out.push(Release::Shift(shift_kc));
            }
        }
        out
    }

    #[test]
    fn press_records_keycode_and_shift() {
        // 'A' on layout 0 needs Shift -> must record both the key kc and the Shift_L kc.
        let mut map = HashMap::new();
        assert!(press(&mut map, 0, 0x41));
        assert_eq!(map.get(&0x41), Some(&(38u32, 50u32)));
    }

    #[test]
    fn press_records_zero_shift_when_unshifted() {
        // Return is unshifted -> shift_kc sentinel must be 0 so release skips synthetic Shift.
        let mut map = HashMap::new();
        assert!(press(&mut map, 0, 0xFF0D));
        assert_eq!(map.get(&0xFF0D), Some(&(36u32, 0u32)));
    }

    #[test]
    fn unresolved_keysym_records_nothing() {
        let mut map = HashMap::new();
        assert!(!press(&mut map, 0, 0xDEAD));
        assert!(map.is_empty());
    }

    #[test]
    fn release_replays_recorded_keycodes_not_a_reresolve() {
        // Press 'A' on layout 0 (kc 38 + Shift 50). THEN the layout switches to 1 mid-keystroke.
        let mut map = HashMap::new();
        assert!(press(&mut map, 0, 0x41));

        let layout_at_release = 1; // user/compositor switched layout after press

        let fixed = release_fixed(&mut map, 0x41);
        let buggy = release_reresolve(layout_at_release, 0x41);

        // The FIX releases exactly what was pressed: kc 38 then Shift 50.
        assert_eq!(fixed, vec![Release::Key(38), Release::Shift(50)]);

        // The buggy re-resolve would release kc 24 (layout 1's 'A') and NO shift -> kc 38 and
        // Shift 50 stay logically held down. Prove the two paths diverge here.
        assert_eq!(buggy, vec![Release::Key(24)]);
        assert_ne!(fixed, buggy, "fix must NOT match the re-resolve path under a layout switch");
    }

    #[test]
    fn release_consumes_the_record_no_double_release() {
        // remove() must take the entry so a duplicate key-up is a no-op (no phantom release).
        let mut map = HashMap::new();
        assert!(press(&mut map, 0, 0x41));
        let first = release_fixed(&mut map, 0x41);
        assert_eq!(first, vec![Release::Key(38), Release::Shift(50)]);
        assert!(map.is_empty());
        let second = release_fixed(&mut map, 0x41);
        assert!(second.is_empty(), "second key-up must release nothing");
    }

    #[test]
    fn release_without_prior_press_is_noop() {
        // A stray key-up that was never recorded must not inject anything.
        let mut map = HashMap::new();
        assert!(release_fixed(&mut map, 0x41).is_empty());
    }

    #[test]
    fn no_layout_change_fix_and_reresolve_agree() {
        // Sanity: when the layout is stable, the fix and the (would-be) re-resolve must agree,
        // so the fix is not changing correct behavior in the common case.
        let mut map = HashMap::new();
        assert!(press(&mut map, 0, 0x41));
        let fixed = release_fixed(&mut map, 0x41);
        let same_layout = release_reresolve(0, 0x41);
        assert_eq!(fixed, same_layout);
    }
}
