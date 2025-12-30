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
use pyo3::types::{PyAny, PyBytes, PyModule};
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
            Bind, ExportMem, ImportAll, ImportEgl, ImportMem,
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

pub use encoders::nvenc;
pub use encoders::software::StripeState;
pub use encoders::vaapi;

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
        }
    }
}

pub enum ThreadCommand {
    StartCapture(Py<PyAny>, RustCaptureSettings),
    StopCapture,
    SetCursorCallback(Py<PyAny>),
    KeyboardKey { scancode: u32, state: u32 },
    PointerMotion { x: f64, y: f64 },
    PointerRelativeMotion { dx: f64, dy: f64 },
    PointerButton { btn: u32, state: u32 },
    PointerAxis { x: f64, y: f64 },
    UpdateCursorConfig { render_on_framebuffer: bool },
}

fn get_gpu_driver(card_index: i32) -> String {
    let path = format!("/sys/class/drm/renderD{}/device/driver", 128 + card_index);
    match std::fs::read_link(&path) {
        Ok(link_path) => link_path.to_string_lossy().to_lowercase(),
        Err(_) => String::new(),
    }
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
fn run_wayland_thread(command_rx: smithay::reexports::calloop::channel::Channel<ThreadCommand>) {
    let mut width: i32 = 1024;
    let mut height: i32 = 768;

    if let Ok(res_str) = std::env::var("MAX_RES") {
        let parts: Vec<&str> = res_str.split('x').collect();
        if parts.len() == 2 {
            if let (Ok(w), Ok(h)) = (parts[0].parse::<i32>(), parts[1].parse::<i32>()) {
                width = w;
                height = h;
                println!("[Wayland] Resolution set via MAX_RES: {}x{}", width, height);
            }
        }
    }
    if let Ok(w_str) = std::env::var("SELKIES_MANUAL_WIDTH") {
        if let Ok(w) = w_str.parse::<i32>() {
            width = w;
            println!("[Wayland] Width override via SELKIES_MANUAL_WIDTH: {}", width);
        }
    }

    if let Ok(h_str) = std::env::var("SELKIES_MANUAL_HEIGHT") {
        if let Ok(h) = h_str.parse::<i32>() {
            height = h;
            println!("[Wayland] Height override via SELKIES_MANUAL_HEIGHT: {}", height);
        }
    }

    let mut event_loop = EventLoop::<AppState>::try_new().expect("Unable to create event_loop");
    let display: Display<AppState> = Display::new().unwrap();
    let dh: DisplayHandle = display.handle();

    let dri_node = std::env::var("DRINODE").unwrap_or_default();
    let use_gpu = !dri_node.is_empty();

    let mut gles_renderer = None;
    let mut pixman_renderer = None;
    let mut offscreen_buffer: Option<(BufferObject<()>, Dmabuf)> = None;
    let mut dmabuf_global = None;
    let mut gbm_device_raw = None;
    let mut dmabuf_state = DmabufState::new();

    if use_gpu {
        println!("[Wayland] Initializing GL Renderer using device: {}", dri_node);
        let device_path = std::path::Path::new(&dri_node);

        let file = File::options()
            .read(true)
            .write(true)
            .open(device_path)
            .expect("Failed to open render device");

        let file_for_alloc = file
            .try_clone()
            .expect("Failed to clone file for GBM Allocator");
        let gbm_allocator =
            RawGbmDevice::new(file_for_alloc).expect("Failed to create Raw GBM Device");

        let gbm = GbmDevice::new(file).expect("Failed to create GBM device");
        let egl = unsafe { EGLDisplay::new(gbm) }.expect("Failed to create EGL display");
        let context = EGLContext::new(&egl).expect("Failed to create EGL context");

        let mut renderer =
            unsafe { GlesRenderer::new(context) }.expect("Failed to init GlesRenderer");
        if let Err(e) = renderer.bind_wl_display(&dh) {
            println!("[Wayland] Warning: Failed to bind EGL to Wayland Display (Optional): {:?}", e);
        }

        let formats = Bind::<Dmabuf>::supported_formats(&renderer)
            .expect("Failed to query formats")
            .into_iter()
            .collect::<Vec<_>>();

        let node = DrmNode::from_path(device_path).expect("Failed to create DrmNode");
        let dmabuf_default_feedback =
            DmabufFeedbackBuilder::new(node.dev_id(), formats.clone()).build();

        dmabuf_global = Some(if let Ok(default_feedback) = dmabuf_default_feedback {
            dmabuf_state.create_global_with_default_feedback::<AppState>(&dh, &default_feedback)
        } else {
            dmabuf_state.create_global::<AppState>(&dh, formats)
        });

        let bo = gbm_allocator
            .create_buffer_object(
                width as u32,
                height as u32,
                GbmFormat::Argb8888,
                BufferObjectFlags::RENDERING,
            )
            .expect("Failed to allocate GBM buffer");

        let dmabuf = create_dmabuf_from_bo(&bo);
        offscreen_buffer = Some((bo, dmabuf));

        gbm_device_raw = Some(gbm_allocator);
        gles_renderer = Some(renderer);
    } else {
        println!("[Wayland] DRINODE unset. Initializing Software Renderer (Pixman).");
        pixman_renderer = Some(PixmanRenderer::new().expect("Failed to init PixmanRenderer"));
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
        use_gpu,
        video_encoder: None,
        vaapi_state: StripeState::default(),
        cursor_helper: Cursor::load(),
        overlay_state: OverlayState::default(),
        current_cursor_icon: None,
        cursor_buffer: None,
        cursor_cache: std::collections::HashMap::new(),
        render_cursor_on_framebuffer: false,
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
                CalloopEvent::Msg(ThreadCommand::StartCapture(cb, settings)) => {
                    if let Some(output) = state.outputs.first() {
                        let current_mode = output.current_mode().unwrap();
                        let current_w = current_mode.size.w;
                        let current_h = current_mode.size.h;
                        let current_scale = output.current_scale().fractional_scale();
                        let current_refresh = current_mode.refresh;
                        let target_refresh = (settings.target_fps * 1000.0).round() as i32;

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

                            let scale = settings.scale.max(0.1);
                            let logical_width = (settings.width as f64 / scale).round() as i32;
                            let logical_height = (settings.height as f64 / scale).round() as i32;

                            for window in state.space.elements() {
                                if let Some(surface) = window.wl_surface() {
                                    output.enter(&surface);
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

                            let pixel_count = (settings.width * settings.height) as usize;
                            state.frame_buffer = vec![0u8; pixel_count * 4];
                            if settings.h264_fullcolor {
                                state.nv12_buffer = vec![0u8; pixel_count * 3];
                            } else {
                                state.nv12_buffer = vec![0u8; pixel_count * 3 / 2];
                            }

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

                            match NvencEncoder::new(&settings, egl_display) {
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
                                match VaapiEncoder::new(&settings) {
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

                    let watermark_active = !settings.watermark_path.is_empty()
                        && settings.watermark_location_enum != 0;
                    let mut different_gpu = false;

                    if state.video_encoder.is_some() {
                        let dri_node = std::env::var("DRINODE").unwrap_or_default();
                        let encode_node_idx = settings.vaapi_render_node_index;
                        if !dri_node.is_empty() && encode_node_idx >= 0 {
                            if !dri_node.contains(&format!("renderD{}", 128 + encode_node_idx)) {
                                different_gpu = true;
                            }
                        }
                    }

                    if watermark_active {
                        println!("[Wayland] Decision: Watermark is active -> Forcing Readback (CPU path for pixels).");
                    }
                    if different_gpu {
                        println!("[Wayland] Decision: Rendering and Encoding GPUs differ -> Forcing Readback (CPU path for pixels).");
                    }
                    if state.video_encoder.is_none() {
                        println!("[Wayland] Decision: No GPU Encoder available -> Using CPU Software Encoding.");
                    } else if !watermark_active && !different_gpu {
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

                    state.overlay_state.load_watermark(&settings.watermark_path);
                    state.callback = Some(cb);
                    state.is_capturing = true;
                    state.settings = settings.clone();
                    state.encoded_frame_count = 0;
                    state.total_stripes_encoded = 0;
                    state.last_log_time = Instant::now();
                    state.frame_counter = 0;
                    state.stripes.clear();
                    state.vaapi_state = StripeState::default();
                }
                CalloopEvent::Msg(ThreadCommand::StopCapture) => {
                    println!("[Wayland] Capture loop stopped.");
                    state.is_capturing = false;
                    state.callback = None;
                    state.video_encoder = None;
                }
                CalloopEvent::Msg(ThreadCommand::SetCursorCallback(cb)) => {
                    state.cursor_callback = Some(cb);
                }
                CalloopEvent::Msg(ThreadCommand::KeyboardKey { scancode, state: key_state_val }) => {
                    let key_state = if key_state_val > 0 { KeyState::Pressed } else { KeyState::Released };
                    let serial = next_serial();
                    let time = wayland_time();
                    if let Some(keyboard) = state.seat.get_keyboard() {
                        keyboard.input(state, Keycode::new(scancode), key_state, serial, time, |_, _, _| {
                            FilterResult::<()>::Forward
                        });
                    }
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
                        pointer.button(state, &ButtonEvent { button: btn, state: button_state, serial, time });
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
    event_loop
        .handle()
        .insert_source(timer, move |_, _, state| {
            let loop_start_time = Instant::now();
            state.space.refresh();

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
                        let watermark_active = state.settings.watermark_location_enum != 0 && !state.settings.watermark_path.is_empty();
                        let mut different_gpu = false;

                        if state.video_encoder.is_some() {
                            let dri_node = std::env::var("DRINODE").unwrap_or_default();
                            let encode_node_idx = state.settings.vaapi_render_node_index;
                            if !dri_node.is_empty() && encode_node_idx >= 0 {
                                if !dri_node.contains(&format!("renderD{}", 128 + encode_node_idx)) {
                                    different_gpu = true;
                                }
                            }
                        }

                        let is_readback = !rendering_gpu || !encoding_gpu_avail || watermark_active || different_gpu;
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

                    println!("Res: {}x{} Mode: {} Stripes: {} EncFPS: {:.2} EncStripes/s: {:.2}",
                        state.settings.width, state.settings.height, mode_str, n_stripes, actual_fps, stripes_per_sec);
                }
                state.encoded_frame_count = 0;
                state.total_stripes_encoded = 0;
                state.last_log_time = now;
            }

            if !state.is_capturing {
                return TimeoutAction::ToDuration(Duration::from_millis(16));
            }

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
                let watermark_active = state.settings.watermark_location_enum != 0 && !state.settings.watermark_path.is_empty();

                let mut different_gpu = false;
                if state.video_encoder.is_some() {
                    let dri_node = std::env::var("DRINODE").unwrap_or_default();
                    let encode_node_idx = state.settings.vaapi_render_node_index;
                    if !dri_node.is_empty() && encode_node_idx >= 0 {
                        if !dri_node.contains(&format!("renderD{}", 128 + encode_node_idx)) {
                            different_gpu = true;
                        }
                    }
                }

                let needs_readback = !rendering_gpu || !encoding_gpu_avail || watermark_active || different_gpu;

                if rendering_gpu {
                    if let Some(renderer) = state.gles_renderer.as_mut() {
                        if let Some((_bo, dmabuf)) = state.offscreen_buffer.as_mut() {
                            match renderer.bind(dmabuf) {
                                Ok(mut frame) => {
                                    let mut elements: Vec<CompositionElements<GlesRenderer, WaylandSurfaceRenderElement<GlesRenderer>>> = Vec::new();

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

                                        draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Background);
                                        draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Bottom);
                                    }

                                    if let Ok(space_elements) = smithay::desktop::space::space_render_elements(renderer, [&state.space], &output, 1.0) {
                                        elements.extend(space_elements.into_iter().map(CompositionElements::Space));
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

                                        draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Top);
                                        draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Overlay);
                                    }

                                    for window in state.space.elements() {
                                        if let Some(surface) = window.wl_surface() {
                                            let window_loc = state.space.element_location(window).unwrap_or_default();
                                            let popups = PopupManager::popups_for_surface(&surface);
                                            
                                            for (popup, location) in popups {
                                                let popup_surface = popup.wl_surface(); {
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
                                        }
                                    }

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
                                        let read_rect = Rectangle::new((0,0).into(), (width, height).into());
                                        if let Ok(mapping) = renderer.copy_framebuffer(&mut frame, read_rect, Fourcc::Abgr8888) {
                                            if let Ok(data) = renderer.map_texture(&mapping) {
                                                state.frame_buffer.copy_from_slice(data);
                                                state.overlay_state.apply(&mut state.frame_buffer, width, height, false);

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
                                        let mut elements: Vec<WaylandSurfaceRenderElement<PixmanRenderer>> = Vec::new();

                                        {
                                            let layer_map = layer_map_for_output(&output);

                                            let draw_layer = |renderer: &mut PixmanRenderer, elements: &mut Vec<WaylandSurfaceRenderElement<PixmanRenderer>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
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
                                                                elements.push(e);
                                                            }
                                                        }
                                                    }
                                                }
                                            };

                                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Background);
                                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Bottom);
                                        }

                                        for window in state.space.elements_for_output(&output) {
                                            let loc = state.space.element_location(window).unwrap_or_default();
                                            elements.extend(window.render_elements(renderer, loc.to_physical_precise_round(1), Scale::from(1.0), 1.0));
                                            
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
                                                            elements.push(e);
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        {
                                            let layer_map = layer_map_for_output(&output);

                                            let draw_layer = |renderer: &mut PixmanRenderer, elements: &mut Vec<WaylandSurfaceRenderElement<PixmanRenderer>>, target_layer: smithay::wayland::shell::wlr_layer::Layer| {
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
                                                                elements.push(e);
                                                            }
                                                        }
                                                    }
                                                }
                                            };

                                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Top);
                                            draw_layer(renderer, &mut elements, smithay::wayland::shell::wlr_layer::Layer::Overlay);
                                        }

                                let render_age = if state.overlay_state.is_animated() { 0 } else { 1 };
                                match damage_tracker.render_output(renderer, &mut frame, render_age, &elements, [0.1, 0.1, 0.1, 1.0]) {
                                    Ok(result) => {
                                        render_success = true;
                                        if let Some(damage) = result.damage { damage_rects = damage.clone(); }

                                        state.overlay_state.apply(&mut state.frame_buffer, width, height, true);

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

                    if let Some(ref mut encoder) = state.video_encoder {
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

                        if is_dirty {
                            send_frame = true;
                            st.no_motion_frame_count = 0;
                            st.paint_over_sent = false;
                            st.h264_burst_frames_remaining = 0;
                            target_qp = normal_qp;
                        } else if !send_frame {
                            st.no_motion_frame_count += 1;

                            if use_paint_over && st.no_motion_frame_count >= trigger_frames && !st.paint_over_sent {
                                send_frame = true;
                                st.paint_over_sent = true;
                                force_idr = true;
                                target_qp = paint_qp;
                                st.h264_burst_frames_remaining = burst - 1;
                            }
                        }

                        if send_frame {
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
                                        #[allow(deprecated)]
                                        Python::with_gil(|py| {
                                            let py_bytes = PyBytes::new(py, &data);
                                            if let Err(e) = cb.call1(py, (py_bytes,)) { eprintln!("Callback error: {:?}", e); }
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

                        let encoded_packets = encoders::software::encode_cpu(
                            &mut state.stripes,
                            &state.frame_buffer,
                            width,
                            height,
                            &damage_rects,
                            &state.settings,
                            state.frame_counter,
                            state.use_gpu,
                        );

                        if !encoded_packets.is_empty() {
                            state.encoded_frame_count += 1;
                            state.total_stripes_encoded += encoded_packets.len() as u32;
                            if let Some(ref cb) = state.callback {
                                #[allow(deprecated)]
                                Python::with_gil(|py| {
                                    for packet in encoded_packets {
                                        let py_bytes = PyBytes::new(py, &packet);
                                        if let Err(e) = cb.call1(py, (py_bytes,)) { eprintln!("Callback error: {:?}", e); }
                                    }
                                });
                            }
                        }
                    }
                    state.frame_counter = state.frame_counter.wrapping_add(1);
                }
            }
            let work_elapsed = loop_start_time.elapsed();
            let target_frame_duration = Duration::from_secs_f64(1.0 / state.settings.target_fps);
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
    fn new() -> Self {
        let (tx, rx) = smithay::reexports::calloop::channel::channel();
        thread::spawn(move || {
            run_wayland_thread(rx);
        });
        WaylandBackend { tx }
    }

    fn start_capture(&mut self, callback: Py<PyAny>, settings: &Bound<'_, PyAny>) -> PyResult<()> {
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
        };

        self.tx
            .send(ThreadCommand::StartCapture(callback, rust_settings))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to send start command: {}", e)))?;
        Ok(())
    }

    fn stop_capture(&mut self) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::StopCapture)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to send stop command: {}", e)))?;
        Ok(())
    }

    fn set_cursor_callback(&mut self, callback: Py<PyAny>) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::SetCursorCallback(callback))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set cursor callback: {}", e)))?;
        Ok(())
    }

    fn inject_key(&mut self, scancode: u32, state: u32) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::KeyboardKey { scancode, state })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject key: {}", e)))?;
        Ok(())
    }

    fn inject_mouse_move(&mut self, x: f64, y: f64) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::PointerMotion { x, y })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject motion: {}", e)))?;
        Ok(())
    }

    fn inject_relative_mouse_move(&mut self, dx: f64, dy: f64) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::PointerRelativeMotion { dx, dy })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject relative motion: {}", e)))?;
        Ok(())
    }

    fn inject_mouse_button(&mut self, btn: u32, state: u32) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::PointerButton { btn, state })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject button: {}", e)))?;
        Ok(())
    }

    fn inject_mouse_scroll(&mut self, x: f64, y: f64) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::PointerAxis { x, y })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to inject axis: {}", e)))?;
        Ok(())
    }

    fn set_cursor_rendering(&mut self, enabled: bool) -> PyResult<()> {
        self.tx
            .send(ThreadCommand::UpdateCursorConfig { render_on_framebuffer: enabled })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set cursor config: {}", e)))?;
        Ok(())
    }
}

#[pymodule]
fn pixelflux_wayland(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WaylandBackend>()?;
    Ok(())
}
