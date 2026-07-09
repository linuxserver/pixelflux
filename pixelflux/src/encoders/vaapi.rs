/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::mem;
use std::os::fd::AsRawFd;
use std::ptr;
use std::sync::Once;

use ffmpeg_sys_next as ff;
use libc::{close, dup};

use std::sync::Arc;

use crate::recording_sink::RecordingSink;
use crate::RustCaptureSettings;
use smithay::backend::allocator::{dmabuf::Dmabuf, Buffer};

static FF_INIT: Once = Once::new();
const AV_DRM_MAX_PLANES: usize = 4;
const QP_HYSTERESIS_LIMIT: u32 = 60;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMObjectDescriptor {
    pub fd: c_int,
    pub size: usize,
    pub format_modifier: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMPlaneDescriptor {
    pub object_index: c_int,
    pub offset: isize,
    pub pitch: isize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMLayerDescriptor {
    pub format: u32,
    pub nb_planes: c_int,
    pub planes: [AVDRMPlaneDescriptor; AV_DRM_MAX_PLANES],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMFrameDescriptor {
    pub nb_objects: c_int,
    pub objects: [AVDRMObjectDescriptor; AV_DRM_MAX_PLANES],
    pub nb_layers: c_int,
    pub layers: [AVDRMLayerDescriptor; AV_DRM_MAX_PLANES],
}

struct DmabufResources {
    fds: Vec<c_int>,
}

unsafe extern "C" fn release_drm_frame(opaque: *mut c_void, data: *mut u8) {
    // FFmpeg (C) invokes this on buffer teardown: a panic must not unwind across the extern "C"
    // boundary (the compiler guard would abort the process), so catch it here.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let resources = Box::from_raw(opaque as *mut DmabufResources);
        for &fd in &resources.fds {
            close(fd);
        }
        // Free the descriptor FFmpeg owns now (null only on the error path, where
        // the caller frees it).
        if !data.is_null() {
            ff::av_free(data as *mut c_void);
        }
    }));
}

fn ff_err_str(err: i32) -> String {
    unsafe {
        let mut errbuf = [0 as c_char; 128];
        ff::av_strerror(err, errbuf.as_mut_ptr(), 128);
        CStr::from_ptr(errbuf.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}

// Hardware-accelerated H.264 encoding via VAAPI (FFmpeg).
pub struct VaapiEncoder {
    encoder_ctx: *mut ff::AVCodecContext,
    codec: *const ff::AVCodec,

    #[allow(dead_code)]
    hw_device_ctx: *mut ff::AVBufferRef,
    #[allow(dead_code)]
    drm_device_ctx: *mut ff::AVBufferRef,
    #[allow(dead_code)]
    drm_frames_ctx: *mut ff::AVBufferRef,
    
    enc_frames_ctx: *mut ff::AVBufferRef,

    filter_graph: *mut ff::AVFilterGraph,
    buffersrc_ctx: *mut ff::AVFilterContext,
    buffersink_ctx: *mut ff::AVFilterContext,

    video_frame: *mut ff::AVFrame,
    sw_frame: *mut ff::AVFrame,
    hw_frame: *mut ff::AVFrame,

    packet: *mut ff::AVPacket,

    width: i32,
    height: i32,
    fps: i32,

    current_qp: u32,
    qp_hysteresis_counter: u32,

    // Live rate-control state, tracked so a reconfigure re-opens the codec only on a real change.
    cbr_mode: bool,
    current_bitrate_kbps: i32,
    current_vbv_mult: f64,
    current_kf_s: f64,

    recording_sink: Option<Arc<RecordingSink>>,
    omit_stripe_headers: bool,
}

unsafe impl Send for VaapiEncoder {}

impl Drop for VaapiEncoder {
    fn drop(&mut self) {
        unsafe {
            if !self.packet.is_null() {
                ff::av_packet_free(&mut self.packet);
            }
            if !self.video_frame.is_null() {
                ff::av_frame_free(&mut self.video_frame);
            }
            if !self.sw_frame.is_null() {
                ff::av_frame_free(&mut self.sw_frame);
            }
            if !self.hw_frame.is_null() {
                ff::av_frame_free(&mut self.hw_frame);
            }

            if !self.filter_graph.is_null() {
                ff::avfilter_graph_free(&mut self.filter_graph);
            }
            if !self.encoder_ctx.is_null() {
                ff::avcodec_free_context(&mut self.encoder_ctx);
            }

            if !self.enc_frames_ctx.is_null() {
                ff::av_buffer_unref(&mut self.enc_frames_ctx);
            }
            if !self.drm_frames_ctx.is_null() {
                ff::av_buffer_unref(&mut self.drm_frames_ctx);
            }
            if !self.hw_device_ctx.is_null() {
                ff::av_buffer_unref(&mut self.hw_device_ctx);
            }
            if !self.drm_device_ctx.is_null() {
                ff::av_buffer_unref(&mut self.drm_device_ctx);
            }
        }
    }
}

impl VaapiEncoder {
    /// Initializes the VAAPI encoder, deriving context from a DRM render node.
    ///
    /// Sets up the hardware device context, derives the VAAPI context, allocates
    /// frame contexts, and configures the FFmpeg filter graph for color conversion.
    ///
    /// Dmabuf input (Wayland): the source frame is a DRM-PRIME dmabuf mapped into a VAAPI
    /// surface by the filter graph.
    pub fn new(
        settings: &RustCaptureSettings,
        recording_sink: Option<Arc<RecordingSink>>,
    ) -> Result<Self, String> {
        Self::new_impl(settings, recording_sink, false)
    }

    /// Host-ARGB input (X11): the source is a CPU BGRA frame uploaded to a VAAPI surface, then
    /// converted ARGB->NV12 by VA-VPP (scale_vaapi) on the GPU before encode -- no CPU CSC.
    pub fn new_host(
        settings: &RustCaptureSettings,
        recording_sink: Option<Arc<RecordingSink>>,
    ) -> Result<Self, String> {
        Self::new_impl(settings, recording_sink, true)
    }

    fn new_impl(
        settings: &RustCaptureSettings,
        recording_sink: Option<Arc<RecordingSink>>,
        host_input: bool,
    ) -> Result<Self, String> {
        FF_INIT.call_once(|| {});

        let width = settings.width;
        let height = settings.height;
        let fps = settings.target_fps as i32;

        unsafe {
            let mut drm_device_ctx: *mut ff::AVBufferRef = ptr::null_mut();
            let render_node = if settings.encode_node_index >= 0 {
                format!("/dev/dri/renderD{}", 128 + settings.encode_node_index)
            } else {
                "/dev/dri/renderD128".to_string()
            };
            let device_url = CString::new(render_node).unwrap();

            let ret = ff::av_hwdevice_ctx_create(
                &mut drm_device_ctx,
                ff::AVHWDeviceType::AV_HWDEVICE_TYPE_DRM,
                device_url.as_ptr(),
                ptr::null_mut(),
                0,
            );
            if ret < 0 {
                return Err(format!("Failed to create DRM device: {}", ff_err_str(ret)));
            }

            let mut hw_device_ctx: *mut ff::AVBufferRef = ptr::null_mut();
            let ret = ff::av_hwdevice_ctx_create_derived(
                &mut hw_device_ctx,
                ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                drm_device_ctx,
                0,
            );
            if ret < 0 {
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err(format!(
                    "Failed to derive VAAPI device: {}",
                    ff_err_str(ret)
                ));
            }

            let mut drm_frames_ref = ff::av_hwframe_ctx_alloc(drm_device_ctx);
            if drm_frames_ref.is_null() {
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to alloc DRM frames ctx".into());
            }

            let drm_frames = (*drm_frames_ref).data as *mut ff::AVHWFramesContext;
            (*drm_frames).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME;
            (*drm_frames).sw_format = ff::AVPixelFormat::AV_PIX_FMT_BGRA;
            (*drm_frames).width = width;
            (*drm_frames).height = height;
            (*drm_frames).initial_pool_size = 0;

            if ff::av_hwframe_ctx_init(drm_frames_ref) < 0 {
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to init DRM frames ctx".into());
            }

            let codec_name = CString::new("h264_vaapi").unwrap();
            let codec = ff::avcodec_find_encoder_by_name(codec_name.as_ptr());
            if codec.is_null() {
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("h264_vaapi encoder not found".into());
            }

            let aligned_width = (width + 15) & !15;
            let aligned_height = (height + 31) & !31;

            let mut enc_frames_ref = ff::av_hwframe_ctx_alloc(hw_device_ctx);
            if enc_frames_ref.is_null() {
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to allocate encoder frames ctx".into());
            }
            let enc_frames = (*enc_frames_ref).data as *mut ff::AVHWFramesContext;
            (*enc_frames).format = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
            (*enc_frames).sw_format = ff::AVPixelFormat::AV_PIX_FMT_NV12;
            (*enc_frames).width = aligned_width;
            (*enc_frames).height = aligned_height;
            (*enc_frames).initial_pool_size = 20;

            if ff::av_hwframe_ctx_init(enc_frames_ref) < 0 {
                ff::av_buffer_unref(&mut enc_frames_ref);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to init encoder frames ctx".into());
            }

            // Keep a reference for restarting the encoder
            let mut saved_enc_frames_ctx = ff::av_buffer_ref(enc_frames_ref);

            let mut encoder_ctx = ff::avcodec_alloc_context3(codec);
            if encoder_ctx.is_null() {
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut enc_frames_ref);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to allocate encoder context".into());
            }
            (*encoder_ctx).width = width;
            (*encoder_ctx).height = height;
            (*encoder_ctx).time_base = ff::AVRational { num: 1, den: fps };
            (*encoder_ctx).framerate = ff::AVRational { num: fps, den: 1 };
            (*encoder_ctx).pix_fmt = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
            (*encoder_ctx).hw_device_ctx = ff::av_buffer_ref(hw_device_ctx);
            (*encoder_ctx).hw_frames_ctx = ff::av_buffer_ref(enc_frames_ref);
            (*encoder_ctx).max_b_frames = 0;
            (*encoder_ctx).gop_size = std::ffi::c_int::MAX;
            // Multiple slices help client decoders parallelize; >4 upsets Chromium.
            (*encoder_ctx).slices = 4;
            // Driver quality level biased toward speed (VA quality range: higher = faster).
            (*encoder_ctx).compression_level = 6;

            ff::av_buffer_unref(&mut enc_frames_ref);

            let mut opts: *mut ff::AVDictionary = ptr::null_mut();
            let set_opt = |d: &mut *mut ff::AVDictionary, k: &str, v: &str| {
                let ck = CString::new(k).unwrap();
                let cv = CString::new(v).unwrap();
                ff::av_dict_set(d, ck.as_ptr(), cv.as_ptr(), 0);
            };

            if settings.video_cbr_mode {
                // Constant bitrate: VA-API reads the target rate from the codec context's
                // bit_rate; the rc_mode opt just selects the algorithm.
                let bps = (settings.video_bitrate_kbps.max(0) as i64).saturating_mul(1000);
                let vbv = crate::encoders::vbv_bits(
                    bps.min(u32::MAX as i64) as u32,
                    settings.target_fps,
                    settings.keyframe_interval_s,
                    settings.video_vbv_multiplier,
                );
                (*encoder_ctx).bit_rate = bps;
                (*encoder_ctx).rc_max_rate = bps;
                (*encoder_ctx).rc_buffer_size = vbv.min(i32::MAX as u32) as i32;
                set_opt(&mut opts, "rc_mode", "CBR");
            } else {
                set_opt(&mut opts, "rc_mode", "CQP");
                set_opt(&mut opts, "qp", &settings.video_crf.to_string());
            }
            set_opt(&mut opts, "async_depth", "1");
            set_opt(&mut opts, "profile", "high");
            set_opt(&mut opts, "level", "4.1");

            let ret = ff::avcodec_open2(encoder_ctx, codec, &mut opts);
            ff::av_dict_free(&mut opts);
            if ret < 0 {
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err(format!("Failed to open encoder: {}", ff_err_str(ret)));
            }

            let mut filter_graph = ff::avfilter_graph_alloc();
            let buffersrc = ff::avfilter_get_by_name(CString::new("buffer").unwrap().as_ptr());
            let buffersink =
                ff::avfilter_get_by_name(CString::new("buffersink").unwrap().as_ptr());
            let name_in = CString::new("in").unwrap();
            let name_out = CString::new("out").unwrap();

            let buffersrc_ctx =
                ff::avfilter_graph_alloc_filter(filter_graph, buffersrc, name_in.as_ptr());

            let par = ff::av_buffersrc_parameters_alloc();
            if par.is_null() {
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to alloc buffersrc parameters".into());
            }
            if host_input {
                // Software BGRA in; hwupload (below) stages it onto a VAAPI surface.
                (*par).format = ff::AVPixelFormat::AV_PIX_FMT_BGRA as i32;
            } else {
                (*par).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME as i32;
                (*par).hw_frames_ctx = ff::av_buffer_ref(drm_frames_ref);
            }
            (*par).width = width;
            (*par).height = height;
            (*par).time_base = ff::AVRational { num: 1, den: fps };

            let ret = ff::av_buffersrc_parameters_set(buffersrc_ctx, par);
            if !(*par).hw_frames_ctx.is_null() {
                ff::av_buffer_unref(&mut (*par).hw_frames_ctx);
            }
            ff::av_free(par as *mut c_void);
            if ret < 0 {
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err(format!(
                    "Failed to set buffersrc parameters: {}",
                    ff_err_str(ret)
                ));
            }

            let args_str = format!(
                "video_size={}x{}:time_base=1/{}:pixel_aspect=1/1",
                width, height, fps
            );
            let args = CString::new(args_str).unwrap();
            if ff::avfilter_init_str(buffersrc_ctx, args.as_ptr()) < 0 {
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to init buffersrc".into());
            }

            let mut buffersink_ctx: *mut ff::AVFilterContext = ptr::null_mut();
            if ff::avfilter_graph_create_filter(
                &mut buffersink_ctx,
                buffersink,
                name_out.as_ptr(),
                ptr::null(),
                ptr::null_mut(),
                filter_graph,
            ) < 0
            {
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to create buffersink".into());
            }

            // VA-VPP convert ARGB->NV12 on the GPU BEFORE encode (VA implementations vary, so an
            // explicit convert is safer than relying on encoder-side RGB CSC). Pin BT.709 limited
            // to match the NVENC/x264 4:2:0 output so all backends agree on color. Host input
            // (X11) uploads the CPU BGRA frame to a VAAPI surface (hwupload); dmabuf input
            // (Wayland) maps the DRM-PRIME buffer (hwmap). Both then run scale_vaapi (VA-VPP).
            let stage = if host_input { "hwupload" } else { "hwmap" };
            let filters_desc = CString::new(format!(
                "{},scale_vaapi=w={}:h={}:format=nv12:out_color_matrix=bt709:out_range=tv",
                stage, width, height
            ))
            .unwrap();
            // The one-shot parser initializes filters DURING the parse, and hwupload fails
            // its init without a device (the host path's buffersrc is plain BGRA, so no
            // frames context to derive one from). Stage the graph the way the ffmpeg CLI
            // does: parse -> create filters -> attach the device -> init+link -> wire our
            // endpoints to the chain's dangling pads.
            let mut seg: *mut ff::AVFilterGraphSegment = ptr::null_mut();
            let mut seg_inputs: *mut ff::AVFilterInOut = ptr::null_mut();
            let mut seg_outputs: *mut ff::AVFilterInOut = ptr::null_mut();
            let seg_ok = ff::avfilter_graph_segment_parse(
                filter_graph,
                filters_desc.as_ptr(),
                0,
                &mut seg,
            ) >= 0
                && ff::avfilter_graph_segment_create_filters(seg, 0) >= 0
                && {
                    for i in 0..(*filter_graph).nb_filters {
                        let f = *(*filter_graph).filters.add(i as usize);
                        if (*f).hw_device_ctx.is_null() {
                            (*f).hw_device_ctx = ff::av_buffer_ref(hw_device_ctx);
                        }
                    }
                    ff::avfilter_graph_segment_apply(seg, 0, &mut seg_inputs, &mut seg_outputs)
                        >= 0
                }
                && !seg_inputs.is_null()
                && !seg_outputs.is_null()
                && ff::avfilter_link(
                    buffersrc_ctx,
                    0,
                    (*seg_inputs).filter_ctx,
                    (*seg_inputs).pad_idx as u32,
                ) >= 0
                && ff::avfilter_link(
                    (*seg_outputs).filter_ctx,
                    (*seg_outputs).pad_idx as u32,
                    buffersink_ctx,
                    0,
                ) >= 0;
            ff::avfilter_inout_free(&mut seg_inputs);
            ff::avfilter_inout_free(&mut seg_outputs);
            ff::avfilter_graph_segment_free(&mut seg);
            if !seg_ok {
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to build filter graph".into());
            }

            if ff::avfilter_graph_config(filter_graph, ptr::null_mut()) < 0 {
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to config filter graph".into());
            }

            let mut video_frame = ff::av_frame_alloc();
            let mut sw_frame = ff::av_frame_alloc();
            let mut hw_frame = ff::av_frame_alloc();

            if ff::av_hwframe_get_buffer((*encoder_ctx).hw_frames_ctx, hw_frame, 0) < 0 {
                ff::av_frame_free(&mut hw_frame);
                ff::av_frame_free(&mut sw_frame);
                ff::av_frame_free(&mut video_frame);
                ff::avfilter_graph_free(&mut filter_graph);
                ff::avcodec_free_context(&mut encoder_ctx);
                ff::av_buffer_unref(&mut saved_enc_frames_ctx);
                ff::av_buffer_unref(&mut drm_frames_ref);
                ff::av_buffer_unref(&mut hw_device_ctx);
                ff::av_buffer_unref(&mut drm_device_ctx);
                return Err("Failed to allocate HW frame for NV12 path".into());
            }

            Ok(Self {
                encoder_ctx,
                codec,
                hw_device_ctx,
                drm_device_ctx,
                drm_frames_ctx: drm_frames_ref,
                enc_frames_ctx: saved_enc_frames_ctx,
                filter_graph,
                buffersrc_ctx,
                buffersink_ctx,
                video_frame,
                sw_frame,
                hw_frame,
                packet: ff::av_packet_alloc(),
                width,
                height,
                fps,
                current_qp: settings.video_crf as u32,
                qp_hysteresis_counter: 0,
                cbr_mode: settings.video_cbr_mode,
                current_bitrate_kbps: settings.video_bitrate_kbps,
                current_vbv_mult: settings.video_vbv_multiplier,
                current_kf_s: settings.keyframe_interval_s,
                recording_sink,
                omit_stripe_headers: settings.omit_stripe_headers,
            })
        }
    }

    /// Re-opens the codec context in place, applying the current rate-control state.
    ///
    /// The VA device, encoder frames pool and filter graph persist; only the AVCodecContext is
    /// rebuilt. CBR reprograms bit_rate / rc_max_rate / rc_buffer_size from the tracked bitrate
    /// and VBV; CQP reprograms the quantizer. A freshly opened context emits an IDR as its first
    /// frame, so the reference chain self-heals. This replaces any driver-side live reconfigure,
    /// which is flaky on some drivers.
    unsafe fn reopen_codec(&mut self, qp: u32) -> Result<(), String> {
        if !self.encoder_ctx.is_null() {
            ff::avcodec_free_context(&mut self.encoder_ctx);
        }

        self.encoder_ctx = ff::avcodec_alloc_context3(self.codec);
        if self.encoder_ctx.is_null() {
            return Err("Failed to re-alloc encoder context".into());
        }

        (*self.encoder_ctx).width = self.width;
        (*self.encoder_ctx).height = self.height;
        (*self.encoder_ctx).time_base = ff::AVRational { num: 1, den: self.fps };
        (*self.encoder_ctx).framerate = ff::AVRational { num: self.fps, den: 1 };
        (*self.encoder_ctx).pix_fmt = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
        (*self.encoder_ctx).hw_device_ctx = ff::av_buffer_ref(self.hw_device_ctx);
        (*self.encoder_ctx).hw_frames_ctx = ff::av_buffer_ref(self.enc_frames_ctx);
        (*self.encoder_ctx).max_b_frames = 0;
        (*self.encoder_ctx).gop_size = std::ffi::c_int::MAX;
        (*self.encoder_ctx).slices = 4;
        (*self.encoder_ctx).compression_level = 6;

        let mut opts: *mut ff::AVDictionary = ptr::null_mut();
        let set_opt = |d: &mut *mut ff::AVDictionary, k: &str, v: &str| {
            let ck = CString::new(k).unwrap();
            let cv = CString::new(v).unwrap();
            ff::av_dict_set(d, ck.as_ptr(), cv.as_ptr(), 0);
        };

        if self.cbr_mode {
            let bps = (self.current_bitrate_kbps.max(0) as i64).saturating_mul(1000);
            let vbv = crate::encoders::vbv_bits(
                bps.min(u32::MAX as i64) as u32,
                self.fps.max(1) as f64,
                self.current_kf_s,
                self.current_vbv_mult,
            );
            (*self.encoder_ctx).bit_rate = bps;
            (*self.encoder_ctx).rc_max_rate = bps;
            (*self.encoder_ctx).rc_buffer_size = vbv.min(i32::MAX as u32) as i32;
            set_opt(&mut opts, "rc_mode", "CBR");
        } else {
            set_opt(&mut opts, "rc_mode", "CQP");
            set_opt(&mut opts, "qp", &qp.to_string());
        }
        set_opt(&mut opts, "async_depth", "1");
        set_opt(&mut opts, "profile", "high");
        set_opt(&mut opts, "level", "4.1");

        let ret = ff::avcodec_open2(self.encoder_ctx, self.codec, &mut opts);
        ff::av_dict_free(&mut opts);

        if ret < 0 {
            return Err(format!("Failed to re-open encoder: {}", ff_err_str(ret)));
        }

        self.current_qp = qp;
        Ok(())
    }

    /// Updates the quantization parameter (QP) with hysteresis.
    ///
    /// If QP decreases (higher quality paint-over), it re-opens immediately.
    /// If QP increases (lower quality motion), it waits for the hysteresis limit
    /// to avoid blinking artifacts. No-op in CBR, where quality is driven by the
    /// bitrate target and a per-frame QP change would otherwise re-open in CQP and
    /// drop the configured bitrate.
    unsafe fn update_qp(&mut self, target_qp: u32) -> Result<(), String> {
        if self.cbr_mode {
            return Ok(());
        }

        if target_qp == self.current_qp {
            self.qp_hysteresis_counter = 0;
            return Ok(());
        }

        if target_qp < self.current_qp {
            self.qp_hysteresis_counter = 0;
            self.reopen_codec(target_qp)?;
        } else {
            self.qp_hysteresis_counter += 1;
            if self.qp_hysteresis_counter > QP_HYSTERESIS_LIMIT {
                self.qp_hysteresis_counter = 0;
                self.reopen_codec(target_qp)?;
            }
        }

        Ok(())
    }

    /// Apply a live rate-control / framerate change from updated settings. Guarded so it only
    /// re-opens the codec when the CBR bitrate/VBV (CBR only) or the target fps actually change,
    /// making it cheap to call every frame like the NVENC path. The re-open carries the current
    /// QP so a CQP stream keeps its quantizer across an fps change.
    pub fn reconfigure_rate(&mut self, settings: &RustCaptureSettings) {
        unsafe {
            let mut changed = false;
            if self.cbr_mode
                && (settings.video_bitrate_kbps != self.current_bitrate_kbps
                    || settings.video_vbv_multiplier != self.current_vbv_mult)
            {
                changed = true;
            }
            let new_fps = settings.target_fps.max(1.0) as i32;
            if new_fps != self.fps {
                changed = true;
            }
            if !changed {
                return;
            }
            self.fps = new_fps;
            self.current_bitrate_kbps = settings.video_bitrate_kbps;
            self.current_vbv_mult = settings.video_vbv_multiplier;
            self.current_kf_s = settings.keyframe_interval_s;
            if let Err(e) = self.reopen_codec(self.current_qp) {
                eprintln!("[vaapi] rate reconfigure failed: {e}");
            }
        }
    }

    /// Retrieves encoded packets from the encoder and formats them with the custom header.
    unsafe fn collect_packet(&mut self, frame_number: u64, output: &mut Vec<u8>) {
        while ff::avcodec_receive_packet(self.encoder_ctx, self.packet) == 0 {
            let size = (*self.packet).size as usize;
            let data = (*self.packet).data;
            let is_key = ((*self.packet).flags & ff::AV_PKT_FLAG_KEY) != 0;

            let header_sz = if self.omit_stripe_headers { 0 } else { 10 };
            output.reserve(header_sz + size);
            if !self.omit_stripe_headers {
                output.push(0x04);
                output.push(if is_key { 0x01 } else { 0x00 });
                output.extend_from_slice(&(frame_number as u16).to_be_bytes());
                output.extend_from_slice(&0u16.to_be_bytes());
                output.extend_from_slice(&(self.width as u16).to_be_bytes());
                output.extend_from_slice(&(self.height as u16).to_be_bytes());
            }

            let slice = std::slice::from_raw_parts(data, size);
            output.extend_from_slice(slice);
            if let Some(ref sink) = self.recording_sink {
                sink.write_frame(slice);
            }

            ff::av_packet_unref(self.packet);
        }
    }

    /// Encodes a DMA-BUF frame by importing it via DRM and passing it through the filter graph.
    ///
    /// The filter graph handles mapping the DRM frame to a VAAPI surface and converting colorspace.
    pub fn encode_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        frame_number: u64,
        qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.update_qp(qp)?;

            let desc_size = mem::size_of::<AVDRMFrameDescriptor>();
            let desc_ptr = ff::av_mallocz(desc_size) as *mut AVDRMFrameDescriptor;
            if desc_ptr.is_null() {
                return Err("OOM".into());
            }

            let mut resources = DmabufResources { fds: Vec::new() };
            let strides: Vec<u32> = dmabuf.strides().collect();

            (*desc_ptr).nb_objects = dmabuf.handles().count() as i32;
            (*desc_ptr).nb_layers = 1;

            for (i, (handle, _)) in dmabuf.handles().zip(dmabuf.offsets()).enumerate() {
                let fd = dup(handle.as_raw_fd());
                if fd < 0 {
                    for &dup_fd in &resources.fds {
                        close(dup_fd);
                    }
                    ff::av_free(desc_ptr as *mut c_void);
                    return Err("Failed to dup fd".into());
                }
                resources.fds.push(fd);
                (*desc_ptr).objects[i].fd = fd;
                
                let stride = strides.get(i).copied().unwrap_or(strides[0]);
                let aligned_height = (self.height + 31) & !31;
                (*desc_ptr).objects[i].size = (stride as usize) * (aligned_height as usize);
                
                (*desc_ptr).objects[i].format_modifier = u64::from(dmabuf.format().modifier);
            }

            (*desc_ptr).layers[0].format = dmabuf.format().code as u32;
            (*desc_ptr).layers[0].nb_planes = dmabuf.num_planes() as i32;

            for (i, (stride, offset)) in dmabuf.strides().zip(dmabuf.offsets()).enumerate() {
                (*desc_ptr).layers[0].planes[i].object_index = i as i32;
                (*desc_ptr).layers[0].planes[i].offset = offset as isize;
                (*desc_ptr).layers[0].planes[i].pitch = stride as isize;
            }

            if dmabuf.handles().count() == 1 && dmabuf.num_planes() > 1 {
                for i in 0..dmabuf.num_planes() {
                    (*desc_ptr).layers[0].planes[i].object_index = 0;
                }
            }

            ff::av_frame_unref(self.video_frame);
            (*self.video_frame).width = self.width;
            (*self.video_frame).height = self.height;
            (*self.video_frame).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME as i32;
            (*self.video_frame).data[0] = desc_ptr as *mut u8;

            let opaque = Box::into_raw(Box::new(resources));
            let buf_ref = ff::av_buffer_create(
                desc_ptr as *mut u8,
                desc_size,
                Some(release_drm_frame),
                opaque as *mut c_void,
                0,
            );

            if buf_ref.is_null() {
                release_drm_frame(opaque as *mut c_void, ptr::null_mut());
                ff::av_free(desc_ptr as *mut c_void);
                return Err("Failed to create buffer ref".into());
            }
            (*self.video_frame).buf[0] = buf_ref;
            (*self.video_frame).pts = frame_number as i64;
            (*self.video_frame).hw_frames_ctx = ff::av_buffer_ref(self.drm_frames_ctx);

            if ff::av_buffersrc_add_frame(self.buffersrc_ctx, self.video_frame) < 0 {
                // add_frame (no KEEP_REF) only consumes the frame's refs on success;
                // on error "the input frame is not touched", so buf[0] is still live
                // and this unref is required to release it (-> release_drm_frame closes
                // the dup'd dmabuf fds). Don't add a manual fd close: that double-closes.
                ff::av_frame_unref(self.video_frame);
                return Err("Failed to feed filter graph".into());
            }

            let mut output = Vec::new();
            let mut filtered_frame = ff::av_frame_alloc();

            while ff::av_buffersink_get_frame(self.buffersink_ctx, filtered_frame) >= 0 {
                if force_idr {
                    (*filtered_frame).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_I;
                }

                if ff::avcodec_send_frame(self.encoder_ctx, filtered_frame) < 0 {
                    ff::av_frame_free(&mut filtered_frame);
                    return Err("Failed to send frame to encoder".into());
                }
                ff::av_frame_unref(filtered_frame);

                self.collect_packet(frame_number, &mut output);
            }
            ff::av_frame_free(&mut filtered_frame);

            Ok(output)
        }
    }

    /// Encodes one host BGRA frame using GPU postprocessing (X11 path).
    ///
    /// The CPU frame is staged into a VAAPI surface (hwupload) and converted ARGB->NV12 by
    /// VA-VPP (scale_vaapi) on the GPU before encode -- there is no CPU colorspace conversion.
    /// Only valid on an encoder built with `new_host`.
    pub fn encode_host_argb(
        &mut self,
        bgra: &[u8],
        stride: usize,
        frame_number: u64,
        qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.update_qp(qp)?;

            let h = self.height as usize;
            let needed = stride.checked_mul(h).ok_or("stride overflow")?;
            if bgra.len() < needed {
                return Err("Input buffer too small".into());
            }

            // Build a fresh refcounted BGRA frame and stage the host rows into it (the only copy;
            // it is data movement for the GPU upload, not a colorspace conversion).
            ff::av_frame_unref(self.video_frame);
            (*self.video_frame).width = self.width;
            (*self.video_frame).height = self.height;
            (*self.video_frame).format = ff::AVPixelFormat::AV_PIX_FMT_BGRA as i32;
            if ff::av_frame_get_buffer(self.video_frame, 0) < 0 {
                return Err("Failed to allocate host BGRA frame".into());
            }
            let dst = (*self.video_frame).data[0];
            let dst_stride = (*self.video_frame).linesize[0] as usize;
            let row_bytes = (self.width as usize) * 4;
            let copy_bytes = row_bytes.min(stride).min(dst_stride);
            for row in 0..h {
                ptr::copy_nonoverlapping(
                    bgra.as_ptr().add(row * stride),
                    dst.add(row * dst_stride),
                    copy_bytes,
                );
            }
            (*self.video_frame).pts = frame_number as i64;

            if ff::av_buffersrc_add_frame(self.buffersrc_ctx, self.video_frame) < 0 {
                ff::av_frame_unref(self.video_frame);
                return Err("Failed to feed filter graph".into());
            }

            let mut output = Vec::new();
            let mut filtered_frame = ff::av_frame_alloc();
            while ff::av_buffersink_get_frame(self.buffersink_ctx, filtered_frame) >= 0 {
                if force_idr {
                    (*filtered_frame).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_I;
                }
                if ff::avcodec_send_frame(self.encoder_ctx, filtered_frame) < 0 {
                    ff::av_frame_free(&mut filtered_frame);
                    return Err("Failed to send frame to encoder".into());
                }
                ff::av_frame_unref(filtered_frame);
                self.collect_packet(frame_number, &mut output);
            }
            ff::av_frame_free(&mut filtered_frame);

            Ok(output)
        }
    }

    /// Encodes raw NV12 pixel data by uploading it from CPU memory to the GPU.
    pub fn encode_raw(
        &mut self,
        nv12_pixels: &[u8],
        frame_number: u64,
        qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.update_qp(qp)?;

            let width = self.width as usize;
            let height = self.height as usize;
            let required_size = width * height + (width * height / 2);

            if nv12_pixels.len() < required_size {
                return Err("Input buffer too small".into());
            }

            ff::av_frame_unref(self.sw_frame);
            (*self.sw_frame).format = ff::AVPixelFormat::AV_PIX_FMT_NV12 as i32;
            (*self.sw_frame).width = self.width;
            (*self.sw_frame).height = self.height;

            (*self.sw_frame).data[0] = nv12_pixels.as_ptr() as *mut u8;
            (*self.sw_frame).linesize[0] = self.width;

            (*self.sw_frame).data[1] = nv12_pixels.as_ptr().add(width * height) as *mut u8;
            (*self.sw_frame).linesize[1] = self.width;

            // get_buffer needs an empty frame; without this the prior surface leaks.
            ff::av_frame_unref(self.hw_frame);
            if ff::av_hwframe_get_buffer((*self.encoder_ctx).hw_frames_ctx, self.hw_frame, 0) < 0 {
                return Err("Failed to allocate HW frame for NV12 path".into());
            }
            (*self.hw_frame).width = self.width;
            (*self.hw_frame).height = self.height;

            if ff::av_hwframe_transfer_data(self.hw_frame, self.sw_frame, 0) < 0 {
                return Err("Failed to upload frame to GPU".into());
            }

            ff::av_frame_unref(self.sw_frame);

            (*self.hw_frame).pts = frame_number as i64;
            // Force keyframes via pict_type (AV_PKT_FLAG_KEY is a packet flag, not a frame flag).
            if force_idr {
                (*self.hw_frame).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_I;
            } else {
                (*self.hw_frame).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_NONE;
            }

            if ff::avcodec_send_frame(self.encoder_ctx, self.hw_frame) < 0 {
                return Err("Error sending frame to encoder".into());
            }

            let mut output = Vec::new();
            self.collect_packet(frame_number, &mut output);

            Ok(output)
        }
    }
}

#[cfg(test)]
mod graph_ordering_tests {
    use super::*;

    /// Builds the same shaped graph as new_impl's host path -- explicit buffersrc/sink
    /// endpoints plus an "hwupload,..." chain -- against a CUDA device (hwupload is the
    /// same generic filter VAAPI uses; this box has no VAAPI device). `staged` selects
    /// the segment API (parse -> create -> attach -> init/link) vs the one-shot parser
    /// with the device attached afterwards.
    unsafe fn build_hwupload_graph(staged: bool) -> Result<(), String> {
        let mut dev: *mut ff::AVBufferRef = ptr::null_mut();
        if ff::av_hwdevice_ctx_create(
            &mut dev,
            ff::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
            ptr::null(),
            ptr::null_mut(),
            0,
        ) < 0
        {
            return Err("no CUDA device".into());
        }

        let graph = ff::avfilter_graph_alloc();
        let mut src: *mut ff::AVFilterContext = ptr::null_mut();
        let args = CString::new("video_size=64x64:pix_fmt=bgra:time_base=1/30").unwrap();
        let r = ff::avfilter_graph_create_filter(
            &mut src,
            ff::avfilter_get_by_name(CString::new("buffer").unwrap().as_ptr()),
            CString::new("in").unwrap().as_ptr(),
            args.as_ptr(),
            ptr::null_mut(),
            graph,
        );
        assert!(r >= 0, "buffersrc create");
        let mut sink: *mut ff::AVFilterContext = ptr::null_mut();
        let r = ff::avfilter_graph_create_filter(
            &mut sink,
            ff::avfilter_get_by_name(CString::new("buffersink").unwrap().as_ptr()),
            CString::new("out").unwrap().as_ptr(),
            ptr::null(),
            ptr::null_mut(),
            graph,
        );
        assert!(r >= 0, "buffersink create");

        let desc = CString::new("hwupload,hwdownload,format=bgra").unwrap();
        let result: Result<(), String> = if staged {
            let mut seg: *mut ff::AVFilterGraphSegment = ptr::null_mut();
            let mut ins: *mut ff::AVFilterInOut = ptr::null_mut();
            let mut outs: *mut ff::AVFilterInOut = ptr::null_mut();
            let ok = ff::avfilter_graph_segment_parse(graph, desc.as_ptr(), 0, &mut seg) >= 0
                && ff::avfilter_graph_segment_create_filters(seg, 0) >= 0
                && {
                    for i in 0..(*graph).nb_filters {
                        let f = *(*graph).filters.add(i as usize);
                        if (*f).hw_device_ctx.is_null() {
                            (*f).hw_device_ctx = ff::av_buffer_ref(dev);
                        }
                    }
                    ff::avfilter_graph_segment_apply(seg, 0, &mut ins, &mut outs) >= 0
                }
                && !ins.is_null()
                && !outs.is_null()
                && ff::avfilter_link(src, 0, (*ins).filter_ctx, (*ins).pad_idx as u32) >= 0
                && ff::avfilter_link((*outs).filter_ctx, (*outs).pad_idx as u32, sink, 0) >= 0;
            ff::avfilter_inout_free(&mut ins);
            ff::avfilter_inout_free(&mut outs);
            ff::avfilter_graph_segment_free(&mut seg);
            if ok { Ok(()) } else { Err("segment build failed".into()) }
        } else {
            let mut inputs = ff::avfilter_inout_alloc();
            let mut outputs = ff::avfilter_inout_alloc();
            (*inputs).name = ff::av_strdup(CString::new("in").unwrap().as_ptr());
            (*inputs).filter_ctx = src;
            (*inputs).pad_idx = 0;
            (*inputs).next = ptr::null_mut();
            (*outputs).name = ff::av_strdup(CString::new("out").unwrap().as_ptr());
            (*outputs).filter_ctx = sink;
            (*outputs).pad_idx = 0;
            (*outputs).next = ptr::null_mut();
            let r = ff::avfilter_graph_parse_ptr(
                graph,
                desc.as_ptr(),
                &mut outputs,
                &mut inputs,
                ptr::null_mut(),
            );
            ff::avfilter_inout_free(&mut inputs);
            ff::avfilter_inout_free(&mut outputs);
            if r < 0 {
                Err(format!("parse failed before the attach loop could run: {}", ff_err_str(r)))
            } else {
                for i in 0..(*graph).nb_filters {
                    let f = *(*graph).filters.add(i as usize);
                    if (*f).hw_device_ctx.is_null() {
                        (*f).hw_device_ctx = ff::av_buffer_ref(dev);
                    }
                }
                Ok(())
            }
        };

        let result = result.and_then(|()| {
            let r = ff::avfilter_graph_config(graph, ptr::null_mut());
            if r < 0 { Err(format!("config failed: {}", ff_err_str(r))) } else { Ok(()) }
        });

        // Prove the staged graph actually passes pixels, not just configures.
        let result = result.and_then(|()| {
            let frame = ff::av_frame_alloc();
            (*frame).format = ff::AVPixelFormat::AV_PIX_FMT_BGRA as i32;
            (*frame).width = 64;
            (*frame).height = 64;
            if ff::av_frame_get_buffer(frame, 0) < 0 {
                return Err("frame alloc".into());
            }
            for y in 0..64 {
                let row = (*frame).data[0].add(y * (*frame).linesize[0] as usize);
                std::ptr::write_bytes(row, 0x80, 64 * 4);
            }
            let mut fr = frame;
            let ok = ff::av_buffersrc_add_frame(src, fr) >= 0 && {
                let out = ff::av_frame_alloc();
                let got = ff::av_buffersink_get_frame(sink, out) >= 0
                    && (*out).width == 64
                    && !(*out).data[0].is_null();
                let mut o = out;
                ff::av_frame_free(&mut o);
                got
            };
            ff::av_frame_free(&mut fr);
            if ok { Ok(()) } else { Err("frame did not flow through the graph".into()) }
        });

        let mut g = graph;
        ff::avfilter_graph_free(&mut g);
        ff::av_buffer_unref(&mut dev);
        result
    }

    /// The one-shot parser initializes hwupload during the parse, before any attach loop
    /// can run -- reproducing the "always falls back to software" failure; the staged
    /// segment build must succeed end-to-end on the same machine.
    #[test]
    #[ignore]
    fn gpu_hwupload_device_attach_ordering() {
        unsafe {
            let old = build_hwupload_graph(false);
            let new = build_hwupload_graph(true);
            println!("one-shot parse+attach-after: {old:?}");
            println!("staged segment build:        {new:?}");
            assert!(new.is_ok(), "staged build must work: {new:?}");
            assert!(old.is_err(), "expected the one-shot ordering to fail on this FFmpeg");
        }
    }
}
