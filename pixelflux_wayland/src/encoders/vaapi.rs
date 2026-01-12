use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::mem;
use std::os::fd::AsRawFd;
use std::ptr;
use std::sync::Once;

use ffmpeg_sys_next as ff;
use libc::{close, dup};

use crate::RustCaptureSettings;
use smithay::backend::allocator::{dmabuf::Dmabuf, Buffer};

static FF_INIT: Once = Once::new();
const AV_DRM_MAX_PLANES: usize = 4;
const QP_HYSTERESIS_LIMIT: u32 = 60;

/// @brief Describes a DRM object (file descriptor) for FFmpeg interop.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMObjectDescriptor {
    pub fd: c_int,
    pub size: usize,
    pub format_modifier: u64,
}

/// @brief Describes a specific plane within a DRM layer.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMPlaneDescriptor {
    pub object_index: c_int,
    pub offset: isize,
    pub pitch: isize,
}

/// @brief Describes a layer in a DRM frame, containing multiple planes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMLayerDescriptor {
    pub format: u32,
    pub nb_planes: c_int,
    pub planes: [AVDRMPlaneDescriptor; AV_DRM_MAX_PLANES],
}

/// @brief Top-level descriptor for passing DMA-BUF frames to FFmpeg.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AVDRMFrameDescriptor {
    pub nb_objects: c_int,
    pub objects: [AVDRMObjectDescriptor; AV_DRM_MAX_PLANES],
    pub nb_layers: c_int,
    pub layers: [AVDRMLayerDescriptor; AV_DRM_MAX_PLANES],
}

/// @brief Container for file descriptors to ensure they are closed after use.
struct DmabufResources {
    fds: Vec<c_int>,
}

/// @brief Callback function used by FFmpeg to release custom DRM frames.
unsafe extern "C" fn release_drm_frame(opaque: *mut c_void, _data: *mut u8) {
    let resources = Box::from_raw(opaque as *mut DmabufResources);
    for &fd in &resources.fds {
        close(fd);
    }
}

/// @brief Helper to convert FFmpeg error codes into Rust strings.
fn ff_err_str(err: i32) -> String {
    unsafe {
        let mut errbuf = [0 as c_char; 128];
        ff::av_strerror(err, errbuf.as_mut_ptr(), 128);
        CStr::from_ptr(errbuf.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}

/// @brief Handles hardware-accelerated H.264 encoding via VAAPI.
///
/// Manages the FFmpeg VAAPI context, hardware device derivation from DRM,
/// filter graphs for format conversion, and the encoding loop.
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
}

unsafe impl Send for VaapiEncoder {}
unsafe impl Sync for VaapiEncoder {}

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
    /// @brief Initializes the VAAPI encoder, deriving context from a DRM render node.
    ///
    /// Sets up the hardware device context, derives the VAAPI context, allocates
    /// frame contexts, and configures the FFmpeg filter graph for color conversion.
    ///
    /// @input settings: Capture configuration (resolution, FPS, QP, render node).
    /// @return Result containing the new VaapiEncoder instance.
    pub fn new(settings: &RustCaptureSettings) -> Result<Self, String> {
        FF_INIT.call_once(|| {});

        let width = settings.width;
        let height = settings.height;
        let fps = settings.target_fps as i32;

        unsafe {
            let mut drm_device_ctx: *mut ff::AVBufferRef = ptr::null_mut();
            let render_node = if settings.vaapi_render_node_index >= 0 {
                format!("/dev/dri/renderD{}", 128 + settings.vaapi_render_node_index)
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

            let drm_frames_ref = ff::av_hwframe_ctx_alloc(drm_device_ctx);
            if drm_frames_ref.is_null() {
                return Err("Failed to alloc DRM frames ctx".into());
            }

            let drm_frames = (*drm_frames_ref).data as *mut ff::AVHWFramesContext;
            (*drm_frames).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME;
            (*drm_frames).sw_format = ff::AVPixelFormat::AV_PIX_FMT_BGRA;
            (*drm_frames).width = width;
            (*drm_frames).height = height;
            (*drm_frames).initial_pool_size = 0;

            if ff::av_hwframe_ctx_init(drm_frames_ref) < 0 {
                return Err("Failed to init DRM frames ctx".into());
            }

            let codec_name = CString::new("h264_vaapi").unwrap();
            let codec = ff::avcodec_find_encoder_by_name(codec_name.as_ptr());
            if codec.is_null() {
                return Err("h264_vaapi encoder not found".into());
            }

            let mut enc_frames_ref = ff::av_hwframe_ctx_alloc(hw_device_ctx);
            let enc_frames = (*enc_frames_ref).data as *mut ff::AVHWFramesContext;
            (*enc_frames).format = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
            (*enc_frames).sw_format = ff::AVPixelFormat::AV_PIX_FMT_NV12;
            (*enc_frames).width = width;
            (*enc_frames).height = height;
            (*enc_frames).initial_pool_size = 20;

            if ff::av_hwframe_ctx_init(enc_frames_ref) < 0 {
                return Err("Failed to init encoder frames ctx".into());
            }

            // Keep a reference for restarting the encoder
            let saved_enc_frames_ctx = ff::av_buffer_ref(enc_frames_ref);

            let encoder_ctx = ff::avcodec_alloc_context3(codec);
            (*encoder_ctx).width = width;
            (*encoder_ctx).height = height;
            (*encoder_ctx).time_base = ff::AVRational { num: 1, den: fps };
            (*encoder_ctx).framerate = ff::AVRational { num: fps, den: 1 };
            (*encoder_ctx).pix_fmt = ff::AVPixelFormat::AV_PIX_FMT_VAAPI;
            (*encoder_ctx).hw_device_ctx = ff::av_buffer_ref(hw_device_ctx);
            (*encoder_ctx).hw_frames_ctx = ff::av_buffer_ref(enc_frames_ref);
            (*encoder_ctx).max_b_frames = 0;
            (*encoder_ctx).gop_size = std::ffi::c_int::MAX; 

            ff::av_buffer_unref(&mut enc_frames_ref);

            let mut opts: *mut ff::AVDictionary = ptr::null_mut();
            let set_opt = |d: &mut *mut ff::AVDictionary, k: &str, v: &str| {
                let ck = CString::new(k).unwrap();
                let cv = CString::new(v).unwrap();
                ff::av_dict_set(d, ck.as_ptr(), cv.as_ptr(), 0);
            };

            set_opt(&mut opts, "rc_mode", "CQP");
            set_opt(&mut opts, "qp", &settings.h264_crf.to_string());
            set_opt(&mut opts, "async_depth", "1");
            set_opt(&mut opts, "profile", "high");
            set_opt(&mut opts, "level", "4.1");

            let ret = ff::avcodec_open2(encoder_ctx, codec, &mut opts);
            if ret < 0 {
                return Err(format!("Failed to open encoder: {}", ff_err_str(ret)));
            }
            ff::av_dict_free(&mut opts);

            let filter_graph = ff::avfilter_graph_alloc();
            let buffersrc = ff::avfilter_get_by_name(CString::new("buffer").unwrap().as_ptr());
            let buffersink =
                ff::avfilter_get_by_name(CString::new("buffersink").unwrap().as_ptr());
            let name_in = CString::new("in").unwrap();
            let name_out = CString::new("out").unwrap();

            let buffersrc_ctx =
                ff::avfilter_graph_alloc_filter(filter_graph, buffersrc, name_in.as_ptr());

            let par = ff::av_buffersrc_parameters_alloc();
            (*par).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME as i32;
            (*par).hw_frames_ctx = ff::av_buffer_ref(drm_frames_ref);
            (*par).width = width;
            (*par).height = height;
            (*par).time_base = ff::AVRational { num: 1, den: fps };

            let ret = ff::av_buffersrc_parameters_set(buffersrc_ctx, par);
            ff::av_free(par as *mut c_void);
            if ret < 0 {
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
                return Err("Failed to create buffersink".into());
            }

            let mut inputs = ff::avfilter_inout_alloc();
            let mut outputs = ff::avfilter_inout_alloc();
            (*inputs).name = ff::av_strdup(name_in.as_ptr());
            (*inputs).filter_ctx = buffersrc_ctx;
            (*inputs).pad_idx = 0;
            (*inputs).next = ptr::null_mut();
            (*outputs).name = ff::av_strdup(name_out.as_ptr());
            (*outputs).filter_ctx = buffersink_ctx;
            (*outputs).pad_idx = 0;
            (*outputs).next = ptr::null_mut();

            let filters_desc = CString::new("hwmap,scale_vaapi=format=nv12").unwrap();
            if ff::avfilter_graph_parse_ptr(
                filter_graph,
                filters_desc.as_ptr(),
                &mut outputs,
                &mut inputs,
                ptr::null_mut(),
            ) < 0
            {
                return Err("Failed to parse filter graph".into());
            }

            for i in 0..(*filter_graph).nb_filters {
                let f = *(*filter_graph).filters.add(i as usize);
                if (*f).hw_device_ctx.is_null() {
                    (*f).hw_device_ctx = ff::av_buffer_ref(hw_device_ctx);
                }
            }

            if ff::avfilter_graph_config(filter_graph, ptr::null_mut()) < 0 {
                return Err("Failed to config filter graph".into());
            }

            let video_frame = ff::av_frame_alloc();
            let sw_frame = ff::av_frame_alloc();
            let hw_frame = ff::av_frame_alloc();

            if ff::av_hwframe_get_buffer((*encoder_ctx).hw_frames_ctx, hw_frame, 0) < 0 {
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
                current_qp: settings.h264_crf as u32,
                qp_hysteresis_counter: 0,
            })
        }
    }

    /// @brief Completely restarts the encoder context with a new QP.
    ///
    /// This is required because VAAPI dynamic QP updates are flaky or unsupported
    /// on some drivers, necessitating a full stream stop/start to apply changes cleanly.
    unsafe fn restart_encoder(&mut self, new_qp: u32) -> Result<(), String> {
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

        let mut opts: *mut ff::AVDictionary = ptr::null_mut();
        let set_opt = |d: &mut *mut ff::AVDictionary, k: &str, v: &str| {
            let ck = CString::new(k).unwrap();
            let cv = CString::new(v).unwrap();
            ff::av_dict_set(d, ck.as_ptr(), cv.as_ptr(), 0);
        };

        set_opt(&mut opts, "rc_mode", "CQP");
        set_opt(&mut opts, "qp", &new_qp.to_string());
        set_opt(&mut opts, "async_depth", "1");
        set_opt(&mut opts, "profile", "high");
        set_opt(&mut opts, "level", "4.1");

        let ret = ff::avcodec_open2(self.encoder_ctx, self.codec, &mut opts);
        ff::av_dict_free(&mut opts);

        if ret < 0 {
            return Err(format!("Failed to re-open encoder: {}", ff_err_str(ret)));
        }

        self.current_qp = new_qp;
        Ok(())
    }

    /// @brief Updates the quantization parameter (QP) with hysteresis.
    ///
    /// If QP decreases (higher quality paint-over), it restarts immediately.
    /// If QP increases (lower quality motion), it waits for the hysteresis limit
    /// to avoid blinking artifacts.
    unsafe fn update_qp(&mut self, target_qp: u32) -> Result<(), String> {
        if target_qp == self.current_qp {
            self.qp_hysteresis_counter = 0;
            return Ok(()).into();
        }

        if target_qp < self.current_qp {
            self.qp_hysteresis_counter = 0;
            self.restart_encoder(target_qp)?;
        } else {
            self.qp_hysteresis_counter += 1;
            if self.qp_hysteresis_counter > QP_HYSTERESIS_LIMIT {
                self.qp_hysteresis_counter = 0;
                self.restart_encoder(target_qp)?;
            }
        }

        Ok(())
    }

    /// @brief Retrieves encoded packets from the encoder and formats them with the custom header.
    unsafe fn collect_packet(&mut self, frame_number: u64, output: &mut Vec<u8>) {
        while ff::avcodec_receive_packet(self.encoder_ctx, self.packet) == 0 {
            let size = (*self.packet).size as usize;
            let data = (*self.packet).data;
            let is_key = ((*self.packet).flags & ff::AV_PKT_FLAG_KEY) != 0;

            output.reserve(10 + size);
            output.push(0x04);
            output.push(if is_key { 0x01 } else { 0x00 });
            output.extend_from_slice(&(frame_number as u16).to_be_bytes());
            output.extend_from_slice(&0u16.to_be_bytes());
            output.extend_from_slice(&(self.width as u16).to_be_bytes());
            output.extend_from_slice(&(self.height as u16).to_be_bytes());

            let slice = std::slice::from_raw_parts(data, size);
            output.extend_from_slice(slice);

            ff::av_packet_unref(self.packet);
        }
    }

    /// @brief Encodes a DMA-BUF frame by importing it via DRM and passing it through the filter graph.
    ///
    /// The filter graph handles mapping the DRM frame to a VAAPI surface and converting colorspace.
    ///
    /// @input dmabuf: The source DMA buffer.
    /// @input frame_number: Frame index.
    /// @input qp: Quality parameter.
    /// @input force_idr: Force keyframe generation.
    /// @return Result containing the encoded packet.
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

            (*desc_ptr).nb_objects = dmabuf.handles().count() as i32;
            (*desc_ptr).nb_layers = 1;

            for (i, (handle, _)) in dmabuf.handles().zip(dmabuf.offsets()).enumerate() {
                let fd = dup(handle.as_raw_fd());
                if fd < 0 {
                    ff::av_free(desc_ptr as *mut c_void);
                    return Err("Failed to dup fd".into());
                }
                resources.fds.push(fd);
                (*desc_ptr).objects[i].fd = fd;
                (*desc_ptr).objects[i].size = (self.width * self.height * 4) as usize;
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

    /// @brief Encodes raw NV12 pixel data by uploading it from CPU memory to the GPU.
    ///
    /// @input nv12_pixels: Raw byte slice of NV12 data.
    /// @input frame_number: Frame index.
    /// @input qp: Quality parameter.
    /// @input force_idr: Force keyframe generation.
    /// @return Result containing the encoded packet.
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

            if ff::av_hwframe_transfer_data(self.hw_frame, self.sw_frame, 0) < 0 {
                return Err("Failed to upload frame to GPU".into());
            }

            ff::av_frame_unref(self.sw_frame);

            (*self.hw_frame).pts = frame_number as i64;
            if force_idr {
                (*self.hw_frame).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_I;
                (*self.hw_frame).flags |= ff::AV_PKT_FLAG_KEY;
            } else {
                (*self.hw_frame).pict_type = ff::AVPictureType::AV_PICTURE_TYPE_NONE;
                (*self.hw_frame).flags &= !ff::AV_PKT_FLAG_KEY;
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
