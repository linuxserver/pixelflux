use crate::RustCaptureSettings;
use rayon::prelude::*;
use smithay::utils::{Physical, Rectangle};
use std::ffi::CString;
use std::ptr;
use yuv::{BufferStoreMut, YuvConversionMode, YuvPlanarImageMut, YuvRange, YuvStandardMatrix};

/// @brief Maximum number of stripes used for CPU encoding.
pub const MAX_STRIPE_CAPACITY: usize = 64;

/// @brief Wrapper around x264-sys for CPU-based H.264 encoding.
///
/// Manages the raw C pointer to the x264 encoder state and handles
/// configuration, cleanup, and frame encoding.
pub struct H264EncoderWrapper {
    encoder: *mut x264_sys::x264_t,
    pub width: i32,
    pub height: i32,
    current_crf: i32,
    pub is_i444: bool,
    #[allow(dead_code)]
    full_range: bool,
}

unsafe impl Send for H264EncoderWrapper {}

impl Drop for H264EncoderWrapper {
    fn drop(&mut self) {
        if !self.encoder.is_null() {
            unsafe { x264_sys::x264_encoder_close(self.encoder) };
            self.encoder = ptr::null_mut();
        }
    }
}

impl H264EncoderWrapper {
    /// @brief Initializes a new x264 encoder instance with zerolatency tuning.
    /// @input width: The width of the frame to encode.
    /// @input height: The height of the frame to encode.
    /// @input crf: The Constant Rate Factor quality setting.
    /// @input is_i444: True for YUV444 (full color), false for YUV420.
    /// @input fps: The target framerate.
    /// @return Option<Self>: The wrapper instance or None if initialization fails.
    pub fn new(width: i32, height: i32, crf: i32, is_i444: bool, fps: f64) -> Option<Self> {
        unsafe {
            let mut param: x264_sys::x264_param_t = std::mem::zeroed();
            let preset = CString::new("ultrafast").unwrap();
            let tune = CString::new("zerolatency").unwrap();

            if x264_sys::x264_param_default_preset(&mut param, preset.as_ptr(), tune.as_ptr()) < 0 {
                return None;
            }

            param.i_width = width;
            param.i_height = height;
            param.i_fps_num = if fps < 1.0 { 30 } else { fps as u32 };
            param.i_fps_den = 1;
            param.i_keyint_max = x264_sys::X264_KEYINT_MAX_INFINITE as i32;
            param.rc.i_rc_method = x264_sys::X264_RC_CRF as i32;
            param.rc.f_rf_constant = crf as f32;
            param.i_csp = if is_i444 {
                x264_sys::X264_CSP_I444
            } else {
                x264_sys::X264_CSP_I420
            } as i32;
            param.vui.b_fullrange = if is_i444 { 1 } else { 0 };
            param.vui.i_colorprim = 1;
            param.vui.i_transfer = 1;
            param.vui.i_colmatrix = 1;

            let profile = CString::new(if is_i444 { "high444" } else { "baseline" }).unwrap();
            x264_sys::x264_param_apply_profile(&mut param, profile.as_ptr());

            param.i_threads = 1;
            param.b_repeat_headers = 1;
            param.b_annexb = 1;
            param.i_log_level = x264_sys::X264_LOG_NONE as i32;

            let encoder = x264_sys::x264_encoder_open(&mut param);
            if encoder.is_null() {
                None
            } else {
                Some(Self {
                    encoder,
                    width,
                    height,
                    current_crf: crf,
                    is_i444,
                    full_range: param.vui.b_fullrange == 1,
                })
            }
        }
    }

    /// @brief Updates the Rate Factor (CRF) dynamically without recreating the encoder.
    /// @input new_crf: The new quality value.
    pub fn reconfigure_crf(&mut self, new_crf: i32) {
        if self.current_crf == new_crf {
            return;
        }
        unsafe {
            let mut param: x264_sys::x264_param_t = std::mem::zeroed();
            x264_sys::x264_encoder_parameters(self.encoder, &mut param);
            param.rc.f_rf_constant = new_crf as f32;
            if x264_sys::x264_encoder_reconfig(self.encoder, &mut param) == 0 {
                self.current_crf = new_crf;
            }
        }
    }

    /// @brief Encodes YUV planes into H.264 NAL units and prepends a custom header.
    /// @input y: Luma plane data.
    /// @input u: Chroma U plane data.
    /// @input v: Chroma V plane data.
    /// @input y_stride: Stride for Y plane.
    /// @input u_stride: Stride for U plane.
    /// @input v_stride: Stride for V plane.
    /// @input frame_id: Monotonically increasing frame index.
    /// @input force_idr: Whether to force an IDR (Keyframe).
    /// @input fixed_header: Custom header bytes to prepend.
    /// @input output_buf: Buffer to store the resulting packet.
    /// @return bool: True if encoding was successful, false otherwise.
    pub fn encode_with_headers(
        &mut self,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        y_stride: i32,
        u_stride: i32,
        v_stride: i32,
        frame_id: i64,
        force_idr: bool,
        fixed_header: &[u8],
        output_buf: &mut Vec<u8>,
    ) -> bool {
        unsafe {
            let mut pic_in: x264_sys::x264_picture_t = std::mem::zeroed();
            x264_sys::x264_picture_init(&mut pic_in);

            pic_in.img.i_csp = if self.is_i444 {
                x264_sys::X264_CSP_I444
            } else {
                x264_sys::X264_CSP_I420
            } as i32;
            pic_in.img.i_plane = 3;
            pic_in.img.plane[0] = y.as_ptr() as *mut u8;
            pic_in.img.plane[1] = u.as_ptr() as *mut u8;
            pic_in.img.plane[2] = v.as_ptr() as *mut u8;
            pic_in.img.i_stride[0] = y_stride;
            pic_in.img.i_stride[1] = u_stride;
            pic_in.img.i_stride[2] = v_stride;
            pic_in.i_pts = frame_id;
            pic_in.i_type = if force_idr {
                x264_sys::X264_TYPE_IDR
            } else {
                x264_sys::X264_TYPE_AUTO
            } as i32;

            let mut pic_out: x264_sys::x264_picture_t = std::mem::zeroed();
            let mut nals: *mut x264_sys::x264_nal_t = ptr::null_mut();
            let mut i_nals: i32 = 0;

            let frame_size = x264_sys::x264_encoder_encode(
                self.encoder,
                &mut nals,
                &mut i_nals,
                &mut pic_in,
                &mut pic_out,
            );

            if frame_size > 0 {
                let header_len = 2 + fixed_header.len();
                let total_len = header_len + frame_size as usize;

                output_buf.clear();
                output_buf.reserve(total_len);
                output_buf.push(0x04);

                let type_byte = if pic_out.i_type == x264_sys::X264_TYPE_IDR as i32 {
                    0x01
                } else if pic_out.i_type == x264_sys::X264_TYPE_I as i32 {
                    0x02
                } else {
                    0x00
                };
                output_buf.push(type_byte);
                output_buf.extend_from_slice(fixed_header);

                let nal_slice = std::slice::from_raw_parts(nals, i_nals as usize);
                for nal in nal_slice {
                    let payload = std::slice::from_raw_parts(nal.p_payload, nal.i_payload as usize);
                    output_buf.extend_from_slice(payload);
                }
                return true;
            }
        }
        false
    }
}

/// @brief State tracking for individual horizontal stripes in CPU encoding mode.
///
/// Stores buffers, encoder instances, and motion counters for a specific
/// slice of the screen to facilitate parallel encoding.
#[derive(Default)]
pub struct StripeState {
    pub no_motion_frame_count: u32,
    pub paint_over_sent: bool,
    pub h264_encoder: Option<H264EncoderWrapper>,
    pub h264_burst_frames_remaining: i32,
    pub y_buf: Vec<u8>,
    pub u_buf: Vec<u8>,
    pub v_buf: Vec<u8>,
    pub packet_buf: Vec<u8>,
}

/// @brief Main CPU encoding logic handling threading, striping, and format conversion.
///
/// Divides the screen into horizontal stripes, checks for damage/motion, converts
/// color formats (RGBA/BGRA to YUV), and encodes using either TurboJPEG or x264.
///
/// @input stripes: Mutable vector of state objects for each thread/stripe.
/// @input raw_pixels: The raw framebuffer data.
/// @input width: Frame width.
/// @input height: Frame height.
/// @input damage_rects: Regions of the screen that changed since last frame.
/// @input settings: Capture settings (Quality, FPS, etc).
/// @input frame_counter: Current frame index.
/// @input use_gpu: Whether the input buffer came from GPU (affects pixel format).
/// @return Vec<Vec<u8>>: A collection of encoded packets for the changed stripes.
pub fn encode_cpu(
    stripes: &mut Vec<StripeState>,
    raw_pixels: &[u8],
    width: i32,
    height: i32,
    damage_rects: &[Rectangle<i32, Physical>],
    settings: &RustCaptureSettings,
    frame_counter: u16,
    use_gpu: bool,
) -> Vec<Vec<u8>> {
    let num_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let min_stripe_height = 64;
    let mut n_processing_stripes = num_cores;

    if settings.output_mode == 1 && settings.h264_fullframe {
        n_processing_stripes = 1;
    } else if height < min_stripe_height {
        n_processing_stripes = 1;
    } else {
        let max_stripes_by_height = (height as usize) / (min_stripe_height as usize);
        n_processing_stripes = n_processing_stripes.min(max_stripes_by_height).max(1);
    }

    if stripes.len() != n_processing_stripes {
        stripes.resize_with(n_processing_stripes, StripeState::default);
    }

    let mut stripe_geometries = Vec::with_capacity(n_processing_stripes);
    let mut current_y = 0;
    let h_usize = height as usize;
    let n = n_processing_stripes;
    let base_h = (h_usize / n) & !1; 
    
    let total_used = base_h * n;
    let remainder = h_usize - total_used;
    let stripes_with_extra = remainder / 2;

    for i in 0..n {
        let extra = if i < stripes_with_extra { 2 } else { 0 };
        let s_h = base_h + extra;
        stripe_geometries.push((current_y, s_h));
        current_y += s_h;
    }
    let mut stripe_is_dirty = vec![false; n_processing_stripes];
    if !damage_rects.is_empty() {
        for rect in damage_rects {
            let r_y_start = rect.loc.y.max(0) as usize;
            let r_y_end = (rect.loc.y + rect.size.h).min(height as i32) as usize;
            if r_y_start < r_y_end {
                for (i, &(s_y, s_h)) in stripe_geometries.iter().enumerate() {
                    let s_end = s_y + s_h;
                    if r_y_start < s_end && r_y_end > s_y {
                        stripe_is_dirty[i] = true;
                    }
                }
            }
        }
    }

    let width_usize = width as usize;
    let output_mode = settings.output_mode;
    let h264_crf = settings.h264_crf;
    let h264_po_crf = settings.h264_paintover_crf;
    let h264_burst = settings.h264_paintover_burst_frames;
    let h264_fullcolor = settings.h264_fullcolor;
    let h264_streaming = settings.h264_streaming_mode;
    let jpeg_q = settings.jpeg_quality;
    let paint_q = settings.paint_over_jpeg_quality;
    let trigger_frames = settings.paint_over_trigger_frames;
    let use_paint_over = settings.use_paint_over_quality;
    let target_fps = settings.target_fps;

    stripes
        .par_iter_mut()
        .enumerate()
        .filter_map(|(i, stripe_state)| {
            if i >= stripe_geometries.len() {
                return None;
            }
            let (y_start, actual_height) = stripe_geometries[i];
            let start_idx = y_start * width_usize * 4;
            let end_idx = start_idx + (actual_height * width_usize * 4);
            let stripe_bytes = &raw_pixels[start_idx..end_idx];

            let mut send_this_stripe = false;
            let mut quality_or_crf = if output_mode == 0 { jpeg_q } else { h264_crf };
            let mut force_idr = false;
            let is_dirty = stripe_is_dirty[i];

            if output_mode == 1 && stripe_state.h264_burst_frames_remaining > 0 {
                send_this_stripe = true;
                quality_or_crf = h264_po_crf;
                stripe_state.h264_burst_frames_remaining -= 1;

                if is_dirty {
                    stripe_state.h264_burst_frames_remaining = 0;
                    stripe_state.paint_over_sent = false;
                    quality_or_crf = h264_crf;
                }
            }

            if !send_this_stripe && output_mode == 1 && h264_streaming {
                send_this_stripe = true;
            }

            if is_dirty {
                send_this_stripe = true;
                stripe_state.no_motion_frame_count = 0;
                stripe_state.paint_over_sent = false;
                stripe_state.h264_burst_frames_remaining = 0;
                quality_or_crf = if output_mode == 0 { jpeg_q } else { h264_crf };
            } else if !send_this_stripe {
                stripe_state.no_motion_frame_count += 1;

                if use_paint_over
                    && stripe_state.no_motion_frame_count >= trigger_frames
                    && !stripe_state.paint_over_sent
                {
                    if output_mode == 0 && paint_q > jpeg_q {
                        send_this_stripe = true;
                        quality_or_crf = paint_q;
                        stripe_state.paint_over_sent = true;
                    } else if output_mode == 1 && h264_po_crf < h264_crf {
                        send_this_stripe = true;
                        stripe_state.paint_over_sent = true;
                        quality_or_crf = h264_po_crf;
                        force_idr = true;
                        stripe_state.h264_burst_frames_remaining = h264_burst - 1;
                    }
                }
            }

            if send_this_stripe {
                if output_mode == 0 {
                    let mut compressor = turbojpeg::Compressor::new().ok()?;
                    compressor.set_quality(quality_or_crf).ok()?;
                    let pixel_format = if use_gpu {
                        turbojpeg::PixelFormat::RGBA
                    } else {
                        turbojpeg::PixelFormat::BGRA
                    };
                    let img = turbojpeg::Image {
                        pixels: stripe_bytes,
                        width: width_usize,
                        pitch: width_usize * 4,
                        height: actual_height,
                        format: pixel_format,
                    };
                    stripe_state.packet_buf.clear();
                    stripe_state
                        .packet_buf
                        .extend_from_slice(&frame_counter.to_be_bytes());
                    stripe_state
                        .packet_buf
                        .extend_from_slice(&(y_start as u16).to_be_bytes());
                    match compressor.compress_to_vec(img) {
                        Ok(jpeg) => {
                            stripe_state.packet_buf.extend_from_slice(&jpeg);
                            Some(stripe_state.packet_buf.clone())
                        }
                        Err(_) => None,
                    }
                } else {
                    let needs_reinit = if let Some(ref enc) = stripe_state.h264_encoder {
                        enc.width != width_usize as i32
                            || enc.height != actual_height as i32
                            || enc.is_i444 != h264_fullcolor
                    } else {
                        true
                    };

                    if needs_reinit {
                        stripe_state.h264_encoder = H264EncoderWrapper::new(
                            width_usize as i32,
                            actual_height as i32,
                            quality_or_crf,
                            h264_fullcolor,
                            target_fps,
                        );
                        force_idr = true;
                    } else if let Some(ref mut enc) = stripe_state.h264_encoder {
                        enc.reconfigure_crf(quality_or_crf);
                    }

                    if let Some(ref mut enc) = stripe_state.h264_encoder {
                        let y_size = width_usize * actual_height;
                        let uv_size = if h264_fullcolor { y_size } else { y_size / 4 };
                        if stripe_state.y_buf.len() != y_size {
                            stripe_state.y_buf.resize(y_size, 0);
                        }
                        if stripe_state.u_buf.len() != uv_size {
                            stripe_state.u_buf.resize(uv_size, 0);
                        }
                        if stripe_state.v_buf.len() != uv_size {
                            stripe_state.v_buf.resize(uv_size, 0);
                        }

                        let y_stride = width_usize as i32;
                        let uv_stride = if h264_fullcolor {
                            width_usize
                        } else {
                            width_usize / 2
                        } as i32;

                        let mut planar_image = YuvPlanarImageMut {
                            y_plane: BufferStoreMut::Borrowed(&mut stripe_state.y_buf),
                            y_stride: y_stride as u32,
                            u_plane: BufferStoreMut::Borrowed(&mut stripe_state.u_buf),
                            u_stride: uv_stride as u32,
                            v_plane: BufferStoreMut::Borrowed(&mut stripe_state.v_buf),
                            v_stride: uv_stride as u32,
                            width: width_usize as u32,
                            height: actual_height as u32,
                        };

                        if h264_fullcolor {
                            if use_gpu {
                                let _ = yuv::rgba_to_yuv444(
                                    &mut planar_image,
                                    stripe_bytes,
                                    (width_usize * 4) as u32,
                                    YuvRange::Full,
                                    YuvStandardMatrix::Bt709,
                                    YuvConversionMode::Balanced,
                                );
                            } else {
                                let _ = yuv::bgra_to_yuv444(
                                    &mut planar_image,
                                    stripe_bytes,
                                    (width_usize * 4) as u32,
                                    YuvRange::Full,
                                    YuvStandardMatrix::Bt709,
                                    YuvConversionMode::Balanced,
                                );
                            }
                        } else {
                            if use_gpu {
                                let _ = yuv::rgba_to_yuv420(
                                    &mut planar_image,
                                    stripe_bytes,
                                    (width_usize * 4) as u32,
                                    YuvRange::Limited,
                                    YuvStandardMatrix::Bt709,
                                    YuvConversionMode::Balanced,
                                );
                            } else {
                                let _ = yuv::bgra_to_yuv420(
                                    &mut planar_image,
                                    stripe_bytes,
                                    (width_usize * 4) as u32,
                                    YuvRange::Limited,
                                    YuvStandardMatrix::Bt709,
                                    YuvConversionMode::Balanced,
                                );
                            }
                        }

                        let mut fixed_header = [0u8; 8];
                        fixed_header[0..2].copy_from_slice(&frame_counter.to_be_bytes());
                        fixed_header[2..4].copy_from_slice(&(y_start as u16).to_be_bytes());
                        fixed_header[4..6].copy_from_slice(&(width_usize as u16).to_be_bytes());
                        fixed_header[6..8].copy_from_slice(&(actual_height as u16).to_be_bytes());

                        if enc.encode_with_headers(
                            &stripe_state.y_buf,
                            &stripe_state.u_buf,
                            &stripe_state.v_buf,
                            y_stride,
                            uv_stride,
                            uv_stride,
                            frame_counter as i64,
                            force_idr,
                            &fixed_header,
                            &mut stripe_state.packet_buf,
                        ) {
                            Some(stripe_state.packet_buf.clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
            } else {
                None
            }
        })
        .collect()
}
