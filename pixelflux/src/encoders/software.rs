/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use crate::recording_sink::RecordingSink;
use crate::RustCaptureSettings;
use rayon::prelude::*;
use smithay::utils::{Physical, Rectangle};
use std::ffi::CString;
use std::ptr;
use std::sync::Arc;
use yuv::{BufferStoreMut, YuvConversionMode, YuvPlanarImageMut, YuvRange, YuvStandardMatrix};

// Maximum number of stripes used for CPU encoding.
pub const MAX_STRIPE_CAPACITY: usize = 64;

/// BGRA/RGBA -> planar YUV, split into up to `bands` horizontal bands converted in
/// parallel. Full-frame encoders convert the whole image in one call (the striped path
/// already parallelizes across stripes), so the band split gives the conversion the
/// same multi-threading the encoder itself gets. Band edges stay even so 4:2:0 chroma
/// pairs never straddle a seam; `bands == 1` is a plain single-threaded conversion.
#[allow(clippy::too_many_arguments)]
pub(crate) fn convert_to_yuv_mt(
    src: &[u8],
    src_stride: u32,
    width: usize,
    height: usize,
    rgba_input: bool,
    i444: bool,
    y_buf: &mut [u8],
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    bands: usize,
) -> Result<(), yuv::YuvError> {
    let y_stride = width;
    let uv_stride = if i444 { width } else { width / 2 };

    let convert_band = |src_band: &[u8], y: &mut [u8], u: &mut [u8], v: &mut [u8], h: usize| {
        let mut img = YuvPlanarImageMut {
            y_plane: BufferStoreMut::Borrowed(y),
            y_stride: y_stride as u32,
            u_plane: BufferStoreMut::Borrowed(u),
            u_stride: uv_stride as u32,
            v_plane: BufferStoreMut::Borrowed(v),
            v_stride: uv_stride as u32,
            width: width as u32,
            height: h as u32,
        };
        match (i444, rgba_input) {
            (true, true) => yuv::rgba_to_yuv444(
                &mut img, src_band, src_stride, YuvRange::Full,
                YuvStandardMatrix::Bt709, YuvConversionMode::Fast,
            ),
            (true, false) => yuv::bgra_to_yuv444(
                &mut img, src_band, src_stride, YuvRange::Full,
                YuvStandardMatrix::Bt709, YuvConversionMode::Fast,
            ),
            (false, true) => yuv::rgba_to_yuv420(
                &mut img, src_band, src_stride, YuvRange::Limited,
                YuvStandardMatrix::Bt709, YuvConversionMode::Fast,
            ),
            (false, false) => yuv::bgra_to_yuv420(
                &mut img, src_band, src_stride, YuvRange::Limited,
                YuvStandardMatrix::Bt709, YuvConversionMode::Fast,
            ),
        }
    };

    // Even-aligned band height; a band smaller than 2 rows isn't worth a thread.
    let band_h = ((height / bands.max(1)) & !1).max(2);
    if bands <= 1 || height <= band_h {
        return convert_band(src, y_buf, u_buf, v_buf, height);
    }

    let uv_rows = |rows: usize| if i444 { rows } else { rows / 2 };
    let mut results: Vec<Result<(), yuv::YuvError>> = Vec::new();
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        let (mut src_rest, mut y_rest, mut u_rest, mut v_rest) = (src, y_buf, u_buf, v_buf);
        let mut row = 0;
        while row < height {
            let h = if height - row < band_h + 2 { height - row } else { band_h };
            let (src_band, s_next) = src_rest.split_at(h * src_stride as usize);
            let (y_band, y_next) = y_rest.split_at_mut(h * y_stride);
            let (u_band, u_next) = u_rest.split_at_mut(uv_rows(h) * uv_stride);
            let (v_band, v_next) = v_rest.split_at_mut(uv_rows(h) * uv_stride);
            src_rest = s_next;
            y_rest = y_next;
            u_rest = u_next;
            v_rest = v_next;
            row += h;
            handles.push(s.spawn(move || convert_band(src_band, y_band, u_band, v_band, h)));
        }
        for hnd in handles {
            results.push(hnd.join().unwrap_or(Err(yuv::YuvError::PointerOverflow)));
        }
    });
    results.into_iter().collect()
}

thread_local! {
    /// One reusable libjpeg-turbo compressor per worker thread (rayon striped path or the encode
    /// thread in single-stripe mode) -- avoids a tjInitCompress/tjDestroy per stripe per frame.
    static JPEG_COMPRESSOR: std::cell::RefCell<Option<turbojpeg::Compressor>> =
        const { std::cell::RefCell::new(None) };
}

/// Process-global lock serializing libx264 encoder open/close. libx264 mutates global state
/// during x264_encoder_open/close, so concurrent open/close from parallel stripe encoders (or
/// multiple capture instances in one process) can corrupt the heap. Held only around open/close,
/// never during encode.
static X264_OPEN_CLOSE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Wraps the raw x264-sys encoder pointer for CPU H.264 encoding.
pub struct H264EncoderWrapper {
    encoder: *mut x264_sys::x264_t,
    pub width: i32,
    pub height: i32,
    current_crf: i32,
    pub is_i444: bool,
    is_cbr: bool,
    current_bitrate: i32,
    current_vbv: i32,
    current_fps: u32,
    #[allow(dead_code)]
    full_range: bool,
}

unsafe impl Send for H264EncoderWrapper {}

impl Drop for H264EncoderWrapper {
    fn drop(&mut self) {
        if !self.encoder.is_null() {
            let _guard = X264_OPEN_CLOSE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            unsafe { x264_sys::x264_encoder_close(self.encoder) };
            self.encoder = ptr::null_mut();
        }
    }
}

impl H264EncoderWrapper {
    /// Creates an x264 encoder instance tuned for zerolatency.
    #[allow(clippy::too_many_arguments)]
    pub fn new(width: i32, height: i32, crf: i32, is_i444: bool, fps: f64, threads: i32,
               cbr_mode: bool, bitrate_kbps: i32, vbv_kbit: i32,
               min_qp: i32, max_qp: i32) -> Option<Self> {
        unsafe {
            let mut param: x264_sys::x264_param_t = std::mem::zeroed();
            let preset = CString::new("superfast").unwrap();
            let tune = CString::new("zerolatency").unwrap();

            if x264_sys::x264_param_default_preset(&mut param, preset.as_ptr(), tune.as_ptr()) < 0 {
                return None;
            }

            param.i_width = width;
            param.i_height = height;
            param.i_fps_num = if fps < 1.0 { 30 } else { fps as u32 };
            param.i_fps_den = 1;
            param.i_keyint_max = x264_sys::X264_KEYINT_MAX_INFINITE as i32;
            // On-demand keyframes only: disable adaptive scene-cut so a scene change
            // cannot inject an unrequested IDR (infinite GOP + forced IDR is the model).
            param.i_scenecut_threshold = 0;
            if cbr_mode {
                // ABR with a VBV cap (kbit, precomputed by the caller from the
                // frame-time multiplier policy); no filler.
                let bk = bitrate_kbps.saturating_abs();
                param.rc.i_rc_method = x264_sys::X264_RC_ABR as i32;
                param.rc.i_bitrate = bk;
                param.rc.i_vbv_max_bitrate = bk;
                param.rc.i_vbv_buffer_size = vbv_kbit.max(1);
                param.rc.b_filler = 0;
                // Optional RC clamp: max = legibility floor, min = waste ceiling.
                if min_qp > 0 {
                    param.rc.i_qp_min = min_qp.min(51);
                }
                if max_qp > 0 {
                    param.rc.i_qp_max = max_qp.min(51);
                }
            } else {
                param.rc.i_rc_method = x264_sys::X264_RC_CRF as i32;
                param.rc.f_rf_constant = crf as f32;
            }
            param.i_csp = if is_i444 {
                x264_sys::X264_CSP_I444
            } else {
                x264_sys::X264_CSP_I420
            } as i32;
            param.vui.b_fullrange = if is_i444 { 1 } else { 0 };
            param.vui.i_colorprim = 1;
            param.vui.i_transfer = 1;
            param.vui.i_colmatrix = 1;

            let profile = CString::new(if is_i444 { "high444" } else { "high" }).unwrap();
            x264_sys::x264_param_apply_profile(&mut param, profile.as_ptr());
            // Pin the High-profile coding tools on regardless of preset defaults (the
            // HW encoders already ship High/CABAC and every targeted decoder handles it).
            param.b_cabac = 1;
            param.analyse.b_transform_8x8 = 1;

            param.i_threads = threads;
            param.b_repeat_headers = 1;
            param.b_annexb = 1;
            param.i_log_level = x264_sys::X264_LOG_NONE;

            let encoder = {
                let _guard = X264_OPEN_CLOSE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                x264_sys::x264_encoder_open(&mut param)
            };
            if encoder.is_null() {
                None
            } else {
                Some(Self {
                    encoder,
                    width,
                    height,
                    current_crf: crf,
                    is_i444,
                    is_cbr: cbr_mode,
                    current_bitrate: bitrate_kbps.saturating_abs(),
                    current_vbv: vbv_kbit,
                    current_fps: if fps < 1.0 { 30 } else { fps as u32 },
                    full_range: param.vui.b_fullrange == 1,
                })
            }
        }
    }

    /// Updates CRF live, without recreating the encoder.
    pub fn reconfigure_crf(&mut self, new_crf: i32) {
        if self.is_cbr || self.current_crf == new_crf {
            return; // CBR is bitrate-controlled; CRF reconfig doesn't apply.
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

    /// Applies a runtime bitrate/VBV (CBR only) and framerate change without recreating
    /// the encoder. Cheap to call every frame; reconfigures only when a value actually changed.
    pub fn reconfigure_rate(&mut self, bitrate_kbps: i32, vbv_kbit: i32, fps: f64) {
        let bk = bitrate_kbps.saturating_abs();
        let new_fps = if fps < 1.0 { 30 } else { fps as u32 };
        let rate_changed =
            self.is_cbr && (self.current_bitrate != bk || self.current_vbv != vbv_kbit);
        let fps_changed = self.current_fps != new_fps;
        if !rate_changed && !fps_changed {
            return;
        }
        unsafe {
            let mut param: x264_sys::x264_param_t = std::mem::zeroed();
            x264_sys::x264_encoder_parameters(self.encoder, &mut param);
            if rate_changed {
                param.rc.i_bitrate = bk;
                param.rc.i_vbv_max_bitrate = bk;
                param.rc.i_vbv_buffer_size = vbv_kbit.max(1);
            }
            if fps_changed {
                param.i_fps_num = new_fps;
                param.i_fps_den = 1;
            }
            if x264_sys::x264_encoder_reconfig(self.encoder, &mut param) == 0 {
                if rate_changed {
                    self.current_bitrate = bk;
                    self.current_vbv = vbv_kbit;
                }
                if fps_changed {
                    self.current_fps = new_fps;
                }
            }
        }
    }

    /// Encodes YUV planes into H.264 NAL units, prepending a custom header.
    #[allow(clippy::too_many_arguments)]
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
        omit_headers: bool,
        output_buf: &mut Vec<u8>,
        recording_sink: Option<&Arc<RecordingSink>>,
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
                let header_len = if omit_headers { 0 } else { 2 + fixed_header.len() };
                let total_len = header_len + frame_size as usize;

                output_buf.clear();
                output_buf.reserve(total_len);

                if !omit_headers {
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
                }

                let nal_slice = std::slice::from_raw_parts(nals, i_nals as usize);
                for nal in nal_slice {
                    let payload = std::slice::from_raw_parts(nal.p_payload, nal.i_payload as usize);
                    output_buf.extend_from_slice(payload);
                    if let Some(sink) = recording_sink {
                        sink.write_frame(payload);
                    }
                }
                return true;
            }
        }
        false
    }
}

// Per-stripe state (buffers, encoder, motion counters) for parallel CPU encoding.
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
    // Content-hash damage state, used by sources without external damage (X11).
    pub last_hash: u64,
    pub consecutive_changes: u32,
    pub in_damage_block: bool,
    pub damage_block_frames_remaining: i32,
    pub hash_at_block_start: u64,
}

/// Fast 64-bit content hash for change detection. NOT cryptographic; a collision between two
/// distinct stripes is ~2^-64 -- and the next real content change (or a requested keyframe)
/// repaints a missed update anyway. Used ONLY to compare a stripe against its previous frame in memory
/// (never persisted or sent on the wire), so the exact value is irrelevant -- only that identical
/// bytes hash identically.
///
/// Eight independent FNV-1a lanes over interleaved 8-byte words break the serial multiply
/// dependency chain of a single accumulator (the bottleneck at ~6.5 GB/s single-thread), letting
/// the CPU keep several multiplies in flight; the lanes fold into one 64-bit value at the end.
fn fast_hash(bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x100000001b3;
    const SEED: u64 = 0xcbf29ce484222325;
    const LANES: usize = 8;
    const STRIDE: usize = LANES * 8; // 64 bytes per interleaved block

    let mut h = [SEED; LANES];
    let mut blocks = bytes.chunks_exact(STRIDE);
    for b in &mut blocks {
        for (lane, acc) in h.iter_mut().enumerate() {
            let off = lane * 8;
            let w = u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
            *acc = (*acc ^ w).wrapping_mul(PRIME);
        }
    }
    // Fold the eight lanes into one accumulator.
    let mut acc = SEED;
    for lane in h {
        acc = (acc ^ lane).wrapping_mul(PRIME);
    }
    // Tail (< 64 bytes): whole 8-byte words then the trailing bytes.
    let rem = blocks.remainder();
    let mut words = rem.chunks_exact(8);
    for w in &mut words {
        let w = u64::from_le_bytes(w.try_into().unwrap());
        acc = (acc ^ w).wrapping_mul(PRIME);
    }
    for &byte in words.remainder() {
        acc = (acc ^ byte as u64).wrapping_mul(PRIME);
    }
    acc
}

impl StripeState {
    /// Content-hash damage detection for sources without external damage (X11): reports
    /// whether this stripe changed, maintaining a damage block so a continuously-moving
    /// region keeps sending for `duration` frames once `threshold` consecutive changes are
    /// seen, re-hashing only at block end to decide whether to extend the block or exit it.
    pub fn content_dirty(&mut self, bytes: &[u8], threshold: u32, duration: i32) -> bool {
        if self.in_damage_block {
            self.damage_block_frames_remaining -= 1;
            if self.damage_block_frames_remaining <= 0 {
                let h = fast_hash(bytes);
                if h != self.hash_at_block_start {
                    self.damage_block_frames_remaining = duration;
                    self.hash_at_block_start = h;
                } else {
                    self.in_damage_block = false;
                    self.consecutive_changes = 0;
                }
                self.last_hash = h;
            }
            return true;
        }
        let h = fast_hash(bytes);
        let changed = h != self.last_hash;
        self.last_hash = h;
        if changed {
            self.consecutive_changes += 1;
            if self.consecutive_changes >= threshold {
                self.in_damage_block = true;
                self.damage_block_frames_remaining = duration;
                self.hash_at_block_start = h;
            }
        } else {
            self.consecutive_changes = 0;
        }
        changed
    }
}

/// Encoded stripe payload plus the metadata the consumer needs as frame attributes
/// (so it stays available even when the per-stripe header is omitted). data_type is the
/// wire codec tag: JPEG=1, H.264=2.
pub struct EncodedStripe {
    pub data: Vec<u8>,
    pub data_type: i32,
    pub stripe_y_start: i32,
    pub stripe_height: i32,
    pub frame_id: i32,
}

/// Main CPU encoding entry: divides the screen into horizontal stripes, checks
/// damage/motion, converts RGBA/BGRA to YUV, and encodes each with TurboJPEG or
/// x264. `force_idr_all` forces a send + IDR on every H.264 stripe this frame
/// (on-demand or the configured keyframe interval); JPEG stripes ignore it.
#[allow(clippy::too_many_arguments)]
pub fn encode_cpu(
    stripes: &mut Vec<StripeState>,
    raw_pixels: &[u8],
    width: i32,
    height: i32,
    damage_rects: &[Rectangle<i32, Physical>],
    settings: &RustCaptureSettings,
    frame_counter: u16,
    use_gpu: bool,
    // When true (X11, no compositor damage), per-stripe content hashing drives damage
    // instead of `damage_rects`; see StripeState::content_dirty.
    hash_damage: bool,
    recording_sink: Option<&Arc<RecordingSink>>,
    force_idr_all: bool,
) -> Vec<EncodedStripe> {
    let num_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let min_stripe_height = 64;
    let mut n_processing_stripes = num_cores;

    if (settings.output_mode == 1 && settings.video_fullframe) || height < min_stripe_height {
        n_processing_stripes = 1;
    } else {
        let max_stripes_by_height = (height as usize) / (min_stripe_height as usize);
        n_processing_stripes = n_processing_stripes.min(max_stripes_by_height).max(1);
    }

    if stripes.len() != n_processing_stripes {
        stripes.resize_with(n_processing_stripes, StripeState::default);
    }

    let stripe_geometries =
        compute_stripe_geometries(height as usize, n_processing_stripes, settings.output_mode);
    let mut stripe_is_dirty = vec![false; n_processing_stripes];
    if !damage_rects.is_empty() {
        for rect in damage_rects {
            let r_y_start = rect.loc.y.max(0) as usize;
            let r_y_end = (rect.loc.y + rect.size.h).min(height) as usize;
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
    let video_crf = settings.video_crf;
    let video_po_crf = settings.video_paintover_crf;
    let video_burst = settings.video_paintover_burst_frames;
    let video_fullcolor = settings.video_fullcolor;
    let video_streaming = settings.video_streaming_mode;
    let jpeg_q = settings.jpeg_quality;
    let paint_q = settings.paint_over_jpeg_quality;
    let trigger_frames = settings.paint_over_trigger_frames;
    let use_paint_over = settings.use_paint_over_quality;
    // Burst/recovery frames use the paint-over CRF when it is enabled and actually better,
    // otherwise the base CRF (a recovery burst still needs to stream so CBR can refine).
    let burst_crf = if use_paint_over && video_po_crf < video_crf { video_po_crf } else { video_crf };
    let target_fps = settings.target_fps;
    let omit_headers = settings.omit_stripe_headers;
    let damage_block_threshold = settings.damage_block_threshold;
    let damage_block_duration = settings.damage_block_duration as i32;
    let video_cbr = settings.video_cbr_mode;
    let video_bitrate = settings.video_bitrate_kbps;
    // CBR VBV in kbit from the frame-time multiplier policy, recomputed here so live
    // bitrate/fps changes rescale it.
    let video_vbv = (crate::encoders::vbv_bits(
        (video_bitrate.max(0) as u32).saturating_mul(1000),
        target_fps,
        settings.keyframe_interval_s,
        settings.video_vbv_multiplier,
    ) / 1000)
        .max(1) as i32;
    // Single full-frame stripe: four x264 threads (zerolatency uses sliced-threads, so
    // this is four slices — more upsets some Chromium decoders). Striped mode keeps
    // 1 thread per stripe -- parallelism comes from rayon across stripes.
    let h264_threads = if n_processing_stripes == 1 { 4 } else { 1 };
    let csc_bands = if n_processing_stripes == 1 { 4 } else { 1 };
    let stripe_sink: Option<Arc<RecordingSink>> = if n_processing_stripes == 1 {
        recording_sink.cloned()
    } else {
        None
    };

    let stripe_body = |(i, stripe_state): (usize, &mut StripeState)| -> Option<EncodedStripe> {
            if i >= stripe_geometries.len() {
                return None;
            }
            let (y_start, actual_height) = stripe_geometries[i];
            let start_idx = y_start * width_usize * 4;
            let end_idx = start_idx + (actual_height * width_usize * 4);
            let stripe_bytes = &raw_pixels[start_idx..end_idx];

            let mut send_this_stripe = false;
            let mut quality_or_crf = if output_mode == 0 { jpeg_q } else { video_crf };
            let mut force_idr = false;
            let is_dirty = if !hash_damage {
                stripe_is_dirty[i]
            } else if output_mode == 1 && video_streaming {
                // Streaming H.264 sends every stripe unconditionally below, so the content
                // hash is unused here — skip it.
                false
            } else {
                stripe_state.content_dirty(stripe_bytes, damage_block_threshold, damage_block_duration)
            };

            if output_mode == 1 && stripe_state.h264_burst_frames_remaining > 0 {
                send_this_stripe = true;
                quality_or_crf = burst_crf;
                stripe_state.h264_burst_frames_remaining -= 1;

                if is_dirty {
                    stripe_state.h264_burst_frames_remaining = 0;
                    stripe_state.paint_over_sent = false;
                    quality_or_crf = video_crf;
                }
            }

            if !send_this_stripe && output_mode == 1 && video_streaming {
                send_this_stripe = true;
            }

            if is_dirty {
                send_this_stripe = true;
                stripe_state.no_motion_frame_count = 0;
                stripe_state.paint_over_sent = false;
                stripe_state.h264_burst_frames_remaining = 0;
                quality_or_crf = if output_mode == 0 { jpeg_q } else { video_crf };
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
                    } else if output_mode == 1 && video_po_crf < video_crf {
                        send_this_stripe = true;
                        stripe_state.paint_over_sent = true;
                        quality_or_crf = video_po_crf;
                        force_idr = true;
                        stripe_state.h264_burst_frames_remaining = video_burst - 1;
                    }
                }
            }

            // Recovery IDR: force a send + IDR even on a static stripe so a reconnecting
            // client can resume. The keyframe is base-quality -- and CBR rate control crashes
            // it further -- and a damage-gated static stream sends nothing afterward to refine
            // it (turbo/streaming mode never hits this). Arm a burst so the encoder keeps
            // streaming briefly and recovers; skip if a burst/paint-over is already pending so
            // it can't preempt one. no_motion_frame_count is left alone.
            if force_idr_all && output_mode == 1 {
                send_this_stripe = true;
                force_idr = true;
                if stripe_state.h264_burst_frames_remaining <= 0 && video_burst > 0 {
                    stripe_state.paint_over_sent = true;
                    stripe_state.h264_burst_frames_remaining = video_burst;
                }
            }

            if send_this_stripe {
                if output_mode == 0 {
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
                    // Reuse this worker thread's compressor (created once) instead of a fresh one
                    // per stripe per frame.
                    JPEG_COMPRESSOR.with(|cell| -> Option<EncodedStripe> {
                        let mut slot = cell.borrow_mut();
                        if slot.is_none() {
                            *slot = Some(turbojpeg::Compressor::new().ok()?);
                        }
                        let compressor = slot.as_mut().unwrap();
                        compressor.set_quality(quality_or_crf).ok()?;
                        let jpeg = compressor.compress_to_vec(img).ok()?;
                        // Header-less: hand the encoded buffer straight through (drops the extra
                        // copy into packet_buf). With headers: emit the full 6-byte stripe header
                        // (0x03 JPEG type + reserved byte + frame#/y), matching the H.264 path's
                        // native 0x04 so the WS transport sends the buffer with no Python re-frame.
                        let data = if omit_headers {
                            jpeg
                        } else {
                            stripe_state.packet_buf.clear();
                            stripe_state.packet_buf.push(0x03);
                            stripe_state.packet_buf.push(0x00);
                            stripe_state
                                .packet_buf
                                .extend_from_slice(&frame_counter.to_be_bytes());
                            stripe_state
                                .packet_buf
                                .extend_from_slice(&(y_start as u16).to_be_bytes());
                            stripe_state.packet_buf.extend_from_slice(&jpeg);
                            std::mem::take(&mut stripe_state.packet_buf)
                        };
                        Some(EncodedStripe {
                            data,
                            data_type: 1,
                            stripe_y_start: y_start as i32,
                            // Report the actual stripe height (JPEG carries it as frame metadata too).
                            stripe_height: actual_height as i32,
                            frame_id: frame_counter as i32,
                        })
                    })
                } else {
                    let needs_reinit = if let Some(ref enc) = stripe_state.h264_encoder {
                        enc.width != width_usize as i32
                            || enc.height != actual_height as i32
                            || enc.is_i444 != video_fullcolor
                    } else {
                        true
                    };

                    if needs_reinit {
                        stripe_state.h264_encoder = H264EncoderWrapper::new(
                            width_usize as i32,
                            actual_height as i32,
                            quality_or_crf,
                            video_fullcolor,
                            target_fps,
                            h264_threads,
                            video_cbr,
                            video_bitrate,
                            video_vbv,
                            settings.video_min_qp,
                            settings.video_max_qp,
                        );
                        force_idr = true;
                    } else if let Some(ref mut enc) = stripe_state.h264_encoder {
                        enc.reconfigure_crf(quality_or_crf);
                        enc.reconfigure_rate(video_bitrate, video_vbv, target_fps);
                    }

                    if let Some(ref mut enc) = stripe_state.h264_encoder {
                        let y_size = width_usize * actual_height;
                        let uv_size = if video_fullcolor { y_size } else { y_size / 4 };
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
                        let uv_stride =
                            (if video_fullcolor { width_usize } else { width_usize / 2 }) as i32;
                        let conversion_result = convert_to_yuv_mt(
                            stripe_bytes,
                            (width_usize * 4) as u32,
                            width_usize,
                            actual_height,
                            use_gpu,
                            video_fullcolor,
                            &mut stripe_state.y_buf,
                            &mut stripe_state.u_buf,
                            &mut stripe_state.v_buf,
                            csc_bands,
                        );

                        // Skip the stripe on conversion failure instead of encoding garbage.
                        if let Err(e) = conversion_result {
                            eprintln!(
                                "[software] YUV conversion failed for {}x{} stripe: {:?}; skipping",
                                width_usize, actual_height, e
                            );
                            return None;
                        }

                        let mut fixed_header = [0u8; 8];
                        fixed_header[0..2].copy_from_slice(&frame_counter.to_be_bytes());
                        fixed_header[2..4].copy_from_slice(&(y_start as u16).to_be_bytes());
                        fixed_header[4..6].copy_from_slice(&(width_usize as u16).to_be_bytes());
                        fixed_header[6..8].copy_from_slice(&(actual_height as u16).to_be_bytes());

                        let force_idr_for_recording = stripe_sink
                            .as_ref()
                            .map(|s| s.should_force_idr())
                            .unwrap_or(false);

                        if enc.encode_with_headers(
                            &stripe_state.y_buf,
                            &stripe_state.u_buf,
                            &stripe_state.v_buf,
                            y_stride,
                            uv_stride,
                            uv_stride,
                            frame_counter as i64,
                            force_idr || force_idr_for_recording,
                            &fixed_header,
                            omit_headers,
                            &mut stripe_state.packet_buf,
                            stripe_sink.as_ref(),
                        ) {
                            Some(EncodedStripe {
                                data: std::mem::take(&mut stripe_state.packet_buf),
                                data_type: 2,
                                stripe_y_start: y_start as i32,
                                stripe_height: actual_height as i32,
                                frame_id: frame_counter as i32,
                            })
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
    };
    // Single full-frame stripe: run inline (sequential) -- empirically faster than a one-element
    // rayon dispatch. Striped mode keeps rayon to parallelize the encode across stripes.
    if n_processing_stripes <= 1 {
        stripes.iter_mut().enumerate().filter_map(&stripe_body).collect()
    } else {
        stripes.par_iter_mut().enumerate().filter_map(&stripe_body).collect()
    }
}

/// Splits `height` rows into `n` stripes. JPEG (output_mode 0) covers every row;
/// H.264 keeps even stripe heights for 4:2:0, leaving any trailing odd row uncovered.
fn compute_stripe_geometries(height: usize, n: usize, output_mode: i32) -> Vec<(usize, usize)> {
    let mut geoms = Vec::with_capacity(n);
    let mut current_y = 0;
    if output_mode == 0 {
        let base_h = height / n;
        let remainder = height - base_h * n;
        for i in 0..n {
            let s_h = base_h + if i < remainder { 1 } else { 0 };
            geoms.push((current_y, s_h));
            current_y += s_h;
        }
    } else {
        let base_h = (height / n) & !1;
        let remainder = height - base_h * n;
        let stripes_with_extra = remainder / 2;
        for i in 0..n {
            let s_h = base_h + if i < stripes_with_extra { 2 } else { 0 };
            geoms.push((current_y, s_h));
            current_y += s_h;
        }
    }
    geoms
}

#[cfg(test)]
mod tests {
    use super::{compute_stripe_geometries, StripeState};

    #[test]
    fn content_dirty_detects_change_and_damage_block() {
        let mut st = StripeState::default();
        let a = vec![1u8; 256];
        let b = vec![2u8; 256];
        assert!(st.content_dirty(&a, 2, 3)); // changed vs zero-initialized hash
        assert!(!st.content_dirty(&a, 2, 3)); // stable
        assert!(st.content_dirty(&b, 2, 3)); // change #1 (consecutive=1)
        assert!(st.content_dirty(&a, 2, 3)); // change #2 -> enters damage block
        assert!(st.in_damage_block);
        assert!(st.content_dirty(&a, 2, 3)); // block holds dirty, no re-hash (rem 3->2)
        assert!(st.content_dirty(&a, 2, 3)); // block (2->1)
        assert!(st.content_dirty(&a, 2, 3)); // block end (1->0): re-hash, unchanged -> exit
        assert!(!st.in_damage_block);
        assert!(!st.content_dirty(&a, 2, 3)); // stable again
    }

    fn covered(geoms: &[(usize, usize)]) -> usize {
        geoms.iter().map(|&(_, h)| h).sum()
    }

    fn assert_contiguous(geoms: &[(usize, usize)]) {
        let mut y = 0;
        for &(sy, sh) in geoms {
            assert_eq!(sy, y, "stripes must be contiguous");
            y += sh;
        }
    }

    #[test]
    fn jpeg_covers_every_row_including_odd() {
        for &h in &[1usize, 63, 720, 721, 1079, 1080, 1081] {
            for &n in &[1usize, 2, 3, 8, 16] {
                let g = compute_stripe_geometries(h, n, 0);
                assert_eq!(g.len(), n);
                assert_eq!(covered(&g), h, "JPEG must cover full height h={} n={}", h, n);
                assert_contiguous(&g);
            }
        }
    }

    #[test]
    fn h264_stripes_even_and_within_bounds() {
        for &h in &[64usize, 720, 721, 1080, 1081] {
            for &n in &[1usize, 2, 8] {
                let g = compute_stripe_geometries(h, n, 1);
                assert_eq!(g.len(), n);
                for &(_, sh) in &g {
                    assert_eq!(sh % 2, 0, "H.264 stripe heights must be even h={} n={}", h, n);
                }
                assert_contiguous(&g);
                assert!(covered(&g) <= h);
                assert!(h - covered(&g) <= 1, "at most one trailing odd row uncovered");
            }
        }
    }
}

#[cfg(test)]
mod qp_bound_sweep {
    //! Invariants under test: the CBR QP clamp reaches libx264/OpenH264 (a max clamp must
    //! raise worst-case fidelity on rate-starved text at the cost of bitrate overshoot;
    //! a min clamp must cut spend on over-budgeted content) and defaults (0) leave the
    //! encoders' own behavior untouched. The printed table is the measurement behind the
    //! shipped guidance for video_min_qp/video_max_qp.
    use super::H264EncoderWrapper;
    use crate::encoders::oh264::Openh264Encoder;
    use crate::RustCaptureSettings;
    use openh264::decoder::Decoder;
    use openh264::formats::YUVSource;

    const W: usize = 1280;
    const H: usize = 720;
    const FRAMES: usize = 60;

    // Terminal-like text: an 8x12 glyph grid from an LCG, scrolled 4 px per frame.
    // Worst case for screen-share RC: dense high-contrast detail, full-frame motion.
    fn text_luma(frame: usize) -> Vec<u8> {
        let mut y = vec![18u8; W * H];
        let scroll = frame * 4;
        for row in 0..H {
            let srow = row + scroll;
            let cell_y = srow / 12;
            let in_glyph_y = srow % 12;
            if in_glyph_y >= 10 {
                continue; // line spacing
            }
            for col in 0..W {
                let cell_x = col / 8;
                let in_glyph_x = col % 8;
                if in_glyph_x >= 7 {
                    continue; // char spacing
                }
                let mut s = (cell_x as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add((cell_y as u32).wrapping_mul(40503))
                    .wrapping_add(1);
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                // ~40% lit pixels per glyph row pattern
                if (s >> ((in_glyph_y * 3 + in_glyph_x) % 29)) & 1 == 1 {
                    y[row * W + col] = 224;
                }
            }
        }
        y
    }

    fn encode_x264(cbr: bool, kbps: i32, crf: i32, min_qp: i32, max_qp: i32) -> Vec<Vec<u8>> {
        let mut enc = H264EncoderWrapper::new(
            W as i32, H as i32, crf, false, 60.0, 4, cbr, kbps, 50, min_qp, max_qp,
        )
        .expect("x264 init");
        let u = vec![128u8; (W / 2) * (H / 2)];
        let v = vec![128u8; (W / 2) * (H / 2)];
        (0..FRAMES)
            .map(|i| {
                let y = text_luma(i);
                let mut out = Vec::new();
                enc.encode_with_headers(
                    &y, &u, &v, W as i32, (W / 2) as i32, (W / 2) as i32,
                    i as i64, i == 0, &[], true, &mut out, None,
                );
                out
            })
            .collect()
    }

    fn encode_oh264(cbr: bool, kbps: i32, crf: i32, min_qp: i32, max_qp: i32) -> Vec<Vec<u8>> {
        let s = RustCaptureSettings {
            width: W as i32,
            height: H as i32,
            target_fps: 60.0,
            output_mode: 1,
            video_cbr_mode: cbr,
            video_bitrate_kbps: kbps,
            video_crf: crf,
            video_min_qp: min_qp,
            video_max_qp: max_qp,
            ..Default::default()
        };
        let mut enc = Openh264Encoder::new(&s, None).expect("oh264 init");
        (0..FRAMES)
            .map(|i| {
                let y = text_luma(i);
                let mut bgra = vec![255u8; W * H * 4];
                for (px, &l) in bgra.chunks_exact_mut(4).zip(y.iter()) {
                    px[0] = l;
                    px[1] = l;
                    px[2] = l;
                }
                enc.encode_host_argb(&bgra, W * 4, i as u64, i == 0, false)
                    .expect("oh264 encode")
            })
            .collect()
    }

    fn decode_luma(frames: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut dec = Decoder::new().expect("decoder");
        let mut out = Vec::new();
        for f in frames {
            if f.is_empty() {
                continue;
            }
            if let Ok(Some(img)) = dec.decode(f) {
                let (w, h) = img.dimensions();
                let stride = img.strides().0;
                let mut y = vec![0u8; w * h];
                for r in 0..h {
                    y[r * w..r * w + w].copy_from_slice(&img.y()[r * stride..r * stride + w]);
                }
                out.push(y);
            }
        }
        out
    }

    fn mean_psnr(a: &[Vec<u8>], b: &[Vec<u8>]) -> f64 {
        let n = a.len().min(b.len());
        let mut acc = 0.0;
        for i in 0..n {
            let mse: f64 = a[i]
                .iter()
                .zip(b[i].iter())
                .map(|(&x, &y)| {
                    let d = x as f64 - y as f64;
                    d * d
                })
                .sum::<f64>()
                / a[i].len() as f64;
            acc += if mse <= 0.0 { 99.0 } else { 10.0 * (255.0f64 * 255.0 / mse).log10() };
        }
        acc / n.max(1) as f64
    }

    fn kbps(frames: &[Vec<u8>]) -> f64 {
        frames.iter().map(|f| f.len()).sum::<usize>() as f64 * 8.0 * 60.0
            / FRAMES as f64
            / 1000.0
    }

    #[test]
    fn cbr_qp_bound_sweep_diagnostic() {
        // Near-lossless references (per encoder, so CSC differences cancel out).
        let ref_x264 = decode_luma(&encode_x264(false, 0, 12, 0, 0));
        let ref_oh264 = decode_luma(&encode_oh264(false, 0, 12, 0, 0));

        println!("scrolling-text 720p60 @ 2 Mbps CBR (PSNR vs own CRF/QP-12 decode):");
        let mut rows = Vec::new();
        for &max_qp in &[0i32, 45, 40, 35, 30] {
            let x = encode_x264(true, 2000, 25, 0, max_qp);
            let o = encode_oh264(true, 2000, 25, 0, max_qp);
            let px = mean_psnr(&decode_luma(&x), &ref_x264);
            let po = mean_psnr(&decode_luma(&o), &ref_oh264);
            println!(
                "  max_qp {:>2}: x264 {:>8.1} kbps / {:>5.2} dB | oh264 {:>8.1} kbps / {:>5.2} dB",
                max_qp, kbps(&x), px, kbps(&o), po
            );
            rows.push((max_qp, kbps(&x), px, kbps(&o), po));
        }
        // min-QP sweep on an over-provisioned budget: spend must drop, fidelity ~held.
        println!("scrolling-text 720p60 @ 12 Mbps CBR, min-QP sweep:");
        for &min_qp in &[0i32, 10, 15] {
            let x = encode_x264(true, 12000, 25, min_qp, 0);
            let px = mean_psnr(&decode_luma(&x), &ref_x264);
            println!("  min_qp {:>2}: x264 {:>8.1} kbps / {:>5.2} dB", min_qp, kbps(&x), px);
        }

        // The clamp must be plumbed: capping max QP at 30 on starved content must lift
        // fidelity vs the unclamped run (paid for in bitrate) on BOTH encoders.
        let base = &rows[0];
        let capped = rows.last().unwrap();
        assert!(
            capped.2 > base.2 + 0.5,
            "x264 max-QP clamp had no effect: {:.2} vs {:.2} dB",
            capped.2,
            base.2
        );
        assert!(
            capped.4 > base.4 + 0.5,
            "oh264 max-QP clamp had no effect: {:.2} vs {:.2} dB",
            capped.4,
            base.4
        );
    }
}
