/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Software H.264 via Cisco OpenH264 (BSD-licensed). A full-frame, 4:2:0-only,
//! Annex-B alternative to the x264 stripe path, selected with `use_openh264`.
//!
//! Keyframes are driven externally (an effectively infinite intra period plus
//! on-demand `force_intra_frame`), matching the NVENC/x264 strict-GOP streaming
//! behavior. Host ARGB is converted to I420 with the same BT.709 path the x264
//! encoder uses, then fed to OpenH264 as borrowed planes.

use crate::recording_sink::RecordingSink;
use std::sync::Arc;
use crate::RustCaptureSettings;
use openh264::encoder::{
    BitRate, Complexity, Encoder, EncoderConfig, FrameRate, FrameType, IntraFramePeriod, QpRange,
    RateControlMode, UsageType,
};
use openh264::formats::YUVSlices;
use openh264::OpenH264API;

/// Large enough that OpenH264 never inserts its own periodic IDR within a session;
/// recovery keyframes are driven by the pipeline's force-IDR decision instead.
const INFINITE_INTRA_PERIOD: u32 = 300_000;

/// Oversized bitrate ceiling (well under level 5.2's cap): CRF's rate budget, and the max
/// bitrate live CBR changes lift the session to -- init pins the max to the starting target,
/// which would otherwise reject any raise above it.
const BITRATE_CEILING_BPS: u32 = 100_000_000;

pub struct Openh264Encoder {
    encoder: Encoder,
    width: usize,
    height: usize,
    y_buf: Vec<u8>,
    u_buf: Vec<u8>,
    v_buf: Vec<u8>,
    current_bitrate_bps: i32,
    // true = CBR (RC_BITRATE_MODE, target bitrate); false = CRF/CQP (RC_BITRATE_MODE, pinned QP).
    is_cbr: bool,
    omit_stripe_headers: bool,
    recording_sink: Option<Arc<RecordingSink>>,
}

impl Openh264Encoder {
    /// Build an OpenH264 encoder from the capture settings. Dimensions are rounded
    /// down to even values (4:2:0 requires it). Returns None on init failure so the
    /// caller can fall back to the x264 software path. Like the NVENC/VAAPI encoders,
    /// it writes raw Annex-B to `recording_sink` itself and self-prepends the wire header.
    pub fn new(settings: &RustCaptureSettings, recording_sink: Option<Arc<RecordingSink>>) -> Option<Self> {
        if settings.video_fullcolor {
            // Surfaced like the VAAPI limitation; NVENC/x264 honor full color instead.
            eprintln!("[openh264] 4:4:4 full-color requested; OpenH264 is 4:2:0-only, encoding 4:2:0.");
        }
        let width = (settings.width.max(2) as usize) & !1;
        let height = (settings.height.max(2) as usize) & !1;
        let bps = (settings.video_bitrate_kbps.max(1) as u32).saturating_mul(1000);
        let fps = if settings.target_fps < 1.0 { 30.0 } else { settings.target_fps as f32 };
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).clamp(1, 4))
            .unwrap_or(1) as u16;
        let is_cbr = settings.video_cbr_mode;
        // Floor 1, not 0: any zero in the QP pair makes openh264's ParamValidation discard
        // the WHOLE range and substitute its screen-content defaults [26,35]; explicit
        // non-zero values are honored (the library clips the min up to its own floor, 12).
        let crf = settings.video_crf.clamp(1, 51) as u8;

        let base = EncoderConfig::new()
            .max_frame_rate(FrameRate::from_hz(fps))
            .usage_type(UsageType::ScreenContentRealTime)
            .complexity(Complexity::Low)
            // Frame skip ON so OpenH264 actually holds the target bitrate — skip-less bitrate
            // control is only approximate (the encoder warns). A skipped frame isn't encoded and
            // the next P-frame references the last ENCODED frame, so the reference chain is intact.
            .skip_frames(true)
            // Auto-disabled for screen content; set explicitly to avoid the init warnings.
            .adaptive_quantization(false)
            .background_detection(false)
            // Scene-change detection stays ON: OpenH264's ParamValidation force-enables
            // it for ScreenContentRealTime (setting false only produces a warning), so
            // its content-adaptive IDRs are inherent to this encoder; the wire header
            // labels them correctly.
            // Internal stderr trace follows the capture's debug flag (else WELS_LOG_QUIET).
            .debug(settings.debug_logging)
            .intra_frame_period(IntraFramePeriod::from_num_frames(INFINITE_INTRA_PERIOD))
            .num_threads(threads);
        let config = if is_cbr {
            // CBR: RC_BITRATE_MODE with the QP range made EXPLICIT — the crate's (0,51)
            // default trips the same zero-check substitution as above, capping the RC at
            // QP 35 where hard content overshoots any target; (1,51) gives it the full
            // swing. Frame-skip stays off (selkies owns congestion), so targets below
            // the QP-51 floor remain best-effort — openh264 itself warns that skip-less
            // bitrate control is approximate.
            let min_qp = settings.video_min_qp.clamp(1, 51) as u8;
            let max_qp = if settings.video_max_qp > 0 {
                settings.video_max_qp.clamp(min_qp as i32, 51) as u8
            } else {
                51
            };
            base.bitrate(BitRate::from_bps(bps))
                .rate_control_mode(RateControlMode::Bitrate)
                .qp(QpRange::new(min_qp, max_qp))
        } else {
            // CRF (CQP): pin the QP to a single value (min==max==video_crf) under bitrate-mode RC
            // with a bitrate budget so large the QP clamp always dominates, so every frame uses the
            // constant QP = video_crf, matching NVENC CONSTQP / VAAPI CQP. (Empirically RC_OFF ignores
            // the QP range and RC_QUALITY rejects a min==max range; bitrate-mode + pinned QP is what
            // actually holds the QP constant.)
            base.bitrate(BitRate::from_bps(BITRATE_CEILING_BPS))
                .rate_control_mode(RateControlMode::Bitrate)
                .qp(QpRange::new(crf, crf))
        };

        let encoder = Encoder::with_api_config(OpenH264API::from_source(), config).ok()?;

        let (cw, ch) = (width / 2, height / 2);
        let mut me = Self {
            encoder,
            width,
            height,
            y_buf: vec![0u8; width * height],
            u_buf: vec![0u8; cw * ch],
            v_buf: vec![0u8; cw * ch],
            current_bitrate_bps: bps as i32,
            is_cbr,
            omit_stripe_headers: settings.omit_stripe_headers,
            recording_sink: None,
        };
        me.enable_four_slices();
        me.recording_sink = recording_sink;
        Some(me)
    }

    /// Re-init with four fixed slices per frame: encoder threads can parallelize and
    /// client decoders slice-parallelize (more than four upsets some Chromium
    /// decoders). The crate initializes the underlying encoder lazily on the FIRST
    /// encode and only exposes single/size-limited slicing, so prime it with one
    /// throwaway frame, patch the then-live parameter set, and re-init. Best-effort:
    /// any failure leaves a working single-slice encoder.
    fn enable_four_slices(&mut self) {
        let slices = YUVSlices::new(
            (&self.y_buf, &self.u_buf, &self.v_buf),
            (self.width, self.height),
            (self.width, self.width / 2, self.width / 2),
        );
        if self.encoder.encode(&slices).is_err() {
            return;
        }
        unsafe {
            let mut params: openh264_sys2::SEncParamExt = std::mem::zeroed();
            let raw = self.encoder.raw_api();
            if raw.get_option(
                openh264_sys2::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                &mut params as *mut _ as *mut std::ffi::c_void,
            ) != 0
            {
                return;
            }
            params.sSpatialLayers[0].sSliceArgument.uiSliceMode =
                openh264_sys2::SM_FIXEDSLCNUM_SLICE;
            params.sSpatialLayers[0].sSliceArgument.uiSliceNum = 4;
            let ret = raw.set_option(
                openh264_sys2::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                &mut params as *mut _ as *mut std::ffi::c_void,
            );
            if ret != 0 {
                eprintln!("[openh264] 4-slice re-init rejected ({ret}); staying single-slice");
            }
        }
        // The throwaway frame consumed the initial IDR; the first delivered frame
        // must still be one.
        self.encoder.force_intra_frame();
    }

    /// Apply a live bitrate (kbps) / framerate change. OpenH264 honors these via SetOption
    /// without re-initializing, so the reference chain (and infinite GOP) is preserved -- this
    /// is what lets the web UI's bitrate slider take effect mid-stream, like NVENC and x264.
    pub fn reconfigure_rate(&mut self, bitrate_kbps: i32, fps: f64) {
        // Bitrate applies only in CBR mode; in CRF/CQP the QP is fixed at construction (a CRF
        // change restarts the capture, rebuilding the encoder — there is no live-QP setter here).
        if self.is_cbr {
            let bps = bitrate_kbps.max(1).saturating_mul(1000);
            if bps != self.current_bitrate_bps {
                self.set_live_bitrate(bps);
            }
        }
        if fps >= 1.0 {
            let mut rate = fps as f32;
            unsafe {
                self.encoder.raw_api().set_option(
                    openh264_sys2::ENCODER_OPTION_FRAME_RATE,
                    std::ptr::addr_of_mut!(rate).cast(),
                );
            }
        }
    }

    /// Move the CBR target to `bps`. Init pins OpenH264's max bitrate to the construction-time
    /// target (EncoderConfig has no separate max), and SetOption rejects -- after partially
    /// applying -- any target above the max, so lift the ceiling first (overall and layer 0;
    /// verification checks the per-layer value). Failures keep or restore the previous rate.
    fn set_live_bitrate(&mut self, bps: i32) {
        for layer in [openh264_sys2::SPATIAL_LAYER_ALL, openh264_sys2::SPATIAL_LAYER_0] {
            let ret = self.set_bitrate_option(
                openh264_sys2::ENCODER_OPTION_MAX_BITRATE,
                layer,
                BITRATE_CEILING_BPS as i32,
            );
            if ret != 0 {
                eprintln!(
                    "[openh264] live bitrate change to {bps} bps failed raising the max-bitrate ceiling \
                     (code {ret}); keeping {} bps",
                    self.current_bitrate_bps
                );
                return;
            }
        }
        let ret =
            self.set_bitrate_option(openh264_sys2::ENCODER_OPTION_BITRATE, openh264_sys2::SPATIAL_LAYER_ALL, bps);
        if ret == 0 {
            self.current_bitrate_bps = bps;
        } else {
            // A failed SetOption already mutated the target; re-apply the accepted rate so the
            // encoder stays in sync with current_bitrate_bps.
            self.set_bitrate_option(
                openh264_sys2::ENCODER_OPTION_BITRATE,
                openh264_sys2::SPATIAL_LAYER_ALL,
                self.current_bitrate_bps,
            );
            eprintln!(
                "[openh264] live bitrate change to {bps} bps failed (code {ret}); keeping {} bps",
                self.current_bitrate_bps
            );
        }
    }

    /// One SBitrateInfo-shaped SetOption call (BITRATE / MAX_BITRATE); returns 0 on success.
    fn set_bitrate_option(
        &mut self,
        option: openh264_sys2::ENCODER_OPTION,
        layer: openh264_sys2::LAYER_NUM,
        bps: i32,
    ) -> i32 {
        let mut info: openh264_sys2::SBitrateInfo = unsafe { std::mem::zeroed() };
        info.iLayer = layer;
        info.iBitrate = bps;
        unsafe { self.encoder.raw_api().set_option(option, std::ptr::addr_of_mut!(info).cast()) }
    }

    /// Encode one host frame to H.264. `stride` is bytes per row. `rgba_input`
    /// selects the byte order: false = B,G,R,A (X11 XShm), true = R,G,B,A (Wayland GL
    /// readback). When `force_idr` is set, the frame is emitted as an IDR.
    /// Output is the 10-byte wire header + Annex-B (bare Annex-B when
    /// `omit_stripe_headers`); raw Annex-B is also fed to the recording sink.
    pub fn encode_host_argb(
        &mut self,
        argb: &[u8],
        stride: usize,
        frame_number: u64,
        force_idr: bool,
        rgba_input: bool,
    ) -> Result<Vec<u8>, String> {
        if force_idr {
            self.encoder.force_intra_frame();
        }
        let cw = self.width / 2;
        crate::encoders::software::convert_to_yuv_mt(
            argb,
            stride as u32,
            self.width,
            self.height,
            rgba_input,
            false,
            &mut self.y_buf,
            &mut self.u_buf,
            &mut self.v_buf,
            4,
        )
        .map_err(|e| format!("rgb-to-yuv420 failed: {e:?}"))?;

        let slices = YUVSlices::new(
            (&self.y_buf, &self.u_buf, &self.v_buf),
            (self.width, self.height),
            (self.width, cw, cw),
        );
        match self.encoder.encode(&slices) {
            Ok(bitstream) => {
                let header_sz = if self.omit_stripe_headers { 0 } else { 10 };
                let mut out = Vec::with_capacity(header_sz);
                if header_sz != 0 {
                    // Same header as the NVENC/VAAPI/x264 full-frame paths; type byte from the
                    // ACTUAL encoded picture type (IDR=0x01, I=0x02, P=0x00).
                    let type_hdr = match bitstream.frame_type() {
                        FrameType::IDR => 0x01u8,
                        FrameType::I => 0x02u8,
                        _ => 0x00u8,
                    };
                    out.push(0x04);
                    out.push(type_hdr);
                    out.extend_from_slice(&(frame_number as u16).to_be_bytes());
                    out.extend_from_slice(&0u16.to_be_bytes());
                    out.extend_from_slice(&(self.width as u16).to_be_bytes());
                    out.extend_from_slice(&(self.height as u16).to_be_bytes());
                }
                bitstream.write_vec(&mut out);
                if out.len() == header_sz {
                    // No payload (e.g. a skipped frame): emit nothing rather than a bare header.
                    return Ok(Vec::new());
                }
                if let Some(ref sink) = self.recording_sink {
                    sink.write_frame(&out[header_sz..]);
                }
                Ok(out)
            }
            Err(e) => Err(format!("openh264 encode failed: {e:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A busy, motion-carrying frame so the rate controller produces non-trivial P-frames.
    // High-entropy (per-pixel) content: fine under CBR (the RC raises QP to fit) but pathological
    // for a pinned low QP, so the CRF test uses the compressible gradient_frame instead.
    fn busy_frame(w: usize, h: usize, t: usize) -> Vec<u8> {
        let mut f = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let v = (((x + t) * 7 + (y + t) * 13) & 0xFF) as u8;
                f[i] = v;
                f[i + 1] = v.wrapping_mul(3);
                f[i + 2] = v.wrapping_add((x + t) as u8);
                f[i + 3] = 255;
            }
        }
        f
    }

    // A smooth diagonal gradient (moderate, realistic screen-like detail): compressible enough not
    // to overflow at a low pinned QP, yet detailed enough that the QP visibly scales output size.
    fn gradient_frame(w: usize, h: usize, t: usize) -> Vec<u8> {
        let mut f = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let v = ((x * 3 + y * 5 + t * 4) & 0xFF) as u8;
                f[i] = v;
                f[i + 1] = v.wrapping_add(40);
                f[i + 2] = v.wrapping_add(80);
                f[i + 3] = 255;
            }
        }
        f
    }

    #[test]
    fn encodes_annexb_idr_then_p() {
        let s = RustCaptureSettings {
            width: 128,
            height: 96,
            video_bitrate_kbps: 2000,
            target_fps: 30.0,
            ..Default::default()
        };
        let mut enc = Openh264Encoder::new(&s, None).expect("openh264 init");
        let stride = 128 * 4;
        let idr = enc.encode_host_argb(&busy_frame(128, 96, 0), stride, 0, true, false).expect("encode idr");
        assert!(idr.len() > 10, "IDR frame should produce output");
        // Wire header: 0x04, type, frame_id u16, y_start u16, width u16, height u16.
        assert_eq!(idr[0], 0x04);
        assert_eq!(idr[1], 0x01, "forced first frame must be typed IDR");
        assert_eq!(&idr[2..6], &[0, 0, 0, 0], "frame_id 0 and y_start 0");
        assert_eq!(&idr[6..10], &[0, 128, 0, 96], "width/height big-endian");
        assert!(
            idr[10..].starts_with(&[0, 0, 0, 1]) || idr[10..].starts_with(&[0, 0, 1]),
            "payload must be Annex-B (start code prefixed)"
        );
        // Re-encode the same content: no scene change, so this must be a P frame.
        let p = enc.encode_host_argb(&busy_frame(128, 96, 0), stride, 7, false, false).expect("encode p");
        assert!(p.len() > 10, "second frame should produce output");
        assert_eq!(p[1], 0x00, "unforced static second frame must be typed delta");
        assert_eq!(&p[2..4], &[0, 7], "frame_id must come from frame_number");
    }

    #[test]
    fn omit_stripe_headers_yields_bare_annexb() {
        let s = RustCaptureSettings {
            width: 128,
            height: 96,
            video_bitrate_kbps: 2000,
            target_fps: 30.0,
            omit_stripe_headers: true,
            ..Default::default()
        };
        let mut enc = Openh264Encoder::new(&s, None).expect("openh264 init");
        let out = enc.encode_host_argb(&busy_frame(128, 96, 0), 128 * 4, 0, true, false).expect("encode");
        assert!(
            out.starts_with(&[0, 0, 0, 1]) || out.starts_with(&[0, 0, 1]),
            "omit_stripe_headers output must be bare Annex-B"
        );
    }

    #[test]
    fn lower_bitrate_yields_smaller_output() {
        let (w, h, stride) = (256usize, 192usize, 256 * 4);
        let encode_run = |kbps: i32| -> usize {
            let s = RustCaptureSettings {
                width: w as i32,
                height: h as i32,
                video_cbr_mode: true, // bitrate only controls output in CBR mode
                video_bitrate_kbps: kbps,
                target_fps: 30.0,
                ..Default::default()
            };
            let mut e = Openh264Encoder::new(&s, None).unwrap();
            let _ = e.encode_host_argb(&busy_frame(w, h, 0), stride, 0, true, false).unwrap();
            (1..24).map(|t| e.encode_host_argb(&busy_frame(w, h, t), stride, t as u64, false, false).unwrap().len()).sum()
        };
        let high = encode_run(8000);
        let low = encode_run(200);
        assert!(low < high, "lower target bitrate should compress harder (low={low}, high={high})");
    }

    #[test]
    fn live_reconfigure_rate_takes_effect() {
        let (w, h, stride) = (256usize, 192usize, 256 * 4);
        let s = RustCaptureSettings {
            width: w as i32,
            height: h as i32,
            video_cbr_mode: true, // live bitrate only applies in CBR mode
            video_bitrate_kbps: 8000,
            target_fps: 30.0,
            ..Default::default()
        };
        let mut e = Openh264Encoder::new(&s, None).unwrap();
        let _ = e.encode_host_argb(&busy_frame(w, h, 0), stride, 0, true, false).unwrap();
        let high: usize =
            (1..24).map(|t| e.encode_host_argb(&busy_frame(w, h, t), stride, t as u64, false, false).unwrap().len()).sum();
        // Drop the bitrate live (as the web UI slider does) and re-measure.
        e.reconfigure_rate(200, 30.0);
        let low: usize =
            (24..48).map(|t| e.encode_host_argb(&busy_frame(w, h, t), stride, t as u64, false, false).unwrap().len()).sum();
        assert!(low < high, "live bitrate drop should shrink output (low={low}, high={high})");
    }

    #[test]
    fn live_reconfigure_rate_raise_takes_effect() {
        // Raising above the session's initial bitrate requires lifting OpenH264's max-bitrate
        // ceiling, which init pins to the starting target.
        let (w, h, stride) = (256usize, 192usize, 256 * 4);
        let s = RustCaptureSettings {
            width: w as i32,
            height: h as i32,
            video_cbr_mode: true, // live bitrate only applies in CBR mode
            video_bitrate_kbps: 300,
            target_fps: 30.0,
            ..Default::default()
        };
        let mut e = Openh264Encoder::new(&s, None).unwrap();
        let _ = e.encode_host_argb(&busy_frame(w, h, 0), stride, 0, true, false).unwrap();
        let low: usize =
            (1..24).map(|t| e.encode_host_argb(&busy_frame(w, h, t), stride, t as u64, false, false).unwrap().len()).sum();
        // Raise the bitrate live, far above the session's initial value.
        e.reconfigure_rate(8000, 30.0);
        assert_eq!(e.current_bitrate_bps, 8_000_000, "live raise must be accepted, not rejected");
        let high: usize =
            (24..48).map(|t| e.encode_host_argb(&busy_frame(w, h, t), stride, t as u64, false, false).unwrap().len()).sum();
        assert!(high > low * 3 / 2, "live bitrate raise should grow output (low={low}, high={high})");
    }

    #[test]
    fn crf_mode_qp_scales_output() {
        // CRF/CQP mode (RC_OFF + QpRange): a higher video_crf (higher QP, stronger compression)
        // must produce SMALLER output than a lower CRF, confirming the constant QP is applied.
        let (w, h, stride) = (256usize, 192usize, 256 * 4);
        let run = |crf: i32| -> usize {
            let s = RustCaptureSettings {
                width: w as i32,
                height: h as i32,
                video_cbr_mode: false, // CRF/CQP mode
                video_crf: crf,
                target_fps: 30.0,
                ..Default::default()
            };
            let mut e = Openh264Encoder::new(&s, None).unwrap();
            // Include the IDR (frame 0), where the constant QP most affects size.
            let mut total = e.encode_host_argb(&gradient_frame(w, h, 0), stride, 0, true, false).unwrap().len();
            total += (1..24)
                .map(|t| e.encode_host_argb(&gradient_frame(w, h, t), stride, t as u64, false, false).unwrap().len())
                .sum::<usize>();
            total
        };
        let high_quality = run(18); // low QP -> larger
        let low_quality = run(40); // high QP -> smaller
        assert!(
            low_quality < high_quality,
            "higher CRF must compress harder (crf40={low_quality}, crf18={high_quality})"
        );
    }

    #[test]
    fn rgba_and_bgra_inputs_both_encode() {
        // The Wayland GLES readback delivers RGBA; X11 delivers BGRA. Both byte orders must
        // encode valid Annex-B (color correctness is exercised end-to-end, not here).
        let (w, h, stride) = (128usize, 96usize, 128 * 4);
        let s = RustCaptureSettings {
            width: w as i32,
            height: h as i32,
            video_bitrate_kbps: 2000,
            target_fps: 30.0,
            ..Default::default()
        };
        let frame = busy_frame(w, h, 0);
        for rgba in [false, true] {
            let mut e = Openh264Encoder::new(&s, None).unwrap();
            let out = e.encode_host_argb(&frame, stride, 0, true, rgba).expect("encode");
            assert!(out.len() > 10, "rgba_input={rgba} must produce output");
            assert_eq!(out[0], 0x04, "rgba_input={rgba} output must carry the wire header");
            assert!(
                out[10..].starts_with(&[0, 0, 0, 1]) || out[10..].starts_with(&[0, 0, 1]),
                "rgba_input={rgba} payload must be Annex-B"
            );
        }
    }
}

#[cfg(test)]
mod slice_tests {
    use super::*;

    #[test]
    fn four_slices_per_frame() {
        let s = RustCaptureSettings {
            width: 640,
            height: 480,
            output_mode: 1,
            video_bitrate_kbps: 2000,
            video_cbr_mode: true,
            ..Default::default()
        };
        let mut enc = Openh264Encoder::new(&s, None).expect("encoder");
        let frame = vec![0x80u8; 640 * 480 * 4];
        let out = enc.encode_host_argb(&frame, 640 * 4, 0, true, false).expect("encode");
        let mut nals = 0;
        let mut i = 0;
        while i + 3 < out.len() {
            if &out[i..i + 3] == b"\x00\x00\x01" {
                nals += 1;
                i += 3;
            } else {
                i += 1;
            }
        }
        // SPS + PPS + four IDR slices.
        assert!(nals >= 6, "expected 4-slice IDR (>=6 NALs), got {nals}");
    }

    // Deterministic per-frame noise: the worst case for RC (incompressible, and screen-mode
    // scene-change detection fires every frame).
    fn noise_frame(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut x = seed.wrapping_mul(2654435761).max(1);
        let mut v = vec![0u8; w * h * 4];
        for px in v.chunks_exact_mut(4) {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            px[0] = x as u8;
            px[1] = (x >> 8) as u8;
            px[2] = (x >> 16) as u8;
            px[3] = 255;
        }
        v
    }

    #[test]
    fn cbr_rc_diagnostic_on_noise() {
        let (w, h) = (1280usize, 720usize);
        let stride = w * 4;
        let run = |cbr: bool, kbps: i32, crf: i32| -> (usize, usize) {
            let s = RustCaptureSettings {
                width: w as i32,
                height: h as i32,
                video_cbr_mode: cbr,
                video_bitrate_kbps: kbps,
                video_crf: crf,
                target_fps: 60.0,
                ..Default::default()
            };
            let mut e = Openh264Encoder::new(&s, None).unwrap();
            let mut total = 0usize;
            let mut idrs = 0usize;
            for t in 0..60u64 {
                let out = e
                    .encode_host_argb(&noise_frame(w, h, t as u32), stride, t, t == 0, false)
                    .unwrap();
                total += out.len();
                // NAL scan for IDR slices (type 5).
                for i in 0..out.len().saturating_sub(4) {
                    if out[i] == 0 && out[i + 1] == 0 && out[i + 2] == 1 && (out[i + 3] & 0x1F) == 5 {
                        idrs += 1;
                        break;
                    }
                }
            }
            (total / 60, idrs)
        };
        let (cbr4m, idr4m) = run(true, 4000, 25);
        let (cbr500k, _) = run(true, 500, 25);
        let (qp51, _) = run(false, 1, 51);
        let (qp25, _) = run(false, 1, 25);
        println!(
            "noise 720p60 bytes/frame: CBR-4Mbps={cbr4m} (IDR frames {idr4m}/60) \
             CBR-500kbps={cbr500k} QP51-floor={qp51} QP25={qp25}"
        );
        // Invariant: the RC must respond monotonically to the CBR target on hard content.
        // A zero anywhere in the configured QP pair makes openh264 substitute its [26,35]
        // screen defaults, under which noise emits the same bytes at any target — the
        // explicit range in `new` is what this guards. The RC still can't reach the QP51
        // floor here: screen mode force-enables scene-change detection (every noise frame
        // becomes an IDR) and RcCalculateIdrQp clamps IDR QP to <=40 (iQpRangeArray)
        // regardless of the session max — an upstream RC ceiling, not a config defect.
        assert!(
            cbr500k * 4 < cbr4m * 3,
            "CBR-on-noise no longer responds to its target: 500kbps={cbr500k} B/f vs 4Mbps={cbr4m} B/f \
             (QP51 floor {qp51}, QP25 {qp25})"
        );
        assert!(qp51 * 4 < qp25, "QP pinning sanity: 51 must compress far harder than 25");
    }

    #[test]
    fn four_slice_reinit_preserves_rc_params() {
        let (w, h) = (1280usize, 720usize);
        let s = RustCaptureSettings {
            width: w as i32,
            height: h as i32,
            video_cbr_mode: true,
            video_bitrate_kbps: 4000,
            target_fps: 60.0,
            ..Default::default()
        };
        let mut e = Openh264Encoder::new(&s, None).unwrap();
        unsafe {
            let mut p: openh264_sys2::SEncParamExt = std::mem::zeroed();
            let ret = e.encoder.raw_api().get_option(
                openh264_sys2::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                &mut p as *mut _ as *mut std::ffi::c_void,
            );
            assert_eq!(ret, 0, "GetOption(SVC_ENCODE_PARAM_EXT)");
            println!(
                "post-init params: iRCMode={} iTargetBitrate={} iMaxBitrate={} fMaxFrameRate={} \
                 iMaxQp={} iMinQp={} bEnableFrameSkip={} uiIntraPeriod={} layer0(target={} max={})",
                p.iRCMode, p.iTargetBitrate, p.iMaxBitrate, p.fMaxFrameRate, p.iMaxQp, p.iMinQp,
                p.bEnableFrameSkip, p.uiIntraPeriod,
                p.sSpatialLayers[0].iSpatialBitrate, p.sSpatialLayers[0].iMaxSpatialBitrate,
            );
            assert_eq!(p.iTargetBitrate, 4_000_000, "target bitrate survives the 4-slice re-init");
            assert!(p.fMaxFrameRate > 1.0, "frame rate survives the 4-slice re-init");
            assert_eq!(p.iMaxQp, 51, "RC QP headroom survives the 4-slice re-init");
            assert!(p.bEnableFrameSkip, "CBR enables frame skip so the bitrate is actually held");
        }
    }
}

#[cfg(test)]
mod rebuild_cost {
    use super::*;

    // The empirical floor for an OpenH264 resolution change: its SetOption
    // (SVC_ENCODE_PARAM_EXT) re-initializes the encoder core internally, so a rebuild
    // costs the same order -- this prints that cost to show it is milliseconds.
    #[test]
    fn construction_cost_is_milliseconds() {
        let s = RustCaptureSettings {
            width: 1920,
            height: 1080,
            output_mode: 1,
            use_openh264: true,
            video_bitrate_kbps: 8000,
            ..Default::default()
        };
        let t = std::time::Instant::now();
        let mut e = Openh264Encoder::new(&s, None).expect("init");
        let init_ms = t.elapsed().as_secs_f64() * 1000.0;
        let frame = vec![128u8; 1920 * 1080 * 4];
        let t = std::time::Instant::now();
        let out = e.encode_host_argb(&frame, 1920 * 4, 0, true, false).expect("encode");
        let first_ms = t.elapsed().as_secs_f64() * 1000.0;
        assert!(!out.is_empty());
        println!("openh264 1080p init={init_ms:.1}ms first_frame={first_ms:.1}ms");
        assert!(init_ms < 100.0, "OpenH264 init unexpectedly slow: {init_ms}ms");
    }
}
