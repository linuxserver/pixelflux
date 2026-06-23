/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Source-agnostic frame-processing logic shared by the Wayland (dmabuf /
//! compositor-damage) and X11 (host-ARGB / stripe-hash-damage) capture paths, so
//! both get identical paint-over and recovery-keyframe behavior.

use crate::encoders::nvenc::NvencEncoder;
use crate::encoders::oh264::Openh264Encoder;
use crate::encoders::software::{encode_cpu, EncodedStripe, StripeState};
use crate::encoders::vaapi::VaapiEncoder;
use crate::recording_sink::RecordingSink;
use crate::RustCaptureSettings;
use std::sync::Arc;

/// Outcome of the full-frame H.264 (NVENC/VAAPI) send decision.
pub struct HwFrameDecision {
    pub send: bool,
    pub force_idr: bool,
    pub target_qp: u32,
}

/// True when an optional periodic keyframe is due this tick. `keyframe_interval_s
/// <= 0` (the default) keeps the GOP infinite — IDRs happen only on demand, which holds
/// bitrate and quality steady instead of spiking on a wall-clock schedule. A positive
/// interval restores scheduled recovery keyframes every ~N seconds for consumers that
/// cannot request one.
pub fn periodic_idr_due(settings: &RustCaptureSettings, frame_counter: u16) -> bool {
    let secs = settings.keyframe_interval_s;
    if secs <= 0.0 {
        return false;
    }
    let safe_fps = settings.target_fps.max(1.0);
    let interval = ((safe_fps * secs).round() as u64).max(1);
    (frame_counter as u64).is_multiple_of(interval)
}

/// Decide whether to emit a full-frame H.264 picture this frame, at what QP, and
/// whether to force an IDR -- advancing the paint-over bookkeeping in `st`.
///
/// `is_dirty` is the motion signal (compositor damage on Wayland, a stripe-hash
/// change on X11); `is_animated` forces a send for animated overlays; `requested_idr`
/// is an on-demand keyframe request. The GOP is infinite by default: IDRs are emitted
/// for an explicit request (client join/reset, recording cadence) or the optional
/// configured periodic interval — every encoder already keyframes its genuinely-first
/// frame on its own.
pub fn decide_hw_fullframe(
    st: &mut StripeState,
    settings: &RustCaptureSettings,
    frame_counter: u16,
    is_dirty: bool,
    is_animated: bool,
    requested_idr: bool,
) -> HwFrameDecision {
    let normal_qp = settings.video_crf as u32;
    let paint_qp = settings.video_paintover_crf as u32;
    let trigger_frames = settings.paint_over_trigger_frames;
    let use_paint_over = settings.use_paint_over_quality;
    let burst = settings.video_paintover_burst_frames;
    let streaming = settings.video_streaming_mode;
    // Burst/recovery frames use the paint-over QP when it is enabled and actually better,
    // otherwise the base QP (a recovery burst still needs to stream so CBR can refine).
    let burst_qp = if use_paint_over && paint_qp < normal_qp { paint_qp } else { normal_qp };

    let mut send_frame = false;
    let mut force_idr = false;
    let mut target_qp = normal_qp;

    if st.h264_burst_frames_remaining > 0 {
        send_frame = true;
        target_qp = burst_qp;
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

    let recovery_idr = requested_idr || periodic_idr_due(settings, frame_counter);

    if is_dirty {
        // Real motion: full reset of paint-over bookkeeping (the screen changed).
        send_frame = true;
        force_idr = recovery_idr;
        st.no_motion_frame_count = 0;
        st.paint_over_sent = false;
        st.h264_burst_frames_remaining = 0;
        target_qp = normal_qp;
    } else if recovery_idr {
        // Recovery keyframe on a STATIC screen (a joining/reset/resuming client needs a
        // decode entry point). The keyframe is base-quality -- and CBR rate control crashes
        // it further -- and a damage-gated static stream sends nothing afterward to refine
        // it (turbo/streaming mode never hits this because it always sends). Arm a burst so
        // the encoder keeps streaming briefly and recovers. Leave no_motion_frame_count
        // alone so a periodic recovery can't restart the paint-over countdown and starve it.
        send_frame = true;
        force_idr = true;
        if st.h264_burst_frames_remaining <= 0 {
            target_qp = normal_qp;
            if burst > 0 {
                st.paint_over_sent = true;
                st.h264_burst_frames_remaining = burst;
            }
        }
    } else if !send_frame {
        st.no_motion_frame_count += 1;

        if use_paint_over
            && st.no_motion_frame_count >= trigger_frames
            && !st.paint_over_sent
            && paint_qp < normal_qp
        {
            // Quality refresh via low-QP P-frames only: the burst refines the static
            // image against the existing reference chain, so no IDR (and its bitrate
            // spike) is needed. Encoders keep intra-MB choice where residuals demand it.
            send_frame = true;
            st.paint_over_sent = true;
            target_qp = paint_qp;
            st.h264_burst_frames_remaining = burst - 1;
        }
    }

    HwFrameDecision { send: send_frame, force_idr, target_qp }
}

/// Hardware encoder bound to the X11 (host-ARGB) pipeline. Software JPEG/x264 needs no
/// persistent encoder object (encode_cpu owns per-stripe x264 state), so it is `None`.
#[allow(clippy::large_enum_variant)]
enum X11Encoder {
    None,
    Nvenc(NvencEncoder),
    Vaapi(VaapiEncoder),
    Openh264(Openh264Encoder),
}

/// Persistent per-capture context for the X11 host-ARGB path. The caller hands a borrowed ARGB
/// frame to `process()` each tick. There is no compositor here, so damage comes from whole-frame
/// or per-stripe content hashing; full-frame H.264 goes through `decide_hw_fullframe`, while
/// striped JPEG/x264 goes through `encode_cpu` with `hash_damage=true`.
pub struct X11Pipeline {
    settings: RustCaptureSettings,
    stripes: Vec<StripeState>,
    hw: X11Encoder,
    hw_state: StripeState,
    frame_counter: u16,
    pending_force_idr: bool,
    // Optional Unix-socket H.264 fan-out (parity with the Wayland path); None unless
    // recording_socket is set. HW encoders write to it internally; the CPU/OpenH264 paths
    // are fed from process().
    recording_sink: Option<Arc<RecordingSink>>,
}

impl X11Pipeline {
    /// Build the context, picking the hardware encoder by the selected device's
    /// kernel driver (NVIDIA -> NVENC, other GPUs -> VA-API; encode_node_index auto
    /// = ID 0, the first GPU). EGL is unused on the CPU-ARGB path, so a null
    /// display is passed. Falls back to software on HW init failure.
    ///
    /// `recording_sink` is bound once per capture and OWNED BY THE CALLER: the pipeline is
    /// rebuilt on auto-adjust resizes, and re-binding the sink there would tear down the
    /// socket listener and disconnect attached recorders mid-recording.
    pub fn new(settings: RustCaptureSettings, recording_sink: Option<Arc<RecordingSink>>) -> Self {
        let hw = if settings.output_mode == 1 && settings.use_openh264 {
            // Explicit opt-in to the OpenH264 software encoder (full-frame, like the HW path).
            match Openh264Encoder::new(&settings, recording_sink.clone()) {
                Some(e) => X11Encoder::Openh264(e),
                None => {
                    eprintln!("[x11] OpenH264 init failed; falling back to software x264");
                    X11Encoder::None
                }
            }
        } else if settings.output_mode == 1 && !settings.use_cpu && settings.encode_node_index != -1 {
            // Hardware-first on the selected device (encode_node_index: auto resolves to
            // ID 0, the first GPU; -1 above = explicit software): an NVIDIA driver (or no
            // detectable GPU — the attempt is cheap and falls back) goes NVENC, any other
            // GPU driver goes VA-API.
            let encode_driver =
                crate::get_gpu_driver(settings.encode_node_index.max(0));
            if !encode_driver.is_empty() && !encode_driver.contains("nvidia") {
                if settings.video_fullcolor {
                    // VAAPI has no reliable 4:4:4 H.264 profile; the x264 software path does
                    // (high444), so defer full-color to it instead of a silent CPU fallback.
                    eprintln!("[x11] 4:4:4 full-color requested; VAAPI lacks it, using software x264");
                    X11Encoder::None
                } else {
                    // VAAPI: upload host ARGB to a VAAPI surface, VA-VPP converts to NV12 on the GPU.
                    match VaapiEncoder::new_host(&settings, recording_sink.clone()) {
                        Ok(e) => X11Encoder::Vaapi(e),
                        Err(err) => {
                            eprintln!("[x11] VAAPI init failed ({err}); falling back to software");
                            X11Encoder::None
                        }
                    }
                }
            } else {
                match NvencEncoder::new(&settings, std::ptr::null(), recording_sink.clone()) {
                    Ok(e) => X11Encoder::Nvenc(e),
                    Err(err) => {
                        eprintln!("[x11] NVENC init failed ({err}); falling back to software");
                        X11Encoder::None
                    }
                }
            }
        } else {
            X11Encoder::None
        };
        Self {
            settings,
            stripes: Vec::new(),
            hw,
            hw_state: StripeState::default(),
            frame_counter: 0,
            pending_force_idr: false,
            recording_sink,
        }
    }

    /// Request an on-demand keyframe on the next processed frame.
    pub fn request_idr(&mut self) {
        self.pending_force_idr = true;
    }

    /// Adapt the live pipeline to recreated capture surfaces (and possibly a new size)
    /// without rebuilding it. Returns false when the active encoder cannot follow in
    /// place (VAAPI/OpenH264 resizes, or a rejected NVENC reconfigure) and the caller
    /// must rebuild. `settings` carries the new geometry plus the current live rates.
    pub fn reshape(&mut self, settings: &RustCaptureSettings, size_changed: bool) -> bool {
        if !size_changed {
            // Same dimensions, new shm segments: only NVENC keys state to the old
            // base addresses (its pinned-host cache); everything else hashes content.
            if let X11Encoder::Nvenc(enc) = &mut self.hw {
                enc.release_pinned_hosts();
            }
            self.settings = settings.clone();
            return true;
        }
        match &mut self.hw {
            X11Encoder::Nvenc(enc) => {
                if let Err(e) = enc.reconfigure_resolution(settings) {
                    eprintln!("[x11] NVENC in-place resize unavailable ({e}); rebuilding");
                    return false;
                }
            }
            // The striped x264/JPEG path re-derives per-stripe state from the settings on
            // the next process(); clearing it below is the whole resize.
            X11Encoder::None => {}
            // VAAPI surfaces and OpenH264 sessions are fixed-size; rebuild.
            _ => return false,
        }
        self.settings = settings.clone();
        self.stripes.clear();
        self.hw_state = StripeState::default();
        true
    }

    /// Apply a runtime rate-control / framerate change: the CBR target bitrate + VBV (kbps /
    /// kb; ignored unless CBR is active) and the target fps. NVENC and OpenH264 reconfigure their
    /// live session immediately; VAAPI re-opens its codec context to apply the new rate; the x264
    /// software path picks the new values up on the next `process()` (encode_cpu reads the updated
    /// settings and reconfigures each stripe).
    pub fn update_rate(&mut self, bitrate_kbps: i32, vbv_multiplier: f64, fps: f64) {
        self.settings.video_bitrate_kbps = bitrate_kbps;
        self.settings.video_vbv_multiplier = vbv_multiplier;
        if fps > 0.0 {
            self.settings.target_fps = fps;
        }
        match &mut self.hw {
            X11Encoder::Nvenc(enc) => enc.reconfigure_rate(&self.settings),
            X11Encoder::Openh264(enc) => enc.reconfigure_rate(bitrate_kbps, fps),
            X11Encoder::Vaapi(enc) => enc.reconfigure_rate(&self.settings),
            _ => {}
        }
    }

    /// Apply live per-frame tunables (quality, paint-over, streaming mode, keyframe
    /// cadence); every encoder re-reads them from the settings on the next process().
    pub fn update_tunables(&mut self, t: &crate::LiveTunables) {
        t.apply_to(&mut self.settings);
    }

    /// Encode one host-ARGB frame (B,G,R,A order; `stride` bytes per row) and return the
    /// encoded stripes (empty when nothing changed). Borrows `argb` for the call only.
    pub fn process(&mut self, argb: &[u8], stride: usize) -> Vec<EncodedStripe> {
        let width = self.settings.width;
        let height = self.settings.height;
        let requested = self.pending_force_idr;
        let threshold = self.settings.damage_block_threshold;
        let duration = self.settings.damage_block_duration as i32;

        let out = if !matches!(self.hw, X11Encoder::None) {
            // Full-frame H.264 on a HW encoder (NVENC or VAAPI). Frame-level damage via
            // whole-frame content hashing; streaming mode sends every frame, so the content
            // hash is unused there — skip it.
            let is_dirty = if self.settings.video_streaming_mode {
                false
            } else {
                self.hw_state.content_dirty(argb, threshold, duration)
            };
            let d = decide_hw_fullframe(
                &mut self.hw_state,
                &self.settings,
                self.frame_counter,
                is_dirty,
                false,
                requested,
            );
            if d.send {
                let fc = self.frame_counter as u64;
                // Recording sink forces an IDR on connect / every N frames so a late recorder
                // starts on a keyframe (parity with the Wayland path).
                let force_idr = d.force_idr
                    || self.recording_sink.as_ref().map(|s| s.should_force_idr()).unwrap_or(false);
                // Identical decision for both HW encoders; only the submission differs: NVENC
                // encodes ARGB directly, VAAPI uploads + VA-VPP converts to NV12 on the GPU.
                let res = match &mut self.hw {
                    X11Encoder::Nvenc(enc) => {
                        enc.encode_cpu_argb(argb, stride, fc, d.target_qp, force_idr)
                    }
                    X11Encoder::Vaapi(enc) => {
                        enc.encode_host_argb(argb, stride, fc, d.target_qp, force_idr)
                    }
                    // OpenH264 is bitrate-controlled, so the paint-over QP is not applied here.
                    // X11 host pixels are BGRA (rgba_input=false).
                    X11Encoder::Openh264(enc) => {
                        enc.encode_host_argb(argb, stride, fc, force_idr, false)
                    }
                    X11Encoder::None => unreachable!(),
                };
                match res {
                    Ok(data) if !data.is_empty() => {
                        vec![EncodedStripe {
                            data,
                            data_type: 2,
                            stripe_y_start: 0,
                            stripe_height: height,
                            frame_id: self.frame_counter as i32,
                        }]
                    }
                    Ok(_) => Vec::new(),
                    Err(e) => {
                        eprintln!("[x11] HW encode error: {e}");
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            }
        } else {
            // Invariant: the software path (encode_cpu / content_dirty) indexes rows at width*4;
            // it does NOT thread the producer stride through. This holds for every current
            // producer (X11 XShm BGRA and the Wayland readback both deliver tightly-packed
            // width*4 rows). A future producer with a padded stride would need encode_cpu taught
            // to honor `stride` -- the HW paths above already pass it through.
            debug_assert_eq!(
                stride,
                width as usize * 4,
                "software encode path assumes tightly-packed rows (stride == width*4)"
            );
            // Infinite GOP by default: stripes IDR on demand (or on the optional
            // configured interval). Each stripe encoder keyframes its own first frame
            // at (re)init, so no first-frame force is needed here.
            let force_idr_all = self.settings.output_mode == 1
                && (requested || periodic_idr_due(&self.settings, self.frame_counter));
            encode_cpu(
                &mut self.stripes,
                argb,
                width,
                height,
                &[],
                &self.settings,
                self.frame_counter,
                false,
                true,
                self.recording_sink.as_ref(),
                force_idr_all,
            )
        };

        self.pending_force_idr = false;
        self.frame_counter = self.frame_counter.wrapping_add(1);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x11_software_emits_on_change_and_stays_quiet_when_static() {
        let s = RustCaptureSettings {
            width: 128,
            height: 128,
            output_mode: 0, // JPEG
            use_cpu: true, // force the software path (no NVENC in the test)
            jpeg_quality: 60,
            use_paint_over_quality: false, // keep the assertion about static frames clean
            ..Default::default()
        };
        let mut p = X11Pipeline::new(s, None);
        let stride = 128 * 4;
        let frame_a = vec![10u8; stride * 128];
        let mut frame_b = frame_a.clone();
        for px in frame_b.iter_mut().take(stride * 40) {
            *px = 200; // change the top rows so some stripes differ
        }
        let n1 = p.process(&frame_a, stride).len();
        let n2 = p.process(&frame_a, stride).len();
        let n3 = p.process(&frame_b, stride).len();
        assert!(n1 > 0, "first frame should emit (all stripes dirty vs init)");
        assert_eq!(n2, 0, "identical static frame should emit nothing");
        assert!(n3 > 0, "changed frame should emit dirty stripes");
    }

    #[test]
    fn x11_software_h264_streams_recovery_burst_after_requested_idr() {
        // Software x264 (the reported repro), paint-over OFF, non-turbo: a requested IDR on a
        // static screen must be followed by a short recovery burst so rate control can refine
        // the keyframe, instead of the stream going silent and stranding a crashed keyframe.
        let s = RustCaptureSettings {
            width: 128,
            height: 128,
            output_mode: 1, // H.264
            use_cpu: true,  // software x264 path (hw = None)
            video_crf: 25,
            video_paintover_burst_frames: 5,
            use_paint_over_quality: false,
            video_streaming_mode: false,
            target_fps: 60.0,
            ..Default::default()
        };
        let mut p = X11Pipeline::new(s, None);
        let stride = 128 * 4;
        let frame = vec![10u8; stride * 128];
        assert!(!p.process(&frame, stride).is_empty(), "first frame emits");
        for _ in 0..4 {
            let _ = p.process(&frame, stride);
        }
        assert!(p.process(&frame, stride).is_empty(), "static screen is quiet before the request");
        // Resume: request a keyframe. The IDR frame plus the next `burst` static frames all emit.
        p.request_idr();
        assert!(!p.process(&frame, stride).is_empty(), "requested IDR emits on a static screen");
        for i in 0..5 {
            assert!(!p.process(&frame, stride).is_empty(), "recovery burst frame {i} streams while static");
        }
        assert!(p.process(&frame, stride).is_empty(), "stream goes quiet again after the recovery burst");
    }

    fn settings() -> RustCaptureSettings {
        RustCaptureSettings {
            video_crf: 25,
            video_paintover_crf: 18,
            paint_over_trigger_frames: 3,
            use_paint_over_quality: true,
            video_paintover_burst_frames: 5,
            video_streaming_mode: false,
            target_fps: 60.0,
            ..Default::default()
        }
    }

    #[test]
    fn static_frames_stay_silent_without_request() {
        let mut s = settings();
        s.use_paint_over_quality = false;
        let mut st = StripeState::default();
        // No motion, no request -> nothing is emitted (infinite GOP, no scheduled IDR),
        // regardless of where the frame counter sits.
        for fc in [0u16, 1, 120, 240] {
            let d = decide_hw_fullframe(&mut st, &s, fc, false, false, false);
            assert!(!d.send && !d.force_idr, "frame {fc} should stay idle");
        }
    }

    #[test]
    fn requested_idr_forces_send_even_on_static() {
        let s = settings();
        let mut st = StripeState::default();
        // Resume/join: requested IDR on a static screen -> keyframe at the base QP...
        let d = decide_hw_fullframe(&mut st, &s, 5, false, false, true);
        assert!(d.send && d.force_idr);
        assert_eq!(d.target_qp, 25);
        // ...and a recovery burst is armed so the static screen refines afterward instead of
        // staying stuck at the (CBR-crashed) keyframe.
        assert_eq!(st.h264_burst_frames_remaining, 5);
        // The next static frame streams a burst frame at the paint QP (no new keyframe).
        let d = decide_hw_fullframe(&mut st, &s, 6, false, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 18);
        assert_eq!(st.h264_burst_frames_remaining, 4);
        // Motion aborts the burst and resets to base QP.
        let d = decide_hw_fullframe(&mut st, &s, 7, true, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 25);
        assert_eq!(st.h264_burst_frames_remaining, 0);
    }

    #[test]
    fn requested_idr_recovers_even_without_paint_over() {
        // The reported CBR case: paint-over OFF still needs frames after the forced keyframe
        // so rate control can refine the static image (turbo/streaming mode masks this).
        let mut s = settings();
        s.use_paint_over_quality = false;
        let mut st = StripeState::default();
        let d = decide_hw_fullframe(&mut st, &s, 5, false, false, true);
        assert!(d.send && d.force_idr);
        assert_eq!(st.h264_burst_frames_remaining, 5, "recovery burst armed without paint-over");
        // Recovery frames stream at the base QP (CBR ignores QP; the point is to keep sending).
        let d = decide_hw_fullframe(&mut st, &s, 6, false, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 25);
    }

    #[test]
    fn configured_interval_restores_scheduled_keyframes() {
        let mut s = settings();
        s.keyframe_interval_s = 2.0; // 120 frames at 60 fps
        assert!(periodic_idr_due(&s, 0));
        assert!(!periodic_idr_due(&s, 1));
        assert!(periodic_idr_due(&s, 120));
        let mut st = StripeState::default();
        let d = decide_hw_fullframe(&mut st, &s, 120, false, false, false);
        assert!(d.send && d.force_idr, "interval keyframe fires on a static screen");
        s.keyframe_interval_s = 0.0;
        assert!(!periodic_idr_due(&s, 0) && !periodic_idr_due(&s, 120));
    }

    #[test]
    fn paint_over_fires_after_trigger_then_bursts() {
        let s = settings();
        let mut st = StripeState::default();
        // Static frames accumulate no-motion count.
        for fc in 1..=2 {
            let d = decide_hw_fullframe(&mut st, &s, fc, false, false, false);
            assert!(!d.send, "frame {fc} should stay idle");
        }
        // 3rd static frame hits trigger -> paint-over at paint QP as a refining P frame
        // (no IDR spike), burst armed.
        let d = decide_hw_fullframe(&mut st, &s, 3, false, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 18);
        assert_eq!(st.h264_burst_frames_remaining, 4);
        // 4th frame: burst continues at paint QP, still no forced IDR.
        let d = decide_hw_fullframe(&mut st, &s, 4, false, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 18);
        assert_eq!(st.h264_burst_frames_remaining, 3);
    }

    #[test]
    fn motion_resets_paintover_and_uses_normal_qp() {
        let s = settings();
        let mut st = StripeState {
            paint_over_sent: true,
            h264_burst_frames_remaining: 2,
            no_motion_frame_count: 9,
            ..Default::default()
        };
        let d = decide_hw_fullframe(&mut st, &s, 7, true, false, false);
        assert!(d.send);
        assert_eq!(d.target_qp, 25);
        assert!(!st.paint_over_sent);
        assert_eq!(st.h264_burst_frames_remaining, 0);
        assert_eq!(st.no_motion_frame_count, 0);
    }
}

#[cfg(test)]
mod vbv_tests {
    #[test]
    fn vbv_policy() {
        use crate::encoders::vbv_bits;
        let frame = 4_000_000f64 / 60.0;
        // Infinite GOP default: 1.5 frames.
        assert_eq!(vbv_bits(4_000_000, 60.0, 0.0, 0.0), (frame * 1.5).round() as u32);
        // Scheduled keyframes: relaxed to 3 frames.
        assert_eq!(vbv_bits(4_000_000, 60.0, 2.0, 0.0), (frame * 3.0).round() as u32);
        // Explicit multiplier wins and rescales with bitrate.
        assert_eq!(vbv_bits(4_000_000, 60.0, 2.0, 1.0), frame.round() as u32);
        assert_eq!(vbv_bits(8_000_000, 60.0, 0.0, 1.0), (2.0 * frame).round() as u32);
    }
}
