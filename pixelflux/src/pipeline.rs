/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Frame-processing policy shared by the two capture backends. It lives in its own module for one
//! reason: the Wayland path (dmabuf, compositor damage) and the X11 path (host-ARGB, stripe-hash
//! damage) capture pixels in completely different ways, but a viewer must never be able to tell
//! which one produced a frame — a paint-over refresh or a recovery keyframe has to behave
//! identically either way. Keeping the decision logic here, source-agnostic, is what guarantees it.

use crate::encoders::nvenc::NvencEncoder;
use crate::encoders::oh264::Openh264Encoder;
use crate::encoders::software::{encode_cpu, EncodedStripe, StripeState};
use crate::encoders::vaapi::VaapiEncoder;
use crate::recording_sink::RecordingSink;
use crate::RustCaptureSettings;
use std::sync::Arc;

/// @brief Outcome of the full-frame H.264 send decision produced by `decide_hw_fullframe`.
pub struct HwFrameDecision {
    pub send: bool,
    pub force_idr: bool,
    pub target_qp: u32,
}

/// @brief Whether a scheduled keyframe is due this tick — the escape hatch for consumers that
/// cannot ask for one on their own.
///
/// The default (`keyframe_interval_s <= 0`) is an infinite GOP with no scheduled IDRs at all,
/// because a wall-clock keyframe cadence spikes bitrate and dents quality on a screen that isn't
/// changing. A positive interval buys that back as a fixed ~N-second recovery cadence, which is
/// only worth its cost for a consumer that has no channel to request an on-demand keyframe.
pub fn periodic_idr_due(settings: &RustCaptureSettings, frame_counter: u16) -> bool {
    let secs = settings.keyframe_interval_s;
    if secs <= 0.0 {
        return false;
    }
    let safe_fps = settings.target_fps.max(1.0);
    let interval = ((safe_fps * secs).round() as u64).max(1);
    (frame_counter as u64).is_multiple_of(interval)
}

/// @brief The one send / quality / keyframe policy every full-frame H.264 encoder obeys, shaped so
/// a static screen costs almost nothing to stream yet a client that just joined or reset can always
/// recover a clean picture.
///
/// **Why it is shaped this way.** A wall-clock keyframe schedule would waste bitrate and dent
/// quality while nothing on screen moves, so the GOP is left infinite and an IDR is forced only
/// when some consumer genuinely needs a fresh decode entry point. But a single forced keyframe is
/// not enough on its own: under CBR that lone keyframe decodes poorly, and a damage-gated static
/// stream would then send nothing more to sharpen it. So every forced IDR is followed by a short
/// "recovery burst" that keeps streaming until rate control converges. All of this lives in one
/// function precisely so the three encoders (NVENC / VAAPI / OpenH264) cannot drift apart.
///
/// Inputs: `is_dirty` is the motion signal (Wayland compositor damage, or an X11 stripe-hash
/// change); `is_animated` forces a send for animated overlays; `requested_idr` is an on-demand
/// request (client join / reset, recording cadence); and `st` carries the paint-over bookkeeping
/// this function advances. The states are resolved in priority order, each choosing the send flag
/// and QP for the reason noted:
///
/// 1. **Recovery burst in progress** — keep streaming the static screen so CBR can refine the
///    keyframe just sent; real motion aborts the burst and drops back to the base QP. `burst_qp` is
///    the paint-over QP only when that is enabled and actually lower, since the goal here is to
///    sharpen, not just to keep bytes flowing.
/// 2. **Always-on modes** — streaming mode and animated overlays have no "static" state to optimize
///    for, so they send every frame unconditionally.
/// 3. **Motion** — the screen changed, so send at the base QP, force an IDR only if one is due, and
///    reset the paint-over bookkeeping (the still image it was refining is gone).
/// 4. **Recovery keyframe on a static screen** — the join / reset case the whole burst mechanism
///    exists for: force the IDR and arm the burst, deliberately leaving `no_motion_frame_count`
///    untouched so a periodic recovery cannot restart and then starve the paint-over countdown.
/// 5. **Paint-over refresh** — once the screen has sat idle for `paint_over_trigger_frames`, spend a
///    few low-QP P-frames sharpening the still image against the existing reference chain (no IDR,
///    so no bitrate spike), because an idle screen would otherwise stay stuck at whatever quality
///    the last motion happened to leave behind.
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
        send_frame = true;
        force_idr = recovery_idr;
        st.no_motion_frame_count = 0;
        st.paint_over_sent = false;
        st.h264_burst_frames_remaining = 0;
        target_qp = normal_qp;
    } else if recovery_idr {
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
            send_frame = true;
            st.paint_over_sent = true;
            target_qp = paint_qp;
            st.h264_burst_frames_remaining = burst - 1;
        }
    }

    HwFrameDecision { send: send_frame, force_idr, target_qp }
}

/// @brief Hardware encoder bound to the X11 (host-ARGB) pipeline. Software JPEG/x264 needs no
/// persistent encoder object (`encode_cpu` owns per-stripe x264 state), so it is `None`.
#[allow(clippy::large_enum_variant)]
enum X11Encoder {
    None,
    Nvenc(NvencEncoder),
    Vaapi(VaapiEncoder),
    Openh264(Openh264Encoder),
}

/// @brief Everything the X11 host-ARGB path has to remember between frames.
///
/// Unlike the Wayland backend, X11 capture has no compositor to report what changed, so this
/// context exists to hold the state that stands in for that missing damage signal: the per-stripe
/// hashes and the persistent encoder session that let `process()` discover damage by comparing
/// content frame-to-frame. Full-frame H.264 runs through `decide_hw_fullframe`; striped JPEG/x264
/// runs through `encode_cpu` with `hash_damage=true`.
///
/// `recording_sink` is the optional Unix-socket H.264 fan-out (parity with the Wayland path),
/// `None` unless a recording socket is configured. It is owned here rather than created per frame
/// because the hardware encoders write to it internally; the software CPU / OpenH264 paths are fed
/// from `process()`.
pub struct X11Pipeline {
    settings: RustCaptureSettings,
    stripes: Vec<StripeState>,
    hw: X11Encoder,
    hw_state: StripeState,
    frame_counter: u16,
    pending_force_idr: bool,
    recording_sink: Option<Arc<RecordingSink>>,
}

impl X11Pipeline {
    /// @brief Build the context, choosing the full-frame encoder for the X11 host-ARGB path.
    ///
    /// The encoder is selected from the settings and the chosen device's kernel driver:
    ///
    /// - **OpenH264** when H.264 output is selected and `use_openh264` is set: an explicit opt-in
    ///   to the software OpenH264 encoder, which is full-frame like the hardware path.
    /// - **Hardware** when H.264 is selected, software is not forced, and a device is chosen
    ///   (`encode_node_index`: auto resolves to ID 0, the first GPU; `-1` means explicit software).
    ///   An NVIDIA driver — or no detectable GPU, since the attempt is cheap and falls back — goes
    ///   to NVENC; any other GPU driver goes to VA-API. The one exception is a 4:4:4 full-color
    ///   request, which defers to software x264: VA-API has no reliable 4:4:4 H.264 profile while
    ///   x264's high444 does, so this avoids a silent CPU fallback.
    /// - **Software** (`X11Encoder::None`) otherwise, and on any hardware init failure.
    ///
    /// EGL is unused on the CPU-ARGB path, so NVENC is handed a null display. `recording_sink` is
    /// bound once per capture and OWNED BY THE CALLER: the pipeline is rebuilt on auto-adjust
    /// resizes, and re-binding the sink there would tear down the socket listener and disconnect
    /// attached recorders mid-recording.
    pub fn new(settings: RustCaptureSettings, recording_sink: Option<Arc<RecordingSink>>) -> Self {
        let hw = if settings.output_mode == 1 && settings.use_openh264 {
            println!("[x11] OpenH264 software encoder selected.");
            match Openh264Encoder::new(&settings, recording_sink.clone()) {
                Some(e) => X11Encoder::Openh264(e),
                None => {
                    eprintln!("[x11] OpenH264 init failed; falling back to software x264");
                    X11Encoder::None
                }
            }
        } else if settings.output_mode == 1 && !settings.use_cpu && settings.encode_node_index != -1 {
            let encode_driver =
                crate::get_gpu_driver(settings.encode_node_index.max(0));
            println!("[x11] Encode Node Index: {} | Driver: {}", settings.encode_node_index.max(0), encode_driver);
            if !encode_driver.is_empty() && !encode_driver.contains("nvidia") {
                if settings.video_fullcolor {
                    println!("[x11] 4:4:4 full-color requested. VAAPI does not support this profile reliably. Falling back to CPU.");
                    X11Encoder::None
                } else {
                    println!("[x11] Initializing Unified VAAPI Encoder...");
                    match VaapiEncoder::new_host(&settings, recording_sink.clone()) {
                        Ok(e) => {
                            println!("[x11] VAAPI Encoder initialized successfully.");
                            X11Encoder::Vaapi(e)
                        }
                        Err(err) => {
                            eprintln!("[x11] Failed to init VAAPI: {err}. Falling back to CPU.");
                            X11Encoder::None
                        }
                    }
                }
            } else {
                println!("[x11] Nvidia Encoder detected. Initializing NVENC...");
                match NvencEncoder::new(&settings, std::ptr::null(), recording_sink.clone()) {
                    Ok(e) => {
                        println!("[x11] NVENC Encoder initialized successfully.");
                        X11Encoder::Nvenc(e)
                    }
                    Err(err) => {
                        eprintln!("[x11] Failed to init NVENC: {err}. Falling back to CPU.");
                        X11Encoder::None
                    }
                }
            }
        } else {
            if settings.output_mode == 1 {
                println!("[x11] No GPU Encoder available -> Using CPU Software Encoding.");
            }
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

    /// @brief Request an on-demand keyframe on the next processed frame.
    pub fn request_idr(&mut self) {
        self.pending_force_idr = true;
    }

    /// @brief Return a human-readable encoder type string for logging.
    pub fn encoder_name(&self) -> &str {
        match &self.hw {
            X11Encoder::Nvenc(_) => "NVENC",
            X11Encoder::Vaapi(_) => "VAAPI",
            X11Encoder::Openh264(_) => "OpenH264",
            X11Encoder::None => "CPU",
        }
    }

    /// @brief Adapt the live pipeline to recreated capture surfaces — and possibly a new size —
    /// without rebuilding it.
    ///
    /// Returns `false` when the active encoder cannot follow in place and the caller must rebuild;
    /// `settings` carries the new geometry plus the current live rates. Behavior per encoder:
    ///
    /// - **Same dimensions, new shm segments**: only NVENC keys state to the old base addresses
    ///   (its pinned-host cache), so its pinned hosts are released; every other path hashes content
    ///   and needs nothing.
    /// - **New dimensions, NVENC**: reconfigured in place; a rejected reconfigure returns `false`.
    /// - **New dimensions, software x264/JPEG**: the striped path re-derives per-stripe state from
    ///   the settings on the next `process()`, so clearing that state below is the whole resize.
    /// - **New dimensions, VAAPI or OpenH264**: their surfaces / sessions are fixed-size, so this
    ///   returns `false` to force a rebuild.
    pub fn reshape(&mut self, settings: &RustCaptureSettings, size_changed: bool) -> bool {
        if !size_changed {
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
            X11Encoder::None => {}
            _ => return false,
        }
        self.settings = settings.clone();
        self.stripes.clear();
        self.hw_state = StripeState::default();
        true
    }

    /// @brief Apply a runtime rate-control / framerate change: the CBR target bitrate + VBV (kbps /
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

    /// @brief Apply live per-frame tunables (quality, paint-over, streaming mode, keyframe
    /// cadence); every encoder re-reads them from the settings on the next process().
    pub fn update_tunables(&mut self, t: &crate::LiveTunables) {
        t.apply_to(&mut self.settings);
    }

    /// @brief Encode one host-ARGB frame (B,G,R,A order, `stride` bytes per row) and return the
    /// encoded stripes — empty when nothing changed. Borrows `argb` for the call only.
    ///
    /// Two paths, chosen by whether a hardware / OpenH264 encoder is bound:
    ///
    /// - **Full-frame H.264** (`hw` set): frame-level damage comes from whole-frame content
    ///   hashing, except in streaming mode which sends every frame and skips the hash.
    ///   `decide_hw_fullframe` makes the identical send decision for every encoder; only the
    ///   submission differs — NVENC encodes ARGB directly, VAAPI uploads and VA-VPP converts to
    ///   NV12 on the GPU, and OpenH264 (bitrate-controlled, so the paint-over QP does not apply)
    ///   takes BGRA host pixels. A configured recording sink forces an IDR on connect and every N
    ///   frames so a late recorder starts on a keyframe, matching the Wayland path.
    /// - **Striped software JPEG/x264** (`hw` is `None`): `encode_cpu` hashes per stripe. The GOP
    ///   is infinite, so stripes keyframe only on demand or on the optional configured interval;
    ///   each stripe encoder keyframes its own first frame at (re)init, and an explicit request
    ///   also forces a full JPEG resend so a joining viewer receives every stripe.
    ///
    /// The software path assumes tightly-packed `width*4` rows: `encode_cpu` / `content_dirty`
    /// index rows at `width*4` and do not thread the producer `stride` through. Every current
    /// producer satisfies this (X11 XShm BGRA and the Wayland readback both deliver packed rows);
    /// a future padded-stride producer would need `encode_cpu` taught to honor `stride`, which the
    /// hardware paths above already pass through. The `debug_assert` guards that invariant.
    pub fn process(&mut self, argb: &[u8], stride: usize) -> Vec<EncodedStripe> {
        let width = self.settings.width;
        let height = self.settings.height;
        let requested = self.pending_force_idr;
        let threshold = self.settings.damage_block_threshold;
        let duration = self.settings.damage_block_duration as i32;

        let out = if !matches!(self.hw, X11Encoder::None) {
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
                let force_idr = d.force_idr
                    || self.recording_sink.as_ref().map(|s| s.should_force_idr()).unwrap_or(false);
                let res = match &mut self.hw {
                    X11Encoder::Nvenc(enc) => {
                        enc.encode_cpu_argb(argb, stride, fc, d.target_qp, force_idr)
                    }
                    X11Encoder::Vaapi(enc) => {
                        enc.encode_host_argb(argb, stride, fc, d.target_qp, force_idr)
                    }
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
            debug_assert_eq!(
                stride,
                width as usize * 4,
                "software encode path assumes tightly-packed rows (stride == width*4)"
            );
            let force_idr_all = requested
                || (self.settings.output_mode == 1
                    && periodic_idr_due(&self.settings, self.frame_counter));
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

    /// @brief Software JPEG path emits on change and stays silent while static: the first frame
    /// sends every stripe (all dirty vs init), an identical static frame sends nothing, and a
    /// frame with changed top rows re-sends the dirty stripes. `use_cpu` forces the software path
    /// and paint-over is disabled to keep the static-frame assertions clean.
    #[test]
    fn x11_software_emits_on_change_and_stays_quiet_when_static() {
        let s = RustCaptureSettings {
            width: 128,
            height: 128,
            output_mode: 0,
            use_cpu: true,
            jpeg_quality: 60,
            use_paint_over_quality: false,
            ..Default::default()
        };
        let mut p = X11Pipeline::new(s, None);
        let stride = 128 * 4;
        let frame_a = vec![10u8; stride * 128];
        let mut frame_b = frame_a.clone();
        for px in frame_b.iter_mut().take(stride * 40) {
            *px = 200;
        }
        let n1 = p.process(&frame_a, stride).len();
        let n2 = p.process(&frame_a, stride).len();
        let n3 = p.process(&frame_b, stride).len();
        assert!(n1 > 0, "first frame should emit (all stripes dirty vs init)");
        assert_eq!(n2, 0, "identical static frame should emit nothing");
        assert!(n3 > 0, "changed frame should emit dirty stripes");
    }

    /// @brief Software x264, paint-over off, non-streaming: a requested IDR on a static screen is
    /// followed by a short recovery burst so rate control can refine the keyframe, instead of the
    /// stream going silent and stranding an unrefined keyframe. The stream goes quiet again once
    /// the burst ends.
    #[test]
    fn x11_software_h264_streams_recovery_burst_after_requested_idr() {
        let s = RustCaptureSettings {
            width: 128,
            height: 128,
            output_mode: 1,
            use_cpu: true,
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

    /// @brief With no motion and no request, nothing is emitted at any frame-counter position —
    /// the GOP is infinite, so there is no scheduled IDR to break the silence.
    #[test]
    fn static_frames_stay_silent_without_request() {
        let mut s = settings();
        s.use_paint_over_quality = false;
        let mut st = StripeState::default();
        for fc in [0u16, 1, 120, 240] {
            let d = decide_hw_fullframe(&mut st, &s, fc, false, false, false);
            assert!(!d.send && !d.force_idr, "frame {fc} should stay idle");
        }
    }

    /// @brief A requested IDR on a static screen (client resume / join) emits a base-QP keyframe
    /// and arms a recovery burst; subsequent static frames stream burst frames at the paint QP
    /// with no new keyframe, and real motion aborts the burst and reverts to the base QP.
    #[test]
    fn requested_idr_forces_send_even_on_static() {
        let s = settings();
        let mut st = StripeState::default();
        let d = decide_hw_fullframe(&mut st, &s, 5, false, false, true);
        assert!(d.send && d.force_idr);
        assert_eq!(d.target_qp, 25);
        assert_eq!(st.h264_burst_frames_remaining, 5);
        let d = decide_hw_fullframe(&mut st, &s, 6, false, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 18);
        assert_eq!(st.h264_burst_frames_remaining, 4);
        let d = decide_hw_fullframe(&mut st, &s, 7, true, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 25);
        assert_eq!(st.h264_burst_frames_remaining, 0);
    }

    /// @brief With paint-over off, a forced keyframe still needs following frames so CBR rate
    /// control can refine the static image (streaming mode would mask this by always sending);
    /// the recovery burst supplies them at the base QP.
    #[test]
    fn requested_idr_recovers_even_without_paint_over() {
        let mut s = settings();
        s.use_paint_over_quality = false;
        let mut st = StripeState::default();
        let d = decide_hw_fullframe(&mut st, &s, 5, false, false, true);
        assert!(d.send && d.force_idr);
        assert_eq!(st.h264_burst_frames_remaining, 5, "recovery burst armed without paint-over");
        let d = decide_hw_fullframe(&mut st, &s, 6, false, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 25);
    }

    #[test]
    fn configured_interval_restores_scheduled_keyframes() {
        let mut s = settings();
        s.keyframe_interval_s = 2.0;
        assert!(periodic_idr_due(&s, 0));
        assert!(!periodic_idr_due(&s, 1));
        assert!(periodic_idr_due(&s, 120));
        let mut st = StripeState::default();
        let d = decide_hw_fullframe(&mut st, &s, 120, false, false, false);
        assert!(d.send && d.force_idr, "interval keyframe fires on a static screen");
        s.keyframe_interval_s = 0.0;
        assert!(!periodic_idr_due(&s, 0) && !periodic_idr_due(&s, 120));
    }

    /// @brief After `paint_over_trigger_frames` idle frames, paint-over fires as a refining
    /// P-frame at the paint QP (no IDR spike) and arms a burst; the next frame continues the burst
    /// at the paint QP, still without a forced IDR.
    #[test]
    fn paint_over_fires_after_trigger_then_bursts() {
        let s = settings();
        let mut st = StripeState::default();
        for fc in 1..=2 {
            let d = decide_hw_fullframe(&mut st, &s, fc, false, false, false);
            assert!(!d.send, "frame {fc} should stay idle");
        }
        let d = decide_hw_fullframe(&mut st, &s, 3, false, false, false);
        assert!(d.send && !d.force_idr);
        assert_eq!(d.target_qp, 18);
        assert_eq!(st.h264_burst_frames_remaining, 4);
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
    /// @brief VBV sizing policy: an infinite GOP uses 1.5 frames of headroom, scheduled keyframes
    /// relax to 3 frames, and an explicit multiplier overrides both and rescales with bitrate.
    #[test]
    fn vbv_policy() {
        use crate::encoders::vbv_bits;
        let frame = 4_000_000f64 / 60.0;
        assert_eq!(vbv_bits(4_000_000, 60.0, 0.0, 0.0), (frame * 1.5).round() as u32);
        assert_eq!(vbv_bits(4_000_000, 60.0, 2.0, 0.0), (frame * 3.0).round() as u32);
        assert_eq!(vbv_bits(4_000_000, 60.0, 2.0, 1.0), frame.round() as u32);
        assert_eq!(vbv_bits(8_000_000, 60.0, 0.0, 1.0), (2.0 * frame).round() as u32);
    }
}
