//! Optional out-of-band sink for the encoded H.264 elementary stream.
//!
//! When enabled (via `RustCaptureSettings::recording_socket` or the
//! `PIXELFLUX_RECORDING_SOCKET` environment variable), pixelflux binds a Unix
//! domain socket and accepts an arbitrary number of consumers. Each encoder
//! calls [`RecordingSink::write_frame`] once per produced NAL unit; the sink
//! fans the bytes out to every connected client.
//!
//! The wire format is the encoder's native Annex-B H.264 — exactly the bytes
//! that already get concatenated into the Python callback's `output_buf`,
//! minus pixelflux's 2-byte custom framing header. This makes the stream
//! consumable by `ffmpeg -f h264 -i unix:///<path> -c:v copy ...` with no
//! custom parser on the consumer side.
//!
//! ## Lifecycle
//!
//! * `RecordingSink::try_bind` is called at encoder construction. It removes
//!   any stale socket file, binds a non-blocking [`UnixListener`], and spawns
//!   a dedicated accept thread that polls every 50 ms for new clients.
//! * Each accepted client gets a 100 ms write timeout (`SO_SNDTIMEO`). A slow
//!   or hung consumer is closed on the next encode call rather than back-
//!   pressuring pixelflux's encode loop.
//! * `write_frame` iterates the connected clients under a mutex, attempts a
//!   blocking `write_all` (bounded by the timeout), and prunes any client
//!   whose write returned an error (`EPIPE`, `ETIMEDOUT`, …).
//! * On `Drop`, the listener thread is signalled to exit and the socket file
//!   is removed.
//!
//! ## Mid-stream joiners
//!
//! A client that connects mid-session sees arbitrary frames before the next
//! IDR ("keyframe"). Standard H.264 decoders skip until the next IDR finds
//! them; ffmpeg handles this transparently. Forcing an IDR on every new
//! connection is left as a future enhancement — kept out of scope here to
//! keep the patch minimal.

use std::fs;
use std::io::{ErrorKind, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Per-client write timeout. A slow consumer is closed rather than blocking
/// the encoder thread.
const WRITE_TIMEOUT: Duration = Duration::from_millis(100);

/// Cadence at which the accept thread polls for new clients.
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Environment variable consulted as a fallback when
/// `RustCaptureSettings::recording_socket` is empty.
pub const RECORDING_SOCKET_ENV: &str = "PIXELFLUX_RECORDING_SOCKET";

/// Default keyframe cadence — emit an IDR every N encoded frames whenever
/// the sink is active. At pixelflux's typical 30 fps this places an IDR
/// every ~2 s, which lets a mid-stream consumer start decoding within that
/// bound and lets `ffmpeg -f segment -segment_time 2` cut clean segments.
/// Pixelflux's bare encoder uses `KEYINT_MAX_INFINITE` for WebRTC latency;
/// the sink overrides this only when recording is in use.
const DEFAULT_KEYINT_FRAMES: u32 = 60;

pub struct RecordingSink {
    path: String,
    clients: Arc<Mutex<Vec<UnixStream>>>,
    shutdown: Arc<AtomicBool>,
    /// Frames since the last forced IDR. Touched by two threads:
    /// the accept loop resets it to `u32::MAX` whenever a new client
    /// arrives (so the next encode is a keyframe for the new consumer),
    /// and the encoder's `should_force_idr` increments / resets to 0
    /// after each forced keyframe. Periodic IDRs every `keyint_frames`
    /// frames cover the steady-state segment-rotation case.
    frames_since_idr: Arc<AtomicU32>,
    keyint_frames: u32,
}

impl RecordingSink {
    /// Resolve the configured socket path and try to bind it.
    ///
    /// Precedence: `settings_path` (if non-empty) > `PIXELFLUX_RECORDING_SOCKET`
    /// env var (if set and non-empty) > `None` (recording disabled).
    ///
    /// Bind failures are logged but never panic — pixelflux's main pipeline
    /// continues unaffected.
    pub fn try_bind(settings_path: &str) -> Option<Arc<Self>> {
        let path = if !settings_path.is_empty() {
            settings_path.to_string()
        } else {
            match std::env::var(RECORDING_SOCKET_ENV) {
                Ok(p) if !p.is_empty() => p,
                _ => return None,
            }
        };

        match Self::bind(path) {
            Ok(sink) => Some(Arc::new(sink)),
            Err(e) => {
                eprintln!("[recording_sink] bind failed: {:?}", e);
                None
            }
        }
    }

    fn bind(path: String) -> std::io::Result<Self> {
        // Stale socket files can survive an unclean shutdown of a previous
        // pixelflux process; remove unconditionally before bind.
        let _ = fs::remove_file(&path);

        let listener = UnixListener::bind(&path)?;
        listener.set_nonblocking(true)?;

        let clients: Arc<Mutex<Vec<UnixStream>>> = Arc::new(Mutex::new(Vec::new()));
        let shutdown = Arc::new(AtomicBool::new(false));

        // The accept loop and the encode loop both touch this counter:
        // accept resets it to u32::MAX so the next encode emits an IDR
        // (giving the new client a clean decoding entry point); the
        // encoder's should_force_idr increments it and resets to 0 after
        // each forced keyframe. Wrapped in Arc so both threads share it.
        let frames_since_idr = Arc::new(AtomicU32::new(u32::MAX));

        let clients_acc = clients.clone();
        let shutdown_acc = shutdown.clone();
        let frames_since_idr_acc = frames_since_idr.clone();
        let path_log = path.clone();
        thread::spawn(move || {
            eprintln!("[recording_sink] listening on {}", path_log);
            while !shutdown_acc.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        if let Err(e) = stream.set_write_timeout(Some(WRITE_TIMEOUT)) {
                            eprintln!("[recording_sink] set_write_timeout failed: {:?}", e);
                            continue;
                        }
                        let mut guard = clients_acc.lock().unwrap();
                        guard.push(stream);
                        // Trigger an IDR on the encoder's next call so the
                        // newly-attached consumer has SPS/PPS + a keyframe
                        // to start decoding from.
                        frames_since_idr_acc.store(u32::MAX, Ordering::Relaxed);
                        eprintln!(
                            "[recording_sink] client connected; total {}; requesting IDR",
                            guard.len()
                        );
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => {
                        thread::sleep(ACCEPT_POLL_INTERVAL);
                    }
                    Err(e) => {
                        eprintln!("[recording_sink] accept error: {:?}", e);
                        thread::sleep(Duration::from_millis(500));
                    }
                }
            }
            eprintln!("[recording_sink] listener thread exiting");
        });

        Ok(Self {
            path,
            clients,
            shutdown,
            frames_since_idr,
            keyint_frames: DEFAULT_KEYINT_FRAMES,
        })
    }

    /// Advisory hook for encoders: returns `true` when the next encode
    /// should be an IDR (keyframe with SPS+PPS). Encoders should OR this
    /// into their own `force_idr` flag before calling into x264/VAAPI/NVENC.
    ///
    /// Bumps the internal frame counter each call. The very first call
    /// after sink construction returns `true`; subsequent calls return
    /// `true` every `keyint_frames` increments. This is the periodic-IDR
    /// behaviour pixelflux's bare encoder doesn't provide on its own (its
    /// keyint is `INFINITE` for WebRTC latency).
    pub fn should_force_idr(&self) -> bool {
        // Saturating add avoids wrap-around weirdness in the rare case the
        // counter is already u32::MAX from initialisation.
        let prev = self.frames_since_idr.fetch_add(1, Ordering::Relaxed);
        if prev >= self.keyint_frames - 1 {
            self.frames_since_idr.store(0, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Fan a chunk of encoded bytes out to every connected client.
    ///
    /// `data` is expected to be raw Annex-B H.264 — typically a single NAL
    /// unit's `p_payload` (software encoder) or the locked NVENC bitstream
    /// segment (NVENC encoder), or the corresponding VAAPI output. The
    /// boundary between calls does not need to align with H.264 frame
    /// boundaries; ffmpeg's H.264 demuxer parses start codes natively.
    pub fn write_frame(&self, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let mut clients = self.clients.lock().unwrap();
        let mut to_remove: Vec<usize> = Vec::new();
        for (idx, client) in clients.iter_mut().enumerate() {
            if let Err(e) = client.write_all(data) {
                eprintln!("[recording_sink] dropping client (idx {}): {:?}", idx, e);
                to_remove.push(idx);
            }
        }
        for idx in to_remove.into_iter().rev() {
            clients.swap_remove(idx);
        }
    }
}

impl Drop for RecordingSink {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        let _ = fs::remove_file(&self.path);
    }
}
