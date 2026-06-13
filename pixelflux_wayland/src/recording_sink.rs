use std::fs;
use std::io::{ErrorKind, Write};
use std::os::unix::net::UnixListener;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{bounded, Sender, TrySendError};

/// Core settings and state for the out-of-band H.264 recording sink.
///
/// Defines socket connection timeouts, polling intervals, environment fallbacks,
/// keyframe cadence, and the primary RecordingSink structure used to multiplex
/// the elementary stream to connected Unix socket clients.
const WRITE_TIMEOUT: Duration = Duration::from_millis(100);
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(50);
pub const RECORDING_SOCKET_ENV: &str = "PIXELFLUX_RECORDING_SOCKET";
const DEFAULT_KEYINT_FRAMES: u32 = 60;

/// Per-client buffered-frame cap; a client that exceeds it is dropped as too slow.
const CLIENT_QUEUE_CAP: usize = 256;

/// A connected client; bytes are drained to its socket by a dedicated thread.
struct ClientHandle {
    tx: Sender<Arc<Vec<u8>>>,
}

pub struct RecordingSink {
    path: String,
    clients: Arc<Mutex<Vec<ClientHandle>>>,
    shutdown: Arc<AtomicBool>,
    frames_since_idr: Arc<AtomicU32>,
    keyint_frames: u32,
}

impl RecordingSink {
    /// @brief Resolves the configured socket path and tries to bind it.
    ///
    /// @input settings_path: The configured path for the socket.
    /// @return Option containing the new RecordingSink instance.
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

    /// @brief Binds the Unix listener and spawns the accept thread.
    ///
    /// @input path: The file path to bind the socket to.
    /// @return Result containing the new RecordingSink instance.
    fn bind(path: String) -> std::io::Result<Self> {
        let _ = fs::remove_file(&path);

        let listener = UnixListener::bind(&path)?;
        listener.set_nonblocking(true)?;

        let clients: Arc<Mutex<Vec<ClientHandle>>> = Arc::new(Mutex::new(Vec::new()));
        let shutdown = Arc::new(AtomicBool::new(false));

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

                        // Writer thread owns the stream; exits when tx drops or a write fails.
                        let (tx, rx) = bounded::<Arc<Vec<u8>>>(CLIENT_QUEUE_CAP);
                        thread::spawn(move || {
                            let mut stream = stream;
                            for buf in rx.iter() {
                                if stream.write_all(&buf).is_err() {
                                    break;
                                }
                            }
                        });

                        let mut guard = clients_acc.lock().unwrap();
                        guard.push(ClientHandle { tx });

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

    /// @brief Advisory hook indicating if the next encode should be an IDR.
    ///
    /// @return True if the next frame should be a keyframe.
    pub fn should_force_idr(&self) -> bool {
        let prev = self.frames_since_idr.fetch_add(1, Ordering::Relaxed);
        if idr_due(prev, self.keyint_frames) {
            self.frames_since_idr.store(0, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// @brief Fans encoded bytes out to every client without blocking the caller.
    ///
    /// Bytes go to each client's bounded queue; a full (too slow) or disconnected
    /// client is dropped.
    ///
    /// @input data: The raw Annex-B H.264 byte slice.
    pub fn write_frame(&self, data: &[u8]) {
        if data.is_empty() {
            return;
        }

        // One copy, shared across clients via Arc.
        let buf = Arc::new(data.to_vec());

        let mut clients = self.clients.lock().unwrap();
        let mut to_remove: Vec<usize> = Vec::new();

        for (idx, client) in clients.iter().enumerate() {
            match client.tx.try_send(buf.clone()) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    eprintln!("[recording_sink] dropping slow client (idx {})", idx);
                    to_remove.push(idx);
                }
                Err(TrySendError::Disconnected(_)) => {
                    eprintln!("[recording_sink] dropping disconnected client (idx {})", idx);
                    to_remove.push(idx);
                }
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

/// True when a keyframe is due. saturating_sub avoids underflow if keyint is 0.
fn idr_due(frames_since_idr: u32, keyint_frames: u32) -> bool {
    frames_since_idr >= keyint_frames.saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::idr_due;

    #[test]
    fn first_frame_forces_idr() {
        // frames_since_idr is initialized to u32::MAX, so the first call is due.
        assert!(idr_due(u32::MAX, 60));
    }

    #[test]
    fn cadence_matches_keyint() {
        let keyint = 60;
        assert!(!idr_due(0, keyint));
        assert!(!idr_due(58, keyint));
        assert!(idr_due(59, keyint));
        assert!(idr_due(60, keyint));
    }

    #[test]
    fn keyint_zero_and_one_do_not_underflow() {
        assert!(idr_due(0, 0));
        assert!(idr_due(u32::MAX, 0));
        assert!(idr_due(0, 1));
        assert!(idr_due(1, 1));
    }
}
