//! Unix-socket H.264 fan-out for external recording.
//!
//! A background listener thread binds a Unix domain socket and accepts an arbitrary number of
//! concurrent clients. Each encoded H.264 stripe written through [`RecordingSink`] is broadcast
//! to every connected client as a plain Annex-B elementary stream — the 10-byte wire header
//! (tag, keyframe flag, frame number, y-start, width, height) is stripped so the output is
//! directly muxable.
//!
//! The sink is created once per capture session and shared (via `Arc`) between the encode
//! threads and the delivery layer. When a new client connects, [`RecordingSink::should_force_idr`]
//! returns `true` on the next call, signalling the encoder to emit an IDR so the fresh consumer
//! can start decoding from a clean reference frame.

use std::fs;
use std::io::{ErrorKind, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Write timeout applied to each accepted client stream. Keeps a slow consumer from blocking
/// the encode thread; a stalled write simply drops that client.
const WRITE_TIMEOUT: Duration = Duration::from_millis(100);

/// How often the non-blocking accept loop retries when no client is waiting.
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Environment variable consulted as a fallback when the Python `CaptureSettings.recording_socket`
/// field is empty.
pub const RECORDING_SOCKET_ENV: &str = "PIXELFLUX_RECORDING_SOCKET";

/// Unix-socket fan-out that broadcasts every encoded H.264 frame to connected consumers.
///
/// Internally a listener thread accepts connections on a [`UnixListener`] and appends the
/// resulting streams to a shared `Vec<UnixStream>`. The encode / delivery threads then call
/// [`write_encoded_frame`] to push data to every client; clients that fall behind or disconnect
/// are removed automatically.
///
/// [`write_encoded_frame`]: RecordingSink::write_encoded_frame
pub struct RecordingSink {
    /// Filesystem path of the Unix socket; removed on drop.
    path: String,
    /// Connected client streams, shared between the accept thread and the write path.
    clients: Arc<Mutex<Vec<UnixStream>>>,
    /// Signals the accept thread to exit; set in [`Drop`].
    shutdown: Arc<AtomicBool>,
    /// Flipped to `true` each time a new client connects; consumed by [`should_force_idr`].
    ///
    /// [`should_force_idr`]: RecordingSink::should_force_idr
    client_connected: Arc<AtomicBool>,
}

impl RecordingSink {
    /// Try to bind a Unix socket at `settings_path` (or the `PIXELFLUX_RECORDING_SOCKET`
    /// environment variable when the path is empty).
    ///
    /// Returns `None` when no path is configured or when the bind fails. On success the
    /// listener thread is spawned and the sink is ready to accept clients immediately.
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

    /// Create the socket, spawn the accept thread, and return the sink.
    fn bind(path: String) -> std::io::Result<Self> {
        let _ = fs::remove_file(&path);
        let listener = UnixListener::bind(&path)?;
        listener.set_nonblocking(true)?;

        let clients = Arc::new(Mutex::new(Vec::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let client_connected = Arc::new(AtomicBool::new(false));

        let clients_acc = clients.clone();
        let shutdown_acc = shutdown.clone();
        let client_connected_acc = client_connected.clone();
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
                        client_connected_acc.store(true, Ordering::Relaxed);
                        eprintln!(
                            "[recording_sink] client connected; total {}",
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
            client_connected,
        })
    }

    /// Returns `true` exactly once after a new client connects, signalling that the next
    /// encode should produce an IDR so the consumer starts from a clean reference frame.
    pub fn should_force_idr(&self) -> bool {
        self.client_connected.swap(false, Ordering::Relaxed)
    }

    /// Write `data` to every connected client, silently removing any that are disconnected
    /// or too slow.
    fn write_all_clients(&self, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let mut clients = self.clients.lock().unwrap();
        if clients.is_empty() {
            return;
        }
        let mut to_remove: Vec<usize> = Vec::new();
        for (idx, client) in clients.iter_mut().enumerate() {
            if let Err(e) = client.write_all(data) {
                if matches!(
                    e.kind(),
                    ErrorKind::BrokenPipe | ErrorKind::ConnectionReset | ErrorKind::UnexpectedEof
                ) {
                    to_remove.push(idx);
                } else {
                    eprintln!("[recording_sink] dropping client (idx {}): {:?}", idx, e);
                    to_remove.push(idx);
                }
            }
        }
        for idx in to_remove.into_iter().rev() {
            clients.swap_remove(idx);
        }
    }

    /// Write a single encoded stripe to all connected clients.
    ///
    /// Only H.264 frames (`data_type == 2`) are forwarded. When the frame carries the
    /// pixelflux wire header (`0x04` tag, 10 bytes) the header is stripped so the output
    /// is a plain Annex-B elementary stream suitable for direct muxing.
    pub fn write_encoded_frame(&self, stripe: &crate::encoders::software::EncodedStripe) {
        if stripe.data.is_empty() || stripe.data_type != 2 {
            return;
        }
        let data = &stripe.data;
        if data.len() > 10 && data[0] == 0x04 {
            self.write_all_clients(&data[10..]);
        } else {
            self.write_all_clients(data);
        }
    }
}

impl Drop for RecordingSink {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        let _ = fs::remove_file(&self.path);
    }
}
