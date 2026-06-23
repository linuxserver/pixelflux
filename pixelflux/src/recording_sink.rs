/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

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
/// Defines socket connection timeouts, polling intervals, keyframe cadence, and
/// the primary RecordingSink structure used to multiplex the elementary stream
/// to connected Unix socket clients.
const WRITE_TIMEOUT: Duration = Duration::from_millis(100);
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(50);
const DEFAULT_KEYINT_FRAMES: u32 = 60;

/// Per-client buffered-frame cap; a client that exceeds it is dropped as too slow.
const CLIENT_QUEUE_CAP: usize = 256;

/// A connected client; bytes are drained to its socket by a dedicated thread.
struct ClientHandle {
    tx: Sender<Arc<Vec<u8>>>,
    /// Signals the writer thread to stop promptly (without draining its backlog)
    /// when the client is dropped from the sink.
    stop: Arc<AtomicBool>,
}

pub struct RecordingSink {
    path: String,
    clients: Arc<Mutex<Vec<ClientHandle>>>,
    shutdown: Arc<AtomicBool>,
    frames_since_idr: Arc<AtomicU32>,
    keyint_frames: u32,
}

impl RecordingSink {
    /// @brief Binds the configured socket path, if one is set.
    ///
    /// @input settings_path: The configured path for the socket ("" = recording off).
    /// @return Option containing the new RecordingSink instance.
    pub fn try_bind(settings_path: &str) -> Option<Arc<Self>> {
        if settings_path.is_empty() {
            return None;
        }
        match Self::bind(settings_path.to_string()) {
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

                        // Writer thread owns the stream; exits when tx drops, the
                        // stop flag is set, or a hard write error occurs.
                        let (tx, rx) = bounded::<Arc<Vec<u8>>>(CLIENT_QUEUE_CAP);
                        let stop = Arc::new(AtomicBool::new(false));
                        let stop_writer = stop.clone();
                        thread::spawn(move || {
                            let mut stream = stream;
                            for buf in rx.iter() {
                                if stop_writer.load(Ordering::Relaxed) {
                                    break;
                                }
                                if let Err(e) = write_all_frame(&mut stream, &buf, &stop_writer) {
                                    // A soft timeout/would-block is retried inside
                                    // write_all_frame; reaching here means a hard
                                    // error (or a requested stop). Surface the
                                    // concrete reason and drop the client.
                                    eprintln!(
                                        "[recording_sink] writer thread exiting; write failed: {:?}",
                                        e
                                    );
                                    break;
                                }
                            }
                        });

                        let mut guard = clients_acc.lock().unwrap();
                        guard.push(ClientHandle { tx, stop });

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

        let mut clients = self.clients.lock().unwrap();
        // Skip the per-frame heap alloc + copy entirely when there is no recorder
        // attached (the normal optional-tap case where the socket is configured
        // but unconnected).
        if clients.is_empty() {
            return;
        }

        // One copy, shared across clients via Arc.
        let buf = Arc::new(data.to_vec());

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
            // Signal the writer to stop promptly rather than draining its 256-frame
            // backlog before noticing the dropped tx.
            let removed = clients.swap_remove(idx);
            removed.stop.store(true, Ordering::Relaxed);
        }
    }
}

impl Drop for RecordingSink {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Drop every client immediately so their writer threads/FDs are reclaimed
        // now instead of via the accept thread's ~50 ms poll. Signal stop first so
        // a writer mid-frame exits without draining its backlog.
        if let Ok(mut clients) = self.clients.lock() {
            for client in clients.iter() {
                client.stop.store(true, Ordering::Relaxed);
            }
            clients.clear();
        }
        let _ = fs::remove_file(&self.path);
    }
}

/// Writes a full frame to the client, resuming from the last written offset on a
/// soft timeout rather than dropping the connection mid-frame.
///
/// `set_write_timeout` makes `write` return `TimedOut`/`WouldBlock` after pushing
/// only some leading bytes of a frame; bailing out there would emit a truncated
/// Annex-B NAL to a merely-slow-but-healthy reader. Instead we retry the unwritten
/// remainder until the whole frame lands or a hard error occurs. A permanently
/// stuck client is bounded elsewhere: its queue fills, `write_frame` drops it from
/// the client list, and `stop` is set — which this loop observes between retries.
fn write_all_frame<W: Write>(stream: &mut W, buf: &[u8], stop: &AtomicBool) -> std::io::Result<()> {
    let mut written = 0usize;
    while written < buf.len() {
        if stop.load(Ordering::Relaxed) {
            return Err(std::io::Error::other(
                "writer stopped (client dropped)",
            ));
        }
        match stream.write(&buf[written..]) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    ErrorKind::WriteZero,
                    "failed to write whole frame",
                ));
            }
            Ok(n) => written += n,
            // Soft errors: the partial write already advanced `written`, so retry the
            // remainder. The blocking write timeout paces the loop; add a small sleep
            // for the (non-default) non-blocking WouldBlock case to avoid a hot spin.
            Err(ref e) if e.kind() == ErrorKind::TimedOut => {}
            Err(ref e) if e.kind() == ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
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
