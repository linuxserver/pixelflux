/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Out-of-band H.264 recording sink. Its reason to exist is separation: a
//! recorder needs the exact encoded stream the viewers see, but neither side
//! should be able to disturb the other — so this fans the encoder's raw Annex-B
//! elementary stream to clients on a private Unix domain socket, entirely apart
//! from the live viewer transport (WebSocket / WebRTC). A recorder attaching,
//! stalling, or detaching cannot perturb what viewers receive, and recording
//! works even with no viewer connected at all. Recording is also strictly
//! optional: with no socket path configured, `RecordingSink::try_bind` returns
//! `None` and there is no sink for the caller to write frames to.

use std::fs;
use std::io::{ErrorKind, Write};
use std::os::unix::net::UnixListener;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{bounded, Sender, TrySendError};

/// @brief Bounds how long a single `write` to a recorder's socket may block, so a
/// stalled reader cannot wedge its writer thread indefinitely. Without it, a
/// recorder that stops draining would leave `write` blocked forever, and the
/// writer could never again check its `stop` flag or make progress; the timeout
/// instead surfaces as a soft `TimedOut`/`WouldBlock` that `write_all_frame`
/// retries, keeping the thread responsive to teardown between attempts while a
/// merely-slow-but-healthy reader still gets its frames.
const WRITE_TIMEOUT: Duration = Duration::from_millis(100);

/// @brief Throttles the accept thread's poll loop, trading a little connection
/// pickup latency for not burning a CPU on empty polls. The listener is
/// deliberately nonblocking so the thread can observe `shutdown` between attempts
/// instead of parking forever inside `accept()`; the price of that choice is that
/// a bare `accept()` returns at once when nothing is pending, so the thread sleeps
/// this long before trying again. The same interval also caps how long teardown
/// waits for the thread to notice `shutdown` and exit.
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// @brief Bounds how long the recorded stream can run without a decode entry
/// point — the reason a periodic keyframe cadence exists at all. An infinite GOP
/// would leave a saved file decodable only from its very first frame and
/// unrecoverable after any gap, so `should_force_idr` requests a fresh IDR every
/// this-many frames to keep the recording seekable and self-healing. A newly
/// connected client short-circuits the same counter to force one immediately (see
/// `bind`), so a mid-stream joiner need not wait out the cadence.
const DEFAULT_KEYINT_FRAMES: u32 = 60;

/// @brief Bounds each recorder's backlog so one slow reader can neither grow
/// memory without limit nor push back on the shared encode thread. Fan-out is a
/// non-blocking `try_send` into this per-client queue; a reader that falls this
/// far behind is judged unable to keep up and dropped, sacrificing that one
/// recording to protect the live pipeline and every other client. The depth is
/// the jitter tolerance — deep enough to ride out a brief hiccup, shallow enough
/// to cap the memory a doomed client can waste.
const CLIENT_QUEUE_CAP: usize = 256;

/// @brief The sink's handle to one connected recorder — deliberately just the
/// feed end and a kill switch, never the socket itself. Giving each client its
/// own bounded channel and a dedicated writer thread that alone owns the stream
/// is what keeps one slow reader from stalling the fan-out or its peers; the sink
/// therefore holds only what it needs, `tx` to enqueue frames and `stop` to tear
/// the writer down.
struct ClientHandle {
    tx: Sender<Arc<Vec<u8>>>,
    /// Signals the writer thread to exit promptly rather than draining
    /// whatever frames are still queued, once the client is dropped from the
    /// sink (or the sink itself shuts down).
    stop: Arc<AtomicBool>,
}

/// @brief Taps the encoder's H.264 elementary stream to recorders over a Unix
/// socket, kept out of band from the live viewer transport so recording neither
/// perturbs nor depends on what viewers see. That separation is the point: a
/// recorder can attach, stall, or detach without touching the WebSocket / WebRTC
/// path, and recording runs even with no viewer connected — which a tee off the
/// viewer stream could not promise.
///
/// `bind` owns the listener and spawns the accept thread; each accepted
/// connection gets its own bounded queue and writer thread (see `ClientHandle`)
/// so one slow reader cannot stall frame delivery to the others or to the caller
/// (the encode thread). `frames_since_idr` and `keyint_frames` drive
/// `should_force_idr`, which the encoder folds into its own keyframe decision —
/// forcing an IDR on every new connection and on a periodic cadence — so a
/// recorder that joins mid-stream always lands on a decodable entry point.
pub struct RecordingSink {
    path: String,
    clients: Arc<Mutex<Vec<ClientHandle>>>,
    shutdown: Arc<AtomicBool>,
    frames_since_idr: Arc<AtomicU32>,
    keyint_frames: u32,
}

impl RecordingSink {
    /// @brief The best-effort entry point to recording: yields a shared sink only
    /// when recording is both configured and successfully bound, and `None`
    /// otherwise — because recording is an optional tap that must never block or
    /// crash capture. An empty `settings_path` means the feature is simply off; a
    /// `bind` failure (e.g. the path cannot be bound) is logged to stderr and
    /// swallowed rather than propagated, so a broken sink degrades to "no
    /// recording" instead of taking the pipeline down with it. Success is wrapped
    /// in an `Arc` so it can be shared with whatever encoder(s) feed it.
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

    /// @brief Stand up the socket and the one long-lived thread that owns it —
    /// together the entire listening half of the sink. A single dedicated accept
    /// thread, rather than polling folded into the encode path, is what lets
    /// recorders connect at any point during capture without the frame path ever
    /// waiting on connection setup. The steps, and why each is here:
    ///
    /// 1. **Stale socket cleanup**: best-effort `remove_file` on `path` first,
    ///    since `UnixListener::bind` fails if a socket file is already there
    ///    (e.g. left over from a prior crashed run).
    /// 2. **Nonblocking listener**: `accept()` never blocks the accept
    ///    thread, so the thread can also check the `shutdown` flag promptly
    ///    between connection attempts instead of sitting blocked in
    ///    `accept()`.
    /// 3. **Startup IDR arming**: `frames_since_idr` starts at `u32::MAX`, so
    ///    the very first `should_force_idr()` call after bind is already due
    ///    — regardless of whether any client has connected by the time the
    ///    first frame is encoded.
    /// 4. **Accept loop** (runs on the spawned thread until `shutdown`):
    ///    - **New connection**: sets a `WRITE_TIMEOUT` write timeout on the
    ///      stream, abandoning the connection if that fails (a bad fd is not
    ///      worth retrying). A dedicated writer thread is spawned that pulls
    ///      frames from a fresh bounded channel, writing each via
    ///      `write_all_frame` and exiting (with a logged reason) on any
    ///      error — soft errors are already retried inside
    ///      `write_all_frame`, so an error here means a hard failure, and the
    ///      loop also exits early if `stop` is set. The channel's sender and
    ///      the `stop` flag become a `ClientHandle` pushed into the shared
    ///      `clients` list, and `frames_since_idr` resets to `u32::MAX` so the
    ///      encoder's next frame is forced to an IDR — the newly joined
    ///      client's (and, since the stream is shared, every other client's)
    ///      next frame is guaranteed decodable.
    ///    - **`WouldBlock`**: no connection is pending; sleeps
    ///      `ACCEPT_POLL_INTERVAL` before polling again.
    ///    - **Other error**: logs and backs off 500ms so a persistent accept
    ///      failure does not spin the thread hot.
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

    /// @brief The sink's answer to "does the recorded stream need a keyframe
    /// now?", which the encoder folds into its own keyframe decision — the single
    /// hook through which recording bends the encoder's GOP to its needs (an
    /// immediate entry point for a joiner, a periodic one for seekability) without
    /// the encoder having to track recorders itself. The counter lives here
    /// because only the sink knows when a client connected.
    ///
    /// `frames_since_idr` is atomically incremented on every call, and `idr_due`
    /// is evaluated against the count *before* that increment. A due result resets
    /// the counter to zero, so the following `keyint_frames` calls return `false`
    /// again. Bind time and every new client connection force the counter to
    /// `u32::MAX` (see `bind`), which makes it due on the very next call — that is
    /// how a joining client is guaranteed to receive an IDR as its first frame.
    pub fn should_force_idr(&self) -> bool {
        let prev = self.frames_since_idr.fetch_add(1, Ordering::Relaxed);
        if idr_due(prev, self.keyint_frames) {
            self.frames_since_idr.store(0, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// @brief Hand one encoded frame to every recorder without ever blocking the
    /// caller — which matters because the caller is the real-time encode thread,
    /// and a stall here would hitch the live stream for viewers, not just for
    /// recorders. Every choice below serves that constraint: non-blocking sends, a
    /// single shared copy, and dropping a client rather than waiting on it.
    ///
    /// `data` is a raw Annex-B H.264 byte slice (a full frame or access unit).
    ///
    /// 1. **Early-outs**: an empty `data` is ignored, and — the common case
    ///    when the socket is configured but nothing has connected yet — an
    ///    empty client list returns before paying for the per-frame heap
    ///    allocation below.
    /// 2. **One shared copy**: `data` is copied once into an `Arc<Vec<u8>>` so
    ///    every client's queue holds a cheap clone of the same allocation
    ///    instead of one copy per client.
    /// 3. **Non-blocking fan-out**: each client's bounded channel gets a
    ///    `try_send`, which never blocks the caller. A `Full` queue (the
    ///    client's writer thread cannot keep up with the incoming rate) or a
    ///    `Disconnected` sender (the writer thread has exited) marks that
    ///    client index for removal; a healthy send is silently a no-op here.
    /// 4. **Removal**: marked indices are removed in reverse order with
    ///    `swap_remove` so earlier indices stay valid as later ones are
    ///    removed. Each removed client's `stop` flag is set so its writer
    ///    thread notices and exits promptly instead of draining up to
    ///    `CLIENT_QUEUE_CAP` queued frames first.
    pub fn write_frame(&self, data: &[u8]) {
        if data.is_empty() {
            return;
        }

        let mut clients = self.clients.lock().unwrap();
        if clients.is_empty() {
            return;
        }

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
            let removed = clients.swap_remove(idx);
            removed.stop.store(true, Ordering::Relaxed);
        }
    }
}

impl Drop for RecordingSink {
    /// @brief Tear the sink down promptly and completely — stop accepting, end
    /// every writer thread, free the socket path — so no thread or bound node
    /// outlives the sink. The subtlety it exists to handle is that a writer thread
    /// can be stuck in either of two states, and each needs a different nudge to
    /// exit; miss either and that thread leaks.
    ///
    /// `shutdown` governs only the accept thread's loop condition (it stops taking
    /// connections and exits on its next poll); writer threads never look at it.
    /// Each writer is instead released by two signals, one per possible state: its
    /// `stop` flag is set first so a writer mid-retry inside `write_all_frame`
    /// aborts rather than finishing a stalled write, then `clients.clear()` drops
    /// every queue sender, disconnecting each channel to wake a writer parked idle
    /// in `rx.iter()` waiting for its next frame. Clearing the list here also makes
    /// teardown immediate rather than leaving the writers alive until the accept
    /// thread happens to exit and release its own clone of the client list. The
    /// socket file is removed last so a later bind at the same path does not
    /// collide with a leftover node.
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Ok(mut clients) = self.clients.lock() {
            for client in clients.iter() {
                client.stop.store(true, Ordering::Relaxed);
            }
            clients.clear();
        }
        let _ = fs::remove_file(&self.path);
    }
}

/// @brief Get one whole frame onto a recorder's socket, or fail cleanly — never
/// leave a partial frame behind, because a truncated Annex-B NAL corrupts the
/// recorded stream from that point on. Since the socket carries a write timeout,
/// even a healthy reader can absorb only part of a `write` at a time, so this
/// resumes from the last written offset across soft timeouts instead of
/// surrendering the connection mid-frame.
///
/// 1. **Why retry at all**: the stream has a write timeout set, so a
///    slow-but-healthy reader can make `write` return `TimedOut`/`WouldBlock`
///    after only some leading bytes of a frame land. Bailing out there would
///    emit a truncated Annex-B NAL, so the loop instead retries the unwritten
///    remainder until the whole frame lands or a hard error occurs.
/// 2. **Soft errors** (`TimedOut`, `WouldBlock`, `Interrupted`): `written`
///    already reflects any partial progress, so these just loop back and
///    retry the remainder. The stream's write timeout naturally paces the
///    `TimedOut` case; a small sleep is added for `WouldBlock` so a socket
///    that returns it without actually blocking does not spin the retry loop
///    hot.
/// 3. **Bounding a permanently stuck client**: this function alone would
///    retry forever against a dead peer. That is bounded elsewhere — the
///    client's bounded queue fills up, `write_frame` drops it from the client
///    list and sets `stop`, which this loop checks at the top of every
///    iteration, aborting with an error as soon as it is set.
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

/// @brief The pure cadence arithmetic behind `should_force_idr`, pulled out so
/// its off-by-one and its degenerate-cadence guard can be reasoned about and
/// unit-tested free of the atomics and threading that surround the real counter.
///
/// Due when `frames_since_idr >= keyint_frames - 1`, computed with
/// `saturating_sub` so a `keyint_frames` of 0 cannot underflow and instead safely
/// makes every call due. That `- 1` is what makes a cadence of `keyint_frames`
/// fire on the `keyint_frames`-th call after `should_force_idr` last reset the
/// counter to zero.
fn idr_due(frames_since_idr: u32, keyint_frames: u32) -> bool {
    frames_since_idr >= keyint_frames.saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::idr_due;

    /// @brief The very first call after construction is always due:
    /// `frames_since_idr` starts at `u32::MAX`, which is what lets a client
    /// that joins before any frame has been encoded still receive an IDR as
    /// its first frame.
    #[test]
    fn first_frame_forces_idr() {
        assert!(idr_due(u32::MAX, 60));
    }

    /// @brief A cadence of 60 is not due until the 59th `frames_since_idr`
    /// value (the 60th call after a reset) and stays due afterward, matching
    /// the reset-on-due bookkeeping in `should_force_idr`.
    #[test]
    fn cadence_matches_keyint() {
        let keyint = 60;
        assert!(!idr_due(0, keyint));
        assert!(!idr_due(58, keyint));
        assert!(idr_due(59, keyint));
        assert!(idr_due(60, keyint));
    }

    /// @brief A `keyint_frames` of 0 or 1 cannot underflow the saturating
    /// subtraction and instead makes every call due, at any frame count.
    #[test]
    fn keyint_zero_and_one_do_not_underflow() {
        assert!(idr_due(0, 0));
        assert!(idr_due(u32::MAX, 0));
        assert!(idr_due(0, 1));
        assert!(idr_due(1, 1));
    }
}
