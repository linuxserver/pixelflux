/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! HTTP server implementing the [Anthropic Computer Use](https://github.com/anthropics/claude-quickstarts/tree/main/computer-use-demo) specification.
//!
//! Enabled by setting the `PIXELFLUX_CU` environment variable to the listen port. The server
//! handles `POST /computer-use` requests for screenshots, mouse/keyboard injection, scrolling,
//! and cursor position queries — all targeting the Wayland compositor owned by the same process.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use std::io::Cursor;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use image::{ImageBuffer, Rgba, ImageFormat};
use serde::Deserialize;
use tiny_http;

use crate::ThreadCommand;

fn clamp<T: PartialOrd>(v: T, lo: T, hi: T) -> T {
    if v < lo { lo } else if v > hi { hi } else { v }
}

/// Turn a raw framebuffer into a PNG the Computer Use agent can actually look at.
///
/// The `screenshot` and `zoom` crop paths have to hand the agent an image, and the API carries it
/// as base64 PNG, so this encodes a flat RGBA buffer (with its dimensions) through the `image`
/// crate and returns the PNG bytes for that response payload.
pub fn encode_png_rgba(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let img = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, data.to_vec())
        .ok_or("Failed to create image buffer")?;
    let mut png = Vec::new();
    img.write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
        .map_err(|e| format!("PNG encode error: {}", e))?;
    Ok(png)
}

fn scancode_for_char(c: char) -> Option<(u32, bool)> {
    Some(match c {
        'a' | 'A' => (38, c.is_uppercase()),
        'b' | 'B' => (56, c.is_uppercase()),
        'c' | 'C' => (54, c.is_uppercase()),
        'd' | 'D' => (40, c.is_uppercase()),
        'e' | 'E' => (26, c.is_uppercase()),
        'f' | 'F' => (41, c.is_uppercase()),
        'g' | 'G' => (42, c.is_uppercase()),
        'h' | 'H' => (43, c.is_uppercase()),
        'i' | 'I' => (31, c.is_uppercase()),
        'j' | 'J' => (44, c.is_uppercase()),
        'k' | 'K' => (45, c.is_uppercase()),
        'l' | 'L' => (46, c.is_uppercase()),
        'm' | 'M' => (58, c.is_uppercase()),
        'n' | 'N' => (57, c.is_uppercase()),
        'o' | 'O' => (32, c.is_uppercase()),
        'p' | 'P' => (33, c.is_uppercase()),
        'q' | 'Q' => (24, c.is_uppercase()),
        'r' | 'R' => (27, c.is_uppercase()),
        's' | 'S' => (39, c.is_uppercase()),
        't' | 'T' => (28, c.is_uppercase()),
        'u' | 'U' => (30, c.is_uppercase()),
        'v' | 'V' => (55, c.is_uppercase()),
        'w' | 'W' => (25, c.is_uppercase()),
        'x' | 'X' => (53, c.is_uppercase()),
        'y' | 'Y' => (29, c.is_uppercase()),
        'z' | 'Z' => (52, c.is_uppercase()),
        '0' => (19, false), '1' => (10, false), '2' => (11, false),
        '3' => (12, false), '4' => (13, false), '5' => (14, false),
        '6' => (15, false), '7' => (16, false), '8' => (17, false),
        '9' => (18, false),
        '!' => (10, true), '@' => (11, true), '#' => (12, true),
        '$' => (13, true), '%' => (14, true), '^' => (15, true),
        '&' => (16, true), '*' => (17, true), '(' => (18, true),
        ')' => (19, true),
        '-' => (20, false), '_' => (20, true),
        '=' => (21, false), '+' => (21, true),
        '[' => (34, false), '{' => (34, true),
        ']' => (35, false), '}' => (35, true),
        ';' => (47, false), ':' => (47, true),
        '\'' => (48, false), '"' => (48, true),
        '`' => (49, false), '~' => (49, true),
        '\\' => (51, false), '|' => (51, true),
        ',' => (59, false), '<' => (59, true),
        '.' => (60, false), '>' => (60, true),
        '/' => (61, false), '?' => (61, true),
        ' ' => (65, false),
        '\t' => (23, false),
        '\n' => (36, false),
        _ => return None,
    })
}

fn scancode_for_keyname(name: &str) -> Option<u32> {
    Some(match name.to_lowercase().as_str() {
        "return" | "enter" => 36,
        "tab" => 23,
        "escape" | "esc" => 9,
        "backspace" => 22,
        "delete" => 119,
        "insert" => 118,
        "home" => 110,
        "end" => 115,
        "pageup" => 112,
        "pagedown" => 117,
        "up" => 111,
        "down" => 116,
        "left" => 113,
        "right" => 114,
        "space" => 65,
        "capslock" => 66,
        "numlock" => 77,
        "scrolllock" => 78,
        "printscreen" | "sysrq" | "print" => 107,
        "pause" | "break" => 127,
        "f1" => 67, "f2" => 68, "f3" => 69, "f4" => 70,
        "f5" => 71, "f6" => 72, "f7" => 73, "f8" => 74,
        "f9" => 75, "f10" => 76, "f11" => 95, "f12" => 96,
        "f13" => 191, "f14" => 192, "f15" => 193, "f16" => 194,
        "f17" => 195, "f18" => 196, "f19" => 197, "f20" => 198,
        "f21" => 199, "f22" => 200, "f23" => 201, "f24" => 202,
        "ctrl" | "lctrl" | "leftctrl" => 37,
        "rctrl" | "rightctrl" => 105,
        "shift" | "lshift" | "leftshift" => 50,
        "rshift" | "rightshift" => 62,
        "alt" | "lalt" | "leftalt" => 64,
        "ralt" | "rightalt" => 108,
        "super" | "meta" | "lsuper" | "leftmeta" | "leftsuper" | "windows" | "leftwindows" => 133,
        "rsuper" | "rightmeta" | "rightsuper" | "rightwindows" => 134,
        "menu" | "compose" => 135,
        "kp_0" | "kp0" => 90, "kp_1" | "kp1" => 87,
        "kp_2" | "kp2" => 88, "kp_3" | "kp3" => 89,
        "kp_4" | "kp4" => 83, "kp_5" | "kp5" => 84,
        "kp_6" | "kp6" => 85, "kp_7" | "kp7" => 79,
        "kp_8" | "kp8" => 80, "kp_9" | "kp9" => 81,
        "kp_decimal" | "kp_dot" => 91,
        "kp_divide" | "kp_slash" => 106,
        "kp_multiply" | "kp_asterisk" => 63,
        "kp_subtract" | "kp_minus" => 82,
        "kp_add" | "kp_plus" => 86,
        "kp_enter" => 104,
        _ => return None,
    })
}

fn is_modifier(scancode: u32) -> bool {
    matches!(scancode, 37 | 105 | 50 | 62 | 64 | 108 | 133 | 134)
}

fn send_key(tx: &smithay::reexports::calloop::channel::Sender<ThreadCommand>, scancode: u32, pressed: bool) {
    let _ = tx.send(ThreadCommand::KeyboardKey {
        scancode,
        state: if pressed { 1 } else { 0 },
    });
}

fn send_mouse_move(tx: &smithay::reexports::calloop::channel::Sender<ThreadCommand>, x: f64, y: f64) {
    let _ = tx.send(ThreadCommand::PointerMotion { x, y });
}

const BTN_LEFT: u32 = 0x110;
const BTN_RIGHT: u32 = 0x111;
const BTN_MIDDLE: u32 = 0x112;

fn send_mouse_button(tx: &smithay::reexports::calloop::channel::Sender<ThreadCommand>, btn: u32, pressed: bool) {
    let _ = tx.send(ThreadCommand::PointerButton {
        btn,
        state: if pressed { 1 } else { 0 },
    });
}

fn send_scroll(tx: &smithay::reexports::calloop::channel::Sender<ThreadCommand>, x: f64, y: f64) {
    let _ = tx.send(ThreadCommand::PointerAxis { x, y });
}

fn screenshot(
    tx: &smithay::reexports::calloop::channel::Sender<ThreadCommand>,
) -> Result<Vec<u8>, String> {
    let (resp_tx, resp_rx) = mpsc::channel();
    tx.send(ThreadCommand::CuScreenshot { resp: resp_tx })
        .map_err(|_| "Failed to request screenshot".to_string())?;
    resp_rx.recv().map_err(|_| "Screenshot failed".to_string())
}

fn cursor_position(
    tx: &smithay::reexports::calloop::channel::Sender<ThreadCommand>,
) -> Result<(f64, f64), String> {
    let (resp_tx, resp_rx) = mpsc::channel();
    tx.send(ThreadCommand::CuCursorPosition { resp: resp_tx })
        .map_err(|_| "Failed to request cursor position".to_string())?;
    resp_rx.recv().map_err(|_| "Cursor position failed".to_string())
}

fn get_fb_size(
    tx: &smithay::reexports::calloop::channel::Sender<ThreadCommand>,
) -> Result<(i32, i32), String> {
    let (resp_tx, resp_rx) = mpsc::channel();
    tx.send(ThreadCommand::CuGetInfo { resp: resp_tx })
        .map_err(|_| "Failed to request compositor info".to_string())?;
    let (w, h, _) = resp_rx.recv().map_err(|_| "Compositor info request failed".to_string())?;
    Ok((w, h))
}

#[derive(Deserialize)]
struct CuActionRequest {
    action: String,
    coordinate: Option<[f64; 2]>,
    start_coordinate: Option<[f64; 2]>,
    text: Option<String>,
    key: Option<String>,
    scroll_direction: Option<String>,
    scroll_amount: Option<i32>,
    duration: Option<f64>,
    region: Option<[f64; 4]>,
}

fn ok_json() -> String {
    "{\"result\":\"ok\"}".to_string()
}

fn handle_action(
    req: CuActionRequest,
    tx: &smithay::reexports::calloop::channel::Sender<ThreadCommand>,
) -> String {
    let result = handle_action_inner(req, tx);
    match result {
        Ok(response) => response,
        Err(e) => format!("{{\"error\":\"{}\"}}", e.replace('"', "\\\"")),
    }
}

/// Turn one parsed Computer Use request into a real action on the captured desktop — the
/// bridge that lets an external AI agent see and drive the session.
///
/// Every action ultimately becomes a compositor thread command or a framebuffer read, so this is
/// where the API's vocabulary meets the running compositor. Coordinates are clamped to the current
/// framebuffer so a mistaken agent click cannot address off-screen pixels; keyboard and pointer
/// actions translate into input events; and actions that must look at the screen (`screenshot`,
/// `zoom`) request a fresh capture first, so the agent reasons about the current frame rather than
/// a stale one. Returns the JSON response, or an error string explaining why the action could not
/// run.
fn handle_action_inner(
    req: CuActionRequest,
    tx: &smithay::reexports::calloop::channel::Sender<ThreadCommand>,
) -> Result<String, String> {
    let sleep_ms = |ms: u64| thread::sleep(Duration::from_millis(ms));

    let (fb_w, fb_h) = get_fb_size(tx)?;

    let handle_coord = |coord: [f64; 2]| -> (f64, f64) {
        (
            clamp(coord[0], 0.0, (fb_w - 1) as f64),
            clamp(coord[1], 0.0, (fb_h - 1) as f64),
        )
    };

    let handle_modifier = |mod_name: &str| -> Option<u32> {
        let sc = scancode_for_keyname(mod_name)?;
        if is_modifier(sc) { Some(sc) } else { None }
    };

    match req.action.as_str() {
        "screenshot" => {
            let png = screenshot(tx)?;
            let b64 = BASE64.encode(&png);
            Ok(format!("{{\"data\":\"{}\"}}", b64))
        }

        "mouse_move" => {
            let coord = req.coordinate.ok_or("Missing coordinate")?;
            let (fx, fy) = handle_coord(coord);
            send_mouse_move(tx, fx, fy);
            Ok(ok_json())
        }

        "left_click" | "right_click" | "middle_click" => {
            let btn: u32 = match req.action.as_str() {
                "left_click" => BTN_LEFT,
                "right_click" => BTN_RIGHT,
                _ => BTN_MIDDLE,
            };
            if let Some(coord) = req.coordinate {
                let (fx, fy) = handle_coord(coord);
                send_mouse_move(tx, fx, fy);
                sleep_ms(30);
            }
            if let Some(ref mod_name) = req.text {
                if let Some(sc) = handle_modifier(mod_name) {
                    send_key(tx, sc, true);
                    sleep_ms(20);
                }
            }
            send_mouse_button(tx, btn, true);
            sleep_ms(20);
            send_mouse_button(tx, btn, false);
            if let Some(ref mod_name) = req.text {
                if let Some(sc) = handle_modifier(mod_name) {
                    sleep_ms(10);
                    send_key(tx, sc, false);
                }
            }
            Ok(ok_json())
        }

        "double_click" | "triple_click" => {
            let n = if req.action == "double_click" { 2 } else { 3 };
            if let Some(coord) = req.coordinate {
                let (fx, fy) = handle_coord(coord);
                send_mouse_move(tx, fx, fy);
                sleep_ms(30);
            }
            if let Some(ref mod_name) = req.text {
                if let Some(sc) = handle_modifier(mod_name) {
                    send_key(tx, sc, true);
                    sleep_ms(20);
                }
            }
            for _ in 0..n {
                send_mouse_button(tx, BTN_LEFT, true);
                sleep_ms(10);
                send_mouse_button(tx, BTN_LEFT, false);
                sleep_ms(10);
            }
            if let Some(ref mod_name) = req.text {
                if let Some(sc) = handle_modifier(mod_name) {
                    sleep_ms(10);
                    send_key(tx, sc, false);
                }
            }
            Ok(ok_json())
        }

        "left_click_drag" => {
            let start = req.start_coordinate.ok_or("Missing start_coordinate")?;
            let end = req.coordinate.ok_or("Missing coordinate")?;
            let (sx, sy) = handle_coord(start);
            let (ex, ey) = handle_coord(end);
            send_mouse_move(tx, sx, sy);
            sleep_ms(30);
            send_mouse_button(tx, BTN_LEFT, true);
            sleep_ms(30);
            send_mouse_move(tx, ex, ey);
            sleep_ms(30);
            send_mouse_button(tx, BTN_LEFT, false);
            Ok(ok_json())
        }

        "left_mouse_down" => {
            send_mouse_button(tx, BTN_LEFT, true);
            Ok(ok_json())
        }

        "left_mouse_up" => {
            send_mouse_button(tx, BTN_LEFT, false);
            Ok(ok_json())
        }

        "type" => {
            let text = req.text.as_deref().ok_or("Missing text")?;
            for chunk in text.as_bytes().chunks(50) {
                let chunk_str = std::str::from_utf8(chunk).map_err(|_| "Invalid UTF-8")?;
                for ch in chunk_str.chars() {
                    if ch == '\n' {
                        send_key(tx, 36, true);
                        sleep_ms(10);
                        send_key(tx, 36, false);
                        sleep_ms(10);
                        continue;
                    }
                    if let Some((sc, need_shift)) = scancode_for_char(ch) {
                        if need_shift {
                            send_key(tx, 50, true);
                            sleep_ms(5);
                            send_key(tx, sc, true);
                            sleep_ms(10);
                            send_key(tx, sc, false);
                            send_key(tx, 50, false);
                        } else {
                            send_key(tx, sc, true);
                            sleep_ms(10);
                            send_key(tx, sc, false);
                        }
                        sleep_ms(8);
                    }
                }
                sleep_ms(20);
            }
            Ok(ok_json())
        }

        "key" => {
            let text = req.text.as_deref().ok_or("Missing text")?;
            let parts: Vec<&str> = text.split('+').collect();
            let mut mods: Vec<u32> = Vec::new();
            let mut main_key: Option<u32> = None;
            for part in &parts {
                let trimmed = part.trim();
                if let Some(sc) = scancode_for_keyname(trimmed) {
                    if is_modifier(sc) {
                        mods.push(sc);
                    } else {
                        main_key = Some(sc);
                    }
                }
            }
            for &sc in &mods {
                send_key(tx, sc, true);
                sleep_ms(10);
            }
            if let Some(sc) = main_key {
                send_key(tx, sc, true);
                sleep_ms(30);
                send_key(tx, sc, false);
            } else if let Some(&last_mod) = mods.last() {
                send_key(tx, last_mod, true);
                sleep_ms(30);
                send_key(tx, last_mod, false);
            }
            for &sc in mods.iter().rev() {
                sleep_ms(10);
                send_key(tx, sc, false);
            }
            Ok(ok_json())
        }

        "hold_key" => {
            let text = req.text.as_deref().ok_or("Missing text")?;
            let duration = req.duration.ok_or("Missing duration")?;
            let duration = duration.min(100.0);
            if let Some(sc) = scancode_for_keyname(text) {
                send_key(tx, sc, true);
                sleep_ms((duration * 1000.0) as u64);
                send_key(tx, sc, false);
            }
            Ok(ok_json())
        }

        "scroll" => {
            let dir = req.scroll_direction.as_deref().ok_or("Missing scroll_direction")?;
            let amount = req.scroll_amount.unwrap_or(1).max(0) as f64;
            if let Some(coord) = req.coordinate {
                let (fx, fy) = handle_coord(coord);
                send_mouse_move(tx, fx, fy);
                sleep_ms(30);
            }
            if let Some(ref mod_name) = req.text {
                if let Some(sc) = handle_modifier(mod_name) {
                    send_key(tx, sc, true);
                    sleep_ms(20);
                }
            }
            let (dx, dy) = match dir {
                "up" => (0.0, -amount),
                "down" => (0.0, amount),
                "left" => (-amount, 0.0),
                "right" => (amount, 0.0),
                _ => return Err(format!("Invalid scroll_direction: {}", dir)),
            };
            send_scroll(tx, dx, dy);
            sleep_ms(30);
            if let Some(ref mod_name) = req.text {
                if let Some(sc) = handle_modifier(mod_name) {
                    sleep_ms(10);
                    send_key(tx, sc, false);
                }
            }
            Ok(ok_json())
        }

        "cursor_position" => {
            let (x, y) = cursor_position(tx)?;
            Ok(format!("{{\"text\":\"X={},Y={}\"}}", x.round() as i32, y.round() as i32))
        }

        "wait" => {
            let duration = req.duration.unwrap_or(1.0).min(100.0);
            sleep_ms((duration * 1000.0) as u64);
            Ok(ok_json())
        }

        "zoom" => {
            let region = req.region.ok_or("Missing region")?;
            let (x0, y0, x1, y1) = (region[0], region[1], region[2], region[3]);
            let left = clamp(x0.round() as u32, 0, fb_w as u32 - 1);
            let top = clamp(y0.round() as u32, 0, fb_h as u32 - 1);
            let right = clamp(x1.round() as u32, left + 1, fb_w as u32);
            let bottom = clamp(y1.round() as u32, top + 1, fb_h as u32);
            let crop_w = right - left;
            let crop_h = bottom - top;
            let png = screenshot(tx)?;
            if crop_w > 0 && crop_h > 0 {
                if let Ok(img) = image::load_from_memory(&png) {
                    let cropped = img.crop_imm(left, top, crop_w, crop_h);
                    let mut out = Vec::new();
                    cropped.write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
                        .map_err(|e| format!("Crop/encode error: {}", e))?;
                    let b64 = BASE64.encode(&out);
                    return Ok(format!("{{\"data\":\"{}\"}}", b64));
                }
            }
            let b64 = BASE64.encode(&png);
            Ok(format!("{{\"data\":\"{}\"}}", b64))
        }

        _ => Err(format!("Unknown action: {}", req.action)),
    }
}

/// Expose the captured desktop to an AI agent over HTTP, so a Computer Use client can drive
/// the session much as a human viewer would.
///
/// It runs on its own thread listening on `0.0.0.0:<port>` for POST `/computer-use` JSON actions
/// (screenshot, mouse_move, click, key, scroll, …). Framebuffer dimensions are re-queried from the
/// compositor on every request rather than cached, because the stream can start, stop, or resize
/// underneath the agent and a stale size would misplace every coordinate. A screenshot forces a
/// one-frame GPU readback when the pipeline is in zero-copy mode, since that path never otherwise
/// brings host pixels back to the CPU and the agent needs an image to look at.
pub fn run_cu_server(
    tx: smithay::reexports::calloop::channel::Sender<ThreadCommand>,
    port: u16,
) {
    println!(
        "[ComputerUse] Server listening on port {}",
        port,
    );

    let server = match tiny_http::Server::http(format!("0.0.0.0:{}", port)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[ComputerUse] Failed to start server: {}", e);
            return;
        }
    };

    for mut request in server.incoming_requests() {
        let mut body = String::new();
        if let Err(e) = request.as_reader().read_to_string(&mut body) {
            let _ = request.respond(tiny_http::Response::from_string(format!(
                "{{\"error\":\"{}\"}}", e
            ))
            .with_status_code(400)
            .with_header(
                "Content-Type: application/json".parse::<tiny_http::Header>().unwrap()
            ));
            continue;
        }

        let parsed: CuActionRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => {
                let _ = request.respond(tiny_http::Response::from_string(format!(
                    "{{\"error\":\"Invalid JSON: {}\"}}", e
                ))
                .with_status_code(400)
                .with_header(
                    "Content-Type: application/json".parse::<tiny_http::Header>().unwrap()
                ));
                continue;
            }
        };

        let json_response = handle_action(parsed, &tx);
        let _ = request.respond(
            tiny_http::Response::from_string(json_response)
                .with_status_code(200)
                .with_header(
                    "Content-Type: application/json".parse::<tiny_http::Header>().unwrap()
                )
        );
    }
}
