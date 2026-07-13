/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Encoder backends shared by the X11 and Wayland capture pipelines.
//!
//! The active encoder is chosen at capture start based on the operator settings and available
//! hardware. The software path ([`software`]) handles striped JPEG and x264; the hardware paths
//! ([`nvenc`], [`vaapi`], [`oh264`]) produce full-frame H.264. [`overlay`] composites PNG
//! watermarks onto frames before encoding.

/// CPU-based x264 / JPEG striped encoder with per-stripe change detection.
pub mod software;
/// NVIDIA NVENC hardware H.264 encoder loaded via runtime `libcuda` / `libnvidia-encode`.
pub mod nvenc;
/// VA-API hardware H.264 encoder for Intel / AMD GPUs via FFmpeg.
pub mod vaapi;
/// Cisco OpenH264 software H.264 encoder (BSD-licensed, full-frame, bitrate-controlled).
pub mod oh264;
/// PNG watermark overlay composited onto frames before encoding.
pub mod overlay;
