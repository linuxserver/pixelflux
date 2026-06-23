/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(deref_nullptr)]
#![allow(clippy::all)]
#![allow(unknown_lints)]
#![allow(unpredictable_function_pointer_comparisons)]
#![allow(unnecessary_transmutes)]

// The bindgen output (types/enums/functions + the regex-extracted NV_ENC_*_VER constants and
// codec/preset GUIDs). Regenerated in place by build.rs under the `regen` feature.
include!("bindgen/nvenc.rs");

// Loader entry points bindgen doesn't emit: the dlsym symbol names + their signatures, and the
// runtime library name. NvEncodeAPIGetMaxSupportedVersion drives the API-version negotiation.
pub const NV_ENCODE_API_GET_MAX_SUPPORTED_VERSION_FN_NAME: &[u8] =
    b"NvEncodeAPIGetMaxSupportedVersion\0";
pub type NvEncodeApiGetMaxSupportedVersionFn =
    unsafe extern "C" fn(version: *mut u32) -> NVENCSTATUS;

pub const NV_ENCODE_API_CREATE_INSTANCE_FN_NAME: &[u8] = b"NvEncodeAPICreateInstance\0";
pub type NvEncodeApiCreateInstanceFn =
    unsafe extern "C" fn(functionList: *mut NV_ENCODE_API_FUNCTION_LIST) -> NVENCSTATUS;

#[cfg(not(windows))]
pub const NVENC_DLL_NAME: &str = "libnvidia-encode.so.1";
#[cfg(windows)]
pub const NVENC_DLL_NAME: &str = "nvEncodeAPI64.dll";
