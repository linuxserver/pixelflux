/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::os::unix::io::AsRawFd;
use std::ptr;
use std::sync::Arc;

use libloading::{Library, Symbol};
use smithay::backend::allocator::{dmabuf::Dmabuf, Buffer};

use crate::recording_sink::RecordingSink;
use crate::RustCaptureSettings;
use nvcodec_sys::cuda::*;
use nvcodec_sys::*;

/// @brief EGL constants and type definitions for C interop.
type EGLDisplay = *const c_void;
type EGLImageKHR = *mut c_void;
type EGLint = i32;
type EGLenum = u32;
type EGLBoolean = u32;

const EGL_NO_IMAGE_KHR: EGLImageKHR = ptr::null_mut();
const EGL_LINUX_DMA_BUF_EXT: u32 = 0x3270;
const EGL_DMA_BUF_PLANE0_FD_EXT: EGLint = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EGLint = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: EGLint = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: EGLint = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: EGLint = 0x3444;
const EGL_WIDTH: EGLint = 0x3057;
const EGL_HEIGHT: EGLint = 0x3056;
const EGL_LINUX_DRM_FOURCC_EXT: EGLint = 0x3271;
const EGL_NONE: EGLint = 0x3038;

/// @brief CUDA types specifically used for EGL interop.
type CUgraphicsResource = *mut c_void;

/// @brief Represents a frame mapped from EGL to CUDA.
#[repr(C)]
#[derive(Clone, Copy)]
struct CUeglFrame {
    frame: CUeglFrameUnion,
    width: u32,
    height: u32,
    depth: u32,
    pitch: u32,
    plane_count: u32,
    num_channels: u32,
    frame_type: u32,
    egl_color_format: u32,
    cu_format: u32,
}

/// @brief Union for frame data pointers (array vs pitch linear).
#[repr(C)]
#[derive(Clone, Copy)]
union CUeglFrameUnion {
    p_array: [CUarray; 3],
    p_pitch: [*mut c_void; 3],
}

type EglCreateImageKhrFn = unsafe extern "C" fn(
    dpy: EGLDisplay,
    ctx: *mut c_void,
    target: EGLenum,
    buffer: *mut c_void,
    attrib_list: *const EGLint,
) -> EGLImageKHR;
type EglDestroyImageKhrFn = unsafe extern "C" fn(dpy: EGLDisplay, image: EGLImageKHR) -> EGLBoolean;

/// @brief dynamically loaded EGL function pointers.
struct EglFunctions {
    _lib: Library,
    eglGetProcAddress: unsafe extern "C" fn(procname: *const c_char) -> *mut c_void,
    eglCreateImageKHR: EglCreateImageKhrFn,
    eglDestroyImageKHR: EglDestroyImageKhrFn,
}

/// @brief Dynamically loaded CUDA function pointers.
struct CudaFunctions {
    _lib: Library,
    cuInit: unsafe extern "C" fn(flags: u32) -> CUresult,
    cuDeviceGet: unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult,
    cuDeviceGetByPCIBusId: unsafe extern "C" fn(dev: *mut CUdevice, pciBusId: *const c_char) -> CUresult,
    cuCtxCreate_v2: unsafe extern "C" fn(
        pctx: *mut CUcontext,
        flags: u32,
        dev: CUdevice,
    ) -> CUresult,
    cuCtxPushCurrent_v2: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    cuCtxPopCurrent_v2: unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult,
    cuCtxDestroy_v2: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    cuMemAlloc_v2: unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    cuMemAllocPitch_v2: unsafe extern "C" fn(
        dptr: *mut CUdeviceptr,
        pPitch: *mut usize,
        WidthInBytes: usize,
        Height: usize,
        ElementSizeBytes: u32,
    ) -> CUresult,
    cuMemFree_v2: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
    cuMemcpyHtoD_v2: unsafe extern "C" fn(
        dstDevice: CUdeviceptr,
        srcHost: *const c_void,
        ByteCount: usize,
    ) -> CUresult,
    cuMemcpyDtoH_v2: unsafe extern "C" fn(
        dstHost: *mut c_void,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
    ) -> CUresult,
    cuMemcpy2D_v2: unsafe extern "C" fn(pCopy: *const CUDA_MEMCPY2D) -> CUresult,
    cuMemHostRegister_v2: unsafe extern "C" fn(p: *mut c_void, bytesize: usize, flags: u32) -> CUresult,
    cuMemHostUnregister: unsafe extern "C" fn(p: *mut c_void) -> CUresult,
    cuGraphicsEGLRegisterImage: unsafe extern "C" fn(
        pCudaResource: *mut CUgraphicsResource,
        image: EGLImageKHR,
        flags: u32,
    ) -> CUresult,
    cuGraphicsUnregisterResource: unsafe extern "C" fn(resource: CUgraphicsResource) -> CUresult,
    cuGraphicsResourceGetMappedEglFrame: unsafe extern "C" fn(
        pEglFrame: *mut CUeglFrame,
        resource: CUgraphicsResource,
        index: u32,
        mipLevel: u32,
    ) -> CUresult,
    cuDeviceGetCount: unsafe extern "C" fn(count: *mut i32) -> CUresult,
    cuDeviceGetName: unsafe extern "C" fn(name: *mut c_char, len: i32, dev: CUdevice) -> CUresult,
    cuDeviceGetUuid: unsafe extern "C" fn(uuid: *mut CUuuid, dev: CUdevice) -> CUresult,
    cuGetErrorName: unsafe extern "C" fn(error: CUresult, pStr: *mut *const c_char) -> CUresult,
}

/// @brief Dynamically loaded NVENC API entry point.
struct NvencLibrary {
    _lib: Library,
    create_instance: unsafe extern "C" fn(
        functionList: *mut NV_ENCODE_API_FUNCTION_LIST,
    ) -> NVENCSTATUS,
    // Optional: lets us cap probing at the driver's max API version (absent on very old drivers).
    get_max_version: Option<unsafe extern "C" fn(*mut u32) -> NVENCSTATUS>,
}

/// Negotiated NVENC API (major, minor), set once per process. None until negotiation runs.
static NVENC_NEG_VER: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();

/// The NVENC structs this encoder version-tags. Struct revisions (bits 16-23) and the 1<<31
/// flag changed across SDKs, so a down-negotiated session must send exactly the words the
/// negotiated SDK defined -- older drivers reject anything else with NV_ENC_ERR_INVALID_VERSION.
#[derive(Clone, Copy, Debug)]
enum NvStruct {
    FunctionList,
    OpenSessionExParams,
    Config,
    RcParams,
    PresetConfig,
    InitializeParams,
    ReconfigureParams,
    RegisterResource,
    MapInputResource,
    CreateBitstreamBuffer,
    PicParams,
    LockBitstream,
}

impl NvStruct {
    /// (struct revision, 1<<31 flag) as defined by the SDK with packed version `api`
    /// ((major<<4)|minor). Sourced from nvEncodeAPI.h at FFmpeg nv-codec-headers tags
    /// n10.0.26.2, n11.0.10.3, n11.1.5.3, n12.0.16.1, n12.1.14.0, n12.2.72.0, n13.0.19.0;
    /// 10.0 is the negotiation floor, so the oldest arm also covers anything below it.
    fn rev(self, api: u32) -> (u32, bool) {
        match self {
            NvStruct::FunctionList => (2, false),
            NvStruct::OpenSessionExParams => (1, false),
            NvStruct::Config => match api {
                0xC2.. => (9, true),
                0xC0..=0xC1 => (8, true),
                _ => (7, true),
            },
            NvStruct::RcParams => (1, false),
            NvStruct::PresetConfig => (if api >= 0xC2 { 5 } else { 4 }, true),
            NvStruct::InitializeParams => match api {
                0xC2.. => (7, true),
                0xC1 => (6, true),
                _ => (5, true),
            },
            NvStruct::ReconfigureParams => (if api >= 0xC2 { 2 } else { 1 }, true),
            NvStruct::RegisterResource => match api {
                0xC2.. => (5, false),
                0xC0..=0xC1 => (4, false),
                _ => (3, false),
            },
            NvStruct::MapInputResource => (4, false),
            NvStruct::CreateBitstreamBuffer => (1, false),
            NvStruct::PicParams => match api {
                0xC2.. => (7, true),
                0xC0..=0xC1 => (6, true),
                _ => (4, true),
            },
            NvStruct::LockBitstream => match api {
                0xC2.. => (2, true),
                0xC1 => (1, true),
                0xC0 => (2, false),
                _ => (1, false),
            },
        }
    }
}

/// The NVENCAPI_STRUCT_VERSION word for `s` at API (major, minor): that SDK's struct revision
/// and flag bit, API major in bits 0-7, minor in bits 24-27, magic 0x7 in bits 28-30. For the
/// pinned nvcodec-sys version this reproduces the compile-time NV_ENC_*_VER constants exactly,
/// so a current driver is byte-for-byte unchanged.
fn nvenc_struct_ver(s: NvStruct, maj: u32, min: u32) -> u32 {
    let (rev, high_bit) = s.rev((maj << 4) | (min & 0xF));
    (maj & 0xFF) | ((min & 0xF) << 24) | (rev << 16) | (0x7 << 28) | ((high_bit as u32) << 31)
}

#[inline]
fn nvenc_cur_ver() -> (u32, u32) {
    NVENC_NEG_VER
        .get()
        .copied()
        .unwrap_or((NVENCAPI_VERSION & 0xFF, (NVENCAPI_VERSION >> 24) & 0xFF))
}

/// Struct-version word for `s` tagged with the negotiated API version.
#[inline]
fn sv(s: NvStruct) -> u32 {
    let (m, n) = nvenc_cur_ver();
    nvenc_struct_ver(s, m, n)
}

/// The raw apiVersion (major | minor<<24) for NvEncOpenEncodeSessionEx.
#[inline]
fn neg_api() -> u32 {
    let (m, n) = nvenc_cur_ver();
    m | (n << 24)
}

/// Probe NVENC API versions newest-first against the driver and remember the highest accepted:
/// the bundled headers are NVENC 13.0 (`pinned`), so a current driver negotiates 13.0 natively
/// while older drivers down-negotiate through 12.x/11.x to 10.0 (~R445). The struct-version words
/// are derived per negotiated version from the revision table, so the 13.0-layout structs are
/// stamped with the exact word each older SDK defined. Set-once per process.
fn nvenc_negotiate(lib: &NvencLibrary) {
    NVENC_NEG_VER.get_or_init(|| {
        let pinned = (NVENCAPI_VERSION & 0xFF, (NVENCAPI_VERSION >> 24) & 0xFF);
        let mut drv_max: u32 = 0; // (major<<4)|minor; 0 = unknown -> rely on createInstance probing
        if let Some(get_max) = lib.get_max_version {
            let mut m: u32 = 0;
            if unsafe { get_max(&mut m) } == NVENCSTATUS::NV_ENC_SUCCESS {
                drv_max = m;
            }
        }
        // Optional cap for testing/pinning a lower version, e.g. PIXELFLUX_NVENC_MAX_API="11.0".
        if let Ok(cap) = std::env::var("PIXELFLUX_NVENC_MAX_API") {
            let mut it = cap.split('.');
            if let (Some(a), Some(b)) = (it.next(), it.next()) {
                if let (Ok(cm), Ok(cn)) = (a.parse::<u32>(), b.parse::<u32>()) {
                    let capv = (cm << 4) | (cn & 0xF);
                    if capv != 0 && (drv_max == 0 || capv < drv_max) {
                        drv_max = capv;
                    }
                }
            }
        }
        let candidates = [pinned, (12, 1), (12, 0), (11, 1), (11, 0), (10, 0)];
        for (maj, min) in candidates {
            let vv = (maj << 4) | min;
            if drv_max != 0 && vv > drv_max {
                continue; // driver can't support this version; skip
            }
            let mut probe = NV_ENCODE_API_FUNCTION_LIST {
                version: nvenc_struct_ver(NvStruct::FunctionList, maj, min),
                ..Default::default()
            };
            let st = unsafe { (lib.create_instance)(&mut probe) };
            // Require every entry point the encode path unwrap()s, not just the session opener:
            // a driver may accept the function-list word yet leave newer entries null.
            if st == NVENCSTATUS::NV_ENC_SUCCESS
                && probe.nvEncOpenEncodeSessionEx.is_some()
                && probe.nvEncInitializeEncoder.is_some()
                && probe.nvEncGetEncodePresetConfigEx.is_some()
                && probe.nvEncEncodePicture.is_some()
                && probe.nvEncLockBitstream.is_some()
            {
                eprintln!("[pixelflux] NVENC API version negotiated: {}.{}", maj, min);
                return (maj, min);
            }
        }
        pinned
    });
}

/// @brief Cache entry for repeated DMABuf imports.
struct CachedDmaBuf {
    egl_image: EGLImageKHR,
    cuda_resource: CUgraphicsResource,
    egl_frame: CUeglFrame,
}

const NV_ENC_H264_PROFILE_HIGH_GUID: GUID = GUID {
    Data1: 0x205b553d,
    Data2: 0x5f01,
    Data3: 0x4d9e,
    Data4: [0x91, 0x84, 0xda, 0x32, 0x77, 0x5b, 0x55, 0x9b],
};

const NV_ENC_H264_PROFILE_HIGH_444_GUID: GUID = GUID {
    Data1: 0x7ac663cb,
    Data2: 0xa598,
    Data3: 0x4960,
    Data4: [0xb8, 0x44, 0x33, 0x9b, 0x26, 0x1a, 0x7d, 0x5c],
};

/// @brief Manages the NVENC H.264 encoder session and CUDA interop resources.
///
/// Handles initialization of CUDA contexts, loading of dynamic libraries,
/// management of input buffers (both DMABuf and Raw), and the encoding loop.
pub struct NvencEncoder {
    encoder_session: *mut c_void,
    cuda_context: CUcontext,
    egl_display: EGLDisplay,
    width: u32,
    height: u32,
    current_qp: u32,
    encode_config: NV_ENC_CONFIG,
    init_params: NV_ENC_INITIALIZE_PARAMS,
    input_device_ptr: CUdeviceptr,
    input_pitch: usize,
    registered_input_resource: NV_ENC_REGISTERED_PTR,
    mapped_input_buffer: NV_ENC_INPUT_PTR,
    nv12_device_ptr: Option<CUdeviceptr>,
    nv12_pitch: usize,
    nv12_registered_resource: Option<NV_ENC_REGISTERED_PTR>,
    nv12_mapped_buffer: Option<NV_ENC_INPUT_PTR>,
    bitstream_buffers: Vec<NV_ENC_OUTPUT_PTR>,
    current_buffer_idx: usize,
    dmabuf_cache: HashMap<i32, CachedDmaBuf>,
    // Page-locked host upload sources: base ptr -> registered len (0 = registration failed).
    pinned_hosts: HashMap<usize, usize>,
    cuda: Arc<CudaFunctions>,
    egl: Arc<EglFunctions>,
    _nvenc_lib: Arc<NvencLibrary>,
    nvenc_funcs: NV_ENCODE_API_FUNCTION_LIST,
    recording_sink: Option<Arc<RecordingSink>>,
    omit_stripe_headers: bool,
    // Effective CUDA device this session is bound to; a reuse across captures must
    // rebuild when the caller now targets a different device.
    node_index: i32,
}

unsafe impl Send for NvencEncoder {}

/// @brief Clean up GPU resources on drop.
///
/// Unregisters resources, frees CUDA memory, destroys the encoder session,
/// and cleans up the CUDA context.
impl Drop for NvencEncoder {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);

            if !self.mapped_input_buffer.is_null() {
                (self.nvenc_funcs.nvEncUnmapInputResource.unwrap())(
                    self.encoder_session,
                    self.mapped_input_buffer,
                );
            }
            if !self.registered_input_resource.is_null() {
                (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                    self.encoder_session,
                    self.registered_input_resource,
                );
            }
            if self.input_device_ptr != 0 {
                (self.cuda.cuMemFree_v2)(self.input_device_ptr);
            }

            if let Some(mapped) = self.nv12_mapped_buffer {
                (self.nvenc_funcs.nvEncUnmapInputResource.unwrap())(
                    self.encoder_session,
                    mapped,
                );
            }
            if let Some(registered) = self.nv12_registered_resource {
                (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                    self.encoder_session,
                    registered,
                );
            }
            if let Some(ptr) = self.nv12_device_ptr {
                (self.cuda.cuMemFree_v2)(ptr);
            }

            for &bs in &self.bitstream_buffers {
                (self.nvenc_funcs.nvEncDestroyBitstreamBuffer.unwrap())(
                    self.encoder_session,
                    bs,
                );
            }

            for (_, cache) in self.dmabuf_cache.drain() {
                (self.cuda.cuGraphicsUnregisterResource)(cache.cuda_resource);
                (self.egl.eglDestroyImageKHR)(self.egl_display, cache.egl_image);
            }

            for (base, len) in &self.pinned_hosts {
                if *len > 0 {
                    (self.cuda.cuMemHostUnregister)(*base as *mut c_void);
                }
            }

            if !self.encoder_session.is_null() {
                (self.nvenc_funcs.nvEncDestroyEncoder.unwrap())(self.encoder_session);
            }

            (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
            (self.cuda.cuCtxDestroy_v2)(self.cuda_context);
        }
    }
}

/// Minimum H.264 level (nvcodec_sys numeric values: 52/60/61/62) whose MaxFS
/// (frame size in MBs) and MaxMBPS (MB rate) fit this resolution+fps per H.264
/// Annex-A Table A-1, floored at 5.2. The level table starts at 5.2 (no 5.1
/// entry) so every resolution up to 4K resolves deterministically to High@5.2.
/// A flat hardcoded 5.2 fails NVENC init above 4K (5.2 MaxFS=36864 MBs ~= 4096x2304),
/// so only >4K steps UP to 6.0/6.1/6.2. We pick the LOWEST fitting level at or above
/// the 5.2 floor so the SPS advertises the smallest level a decoder must support,
/// while the level is fixed from frame 1 (profile stays High) -- no mid-stream bump.
fn min_h264_level(width: u32, height: u32, fps: u32) -> u32 {
    // Frame size in macroblocks and the per-second MB rate.
    let mbs = (width as u64).div_ceil(16) * (height as u64).div_ceil(16);
    let mbps = mbs * fps.max(1) as u64;
    // (numeric level, MaxFS, MaxMBPS) ascending; floored at 5.2 (no 5.1 entry) so typical
    // <=4K streams resolve to a single deterministic High@5.2.
    const LEVELS: [(u32, u64, u64); 4] = [
        (52, 36864, 2073600),   // 5.2 (floor)
        (60, 139264, 4177920),  // 6.0
        (61, 139264, 8355840),  // 6.1
        (62, 139264, 16711680), // 6.2
    ];
    for &(level, max_fs, max_mbps) in &LEVELS {
        if mbs <= max_fs && mbps <= max_mbps {
            return level;
        }
    }
    62 // Above 6.2's limits NVENC has no higher level; best effort.
}

impl NvencEncoder {
    /// @brief Loads the EGL library and required extensions.
    /// @return Result containing the loaded EGL function table.
    fn load_egl() -> Result<EglFunctions, String> {
        unsafe {
            let lib_name = "libEGL.so.1";
            let lib = Library::new(lib_name)
                .or_else(|_| Library::new("libEGL.so"))
                .map_err(|e| format!("Could not load EGL library: {}", e))?;

            let get_proc_addr_sym: Symbol<unsafe extern "C" fn(*const c_char) -> *mut c_void> = lib
                .get(b"eglGetProcAddress\0")
                .map_err(|e| format!("Missing symbol eglGetProcAddress: {}", e))?;

            let eglGetProcAddress = *get_proc_addr_sym;

            let load_extension = |name: &str| -> Result<*mut c_void, String> {
                let c_name = CString::new(name).unwrap();
                let addr = eglGetProcAddress(c_name.as_ptr());
                if addr.is_null() {
                    Err(format!("EGL Extension not found: {}", name))
                } else {
                    Ok(addr)
                }
            };

            let create_addr = load_extension("eglCreateImageKHR")?;
            let destroy_addr = load_extension("eglDestroyImageKHR")?;

            Ok(EglFunctions {
                _lib: lib,
                eglGetProcAddress,
                eglCreateImageKHR: std::mem::transmute::<*mut c_void, EglCreateImageKhrFn>(create_addr),
                eglDestroyImageKHR: std::mem::transmute::<*mut c_void, EglDestroyImageKhrFn>(destroy_addr),
            })
        }
    }

    /// @brief Loads the CUDA library and core symbols.
    /// @return Result containing the loaded CUDA function table.
    fn load_cuda() -> Result<CudaFunctions, String> {
        unsafe {
            let lib_name = if cfg!(windows) {
                "nvcuda.dll"
            } else {
                "libcuda.so.1"
            };
            let lib = Library::new(lib_name)
                .map_err(|e| format!("Could not load CUDA library ({}): {}", lib_name, e))?;

            macro_rules! load {
                ($lib:expr, $name:expr) => {
                    *$lib.get($name).map_err(|e| {
                        format!(
                            "Missing symbol {}: {}",
                            std::str::from_utf8($name).unwrap(),
                            e
                        )
                    })?
                };
            }

            Ok(CudaFunctions {
                cuInit: load!(lib, b"cuInit\0"),
                cuDeviceGet: load!(lib, b"cuDeviceGet\0"),
                cuDeviceGetByPCIBusId: load!(lib, b"cuDeviceGetByPCIBusId\0"),
                cuCtxCreate_v2: load!(lib, b"cuCtxCreate_v2\0"),
                cuCtxPushCurrent_v2: load!(lib, b"cuCtxPushCurrent_v2\0"),
                cuCtxPopCurrent_v2: load!(lib, b"cuCtxPopCurrent_v2\0"),
                cuCtxDestroy_v2: load!(lib, b"cuCtxDestroy_v2\0"),
                cuMemAlloc_v2: load!(lib, b"cuMemAlloc_v2\0"),
                cuMemAllocPitch_v2: load!(lib, b"cuMemAllocPitch_v2\0"),
                cuMemFree_v2: load!(lib, b"cuMemFree_v2\0"),
                cuMemcpyHtoD_v2: load!(lib, b"cuMemcpyHtoD_v2\0"),
                cuMemcpyDtoH_v2: load!(lib, b"cuMemcpyDtoH_v2\0"),
                cuMemcpy2D_v2: load!(lib, b"cuMemcpy2D_v2\0"),
                cuMemHostRegister_v2: load!(lib, b"cuMemHostRegister_v2\0"),
                cuMemHostUnregister: load!(lib, b"cuMemHostUnregister\0"),
                cuGraphicsEGLRegisterImage: load!(lib, b"cuGraphicsEGLRegisterImage\0"),
                cuGraphicsUnregisterResource: load!(lib, b"cuGraphicsUnregisterResource\0"),
                cuGraphicsResourceGetMappedEglFrame: load!(
                    lib,
                    b"cuGraphicsResourceGetMappedEglFrame\0"
                ),
                cuDeviceGetCount: load!(lib, b"cuDeviceGetCount\0"),
                cuDeviceGetName: load!(lib, b"cuDeviceGetName\0"),
                cuDeviceGetUuid: load!(lib, b"cuDeviceGetUuid\0"),
                cuGetErrorName: load!(lib, b"cuGetErrorName\0"),
                _lib: lib,
            })
        }
    }

    /// @brief Loads the NVENC API library.
    /// @return Result containing the loaded NVENC library wrapper.
    fn load_nvenc() -> Result<NvencLibrary, String> {
        unsafe {
            let lib_name = NVENC_DLL_NAME;
            let lib = Library::new(lib_name)
                .map_err(|e| format!("Could not load NVENC library ({}): {}", lib_name, e))?;

            let create_instance = *lib
                .get(NV_ENCODE_API_CREATE_INSTANCE_FN_NAME)
                .map_err(|e| e.to_string())?;
            let get_max_version = lib
                .get::<NvEncodeApiGetMaxSupportedVersionFn>(
                    NV_ENCODE_API_GET_MAX_SUPPORTED_VERSION_FN_NAME,
                )
                .map(|s| *s)
                .ok();
            Ok(NvencLibrary {
                create_instance,
                get_max_version,
                _lib: lib,
            })
        }
    }

    /// @brief Helper to convert CUDA error codes to strings.
    unsafe fn get_error_string(cuda: &CudaFunctions, err: CUresult) -> String {
        let mut p_str: *const c_char = ptr::null();
        if (cuda.cuGetErrorName)(err, &mut p_str) == CUresult::CUDA_SUCCESS && !p_str.is_null() {
            CStr::from_ptr(p_str).to_string_lossy().into_owned()
        } else {
            format!("Unknown CUDA Error ({})", err.0)
        }
    }

    /// @brief Enumerates and prints available CUDA devices for debugging.
    unsafe fn probe_devices(cuda: &CudaFunctions) {
        let mut count = 0;
        if (cuda.cuDeviceGetCount)(&mut count) != CUresult::CUDA_SUCCESS {
            return;
        }
        println!("[NVENC] Found {} CUDA devices:", count);
        for i in 0..count {
            let mut dev = 0;
            (cuda.cuDeviceGet)(&mut dev, i);
            let mut name_buf = [0 as c_char; 256];
            (cuda.cuDeviceGetName)(name_buf.as_mut_ptr(), 256, dev);
            let name = CStr::from_ptr(name_buf.as_ptr()).to_string_lossy();
            println!("[NVENC]   Device {}: {}", i, name);
        }
    }

    /// @brief Retrieves the physical PCI Bus ID for a given DRM render node index.
    fn get_pci_bus_id(render_index: i32) -> Option<String> {
        let path = format!("/sys/class/drm/renderD{}/device", 128 + render_index);
        if let Ok(target) = std::fs::read_link(&path) {
            if let Some(name) = target.file_name() {
                if let Some(name_str) = name.to_str() {
                    return Some(name_str.to_string());
                }
            }
        }
        None
    }

    /// @brief Initializes the NVENC encoder, CUDA context, and primary resources.
    /// @input settings: Capture settings (resolution, FPS, QP).
    /// @input egl_display: The EGL display handle for interop.
    /// @return Result containing the initialized NvencEncoder instance.
    pub fn new(
        settings: &RustCaptureSettings,
        egl_display: *const c_void,
        recording_sink: Option<Arc<RecordingSink>>,
    ) -> Result<Self, String> {
        println!("[NVENC] Initializing...");

        let egl = Arc::new(Self::load_egl()?);
        let cuda = Arc::new(Self::load_cuda()?);
        let nvenc_lib = Arc::new(Self::load_nvenc()?);
        // Negotiate the NVENC API version against the driver (set-once) before tagging structs.
        nvenc_negotiate(&nvenc_lib);

        // libcuda + libnvidia-encode are now loaded: install the multi-GPU GET_ATTACHED_IDS ioctl
        // filter before cuInit enumerates devices (a no-op unless a host GPU is hidden from this
        // container). Must run AFTER the NVIDIA libs are dlopened so their GOTs can be patched.
        crate::nvgpufilter::install();

        static LEAK_ONCE: std::sync::Once = std::sync::Once::new();
        LEAK_ONCE.call_once(|| {
            std::mem::forget(egl.clone());
            std::mem::forget(cuda.clone());
            std::mem::forget(nvenc_lib.clone());
        });

        unsafe {
            let res = (cuda.cuInit)(0);
            if res != CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "Init CUDA failed: {}",
                    Self::get_error_string(&cuda, res)
                ));
            }

            Self::probe_devices(&cuda);

            let mut cu_device: CUdevice = 0;
            let mut device_found = false;

            // Effective encoder device: auto (<0) means device 0.
            if let Some(pci_bus_id) = Self::get_pci_bus_id(settings.encode_node_index.max(0)) {
                let c_pci_bus_id = CString::new(pci_bus_id.clone()).unwrap();
                if (cuda.cuDeviceGetByPCIBusId)(&mut cu_device, c_pci_bus_id.as_ptr()) == CUresult::CUDA_SUCCESS {
                    println!("[NVENC] Bound to CUDA device via PCI Bus ID: {}", pci_bus_id);
                    device_found = true;
                }
            }

            if !device_found {
                let res = (cuda.cuDeviceGet)(&mut cu_device, 0);
                if res != CUresult::CUDA_SUCCESS {
                    return Err("Failed to get default CUDA device".into());
                }
            }

            let mut cu_context: CUcontext = ptr::null_mut();
            let res = (cuda.cuCtxCreate_v2)(&mut cu_context, 0, cu_device);
            if res != CUresult::CUDA_SUCCESS {
                return Err("Failed to create CUDA Context".into());
            }

            let width = settings.width as u32;
            let height = settings.height as u32;
            let mut input_device_ptr: CUdeviceptr = 0;
            let mut input_pitch: usize = 0;

            let res = (cuda.cuMemAllocPitch_v2)(
                &mut input_device_ptr,
                &mut input_pitch,
                (width * 4) as usize,
                height as usize,
                16,
            );
            if res != CUresult::CUDA_SUCCESS {
                (cuda.cuCtxDestroy_v2)(cu_context);
                return Err("Failed to allocate ARGB input buffer on GPU".into());
            }

            let mut function_list = NV_ENCODE_API_FUNCTION_LIST {
                version: sv(NvStruct::FunctionList),
                ..Default::default()
            };
            if (nvenc_lib.create_instance)(&mut function_list) != NVENCSTATUS::NV_ENC_SUCCESS {
                (cuda.cuMemFree_v2)(input_device_ptr);
                (cuda.cuCtxDestroy_v2)(cu_context);
                return Err("NvEncodeAPICreateInstance failed".into());
            }

            let mut session_params = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
                version: sv(NvStruct::OpenSessionExParams),
                deviceType: NV_ENC_DEVICE_TYPE::NV_ENC_DEVICE_TYPE_CUDA,
                device: cu_context as *mut c_void,
                apiVersion: neg_api(),
                ..Default::default()
            };

            let mut encoder_session: *mut c_void = ptr::null_mut();
            let open_fn = function_list.nvEncOpenEncodeSessionEx.unwrap();
            if open_fn(&mut session_params, &mut encoder_session) != NVENCSTATUS::NV_ENC_SUCCESS {
                // Free the already-allocated ARGB buffer + CUDA context on init failure.
                (cuda.cuMemFree_v2)(input_device_ptr);
                (cuda.cuCtxDestroy_v2)(cu_context);
                return Err("Failed to open NVENC session".into());
            }

            let is_444 = settings.video_fullcolor;
            let profile_guid = if is_444 {
                NV_ENC_H264_PROFILE_HIGH_444_GUID
            } else {
                NV_ENC_H264_PROFILE_HIGH_GUID
            };

            let mut config = NV_ENC_CONFIG {
                version: sv(NvStruct::Config),
                ..Default::default()
            };
            let mut preset_config = NV_ENC_PRESET_CONFIG {
                version: sv(NvStruct::PresetConfig),
                presetCfg: config,
                ..Default::default()
            };

            let get_preset_ex = function_list.nvEncGetEncodePresetConfigEx.unwrap();
            let preset_status = get_preset_ex(
                encoder_session,
                NV_ENC_CODEC_H264_GUID,
                NV_ENC_PRESET_P4_GUID,
                NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
                &mut preset_config,
            );
            if preset_status != NVENCSTATUS::NV_ENC_SUCCESS {
                // Proceed with the (zeroed) default config rather than aborting, but surface WHY
                // the preset lookup failed instead of silently encoding with an empty preset.
                let detail = function_list.nvEncGetLastErrorString.and_then(|f| {
                    let p = f(encoder_session);
                    if p.is_null() {
                        None
                    } else {
                        Some(CStr::from_ptr(p).to_string_lossy().into_owned())
                    }
                });
                eprintln!(
                    "[NVENC] nvEncGetEncodePresetConfigEx failed ({preset_status:?}): {}",
                    detail.as_deref().unwrap_or("no error string")
                );
            }

            config = preset_config.presetCfg;
            // Version the NV_ENC_CONFIG; the embedded rcParams is left with the version the preset
            // fill returns (libnvidia-encode's own clients don't stamp it separately).
            config.version = sv(NvStruct::Config);
            config.profileGUID = profile_guid;
            if settings.video_cbr_mode {
                let bps = (settings.video_bitrate_kbps.max(0) as u32).saturating_mul(1000);
                config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR;
                // Quarter-resolution first pass: tighter per-frame rate adherence (fewer
                // over/undershoots to smooth out) for a small encode cost.
                config.rcParams.multiPass = NV_ENC_MULTI_PASS::NV_ENC_TWO_PASS_QUARTER_RESOLUTION;
                config.rcParams.averageBitRate = bps;
                config.rcParams.maxBitRate = bps;
                config.rcParams.vbvBufferSize = crate::encoders::vbv_bits(
                    bps,
                    settings.target_fps,
                    settings.keyframe_interval_s,
                    settings.video_vbv_multiplier,
                );
                // Optional RC clamp: max = legibility floor, min = waste ceiling.
                if settings.video_min_qp > 0 {
                    let q = settings.video_min_qp.min(51) as u32;
                    config.rcParams.set_enableMinQP(1);
                    config.rcParams.minQP.qpInterP = q;
                    config.rcParams.minQP.qpInterB = q;
                    config.rcParams.minQP.qpIntra = q;
                }
                if settings.video_max_qp > 0 {
                    let q = settings.video_max_qp.min(51) as u32;
                    config.rcParams.set_enableMaxQP(1);
                    config.rcParams.maxQP.qpInterP = q;
                    config.rcParams.maxQP.qpInterB = q;
                    config.rcParams.maxQP.qpIntra = q;
                }
            } else {
                config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CONSTQP;
                config.rcParams.constQP.qpInterP = settings.video_crf as u32;
                config.rcParams.constQP.qpInterB = settings.video_crf as u32;
                config.rcParams.constQP.qpIntra = settings.video_crf as u32;
            }
            config.frameIntervalP = 1;
            config.gopLength = 0xFFFFFFFF;
            // No-B-frame stream: emit pictures in decode order with no DPB output lag and
            // say so in the SPS (bitstream_restriction: max_num_reorder_frames=0), so
            // VideoToolbox/MediaCodec-class decoders don't buffer frames "just in case".
            // ffmpeg-class decoders already infer this from pic_order_cnt_type=2.
            config.rcParams.set_zeroReorderDelay(1);
            config.encodeCodecConfig.h264Config.h264VUIParameters.bitstreamRestrictionFlag = 1;
            // Pin an explicit H.264 level + idrPeriod so the very first access unit already
            // declares the final High profile at a deterministic level. Leaving level at
            // AUTOSELECT lets early frames advertise a lower level; when NVENC later bumps it
            // mid-stream, Windows Chromium's D3D11VideoDecoder (and WebCodecs) must re-init and
            // drops frames. Compute the minimum Annex-A level for this resolution+fps, floored
            // at 5.2: <=4K stays High@5.2; >4K needs 6.0/6.1/6.2 (else NVENC init fails).
            // idrPeriod matches the infinite GOP set above.
            config.encodeCodecConfig.h264Config.level =
                min_h264_level(width, height, settings.target_fps as u32);
            config.encodeCodecConfig.h264Config.idrPeriod = 0xFFFFFFFF;
            config.encodeCodecConfig.h264Config.h264VUIParameters.videoSignalTypePresentFlag = 1;
            // Signal the colorimetry in the VUI: video format unspecified, BT.709 primaries/
            // transfer/matrix, matching the BT.709 the encoder produces.
            config.encodeCodecConfig.h264Config.h264VUIParameters.videoFormat =
                NV_ENC_VUI_VIDEO_FORMAT::NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
            config.encodeCodecConfig.h264Config.h264VUIParameters.colourDescriptionPresentFlag = 1;
            config.encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries =
                NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709;
            config.encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics =
                NV_ENC_VUI_TRANSFER_CHARACTERISTIC::NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
            config.encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix =
                NV_ENC_VUI_MATRIX_COEFFS::NV_ENC_VUI_MATRIX_COEFFS_BT709;
            config.encodeCodecConfig.h264Config.chromaFormatIDC = if is_444 { 3 } else { 1 };
            config.encodeCodecConfig.h264Config.h264VUIParameters.videoFullRangeFlag =
                if is_444 { 1 } else { 0 };
            config.encodeCodecConfig.h264Config.set_repeatSPSPPS(1);
            // Pin the entropy coder and access-unit delimiters explicitly instead of
            // inheriting them from the preset table: CABAC for High-profile efficiency,
            // no AUD (browsers and WebCodecs don't need the delimiters).
            config.encodeCodecConfig.h264Config.entropyCodingMode =
                NV_ENC_H264_ENTROPY_CODING_MODE::NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;
            config.encodeCodecConfig.h264Config.set_outputAUD(0);
            // Minimize GOP-to-GOP rate fluctuation; harmless on the infinite-GOP default.
            config.rcParams.set_strictGOPTarget(1);
            // No lookahead: real-time low latency must not depend on the preset default.
            config.rcParams.set_enableLookahead(0);
            config.rcParams.lookaheadDepth = 0;

            let mut init_params = NV_ENC_INITIALIZE_PARAMS {
                version: sv(NvStruct::InitializeParams),
                encodeGUID: NV_ENC_CODEC_H264_GUID,
                presetGUID: NV_ENC_PRESET_P4_GUID,
                tuningInfo: NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
                encodeWidth: width,
                encodeHeight: height,
                darWidth: width,
                darHeight: height,
                frameRateNum: settings.target_fps as u32,
                frameRateDen: 1,
                enablePTD: 1,
                encodeConfig: &mut config,
                // In-place resize headroom: NvEncReconfigureEncoder only accepts new
                // dimensions up to these, so give every session the H.264 level-5.2
                // ceiling (the same floor min_h264_level pins) or the initial size if
                // larger. Resizes beyond this fall back to a session rebuild.
                maxEncodeWidth: width.max(4096),
                maxEncodeHeight: height.max(2304),
                ..Default::default()
            };

            let init_fn = function_list.nvEncInitializeEncoder.unwrap();
            if init_fn(encoder_session, &mut init_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                // The headroom costs real device memory (~290 MiB at 4096x2304); a rig
                // that can't afford it still gets a working session -- resizes beyond
                // the exact dimensions then fall back to a session rebuild.
                init_params.maxEncodeWidth = width;
                init_params.maxEncodeHeight = height;
                if init_fn(encoder_session, &mut init_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                    // Tear down session + ARGB buffer + CUDA context on init failure.
                    (function_list.nvEncDestroyEncoder.unwrap())(encoder_session);
                    (cuda.cuMemFree_v2)(input_device_ptr);
                    (cuda.cuCtxDestroy_v2)(cu_context);
                    return Err("Failed to initialize encoder".into());
                }
                eprintln!("[NVENC] Init with resize headroom failed; running without it.");
            }

            // null the pointer to the soon-to-be-moved local `config`; reconfigure repoints it.
            init_params.encodeConfig = ptr::null_mut();

            let mut reg_res = NV_ENC_REGISTER_RESOURCE {
                version: sv(NvStruct::RegisterResource),
                resourceType: NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                width,
                height,
                resourceToRegister: input_device_ptr as *mut c_void,
                pitch: input_pitch as u32,
                bufferFormat: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
                bufferUsage: NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE,
                ..Default::default()
            };

            let register_fn = function_list.nvEncRegisterResource.unwrap();
            if register_fn(encoder_session, &mut reg_res) != NVENCSTATUS::NV_ENC_SUCCESS {
                // Registration failed (nothing to unregister): tear down session + buffer + context.
                (function_list.nvEncDestroyEncoder.unwrap())(encoder_session);
                (cuda.cuMemFree_v2)(input_device_ptr);
                (cuda.cuCtxDestroy_v2)(cu_context);
                return Err("Failed to register input buffer".into());
            }

            let mut map_params = NV_ENC_MAP_INPUT_RESOURCE {
                version: sv(NvStruct::MapInputResource),
                registeredResource: reg_res.registeredResource,
                ..Default::default()
            };
            let map_fn = function_list.nvEncMapInputResource.unwrap();
            if map_fn(encoder_session, &mut map_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                // Map failed: unregister the resource, then tear down session + buffer + context.
                (function_list.nvEncUnregisterResource.unwrap())(
                    encoder_session,
                    reg_res.registeredResource,
                );
                (function_list.nvEncDestroyEncoder.unwrap())(encoder_session);
                (cuda.cuMemFree_v2)(input_device_ptr);
                (cuda.cuCtxDestroy_v2)(cu_context);
                return Err("Failed to map input buffer".into());
            }

            let mut bitstream_buffers = Vec::new();
            let create_bs_fn = function_list.nvEncCreateBitstreamBuffer.unwrap();
            for _ in 0..4 {
                let mut bitstream_params = NV_ENC_CREATE_BITSTREAM_BUFFER {
                    version: sv(NvStruct::CreateBitstreamBuffer),
                    ..Default::default()
                };
                if create_bs_fn(encoder_session, &mut bitstream_params)
                    != NVENCSTATUS::NV_ENC_SUCCESS
                {
                    // Destroy any bitstream buffers made so far, unmap + unregister the input,
                    // then tear down session + ARGB buffer + context.
                    for &bs in &bitstream_buffers {
                        (function_list.nvEncDestroyBitstreamBuffer.unwrap())(encoder_session, bs);
                    }
                    (function_list.nvEncUnmapInputResource.unwrap())(
                        encoder_session,
                        map_params.mappedResource,
                    );
                    (function_list.nvEncUnregisterResource.unwrap())(
                        encoder_session,
                        reg_res.registeredResource,
                    );
                    (function_list.nvEncDestroyEncoder.unwrap())(encoder_session);
                    (cuda.cuMemFree_v2)(input_device_ptr);
                    (cuda.cuCtxDestroy_v2)(cu_context);
                    return Err("Failed to create bitstream buffer".into());
                }
                bitstream_buffers.push(bitstream_params.bitstreamBuffer);
            }

            println!("[NVENC] Initialized successfully (4:4:4 mode: {}).", is_444);

            Ok(Self {
                encoder_session,
                cuda_context: cu_context,
                egl_display: egl_display as EGLDisplay,
                width,
                height,
                current_qp: settings.video_crf as u32,
                encode_config: config,
                init_params,
                input_device_ptr,
                input_pitch,
                registered_input_resource: reg_res.registeredResource,
                mapped_input_buffer: map_params.mappedResource,
                nv12_device_ptr: None,
                nv12_pitch: 0,
                nv12_registered_resource: None,
                nv12_mapped_buffer: None,
                bitstream_buffers,
                current_buffer_idx: 0,
                dmabuf_cache: HashMap::new(),
                pinned_hosts: HashMap::new(),
                cuda,
                egl,
                _nvenc_lib: nvenc_lib,
                nvenc_funcs: function_list,
                recording_sink,
                omit_stripe_headers: settings.omit_stripe_headers,
                node_index: settings.encode_node_index.max(0),
            })
        }
    }

    /// Reshape the live session to `settings` without tearing it down: the NVENC session,
    /// CUDA context and bitstream buffers survive, so a resize costs a few milliseconds
    /// instead of a full rebuild. Only geometry-dependent state is replaced (ARGB input
    /// surface; raw-plane buffer, dmabuf imports and pinned hosts are dropped for lazy
    /// re-creation). Also folds in the current rate/QP/fps so one call covers a combined
    /// resize+rate change. The next frame is forced IDR with RC state reset.
    ///
    /// Errs when the target needs what a live session cannot change -- another device,
    /// chroma format (4:4:4), RC mode, or dimensions beyond the init-time headroom --
    /// and on driver rejection; the caller falls back to a rebuild.
    pub fn reconfigure_resolution(&mut self, settings: &RustCaptureSettings) -> Result<(), String> {
        let new_w = settings.width as u32;
        let new_h = settings.height as u32;
        // encodeCodecConfig is a C union; the H.264 arm is the one this encoder fills.
        let is_444 =
            unsafe { self.encode_config.encodeCodecConfig.h264Config.chromaFormatIDC == 3 };
        let is_cbr = self.encode_config.rcParams.rateControlMode
            == NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR;
        if settings.encode_node_index.max(0) != self.node_index {
            return Err("encode device changed".into());
        }
        if settings.video_fullcolor != is_444 {
            return Err("chroma format changed".into());
        }
        if settings.video_cbr_mode != is_cbr {
            return Err("rate-control mode changed".into());
        }
        if new_w == 0
            || new_h == 0
            || new_w > self.init_params.maxEncodeWidth
            || new_h > self.init_params.maxEncodeHeight
        {
            return Err(format!(
                "{}x{} outside reconfigure headroom {}x{}",
                new_w, new_h, self.init_params.maxEncodeWidth, self.init_params.maxEncodeHeight
            ));
        }

        unsafe {
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);
            // Release every geometry-dependent input resource. The raw-plane buffer and
            // dmabuf imports are re-created lazily by their encode paths; the ARGB surface
            // is re-created below. Pinned hosts go too: the source shm segments are
            // recreated on resize and may land on the same base addresses.
            if !self.mapped_input_buffer.is_null() {
                (self.nvenc_funcs.nvEncUnmapInputResource.unwrap())(
                    self.encoder_session,
                    self.mapped_input_buffer,
                );
                self.mapped_input_buffer = ptr::null_mut();
            }
            if !self.registered_input_resource.is_null() {
                (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                    self.encoder_session,
                    self.registered_input_resource,
                );
                self.registered_input_resource = ptr::null_mut();
            }
            if self.input_device_ptr != 0 {
                (self.cuda.cuMemFree_v2)(self.input_device_ptr);
                self.input_device_ptr = 0;
            }
            if let Some(mapped) = self.nv12_mapped_buffer.take() {
                (self.nvenc_funcs.nvEncUnmapInputResource.unwrap())(self.encoder_session, mapped);
            }
            if let Some(registered) = self.nv12_registered_resource.take() {
                (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                    self.encoder_session,
                    registered,
                );
            }
            if let Some(ptr) = self.nv12_device_ptr.take() {
                (self.cuda.cuMemFree_v2)(ptr);
            }
            self.nv12_pitch = 0;
            for (_, cache) in self.dmabuf_cache.drain() {
                (self.cuda.cuGraphicsUnregisterResource)(cache.cuda_resource);
                (self.egl.eglDestroyImageKHR)(self.egl_display, cache.egl_image);
            }
            for (base, len) in self.pinned_hosts.drain() {
                if len > 0 {
                    (self.cuda.cuMemHostUnregister)(base as *mut c_void);
                }
            }

            // Reconfigure the session: new dimensions, the level they need, and the
            // current rate/QP state, resetting RC and forcing an IDR so the stream
            // restarts cleanly at the new size.
            self.encode_config.encodeCodecConfig.h264Config.level =
                min_h264_level(new_w, new_h, settings.target_fps as u32);
            if is_cbr {
                let bps = (settings.video_bitrate_kbps.max(0) as u32).saturating_mul(1000);
                self.encode_config.rcParams.averageBitRate = bps;
                self.encode_config.rcParams.maxBitRate = bps;
                self.encode_config.rcParams.vbvBufferSize = crate::encoders::vbv_bits(
                    bps,
                    settings.target_fps,
                    settings.keyframe_interval_s,
                    settings.video_vbv_multiplier,
                );
            } else {
                let qp = settings.video_crf as u32;
                self.encode_config.rcParams.constQP.qpInterP = qp;
                self.encode_config.rcParams.constQP.qpInterB = qp;
                self.encode_config.rcParams.constQP.qpIntra = qp;
                self.current_qp = qp;
            }
            self.init_params.encodeWidth = new_w;
            self.init_params.encodeHeight = new_h;
            self.init_params.darWidth = new_w;
            self.init_params.darHeight = new_h;
            self.init_params.frameRateNum = (settings.target_fps.max(1.0)) as u32;
            self.init_params.frameRateDen = 1;
            self.init_params.encodeConfig = &mut self.encode_config;
            let mut reconfig_params = NV_ENC_RECONFIGURE_PARAMS {
                version: sv(NvStruct::ReconfigureParams),
                reInitEncodeParams: self.init_params,
                ..Default::default()
            };
            reconfig_params.set_resetEncoder(1);
            reconfig_params.set_forceIDR(1);
            let reconfig_fn = self.nvenc_funcs.nvEncReconfigureEncoder.unwrap();
            if reconfig_fn(self.encoder_session, &mut reconfig_params)
                != NVENCSTATUS::NV_ENC_SUCCESS
            {
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err("NvEncReconfigureEncoder rejected the resolution change".into());
            }
            self.width = new_w;
            self.height = new_h;

            // New ARGB input surface at the new size, registered and mapped like init.
            let mut input_device_ptr: CUdeviceptr = 0;
            let mut input_pitch: usize = 0;
            let res = (self.cuda.cuMemAllocPitch_v2)(
                &mut input_device_ptr,
                &mut input_pitch,
                (new_w * 4) as usize,
                new_h as usize,
                16,
            );
            if res != CUresult::CUDA_SUCCESS {
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err("Failed to allocate ARGB input buffer on GPU".into());
            }
            let mut reg_res = NV_ENC_REGISTER_RESOURCE {
                version: sv(NvStruct::RegisterResource),
                resourceType: NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                width: new_w,
                height: new_h,
                resourceToRegister: input_device_ptr as *mut c_void,
                pitch: input_pitch as u32,
                bufferFormat: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
                bufferUsage: NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE,
                ..Default::default()
            };
            let register_fn = self.nvenc_funcs.nvEncRegisterResource.unwrap();
            if register_fn(self.encoder_session, &mut reg_res) != NVENCSTATUS::NV_ENC_SUCCESS {
                (self.cuda.cuMemFree_v2)(input_device_ptr);
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err("Failed to register input buffer".into());
            }
            let mut map_params = NV_ENC_MAP_INPUT_RESOURCE {
                version: sv(NvStruct::MapInputResource),
                registeredResource: reg_res.registeredResource,
                ..Default::default()
            };
            let map_fn = self.nvenc_funcs.nvEncMapInputResource.unwrap();
            if map_fn(self.encoder_session, &mut map_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                    self.encoder_session,
                    reg_res.registeredResource,
                );
                (self.cuda.cuMemFree_v2)(input_device_ptr);
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err("Failed to map input buffer".into());
            }
            self.input_device_ptr = input_device_ptr;
            self.input_pitch = input_pitch;
            self.registered_input_resource = reg_res.registeredResource;
            self.mapped_input_buffer = map_params.mappedResource;
            (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
        }
        self.omit_stripe_headers = settings.omit_stripe_headers;
        Ok(())
    }

    /// Drop every pinned-host registration. Called when the capture's shm segments were
    /// recreated at unchanged dimensions: the new segments often reuse the old base
    /// addresses, so stale registrations would alias them. Uploads re-pin lazily.
    pub fn release_pinned_hosts(&mut self) {
        if self.pinned_hosts.is_empty() {
            return;
        }
        unsafe {
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);
            for (base, len) in self.pinned_hosts.drain() {
                if len > 0 {
                    (self.cuda.cuMemHostUnregister)(base as *mut c_void);
                }
            }
            let _ = (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
        }
    }

    /// Swap the recording fan-out for a session kept alive across capture restarts
    /// (each restart rebinds the socket, so the old sink must not be written to).
    pub fn set_recording_sink(&mut self, sink: Option<Arc<RecordingSink>>) {
        self.recording_sink = sink;
    }

    /// @brief Detects if the quantization parameter (QP) has changed and reconfigures the encoder.
    /// @input target_qp: The new desired QP value.
    /// @return bool: True if reconfiguration occurred, false otherwise.
    unsafe fn reconfigure_if_needed(&mut self, target_qp: u32) -> bool {
        // CBR is bitrate-controlled, so QP-based paint-over reconfigures don't apply.
        if self.encode_config.rcParams.rateControlMode
            == NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR
        {
            return false;
        }
        if self.current_qp != target_qp {
            self.encode_config.rcParams.constQP.qpInterP = target_qp;
            self.encode_config.rcParams.constQP.qpInterB = target_qp;
            self.encode_config.rcParams.constQP.qpIntra = target_qp;
            self.init_params.encodeConfig = &mut self.encode_config;

            // No forced IDR on QP changes: a lower-QP P frame refines the static image
            // against the existing reference chain (paint-over) without an intra-frame
            // bitrate spike; the GOP continues seamlessly across the reconfigure.
            let mut reconfig_params = NV_ENC_RECONFIGURE_PARAMS {
                version: sv(NvStruct::ReconfigureParams),
                reInitEncodeParams: self.init_params,
                ..Default::default()
            };

            let reconfig_fn = self.nvenc_funcs.nvEncReconfigureEncoder.unwrap();
            if reconfig_fn(self.encoder_session, &mut reconfig_params)
                == NVENCSTATUS::NV_ENC_SUCCESS
            {
                self.current_qp = target_qp;
                return true;
            } else {
                eprintln!("[NVENC] Reconfigure failed.");
            }
        }
        false
    }

    /// Apply a runtime rate-control / framerate change to the live session: the CBR target
    /// bitrate + VBV (ignored unless CBR is active) and the target fps. Reconfigures only when
    /// something actually changed, so it is cheap to call every frame.
    pub fn reconfigure_rate(&mut self, settings: &RustCaptureSettings) {
        unsafe {
            let mut changed = false;
            if self.encode_config.rcParams.rateControlMode
                == NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR
            {
                let bps = (settings.video_bitrate_kbps.max(0) as u32).saturating_mul(1000);
                let vbv = crate::encoders::vbv_bits(
                    bps,
                    settings.target_fps,
                    settings.keyframe_interval_s,
                    settings.video_vbv_multiplier,
                );
                if self.encode_config.rcParams.averageBitRate != bps
                    || self.encode_config.rcParams.maxBitRate != bps
                    || self.encode_config.rcParams.vbvBufferSize != vbv
                {
                    self.encode_config.rcParams.averageBitRate = bps;
                    self.encode_config.rcParams.maxBitRate = bps;
                    self.encode_config.rcParams.vbvBufferSize = vbv;
                    changed = true;
                }
            }
            let fps = (settings.target_fps.max(1.0)) as u32;
            if self.init_params.frameRateNum != fps {
                self.init_params.frameRateNum = fps;
                self.init_params.frameRateDen = 1;
                changed = true;
            }
            if !changed {
                return;
            }
            self.init_params.encodeConfig = &mut self.encode_config;
            let mut reconfig_params = NV_ENC_RECONFIGURE_PARAMS {
                version: sv(NvStruct::ReconfigureParams),
                reInitEncodeParams: self.init_params,
                ..Default::default()
            };
            let reconfig_fn = self.nvenc_funcs.nvEncReconfigureEncoder.unwrap();
            if reconfig_fn(self.encoder_session, &mut reconfig_params)
                != NVENCSTATUS::NV_ENC_SUCCESS
            {
                eprintln!("[NVENC] Rate reconfigure failed.");
            }
        }
    }

    /// @brief Submits a frame to NVENC, locks the output bitstream, and retrieves the encoded data.
    /// @input mapped_buffer: The CUDA-mapped input resource containing the image.
    /// @input frame_number: Monotonically increasing frame index.
    /// @input force_idr: If true, forces an IDR (Keyframe).
    /// @return Result containing the encoded packet with custom header.
    unsafe fn submit_frame(
        &mut self,
        mapped_buffer: NV_ENC_INPUT_PTR,
        buffer_format: NV_ENC_BUFFER_FORMAT,
        frame_number: u64,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        let output_bitstream = self.bitstream_buffers[self.current_buffer_idx];
        self.current_buffer_idx = (self.current_buffer_idx + 1) % self.bitstream_buffers.len();

        let mut pic_params = NV_ENC_PIC_PARAMS {
            version: sv(NvStruct::PicParams),
            inputWidth: self.width,
            inputHeight: self.height,
            inputBuffer: mapped_buffer,
            outputBitstream: output_bitstream,
            bufferFmt: buffer_format,
            pictureStruct: NV_ENC_PIC_STRUCT::NV_ENC_PIC_STRUCT_FRAME,
            encodePicFlags: if force_idr {
                NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_FORCEIDR as u32
            } else {
                0
            },
            ..Default::default()
        };

        let encode_fn = self.nvenc_funcs.nvEncEncodePicture.unwrap();
        let res = encode_fn(self.encoder_session, &mut pic_params);
        if res != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("Encode Picture failed: {:?}", res));
        }

        let mut lock_params = NV_ENC_LOCK_BITSTREAM {
            version: sv(NvStruct::LockBitstream),
            outputBitstream: output_bitstream,
            ..Default::default()
        };
        lock_params.set_doNotWait(0);

        let lock_fn = self.nvenc_funcs.nvEncLockBitstream.unwrap();
        if lock_fn(self.encoder_session, &mut lock_params) != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err("Lock Bitstream failed".into());
        }

        let data_ptr = lock_params.bitstreamBufferPtr as *const u8;
        let data_size = lock_params.bitstreamSizeInBytes as usize;
        let header_sz = if self.omit_stripe_headers { 0 } else { 10 };
        let mut output = Vec::with_capacity(header_sz + data_size);

        if !self.omit_stripe_headers {
            // Derive the type byte from the ACTUAL encoded picture type
            // (IDR=0x01, I=0x02, P=0x00), not from the force_idr request.
            let type_hdr = match lock_params.pictureType {
                NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_IDR => 0x01u8,
                NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_I => 0x02u8,
                _ => 0x00u8,
            };
            output.push(0x04);
            output.push(type_hdr);
            output.extend_from_slice(&(frame_number as u16).to_be_bytes());
            output.extend_from_slice(&0u16.to_be_bytes());
            output.extend_from_slice(&(self.width as u16).to_be_bytes());
            output.extend_from_slice(&(self.height as u16).to_be_bytes());
        }

        if data_size > 0 && !data_ptr.is_null() {
            let slice = std::slice::from_raw_parts(data_ptr, data_size);
            output.extend_from_slice(slice);
            if let Some(ref sink) = self.recording_sink {
                sink.write_frame(slice);
            }
        }

        (self.nvenc_funcs.nvEncUnlockBitstream.unwrap())(self.encoder_session, output_bitstream);
        Ok(output)
    }

    /// @brief Encodes a single DMABuf frame by importing it via EGL and mapping it to CUDA.
    /// @input dmabuf: The source Linux DMA buffer.
    /// @input frame_number: Frame index.
    /// @input target_qp: Desired quality parameter.
    /// @input force_idr: Force keyframe generation.
    /// @return Result containing encoded byte vector.
    pub fn encode(
        &mut self,
        dmabuf: &Dmabuf,
        frame_number: u64,
        target_qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.reconfigure_if_needed(target_qp);
            // Extract fd before pushing the context so the `?` can't return with
            // the context left pushed (stack imbalance).
            let fd = dmabuf.handles().next().ok_or("No handles")?.as_raw_fd();
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);

            if !self.dmabuf_cache.contains_key(&fd) {
                let stride = dmabuf.strides().next().unwrap_or(0) as i32;
                let offset = dmabuf.offsets().next().unwrap_or(0) as i32;
                let fmt = dmabuf.format();
                let modifier: u64 = fmt.modifier.into();

                let attribs = [
                    EGL_WIDTH,
                    self.width as i32,
                    EGL_HEIGHT,
                    self.height as i32,
                    EGL_LINUX_DRM_FOURCC_EXT,
                    fmt.code as i32,
                    EGL_DMA_BUF_PLANE0_FD_EXT,
                    fd,
                    EGL_DMA_BUF_PLANE0_OFFSET_EXT,
                    offset,
                    EGL_DMA_BUF_PLANE0_PITCH_EXT,
                    stride,
                    EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                    (modifier & 0xFFFFFFFF) as i32,
                    EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                    (modifier >> 32) as i32,
                    EGL_NONE,
                ];

                let egl_image = (self.egl.eglCreateImageKHR)(
                    self.egl_display,
                    ptr::null_mut(),
                    EGL_LINUX_DMA_BUF_EXT,
                    ptr::null_mut(),
                    attribs.as_ptr(),
                );
                if egl_image == EGL_NO_IMAGE_KHR {
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to create EGLImage".into());
                }

                let mut cuda_resource: CUgraphicsResource = ptr::null_mut();
                if (self.cuda.cuGraphicsEGLRegisterImage)(&mut cuda_resource, egl_image, 1)
                    != CUresult::CUDA_SUCCESS
                {
                    (self.egl.eglDestroyImageKHR)(self.egl_display, egl_image);
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to register EGLImage".into());
                }

                let mut egl_frame: CUeglFrame = std::mem::zeroed();
                if (self.cuda.cuGraphicsResourceGetMappedEglFrame)(
                    &mut egl_frame,
                    cuda_resource,
                    0,
                    0,
                ) != CUresult::CUDA_SUCCESS
                {
                    (self.cuda.cuGraphicsUnregisterResource)(cuda_resource);
                    (self.egl.eglDestroyImageKHR)(self.egl_display, egl_image);
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to map EGL frame".into());
                }

                self.dmabuf_cache.insert(
                    fd,
                    CachedDmaBuf {
                        egl_image,
                        cuda_resource,
                        egl_frame,
                    },
                );
            }

            let cached = self.dmabuf_cache.get(&fd).unwrap();
            let mut copy_params = CUDA_MEMCPY2D {
                srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcHost: ptr::null(),
                srcDevice: 0,
                srcArray: ptr::null_mut(),
                srcPitch: 0,
                dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstHost: ptr::null_mut(),
                dstDevice: self.input_device_ptr,
                dstArray: ptr::null_mut(),
                dstPitch: self.input_pitch,
                WidthInBytes: (self.width * 4) as usize,
                Height: self.height as usize,
                ..Default::default()
            };

            if cached.egl_frame.frame_type == 0 {
                copy_params.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
                copy_params.srcArray = cached.egl_frame.frame.p_array[0];
            } else {
                copy_params.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
                copy_params.srcDevice = cached.egl_frame.frame.p_pitch[0] as CUdeviceptr;
                copy_params.srcPitch = cached.egl_frame.pitch as usize;
            }

            if (self.cuda.cuMemcpy2D_v2)(&copy_params) != CUresult::CUDA_SUCCESS {
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err("Sanitization copy failed".into());
            }

            let result = self.submit_frame(
                self.mapped_input_buffer,
                NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
                frame_number,
                force_idr,
            );
            (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
            result
        }
    }

    /// @brief Encodes a host ARGB frame directly, with no explicit ARGB->NV12 conversion.
    ///
    /// Uploads the packed ARGB rows straight into the registered ARGB input surface
    /// and lets NVENC's hardware CSC produce YUV. Bytes must be in NVENC ARGB order
    /// (B,G,R,A in memory), i.e. the host BGRA layout an XShm grab produces.
    /// `src_stride` is the source row stride in bytes (>= width*4).
    /// @input argb: Host pixel buffer (height rows of width*4 at src_stride).
    /// @input src_stride: Source row stride in bytes.
    /// @input frame_number: Frame index.
    /// @input target_qp: Desired quality parameter.
    /// @input force_idr: Force keyframe generation.
    /// @return Result containing encoded byte vector.
    pub fn encode_cpu_argb(
        &mut self,
        argb: &[u8],
        src_stride: usize,
        frame_number: u64,
        target_qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.reconfigure_if_needed(target_qp);
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);

            let width_bytes = (self.width * 4) as usize;
            let rows = self.height as usize;
            // Source must hold `rows` lines of width*4 bytes at src_stride.
            let needed = if rows == 0 { 0 } else { src_stride * (rows - 1) + width_bytes };
            if src_stride < width_bytes || argb.len() < needed {
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err(format!(
                    "ARGB buffer too small: len={} need>={} (stride={}, {}x{})",
                    argb.len(), needed, src_stride, self.width, self.height
                ));
            }

            // Page-lock the persistent, bounded shm source once so cuMemcpy2D becomes a direct
            // pinned DMA instead of a pageable copy staged through a driver bounce buffer.
            if std::env::var("PIXELFLUX_NVENC_PIN").as_deref() != Ok("0") {
                let base = argb.as_ptr() as usize;
                if let std::collections::hash_map::Entry::Vacant(e) = self.pinned_hosts.entry(base) {
                    let st = (self.cuda.cuMemHostRegister_v2)(argb.as_ptr() as *mut c_void, argb.len(), 0);
                    // 0 sentinel: registration failed -> stay pageable, never re-probe.
                    e.insert(if st == CUresult::CUDA_SUCCESS { argb.len() } else { 0 });
                }
            }

            let copy = CUDA_MEMCPY2D {
                srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                srcHost: argb.as_ptr() as *const c_void,
                srcPitch: src_stride,
                dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstDevice: self.input_device_ptr,
                dstPitch: self.input_pitch,
                WidthInBytes: width_bytes,
                Height: rows,
                ..Default::default()
            };
            if (self.cuda.cuMemcpy2D_v2)(&copy) != CUresult::CUDA_SUCCESS {
                (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                return Err("ARGB host->device copy failed".into());
            }

            let result = self.submit_frame(
                self.mapped_input_buffer,
                NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
                frame_number,
                force_idr,
            );
            (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
            result
        }
    }

    /// @brief Encodes a raw byte array by copying from Host to Device.
    /// @input raw_data: Slice of raw pixel data (NV12 or YUV444).
    /// @input frame_number: Frame index.
    /// @input target_qp: Desired quality parameter.
    /// @input force_idr: Force keyframe generation.
    /// @return Result containing encoded byte vector.
    pub fn encode_raw(
        &mut self,
        raw_data: &[u8],
        frame_number: u64,
        target_qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        unsafe {
            self.reconfigure_if_needed(target_qp);
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);

            let is_444 = self.encode_config.encodeCodecConfig.h264Config.chromaFormatIDC == 3;

            if self.nv12_device_ptr.is_none() {
                let mut d_ptr: CUdeviceptr = 0;
                let mut pitch: usize = 0;

                let alloc_height = if is_444 {
                    self.height * 3
                } else {
                    self.height + (self.height / 2)
                };

                let res = (self.cuda.cuMemAllocPitch_v2)(
                    &mut d_ptr,
                    &mut pitch,
                    self.width as usize,
                    alloc_height as usize,
                    16,
                );
                if res != CUresult::CUDA_SUCCESS {
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to allocate GPU buffer for raw input".into());
                }

                let buffer_fmt = if is_444 {
                    NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_YUV444
                } else {
                    NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12
                };

                let mut reg_res = NV_ENC_REGISTER_RESOURCE {
                    version: sv(NvStruct::RegisterResource),
                    resourceType:
                        NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                    width: self.width,
                    height: self.height,
                    resourceToRegister: d_ptr as *mut c_void,
                    pitch: pitch as u32,
                    bufferFormat: buffer_fmt,
                    bufferUsage: NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE,
                    ..Default::default()
                };

                let register_fn = self.nvenc_funcs.nvEncRegisterResource.unwrap();
                if register_fn(self.encoder_session, &mut reg_res) != NVENCSTATUS::NV_ENC_SUCCESS {
                    (self.cuda.cuMemFree_v2)(d_ptr);
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to register raw input buffer".into());
                }

                let mut map_params = NV_ENC_MAP_INPUT_RESOURCE {
                    version: sv(NvStruct::MapInputResource),
                    registeredResource: reg_res.registeredResource,
                    ..Default::default()
                };
                let map_fn = self.nvenc_funcs.nvEncMapInputResource.unwrap();
                if map_fn(self.encoder_session, &mut map_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                    (self.nvenc_funcs.nvEncUnregisterResource.unwrap())(
                        self.encoder_session,
                        reg_res.registeredResource,
                    );
                    (self.cuda.cuMemFree_v2)(d_ptr);
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to map raw input buffer".into());
                }

                self.nv12_device_ptr = Some(d_ptr);
                self.nv12_pitch = pitch;
                self.nv12_registered_resource = Some(reg_res.registeredResource);
                self.nv12_mapped_buffer = Some(map_params.mappedResource);
            }

            let dev_ptr = self.nv12_device_ptr.unwrap();
            let dev_pitch = self.nv12_pitch;
            let width_bytes = self.width as usize;
            let height = self.height as usize;

            if is_444 {
                let plane_size = width_bytes * height;
                // The Y copy reads exactly `plane_size` bytes from the host slice; refuse rather
                // than read out of bounds if the caller handed us a short buffer.
                if raw_data.len() < plane_size {
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("raw frame smaller than the Y plane (444)".into());
                }

                let copy_y = CUDA_MEMCPY2D {
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: raw_data.as_ptr() as *const c_void,
                    srcPitch: width_bytes,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstDevice: dev_ptr,
                    dstPitch: dev_pitch,
                    WidthInBytes: width_bytes,
                    Height: height,
                    ..Default::default()
                };
                if (self.cuda.cuMemcpy2D_v2)(&copy_y) != CUresult::CUDA_SUCCESS {
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to copy Y plane (444)".into());
                }

                // Each chroma copy reads a FULL plane_size from its offset: require the whole
                // plane to be present (not just its start offset) so we never read past the end.
                if raw_data.len() >= 2 * plane_size {
                    let copy_u = CUDA_MEMCPY2D {
                        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                        srcHost: raw_data[plane_size..].as_ptr() as *const c_void,
                        srcPitch: width_bytes,
                        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                        dstDevice: dev_ptr + (dev_pitch * height) as u64,
                        dstPitch: dev_pitch,
                        WidthInBytes: width_bytes,
                        Height: height,
                        ..Default::default()
                    };
                    if (self.cuda.cuMemcpy2D_v2)(&copy_u) != CUresult::CUDA_SUCCESS {
                        (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                        return Err("Failed to copy U plane (444)".into());
                    }
                }

                if raw_data.len() >= 3 * plane_size {
                    let copy_v = CUDA_MEMCPY2D {
                        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                        srcHost: raw_data[2 * plane_size..].as_ptr() as *const c_void,
                        srcPitch: width_bytes,
                        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                        dstDevice: dev_ptr + (dev_pitch * height * 2) as u64,
                        dstPitch: dev_pitch,
                        WidthInBytes: width_bytes,
                        Height: height,
                        ..Default::default()
                    };
                    if (self.cuda.cuMemcpy2D_v2)(&copy_v) != CUresult::CUDA_SUCCESS {
                        (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                        return Err("Failed to copy V plane (444)".into());
                    }
                }
            } else {
                let y_size = width_bytes * height;
                // The Y copy reads `y_size` bytes; refuse a short buffer rather than read OOB.
                if raw_data.len() < y_size {
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("raw frame smaller than the Y plane".into());
                }
                let copy_y = CUDA_MEMCPY2D {
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: raw_data.as_ptr() as *const c_void,
                    srcPitch: width_bytes,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstDevice: dev_ptr,
                    dstPitch: dev_pitch,
                    WidthInBytes: width_bytes,
                    Height: height,
                    ..Default::default()
                };
                if (self.cuda.cuMemcpy2D_v2)(&copy_y) != CUresult::CUDA_SUCCESS {
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to copy Y plane".into());
                }

                let uv_offset = y_size;
                // The interleaved UV copy reads width_bytes * (height/2) bytes from uv_offset:
                // require that whole span, not just its start, so we never read past the end.
                if raw_data.len() >= uv_offset + width_bytes * (height / 2) {
                    let copy_uv = CUDA_MEMCPY2D {
                        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
                        srcHost: raw_data[uv_offset..].as_ptr() as *const c_void,
                        srcPitch: width_bytes,
                        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                        dstDevice: dev_ptr + (dev_pitch * height) as u64,
                        dstPitch: dev_pitch,
                        WidthInBytes: width_bytes,
                        Height: height / 2,
                        ..Default::default()
                    };
                    if (self.cuda.cuMemcpy2D_v2)(&copy_uv) != CUresult::CUDA_SUCCESS {
                        (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                        return Err("Failed to copy UV plane".into());
                    }
                }
            }

            let raw_format = if self.encode_config.encodeCodecConfig.h264Config.chromaFormatIDC == 3 {
                NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_YUV444
            } else {
                NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12
            };
            let result =
                self.submit_frame(self.nv12_mapped_buffer.unwrap(), raw_format, frame_number, force_idr);
            (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
            result
        }
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;

    fn settings(w: i32, h: i32, fps: f64) -> RustCaptureSettings {
        RustCaptureSettings {
            width: w,
            height: h,
            output_mode: 1,
            target_fps: fps,
            video_crf: 25,
            ..Default::default()
        }
    }

    fn frame(w: usize, h: usize, seed: u8) -> Vec<u8> {
        // Content with structure so encodes are non-trivial.
        let mut f = vec![0u8; w * h * 4];
        for (i, px) in f.chunks_exact_mut(4).enumerate() {
            let v = ((i as u32).wrapping_mul(2654435761) >> 24) as u8;
            px[0] = v.wrapping_add(seed);
            px[1] = v ^ seed;
            px[2] = seed;
            px[3] = 255;
        }
        f
    }

    fn wire_dims(pkt: &[u8]) -> (u16, u16) {
        (
            u16::from_be_bytes([pkt[6], pkt[7]]),
            u16::from_be_bytes([pkt[8], pkt[9]]),
        )
    }

    /// End-to-end in-place resize on a real GPU: grow, shrink, rejection cases, and a
    /// decodable dump. Run with: cargo test gpu_ -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gpu_resolution_reconfigure_roundtrip() {
        let mut s = settings(1280, 720, 60.0);
        let t0 = std::time::Instant::now();
        let mut enc = NvencEncoder::new(&s, ptr::null(), None).expect("NVENC init");
        let init_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut stream: Vec<u8> = Vec::new();
        let f720 = frame(1280, 720, 10);
        for i in 0..5u64 {
            let pkt = enc
                .encode_cpu_argb(&f720, 1280 * 4, i, 25, i == 0)
                .expect("encode 720p");
            assert_eq!(wire_dims(&pkt), (1280, 720));
            stream.extend_from_slice(&pkt[10..]);
        }

        // Grow within headroom.
        s.width = 1920;
        s.height = 1080;
        let t1 = std::time::Instant::now();
        enc.reconfigure_resolution(&s).expect("grow reconfigure");
        let grow_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let f1080 = frame(1920, 1080, 40);
        let pkt = enc
            .encode_cpu_argb(&f1080, 1920 * 4, 5, 25, false)
            .expect("encode 1080p");
        assert_eq!(pkt[0], 0x04);
        assert_eq!(pkt[1], 0x01, "first frame after a resize must be an IDR");
        assert_eq!(wire_dims(&pkt), (1920, 1080));
        stream.extend_from_slice(&pkt[10..]);
        for i in 6..10u64 {
            let pkt = enc
                .encode_cpu_argb(&f1080, 1920 * 4, i, 25, false)
                .expect("encode 1080p");
            assert_eq!(pkt[1], 0x00, "steady frames after the IDR are P frames");
            stream.extend_from_slice(&pkt[10..]);
        }

        // Shrink.
        s.width = 640;
        s.height = 480;
        let t2 = std::time::Instant::now();
        enc.reconfigure_resolution(&s).expect("shrink reconfigure");
        let shrink_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let f480 = frame(640, 480, 70);
        let pkt = enc
            .encode_cpu_argb(&f480, 640 * 4, 10, 25, false)
            .expect("encode 480p");
        assert_eq!(pkt[1], 0x01);
        assert_eq!(wire_dims(&pkt), (640, 480));
        stream.extend_from_slice(&pkt[10..]);

        // Rejections that must fall back to a rebuild, leaving the session usable.
        s.width = 4100;
        s.height = 2400;
        assert!(enc.reconfigure_resolution(&s).is_err(), "beyond headroom");
        s.width = 640;
        s.height = 480;
        s.video_fullcolor = true;
        assert!(enc.reconfigure_resolution(&s).is_err(), "chroma flip");
        s.video_fullcolor = false;
        s.video_cbr_mode = true;
        assert!(enc.reconfigure_resolution(&s).is_err(), "RC mode flip");
        s.video_cbr_mode = false;
        let pkt = enc
            .encode_cpu_argb(&f480, 640 * 4, 11, 25, false)
            .expect("session survives rejected reconfigures");
        stream.extend_from_slice(&pkt[10..]);

        println!(
            "init={init_ms:.1}ms grow(720p->1080p)={grow_ms:.1}ms shrink(1080p->480p)={shrink_ms:.1}ms"
        );
        if let Ok(path) = std::env::var("NVENC_TEST_DUMP") {
            std::fs::write(&path, &stream).unwrap();
            println!("wrote {} bytes to {path}", stream.len());
        }
    }

    /// CBR sessions fold the current rate into the resize reconfigure.
    #[test]
    #[ignore]
    fn gpu_resolution_reconfigure_cbr() {
        let mut s = settings(1280, 720, 60.0);
        s.video_cbr_mode = true;
        s.video_bitrate_kbps = 4000;
        let mut enc = NvencEncoder::new(&s, ptr::null(), None).expect("NVENC init");
        let f720 = frame(1280, 720, 10);
        for i in 0..3u64 {
            enc.encode_cpu_argb(&f720, 1280 * 4, i, 25, i == 0)
                .expect("encode 720p");
        }
        s.width = 1920;
        s.height = 1080;
        s.video_bitrate_kbps = 8000;
        enc.reconfigure_resolution(&s).expect("cbr resize+rate");
        assert_eq!(enc.encode_config.rcParams.averageBitRate, 8_000_000);
        let f1080 = frame(1920, 1080, 40);
        let pkt = enc
            .encode_cpu_argb(&f1080, 1920 * 4, 3, 25, false)
            .expect("encode 1080p");
        assert_eq!(pkt[1], 0x01);
        assert_eq!(wire_dims(&pkt), (1920, 1080));
    }

    /// Prints the device-memory cost of one encoding session (for measuring the
    /// reconfigure-headroom overhead against a build without it).
    #[test]
    #[ignore]
    fn gpu_vram_probe() {
        fn used_mb() -> i64 {
            let out = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
                .output()
                .expect("nvidia-smi");
            String::from_utf8_lossy(&out.stdout).trim().parse().expect("parse MiB")
        }
        let s = settings(1920, 1080, 60.0);
        let before = used_mb();
        let mut enc = NvencEncoder::new(&s, ptr::null(), None).expect("init");
        let f = frame(1920, 1080, 5);
        for i in 0..3u64 {
            enc.encode_cpu_argb(&f, 1920 * 4, i, 25, i == 0).expect("encode");
        }
        println!("VRAM delta for one 1080p session: {} MiB", used_mb() - before);
    }

    /// Sessions starting above the default headroom get their own size as the ceiling
    /// (portrait 4K: taller than the 2304 default while inside NVENC's H.264 4096 cap).
    #[test]
    #[ignore]
    fn gpu_init_above_default_headroom() {
        let s = settings(2160, 4096, 30.0);
        let mut enc = NvencEncoder::new(&s, ptr::null(), None).expect("NVENC init portrait 4K");
        assert_eq!(enc.init_params.maxEncodeWidth, 4096);
        assert_eq!(enc.init_params.maxEncodeHeight, 4096);
        let f = frame(2160, 4096, 20);
        let pkt = enc
            .encode_cpu_argb(&f, 2160 * 4, 0, 25, true)
            .expect("encode portrait 4K");
        assert_eq!(wire_dims(&pkt), (2160, 4096));
    }
}

#[cfg(test)]
mod version_tests {
    use super::*;

    /// Every version-tagged struct, in one fixed order, with its pinned compile-time constant.
    const ALL: [(NvStruct, u32); 12] = [
        (NvStruct::FunctionList, NV_ENCODE_API_FUNCTION_LIST_VER),
        (NvStruct::OpenSessionExParams, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER),
        (NvStruct::Config, NV_ENC_CONFIG_VER),
        (NvStruct::RcParams, NV_ENC_RC_PARAMS_VER),
        (NvStruct::PresetConfig, NV_ENC_PRESET_CONFIG_VER),
        (NvStruct::InitializeParams, NV_ENC_INITIALIZE_PARAMS_VER),
        (NvStruct::ReconfigureParams, NV_ENC_RECONFIGURE_PARAMS_VER),
        (NvStruct::RegisterResource, NV_ENC_REGISTER_RESOURCE_VER),
        (NvStruct::MapInputResource, NV_ENC_MAP_INPUT_RESOURCE_VER),
        (NvStruct::CreateBitstreamBuffer, NV_ENC_CREATE_BITSTREAM_BUFFER_VER),
        (NvStruct::PicParams, NV_ENC_PIC_PARAMS_VER),
        (NvStruct::LockBitstream, NV_ENC_LOCK_BITSTREAM_VER),
    ];

    // For the pinned nvcodec-sys version the table must reproduce every compile-time
    // NV_ENC_*_VER constant exactly, guaranteeing a current driver is byte-for-byte unchanged.
    // Also fails loudly if the bundled header is bumped without extending the revision table.
    #[test]
    fn table_is_identity_for_pinned_version() {
        let maj = NVENCAPI_VERSION & 0xFF;
        let min = (NVENCAPI_VERSION >> 24) & 0xFF;
        for (s, base) in ALL {
            assert_eq!(nvenc_struct_ver(s, maj, min), base, "{:?}", s);
        }
        assert_eq!(maj | (min << 24), NVENCAPI_VERSION);
    }

    // The exact NV_ENC_*_VER words each SDK defined, hardcoded from nvEncodeAPI.h at the FFmpeg
    // nv-codec-headers tag named per row; the table must reproduce them for every negotiable
    // version, in ALL order. (The n10.0.26.2 header spells the flag `1<<31` instead of `1u<<31`;
    // same bit.)
    #[test]
    fn table_matches_historical_headers() {
        #[rustfmt::skip]
        let expected: [(u32, u32, [u32; 12]); 7] = [
            // n10.0.26.2 (SDK 10.0)
            (10, 0, [0x7002000A, 0x7001000A, 0xF007000A, 0x7001000A, 0xF004000A, 0xF005000A, 0xF001000A,
                     0x7003000A, 0x7004000A, 0x7001000A, 0xF004000A, 0x7001000A]),
            // n11.0.10.3 (SDK 11.0)
            (11, 0, [0x7002000B, 0x7001000B, 0xF007000B, 0x7001000B, 0xF004000B, 0xF005000B, 0xF001000B,
                     0x7003000B, 0x7004000B, 0x7001000B, 0xF004000B, 0x7001000B]),
            // n11.1.5.3 (SDK 11.1)
            (11, 1, [0x7102000B, 0x7101000B, 0xF107000B, 0x7101000B, 0xF104000B, 0xF105000B, 0xF101000B,
                     0x7103000B, 0x7104000B, 0x7101000B, 0xF104000B, 0x7101000B]),
            // n12.0.16.1 (SDK 12.0)
            (12, 0, [0x7002000C, 0x7001000C, 0xF008000C, 0x7001000C, 0xF004000C, 0xF005000C, 0xF001000C,
                     0x7004000C, 0x7004000C, 0x7001000C, 0xF006000C, 0x7002000C]),
            // n12.1.14.0 (SDK 12.1)
            (12, 1, [0x7102000C, 0x7101000C, 0xF108000C, 0x7101000C, 0xF104000C, 0xF106000C, 0xF101000C,
                     0x7104000C, 0x7104000C, 0x7101000C, 0xF106000C, 0xF101000C]),
            // n12.2.72.0 (SDK 12.2)
            (12, 2, [0x7202000C, 0x7201000C, 0xF209000C, 0x7201000C, 0xF205000C, 0xF207000C, 0xF202000C,
                     0x7205000C, 0x7204000C, 0x7201000C, 0xF207000C, 0xF202000C]),
            // n13.0.19.0 (SDK 13.0)
            (13, 0, [0x7002000D, 0x7001000D, 0xF009000D, 0x7001000D, 0xF005000D, 0xF007000D, 0xF002000D,
                     0x7005000D, 0x7004000D, 0x7001000D, 0xF007000D, 0xF002000D]),
        ];
        for (maj, min, words) in expected {
            for ((s, _), want) in ALL.iter().zip(words) {
                assert_eq!(
                    nvenc_struct_ver(*s, maj, min),
                    want,
                    "{:?} at {}.{}",
                    s,
                    maj,
                    min
                );
            }
        }
    }
}
