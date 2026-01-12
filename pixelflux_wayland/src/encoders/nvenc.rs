#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::os::unix::io::AsRawFd;
use std::ptr;
use std::sync::Arc;

use libloading::{Library, Symbol};
use smithay::backend::allocator::{dmabuf::Dmabuf, Buffer};

use crate::RustCaptureSettings;
use nvenc_sys::cuda::*;
use nvenc_sys::*;

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

/// @brief dynamically loaded EGL function pointers.
struct EglFunctions {
    _lib: Library,
    eglGetProcAddress: unsafe extern "C" fn(procname: *const c_char) -> *mut c_void,
    eglCreateImageKHR: unsafe extern "C" fn(
        dpy: EGLDisplay,
        ctx: *mut c_void,
        target: EGLenum,
        buffer: *mut c_void,
        attrib_list: *const EGLint,
    ) -> EGLImageKHR,
    eglDestroyImageKHR: unsafe extern "C" fn(dpy: EGLDisplay, image: EGLImageKHR) -> EGLBoolean,
}

/// @brief Dynamically loaded CUDA function pointers.
struct CudaFunctions {
    _lib: Library,
    cuInit: unsafe extern "C" fn(flags: u32) -> CUresult,
    cuDeviceGet: unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult,
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
    cuda: Arc<CudaFunctions>,
    egl: Arc<EglFunctions>,
    _nvenc_lib: Arc<NvencLibrary>,
    nvenc_funcs: NV_ENCODE_API_FUNCTION_LIST,
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

            if !self.encoder_session.is_null() {
                (self.nvenc_funcs.nvEncDestroyEncoder.unwrap())(self.encoder_session);
            }

            (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
            (self.cuda.cuCtxDestroy_v2)(self.cuda_context);
        }
    }
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
                eglCreateImageKHR: std::mem::transmute(create_addr),
                eglDestroyImageKHR: std::mem::transmute(destroy_addr),
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
            let lib_name = if cfg!(windows) {
                "nvEncodeAPI64.dll"
            } else {
                "libnvidia-encode.so.1"
            };
            let lib = Library::new(lib_name)
                .map_err(|e| format!("Could not load NVENC library ({}): {}", lib_name, e))?;

            Ok(NvencLibrary {
                create_instance: *lib
                    .get(b"NvEncodeAPICreateInstance\0")
                    .map_err(|e| e.to_string())?,
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
            format!("Unknown CUDA Error ({})", err as u32)
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

    /// @brief Initializes the NVENC encoder, CUDA context, and primary resources.
    /// @input settings: Capture settings (resolution, FPS, QP).
    /// @input egl_display: The EGL display handle for interop.
    /// @return Result containing the initialized NvencEncoder instance.
    pub fn new(
        settings: &RustCaptureSettings,
        egl_display: *const c_void,
    ) -> Result<Self, String> {
        println!("[NVENC] Initializing...");

        let egl = Arc::new(Self::load_egl()?);
        let cuda = Arc::new(Self::load_cuda()?);
        let nvenc_lib = Arc::new(Self::load_nvenc()?);

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
            (cuda.cuDeviceGet)(&mut cu_device, 0);

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
                version: NV_ENCODE_API_FUNCTION_LIST_VER,
                ..Default::default()
            };
            (nvenc_lib.create_instance)(&mut function_list);

            let mut session_params = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
                version: NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
                deviceType: NV_ENC_DEVICE_TYPE::NV_ENC_DEVICE_TYPE_CUDA,
                device: cu_context as *mut c_void,
                apiVersion: NVENCAPI_VERSION,
                ..Default::default()
            };

            let mut encoder_session: *mut c_void = ptr::null_mut();
            let open_fn = function_list.nvEncOpenEncodeSessionEx.unwrap();
            if open_fn(&mut session_params, &mut encoder_session) != NVENCSTATUS::NV_ENC_SUCCESS {
                return Err("Failed to open NVENC session".into());
            }

            let is_444 = settings.h264_fullcolor;
            let profile_guid = if is_444 {
                NV_ENC_H264_PROFILE_HIGH_444_GUID
            } else {
                NV_ENC_H264_PROFILE_HIGH_GUID
            };

            let mut config = NV_ENC_CONFIG {
                version: NV_ENC_CONFIG_VER,
                ..Default::default()
            };
            let mut preset_config = NV_ENC_PRESET_CONFIG {
                version: NV_ENC_PRESET_CONFIG_VER,
                presetCfg: config,
                ..Default::default()
            };

            let get_preset = function_list.nvEncGetEncodePresetConfig.unwrap();
            get_preset(
                encoder_session,
                NV_ENC_CODEC_H264_GUID,
                NV_ENC_PRESET_LOW_LATENCY_HQ_GUID,
                &mut preset_config,
            );

            config = preset_config.presetCfg;
            config.profileGUID = profile_guid;
            config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CONSTQP;
            config.rcParams.constQP.qpInterP = settings.h264_crf as u32;
            config.rcParams.constQP.qpInterB = settings.h264_crf as u32;
            config.rcParams.constQP.qpIntra = settings.h264_crf as u32;
            config.frameIntervalP = 1; 
            config.gopLength = 0xFFFFFFFF;
            config.encodeCodecConfig.h264Config.h264VUIParameters.videoSignalTypePresentFlag = 1;
            config.encodeCodecConfig.h264Config.h264VUIParameters.videoFormat = 5;
            config.encodeCodecConfig.h264Config.h264VUIParameters.colourDescriptionPresentFlag = 1;
            config.encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries = 1;
            config.encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics = 1;
            config.encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix = 1;
            config.encodeCodecConfig.h264Config.chromaFormatIDC = if is_444 { 3 } else { 1 };
            config.encodeCodecConfig.h264Config.h264VUIParameters.videoFullRangeFlag =
                if is_444 { 1 } else { 0 };

            let mut init_params = NV_ENC_INITIALIZE_PARAMS {
                version: NV_ENC_INITIALIZE_PARAMS_VER,
                encodeGUID: NV_ENC_CODEC_H264_GUID,
                presetGUID: NV_ENC_PRESET_LOW_LATENCY_HQ_GUID,
                encodeWidth: width,
                encodeHeight: height,
                darWidth: width,
                darHeight: height,
                frameRateNum: settings.target_fps as u32,
                frameRateDen: 1,
                enablePTD: 1,
                encodeConfig: &mut config,
                ..Default::default()
            };

            let init_fn = function_list.nvEncInitializeEncoder.unwrap();
            if init_fn(encoder_session, &mut init_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                return Err("Failed to initialize encoder".into());
            }

            let mut reg_res = NV_ENC_REGISTER_RESOURCE {
                version: NV_ENC_REGISTER_RESOURCE_VER,
                resourceType: NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                width: width,
                height: height,
                resourceToRegister: input_device_ptr as *mut c_void,
                pitch: input_pitch as u32,
                bufferFormat: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
                bufferUsage: NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE,
                ..Default::default()
            };

            let register_fn = function_list.nvEncRegisterResource.unwrap();
            if register_fn(encoder_session, &mut reg_res) != NVENCSTATUS::NV_ENC_SUCCESS {
                return Err("Failed to register input buffer".into());
            }

            let mut map_params = NV_ENC_MAP_INPUT_RESOURCE {
                version: NV_ENC_MAP_INPUT_RESOURCE_VER,
                registeredResource: reg_res.registeredResource,
                ..Default::default()
            };
            let map_fn = function_list.nvEncMapInputResource.unwrap();
            if map_fn(encoder_session, &mut map_params) != NVENCSTATUS::NV_ENC_SUCCESS {
                return Err("Failed to map input buffer".into());
            }

            let mut bitstream_buffers = Vec::new();
            let create_bs_fn = function_list.nvEncCreateBitstreamBuffer.unwrap();
            for _ in 0..4 {
                let mut bitstream_params = NV_ENC_CREATE_BITSTREAM_BUFFER {
                    version: NV_ENC_CREATE_BITSTREAM_BUFFER_VER,
                    ..Default::default()
                };
                if create_bs_fn(encoder_session, &mut bitstream_params)
                    != NVENCSTATUS::NV_ENC_SUCCESS
                {
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
                current_qp: settings.h264_crf as u32,
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
                cuda,
                egl,
                _nvenc_lib: nvenc_lib,
                nvenc_funcs: function_list,
            })
        }
    }

    /// @brief Detects if the quantization parameter (QP) has changed and reconfigures the encoder.
    /// @input target_qp: The new desired QP value.
    /// @return bool: True if reconfiguration occurred, false otherwise.
    unsafe fn reconfigure_if_needed(&mut self, target_qp: u32) -> bool {
        if self.current_qp != target_qp {
            self.encode_config.rcParams.constQP.qpInterP = target_qp;
            self.encode_config.rcParams.constQP.qpInterB = target_qp;
            self.encode_config.rcParams.constQP.qpIntra = target_qp;
            self.init_params.encodeConfig = &mut self.encode_config;

            let mut reconfig_params = NV_ENC_RECONFIGURE_PARAMS {
                version: NV_ENC_RECONFIGURE_PARAMS_VER,
                reInitEncodeParams: self.init_params,
                ..Default::default()
            };

            if target_qp < self.current_qp {
                reconfig_params.set_forceIDR(1);
            }

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

    /// @brief Submits a frame to NVENC, locks the output bitstream, and retrieves the encoded data.
    /// @input mapped_buffer: The CUDA-mapped input resource containing the image.
    /// @input frame_number: Monotonically increasing frame index.
    /// @input force_idr: If true, forces an IDR (Keyframe).
    /// @return Result containing the encoded packet with custom header.
    unsafe fn submit_frame(
        &mut self,
        mapped_buffer: NV_ENC_INPUT_PTR,
        frame_number: u64,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        let output_bitstream = self.bitstream_buffers[self.current_buffer_idx];
        self.current_buffer_idx = (self.current_buffer_idx + 1) % self.bitstream_buffers.len();

        let mut pic_params = NV_ENC_PIC_PARAMS {
            version: NV_ENC_PIC_PARAMS_VER,
            inputWidth: self.width,
            inputHeight: self.height,
            inputBuffer: mapped_buffer,
            outputBitstream: output_bitstream,
            bufferFmt: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
            pictureStruct: NV_ENC_PIC_STRUCT::NV_ENC_PIC_STRUCT_FRAME,
            encodePicFlags: if force_idr {
                NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_FORCEIDR as u32
            } else {
                0
            },
            ..Default::default()
        };

        if mapped_buffer == self.nv12_mapped_buffer.unwrap_or(ptr::null_mut()) {
            if self.encode_config.encodeCodecConfig.h264Config.chromaFormatIDC == 3 {
                pic_params.bufferFmt = NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_YUV444;
            } else {
                pic_params.bufferFmt = NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12;
            }
        }

        let encode_fn = self.nvenc_funcs.nvEncEncodePicture.unwrap();
        let res = encode_fn(self.encoder_session, &mut pic_params);
        if res != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("Encode Picture failed: {:?}", res));
        }

        let mut lock_params = NV_ENC_LOCK_BITSTREAM {
            version: NV_ENC_LOCK_BITSTREAM_VER,
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
        let mut output = Vec::with_capacity(10 + data_size);

        output.push(0x04);
        output.push(if force_idr { 0x01 } else { 0x00 });
        output.extend_from_slice(&(frame_number as u16).to_be_bytes());
        output.extend_from_slice(&0u16.to_be_bytes());
        output.extend_from_slice(&(self.width as u16).to_be_bytes());
        output.extend_from_slice(&(self.height as u16).to_be_bytes());

        if data_size > 0 && !data_ptr.is_null() {
            output.extend_from_slice(std::slice::from_raw_parts(data_ptr, data_size));
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
            let _ = (self.cuda.cuCtxPushCurrent_v2)(self.cuda_context);
            let fd = dmabuf.handles().next().ok_or("No handles")?.as_raw_fd();

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

            let result = self.submit_frame(self.mapped_input_buffer, frame_number, force_idr);
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
                    version: NV_ENC_REGISTER_RESOURCE_VER,
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
                    (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
                    return Err("Failed to register raw input buffer".into());
                }

                let mut map_params = NV_ENC_MAP_INPUT_RESOURCE {
                    version: NV_ENC_MAP_INPUT_RESOURCE_VER,
                    registeredResource: reg_res.registeredResource,
                    ..Default::default()
                };
                let map_fn = self.nvenc_funcs.nvEncMapInputResource.unwrap();
                if map_fn(self.encoder_session, &mut map_params) != NVENCSTATUS::NV_ENC_SUCCESS {
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

                if plane_size < raw_data.len() {
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

                if 2 * plane_size < raw_data.len() {
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

                let uv_offset = width_bytes * height;
                if uv_offset < raw_data.len() {
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

            let result =
                self.submit_frame(self.nv12_mapped_buffer.unwrap(), frame_number, force_idr);
            (self.cuda.cuCtxPopCurrent_v2)(ptr::null_mut());
            result
        }
    }
}
