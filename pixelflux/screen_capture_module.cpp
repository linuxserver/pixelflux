/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/*
  ▘    ▜ ▐▘▜     
▛▌▌▚▘█▌▐ ▜▘▐ ▌▌▚▘
▙▌▌▞▖▙▖▐▖▐ ▐▖▙▌▞▖
▌                
*/

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <algorithm>
#include <Python.h>
#include <structmember.h>
#include <X11/Xlib.h>
#include <X11/extensions/XShm.h>
#include <X11/extensions/Xfixes.h>
#include <X11/Xutil.h>
#include <jpeglib.h>
#include <netinet/in.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#define XXH_STATIC_LINKING_ONLY
#include "xxhash.h"
#include <libyuv/convert.h>
#include <libyuv/convert_from.h>
#include <libyuv/convert_from_argb.h>
#include <libyuv/planar_functions.h>
#include <x264.h>
#include <string>
#include <cmath>
#include <dlfcn.h>
#include "nvEncodeAPI.h"
#ifndef STB_IMAGE_IMPLEMENTATION_DEFINED
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_enc_h264.h>
#define VA_ENC_PACKED_HEADER_DATA (VA_ENC_PACKED_HEADER_SEQUENCE | VA_ENC_PACKED_HEADER_PICTURE)
#define STB_IMAGE_IMPLEMENTATION_DEFINED
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"

typedef enum CUresult_enum { CUDA_SUCCESS = 0 } CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef CUresult (*tcuInit)(unsigned int);
typedef CUresult (*tcuDeviceGet)(CUdevice*, int);
typedef CUresult (*tcuCtxCreate)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*tcuCtxDestroy)(CUcontext);

struct CudaFunctions {
  tcuInit pfn_cuInit = nullptr;
  tcuDeviceGet pfn_cuDeviceGet = nullptr;
  tcuCtxCreate pfn_cuCtxCreate = nullptr;
  tcuCtxDestroy pfn_cuCtxDestroy = nullptr;
};

CudaFunctions g_cuda_funcs;
static void* g_cuda_lib_handle = nullptr;

struct NvencEncoderState {
  NV_ENCODE_API_FUNCTION_LIST nvenc_funcs = {0};
  void* encoder_session = nullptr;
  NV_ENC_INITIALIZE_PARAMS init_params = {0};
  NV_ENC_CONFIG encode_config = {0};
  std::vector<NV_ENC_INPUT_PTR> input_buffers;
  std::vector<NV_ENC_OUTPUT_PTR> output_buffers;
  uint32_t current_input_buffer_idx = 0;
  uint32_t current_output_buffer_idx = 0;
  int buffer_pool_size = 4;
  bool initialized = false;
  int initialized_width = 0;
  int initialized_height = 0;
  int initialized_qp = -1;
  NV_ENC_BUFFER_FORMAT initialized_buffer_format = NV_ENC_BUFFER_FORMAT_UNDEFINED;
  CUcontext cuda_context = nullptr;

  NvencEncoderState() {
    nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    init_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
  }
};
NvencEncoderState g_nvenc_state;
std::mutex g_nvenc_mutex;
std::atomic<bool> g_nvenc_force_next_idr_global{true};

static void* g_nvenc_lib_handle = nullptr;
typedef NVENCSTATUS(NVENCAPI* PFN_NvEncodeAPICreateInstance)(
  NV_ENCODE_API_FUNCTION_LIST*);

/**
 * @brief Holds function pointers for the VA-API libraries (libva, libva-drm, libva-x11).
 * This struct is populated by `LoadVaapiApi` using `dlsym` to allow for
 * dynamic loading of the VA-API dependencies, avoiding a hard link-time
 * dependency on the user's system.
 */
struct VaapiFunctions {
    void *va_lib_handle = nullptr;
    void *va_x11_lib_handle = nullptr;
    void *va_drm_lib_handle = nullptr;
    VADisplay (*vaGetDisplay)(Display*) = nullptr;
    VADisplay (*vaGetDisplayDRM)(int) = nullptr;
    VAStatus (*vaInitialize)(VADisplay, int*, int*) = nullptr;
    VAStatus (*vaTerminate)(VADisplay) = nullptr;
    const char * (*vaQueryVendorString)(VADisplay) = nullptr;
    VAStatus (*vaCreateConfig)(VADisplay, VAProfile, VAEntrypoint, VAConfigAttrib*, int, VAConfigID*) = nullptr;
    VAStatus (*vaDestroyConfig)(VADisplay, VAConfigID) = nullptr;
    VAStatus (*vaCreateSurfaces)(VADisplay, unsigned int, unsigned int, unsigned int, VASurfaceID*, unsigned int, VASurfaceAttrib*, unsigned int) = nullptr;
    VAStatus (*vaDestroySurfaces)(VADisplay, VASurfaceID*, int) = nullptr;
    VAStatus (*vaCreateContext)(VADisplay, VAConfigID, int, int, int, VASurfaceID*, int, VAContextID*) = nullptr;
    VAStatus (*vaDestroyContext)(VADisplay, VAContextID) = nullptr;
    VAStatus (*vaCreateBuffer)(VADisplay, VAContextID, VABufferType, unsigned int, unsigned int, void*, VABufferID*) = nullptr;
    VAStatus (*vaDestroyBuffer)(VADisplay, VABufferID) = nullptr;
    VAStatus (*vaBeginPicture)(VADisplay, VAContextID, VASurfaceID) = nullptr;
    VAStatus (*vaRenderPicture)(VADisplay, VAContextID, VABufferID*, int) = nullptr;
    VAStatus (*vaEndPicture)(VADisplay, VAContextID) = nullptr;
    VAStatus (*vaSyncSurface)(VADisplay, VASurfaceID) = nullptr;
    VAStatus (*vaMapBuffer)(VADisplay, VABufferID, void**) = nullptr;
    VAStatus (*vaUnmapBuffer)(VADisplay, VABufferID) = nullptr;
    VAStatus (*vaDeriveImage)(VADisplay, VASurfaceID, VAImage*) = nullptr;
    VAStatus (*vaDestroyImage)(VADisplay, VAImageID) = nullptr;
    VAStatus (*vaCreateImage)(VADisplay, VAImageFormat*, int, int, VAImage*) = nullptr;
    VAStatus (*vaPutImage)(VADisplay, VASurfaceID, VAImageID, int, int, unsigned int, unsigned int, int, int, unsigned int, unsigned int) = nullptr;
    VAStatus (*vaGetConfigAttributes)(VADisplay, VAProfile, VAEntrypoint, VAConfigAttrib*, int) = nullptr;
};

/**
 * @brief Manages the state of a VA-API H.264 encoder session.
 * This struct encapsulates all the necessary handles and configuration for a
 * VA-API encoding pipeline, including the display connection, configuration and
 * context IDs, a pool of surfaces for video frames, and initialization status.
 * It is protected by the global `g_vaapi_mutex`.
 */
struct VaapiEncoderState {
    VaapiFunctions va_funcs = {0};
    VADisplay display = nullptr;
    VAConfigID config_id = VA_INVALID_ID;
    VAContextID context_id = VA_INVALID_ID;
    std::vector<VASurfaceID> surfaces;
    VABufferID coded_buffer_id = VA_INVALID_ID;
    int fd = -1;
    bool initialized = false;
    int initialized_width = 0;
    int initialized_height = 0;
    int initialized_qp = -1;
    unsigned int frame_count = 0;
    VAPictureH264 last_ref_pic;
};

VaapiEncoderState g_vaapi_state;
std::mutex g_vaapi_mutex;
std::atomic<bool> g_vaapi_force_next_idr_global{true};


/**
 * @brief Manages a pool of H.264 encoders and associated picture buffers.
 * This struct provides thread-safe storage and management for x264 encoder
 * instances, input pictures, and their initialization states. This allows
 * different threads to use separate encoder instances, particularly for
 * encoding different stripes of a video frame concurrently.
 */
struct MinimalEncoderStore {
  std::vector<x264_t*> encoders;
  std::vector<x264_picture_t*> pics_in_ptrs;
  std::vector<bool> initialized_flags;
  std::vector<int> initialized_widths;
  std::vector<int> initialized_heights;
  std::vector<int> initialized_crfs;
  std::vector<int> initialized_csps;
  std::vector<int> initialized_colorspaces;
  std::vector<bool> initialized_full_range_flags;
  std::vector<bool> force_idr_flags;
  std::mutex store_mutex;

  /**
   * @brief Ensures that the internal vectors are large enough for the given thread_id.
   * If thread_id is out of bounds, resizes all vectors to accommodate it,
   * initializing new elements to default values.
   * @param thread_id The ID of the thread, used as an index.
   */
  void ensure_size(int thread_id) {
    if (thread_id >= static_cast<int>(encoders.size())) {
      size_t new_size = static_cast<size_t>(thread_id) + 1;
      encoders.resize(new_size, nullptr);
      pics_in_ptrs.resize(new_size, nullptr);
      initialized_flags.resize(new_size, false);
      initialized_widths.resize(new_size, 0);
      initialized_heights.resize(new_size, 0);
      initialized_crfs.resize(new_size, -1);
      initialized_csps.resize(new_size, X264_CSP_NONE);
      initialized_colorspaces.resize(new_size, 0);
      initialized_full_range_flags.resize(new_size, false);
      force_idr_flags.resize(new_size, false);
    }
  }

  /**
   * @brief Resets the store by closing all encoders and freeing resources.
   * Clears all internal vectors, ensuring a clean state. This should be called
   * when encoder settings change significantly (e.g., resolution) or when
   * the capture module is stopped.
   */
  void reset() {
    std::lock_guard<std::mutex> lock(store_mutex);
    for (size_t i = 0; i < encoders.size(); ++i) {
      if (encoders[i]) {
        x264_encoder_close(encoders[i]);
        encoders[i] = nullptr;
      }
      if (pics_in_ptrs[i]) {
        if (i < initialized_flags.size() && initialized_flags[i]) {
          x264_picture_clean(pics_in_ptrs[i]);
        }
        delete pics_in_ptrs[i];
        pics_in_ptrs[i] = nullptr;
      }
    }
    encoders.clear();
    pics_in_ptrs.clear();
    initialized_flags.clear();
    initialized_widths.clear();
    initialized_heights.clear();
    initialized_crfs.clear();
    initialized_csps.clear();
    initialized_colorspaces.clear();
    initialized_full_range_flags.clear();
    force_idr_flags.clear();
  }

  /**
   * @brief Destructor for MinimalEncoderStore.
   * Calls reset() to ensure all resources are released upon destruction.
   */
  ~MinimalEncoderStore() {
    reset();
  }
};

MinimalEncoderStore g_h264_minimal_store;

/**
 * @brief Enumerates the possible output modes for encoding.
 */
enum class OutputMode {
  JPEG = 0, /**< Output frames as JPEG images. */
  H264 = 1  /**< Output frames as H.264 video. */
};

/**
 * @brief Enumerates the data types for encoded stripes.
 */
enum class StripeDataType {
  UNKNOWN = 0, /**< Unknown or uninitialized data type. */
  JPEG    = 1, /**< Data is JPEG encoded. */
  H264    = 2  /**< Data is H.264 encoded. */
};

/**
 * @brief Enumerates the watermark location identifiers
 */
enum class WatermarkLocation {
  NONE = 0, TL = 1, TR = 2, BL = 3, BR = 4, MI = 5, AN = 6
};

/**
 * @brief Holds settings for screen capture and encoding.
 * This struct aggregates all configurable parameters for the capture process,
 * including dimensions, frame rate, quality settings, and output mode.
 */
struct CaptureSettings {
  int capture_width;
  int capture_height;
  int capture_x;
  int capture_y;
  double target_fps;
  int jpeg_quality;
  int paint_over_jpeg_quality;
  bool use_paint_over_quality;
  int paint_over_trigger_frames;
  int damage_block_threshold;
  int damage_block_duration;
  OutputMode output_mode;
  int h264_crf;
  bool h264_fullcolor;
  bool h264_fullframe;
  bool capture_cursor;
  const char* watermark_path;
  WatermarkLocation watermark_location_enum;
  int vaapi_render_node_index;

  /**
   * @brief Default constructor for CaptureSettings.
   * Initializes settings with common default values.
   */
  CaptureSettings()
    : capture_width(1920),
      capture_height(1080),
      capture_x(0),
      capture_y(0),
      target_fps(60.0),
      jpeg_quality(85),
      paint_over_jpeg_quality(95),
      use_paint_over_quality(false),
      paint_over_trigger_frames(10),
      damage_block_threshold(15),
      damage_block_duration(30),
      output_mode(OutputMode::JPEG),
      h264_crf(25),
      h264_fullcolor(false),
      h264_fullframe(false),
      capture_cursor(false),
      watermark_path(nullptr),
      watermark_location_enum(WatermarkLocation::NONE),
      vaapi_render_node_index(-1) {}

  /**
   * @brief Parameterized constructor for CaptureSettings.
   * Allows initializing all settings with specific values.
   * @param cw Capture width.
   * @param ch Capture height.
   * @param cx Capture X offset.
   * @param cy Capture Y offset.
   * @param fps Target frames per second.
   * @param jq JPEG quality.
   * @param pojq Paint-over JPEG quality.
   * @param upoq Use paint-over quality flag.
   * @param potf Paint-over trigger frames.
   * @param dbt Damage block threshold.
   * @param dbd Damage block duration.
   * @param om Output mode (JPEG or H.264).
   * @param crf H.264 Constant Rate Factor.
   * @param h264_fc H.264 full color (I444) flag.
   * @param h264_ff H.264 full frame encoding flag.
   * @param capture_cursor Capture cursor flag.
   */
  CaptureSettings(int cw, int ch, int cx, int cy, double fps, int jq,
                  int pojq, bool upoq, int potf, int dbt, int dbd,
                  OutputMode om = OutputMode::JPEG, int crf = 25,
                  bool h264_fc = false, bool h264_ff = false,
                  bool capture_cursor = false,
                  const char* wm_path = nullptr,
                  WatermarkLocation wm_loc = WatermarkLocation::NONE,
                  int vaapi_idx = -1)
    : capture_width(cw),
      capture_height(ch),
      capture_x(cx),
      capture_y(cy),
      target_fps(fps),
      jpeg_quality(jq),
      paint_over_jpeg_quality(pojq),
      use_paint_over_quality(upoq),
      paint_over_trigger_frames(potf),
      damage_block_threshold(dbt),
      damage_block_duration(dbd),
      output_mode(om),
      h264_crf(crf),
      h264_fullcolor(h264_fc),
      h264_fullframe(h264_ff),
      capture_cursor(capture_cursor),
      watermark_path(wm_path),
      watermark_location_enum(wm_loc),
      vaapi_render_node_index(vaapi_idx) {}
};

/**
 * @brief Represents the result of encoding a single stripe of a frame.
 * Contains the encoded data, its type, dimensions, and frame identifier.
 * This struct uses move semantics for efficient data transfer.
 */
struct StripeEncodeResult {
  StripeDataType type;
  int stripe_y_start;
  int stripe_height;
  int size;
  unsigned char* data;
  int frame_id;

  /**
   * @brief Default constructor for StripeEncodeResult.
   * Initializes members to default/null values.
   */
  StripeEncodeResult()
    : type(StripeDataType::UNKNOWN),
      stripe_y_start(0),
      stripe_height(0),
      size(0),
      data(nullptr),
      frame_id(-1) {}

  /**
   * @brief Move constructor for StripeEncodeResult.
   * Transfers ownership of data from the 'other' object.
   * @param other The StripeEncodeResult to move from.
   */
  StripeEncodeResult(StripeEncodeResult&& other) noexcept;

  /**
   * @brief Move assignment operator for StripeEncodeResult.
   * Transfers ownership of data from the 'other' object, freeing existing data.
   * @param other The StripeEncodeResult to move assign from.
   * @return Reference to this object.
   */
  StripeEncodeResult& operator=(StripeEncodeResult&& other) noexcept;

private:
  StripeEncodeResult(const StripeEncodeResult&) = delete;
  StripeEncodeResult& operator=(const StripeEncodeResult&) = delete;
};

/**
 * @brief Move constructor implementation for StripeEncodeResult.
 * @param other The StripeEncodeResult to move data from.
 */
StripeEncodeResult::StripeEncodeResult(StripeEncodeResult&& other) noexcept
  : type(other.type),
    stripe_y_start(other.stripe_y_start),
    stripe_height(other.stripe_height),
    size(other.size),
    data(other.data),
    frame_id(other.frame_id) {
  other.type = StripeDataType::UNKNOWN;
  other.stripe_y_start = 0;
  other.stripe_height = 0;
  other.size = 0;
  other.data = nullptr;
  other.frame_id = -1;
}

/**
 * @brief Move assignment operator implementation for StripeEncodeResult.
 * @param other The StripeEncodeResult to move data from.
 * @return A reference to this StripeEncodeResult.
 */
StripeEncodeResult& StripeEncodeResult::operator=(StripeEncodeResult&& other) noexcept {
  if (this != &other) {
    if (data) {
      delete[] data;
      data = nullptr;
    }
    type = other.type;
    stripe_y_start = other.stripe_y_start;
    stripe_height = other.stripe_height;
    size = other.size;
    data = other.data;
    frame_id = other.frame_id;

    other.type = StripeDataType::UNKNOWN;
    other.stripe_y_start = 0;
    other.stripe_height = 0;
    other.size = 0;
    other.data = nullptr;
    other.frame_id = -1;
  }
  return *this;
}

/**
 * @brief Dynamically loads the CUDA driver library and resolves required function pointers.
 *
 * This function checks if the library is already loaded. If not, it uses `dlopen`
 * to load `libcuda.so` and `dlsym` to find the addresses of `cuInit`, `cuDeviceGet`,
 * `cuCtxCreate`, and `cuCtxDestroy`. The function pointers are stored in the
 * global `g_cuda_funcs` struct. This must be successful before any NVENC
 * operations that use a CUDA context can be performed.
 *
 * @return true if the library was loaded and all required function pointers were
 *         successfully resolved, false otherwise.
 */
bool LoadCudaApi() {
    if (g_cuda_lib_handle) {
        return true;
    }

    g_cuda_lib_handle = dlopen("libcuda.so", RTLD_LAZY);
    if (!g_cuda_lib_handle) {
        std::cerr << "CUDA_API_LOAD: dlopen failed for libcuda.so" << std::endl;
        return false;
    }

    g_cuda_funcs.pfn_cuInit = (tcuInit)dlsym(g_cuda_lib_handle, "cuInit");
    g_cuda_funcs.pfn_cuDeviceGet = (tcuDeviceGet)dlsym(g_cuda_lib_handle, "cuDeviceGet");
    g_cuda_funcs.pfn_cuCtxCreate = (tcuCtxCreate)dlsym(g_cuda_lib_handle, "cuCtxCreate");
    g_cuda_funcs.pfn_cuCtxDestroy = (tcuCtxDestroy)dlsym(g_cuda_lib_handle, "cuCtxDestroy");

    if (!g_cuda_funcs.pfn_cuInit || !g_cuda_funcs.pfn_cuDeviceGet || !g_cuda_funcs.pfn_cuCtxCreate || !g_cuda_funcs.pfn_cuCtxDestroy) {
        std::cerr << "CUDA_API_LOAD: dlsym failed for one or more CUDA functions." << std::endl;
        dlclose(g_cuda_lib_handle);
        g_cuda_lib_handle = nullptr;
        memset(&g_cuda_funcs, 0, sizeof(CudaFunctions));
        return false;
    }
    return true;
}

/**
 * @brief Unloads the CUDA driver library if it was previously loaded.
 *
 * This function calls `dlclose` on the CUDA library handle and clears the global
 * function pointer struct to ensure a clean state. It should be called when
 * CUDA functionality is no longer needed.
 */
void UnloadCudaApi() {
    if (g_cuda_lib_handle) {
        dlclose(g_cuda_lib_handle);
        g_cuda_lib_handle = nullptr;
        memset(&g_cuda_funcs, 0, sizeof(CudaFunctions));
    }
}

/**
 * @brief Dynamically loads the NVIDIA Encoder (NVENC) library and initializes the API function list.
 *
 * This function checks if the API is already loaded. If not, it attempts to load
 * `libnvidia-encode.so.1` or `libnvidia-encode.so` using `dlopen`. It then uses
 * `dlsym` to get the `NvEncodeAPICreateInstance` function and calls it to populate
 * the global `g_nvenc_state.nvenc_funcs` list, which contains pointers to all
 * other NVENC API functions.
 *
 * @return true if the library was loaded and the function list was successfully
 *         populated, false otherwise.
 */
bool LoadNvencApi() {
  if (g_nvenc_state.nvenc_funcs.nvEncOpenEncodeSessionEx != nullptr) {
    return true;
  }
  if (g_nvenc_lib_handle) {
    dlclose(g_nvenc_lib_handle);
    g_nvenc_lib_handle = nullptr;
  }
  memset(&g_nvenc_state.nvenc_funcs, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
  g_nvenc_state.nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;

  const char* lib_names[] = {"libnvidia-encode.so.1", "libnvidia-encode.so"};
  for (const char* name : lib_names) {
    g_nvenc_lib_handle = dlopen(name, RTLD_LAZY | RTLD_GLOBAL);
    if (g_nvenc_lib_handle) {
      break;
    }
  }

  if (!g_nvenc_lib_handle) {
    return false;
  }

  PFN_NvEncodeAPICreateInstance NvEncodeAPICreateInstance_func_ptr =
    (PFN_NvEncodeAPICreateInstance)dlsym(g_nvenc_lib_handle, "NvEncodeAPICreateInstance");

  if (!NvEncodeAPICreateInstance_func_ptr) {
    dlclose(g_nvenc_lib_handle);
    g_nvenc_lib_handle = nullptr;
    return false;
  }

  NVENCSTATUS status = NvEncodeAPICreateInstance_func_ptr(&g_nvenc_state.nvenc_funcs);
  if (status != NV_ENC_SUCCESS) {
    memset(&g_nvenc_state.nvenc_funcs, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    g_nvenc_state.nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    dlclose(g_nvenc_lib_handle);
    g_nvenc_lib_handle = nullptr;
    return false;
  }
  if (!g_nvenc_state.nvenc_funcs.nvEncOpenEncodeSessionEx) {
    memset(&g_nvenc_state.nvenc_funcs, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    g_nvenc_state.nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    dlclose(g_nvenc_lib_handle);
    g_nvenc_lib_handle = nullptr;
    return false;
  }
  return true;
}

/**
 * @brief Resets the global NVENC encoder, releasing all associated resources.
 *
 * This function is thread-safe. If the encoder is initialized, it destroys all
 * allocated input and output buffers, destroys the encoder session, and destroys
 * the associated CUDA context. It then marks the encoder as uninitialized. This
 * is called when capture settings change (e.g., resolution) or when stopping.
 */
void reset_nvenc_encoder() {
  std::lock_guard<std::mutex> lock(g_nvenc_mutex);

  if (!g_nvenc_state.initialized) {
    return;
  }

  if (g_nvenc_state.encoder_session && g_nvenc_state.nvenc_funcs.nvEncDestroyEncoder) {
    for (NV_ENC_INPUT_PTR& ptr : g_nvenc_state.input_buffers) {
        if (ptr && g_nvenc_state.nvenc_funcs.nvEncDestroyInputBuffer)
            g_nvenc_state.nvenc_funcs.nvEncDestroyInputBuffer(g_nvenc_state.encoder_session, ptr);
        ptr = nullptr;
    }
    g_nvenc_state.input_buffers.clear();

    for (NV_ENC_OUTPUT_PTR& ptr : g_nvenc_state.output_buffers) {
        if (ptr && g_nvenc_state.nvenc_funcs.nvEncDestroyBitstreamBuffer)
            g_nvenc_state.nvenc_funcs.nvEncDestroyBitstreamBuffer(g_nvenc_state.encoder_session, ptr);
        ptr = nullptr;
    }
    g_nvenc_state.output_buffers.clear();

    g_nvenc_state.nvenc_funcs.nvEncDestroyEncoder(g_nvenc_state.encoder_session);
    g_nvenc_state.encoder_session = nullptr;
  }

  if (g_nvenc_state.cuda_context && g_cuda_funcs.pfn_cuCtxDestroy) {
    g_cuda_funcs.pfn_cuCtxDestroy(g_nvenc_state.cuda_context);
    g_nvenc_state.cuda_context = nullptr;
  }

  g_nvenc_state.initialized = false;
}

/**
 * @brief Completely unloads the NVENC library and resets the encoder state.
 *
 * This function is thread-safe. It first calls `reset_nvenc_encoder` to release
 * any active session resources. Then, it calls `dlclose` on the NVENC library
 * handle and clears the global NVENC function list struct.
 */
void unload_nvenc_library_if_loaded() {
  std::unique_lock<std::mutex> lock(g_nvenc_mutex); 
  
  if (g_nvenc_state.initialized) {
    lock.unlock();
    reset_nvenc_encoder();
    lock.lock();
  }

  if (g_nvenc_lib_handle) {
    dlclose(g_nvenc_lib_handle);
    g_nvenc_lib_handle = nullptr;
    memset(&g_nvenc_state.nvenc_funcs, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    g_nvenc_state.nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;
  }
}

/**
 * @brief Initializes or re-initializes the global NVENC encoder with the specified parameters.
 *
 * This function is thread-safe. It checks if an encoder is already initialized
 * with the exact same parameters. If so, it returns true immediately. Otherwise,
 * it resets any existing encoder and proceeds to create a new CUDA context and
 * NVENC session. It configures the encoder for ultra-low-latency H.264 encoding
 * with the given dimensions, QP, FPS, and colorspace, and allocates a pool of
 * input/output buffers.
 *
 * @param width The width of the video frames to be encoded.
 * @param height The height of the video frames to be encoded.
 * @param target_qp The target Quantization Parameter (QP) for constant quality encoding.
 * @param fps The target frames per second for the encoder.
 * @param use_yuv444 If true, configures the encoder for YUV 4:4:4 (full color);
 *                   if false, uses YUV 4:2:0 (NV12).
 * @return true if the encoder was successfully initialized, false on any failure.
 */
bool initialize_nvenc_encoder(int width,
                              int height,
                              int target_qp,
                              double fps,
                              bool use_yuv444) {
  std::lock_guard<std::mutex> lock(g_nvenc_mutex);

  NV_ENC_BUFFER_FORMAT target_buffer_format =
    use_yuv444 ? NV_ENC_BUFFER_FORMAT_YUV444 : NV_ENC_BUFFER_FORMAT_NV12;

  if (g_nvenc_state.initialized && g_nvenc_state.initialized_width == width &&
      g_nvenc_state.initialized_height == height &&
      g_nvenc_state.initialized_qp == target_qp &&
      g_nvenc_state.initialized_buffer_format == target_buffer_format) {
    return true;
  }

  if (g_nvenc_state.initialized) {
    g_nvenc_mutex.unlock();
    reset_nvenc_encoder();
    g_nvenc_mutex.lock();
  }

  if (!LoadCudaApi()) {
    std::cerr << "NVENC_INIT_FATAL: Failed to load CUDA driver API." << std::endl;
    return false;
  }

  if (!g_nvenc_state.nvenc_funcs.nvEncOpenEncodeSessionEx) {
    g_nvenc_state.initialized = false;
    return false;
  }

  CUresult cu_status = g_cuda_funcs.pfn_cuInit(0);
  if (cu_status != CUDA_SUCCESS) {
      std::cerr << "NVENC_INIT_ERROR: cuInit failed with code " << cu_status << std::endl;
      return false;
  }
  CUdevice cu_device;
  cu_status = g_cuda_funcs.pfn_cuDeviceGet(&cu_device, 0);
  if (cu_status != CUDA_SUCCESS) {
      std::cerr << "NVENC_INIT_ERROR: cuDeviceGet failed with code " << cu_status << std::endl;
      return false;
  }
  cu_status = g_cuda_funcs.pfn_cuCtxCreate(&g_nvenc_state.cuda_context, 0, cu_device);
  if (cu_status != CUDA_SUCCESS) {
      std::cerr << "NVENC_INIT_ERROR: cuCtxCreate failed with code " << cu_status << std::endl;
      return false;
  }

  NVENCSTATUS status;
  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS session_params = {0};
  session_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
  session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
  session_params.device = g_nvenc_state.cuda_context;
  session_params.apiVersion = NVENCAPI_VERSION;

  status = g_nvenc_state.nvenc_funcs.nvEncOpenEncodeSessionEx(
    &session_params, &g_nvenc_state.encoder_session);

  if (status != NV_ENC_SUCCESS) {
    std::string error_str = "NVENC_INIT_ERROR: nvEncOpenEncodeSessionEx (CUDA Path) FAILED: " + std::to_string(status);
    std::cerr << error_str << std::endl;
    g_nvenc_state.encoder_session = nullptr;
    g_nvenc_mutex.unlock();
    reset_nvenc_encoder();
    g_nvenc_mutex.lock();
    return false;
  }
  if (!g_nvenc_state.encoder_session) {
    g_nvenc_mutex.unlock();
    reset_nvenc_encoder();
    g_nvenc_mutex.lock();
    return false;
  }

  memset(&g_nvenc_state.init_params, 0, sizeof(g_nvenc_state.init_params));
  g_nvenc_state.init_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
  g_nvenc_state.init_params.encodeGUID = NV_ENC_CODEC_H264_GUID;
  g_nvenc_state.init_params.presetGUID = NV_ENC_PRESET_P3_GUID;
  g_nvenc_state.init_params.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
  g_nvenc_state.init_params.encodeWidth = width;
  g_nvenc_state.init_params.encodeHeight = height;
  g_nvenc_state.init_params.darWidth = width;
  g_nvenc_state.init_params.darHeight = height;
  g_nvenc_state.init_params.frameRateNum = static_cast<uint32_t>(fps < 1.0 ? 30 : fps);
  g_nvenc_state.init_params.frameRateDen = 1;
  g_nvenc_state.init_params.enablePTD = 1;

  NV_ENC_PRESET_CONFIG preset_config = {0};
  preset_config.version = NV_ENC_PRESET_CONFIG_VER;
  preset_config.presetCfg.version = NV_ENC_CONFIG_VER;

  if (g_nvenc_state.nvenc_funcs.nvEncGetEncodePresetConfigEx) {
    status = g_nvenc_state.nvenc_funcs.nvEncGetEncodePresetConfigEx(
      g_nvenc_state.encoder_session,
      g_nvenc_state.init_params.encodeGUID,
      g_nvenc_state.init_params.presetGUID,
      g_nvenc_state.init_params.tuningInfo,
      &preset_config);

    if (status != NV_ENC_SUCCESS) {
      std::cerr << "NVENC_INIT_WARN: nvEncGetEncodePresetConfigEx FAILED: " << status
                << ". Falling back to manual config." << std::endl;
      memset(&g_nvenc_state.encode_config, 0, sizeof(g_nvenc_state.encode_config));
      g_nvenc_state.encode_config.version = NV_ENC_CONFIG_VER;
    } else {
      g_nvenc_state.encode_config = preset_config.presetCfg;
      g_nvenc_state.encode_config.version = NV_ENC_CONFIG_VER;
    }
  } else {
    std::cerr << "NVENC_INIT_WARN: nvEncGetEncodePresetConfigEx not available. Using manual "
                 "config."
              << std::endl;
    memset(&g_nvenc_state.encode_config, 0, sizeof(g_nvenc_state.encode_config));
    g_nvenc_state.encode_config.version = NV_ENC_CONFIG_VER;
  }

  g_nvenc_state.encode_config.profileGUID =
    use_yuv444 ? NV_ENC_H264_PROFILE_HIGH_444_GUID : NV_ENC_H264_PROFILE_HIGH_GUID;
  g_nvenc_state.encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
  g_nvenc_state.encode_config.rcParams.constQP.qpInterP = target_qp;
  g_nvenc_state.encode_config.rcParams.constQP.qpIntra = target_qp;
  g_nvenc_state.encode_config.rcParams.constQP.qpInterB = target_qp;
  g_nvenc_state.encode_config.gopLength = NVENC_INFINITE_GOPLENGTH;
  g_nvenc_state.encode_config.frameIntervalP = 1;

  NV_ENC_CONFIG_H264* h264_cfg = &g_nvenc_state.encode_config.encodeCodecConfig.h264Config;
  h264_cfg->chromaFormatIDC = use_yuv444 ? 3 : 1;
  h264_cfg->h264VUIParameters.videoFullRangeFlag = use_yuv444 ? 1 : 0;
  g_nvenc_state.init_params.encodeConfig = &g_nvenc_state.encode_config;

  status = g_nvenc_state.nvenc_funcs.nvEncInitializeEncoder(g_nvenc_state.encoder_session,
                                                            &g_nvenc_state.init_params);
  if (status != NV_ENC_SUCCESS) {
    std::string error_str =
      "NVENC_INIT_ERROR: nvEncInitializeEncoder FAILED: " + std::to_string(status);
    if (g_nvenc_state.nvenc_funcs.nvEncGetLastErrorString) {
      const char* api_err =
        g_nvenc_state.nvenc_funcs.nvEncGetLastErrorString(g_nvenc_state.encoder_session);
      if (api_err)
        error_str += " - API Error: " + std::string(api_err);
    }
    std::cerr << error_str << std::endl;

    g_nvenc_mutex.unlock();
    reset_nvenc_encoder();
    g_nvenc_mutex.lock();
    return false;
  }

  g_nvenc_state.input_buffers.resize(g_nvenc_state.buffer_pool_size);
  g_nvenc_state.output_buffers.resize(g_nvenc_state.buffer_pool_size);
  for (int i = 0; i < g_nvenc_state.buffer_pool_size; ++i) {
    NV_ENC_CREATE_INPUT_BUFFER icp = {0};
    icp.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
    icp.width = width;
    icp.height = height;
    icp.bufferFmt = target_buffer_format;
    status = g_nvenc_state.nvenc_funcs.nvEncCreateInputBuffer(g_nvenc_state.encoder_session,
                                                              &icp);
    if (status != NV_ENC_SUCCESS) {
      g_nvenc_mutex.unlock();
      reset_nvenc_encoder();
      g_nvenc_mutex.lock();
      return false;
    }
    g_nvenc_state.input_buffers[i] = icp.inputBuffer;
    NV_ENC_CREATE_BITSTREAM_BUFFER ocp = {0};
    ocp.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    status = g_nvenc_state.nvenc_funcs.nvEncCreateBitstreamBuffer(
      g_nvenc_state.encoder_session, &ocp);
    if (status != NV_ENC_SUCCESS) {
      g_nvenc_mutex.unlock();
      reset_nvenc_encoder();
      g_nvenc_mutex.lock();
      return false;
    }
    g_nvenc_state.output_buffers[i] = ocp.bitstreamBuffer;
  }
  g_nvenc_state.initialized_width = width;
  g_nvenc_state.initialized_height = height;
  g_nvenc_state.initialized_qp = target_qp;
  g_nvenc_state.initialized_buffer_format = target_buffer_format;
  g_nvenc_state.initialized = true;
  return true;
}

/**
 * @brief Encodes a full frame of YUV data using the pre-initialized global NVENC encoder.
 *
 * This function is thread-safe. It locks an available input buffer from the pool,
 * copies the provided Y, U, and V plane data into it (converting to NV12 if the
 * input is I420), and then submits it to the encoder. After encoding, it locks the
 * corresponding output bitstream buffer, packages the H.264 data into a
 * `StripeEncodeResult` with a custom header, and returns it. It manages a circular
 * pool of input/output buffers to pipeline encoding operations.
 *
 * @param width The width of the input frame.
 * @param height The height of the input frame.
 * @param y_plane Pointer to the Y (luma) plane data.
 * @param y_stride Stride of the Y plane in bytes.
 * @param u_plane Pointer to the U (chroma) plane data.
 * @param u_stride Stride of the U plane in bytes.
 * @param v_plane Pointer to the V (chroma) plane data.
 * @param v_stride Stride of the V plane in bytes.
 * @param is_i444 True if the input is YUV 4:4:4, false if it is YUV 4:2:0.
 * @param frame_counter The current frame number, used for timestamping.
 * @param force_idr_frame If true, forces the encoder to generate an IDR (key) frame.
 * @return A `StripeEncodeResult` containing the encoded H.264 NAL units and a
 *         custom header. The result's data buffer is dynamically allocated and
 *         must be freed by the caller.
 * @throws std::runtime_error if any NVENC API call fails during the encoding process.
 */
StripeEncodeResult encode_fullframe_nvenc(int width,
                                          int height,
                                          const uint8_t* y_plane, int y_stride,
                                          const uint8_t* u_plane, int u_stride,
                                          const uint8_t* v_plane, int v_stride,
                                          bool is_i444,
                                          int frame_counter,
                                          bool force_idr_frame) {
  StripeEncodeResult result;
  result.type = StripeDataType::H264;
  result.stripe_y_start = 0;
  result.stripe_height = height;
  result.frame_id = frame_counter;

  std::lock_guard<std::mutex> lock(g_nvenc_mutex);

  if (!g_nvenc_state.initialized) {
    throw std::runtime_error("NVENC_ENCODE_FATAL: Not initialized.");
  }

  NV_ENC_INPUT_PTR in_ptr =
    g_nvenc_state.input_buffers[g_nvenc_state.current_input_buffer_idx];
  NV_ENC_OUTPUT_PTR out_ptr =
    g_nvenc_state.output_buffers[g_nvenc_state.current_output_buffer_idx];

  NV_ENC_LOCK_INPUT_BUFFER lip = {0};
  lip.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
  lip.inputBuffer = in_ptr;
  NVENCSTATUS status =
    g_nvenc_state.nvenc_funcs.nvEncLockInputBuffer(g_nvenc_state.encoder_session, &lip);
  if (status != NV_ENC_SUCCESS)
    throw std::runtime_error("NVENC_ENCODE_ERROR: nvEncLockInputBuffer FAILED: " +
                             std::to_string(status));

  unsigned char* locked_buffer = static_cast<unsigned char*>(lip.bufferDataPtr);
  int locked_pitch = lip.pitch;

  if (is_i444) {
    uint8_t* y_dst = locked_buffer;
    uint8_t* u_dst = locked_buffer + static_cast<size_t>(locked_pitch) * height;
    uint8_t* v_dst = u_dst + static_cast<size_t>(locked_pitch) * height;
    libyuv::CopyPlane(y_plane, y_stride, y_dst, locked_pitch, width, height);
    libyuv::CopyPlane(u_plane, u_stride, u_dst, locked_pitch, width, height);
    libyuv::CopyPlane(v_plane, v_stride, v_dst, locked_pitch, width, height);
  } else {
    uint8_t* y_dst = locked_buffer;
    uint8_t* uv_dst = locked_buffer + static_cast<size_t>(locked_pitch) * height;
    libyuv::I420ToNV12(y_plane, y_stride, u_plane, u_stride, v_plane, v_stride,
                        y_dst, locked_pitch, uv_dst, locked_pitch, width, height);
  }

  g_nvenc_state.nvenc_funcs.nvEncUnlockInputBuffer(g_nvenc_state.encoder_session, in_ptr);

  NV_ENC_PIC_PARAMS pp = {0};
  pp.version = NV_ENC_PIC_PARAMS_VER;
  pp.inputBuffer = in_ptr;
  pp.outputBitstream = out_ptr;
  pp.bufferFmt = g_nvenc_state.initialized_buffer_format;
  pp.inputWidth = width;
  pp.inputHeight = height;
  pp.inputPitch = locked_pitch;
  pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
  pp.inputTimeStamp = frame_counter;
  pp.frameIdx = frame_counter;
  if (force_idr_frame) {
    pp.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;
  }

  status =
    g_nvenc_state.nvenc_funcs.nvEncEncodePicture(g_nvenc_state.encoder_session, &pp);
  if (status != NV_ENC_SUCCESS) {
    std::string err_msg = "NVENC_ENCODE_ERROR: nvEncEncodePicture FAILED: " + std::to_string(status);
    throw std::runtime_error(err_msg);
  }

  NV_ENC_LOCK_BITSTREAM lbs = {0};
  lbs.version = NV_ENC_LOCK_BITSTREAM_VER;
  lbs.outputBitstream = out_ptr;
  status =
    g_nvenc_state.nvenc_funcs.nvEncLockBitstream(g_nvenc_state.encoder_session, &lbs);
  if (status != NV_ENC_SUCCESS) {
    throw std::runtime_error("NVENC_ENCODE_ERROR: nvEncLockBitstream FAILED: " + std::to_string(status));
  }

  if (lbs.bitstreamSizeInBytes > 0) {
    const unsigned char TAG = 0x04;
    unsigned char type_hdr = 0x00;
    if (lbs.pictureType == NV_ENC_PIC_TYPE_IDR) type_hdr = 0x01;
    else if (lbs.pictureType == NV_ENC_PIC_TYPE_I) type_hdr = 0x02;

    int header_sz = 10;
    result.data = new unsigned char[lbs.bitstreamSizeInBytes + header_sz];
    result.size = lbs.bitstreamSizeInBytes + header_sz;
    result.data[0] = TAG;
    result.data[1] = type_hdr;
    uint16_t net_val = htons(static_cast<uint16_t>(result.frame_id % 65536));
    std::memcpy(result.data + 2, &net_val, 2);
    net_val = htons(static_cast<uint16_t>(result.stripe_y_start));
    std::memcpy(result.data + 4, &net_val, 2);
    net_val = htons(static_cast<uint16_t>(width));
    std::memcpy(result.data + 6, &net_val, 2);
    net_val = htons(static_cast<uint16_t>(height));
    std::memcpy(result.data + 8, &net_val, 2);
    std::memcpy(result.data + header_sz, lbs.bitstreamBufferPtr, lbs.bitstreamSizeInBytes);
  } else {
    result.size = 0;
    result.data = nullptr;
  }

  g_nvenc_state.nvenc_funcs.nvEncUnlockBitstream(g_nvenc_state.encoder_session, out_ptr);

  g_nvenc_state.current_input_buffer_idx = (g_nvenc_state.current_input_buffer_idx + 1) % g_nvenc_state.buffer_pool_size;
  g_nvenc_state.current_output_buffer_idx = (g_nvenc_state.current_output_buffer_idx + 1) % g_nvenc_state.buffer_pool_size;

  return result;
}

/**
 * @brief Dynamically loads the VA-API libraries and resolves required function pointers.
 *
 * This function uses `dlopen` to load `libva.so.2` (or `.1`) and its backend
 * libraries like `libva-drm.so.2`. It then uses `dlsym` to find the addresses
 * of all necessary VA-API functions and stores them in the global
 * `g_vaapi_state.va_funcs` struct. This must be called successfully before
 * any other VA-API operations.
 *
 * @return true if all libraries were loaded and functions were resolved, false otherwise.
 */
bool LoadVaapiApi() {
    if (g_vaapi_state.va_funcs.vaInitialize) {
        return true;
    }

    g_vaapi_state.va_funcs.va_lib_handle = dlopen("libva.so.2", RTLD_LAZY);
    if (!g_vaapi_state.va_funcs.va_lib_handle) {
        g_vaapi_state.va_funcs.va_lib_handle = dlopen("libva.so.1", RTLD_LAZY);
    }
    if (!g_vaapi_state.va_funcs.va_lib_handle) {
        std::cerr << "VAAPI_API_LOAD: dlopen failed for libva.so" << std::endl;
        return false;
    }

    g_vaapi_state.va_funcs.va_drm_lib_handle = dlopen("libva-drm.so.2", RTLD_LAZY);
    if (!g_vaapi_state.va_funcs.va_drm_lib_handle) {
        g_vaapi_state.va_funcs.va_drm_lib_handle = dlopen("libva-drm.so.1", RTLD_LAZY);
    }
    if (!g_vaapi_state.va_funcs.va_drm_lib_handle) {
        std::cerr << "VAAPI_API_LOAD: dlopen failed for libva-drm.so" << std::endl;
        dlclose(g_vaapi_state.va_funcs.va_lib_handle);
        g_vaapi_state.va_funcs.va_lib_handle = nullptr;
        return false;
    }

    g_vaapi_state.va_funcs.va_x11_lib_handle = dlopen("libva-x11.so.2", RTLD_LAZY);
    if (!g_vaapi_state.va_funcs.va_x11_lib_handle) {
        g_vaapi_state.va_funcs.va_x11_lib_handle = dlopen("libva-x11.so.1", RTLD_LAZY);
    }

    auto unload_all_and_fail = [&]() {
        std::cerr << "VAAPI_API_LOAD: dlsym failed for one or more functions." << std::endl;
        if (g_vaapi_state.va_funcs.va_lib_handle) dlclose(g_vaapi_state.va_funcs.va_lib_handle);
        if (g_vaapi_state.va_funcs.va_x11_lib_handle) dlclose(g_vaapi_state.va_funcs.va_x11_lib_handle);
        if (g_vaapi_state.va_funcs.va_drm_lib_handle) dlclose(g_vaapi_state.va_funcs.va_drm_lib_handle);
        g_vaapi_state.va_funcs = {};
        return false;
    };

    #define LOAD_VA_FUNC(lib, name) \
        g_vaapi_state.va_funcs.name = (decltype(g_vaapi_state.va_funcs.name))dlsym(lib, #name); \
        if (!g_vaapi_state.va_funcs.name) return unload_all_and_fail()

    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_drm_lib_handle, vaGetDisplayDRM);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaInitialize);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaTerminate);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaQueryVendorString);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaCreateConfig);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaDestroyConfig);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaCreateSurfaces);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaDestroySurfaces);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaCreateContext);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaDestroyContext);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaCreateBuffer);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaDestroyBuffer);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaBeginPicture);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaRenderPicture);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaEndPicture);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaSyncSurface);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaMapBuffer);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaUnmapBuffer);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaDeriveImage);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaDestroyImage);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaCreateImage);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaPutImage);
    LOAD_VA_FUNC(g_vaapi_state.va_funcs.va_lib_handle, vaGetConfigAttributes);

    #undef LOAD_VA_FUNC

    return true;
}

/**
 * @brief Resets the global VA-API encoder, releasing all associated resources.
 *
 * This function is thread-safe. If the encoder is initialized, it destroys the
 * VA context, configuration, surfaces, and coded buffer. It also terminates
 * the display connection and closes the DRM device file descriptor. This is
 * called when capture settings change or when stopping the capture.
 */
void UnloadVaapiApi() {
    if (g_vaapi_state.va_funcs.va_lib_handle) {
        dlclose(g_vaapi_state.va_funcs.va_lib_handle);
    }
    if (g_vaapi_state.va_funcs.va_x11_lib_handle) {
        dlclose(g_vaapi_state.va_funcs.va_x11_lib_handle);
    }
    if (g_vaapi_state.va_funcs.va_drm_lib_handle) {
        dlclose(g_vaapi_state.va_funcs.va_drm_lib_handle);
    }
    g_vaapi_state.va_funcs = {};
}

/**
 * @brief Resets the global VA-API encoder, releasing all associated resources.
 *
 * This function is thread-safe. If the encoder is initialized, it destroys the
 * VA context, configuration, surfaces, and coded buffer. It also terminates
 * the display connection and closes the DRM device file descriptor. This is
 * called when capture settings change or when stopping the capture.
 */
void reset_vaapi_encoder() {
    std::lock_guard<std::mutex> lock(g_vaapi_mutex);
    if (!g_vaapi_state.initialized) {
        return;
    }

    auto& funcs = g_vaapi_state.va_funcs;
    if (g_vaapi_state.context_id != VA_INVALID_ID) {
        funcs.vaDestroyContext(g_vaapi_state.display, g_vaapi_state.context_id);
    }
    if (g_vaapi_state.config_id != VA_INVALID_ID) {
        funcs.vaDestroyConfig(g_vaapi_state.display, g_vaapi_state.config_id);
    }
    if (!g_vaapi_state.surfaces.empty()) {
        funcs.vaDestroySurfaces(g_vaapi_state.display, g_vaapi_state.surfaces.data(), g_vaapi_state.surfaces.size());
    }
    if (g_vaapi_state.coded_buffer_id != VA_INVALID_ID) {
        funcs.vaDestroyBuffer(g_vaapi_state.display, g_vaapi_state.coded_buffer_id);
    }
    if (g_vaapi_state.display) {
        funcs.vaTerminate(g_vaapi_state.display);
    }
    if (g_vaapi_state.fd >= 0) {
        close(g_vaapi_state.fd);
    }

    g_vaapi_state = {};
}

/**
 * @brief Completely unloads the VA-API libraries and resets the encoder state.
 *
 * This function is thread-safe. It first calls `reset_vaapi_encoder` to release
 * any active session resources, then calls `UnloadVaapiApi` to `dlclose` the
 * library handles, ensuring a full cleanup.
 */
void unload_vaapi_library_if_loaded() {
    std::unique_lock<std::mutex> lock(g_vaapi_mutex);
    
    if (g_vaapi_state.initialized) {
        lock.unlock();
        reset_vaapi_encoder();
        lock.lock();
    }
    UnloadVaapiApi();
}

/**
 * @brief Scans the system for available VA-API compatible DRM render nodes.
 *
 * This function searches the `/dev/dri/` directory for device files named
 * `renderD*`, which represent GPU render nodes that can be used for
 * hardware-accelerated computation like video encoding without needing a
 * graphical display server.
 *
 * @return A sorted vector of strings, where each string is the full path to a
 *         found render node (e.g., "/dev/dri/renderD128").
 */
std::vector<std::string> find_vaapi_render_nodes() {
    std::vector<std::string> nodes;
    const char* drm_dir_path = "/dev/dri/";
    DIR *dir = opendir(drm_dir_path);
    if (!dir) {
        return nodes;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (strncmp(entry->d_name, "renderD", 7) == 0) {
            nodes.push_back(std::string(drm_dir_path) + entry->d_name);
        }
    }
    closedir(dir);
    std::sort(nodes.begin(), nodes.end());
    return nodes;
}

/**
 * @brief Initializes or re-initializes the global VA-API encoder with specified parameters.
 *
 * This function is thread-safe. It checks if an encoder is already initialized with
 * the same parameters. If not, it resets any existing encoder and proceeds to
 * set up a new VA-API encoding session via the DRM backend. This involves:
 * 1. Finding and opening the specified DRM render node.
 * 2. Getting a `VADisplay` from the file descriptor.
 * 3. Initializing the VA-API library connection.
 * 4. Creating an encoder configuration for H.264 with appropriate attributes (e.g., CQP).
 * 5. Allocating a pool of `VASurfaceID`s to hold input video frames.
 * 6. Creating the main encoding `VAContextID`.
 *
 * @param render_node_idx The index of the DRM render node to use from the list
 *                        found by `find_vaapi_render_nodes`.
 * @param width The width of the video frames to be encoded.
 * @param height The height of the video frames to be encoded.
 * @param qp The target Quantization Parameter (QP) for constant quality encoding.
 * @return true if the encoder was successfully initialized, false on any failure.
 */
bool initialize_vaapi_encoder(int render_node_idx, int width, int height, int qp) {
    std::unique_lock<std::mutex> lock(g_vaapi_mutex);

    if (g_vaapi_state.initialized && g_vaapi_state.initialized_width == width &&
        g_vaapi_state.initialized_height == height && g_vaapi_state.initialized_qp == qp) {
        return true;
    }

    if (g_vaapi_state.initialized) {
        lock.unlock();
        reset_vaapi_encoder();
        lock.lock();
    }

    if (!LoadVaapiApi()) {
        std::cerr << "VAAPI_INIT: Failed to load VAAPI libraries." << std::endl;
        return false;
    }

    auto& funcs = g_vaapi_state.va_funcs;
    std::vector<std::string> nodes = find_vaapi_render_nodes();
    if (nodes.empty()) {
        std::cerr << "VAAPI_INIT: No /dev/dri/renderD nodes found." << std::endl;
        return false;
    }

    std::string node_to_use = (render_node_idx >= 0 && render_node_idx < (int)nodes.size()) ? nodes[render_node_idx] : nodes[0];
    std::cout << "VAAPI_INIT: Using render node: " << node_to_use << std::endl;

    g_vaapi_state.fd = open(node_to_use.c_str(), O_RDWR);
    if (g_vaapi_state.fd < 0) {
        std::cerr << "VAAPI_INIT: Failed to open " << node_to_use << std::endl;
        return false;
    }

    g_vaapi_state.display = funcs.vaGetDisplayDRM(g_vaapi_state.fd);
    if (!g_vaapi_state.display) {
        std::cerr << "VAAPI_INIT: vaGetDisplayDRM failed." << std::endl;
        close(g_vaapi_state.fd);
        g_vaapi_state.fd = -1;
        return false;
    }

    int major_ver, minor_ver;
    VAStatus status = funcs.vaInitialize(g_vaapi_state.display, &major_ver, &minor_ver);
    if (status != VA_STATUS_SUCCESS) {
        std::cerr << "VAAPI_INIT: vaInitialize failed: " << status << std::endl;
        return false;
    }
    std::cout << "libva info: VA-API version " << major_ver << "." << minor_ver << ".0" << std::endl;

    VAProfile va_profile = VAProfileH264ConstrainedBaseline;
    VAEntrypoint entrypoint = VAEntrypointEncSlice;
    std::vector<VAConfigAttrib> attribs;
    attribs.push_back({VAConfigAttribRTFormat, VA_RT_FORMAT_YUV420});

    VAConfigAttrib query_attrib;
    query_attrib.type = VAConfigAttribRateControl;
    if (funcs.vaGetConfigAttributes(g_vaapi_state.display, va_profile, entrypoint, &query_attrib, 1) == VA_STATUS_SUCCESS &&
        (query_attrib.value & VA_RC_CQP)) {
        std::cout << "VAAPI_INIT: Driver supports CQP rate control." << std::endl;
        attribs.push_back({VAConfigAttribRateControl, VA_RC_CQP});
    } else {
        std::cout << "VAAPI_INIT: Driver does NOT support CQP. Skipping rate control attribute." << std::endl;
    }

    query_attrib.type = VAConfigAttribEncPackedHeaders;
    if (funcs.vaGetConfigAttributes(g_vaapi_state.display, va_profile, entrypoint, &query_attrib, 1) == VA_STATUS_SUCCESS &&
        (query_attrib.value & VA_ENC_PACKED_HEADER_DATA)) {
        std::cout << "VAAPI_INIT: Driver supports packed headers." << std::endl;
        attribs.push_back({VAConfigAttribEncPackedHeaders, VA_ENC_PACKED_HEADER_DATA});
    } else {
        std::cout << "VAAPI_INIT: Driver does NOT support packed headers. Skipping attribute." << std::endl;
    }

    status = funcs.vaCreateConfig(g_vaapi_state.display, va_profile, entrypoint, attribs.data(), attribs.size(), &g_vaapi_state.config_id);
    if (status != VA_STATUS_SUCCESS) {
        std::cerr << "VAAPI_INIT: vaCreateConfig failed with Baseline profile: " << status << ". Trying VAProfileH264Main..." << std::endl;
        va_profile = VAProfileH264Main;
        status = funcs.vaCreateConfig(g_vaapi_state.display, va_profile, entrypoint, attribs.data(), attribs.size(), &g_vaapi_state.config_id);
        if (status != VA_STATUS_SUCCESS) {
            std::cerr << "VAAPI_INIT: vaCreateConfig failed with Main profile too: " << status << std::endl;
            std::cerr << "VAAPI_INIT: Retrying with ONLY VAConfigAttribRTFormat..." << std::endl;
            VAConfigAttrib minimal_attrib = {VAConfigAttribRTFormat, VA_RT_FORMAT_YUV420};
            status = funcs.vaCreateConfig(g_vaapi_state.display, va_profile, entrypoint, &minimal_attrib, 1, &g_vaapi_state.config_id);
            if (status != VA_STATUS_SUCCESS) {
                std::cerr << "VAAPI_INIT: Failed even with minimal config. Error: " << status << std::endl;
                return false;
            }
            std::cout << "VAAPI_INIT: Minimal config created successfully. Some features may be disabled." << std::endl;
        }
    }

    const unsigned int num_surfaces = 4;
    g_vaapi_state.surfaces.resize(num_surfaces);
    status = funcs.vaCreateSurfaces(g_vaapi_state.display, VA_RT_FORMAT_YUV420, width, height, g_vaapi_state.surfaces.data(), num_surfaces, nullptr, 0);
    if (status != VA_STATUS_SUCCESS) {
        std::cerr << "VAAPI_INIT: vaCreateSurfaces failed: " << status << std::endl;
        return false;
    }

    status = funcs.vaCreateContext(g_vaapi_state.display, g_vaapi_state.config_id, width, height, VA_PROGRESSIVE, nullptr, 0, &g_vaapi_state.context_id);
    if (status != VA_STATUS_SUCCESS) {
        std::cerr << "VAAPI_INIT: vaCreateContext failed: " << status << std::endl;
        return false;
    }

    g_vaapi_state.initialized = true;
    g_vaapi_state.initialized_width = width;
    g_vaapi_state.initialized_height = height;
    g_vaapi_state.initialized_qp = qp;
    g_vaapi_state.frame_count = 0;
    std::cout << "VAAPI encoder initialized successfully via DRM backend." << std::endl;
    return true;
}

/**
 * @brief Helper function to calculate log2(N) - 4 for H.264 SPS header fields.
 *
 * The H.264 specification requires certain fields in the Sequence Parameter Set
 * (SPS), like `log2_max_frame_num_minus4`, to be encoded in this format. This
 * function computes the value, clamping it to the valid range required by the spec.
 *
 * @param num The input number (e.g., GOP size or max picture order count).
 * @return The calculated value suitable for the SPS field.
 */
static unsigned int get_log2_val_minus4(unsigned int num) {
    unsigned int ret = 0;
    while (num > 0) {
        ret++;
        num >>= 1;
    }
    if (ret < 4) ret = 4;
    if (ret > 16) ret = 16;
    return ret - 4;
}

/**
 * @brief Encodes a full frame of YUV data using the pre-initialized global VA-API encoder.
 *
 * This function is thread-safe. It takes raw I420 YUV data and performs one
 * full frame encoding cycle. The process includes:
 * 1. Uploading the I420 data to a hardware surface in NV12 format via `vaPutImage`.
 * 2. Setting up parameter buffers: SPS (on IDR frames), PPS, and Slice parameters.
 * 3. Executing the encoding pipeline with `vaBeginPicture`, `vaRenderPicture`, `vaEndPicture`.
 * 4. Syncing the resulting surface and mapping the output coded buffer to get the bitstream.
 * 5. Packaging the H.264 bitstream into a `StripeEncodeResult` with a custom header.
 *
 * @param width The width of the input frame.
 * @param height The height of the input frame.
 * @param fps The target frames per second, used for SPS timing info.
 * @param y_plane Pointer to the Y (luma) plane data.
 * @param y_stride Stride of the Y plane.
 * @param u_plane Pointer to the U (chroma) plane data.
 * @param u_stride Stride of the U plane.
 * @param v_plane Pointer to the V (chroma) plane data.
 * @param v_stride Stride of the V plane.
 * @param frame_counter The current frame number.
 * @param force_idr_frame If true, forces the encoder to generate an IDR (key) frame
 *                        and include SPS/PPS headers in the bitstream.
 * @return A `StripeEncodeResult` containing the encoded H.264 data.
 * @throws std::runtime_error if any VA-API call fails during the encoding process.
 */
StripeEncodeResult encode_fullframe_vaapi(int width, int height, double fps,
                                          const uint8_t* y_plane, int y_stride,
                                          const uint8_t* u_plane, int u_stride,
                                          const uint8_t* v_plane, int v_stride,
                                          int frame_counter,
                                          bool force_idr_frame) {
    StripeEncodeResult result;
    result.type = StripeDataType::H264;
    result.stripe_y_start = 0;
    result.stripe_height = height;
    result.frame_id = frame_counter;

    std::lock_guard<std::mutex> lock(g_vaapi_mutex);
    if (!g_vaapi_state.initialized) {
        throw std::runtime_error("VAAPI_ENCODE_FATAL: Not initialized.");
    }

    auto& funcs = g_vaapi_state.va_funcs;
    VASurfaceID current_surface = g_vaapi_state.surfaces[g_vaapi_state.frame_count % g_vaapi_state.surfaces.size()];
    VAStatus status;

    if (g_vaapi_state.frame_count == 0) {
        g_vaapi_state.last_ref_pic = {VA_INVALID_ID, VA_PICTURE_H264_INVALID, 0, 0, 0};
    }

    {
        VAImage image = {};
        image.format.fourcc = VA_FOURCC_NV12;
        image.width = width;
        image.height = height;

        status = funcs.vaCreateImage(g_vaapi_state.display, &image.format, width, height, &image);
        if (status != VA_STATUS_SUCCESS) throw std::runtime_error("vaCreateImage failed: " + std::to_string(status));

        void *image_ptr = nullptr;
        status = funcs.vaMapBuffer(g_vaapi_state.display, image.buf, &image_ptr);
        if (status != VA_STATUS_SUCCESS) {
            funcs.vaDestroyImage(g_vaapi_state.display, image.image_id);
            throw std::runtime_error("vaMapBuffer for VAImage failed: " + std::to_string(status));
        }

        uint8_t* y_dest = (uint8_t*)image_ptr + image.offsets[0];
        uint8_t* uv_dest = (uint8_t*)image_ptr + image.offsets[1];
        libyuv::I420ToNV12(y_plane, y_stride, u_plane, u_stride, v_plane, v_stride,
                           y_dest, image.pitches[0], uv_dest, image.pitches[1], width, height);
        
        funcs.vaUnmapBuffer(g_vaapi_state.display, image.buf);
        status = funcs.vaPutImage(g_vaapi_state.display, current_surface, image.image_id,
                                  0, 0, width, height, 0, 0, width, height);
        funcs.vaDestroyImage(g_vaapi_state.display, image.image_id);
        if (status != VA_STATUS_SUCCESS) throw std::runtime_error("vaPutImage failed: " + std::to_string(status));
    }

    if (g_vaapi_state.coded_buffer_id == VA_INVALID_ID) {
        status = funcs.vaCreateBuffer(g_vaapi_state.display, g_vaapi_state.context_id, VAEncCodedBufferType,
                                      width * height * 3 / 2, 1, nullptr, &g_vaapi_state.coded_buffer_id);
        if (status != VA_STATUS_SUCCESS) throw std::runtime_error("vaCreateBuffer for coded buffer failed.");
    }

    std::vector<VABufferID> param_buffers;
    try {
        if (force_idr_frame) {
            VAEncSequenceParameterBufferH264 sps = {};
            const unsigned int gop_size = 30;
            const unsigned int max_ref_frames_in_gop = 1;

            sps.seq_parameter_set_id = 0;
            sps.level_idc = 41;
            sps.intra_idr_period = gop_size;
            sps.intra_period = gop_size;
            sps.ip_period = 1;
            sps.bits_per_second = 0;
            sps.max_num_ref_frames = max_ref_frames_in_gop;
            sps.picture_width_in_mbs = (width + 15) / 16;
            sps.picture_height_in_mbs = (height + 15) / 16;
            sps.seq_fields.bits.chroma_format_idc = 1;
            sps.seq_fields.bits.frame_mbs_only_flag = 1;
            sps.seq_fields.bits.direct_8x8_inference_flag = 1;
            sps.seq_fields.bits.pic_order_cnt_type = 0;
            unsigned int log2_max_frame_num_val = get_log2_val_minus4(gop_size);
            sps.seq_fields.bits.log2_max_frame_num_minus4 = log2_max_frame_num_val;
            unsigned int poc_val = 1 << (log2_max_frame_num_val + 4 + 1);
            sps.seq_fields.bits.log2_max_pic_order_cnt_lsb_minus4 = get_log2_val_minus4(poc_val);
            sps.vui_parameters_present_flag = 1;
            sps.vui_fields.bits.timing_info_present_flag = 1;
            sps.vui_fields.bits.fixed_frame_rate_flag = 1;
            sps.vui_fields.bits.bitstream_restriction_flag = 1;
            sps.vui_fields.bits.motion_vectors_over_pic_boundaries_flag = 1;
            sps.vui_fields.bits.aspect_ratio_info_present_flag = 1;
            sps.aspect_ratio_idc = 255;
            sps.sar_width = 1;
            sps.sar_height = 1;
            sps.num_units_in_tick = 1;
            sps.time_scale = static_cast<unsigned int>(fps * 2);

            VABufferID buf_id;
            if (funcs.vaCreateBuffer(g_vaapi_state.display, g_vaapi_state.context_id, VAEncSequenceParameterBufferType, sizeof(sps), 1, &sps, &buf_id) != VA_STATUS_SUCCESS) throw std::runtime_error("vaCreateBuffer for SPS failed.");
            param_buffers.push_back(buf_id);
        }

        {
            VAEncPictureParameterBufferH264 pps = {};
            pps.CurrPic = {current_surface, g_vaapi_state.frame_count, 0, static_cast<int32_t>(g_vaapi_state.frame_count * 2), static_cast<int32_t>(g_vaapi_state.frame_count * 2)};
            for (int i = 0; i < 16; ++i) pps.ReferenceFrames[i] = {VA_INVALID_ID, VA_PICTURE_H264_INVALID, 0, 0, 0};
            if (!force_idr_frame) pps.ReferenceFrames[0] = g_vaapi_state.last_ref_pic;
            pps.coded_buf = g_vaapi_state.coded_buffer_id;
            pps.frame_num = g_vaapi_state.frame_count;
            pps.pic_init_qp = g_vaapi_state.initialized_qp;
            pps.pic_fields.bits.idr_pic_flag = force_idr_frame ? 1 : 0;
            pps.pic_fields.bits.reference_pic_flag = 1;
            pps.pic_fields.bits.entropy_coding_mode_flag = 1;
            pps.pic_fields.bits.deblocking_filter_control_present_flag = 1;
            pps.pic_fields.bits.transform_8x8_mode_flag = 1;
            VABufferID buf_id;
            if (funcs.vaCreateBuffer(g_vaapi_state.display, g_vaapi_state.context_id, VAEncPictureParameterBufferType, sizeof(pps), 1, &pps, &buf_id) != VA_STATUS_SUCCESS) throw std::runtime_error("vaCreateBuffer for PPS failed.");
            param_buffers.push_back(buf_id);
        }

        {
            VAEncSliceParameterBufferH264 slice = {};
            slice.slice_type = force_idr_frame ? 2 : 0;
            slice.num_macroblocks = ((width + 15) / 16) * ((height + 15) / 16);
            slice.pic_order_cnt_lsb = (g_vaapi_state.frame_count * 2);
            for (int i = 0; i < 32; ++i) slice.RefPicList0[i] = slice.RefPicList1[i] = {VA_INVALID_ID, VA_PICTURE_H264_INVALID, 0, 0, 0};
            if (!force_idr_frame) slice.RefPicList0[0] = g_vaapi_state.last_ref_pic;
            VABufferID buf_id;
            if (funcs.vaCreateBuffer(g_vaapi_state.display, g_vaapi_state.context_id, VAEncSliceParameterBufferType, sizeof(slice), 1, &slice, &buf_id) != VA_STATUS_SUCCESS) throw std::runtime_error("vaCreateBuffer for Slice failed.");
            param_buffers.push_back(buf_id);
        }
    } catch (const std::runtime_error& e) {
        for (VABufferID buf_id : param_buffers) funcs.vaDestroyBuffer(g_vaapi_state.display, buf_id);
        throw;
    }

    status = funcs.vaBeginPicture(g_vaapi_state.display, g_vaapi_state.context_id, current_surface);
    if (status != VA_STATUS_SUCCESS) throw std::runtime_error("vaBeginPicture failed: " + std::to_string(status));

    status = funcs.vaRenderPicture(g_vaapi_state.display, g_vaapi_state.context_id, param_buffers.data(), param_buffers.size());
    if (status != VA_STATUS_SUCCESS) throw std::runtime_error("vaRenderPicture failed: " + std::to_string(status));

    status = funcs.vaEndPicture(g_vaapi_state.display, g_vaapi_state.context_id);
    if (status != VA_STATUS_SUCCESS) {
        for(VABufferID buf_id : param_buffers) funcs.vaDestroyBuffer(g_vaapi_state.display, buf_id);
        throw std::runtime_error("vaEndPicture failed: " + std::to_string(status));
    }

    for (VABufferID buf_id : param_buffers) funcs.vaDestroyBuffer(g_vaapi_state.display, buf_id);

    status = funcs.vaSyncSurface(g_vaapi_state.display, current_surface);
    if (status != VA_STATUS_SUCCESS) throw std::runtime_error("vaSyncSurface failed: " + std::to_string(status));

    VACodedBufferSegment* coded_segment = nullptr;
    status = funcs.vaMapBuffer(g_vaapi_state.display, g_vaapi_state.coded_buffer_id, (void**)&coded_segment);
    if (status != VA_STATUS_SUCCESS) throw std::runtime_error("vaMapBuffer for coded data failed: " + std::to_string(status));

    if (coded_segment && coded_segment->size > 0 && coded_segment->buf) {
        const unsigned char TAG = 0x04;
        unsigned char type_hdr = (force_idr_frame) ? 0x01 : 0x00;
        int header_sz = 10;
        result.data = new unsigned char[coded_segment->size + header_sz];
        result.size = coded_segment->size + header_sz;

        result.data[0] = TAG;
        result.data[1] = type_hdr;
        uint16_t net_val = htons(static_cast<uint16_t>(result.frame_id % 65536));
        std::memcpy(result.data + 2, &net_val, 2);
        net_val = htons(static_cast<uint16_t>(result.stripe_y_start));
        std::memcpy(result.data + 4, &net_val, 2);
        net_val = htons(static_cast<uint16_t>(width));
        std::memcpy(result.data + 6, &net_val, 2);
        net_val = htons(static_cast<uint16_t>(height));
        std::memcpy(result.data + 8, &net_val, 2);
        std::memcpy(result.data + header_sz, coded_segment->buf, coded_segment->size);
    }

    funcs.vaUnmapBuffer(g_vaapi_state.display, g_vaapi_state.coded_buffer_id);

    g_vaapi_state.last_ref_pic = {current_surface, g_vaapi_state.frame_count, VA_PICTURE_H264_SHORT_TERM_REFERENCE, 0, 0};
    g_vaapi_state.frame_count++;

    return result;
}

/**
 * @brief Callback function type for processing encoded stripes.
 * @param result Pointer to the StripeEncodeResult containing the encoded data.
 * @param user_data User-defined data passed to the callback.
 */
typedef void (*StripeCallback)(StripeEncodeResult* result, void* user_data);

/**
 * @brief Encodes a horizontal stripe of an image from shared memory into JPEG format.
 * @param thread_id Identifier for the calling thread, used for managing encoder resources.
 * @param stripe_y_start The Y-coordinate of the top of the stripe within the full image.
 * @param stripe_height The height of the stripe to encode.
 * @param capture_width_actual The actual width of the stripe (and full image).
 * @param shm_data_base Pointer to the beginning of the full image data in shared memory.
 * @param shm_stride_bytes The stride (bytes per row) of the shared memory image.
 * @param shm_bytes_per_pixel The number of bytes per pixel in the shared memory image (e.g., 4 for BGRX).
 * @param jpeg_quality The JPEG quality setting (0-100).
 * @param frame_counter The identifier of the current frame.
 * @return A StripeEncodeResult containing the JPEG data, or an empty result on failure.
 *         The result data includes a custom 4-byte header: frame_id (uint16_t network byte order)
 *         and stripe_y_start (uint16_t network byte order).
 */
StripeEncodeResult encode_stripe_jpeg(
  int thread_id,
  int stripe_y_start,
  int stripe_height,
  int capture_width_actual,
  const unsigned char* shm_data_base,
  int shm_stride_bytes,
  int shm_bytes_per_pixel,
  int jpeg_quality,
  int frame_counter);

/**
 * @brief Encodes a horizontal stripe of YUV data into H.264 format.
 * @param thread_id Identifier for the calling thread, used for managing encoder resources.
 * @param stripe_y_start The Y-coordinate of the top of the stripe.
 * @param stripe_height The height of the stripe to encode (must be even).
 * @param capture_width_actual The width of the stripe (must be even).
 * @param y_plane_stripe_start Pointer to the start of the Y plane data for this stripe.
 * @param y_stride Stride of the Y plane.
 * @param u_plane_stripe_start Pointer to the start of the U plane data for this stripe.
 * @param u_stride Stride of the U plane.
 * @param v_plane_stripe_start Pointer to the start of the V plane data for this stripe.
 * @param v_stride Stride of the V plane.
 * @param is_i444_input True if the input YUV data is I444, false if I420.
 * @param frame_counter The identifier of the current frame.
 * @param current_crf_setting The H.264 CRF (Constant Rate Factor) to use for encoding.
 * @param colorspace_setting An integer indicating input YUV format (420 for I420, 444 for I444).
 * @param use_full_range True if full range color should be signaled in VUI, false for limited range.
 * @return A StripeEncodeResult containing the H.264 NAL units, or an empty result on failure.
 *         The result data includes a custom 10-byte header: type tag (0x04), frame type,
 *         frame_id (uint16_t), stripe_y_start (uint16_t), width (uint16_t), height (uint16_t),
 *         all multi-byte fields in network byte order.
 */
StripeEncodeResult encode_stripe_h264(
  int thread_id,
  int stripe_y_start,
  int stripe_height,
  int capture_width_actual,
  const uint8_t* y_plane_stripe_start, int y_stride,
  const uint8_t* u_plane_stripe_start, int u_stride,
  const uint8_t* v_plane_stripe_start, int v_stride,
  bool is_i444_input,
  int frame_counter,
  int current_crf_setting,
  int colorspace_setting,
  bool use_full_range);

/**
 * @brief Calculates a hash for a stripe of YUV data.
 * @param y_plane_stripe_start Pointer to the Y plane data for the stripe.
 * @param y_stride Stride of the Y plane.
 * @param u_plane_stripe_start Pointer to the U plane data for the stripe.
 * @param u_stride Stride of the U plane.
 * @param v_plane_stripe_start Pointer to the V plane data for the stripe.
 * @param v_stride Stride of the V plane.
 * @param width Width of the stripe.
 * @param height Height of the stripe.
 * @param is_i420 True if the YUV format is I420 (chroma planes are half width/height),
 *                false if I444 (chroma planes are full width/height).
 * @return A 64-bit hash value of the stripe data, or 0 on error.
 */
uint64_t calculate_yuv_stripe_hash(const uint8_t* y_plane_stripe_start, int y_stride,
                                   const uint8_t* u_plane_stripe_start, int u_stride,
                                   const uint8_t* v_plane_stripe_start, int v_stride,
                                   int width, int height, bool is_i420);

/**
 * @brief Calculates a hash for a stripe of BGR(X) data directly from shared memory.
 * Extracts BGR components for hashing.
 * @param shm_stripe_physical_start Pointer to the start of the stripe data in shared memory.
 * @param shm_stride_bytes Stride (bytes per row) of the shared memory image.
 * @param stripe_width Width of the stripe.
 * @param stripe_height Height of the stripe.
 * @param shm_bytes_per_pixel Bytes per pixel in the shared memory (e.g., 3 for BGR, 4 for BGRX).
 * @return A 64-bit hash value of the BGR data in the stripe, or 0 on error.
 */
uint64_t calculate_bgr_stripe_hash_from_shm(const unsigned char* shm_stripe_physical_start,
                                            int shm_stride_bytes,
                                            int stripe_width, int stripe_height,
                                            int shm_bytes_per_pixel);

/**
 * @brief Calculates a hash for a stripe of RGB data.
 * This function is kept for potential compatibility or specific use cases
 * where RGB data is already prepared in a contiguous buffer.
 * New logic for screen capture should generally use YUV or BGR specific hash functions
 * that operate closer to the source data format.
 * @param rgb_data A vector containing the RGB pixel data for the stripe.
 *                 Assumes tightly packed (R,G,B,R,G,B...).
 * @return A 64-bit hash value of the RGB data, or 0 if rgb_data is empty.
 */
uint64_t calculate_rgb_stripe_hash(const std::vector<unsigned char>& rgb_data);


/**
 * @brief Manages the screen capture process, including settings and threading.
 * This class encapsulates the logic for capturing screen content using XShm,
 * dividing it into stripes, encoding these stripes (JPEG or H.264) based on
 * damage detection and other heuristics, and invoking a callback with the encoded data.
 * It supports dynamic modification of capture settings.
 */
class ScreenCaptureModule {
public:
  int capture_width = 1024;
  int capture_height = 768;
  int capture_x = 0;
  int capture_y = 0;
  double target_fps = 60.0;
  int jpeg_quality = 85;
  int paint_over_jpeg_quality = 95;
  bool use_paint_over_quality = false;
  int paint_over_trigger_frames = 10;
  int damage_block_threshold = 15;
  int damage_block_duration = 30;
  int h264_crf = 25;
  bool h264_fullcolor = false;
  bool h264_fullframe = false;
  bool capture_cursor = false;
  OutputMode output_mode = OutputMode::H264;
  std::string watermark_path_internal;
  WatermarkLocation watermark_location_internal;

  std::atomic<bool> stop_requested;
  std::thread capture_thread;
  StripeCallback stripe_callback = nullptr;
  void* user_data = nullptr;
  int frame_counter = 0;
  int encoded_frame_count = 0;
  int total_stripes_encoded_this_interval = 0;
  mutable std::mutex settings_mutex;
  bool is_nvidia_system_detected = false;
  bool nvenc_operational = false;
  int vaapi_render_node_index = -1;
  bool vaapi_operational = false;

private:
    std::vector<uint8_t> full_frame_y_plane_;
    std::vector<uint8_t> full_frame_u_plane_;
    std::vector<uint8_t> full_frame_v_plane_;
    int full_frame_y_stride_;
    int full_frame_u_stride_;
    int full_frame_v_stride_;
    bool yuv_planes_are_i444_;
    std::vector<uint32_t> watermark_image_data_;
    int watermark_width_;
    int watermark_height_;
    bool watermark_loaded_;
    int watermark_current_x_;
    int watermark_current_y_;
    int watermark_dx_;
    int watermark_dy_;
    mutable std::mutex watermark_data_mutex_;

public:
  /**
   * @brief Default constructor for ScreenCaptureModule.
   * Initializes stop_requested to false and YUV plane strides to 0.
   */
  ScreenCaptureModule() : watermark_path_internal(""),
                          watermark_location_internal(WatermarkLocation::NONE),
                          stop_requested(false),
                          stripe_callback(nullptr),
                          full_frame_y_stride_(0), full_frame_u_stride_(0), full_frame_v_stride_(0),
                          yuv_planes_are_i444_(false),
                          watermark_width_(0),
                          watermark_height_(0),
                          watermark_loaded_(false),
                          watermark_current_x_(0),
                          watermark_current_y_(0),
                          watermark_dx_(2),
                          watermark_dy_(2) {}

  /**
   * @brief Destructor for ScreenCaptureModule.
   * Ensures that the capture process is stopped and resources are released.
   * Calls stop_capture().
   */
  ~ScreenCaptureModule() {
    stop_capture();
  }

  /**
   * @brief Starts the screen capture process in a new thread.
   * If a capture thread is already running, it is stopped first.
   * Resets encoder stores and frame counters. The actual settings used by
   * the capture loop are read from member variables which should be set
   * via modify_settings() before calling start_capture().
   */
  void start_capture() {
    if (capture_thread.joinable()) {
      stop_capture();
    }

    {
        std::lock_guard<std::mutex> lock(g_nvenc_mutex);
        if (LoadNvencApi()) {
            is_nvidia_system_detected = true;
        } else {
            is_nvidia_system_detected = false;
        }
    }

    g_h264_minimal_store.reset();

    nvenc_operational = false;
    g_nvenc_force_next_idr_global = true;

    stop_requested = false;
    frame_counter = 0;
    encoded_frame_count = 0;
    total_stripes_encoded_this_interval = 0;
    if (!watermark_path_internal.empty() && watermark_location_internal != WatermarkLocation::NONE) {
        load_watermark_image();
    }
    capture_thread = std::thread(&ScreenCaptureModule::capture_loop, this);
  }

  /**
   * @brief Stops the screen capture process.
   * Sets the stop_requested flag and waits for the capture thread to join.
   * This is a blocking call.
   */
  void stop_capture() {
    stop_requested = true;
    if (capture_thread.joinable()) {
      capture_thread.join();
    }
    if (g_nvenc_state.initialized) {
      reset_nvenc_encoder();
    }
    if (g_vaapi_state.initialized) {
      reset_vaapi_encoder();
    }
    unload_vaapi_library_if_loaded();
    unload_nvenc_library_if_loaded();
    UnloadCudaApi();
  }

  /**
   * @brief Modifies the capture and encoding settings.
   * This function is thread-safe. The new settings will be picked up by
   * the capture loop at the beginning of its next iteration.
   * If dimensions or H.264 color format change, XShm and encoders may be reinitialized.
   * @param new_settings A CaptureSettings struct containing the new settings.
   */
  void modify_settings(const CaptureSettings& new_settings) {
    std::lock_guard<std::mutex> lock(settings_mutex);
    capture_width = new_settings.capture_width;
    capture_height = new_settings.capture_height;
    capture_x = new_settings.capture_x;
    capture_y = new_settings.capture_y;
    target_fps = new_settings.target_fps;
    jpeg_quality = new_settings.jpeg_quality;
    paint_over_jpeg_quality = new_settings.paint_over_jpeg_quality;
    use_paint_over_quality = new_settings.use_paint_over_quality;
    paint_over_trigger_frames = new_settings.paint_over_trigger_frames;
    damage_block_threshold = new_settings.damage_block_threshold;
    damage_block_duration = new_settings.damage_block_duration;
    output_mode = new_settings.output_mode;
    h264_crf = new_settings.h264_crf;
    h264_fullcolor = new_settings.h264_fullcolor;
    h264_fullframe = new_settings.h264_fullframe;
    capture_cursor = new_settings.capture_cursor;
    vaapi_render_node_index = new_settings.vaapi_render_node_index;
    std::string new_wm_path_str = new_settings.watermark_path ? new_settings.watermark_path : "";
    bool path_actually_changed_in_settings = (watermark_path_internal != new_wm_path_str);
  
    watermark_path_internal = new_wm_path_str;
    watermark_location_internal = new_settings.watermark_location_enum;

    if (path_actually_changed_in_settings) {
        std::lock_guard<std::mutex> data_lock(watermark_data_mutex_);
        watermark_loaded_ = false;
    }
  }

  /**
   * @brief Retrieves the current capture and encoding settings.
   * This function is thread-safe.
   * @return A CaptureSettings struct containing the current settings as known
   *         to the module (may not yet be active in the capture loop if recently modified).
   */
  CaptureSettings get_current_settings() const {
    std::lock_guard<std::mutex> lock(settings_mutex);
    return CaptureSettings(
      capture_width, capture_height, capture_x, capture_y, target_fps,
      jpeg_quality, paint_over_jpeg_quality, use_paint_over_quality,
      paint_over_trigger_frames, damage_block_threshold,
      damage_block_duration, output_mode, h264_crf,
      h264_fullcolor, h264_fullframe, capture_cursor,
      watermark_path_internal.c_str(), watermark_location_internal,
      vaapi_render_node_index
      );
  }

private:

  /**
   * @brief Loads or reloads the watermark image from the configured path.
   * This function is thread-safe. It reads the watermark path and location
   * settings under a mutex. If a valid path is provided, it attempts to load
   * the image using stb_image, converts it to ARGB format, and stores it
   * internally for overlaying. If the path is empty or loading fails,
   * any existing watermark is cleared. For animated watermarks, it initializes
   * or resets the animation parameters.
   */
  void load_watermark_image() {
    std::string path_for_this_load;
    WatermarkLocation location_for_this_load;

    {
      std::lock_guard<std::mutex> settings_lock(settings_mutex);
      path_for_this_load = watermark_path_internal;
      location_for_this_load = watermark_location_internal;
    }

    if (path_for_this_load.empty() || location_for_this_load == WatermarkLocation::NONE) {
      std::lock_guard<std::mutex> data_lock(watermark_data_mutex_);
      if (watermark_loaded_) {
        std::cout << "Watermark cleared or not configured." << std::endl;
      }
      watermark_loaded_ = false;
      watermark_image_data_.clear();
      watermark_width_ = 0;
      watermark_height_ = 0;
      return;
    }
    int temp_w = 0, temp_h = 0, temp_channels = 0;
    unsigned char* stbi_img_data = stbi_load(path_for_this_load.c_str(), &temp_w, &temp_h, &temp_channels, 4);
    std::vector<uint32_t> temp_image_data_argb;
    bool temp_loaded_successfully = false;

    if (stbi_img_data) {
      if (temp_w > 0 && temp_h > 0) {
        temp_image_data_argb.resize(static_cast<size_t>(temp_w) * temp_h);
        for (int y_idx = 0; y_idx < temp_h; ++y_idx) {
          for (int x_idx = 0; x_idx < temp_w; ++x_idx) {
            size_t src_pixel_idx = (static_cast<size_t>(y_idx) * temp_w + x_idx) * 4;
            uint8_t r_val = stbi_img_data[src_pixel_idx + 0];
            uint8_t g_val = stbi_img_data[src_pixel_idx + 1];
            uint8_t b_val = stbi_img_data[src_pixel_idx + 2];
            uint8_t a_val = stbi_img_data[src_pixel_idx + 3];
            temp_image_data_argb[static_cast<size_t>(y_idx) * temp_w + x_idx] =
                (static_cast<uint32_t>(a_val) << 24) |
                (static_cast<uint32_t>(r_val) << 16) |
                (static_cast<uint32_t>(g_val) << 8)  |
                static_cast<uint32_t>(b_val);
          }
        }
        temp_loaded_successfully = true;
      } else {
         std::cerr << "Watermark image loaded with invalid dimensions: " << path_for_this_load
                   << " (" << temp_w << "x" << temp_h << ")" << std::endl;
      }
      stbi_image_free(stbi_img_data);
    } else {
      std::cerr << "Error loading watermark image: " << path_for_this_load
                << " - " << stbi_failure_reason() << std::endl;
    }
    std::lock_guard<std::mutex> data_lock(watermark_data_mutex_);
    if (temp_loaded_successfully) {
      watermark_image_data_ = std::move(temp_image_data_argb);
      watermark_width_ = temp_w;
      watermark_height_ = temp_h;
      watermark_loaded_ = true;
      std::cout << "Watermark loaded: " << path_for_this_load
                << " (" << watermark_width_ << "x" << watermark_height_ << ")" << std::endl;

      if (location_for_this_load == WatermarkLocation::AN) {
        watermark_current_x_ = 0;
        watermark_current_y_ = 0;
        watermark_dx_ = (watermark_dx_ != 0) ? std::abs(watermark_dx_) : 2;
        watermark_dy_ = (watermark_dy_ != 0) ? std::abs(watermark_dy_) : 2;
      }
    } else {
      watermark_loaded_ = false;
      watermark_image_data_.clear();
      watermark_width_ = 0;
      watermark_height_ = 0;
    }
  }

  /**
   * @brief Main loop for the screen capture thread.
   * This loop continuously captures frames from the screen using XShm, processes them,
   * and dispatches encoding tasks. It handles:
   * - X11 and XShm initialization and re-initialization on settings changes.
   * - Frame pacing to achieve the target FPS.
   * - Conversion of captured BGRX frames to YUV if H.264 encoding is active.
   * - Division of the frame into horizontal stripes for parallel processing.
   * - Damage detection per stripe using hash comparison to identify changed regions.
   * - Heuristics for paint-over (sending higher quality for static content) and
   *   damage blocks (sustained encoding for rapidly changing areas).
   * - Asynchronous encoding of stripes (JPEG or H.264) using a thread pool pattern.
   * - Invoking a user-provided callback with the encoded stripe data.
   * - Logging of performance metrics (FPS, encoded stripes/sec).
   * The loop runs until stop_requested is set to true.
   */
  void capture_loop() {
    static bool vaapi_444_warning_shown = false;
    auto start_time_loop = std::chrono::high_resolution_clock::now();
    int frame_count_loop = 0;

    int local_capture_width_actual;
    int local_capture_height_actual;
    int local_capture_x_offset;
    int local_capture_y_offset;
    double local_current_target_fps;
    int local_current_jpeg_quality;
    int local_current_paint_over_jpeg_quality;
    bool local_current_use_paint_over_quality;
    int local_current_paint_over_trigger_frames;
    int local_current_damage_block_threshold;
    int local_current_damage_block_duration;
    int local_current_h264_crf;
    bool local_current_h264_fullcolor;
    bool local_current_h264_fullframe;
    OutputMode local_current_output_mode;
    bool local_current_capture_cursor;
    int local_vaapi_render_node_index;
    int xfixes_event_base = 0;
    int xfixes_error_base = 0;
    std::string local_watermark_path_setting;
    WatermarkLocation local_watermark_location_setting;

    {
      std::lock_guard<std::mutex> lock(settings_mutex);
      local_capture_width_actual = capture_width;
      local_capture_height_actual = capture_height;
      local_capture_x_offset = capture_x;
      local_capture_y_offset = capture_y;
      local_current_target_fps = target_fps;
      local_current_jpeg_quality = jpeg_quality;
      local_current_paint_over_jpeg_quality = paint_over_jpeg_quality;
      local_current_use_paint_over_quality = use_paint_over_quality;
      local_current_paint_over_trigger_frames = paint_over_trigger_frames;
      local_current_damage_block_threshold = damage_block_threshold;
      local_current_damage_block_duration = damage_block_duration;
      local_current_output_mode = output_mode;
      local_current_h264_crf = h264_crf;
      local_current_h264_fullcolor = h264_fullcolor;
      local_current_h264_fullframe = h264_fullframe;
      local_current_capture_cursor = capture_cursor;
      local_vaapi_render_node_index = vaapi_render_node_index;
      local_watermark_path_setting = watermark_path_internal;
      local_watermark_location_setting = watermark_location_internal;
    }
    if (local_current_output_mode == OutputMode::H264) {
      if (local_capture_width_actual % 2 != 0 && local_capture_width_actual > 0) {
        local_capture_width_actual--;
      }
      if (local_capture_height_actual % 2 != 0 && local_capture_height_actual > 0) {
        local_capture_height_actual--;
      }
    }
    if (local_capture_width_actual <=0 || local_capture_height_actual <=0) {
        std::cerr << "Error: Invalid capture dimensions after initial adjustment." << std::endl;
        return;
    }

    this->yuv_planes_are_i444_ = local_current_h264_fullcolor;
    if (local_current_output_mode == OutputMode::H264) {
        size_t y_plane_size = static_cast<size_t>(local_capture_width_actual) *
                              local_capture_height_actual;
        full_frame_y_plane_.resize(y_plane_size);
        full_frame_y_stride_ = local_capture_width_actual;

        if (this->yuv_planes_are_i444_) {
            full_frame_u_plane_.resize(y_plane_size);
            full_frame_v_plane_.resize(y_plane_size);
            full_frame_u_stride_ = local_capture_width_actual;
            full_frame_v_stride_ = local_capture_width_actual;
        } else {
            size_t chroma_plane_size = (static_cast<size_t>(local_capture_width_actual) / 2) *
                                       (static_cast<size_t>(local_capture_height_actual) / 2);
            full_frame_u_plane_.resize(chroma_plane_size);
            full_frame_v_plane_.resize(chroma_plane_size);
            full_frame_u_stride_ = local_capture_width_actual / 2;
            full_frame_v_stride_ = local_capture_width_actual / 2;
        }
    }

    if (!local_watermark_path_setting.empty() && local_watermark_location_setting != WatermarkLocation::NONE) {
        load_watermark_image();
    }

    std::chrono::duration < double > target_frame_duration_seconds =
      std::chrono::duration < double > (1.0 / local_current_target_fps);

    auto next_frame_time =
      std::chrono::high_resolution_clock::now() + target_frame_duration_seconds;

    char* display_env = std::getenv("DISPLAY");
    const char* display_name = display_env ? display_env : ":0";
    Display* display = XOpenDisplay(display_name);
    if (!display) {
      std::cerr << "Error: Failed to open X display " << display_name << std::endl;
      return;
    }
    Window root_window = DefaultRootWindow(display);
    int screen = DefaultScreen(display);

    if (!XShmQueryExtension(display)) {
      std::cerr << "Error: X Shared Memory Extension not available!" << std::endl;
      XCloseDisplay(display);
      return;
    }
    std::cout << "X Shared Memory Extension available." << std::endl;

    if (local_current_capture_cursor) {
      if (!XFixesQueryExtension(display, &xfixes_event_base, &xfixes_error_base)) {
        std::cerr << "Error: XFixes extension not available!" << std::endl;
        XCloseDisplay(display);
        return;
      }

      std::cout << "XFixes Extension available." << std::endl;
    }

    XShmSegmentInfo shminfo;
    memset(&shminfo, 0, sizeof(shminfo));
    XImage* shm_image = nullptr;

    shm_image = XShmCreateImage(
      display, DefaultVisual(display, screen), DefaultDepth(display, screen),
      ZPixmap, nullptr, &shminfo, local_capture_width_actual,
      local_capture_height_actual);
    if (!shm_image) {
      std::cerr << "Error: XShmCreateImage failed for "
                << local_capture_width_actual << "x"
                << local_capture_height_actual << std::endl;
      XCloseDisplay(display);
      return;
    }

    shminfo.shmid = shmget(IPC_PRIVATE,
                           static_cast<size_t>(shm_image->bytes_per_line) * shm_image->height,
                           IPC_CREAT | 0600);
    if (shminfo.shmid < 0) {
      perror("shmget");
      XDestroyImage(shm_image);
      XCloseDisplay(display);
      return;
    }

    shminfo.shmaddr = (char*)shmat(shminfo.shmid, nullptr, 0);
    if (shminfo.shmaddr == (char*)-1) {
      perror("shmat");
      shmctl(shminfo.shmid, IPC_RMID, 0);
      XDestroyImage(shm_image);
      XCloseDisplay(display);
      return;
    }
    shminfo.readOnly = False;
    shm_image->data = shminfo.shmaddr;

    if (!XShmAttach(display, &shminfo)) {
      std::cerr << "Error: XShmAttach failed" << std::endl;
      shmdt(shminfo.shmaddr);
      shmctl(shminfo.shmid, IPC_RMID, 0);
      XDestroyImage(shm_image);
      XCloseDisplay(display);
      return;
    }
    std::cout << "XShm setup complete for " << local_capture_width_actual
              << "x" << local_capture_height_actual << "." << std::endl;

    this->vaapi_operational = false;
    this->nvenc_operational = false;

    if (local_vaapi_render_node_index >= 0 &&
        local_current_output_mode == OutputMode::H264 && local_current_h264_fullframe) {
        if (initialize_vaapi_encoder(local_vaapi_render_node_index, local_capture_width_actual,
                                     local_capture_height_actual, local_current_h264_crf)) {
            this->vaapi_operational = true;
            g_vaapi_force_next_idr_global = true;
            std::cout << "VAAPI Encoder Initialized successfully." << std::endl;
        } else {
            std::cerr << "VAAPI Encoder initialization failed. Falling back to CPU." << std::endl;
        }
    } else {
      if (this->is_nvidia_system_detected &&
          local_current_output_mode == OutputMode::H264 && local_current_h264_fullframe) {
        if (initialize_nvenc_encoder(local_capture_width_actual,
                                     local_capture_height_actual,
                                     local_current_h264_crf,
                                     local_current_target_fps,
                                     local_current_h264_fullcolor)) {
          this->nvenc_operational = true;
          g_nvenc_force_next_idr_global = true;
          std::cout << "NVENC Encoder Initialized successfully." << std::endl;
        } else {
          std::cerr << "NVENC Encoder initialization failed. Falling back to x264." << std::endl;
        }
      } else {
          if (!this->nvenc_operational && g_nvenc_state.initialized) {
            reset_nvenc_encoder();
          }
      }
    }
    int num_cores = std::max(1, (int)std::thread::hardware_concurrency());
    std::cout << "CPU cores available: " << num_cores << std::endl;
    int num_stripes_config = num_cores;

    std::vector<uint64_t> previous_hashes(num_stripes_config, 0);
    std::vector<int> no_motion_frame_counts(num_stripes_config, 0);
    std::vector<bool> paint_over_sent(num_stripes_config, false);
    std::vector<int> current_jpeg_qualities(num_stripes_config);
    std::vector<int> consecutive_stripe_changes(num_stripes_config, 0);
    std::vector<bool> stripe_is_in_damage_block(num_stripes_config, false);
    std::vector<int> stripe_damage_block_frames_remaining(num_stripes_config, 0);
    std::vector<uint64_t> stripe_hash_at_damage_block_start(num_stripes_config, 0);

    for (int i = 0; i < num_stripes_config; ++i) {
      current_jpeg_qualities[i] =
        local_current_use_paint_over_quality
          ? local_current_paint_over_jpeg_quality
          : local_current_jpeg_quality;
    }

    auto last_output_time = std::chrono::high_resolution_clock::now();

    while (!stop_requested) {
      auto current_loop_iter_start_time = std::chrono::high_resolution_clock::now();

      if (current_loop_iter_start_time < next_frame_time) {
        auto time_to_sleep = next_frame_time - current_loop_iter_start_time;
        if (time_to_sleep > std::chrono::milliseconds(0)) {
          std::this_thread::sleep_for(time_to_sleep);
        }
      }
      auto intended_current_frame_time = next_frame_time;
      next_frame_time += target_frame_duration_seconds;

      int old_w = local_capture_width_actual;
      int old_h = local_capture_height_actual;
      bool yuv_config_changed = false;
      std::string previous_watermark_path_in_loop = local_watermark_path_setting;
      WatermarkLocation previous_watermark_location_in_loop = local_watermark_location_setting;
      {
        std::lock_guard<std::mutex> lock(settings_mutex);
        local_capture_width_actual = capture_width;
        local_capture_height_actual = capture_height;
        local_capture_x_offset = capture_x;
        local_capture_y_offset = capture_y;

        if (local_current_target_fps != target_fps) {
          local_current_target_fps = target_fps;
          target_frame_duration_seconds = std::chrono::duration < double > (1.0 / local_current_target_fps);
          next_frame_time = intended_current_frame_time + target_frame_duration_seconds;
        }
        local_current_jpeg_quality = jpeg_quality;
        local_current_paint_over_jpeg_quality = paint_over_jpeg_quality;
        local_current_use_paint_over_quality = use_paint_over_quality;
        local_current_paint_over_trigger_frames = paint_over_trigger_frames;
        local_current_damage_block_threshold = damage_block_threshold;
        local_current_damage_block_duration = damage_block_duration;

        if (local_current_output_mode != output_mode ||
            local_current_h264_fullcolor != h264_fullcolor) {
            yuv_config_changed = true;
        }
        local_current_output_mode = output_mode;
        local_current_h264_crf = h264_crf;
        local_current_h264_fullcolor = h264_fullcolor;
        local_current_h264_fullframe = h264_fullframe;
        local_current_capture_cursor = capture_cursor;
        local_vaapi_render_node_index = vaapi_render_node_index;
        local_watermark_path_setting = watermark_path_internal;
        local_watermark_location_setting = watermark_location_internal;
      }

      bool current_watermark_is_actually_loaded_in_loop;
      {
        std::lock_guard<std::mutex> data_lock(watermark_data_mutex_);
        current_watermark_is_actually_loaded_in_loop = watermark_loaded_;
      }

      bool path_setting_changed_from_last_loop_iter = (local_watermark_path_setting != previous_watermark_path_in_loop);
      bool location_setting_changed_from_last_loop_iter = (local_watermark_location_setting != previous_watermark_location_in_loop);
      bool needs_load_due_to_state = (local_watermark_location_setting != WatermarkLocation::NONE &&
                                      !local_watermark_path_setting.empty() &&
                                      !current_watermark_is_actually_loaded_in_loop);
      bool needs_clear_due_to_state = ( (local_watermark_location_setting == WatermarkLocation::NONE || local_watermark_path_setting.empty()) &&
                                       current_watermark_is_actually_loaded_in_loop);

      if (path_setting_changed_from_last_loop_iter ||
          location_setting_changed_from_last_loop_iter ||
          needs_load_due_to_state ||
          needs_clear_due_to_state ||
          (local_watermark_location_setting == WatermarkLocation::AN && previous_watermark_location_in_loop != WatermarkLocation::AN)
          ) {
          load_watermark_image();
          previous_watermark_path_in_loop = local_watermark_path_setting;
          previous_watermark_location_in_loop = local_watermark_location_setting;
      }

      if (local_current_output_mode == OutputMode::H264) {
        if (local_capture_width_actual % 2 != 0 && local_capture_width_actual > 0) {
          local_capture_width_actual--;
        }
        if (local_capture_height_actual % 2 != 0 && local_capture_height_actual > 0) {
          local_capture_height_actual--;
        }
      }
      if (local_capture_width_actual <=0 || local_capture_height_actual <=0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          continue;
      }

      if (old_w != local_capture_width_actual || old_h != local_capture_height_actual ||
          yuv_config_changed) {
        std::cout << "Capture parameters changed. Re-initializing XShm and YUV planes."
                  << std::endl;

        if (shm_image) {
            if (shminfo.shmaddr && shminfo.shmaddr != (char*)-1) {
                XShmDetach(display, &shminfo);
                shmdt(shminfo.shmaddr);
                shminfo.shmaddr = (char*)-1;
            }
            if (shminfo.shmid != -1 && shminfo.shmid != 0) {
                shmctl(shminfo.shmid, IPC_RMID, 0);
                shminfo.shmid = -1;
            }
            XDestroyImage(shm_image);
            shm_image = nullptr;
            memset(&shminfo, 0, sizeof(shminfo));
        }

        shm_image = XShmCreateImage(
          display, DefaultVisual(display, screen), DefaultDepth(display, screen),
          ZPixmap, nullptr, &shminfo, local_capture_width_actual,
          local_capture_height_actual);
        if (!shm_image) {
          std::cerr << "Error: XShmCreateImage failed during re-init." << std::endl;
          if(display) { XCloseDisplay(display); } display = nullptr; return;
        }
        shminfo.shmid = shmget(
          IPC_PRIVATE, static_cast<size_t>(shm_image->bytes_per_line) * shm_image->height,
          IPC_CREAT | 0600);
        if (shminfo.shmid < 0) {
          perror("shmget re-init"); if(shm_image) { XDestroyImage(shm_image); } shm_image = nullptr;
          if(display) { XCloseDisplay(display); } display = nullptr; return;
        }
        shminfo.shmaddr = (char*)shmat(shminfo.shmid, nullptr, 0);
        if (shminfo.shmaddr == (char*)-1) {
          perror("shmat re-init"); 
          if(shminfo.shmid != -1) { shmctl(shminfo.shmid, IPC_RMID, 0); } shminfo.shmid = -1;
          if(shm_image) { XDestroyImage(shm_image); } shm_image = nullptr;
          if(display) { XCloseDisplay(display); } display = nullptr; return;
        }
        shminfo.readOnly = False;
        shm_image->data = shminfo.shmaddr;
        if (!XShmAttach(display, &shminfo)) {
          if(shminfo.shmaddr != (char*)-1) { shmdt(shminfo.shmaddr); } shminfo.shmaddr = (char*)-1;
          if(shminfo.shmid != -1) { shmctl(shminfo.shmid, IPC_RMID, 0); } shminfo.shmid = -1;
          if(shm_image) { XDestroyImage(shm_image); } shm_image = nullptr;
          if(display) { XCloseDisplay(display); } display = nullptr; return;
        }

        this->yuv_planes_are_i444_ = local_current_h264_fullcolor;
        if (local_current_output_mode == OutputMode::H264) {
            size_t y_plane_size = static_cast<size_t>(local_capture_width_actual) *
                                  local_capture_height_actual;
            full_frame_y_plane_.assign(y_plane_size, 0);
            full_frame_y_stride_ = local_capture_width_actual;

            if (this->yuv_planes_are_i444_) {
                full_frame_u_plane_.assign(y_plane_size, 0);
                full_frame_v_plane_.assign(y_plane_size, 0);
                full_frame_u_stride_ = local_capture_width_actual;
                full_frame_v_stride_ = local_capture_width_actual;
            } else {
                size_t chroma_plane_size =
                    (static_cast<size_t>(local_capture_width_actual) / 2) *
                    (static_cast<size_t>(local_capture_height_actual) / 2);
                full_frame_u_plane_.assign(chroma_plane_size, 0);
                full_frame_v_plane_.assign(chroma_plane_size, 0);
                full_frame_u_stride_ = local_capture_width_actual / 2;
                full_frame_v_stride_ = local_capture_width_actual / 2;
            }
        } else {
            full_frame_y_plane_.clear();
            full_frame_u_plane_.clear();
            full_frame_v_plane_.clear();
        }

        std::cout << "XShm and YUV planes re-initialization complete." << std::endl;
        g_h264_minimal_store.reset();
      }

      if (XShmGetImage(display, root_window, shm_image, local_capture_x_offset, local_capture_y_offset, AllPlanes)) {
        unsigned char* shm_data_ptr = (unsigned char*)shm_image->data;
        int shm_stride_bytes = shm_image->bytes_per_line;
        int shm_bytes_per_pixel = shm_image->bits_per_pixel / 8;
        if (local_current_capture_cursor) {
          XFixesCursorImage *cursor_image = XFixesGetCursorImage(display);
          if (cursor_image) {
            std::vector<uint32_t> converted_cursor_pixels;
            if (cursor_image->width > 0 && cursor_image->height > 0) {
                converted_cursor_pixels.resize(static_cast<size_t>(cursor_image->width) * cursor_image->height);
                for (int r = 0; r < cursor_image->height; ++r) {
                    for (int c = 0; c < cursor_image->width; ++c) {
                        unsigned long raw_pixel = cursor_image->pixels[static_cast<size_t>(r) * cursor_image->width + c];
                        converted_cursor_pixels[static_cast<size_t>(r) * cursor_image->width + c] = static_cast<uint32_t>(raw_pixel);
                    }
                }
            }

            if (!converted_cursor_pixels.empty()) {
                overlay_image(cursor_image->height, cursor_image->width, 
                              converted_cursor_pixels.data(),
                              cursor_image->x - local_capture_x_offset,
                              cursor_image->y - local_capture_y_offset,
                              local_capture_height_actual, local_capture_width_actual, 
                              shm_data_ptr, shm_stride_bytes, shm_bytes_per_pixel);
            }
            XFree(cursor_image);
          }
        }

        bool should_overlay_watermark_this_frame = false;
        int overlay_wm_x = 0;
        int overlay_wm_y = 0;
        int temp_wm_w = 0;
        int temp_wm_h = 0;
        std::vector<uint32_t> local_watermark_data_copy;

        {
          std::lock_guard<std::mutex> data_lock(watermark_data_mutex_);
          if (watermark_loaded_ && local_watermark_location_setting != WatermarkLocation::NONE &&
              !watermark_image_data_.empty() && watermark_width_ > 0 && watermark_height_ > 0) {
            
            should_overlay_watermark_this_frame = true;
            temp_wm_w = watermark_width_;
            temp_wm_h = watermark_height_;
            local_watermark_data_copy = watermark_image_data_;

            if (local_watermark_location_setting == WatermarkLocation::AN) {
              watermark_current_x_ += watermark_dx_;
              watermark_current_y_ += watermark_dy_;
  
              if (watermark_current_x_ + watermark_width_ > local_capture_width_actual) {
                watermark_current_x_ = local_capture_width_actual - watermark_width_;
                if (watermark_current_x_ < 0) {
                  watermark_current_x_ = 0;
                }
                watermark_dx_ *= -1;
              } else if (watermark_current_x_ < 0) {
                watermark_current_x_ = 0;
                watermark_dx_ *= -1;
              }
  
              if (watermark_current_y_ + watermark_height_ > local_capture_height_actual) {
                watermark_current_y_ = local_capture_height_actual - watermark_height_;
                if (watermark_current_y_ < 0) {
                  watermark_current_y_ = 0;
                }
                watermark_dy_ *= -1;
              } else if (watermark_current_y_ < 0) {
                watermark_current_y_ = 0;
                watermark_dy_ *= -1;
              }
              overlay_wm_x = watermark_current_x_;
              overlay_wm_y = watermark_current_y_;
            }
          }
        }

        if (should_overlay_watermark_this_frame) {
          if (local_watermark_location_setting != WatermarkLocation::AN) { 
            switch (local_watermark_location_setting) {
              case WatermarkLocation::TL:
                overlay_wm_x = 0;
                overlay_wm_y = 0;
                break;
              case WatermarkLocation::TR:
                overlay_wm_x = local_capture_width_actual - temp_wm_w;
                overlay_wm_y = 0;
                break;
              case WatermarkLocation::BL:
                overlay_wm_x = 0;
                overlay_wm_y = local_capture_height_actual - temp_wm_h;
                break;
              case WatermarkLocation::BR:
                overlay_wm_x = local_capture_width_actual - temp_wm_w;
                overlay_wm_y = local_capture_height_actual - temp_wm_h;
                break;
              case WatermarkLocation::MI:
                overlay_wm_x = (local_capture_width_actual - temp_wm_w) / 2;
                overlay_wm_y = (local_capture_height_actual - temp_wm_h) / 2;
                break;
              default:
                should_overlay_watermark_this_frame = false;
                break; 
            }
          }

          if (should_overlay_watermark_this_frame) { 
            if (overlay_wm_x < 0) {
              overlay_wm_x = 0;
            }
            if (overlay_wm_y < 0) {
              overlay_wm_y = 0;
            }
            
            overlay_image(temp_wm_h, temp_wm_w,
                          local_watermark_data_copy.data(), 
                          overlay_wm_x, overlay_wm_y,
                          local_capture_height_actual, local_capture_width_actual,
                          shm_data_ptr, shm_stride_bytes, shm_bytes_per_pixel);
          }
        }

        if (local_current_output_mode == OutputMode::H264) {
            bool force_420_conversion = this->vaapi_operational;
            if (force_420_conversion && !vaapi_444_warning_shown) {
              std::cerr << "VAAPI_WARNING: 4:4:4 colorspace is not supported in VAAPI mode. "
                           "Forcing 4:2:0 conversion for encoder."
                        << std::endl;
              vaapi_444_warning_shown = true;
            }
            if (this->yuv_planes_are_i444_ && !force_420_conversion) {
                libyuv::ARGBToI444(shm_data_ptr, shm_stride_bytes,
                                   full_frame_y_plane_.data(), full_frame_y_stride_,
                                   full_frame_u_plane_.data(), full_frame_u_stride_,
                                   full_frame_v_plane_.data(), full_frame_v_stride_,
                                   local_capture_width_actual, local_capture_height_actual);
            } else {
                libyuv::ARGBToI420(shm_data_ptr, shm_stride_bytes,
                                   full_frame_y_plane_.data(), full_frame_y_stride_,
                                   full_frame_u_plane_.data(), full_frame_u_stride_,
                                   full_frame_v_plane_.data(), full_frame_v_stride_,
                                   local_capture_width_actual, local_capture_height_actual);
            }
        }

        std::vector<std::future<StripeEncodeResult>> futures;
        std::vector<std::thread> threads;

        int N_processing_stripes;
        if (local_capture_height_actual <= 0) {
          N_processing_stripes = 0;
        } else {
          if (local_current_output_mode == OutputMode::H264) {
            if (local_current_h264_fullframe) {
              N_processing_stripes = 1;
            } else {
              const int MIN_H264_STRIPE_HEIGHT_PX = 64;
              if (local_capture_height_actual < MIN_H264_STRIPE_HEIGHT_PX) {
                N_processing_stripes = 1;
              } else {
                int max_stripes_by_min_height =
                  local_capture_height_actual / MIN_H264_STRIPE_HEIGHT_PX;
                N_processing_stripes =
                  std::min(num_stripes_config, max_stripes_by_min_height);
                if (N_processing_stripes == 0) N_processing_stripes = 1;
              }
            }
          } else {
            N_processing_stripes =
              std::min(num_stripes_config, local_capture_height_actual);
            if (N_processing_stripes == 0 && local_capture_height_actual > 0) {
              N_processing_stripes = 1;
            }
          }
        }
        if (N_processing_stripes == 0 && local_capture_height_actual > 0) {
           N_processing_stripes = 1;
        }

        if (static_cast<int>(previous_hashes.size()) != N_processing_stripes) {
            previous_hashes.assign(N_processing_stripes, 0);
            no_motion_frame_counts.assign(N_processing_stripes, 0);
            paint_over_sent.assign(N_processing_stripes, false);
            current_jpeg_qualities.resize(N_processing_stripes);
            consecutive_stripe_changes.assign(N_processing_stripes, 0);
            stripe_is_in_damage_block.assign(N_processing_stripes, false);
            stripe_damage_block_frames_remaining.assign(N_processing_stripes, 0);
            stripe_hash_at_damage_block_start.assign(N_processing_stripes, 0);

            for(int k=0; k < N_processing_stripes; ++k) {
                 current_jpeg_qualities[k] = local_current_use_paint_over_quality ?
                                             local_current_paint_over_jpeg_quality :
                                             local_current_jpeg_quality;
            }
        }

        int h264_base_even_height = 0;
        int h264_num_stripes_with_extra_pair = 0;
        int current_y_start_for_stripe = 0;

        if (local_current_output_mode == OutputMode::H264 && !local_current_h264_fullframe &&
            N_processing_stripes > 0 && local_capture_height_actual > 0) {
          int H = local_capture_height_actual;
          int N = N_processing_stripes;
          int base_h = H / N;
          h264_base_even_height = (base_h > 0) ? (base_h - (base_h % 2)) : 0;
          if (h264_base_even_height == 0 && H >= 2) {
            h264_base_even_height = 2;
          } else if (h264_base_even_height == 0 && H > 0 && N == 1) {
             h264_base_even_height = H - (H % 2);
             if (h264_base_even_height == 0 && H >= 2) h264_base_even_height = 2;
          } else if (h264_base_even_height == 0 && H > 0) {
             N_processing_stripes = 0;
          }

          if (h264_base_even_height > 0) {
            int H_base_covered = h264_base_even_height * N;
            int H_remaining = H - H_base_covered;
            if (H_remaining < 0) H_remaining = 0;
            h264_num_stripes_with_extra_pair = H_remaining / 2;
            h264_num_stripes_with_extra_pair =
              std::min(h264_num_stripes_with_extra_pair, N);
          } else if (H > 0 && N_processing_stripes > 0) {
             N_processing_stripes = 0;
          }
        }
        bool any_stripe_encoded_this_frame = false;

        int derived_h264_colorspace_setting;
        bool derived_h264_use_full_range;
        if (local_current_h264_fullcolor) {
          derived_h264_colorspace_setting = 444;
          derived_h264_use_full_range = true;
        } else {
          derived_h264_colorspace_setting = 420;
          derived_h264_use_full_range = false;
        }

        for (int i = 0; i < N_processing_stripes; ++i) {
          int start_y = 0;
          int current_stripe_height = 0;

          if (local_current_output_mode == OutputMode::H264) {
            if (local_current_h264_fullframe) {
                start_y = 0;
                current_stripe_height = local_capture_height_actual;
            } else {
                start_y = current_y_start_for_stripe;
                if (h264_base_even_height > 0) {
                    current_stripe_height = h264_base_even_height;
                    if (i < h264_num_stripes_with_extra_pair) {
                        current_stripe_height += 2;
                    }
                } else if (N_processing_stripes == 1) {
                    current_stripe_height = local_capture_height_actual -
                                            (local_capture_height_actual % 2);
                    if (current_stripe_height == 0 && local_capture_height_actual >=2)
                        current_stripe_height = 2;
                } else {
                    current_stripe_height = 0;
                }
            }
          } else {
            if (N_processing_stripes > 0) {
                int base_stripe_height_jpeg = local_capture_height_actual / N_processing_stripes;
                int remainder_height_jpeg = local_capture_height_actual % N_processing_stripes;
                start_y = i * base_stripe_height_jpeg + std::min(i, remainder_height_jpeg);
                current_stripe_height = base_stripe_height_jpeg +
                                        (i < remainder_height_jpeg ? 1 : 0);
            } else {
                current_stripe_height = 0;
            }
          }

          if (current_stripe_height <= 0) {
            continue;
          }

          if (start_y + current_stripe_height > local_capture_height_actual) {
             current_stripe_height = local_capture_height_actual - start_y;
             if (current_stripe_height <= 0) continue;
             if (local_current_output_mode == OutputMode::H264 && !local_current_h264_fullframe &&
                 current_stripe_height % 2 != 0 && current_stripe_height > 0) {
                 current_stripe_height--;
             }
             if (current_stripe_height <= 0) continue;
          }

          if (local_current_output_mode == OutputMode::H264 && !local_current_h264_fullframe) {
            current_y_start_for_stripe += current_stripe_height;
          }

          uint64_t current_hash = 0;
          bool hash_calculated_this_iteration = false;
          bool send_this_stripe = false;
          bool is_h264_idr_paintover_this_stripe = false;

          if (stripe_is_in_damage_block[i]) {
              send_this_stripe = true;
              stripe_damage_block_frames_remaining[i]--;

              if (stripe_damage_block_frames_remaining[i] == 0) {
                  if (local_current_output_mode == OutputMode::H264) {
                      const uint8_t* y_plane_stripe_ptr = full_frame_y_plane_.data() +
                          static_cast<size_t>(start_y) * full_frame_y_stride_;
                      const uint8_t* u_plane_stripe_ptr = full_frame_u_plane_.data() +
                          (static_cast<size_t>(this->yuv_planes_are_i444_ ?
                           start_y : (start_y / 2)) * full_frame_u_stride_);
                      const uint8_t* v_plane_stripe_ptr = full_frame_v_plane_.data() +
                          (static_cast<size_t>(this->yuv_planes_are_i444_ ?
                           start_y : (start_y / 2)) * full_frame_v_stride_);
                      current_hash = calculate_yuv_stripe_hash(
                          y_plane_stripe_ptr, full_frame_y_stride_,
                          u_plane_stripe_ptr, full_frame_u_stride_,
                          v_plane_stripe_ptr, full_frame_v_stride_,
                          local_capture_width_actual, current_stripe_height,
                          !this->yuv_planes_are_i444_);
                  } else {
                      const unsigned char* shm_stripe_start_ptr = shm_data_ptr +
                          static_cast<size_t>(start_y) * shm_stride_bytes;
                      current_hash = calculate_bgr_stripe_hash_from_shm(
                          shm_stripe_start_ptr, shm_stride_bytes,
                          local_capture_width_actual, current_stripe_height,
                          shm_bytes_per_pixel);
                  }
                  hash_calculated_this_iteration = true;

                  if (current_hash != stripe_hash_at_damage_block_start[i]) {
                      stripe_damage_block_frames_remaining[i] =
                          local_current_damage_block_duration;
                      stripe_hash_at_damage_block_start[i] = current_hash;
                  } else {
                      stripe_is_in_damage_block[i] = false;
                      consecutive_stripe_changes[i] = 0;

                      if (current_hash == previous_hashes[i]) {
                          send_this_stripe = false;
                          no_motion_frame_counts[i]++;
                          if (no_motion_frame_counts[i] >=
                                  local_current_paint_over_trigger_frames &&
                              !paint_over_sent[i]) {
                              if (local_current_output_mode == OutputMode::JPEG &&
                                  local_current_use_paint_over_quality) {
                                  send_this_stripe = true;
                              } else if (local_current_output_mode == OutputMode::H264) {
                                  send_this_stripe = true;
                                  is_h264_idr_paintover_this_stripe = true;
                              }
                              if (send_this_stripe) paint_over_sent[i] = true;
                          }
                      } else {
                          send_this_stripe = true;
                          no_motion_frame_counts[i] = 0;
                          paint_over_sent[i] = false;
                      }
                      if (local_current_output_mode == OutputMode::JPEG) {
                          current_jpeg_qualities[i] = local_current_use_paint_over_quality ?
                                                      local_current_paint_over_jpeg_quality :
                                                      local_current_jpeg_quality;
                      }
                  }
              }
          } else {
              if (local_current_output_mode == OutputMode::H264) {
                  const uint8_t* y_plane_stripe_ptr = full_frame_y_plane_.data() +
                      static_cast<size_t>(start_y) * full_frame_y_stride_;
                  const uint8_t* u_plane_stripe_ptr = full_frame_u_plane_.data() +
                      (static_cast<size_t>(this->yuv_planes_are_i444_ ?
                       start_y : (start_y / 2)) * full_frame_u_stride_);
                  const uint8_t* v_plane_stripe_ptr = full_frame_v_plane_.data() +
                      (static_cast<size_t>(this->yuv_planes_are_i444_ ?
                       start_y : (start_y / 2)) * full_frame_v_stride_);
                  current_hash = calculate_yuv_stripe_hash(
                      y_plane_stripe_ptr, full_frame_y_stride_,
                      u_plane_stripe_ptr, full_frame_u_stride_,
                      v_plane_stripe_ptr, full_frame_v_stride_,
                      local_capture_width_actual, current_stripe_height,
                      !this->yuv_planes_are_i444_);
              } else {
                  const unsigned char* shm_stripe_start_ptr = shm_data_ptr +
                      static_cast<size_t>(start_y) * shm_stride_bytes;
                  current_hash = calculate_bgr_stripe_hash_from_shm(
                      shm_stripe_start_ptr, shm_stride_bytes,
                      local_capture_width_actual, current_stripe_height,
                      shm_bytes_per_pixel);
              }
              hash_calculated_this_iteration = true;

              if (current_hash != previous_hashes[i]) {
                  send_this_stripe = true;
                  no_motion_frame_counts[i] = 0;
                  paint_over_sent[i] = false;
                  consecutive_stripe_changes[i]++;

                  if (local_current_damage_block_threshold > 0 &&
                      consecutive_stripe_changes[i] >=
                          local_current_damage_block_threshold) {
                      //stripe_is_in_damage_block[i] = true;
                      //stripe_damage_block_frames_remaining[i] =
                      //    local_current_damage_block_duration;
                      //stripe_hash_at_damage_block_start[i] = current_hash;
                  }
                  if (local_current_output_mode == OutputMode::JPEG) {
                      current_jpeg_qualities[i] =
                          std::max(current_jpeg_qualities[i] - 1,
                                   local_current_jpeg_quality);
                  }
              } else {
                  send_this_stripe = false;
                  consecutive_stripe_changes[i] = 0;
                  no_motion_frame_counts[i]++;

                  if (no_motion_frame_counts[i] >=
                          local_current_paint_over_trigger_frames &&
                      !paint_over_sent[i]) {
                      if (local_current_output_mode == OutputMode::JPEG &&
                          local_current_use_paint_over_quality) {
                          send_this_stripe = true;
                      } else if (local_current_output_mode == OutputMode::H264) {
                          send_this_stripe = true;
                          is_h264_idr_paintover_this_stripe = true;
                      }
                      if (send_this_stripe) paint_over_sent[i] = true;
                  }
              }
          }

          if (hash_calculated_this_iteration) {
              previous_hashes[i] = current_hash;
          }

          if (send_this_stripe) {
            any_stripe_encoded_this_frame = true;
            total_stripes_encoded_this_interval++;
            if (local_current_output_mode == OutputMode::JPEG) {
              int quality_to_use = current_jpeg_qualities[i];
              if (paint_over_sent[i] && local_current_use_paint_over_quality &&
                  no_motion_frame_counts[i] >= local_current_paint_over_trigger_frames) {
                   quality_to_use = local_current_paint_over_jpeg_quality;
              }

              std::packaged_task<StripeEncodeResult(
                int, int, int, int, const unsigned char*, int, int, int, int)>
                task(encode_stripe_jpeg);
              futures.push_back(task.get_future());
              threads.push_back(std::thread(
                std::move(task), i, start_y, current_stripe_height,
                local_capture_width_actual,
                shm_data_ptr,
                shm_stride_bytes,
                shm_bytes_per_pixel,
                quality_to_use,
                this->frame_counter));
            } else {
              if (this->vaapi_operational) {
                std::packaged_task<StripeEncodeResult()> task([=]() {
                    bool force_idr = g_vaapi_force_next_idr_global.exchange(false);
                    return encode_fullframe_vaapi(
                        local_capture_width_actual, local_capture_height_actual, local_current_target_fps,
                        full_frame_y_plane_.data(), full_frame_y_stride_,
                        full_frame_u_plane_.data(), full_frame_u_stride_,
                        full_frame_v_plane_.data(), full_frame_v_stride_,
                        this->frame_counter, force_idr
                    );
                });
                futures.push_back(task.get_future());
                threads.push_back(std::thread(std::move(task)));
              
              } else if (this->nvenc_operational) {
                std::packaged_task<StripeEncodeResult()> task([=]() {
                    bool force_idr = g_nvenc_force_next_idr_global.exchange(false);
                    return encode_fullframe_nvenc(
                        local_capture_width_actual, local_capture_height_actual,
                        full_frame_y_plane_.data(), full_frame_y_stride_,
                        full_frame_u_plane_.data(), full_frame_u_stride_,
                        full_frame_v_plane_.data(), full_frame_v_stride_,
                        this->yuv_planes_are_i444_, this->frame_counter, force_idr
                    );
                });
                futures.push_back(task.get_future());
                threads.push_back(std::thread(std::move(task)));
              } else {
                int crf_for_encode = local_current_h264_crf;
                if (is_h264_idr_paintover_this_stripe && local_current_h264_crf > 10) {
                  crf_for_encode = 10;
                }
                if (is_h264_idr_paintover_this_stripe) {
                  std::lock_guard<std::mutex> lock(g_h264_minimal_store.store_mutex);
                  g_h264_minimal_store.ensure_size(i);
                  if (i < static_cast<int>(g_h264_minimal_store.force_idr_flags.size())) {
                      g_h264_minimal_store.force_idr_flags[i] = true;
                  }
                }

                const uint8_t* y_plane_for_thread = full_frame_y_plane_.data() +
                    static_cast<size_t>(start_y) * full_frame_y_stride_;
                const uint8_t* u_plane_for_thread = full_frame_u_plane_.data() +
                    (static_cast<size_t>(this->yuv_planes_are_i444_ ?
                     start_y : (start_y / 2)) * full_frame_u_stride_);
                const uint8_t* v_plane_for_thread = full_frame_v_plane_.data() +
                    (static_cast<size_t>(this->yuv_planes_are_i444_ ?
                     start_y : (start_y / 2)) * full_frame_v_stride_);

                std::packaged_task<StripeEncodeResult(
                  int, int, int, int,
                  const uint8_t*, int, const uint8_t*, int, const uint8_t*, int,
                  bool, int, int, int, bool)>
                  task(encode_stripe_h264);
                futures.push_back(task.get_future());
                threads.push_back(std::thread(
                  std::move(task), i, start_y, current_stripe_height,
                  local_capture_width_actual,
                  y_plane_for_thread, full_frame_y_stride_,
                  u_plane_for_thread, full_frame_u_stride_,
                  v_plane_for_thread, full_frame_v_stride_,
                  this->yuv_planes_are_i444_,
                  this->frame_counter,
                  crf_for_encode,
                  derived_h264_colorspace_setting,
                  derived_h264_use_full_range
                  ));
              }
            }
          }
        }


        std::vector<StripeEncodeResult> stripe_results;
        stripe_results.reserve(futures.size());
        for (auto& future : futures) {
          try {
            stripe_results.push_back(future.get());
          } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("NVENC_") != std::string::npos) {
                std::cerr << "ENCODE_THREAD_ERROR: " << e.what() << std::endl;
                std::cerr << "Disabling NVENC for this session due to runtime error." << std::endl;
                this->nvenc_operational = false;
                reset_nvenc_encoder();
                g_nvenc_force_next_idr_global = true;
            } else if (std::string(e.what()).find("VAAPI_") != std::string::npos) {
                std::cerr << "ENCODE_THREAD_ERROR: " << e.what() << std::endl;
                std::cerr << "Disabling VAAPI for this session due to runtime error." << std::endl;
                this->vaapi_operational = false;
                reset_vaapi_encoder();
                g_vaapi_force_next_idr_global = true;
            } else {
                std::cerr << "ENCODE_THREAD_ERROR: " << e.what() << std::endl;
            }
            stripe_results.push_back({});
          }
        }
        futures.clear();

        for (StripeEncodeResult& result : stripe_results) {
          if (stripe_callback != nullptr && result.data != nullptr && result.size > 0) {
            stripe_callback(&result, user_data);
          }
        }
        stripe_results.clear();

        for (auto& thread : threads) {
          if (thread.joinable()) {
            thread.join();
          }
        }
        threads.clear();

        this->frame_counter++;
        if (any_stripe_encoded_this_frame) {
          encoded_frame_count++;
        }
        frame_count_loop++;

        auto current_time_for_fps_log = std::chrono::high_resolution_clock::now();
        auto elapsed_time_for_fps_log =
          std::chrono::duration_cast<std::chrono::seconds>(
            current_time_for_fps_log - start_time_loop);

        if (elapsed_time_for_fps_log.count() >= 1) {
          frame_count_loop = 0;
          start_time_loop = std::chrono::high_resolution_clock::now();
        }

        auto current_output_time_log = std::chrono::high_resolution_clock::now();
        auto output_elapsed_time_log =
          std::chrono::duration_cast<std::chrono::seconds>(
            current_output_time_log - last_output_time);

        if (output_elapsed_time_log.count() >= 1) {
          double actual_fps_val =
            (encoded_frame_count > 0 && output_elapsed_time_log.count() > 0)
            ? static_cast<double>(encoded_frame_count) / output_elapsed_time_log.count()
            : 0.0;
          double total_stripes_per_second_val =
            (total_stripes_encoded_this_interval > 0 && output_elapsed_time_log.count() > 0)
            ? static_cast<double>(total_stripes_encoded_this_interval) /
              output_elapsed_time_log.count()
            : 0.0;

          std::cout << "Res: " << local_capture_width_actual << "x"
                    << local_capture_height_actual
                    << " Mode: "
                    << (local_current_output_mode == OutputMode::JPEG ? "JPEG" : (this->vaapi_operational ? "H264 (VAAPI)" : (this->nvenc_operational ? "H264 (NVENC)" : "H264 (CPU)")))
                    << (local_current_output_mode == OutputMode::H264
                        ? (std::string(local_current_h264_fullcolor ?
                                       " CS_IN:I444" : " CS_IN:I420") +
                           (derived_h264_use_full_range ? " FR" : " LR") +
                           (local_current_h264_fullframe ? " FF" : " Striped"))
                        : std::string(""))
                    << " Stripes: " << N_processing_stripes
                    << (local_current_output_mode == OutputMode::H264
                        ? " CRF:" + std::to_string(local_current_h264_crf)
                        : " Q:" + std::to_string(local_current_jpeg_quality))
                    << " EncFPS: " << std::fixed << std::setprecision(2) << actual_fps_val
                    << " EncStripes/s: " << std::fixed << std::setprecision(2)
                    << total_stripes_per_second_val
                    << std::endl;

          encoded_frame_count = 0;
          total_stripes_encoded_this_interval = 0;
          last_output_time = std::chrono::high_resolution_clock::now();
        }

      } else {
        std::cerr << "Failed to capture XImage using XShmGetImage" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }

    if (display) {
        if (shm_image) {
            if (shminfo.shmaddr && shminfo.shmaddr != (char*)-1) {
                 XShmDetach(display, &shminfo);
                 shmdt(shminfo.shmaddr);
                 shminfo.shmaddr = (char*)-1;
            }
            if (shminfo.shmid != -1 && shminfo.shmid != 0) {
                 shmctl(shminfo.shmid, IPC_RMID, 0);
                 shminfo.shmid = -1;
            }
            XDestroyImage(shm_image);
            shm_image = nullptr;
        }
        XCloseDisplay(display);
        display = nullptr;
    }
    std::cout << "Capture loop stopped. X resources released." << std::endl;
  }

  /**
   * @brief Overlays a 32-bit ARGB image onto a BGR(X) frame buffer with alpha blending support.
   *
   * This function takes a source image in 32-bit ARGB format and overlays it at a specified
   * position (image_x, image_y) onto a destination frame buffer that is assumed to be in
   * BGR or BGRX format. It supports transparency via the alpha channel:
   * - Fully opaque pixels are copied directly.
   * - Partially transparent pixels are blended with the existing pixel color.
   * - Fully transparent pixels are skipped.
   *
   * @param image_height Height of the source image in pixels.
   * @param image_width Width of the source image in pixels.
   * @param image_ptr Pointer to the source image data in 32-bit ARGB format.
   *                  Pixels are stored as uint32_t values: (A << 24) | (R << 16) | (G << 8) | B
   * @param image_x X-coordinate (left) where the image should be placed on the frame.
   * @param image_y Y-coordinate (top) where the image should be placed on the frame.
   * @param frame_height Total height of the destination frame buffer in pixels.
   * @param frame_width Total width of the destination frame buffer in pixels.
   * @param frame_ptr Pointer to the destination frame buffer in BGR or BGRX format.
   *                  Each pixel is represented by 3 or 4 bytes per pixel respectively.
   * @param frame_stride_bytes Number of bytes per row in the destination frame buffer.
   * @param frame_bytes_per_pixel Number of bytes used to represent a single pixel in the frame buffer.
   *                              Expected value is 3 (BGR) or 4 (BGRX).
   */
  void overlay_image(int image_height, int image_width, const uint32_t *image_ptr,
                     int image_x, int image_y, int frame_height, int frame_width,
                     unsigned char *frame_ptr, int frame_stride_bytes, int frame_bytes_per_pixel) {
    for (int y = 0; y < image_height; ++y) {
      for (int x = 0; x < image_width; ++x) {
        uint32_t src_pixel = image_ptr[y * image_width + x];
        uint8_t alpha = (src_pixel >> 24) & 0xFF;
        uint8_t red = (src_pixel >> 16) & 0xFF;
        uint8_t green = (src_pixel >> 8) & 0xFF;
        uint8_t blue = src_pixel & 0xFF;

        int target_x = image_x + x;
        int target_y = image_y + y;

        if (target_y >= 0 && target_y < frame_height &&
            target_x >= 0 && target_x < frame_width) {

          unsigned char *dst_pixel = frame_ptr +
                                      target_y * frame_stride_bytes +
                                      target_x * frame_bytes_per_pixel;

          if (alpha == 255)
          {
            dst_pixel[0] = blue;
            dst_pixel[1] = green;
            dst_pixel[2] = red;
          }
          else if (alpha > 0)
          {
            dst_pixel[0] = (blue * alpha + dst_pixel[0] * (255 - alpha)) / 255;
            dst_pixel[1] = (green * alpha + dst_pixel[1] * (255 - alpha)) / 255;
            dst_pixel[2] = (red * alpha + dst_pixel[2] * (255 - alpha)) / 255;
          }
        }
      }
    }
  }
};

/**
 * @brief Encodes a horizontal stripe of an image from shared memory into JPEG format.
 *
 * This function takes a segment of raw image data (assumed to be in BGRX or similar format
 * where BGR components are accessible) from a shared memory buffer, converts it to RGB,
 * and then compresses it into a JPEG image. The resulting JPEG data is prepended with a
 * custom 4-byte header containing the frame ID and stripe's Y-offset.
 *
 * @param thread_id Identifier for the calling thread, primarily for logging purposes.
 * @param stripe_y_start The Y-coordinate of the top edge of the stripe within the full source image.
 * @param stripe_height The height of the stripe in pixels.
 * @param capture_width_actual The width of the stripe in pixels.
 * @param shm_data_base Pointer to the beginning of the *full* source image data in shared memory.
 *                      The function calculates the offset to the stripe using stripe_y_start.
 * @param shm_stride_bytes The number of bytes from the start of one row of the source image
 *                         to the start of the next row (pitch).
 * @param shm_bytes_per_pixel The number of bytes per pixel in the source shared memory image
 *                            (e.g., 4 for BGRX, 3 for BGR).
 * @param jpeg_quality The desired JPEG quality, ranging from 0 (lowest) to 100 (highest).
 * @param frame_counter An identifier for the current frame, included in the output header.
 * @return A StripeEncodeResult struct.
 *         - If successful, `type` is `StripeDataType::JPEG`, `data` points to the
 *           encoded JPEG (including a 4-byte custom header: frame_id (uint16_t MSB)
 *           and stripe_y_start (uint16_t MSB)), and `size` is the total size of `data`.
 *         - On failure (e.g., invalid input, memory allocation error), `type` is
 *           `StripeDataType::UNKNOWN`, and `data` is `nullptr`.
 *         The caller is responsible for freeing `result.data` using
 *         `free_stripe_encode_result_data` or `delete[]`.
 */
StripeEncodeResult encode_stripe_jpeg(
  int thread_id,
  int stripe_y_start,
  int stripe_height,
  int capture_width_actual,
  const unsigned char* shm_data_base,
  int shm_stride_bytes,
  int shm_bytes_per_pixel,
  int jpeg_quality,
  int frame_counter) {
  StripeEncodeResult result;
  result.type = StripeDataType::JPEG;
  result.stripe_y_start = stripe_y_start;
  result.stripe_height = stripe_height;
  result.frame_id = frame_counter;

  if (!shm_data_base || stripe_height <= 0 || capture_width_actual <= 0 ||
      shm_bytes_per_pixel <=0) {
    std::cerr << "JPEG T" << thread_id
              << ": Invalid input for JPEG encoding from SHM." << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }

  jpeg_compress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  cinfo.image_width = capture_width_actual;
  cinfo.image_height = stripe_height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, jpeg_quality, TRUE);

  unsigned char* jpeg_buffer = nullptr;
  unsigned long jpeg_size_temp = 0;
  jpeg_mem_dest(&cinfo, &jpeg_buffer, &jpeg_size_temp);

  jpeg_start_compress(&cinfo, TRUE);

  std::vector<unsigned char> rgb_row_buffer(static_cast<size_t>(capture_width_actual) * 3);
  JSAMPROW row_pointer[1];
  row_pointer[0] = rgb_row_buffer.data();

  for (int y_in_stripe = 0; y_in_stripe < stripe_height; ++y_in_stripe) {
    const unsigned char* shm_current_row_in_full_frame_ptr =
        shm_data_base + static_cast<size_t>(stripe_y_start + y_in_stripe) * shm_stride_bytes;

    for (int x = 0; x < capture_width_actual; ++x) {
      const unsigned char* shm_pixel =
          shm_current_row_in_full_frame_ptr + static_cast<size_t>(x) * shm_bytes_per_pixel;
      rgb_row_buffer[static_cast<size_t>(x) * 3 + 0] = shm_pixel[2]; // R
      rgb_row_buffer[static_cast<size_t>(x) * 3 + 1] = shm_pixel[1]; // G
      rgb_row_buffer[static_cast<size_t>(x) * 3 + 2] = shm_pixel[0]; // B
    }
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);

  if (jpeg_size_temp > 0 && jpeg_buffer) {
    int padding_size = 4; // For frame_counter and stripe_y_start
    result.data = new (std::nothrow) unsigned char[jpeg_size_temp + padding_size];
    if (!result.data) {
      std::cerr << "JPEG T" << thread_id
                << ": Failed to allocate memory for JPEG output." << std::endl;
      jpeg_destroy_compress(&cinfo);
      if (jpeg_buffer)
        free(jpeg_buffer); // jpeg_mem_dest uses malloc
      result.type = StripeDataType::UNKNOWN;
      return result;
    }

    uint16_t frame_counter_net = htons(static_cast<uint16_t>(frame_counter % 65536));
    uint16_t stripe_y_start_net = htons(static_cast<uint16_t>(stripe_y_start));

    std::memcpy(result.data, &frame_counter_net, 2);
    std::memcpy(result.data + 2, &stripe_y_start_net, 2);
    std::memcpy(result.data + padding_size, jpeg_buffer, jpeg_size_temp);
    result.size = static_cast<int>(jpeg_size_temp) + padding_size;
  } else {
    result.size = 0;
    result.data = nullptr;
  }

  jpeg_destroy_compress(&cinfo);
  if (jpeg_buffer) {
    free(jpeg_buffer); // jpeg_mem_dest uses malloc
  }
  return result;
}

/**
 * @brief Encodes a horizontal stripe of YUV data into H.264 format using x264.
 *
 * This function manages x264 encoder instances (one per `thread_id`) stored in
 * `g_h264_minimal_store`. It initializes or reconfigures the encoder if necessary
 * based on dimensions, colorspace, or CRF changes. The input YUV data is copied
 * into an x264 picture structure and then encoded. The output H.264 NAL units are
 * prepended with a custom 10-byte header.
 *
 * @param thread_id Identifier for the calling thread. This ID is used to access a
 *                  dedicated x264 encoder instance from `g_h264_minimal_store`.
 * @param stripe_y_start The Y-coordinate of the top edge of the stripe.
 * @param stripe_height The height of the stripe in pixels. For H.264, this should
 *                      typically be an even number.
 * @param capture_width_actual The width of the stripe in pixels. For H.264, this should
 *                             typically be an even number.
 * @param y_plane_stripe_start Pointer to the start of the Y (luma) plane data for this stripe.
 * @param y_stride Stride (bytes per row) of the Y plane.
 * @param u_plane_stripe_start Pointer to the start of the U (chroma) plane data for this stripe.
 * @param u_stride Stride (bytes per row) of the U plane.
 * @param v_plane_stripe_start Pointer to the start of the V (chroma) plane data for this stripe.
 * @param v_stride Stride (bytes per row) of the V plane.
 * @param is_i444_input True if the input YUV data is in I444 format (chroma planes have
 *                      the same dimensions as luma). False if I420 (chroma planes are
 *                      half width and half height of luma).
 * @param frame_counter An identifier for the current frame, used for setting PTS (Presentation
 *                      Timestamp) and included in the output header.
 * @param current_crf_setting The H.264 Constant Rate Factor (CRF) to use for encoding.
 *                            Lower values mean higher quality and larger file size.
 * @param colorspace_setting An integer indicating the input YUV format for x264
 *                           (e.g., 420 for I420, 444 for I444). This determines the
 *                           `param.i_csp` for x264.
 * @param use_full_range True if full range color (e.g., JPEG levels 0-255) should be
 *                       signaled in the VUI (Video Usability Information). False for
 *                       limited range (e.g., TV levels 16-235).
 * @return A StripeEncodeResult struct.
 *         - If successful, `type` is `StripeDataType::H264`, `data` points to the
 *           encoded H.264 NAL units (including a 10-byte custom header: type tag (0x04),
 *           frame type, frame_id (uint16_t), stripe_y_start (uint16_t), width (uint16_t),
 *           height (uint16_t); multi-byte fields are MSB), and `size` is the total size.
 *         - On failure, `type` is `StripeDataType::UNKNOWN`, `data` is `nullptr`.
 *         The caller is responsible for freeing `result.data`.
 */
StripeEncodeResult encode_stripe_h264(
  int thread_id,
  int stripe_y_start,
  int stripe_height,
  int capture_width_actual,
  const uint8_t* y_plane_stripe_start, int y_stride,
  const uint8_t* u_plane_stripe_start, int u_stride,
  const uint8_t* v_plane_stripe_start, int v_stride,
  bool is_i444_input,
  int frame_counter,
  int current_crf_setting,
  int colorspace_setting,
  bool use_full_range) {

  StripeEncodeResult result;
  result.type = StripeDataType::H264;
  result.stripe_y_start = stripe_y_start;
  result.stripe_height = stripe_height;
  result.frame_id = frame_counter;
  result.data = nullptr;
  result.size = 0;

  if (!y_plane_stripe_start || !u_plane_stripe_start || !v_plane_stripe_start) {
    std::cerr << "H264 T" << thread_id << ": Error - null YUV plane data for stripe Y"
              << stripe_y_start << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }
  if (stripe_height <= 0 || capture_width_actual <= 0) {
    std::cerr << "H264 T" << thread_id << ": Invalid dimensions ("
              << capture_width_actual << "x" << stripe_height
              << ") for stripe Y" << stripe_y_start << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }
  if (capture_width_actual % 2 != 0 || stripe_height % 2 != 0) {
    std::cerr << "H264 T" << thread_id << ": Warning - Odd dimensions ("
              << capture_width_actual << "x" << stripe_height
              << ") for stripe Y" << stripe_y_start
              << ". Encoder might behave unexpectedly or fail." << std::endl;
  }

  x264_t* current_encoder = nullptr;
  x264_picture_t* current_pic_in_ptr = nullptr;
  int target_x264_csp;
  switch (colorspace_setting) {
    case 444:
      target_x264_csp = X264_CSP_I444;
      break;
    case 420:
    default:
      target_x264_csp = X264_CSP_I420;
      break;
  }

  {
    std::lock_guard<std::mutex> lock(g_h264_minimal_store.store_mutex);
    g_h264_minimal_store.ensure_size(thread_id);

    bool is_first_init = !g_h264_minimal_store.initialized_flags[thread_id];
    bool dims_changed = !is_first_init &&
                        (g_h264_minimal_store.initialized_widths[thread_id] !=
                            capture_width_actual ||
                         g_h264_minimal_store.initialized_heights[thread_id] !=
                            stripe_height);
    bool cs_or_fr_changed = !is_first_init &&
                            (g_h264_minimal_store.initialized_csps[thread_id] !=
                                target_x264_csp ||
                             g_h264_minimal_store.initialized_colorspaces[thread_id] !=
                                colorspace_setting ||
                             g_h264_minimal_store.initialized_full_range_flags[thread_id] !=
                                use_full_range);

    bool needs_crf_reinit = false;
    if (!is_first_init &&
        g_h264_minimal_store.initialized_crfs[thread_id] != current_crf_setting) {
        needs_crf_reinit = true;
    }

    bool perform_full_reinit = is_first_init || dims_changed || cs_or_fr_changed;

    if (perform_full_reinit) {
      if (g_h264_minimal_store.encoders[thread_id]) {
        x264_encoder_close(g_h264_minimal_store.encoders[thread_id]);
        g_h264_minimal_store.encoders[thread_id] = nullptr;
      }
      if (g_h264_minimal_store.pics_in_ptrs[thread_id]) {
        if (g_h264_minimal_store.initialized_flags[thread_id]) {
          x264_picture_clean(g_h264_minimal_store.pics_in_ptrs[thread_id]);
        }
        delete g_h264_minimal_store.pics_in_ptrs[thread_id];
        g_h264_minimal_store.pics_in_ptrs[thread_id] = nullptr;
      }
      g_h264_minimal_store.initialized_flags[thread_id] = false;

      x264_param_t param;
      if (x264_param_default_preset(&param, "ultrafast", "zerolatency") < 0) {
        std::cerr << "H264 T" << thread_id
                  << ": x264_param_default_preset FAILED." << std::endl;
        result.type = StripeDataType::UNKNOWN;
      } else {
        param.i_width = capture_width_actual;
        param.i_height = stripe_height;
        param.i_csp = target_x264_csp;
        param.i_fps_num = 60;
        param.i_fps_den = 1;
        param.i_keyint_max = X264_KEYINT_MAX_INFINITE;
        param.rc.f_rf_constant =
            static_cast<float>(std::max(0, std::min(51, current_crf_setting)));
        param.rc.i_rc_method = X264_RC_CRF;
        param.b_repeat_headers = 1;
        param.b_annexb = 1;
        param.i_sync_lookahead = 0;
        param.i_bframe = 0;
        param.i_threads = 1;
        param.i_log_level = X264_LOG_ERROR;
        param.vui.b_fullrange = use_full_range ? 1 : 0;
        param.vui.i_sar_width = 1;
        param.vui.i_sar_height = 1;
        if (param.i_csp == X264_CSP_I444) {
             param.vui.i_colorprim = 1;
             param.vui.i_transfer = 1;
             param.vui.i_colmatrix = 1;
             x264_param_apply_profile(&param, "high444");
        } else { // I420
           param.vui.i_colorprim = 1;
           param.vui.i_transfer  = 1;
           param.vui.i_colmatrix = 1;
           x264_param_apply_profile(&param, "baseline");
        }
        param.b_aud = 0;

        g_h264_minimal_store.encoders[thread_id] = x264_encoder_open(&param);
        if (!g_h264_minimal_store.encoders[thread_id]) {
          std::cerr << "H264 T" << thread_id << ": x264_encoder_open FAILED." << std::endl;
          result.type = StripeDataType::UNKNOWN;
        } else {
          g_h264_minimal_store.pics_in_ptrs[thread_id] = new (std::nothrow) x264_picture_t();
          if (!g_h264_minimal_store.pics_in_ptrs[thread_id]) {
            std::cerr << "H264 T" << thread_id
                      << ": FAILED to new x264_picture_t." << std::endl;
            x264_encoder_close(g_h264_minimal_store.encoders[thread_id]);
            g_h264_minimal_store.encoders[thread_id] = nullptr;
            result.type = StripeDataType::UNKNOWN;
          } else {
            x264_picture_init(g_h264_minimal_store.pics_in_ptrs[thread_id]);
            if (x264_picture_alloc(g_h264_minimal_store.pics_in_ptrs[thread_id],
                                   param.i_csp, param.i_width, param.i_height) < 0) {
              std::cerr << "H264 T" << thread_id << ": x264_picture_alloc FAILED for CSP "
                        << param.i_csp << "." << std::endl;
              delete g_h264_minimal_store.pics_in_ptrs[thread_id];
              g_h264_minimal_store.pics_in_ptrs[thread_id] = nullptr;
              x264_encoder_close(g_h264_minimal_store.encoders[thread_id]);
              g_h264_minimal_store.encoders[thread_id] = nullptr;
              result.type = StripeDataType::UNKNOWN;
            } else {
              g_h264_minimal_store.initialized_flags[thread_id] = true;
              g_h264_minimal_store.initialized_widths[thread_id] = param.i_width;
              g_h264_minimal_store.initialized_heights[thread_id] = param.i_height;
              g_h264_minimal_store.initialized_crfs[thread_id] = current_crf_setting;
              g_h264_minimal_store.initialized_csps[thread_id] = param.i_csp;
              g_h264_minimal_store.initialized_colorspaces[thread_id] = colorspace_setting;
              g_h264_minimal_store.initialized_full_range_flags[thread_id] = use_full_range;
              g_h264_minimal_store.force_idr_flags[thread_id] = true;
            }
          }
        }
      }
    } else if (needs_crf_reinit) {
      x264_t* encoder_to_reconfig = g_h264_minimal_store.encoders[thread_id];
      if (encoder_to_reconfig) {
        x264_param_t params_for_reconfig;
        x264_encoder_parameters(encoder_to_reconfig, &params_for_reconfig);
        params_for_reconfig.rc.f_rf_constant =
          static_cast<float>(std::max(0, std::min(51, current_crf_setting)));
        if (x264_encoder_reconfig(encoder_to_reconfig, &params_for_reconfig) == 0) {
          g_h264_minimal_store.initialized_crfs[thread_id] = current_crf_setting;
        } else {
          std::cerr << "H264 T" << thread_id
                    << ": x264_encoder_reconfig for CRF FAILED. Old CRF "
                    << g_h264_minimal_store.initialized_crfs[thread_id]
                    << " may persist." << std::endl;
        }
      }
    }

    if (g_h264_minimal_store.initialized_flags[thread_id]) {
      current_encoder = g_h264_minimal_store.encoders[thread_id];
      current_pic_in_ptr = g_h264_minimal_store.pics_in_ptrs[thread_id];
    }
  }

  if (result.type == StripeDataType::UNKNOWN) return result;
  if (!current_encoder || !current_pic_in_ptr) {
    std::cerr << "H264 T" << thread_id << ": Encoder/Pic not ready post-init for Y"
              << stripe_y_start << "." << std::endl;
    result.type = StripeDataType::UNKNOWN; return result;
  }

  int chroma_width_src = is_i444_input ? capture_width_actual : (capture_width_actual / 2);
  int chroma_height_src = is_i444_input ? stripe_height : (stripe_height / 2);

  if (!is_i444_input) {
      if (capture_width_actual % 2 != 0 && chroma_width_src > 0)
        chroma_width_src = (capture_width_actual -1) / 2;
      if (stripe_height % 2 != 0 && chroma_height_src > 0)
        chroma_height_src = (stripe_height-1) / 2;
  }

  libyuv::CopyPlane(y_plane_stripe_start, y_stride,
                    current_pic_in_ptr->img.plane[0], current_pic_in_ptr->img.i_stride[0],
                    capture_width_actual, stripe_height);
  if (chroma_width_src > 0 && chroma_height_src > 0) {
    libyuv::CopyPlane(u_plane_stripe_start, u_stride,
                      current_pic_in_ptr->img.plane[1], current_pic_in_ptr->img.i_stride[1],
                      chroma_width_src, chroma_height_src);
    libyuv::CopyPlane(v_plane_stripe_start, v_stride,
                      current_pic_in_ptr->img.plane[2], current_pic_in_ptr->img.i_stride[2],
                      chroma_width_src, chroma_height_src);
  }

  current_pic_in_ptr->i_pts = static_cast<int64_t>(frame_counter);

  bool force_idr_now = false;
  {
    std::lock_guard<std::mutex> lock(g_h264_minimal_store.store_mutex);
    g_h264_minimal_store.ensure_size(thread_id);
    if (g_h264_minimal_store.initialized_flags[thread_id] &&
        thread_id < static_cast<int>(g_h264_minimal_store.force_idr_flags.size()) &&
        g_h264_minimal_store.force_idr_flags[thread_id]) {
      force_idr_now = true;
    }
  }
  current_pic_in_ptr->i_type = force_idr_now ? X264_TYPE_IDR : X264_TYPE_AUTO;

  x264_nal_t* nals = nullptr;
  int i_nals = 0;
  x264_picture_t pic_out;
  x264_picture_init(&pic_out);

  int frame_size = x264_encoder_encode(current_encoder, &nals, &i_nals,
                                       current_pic_in_ptr, &pic_out);

  if (frame_size < 0) {
    std::cerr << "H264 T" << thread_id << ": x264_encoder_encode FAILED: " << frame_size
              << " (Y" << stripe_y_start << ")" << std::endl;
    result.type = StripeDataType::UNKNOWN; return result;
  }

  if (frame_size > 0) {
    if (force_idr_now && pic_out.b_keyframe &&
        (pic_out.i_type == X264_TYPE_IDR || pic_out.i_type == X264_TYPE_I)) {
      std::lock_guard<std::mutex> lock(g_h264_minimal_store.store_mutex);
      if (thread_id < static_cast<int>(g_h264_minimal_store.force_idr_flags.size())) {
        g_h264_minimal_store.force_idr_flags[thread_id] = false;
      }
    }

    const unsigned char DATA_TYPE_H264_STRIPED_TAG = 0x04;
    unsigned char frame_type_header_byte = 0x00;
    if (pic_out.i_type == X264_TYPE_IDR) frame_type_header_byte = 0x01;
    else if (pic_out.i_type == X264_TYPE_I) frame_type_header_byte = 0x02;

    int header_sz = 10;
    int total_sz = frame_size + header_sz;
    result.data = new (std::nothrow) unsigned char[total_sz];
    if (!result.data) {
      std::cerr << "H264 T" << thread_id << ": new result.data FAILED (Y"
                << stripe_y_start << ")" << std::endl;
      result.type = StripeDataType::UNKNOWN; return result;
    }

    result.data[0] = DATA_TYPE_H264_STRIPED_TAG;
    result.data[1] = frame_type_header_byte;
    uint16_t net_val;
    net_val = htons(static_cast<uint16_t>(result.frame_id % 65536));
    std::memcpy(result.data + 2, &net_val, 2);
    net_val = htons(static_cast<uint16_t>(result.stripe_y_start));
    std::memcpy(result.data + 4, &net_val, 2);
    net_val = htons(static_cast<uint16_t>(capture_width_actual));
    std::memcpy(result.data + 6, &net_val, 2);
    net_val = htons(static_cast<uint16_t>(result.stripe_height));
    std::memcpy(result.data + 8, &net_val, 2);

    unsigned char* payload_ptr = result.data + header_sz;
    size_t bytes_copied = 0;
    for (int k = 0; k < i_nals; ++k) {
      if (bytes_copied + nals[k].i_payload > static_cast<size_t>(frame_size)) {
        std::cerr << "H264 T" << thread_id
                  << ": NAL copy overflow detected (Y" << stripe_y_start << ")" << std::endl;
        delete[] result.data; result.data = nullptr; result.size = 0;
        result.type = StripeDataType::UNKNOWN; return result;
      }
      std::memcpy(payload_ptr + bytes_copied, nals[k].p_payload, nals[k].i_payload);
      bytes_copied += nals[k].i_payload;
    }
    result.size = total_sz;
  } else {
    result.data = nullptr;
    result.size = 0;
  }
  return result;
}

/**
 * @brief Calculates a 64-bit XXH3 hash for a stripe of YUV data.
 *
 * This function processes the Y, U, and V planes of a given YUV image stripe
 * to compute a single hash value. This is typically used for damage detection
 * by comparing hashes of the same stripe across consecutive frames.
 *
 * @param y_plane_stripe_start Pointer to the beginning of the Y (luma) plane data
 *                             for the stripe.
 * @param y_stride The stride (bytes per row) of the Y plane.
 * @param u_plane_stripe_start Pointer to the beginning of the U (chroma) plane data
 *                             for the stripe.
 * @param u_stride The stride (bytes per row) of the U plane.
 * @param v_plane_stripe_start Pointer to the beginning of the V (chroma) plane data
 *                             for the stripe.
 * @param v_stride The stride (bytes per row) of the V plane.
 * @param width The width of the luma plane of the stripe in pixels.
 * @param height The height of the luma plane of the stripe in pixels.
 * @param is_i420 True if the YUV format is I420 (chroma planes are half width and
 *                half height of the luma plane). False if I444 (chroma planes have
 *                the same dimensions as the luma plane).
 * @return A 64-bit hash value representing the content of the YUV stripe.
 *         Returns 0 if input parameters are invalid (e.g., null pointers,
 *         non-positive dimensions).
 */
uint64_t calculate_yuv_stripe_hash(const uint8_t* y_plane_stripe_start, int y_stride,
                                   const uint8_t* u_plane_stripe_start, int u_stride,
                                   const uint8_t* v_plane_stripe_start, int v_stride,
                                   int width, int height, bool is_i420) {
    if (!y_plane_stripe_start || !u_plane_stripe_start || !v_plane_stripe_start ||
        width <= 0 || height <= 0) {
        return 0;
    }

    XXH3_state_t hash_state;
    XXH3_64bits_reset(&hash_state);

    for (int r = 0; r < height; r += 12) {
        XXH3_64bits_update(&hash_state, y_plane_stripe_start +
                           static_cast<size_t>(r) * y_stride, width);
    }

    int chroma_width = is_i420 ? (width / 2) : width;
    int chroma_height = is_i420 ? (height / 2) : height;

    if (is_i420) {
        if (width % 2 != 0 && chroma_width > 0) chroma_width = (width - 1) / 2;
        if (height % 2 != 0 && chroma_height > 0) chroma_height = (height - 1) / 2;
    }

    if (chroma_width <= 0 || chroma_height <= 0) {
        // Dimensions invalid
    } else {
        for (int r = 0; r < chroma_height; r += 12) {
            XXH3_64bits_update(&hash_state, u_plane_stripe_start +
                               static_cast<size_t>(r) * u_stride, chroma_width);
        }
        for (int r = 0; r < chroma_height; r += 12) {
            XXH3_64bits_update(&hash_state, v_plane_stripe_start +
                               static_cast<size_t>(r) * v_stride, chroma_width);
        }
    }
    return XXH3_64bits_digest(&hash_state);
}

/**
 * @brief Calculates a 64-bit XXH3 hash for a stripe of BGR(X) image data from shared memory.
 *
 * This function reads pixel data row by row from the provided shared memory buffer,
 * extracts the B, G, and R components (assuming BGR ordering, e.g., BGRX or BGR),
 * and computes a hash of this BGR data. This is useful for damage detection on
 * image data that is natively in a BGR-like format.
 *
 * @param shm_stripe_physical_start Pointer to the beginning of the stripe's pixel data
 *                                  within the shared memory buffer.
 * @param shm_stride_bytes The stride (bytes per row) of the image in shared memory.
 * @param stripe_width The width of the stripe in pixels.
 * @param stripe_height The height of the stripe in pixels.
 * @param shm_bytes_per_pixel The number of bytes per pixel in the shared memory image
 *                            (e.g., 4 for BGRX, 3 for BGR). It's assumed that the
 *                            blue component is at offset 0, green at 1, and red at 2
 *                            within each pixel.
 * @return A 64-bit hash value representing the BGR content of the stripe.
 *         Returns 0 if input parameters are invalid (e.g., null pointer,
 *         non-positive dimensions, insufficient bytes per pixel).
 */
uint64_t calculate_bgr_stripe_hash_from_shm(const unsigned char* shm_stripe_physical_start,
                                            int shm_stride_bytes,
                                            int stripe_width, int stripe_height,
                                            int shm_bytes_per_pixel) {
    if (!shm_stripe_physical_start || stripe_width <= 0 || stripe_height <= 0 ||
        shm_bytes_per_pixel < 3) {
        return 0;
    }

    XXH3_state_t hash_state;
    XXH3_64bits_reset(&hash_state);

    std::vector<unsigned char> bgr_row_buffer(static_cast<size_t>(stripe_width) * 3);

    for (int r = 0; r < stripe_height; ++r) {
        const unsigned char* shm_row_ptr =
            shm_stripe_physical_start + static_cast<size_t>(r) * shm_stride_bytes;
        for (int x = 0; x < stripe_width; ++x) {
            const unsigned char* shm_pixel =
                shm_row_ptr + static_cast<size_t>(x) * shm_bytes_per_pixel;
            bgr_row_buffer[static_cast<size_t>(x) * 3 + 0] = shm_pixel[0]; // B
            bgr_row_buffer[static_cast<size_t>(x) * 3 + 1] = shm_pixel[1]; // G
            bgr_row_buffer[static_cast<size_t>(x) * 3 + 2] = shm_pixel[2]; // R
        }
        XXH3_64bits_update(&hash_state, bgr_row_buffer.data(), bgr_row_buffer.size());
    }
    return XXH3_64bits_digest(&hash_state);
}

/**
 * @brief Calculates a 64-bit XXH3 hash for a stripe of RGB data.
 *
 * This function takes a vector of unsigned chars representing tightly packed RGB data
 * (R,G,B,R,G,B...) and computes its hash.
 * This function is kept for potential compatibility or specific use cases where RGB data
 * is already prepared in a contiguous buffer. New logic for screen capture should
 * generally use YUV or BGR specific hash functions (like
 * `calculate_yuv_stripe_hash` or `calculate_bgr_stripe_hash_from_shm`)
 * that operate closer to the source data format.
 *
 * @param rgb_data A `std::vector<unsigned char>` containing the RGB pixel data for the stripe.
 *                 The data is assumed to be in R,G,B order, tightly packed.
 * @return A 64-bit hash value of the RGB data.
 *         Returns 0 if `rgb_data` is empty.
 */
uint64_t calculate_rgb_stripe_hash(const std::vector<unsigned char>& rgb_data) {
  if (rgb_data.empty()) return 0;
  return XXH3_64bits(rgb_data.data(), rgb_data.size());
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Python type wrapper for the C++ CaptureSettings struct.
 * This struct allows CaptureSettings to be created and manipulated from Python,
 * with members directly corresponding to the C++ struct's fields. It also
 * safely manages the memory for the watermark_path string.
 */
typedef struct {
  PyObject_HEAD
  CaptureSettings settings;
  char* watermark_path_buffer;
} PyCaptureSettings;


/**
 * @brief Deallocator for PyCaptureSettings objects.
 * Frees the dynamically allocated buffer for the watermark path string before
 * freeing the object itself.
 * @param self A pointer to the PyCaptureSettings object to be deallocated.
 */
static void PyCaptureSettings_dealloc(PyCaptureSettings* self) {
  free(self->watermark_path_buffer);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/**
 * @brief Allocator for PyCaptureSettings objects.
 * Creates a new PyCaptureSettings object and initializes the wrapped C++
 * settings with default values.
 * @param type The Python type object.
 * @param args Unused.
 * @param kwds Unused.
 * @return A new PyObject pointer to the allocated PyCaptureSettings.
 */
static PyObject* PyCaptureSettings_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  PyCaptureSettings* self = (PyCaptureSettings*)type->tp_alloc(type, 0);
  if (self) {
    self->settings = {};
    self->watermark_path_buffer = NULL;
  }
  return (PyObject*)self;
}

/**
 * @brief Getter for the 'watermark_path' attribute of a PyCaptureSettings object.
 * @param self The PyCaptureSettings object.
 * @param closure Unused.
 * @return A Python string object representing the path, or Py_None if not set.
 */
static PyObject* PyCaptureSettings_get_watermark_path(PyCaptureSettings* self, void* closure) {
  if (self->settings.watermark_path == NULL) {
    Py_RETURN_NONE;
  }
  return PyUnicode_FromString(self->settings.watermark_path);
}

/**
 * @brief Setter for the 'watermark_path' attribute of a PyCaptureSettings object.
 * Safely handles converting the Python string to a C-style string and manages
 * the underlying memory buffer.
 * @param self The PyCaptureSettings object.
 * @param value The new Python string value for the path, or Py_None.
 * @param closure Unused.
 * @return 0 on success, -1 on failure.
 */
static int PyCaptureSettings_set_watermark_path(PyCaptureSettings* self, PyObject* value, void* closure) {
  if (value == NULL || value == Py_None) {
    free(self->watermark_path_buffer);
    self->watermark_path_buffer = NULL;
    self->settings.watermark_path = NULL;
    return 0;
  }
  if (!PyUnicode_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "The 'watermark_path' attribute value must be a string or None");
    return -1;
  }
  const char* path_str = PyUnicode_AsUTF8(value);
  if (path_str == NULL) {
    return -1;
  }

  free(self->watermark_path_buffer);
  self->watermark_path_buffer = strdup(path_str);
  self->settings.watermark_path = self->watermark_path_buffer;
  return 0;
}

/**
 * @brief Member definitions for the PyCaptureSettings type.
 * This array maps the fields of the C++ CaptureSettings struct to Python
 * attributes, allowing direct access.
 */
static PyMemberDef PyCaptureSettings_members[] = {
  { "capture_width", T_INT, offsetof(PyCaptureSettings, settings.capture_width), 0, "capture width" },
  { "capture_height", T_INT, offsetof(PyCaptureSettings, settings.capture_height), 0, "capture height" },
  { "capture_x", T_INT, offsetof(PyCaptureSettings, settings.capture_x), 0, "capture x" },
  { "capture_y", T_INT, offsetof(PyCaptureSettings, settings.capture_y), 0, "capture y" },
  { "target_fps", T_DOUBLE, offsetof(PyCaptureSettings, settings.target_fps), 0, "target fps" },
  { "jpeg_quality", T_INT, offsetof(PyCaptureSettings, settings.jpeg_quality), 0, "jpeg quality" },
  { "paint_over_jpeg_quality", T_INT, offsetof(PyCaptureSettings, settings.paint_over_jpeg_quality), 0, "paint over jpeg quality" },
  { "use_paint_over_quality", T_BOOL, offsetof(PyCaptureSettings, settings.use_paint_over_quality), 0, "use paint over quality" },
  { "paint_over_trigger_frames", T_INT, offsetof(PyCaptureSettings, settings.paint_over_trigger_frames), 0, "paint over trigger frames" },
  { "damage_block_threshold", T_INT, offsetof(PyCaptureSettings, settings.damage_block_threshold), 0, "damage block threshold" },
  { "damage_block_duration", T_INT, offsetof(PyCaptureSettings, settings.damage_block_duration), 0, "damage block duration" },
  { "output_mode", T_INT, offsetof(PyCaptureSettings, settings.output_mode), 0, "output mode" },
  { "h264_crf", T_INT, offsetof(PyCaptureSettings, settings.h264_crf), 0, "h264 crf" },
  { "h264_fullcolor", T_BOOL, offsetof(PyCaptureSettings, settings.h264_fullcolor), 0, "h264 fullcolor" },
  { "h264_fullframe", T_BOOL, offsetof(PyCaptureSettings, settings.h264_fullframe), 0, "h264 fullframe" },
  { "capture_cursor", T_BOOL, offsetof(PyCaptureSettings, settings.capture_cursor), 0, "capture cursor" },
  { "watermark_location_enum", T_INT, offsetof(PyCaptureSettings, settings.watermark_location_enum), 0, "watermark location enum" },
  { "vaapi_render_node_index", T_INT, offsetof(PyCaptureSettings, settings.vaapi_render_node_index), 0, "vaapi render node index (-1 to disable)" },
  { NULL }
};

/**
 * @brief Getter/setter definitions for the PyCaptureSettings type.
 * Defines the 'watermark_path' attribute, which requires custom logic for
 * memory management.
 */
static PyGetSetDef PyCaptureSettings_getsetters[] = {
  { "watermark_path", (getter)PyCaptureSettings_get_watermark_path, (setter)PyCaptureSettings_set_watermark_path, "watermark path", NULL },
  { NULL }
};

/**
 * @brief Type object definition for PyCaptureSettings.
 * This structure defines the PyCaptureSettings type for the Python interpreter.
 */
static PyTypeObject PyCaptureSettingsType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "pixelflux.screen_capture_module.CaptureSettings",
  sizeof(PyCaptureSettings),
  0,
  (destructor)PyCaptureSettings_dealloc,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  "Capture Settings",
  0, 0, 0, 0, 0, 0,
  0, // No methods for this type
  PyCaptureSettings_members,
  PyCaptureSettings_getsetters,
  0, 0, 0, 0, 0,
  0, // No __init__ for this type
  0,
  PyCaptureSettings_new,
};

/**
 * @brief Python type wrapper for the C++ StripeEncodeResult struct.
 * This object holds the result of an encoding operation from the C++ side.
 * Ownership of the underlying data buffer is transferred to this Python object,
 * and it is freed when the object is garbage collected.
 */
typedef struct {
  PyObject_HEAD
  StripeEncodeResult result;
} PyStripeEncodeResult;

/**
 * @brief Deallocator for PyStripeEncodeResult objects.
 * The move constructor for the C++ StripeEncodeResult transfers ownership of
 * the data buffer, so its destructor (called implicitly) handles freeing memory.
 * @param self The PyStripeEncodeResult object to be deallocated.
 */
static void PyStripeEncodeResult_dealloc(PyStripeEncodeResult* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/**
 * @brief Getter for the 'data' attribute of a PyStripeEncodeResult object.
 * @param self The PyStripeEncodeResult object.
 * @param closure Unused.
 * @return The encoded data as a Python bytes object, or Py_None if no data.
 */
static PyObject* PyStripeEncodeResult_get_data(PyStripeEncodeResult* self, void* closure) {
  if (!self->result.data || self->result.size <= 0) {
    Py_RETURN_NONE;
  }
  return PyBytes_FromStringAndSize((const char*)self->result.data, self->result.size);
}

/**
 * @brief Getter/setter definitions for the PyStripeEncodeResult type.
 * Exposes the encoded 'data' as a bytes object.
 */
static PyGetSetDef PyStripeEncodeResult_getset[] = {
  { "data", (getter)PyStripeEncodeResult_get_data, NULL, "encoded data", NULL },
  { NULL }
};

/**
 * @brief Member definitions for the PyStripeEncodeResult type.
 * Exposes metadata from the C++ result struct as read-only attributes.
 */
static PyMemberDef PyStripeEncodeResult_members[] = {
  { "type", T_INT, offsetof(PyStripeEncodeResult, result.type), READONLY, "stripe type" },
  { "stripe_y_start", T_INT, offsetof(PyStripeEncodeResult, result.stripe_y_start), READONLY, "stripe y start" },
  { "stripe_height", T_INT, offsetof(PyStripeEncodeResult, result.stripe_height), READONLY, "stripe height" },
  { "size", T_INT, offsetof(PyStripeEncodeResult, result.size), READONLY, "data size" },
  { "frame_id", T_INT, offsetof(PyStripeEncodeResult, result.frame_id), READONLY, "frame id" },
  { NULL }
};

/**
 * @brief Type object definition for PyStripeEncodeResult.
 */
static PyTypeObject PyStripeEncodeResultType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "pixelflux.screen_capture_module.StripeEncodeResult",
  sizeof(PyStripeEncodeResult),
  0,
  (destructor)PyStripeEncodeResult_dealloc,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  "Stripe Encode Result",
  0, 0, 0, 0, 0, 0,
  0, // No methods
  PyStripeEncodeResult_members,
  PyStripeEncodeResult_getset,
  0, 0, 0, 0, 0, 0, 0, 0
};

/**
 * @brief A wrapper object to hold a Python callable for type checking.
 * This allows Python code to pass a callback function that can be validated
 * and stored before being used by the capture module.
 */
typedef struct {
  PyObject_HEAD
  PyObject* callback;
} PyStripeCallback;

/**
 * @brief Initializer for PyStripeCallback objects.
 * Verifies that the object passed during construction is a callable.
 * @param self The PyStripeCallback object.
 * @param args A tuple containing the Python callable.
 * @param kwds Unused.
 * @return 0 on success, -1 on failure.
 */
static int PyStripeCallback_init(PyStripeCallback* self, PyObject* args, PyObject* kwds) {
  PyObject* callback = NULL;
  if (!PyArg_ParseTuple(args, "O:StripeCallback", &callback)) {
    return -1;
  }
  if (!PyCallable_Check(callback)) {
    PyErr_SetString(PyExc_TypeError, "parameter must be a callable");
    return -1;
  }
  Py_INCREF(callback);
  self->callback = callback;
  return 0;
}

/**
 * @brief Deallocator for PyStripeCallback objects.
 * Decrements the reference count of the stored Python callable.
 * @param self The PyStripeCallback object.
 */
static void PyStripeCallback_dealloc(PyStripeCallback* self) {
  Py_XDECREF(self->callback);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/**
 * @brief Type object definition for PyStripeCallback.
 */
static PyTypeObject PyStripeCallbackType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "pixelflux.screen_capture_module.StripeCallback",
  sizeof(PyStripeCallback),
  0,
  (destructor)PyStripeCallback_dealloc,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  "StripeCallback object",
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  (initproc)PyStripeCallback_init,
  PyType_GenericAlloc,
  PyType_GenericNew
};

/**
 * @brief The main Python type that wraps the C++ ScreenCaptureModule.
 * This class exposes the primary functionality of starting and stopping the
 * screen capture process to Python.
 */
typedef struct {
  PyObject_HEAD
  ScreenCaptureModule* module;
  PyObject* py_callback_obj;
} PyScreenCapture;

/**
 * @brief Deallocator for PyScreenCapture objects.
 * Ensures the capture thread is stopped and the C++ module is deleted,
 * then decrements the reference to the Python callback object.
 * @param self The PyScreenCapture object.
 */
static void PyScreenCapture_dealloc(PyScreenCapture* self) {
  if (self->module) {
    self->module->stop_capture();
    delete self->module;
    self->module = NULL;
  }
  Py_XDECREF(self->py_callback_obj);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/**
 * @brief Allocator for PyScreenCapture objects.
 * Creates a new PyScreenCapture object and instantiates the underlying C++
 * ScreenCaptureModule.
 * @param type The Python type object.
 * @param args Unused.
 * @param kwds Unused.
 * @return A new PyObject pointer to the allocated PyScreenCapture.
 */
static PyObject* PyScreenCapture_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  PyScreenCapture* self = (PyScreenCapture*)type->tp_alloc(type, 0);
  if (self) {
    self->module = new ScreenCaptureModule();
    self->py_callback_obj = NULL;
  }
  return (PyObject*)self;
}

/**
 * @brief C-style callback function that acts as a bridge from the C++ capture
 *        thread to the Python interpreter.
 *
 * This function is called by the C++ ScreenCaptureModule from a background
 * thread. It acquires the Python GIL, creates a Python wrapper object for the
 * encode result, calls the user-provided Python callback, and then cleans up.
 * It transfers ownership of the result's data buffer to the Python wrapper
 * via `std::move` to ensure proper memory management.
 *
 * @param result A pointer to the StripeEncodeResult from the C++ side.
 * @param user_data A void pointer to the PyScreenCapture object instance.
 */
static void py_stripe_callback_trampoline(StripeEncodeResult* result, void* user_data) {
  PyGILState_STATE gstate = PyGILState_Ensure();

  PyScreenCapture* py_capture_module = (PyScreenCapture*)user_data;
  if (py_capture_module && py_capture_module->py_callback_obj && result && result->data) {
    PyStripeEncodeResult* py_result_obj = (PyStripeEncodeResult*)PyStripeEncodeResultType.tp_alloc(&PyStripeEncodeResultType, 0);
    if (py_result_obj) {
      py_result_obj->result = std::move(*result);
      PyObject* ret = PyObject_CallFunction(py_capture_module->py_callback_obj, "OO", py_result_obj, py_capture_module);
      Py_XDECREF(ret);
      Py_DECREF(py_result_obj);
    }
  }
  else if (result && result->data) {
    delete[] result->data;
    result->data = nullptr;
  }

  PyGILState_Release(gstate);
}

/**
 * @brief Python method to start the capture process.
 * Takes CaptureSettings and a StripeCallback object, configures the underlying
 * C++ module, sets up the callback trampoline, and starts the capture thread.
 * @param self The PyScreenCapture object.
 * @param args A tuple containing a PyCaptureSettings and PyStripeCallback object.
 * @return Py_None on success, NULL on failure.
 */
static PyObject* PyScreenCapture_start_capture(PyScreenCapture* self, PyObject* args) {
  PyObject* py_settings;
  PyObject* py_callback_wrapper;
  if (!PyArg_ParseTuple(args, "OO", &py_settings, &py_callback_wrapper)) {
    return NULL;
  }

  if (!PyObject_TypeCheck(py_settings, &PyCaptureSettingsType)) {
    PyErr_SetString(PyExc_TypeError, "First argument must be a CaptureSettings object");
    return NULL;
  }
  if (!PyObject_TypeCheck(py_callback_wrapper, &PyStripeCallbackType)) {
    PyErr_SetString(PyExc_TypeError, "Second argument must be a StripeCallback object");
    return NULL;
  }

  PyObject* callable = ((PyStripeCallback*)py_callback_wrapper)->callback;
  Py_XDECREF(self->py_callback_obj);
  Py_INCREF(callable);
  self->py_callback_obj = callable;

  self->module->modify_settings(((PyCaptureSettings*)py_settings)->settings);
  self->module->stripe_callback = py_stripe_callback_trampoline;
  self->module->user_data = (void*)self;
  self->module->start_capture();

  Py_RETURN_NONE;
}

/**
 * @brief Python method to stop the capture process.
 * Signals the background thread to stop and waits for it to join.
 * @param self The PyScreenCapture object.
 * @param ignored Unused.
 * @return Py_None.
 */
static PyObject* PyScreenCapture_stop_capture(PyScreenCapture* self, PyObject* Py_UNUSED(ignored)) {
  if (self->module) {
    self->module->stop_capture();
  }
  Py_RETURN_NONE;
}

/**
 * @brief Method definitions for the PyScreenCapture type.
 */
static PyMethodDef PyScreenCapture_methods[] = {
  { "start_capture", (PyCFunction)PyScreenCapture_start_capture, METH_VARARGS, "Start capture with settings and callback" },
  { "stop_capture", (PyCFunction)PyScreenCapture_stop_capture, METH_NOARGS, "Stop capture" },
  { NULL }
};

/**
 * @brief Type object definition for PyScreenCapture.
 */
static PyTypeObject PyScreenCaptureType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "pixelflux.screen_capture_module.ScreenCapture",
  sizeof(PyScreenCapture),
  0,
  (destructor)PyScreenCapture_dealloc,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  Py_TPFLAGS_DEFAULT,
  "Screen Capture object",
  0, 0, 0, 0, 0, 0,
  PyScreenCapture_methods,
  0, 0, 0, 0, 0, 0, 0, 0, 0,
  PyScreenCapture_new
};

/**
 * @brief Module definition structure.
 */
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "pixelflux.screen_capture_module",
  "Screen capture module (C++ backend)",
  -1,
  NULL,
};

/**
 * @brief The initialization function for the Python module.
 * This is the entry point called by the Python interpreter when the module
 * is imported. It prepares and registers all the custom Python types.
 * @return A new PyObject pointer to the module, or NULL on failure.
 */
PyMODINIT_FUNC PyInit_screen_capture_module(void) {
  if (PyType_Ready(&PyCaptureSettingsType) < 0) return NULL;
  if (PyType_Ready(&PyStripeEncodeResultType) < 0) return NULL;
  if (PyType_Ready(&PyStripeCallbackType) < 0) return NULL;
  if (PyType_Ready(&PyScreenCaptureType) < 0) return NULL;

  PyObject* m = PyModule_Create(&moduledef);
  if (!m) {
    return NULL;
  }

  Py_INCREF(&PyCaptureSettingsType);
  PyModule_AddObject(m, "CaptureSettings", (PyObject*)&PyCaptureSettingsType);

  Py_INCREF(&PyStripeEncodeResultType);
  PyModule_AddObject(m, "StripeEncodeResult", (PyObject*)&PyStripeEncodeResultType);

  Py_INCREF(&PyStripeCallbackType);
  PyModule_AddObject(m, "StripeCallback", (PyObject*)&PyStripeCallbackType);

  Py_INCREF(&PyScreenCaptureType);
  PyModule_AddObject(m, "ScreenCapture", (PyObject*)&PyScreenCaptureType);

  return m;
}

#ifdef __cplusplus
}
#endif
