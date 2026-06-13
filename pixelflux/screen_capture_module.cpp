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
#include <cstdint>
#include <cstddef>
#include <cctype>
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
#include <cstdarg>
#include <dlfcn.h>
#include <link.h>
#include <elf.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
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
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

typedef enum CUresult_enum { CUDA_SUCCESS = 0 } CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef CUresult (*tcuInit)(unsigned int);
typedef CUresult (*tcuDeviceGet)(CUdevice*, int);
typedef CUresult (*tcuCtxCreate)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*tcuCtxDestroy)(CUcontext);
// Push/pop the encoder's CUDA context onto the context-less per-frame encode thread for
// the device-input NVENC calls (push/pop preferred over SetCurrent: saves/restores state).
typedef CUresult (*tcuCtxPushCurrent)(CUcontext);
typedef CUresult (*tcuCtxPopCurrent)(CUcontext*);
typedef CUresult (*tcuCtxSetCurrent)(CUcontext);

/**
 * @brief Holds function pointers for the CUDA driver API.
 * This struct is populated by `LoadCudaApi` using `dlsym` to allow for
 * dynamic loading of the CUDA library (`libcuda.so`), avoiding a hard
 * link-time dependency.
 */
struct CudaFunctions {
  tcuInit pfn_cuInit = nullptr;
  tcuDeviceGet pfn_cuDeviceGet = nullptr;
  tcuCtxCreate pfn_cuCtxCreate = nullptr;
  tcuCtxDestroy pfn_cuCtxDestroy = nullptr;
  // Optional: may be null on older drivers (callers must null-check); not part of the
  // LoadCudaApi success gate, so their absence never disables NVENC.
  tcuCtxPushCurrent pfn_cuCtxPushCurrent = nullptr;
  tcuCtxPopCurrent pfn_cuCtxPopCurrent = nullptr;
  tcuCtxSetCurrent pfn_cuCtxSetCurrent = nullptr;
};

CudaFunctions g_cuda_funcs;
static void* g_cuda_lib_handle = nullptr;
// Guards the one-time GOT ioctl interposer install (InstallNvencGpuFilter). libcuda is
// process-lifetime (never dlclose'd), so the patch never needs re-applying: set once, never reset.
static std::mutex g_nv_filter_mutex;
static bool g_nv_filter_installed = false;
// Issue: include vs omit the per-stripe header, and whether Python defers the free.
// Both toggles are per-ScreenCaptureModule members (emit_stripe_headers_/
// deferred_free_) rather than process-global statics, so multiple modules in one
// process don't clobber each other's wire format / free-ownership. The stripe
// free-function encoders receive emit_stripe_headers_ as an explicit argument.
// (emit_header false lets the WebRTC path skip pixelflux's header and its Python
// [10:] strip; deferred_free true hands buffer ownership to Python -- either way
// pixelflux's StripeEncodeResult has no data-freeing destructor, so Python always
// owns the free.)

/**
 * @brief Manages the state of an NVENC H.264 encoder session.
 * This struct encapsulates all the necessary handles, parameters, and buffer
 * pools for a single NVENC encoding pipeline. It maintains the CUDA context,
 * the encoder session, configuration details, and pools of input/output
 * buffers to facilitate asynchronous encoding.
 */
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
  bool cbr_mode = false;
  int initialized_bitrate_kbps = 0;
  NV_ENC_BUFFER_FORMAT initialized_buffer_format = NV_ENC_BUFFER_FORMAT_UNDEFINED;
  CUcontext cuda_context = nullptr;

  // Gated device-input (E2): cached registered CUDA device resource + the per-frame
  // device buffer the conversion site produced. All zero/null when the gate is off.
  NV_ENC_REGISTERED_PTR registered_resource = nullptr;
  unsigned long long registered_base = 0;
  int registered_w = 0, registered_h = 0, registered_pitch = 0;
  unsigned long long dev_input_base = 0;
  int dev_input_pitch = 0;

  NvencEncoderState() {
    nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    init_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
  }
};

static void* g_nvenc_lib_handle = nullptr;
typedef NVENCSTATUS(NVENCAPI* PFN_NvEncodeAPICreateInstance)(
  NV_ENCODE_API_FUNCTION_LIST*);

/**
 * @brief Manages the state of a VA-API H.264 encoder session using libavcodec.
 * This struct encapsulates all necessary libav objects for a VA-API hardware-
 * accelerated encoding pipeline. This includes the hardware device context,
 * hardware frame context for surface allocation, the codec context for the
 * h264_vaapi encoder, and reusable frame/packet objects.
 */
struct VaapiEncoderState {
    AVBufferRef *hw_device_ctx = nullptr;
    AVBufferRef *hw_frames_ctx = nullptr;
    AVCodecContext *codec_ctx = nullptr;
    AVFrame *sw_frame = nullptr;
    AVFrame *hw_frame = nullptr;
    AVPacket *packet = nullptr;
    bool initialized = false;
    int initialized_width = 0;
    int initialized_height = 0;
    int initialized_qp = -1;
    bool initialized_is_444 = false;
    unsigned int frame_count = 0;
    bool initialized_cbr = false;
    int initialized_bitrate_kbps = 0;
};


/**
 * @brief Custom X11 error handler specifically for the XShmAttach call.
 * This function is temporarily installed as the X11 error handler. It catches
 * any error, sets the g_shm_attach_failed flag to true, and returns 0 to
 * signal that the error has been "handled," preventing program termination.
 */
static bool g_shm_attach_failed = false;
static int shm_attach_error_handler(Display* dpy, XErrorEvent* ev) {
    g_shm_attach_failed = true;
    return 0;
}

/**
 * @brief Manages a pool of H.264 encoders and associated picture buffers.
 * This struct provides thread-safe storage and management for x264 encoder
 * instances, input pictures, and their initialization states. This allows
 * different threads to use separate encoder instances, particularly for
 * encoding different stripes of a video frame concurrently.
 */
struct MinimalEncoderStore {
  std::vector<x264_t*> encoders;
  std::vector<bool> initialized_flags;
  std::vector<int> initialized_widths;
  std::vector<int> initialized_heights;
  std::vector<int> initialized_crfs;
  std::vector<int> initialized_csps;
  std::vector<int> initialized_colorspaces;
  std::vector<bool> initialized_full_range_flags;
  std::vector<bool> force_idr_flags;
  std::vector<bool> initialized_cbr_flags;
  std::vector<int> initialized_bitrates;
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
      initialized_flags.resize(new_size, false);
      initialized_widths.resize(new_size, 0);
      initialized_heights.resize(new_size, 0);
      initialized_crfs.resize(new_size, -1);
      initialized_csps.resize(new_size, X264_CSP_NONE);
      initialized_colorspaces.resize(new_size, 0);
      initialized_full_range_flags.resize(new_size, false);
      force_idr_flags.resize(new_size, false);
      initialized_cbr_flags.resize(new_size, false);
      initialized_bitrates.resize(new_size, 0);
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
    }
    encoders.clear();
    initialized_flags.clear();
    initialized_widths.clear();
    initialized_heights.clear();
    initialized_crfs.clear();
    initialized_csps.clear();
    initialized_colorspaces.clear();
    initialized_full_range_flags.clear();
    force_idr_flags.clear();
    initialized_cbr_flags.clear();
    initialized_bitrates.clear();
  }

  /**
   * @brief Destructor for MinimalEncoderStore.
   * Calls reset() to ensure all resources are released upon destruction.
   */
  ~MinimalEncoderStore() {
    reset();
  }
};

/**
 * @brief Enumerates the possible output modes for encoding.
 */
enum class OutputMode : int {
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
enum class WatermarkLocation : int {
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
  double scale;
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
  int h264_paintover_crf;
  int h264_paintover_burst_frames;
  bool h264_fullcolor;
  bool h264_fullframe;
  bool h264_streaming_mode;
  bool capture_cursor;
  const char* watermark_path;
  WatermarkLocation watermark_location_enum;
  int vaapi_render_node_index;
  bool use_cpu;
  bool debug_logging;
  bool h264_cbr_mode;
  int h264_bitrate_kbps;
  int h264_vbv_buffer_size_kb;
  bool auto_adjust_screen_capture_size;
  bool omit_stripe_headers;   // append-only; false(default)=prepend per-stripe header (WS), true=omit (WebRTC)
  bool deferred_free;         // append-only; true=transfer buffer ownership to Python (zero-copy), false(default)=free here
  const char* vaapi_render_node_path;  // append-only; explicit /dev/dri/renderD* path (authoritative); null/empty = use index/AUTO_GPU

  /**
   * @brief Default constructor for CaptureSettings.
   * Initializes settings with common default values.
   */
  CaptureSettings()
    : capture_width(1920),
      capture_height(1080),
      scale(1.0),
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
      h264_paintover_crf(18),
      h264_paintover_burst_frames(5),
      h264_fullcolor(false),
      h264_fullframe(false),
      h264_streaming_mode(false),
      capture_cursor(false),
      watermark_path(nullptr),
      watermark_location_enum(WatermarkLocation::NONE),
      vaapi_render_node_index(-1),
      use_cpu(false),
      debug_logging(false),
      h264_cbr_mode(false),
      h264_bitrate_kbps(4000),
      h264_vbv_buffer_size_kb(0),
      auto_adjust_screen_capture_size(false),
      omit_stripe_headers(false),
      deferred_free(false),
      vaapi_render_node_path(nullptr) {}

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
   * @param h264_po_crf H.264 paint-over CRF.
   * @param h264_po_burst H.264 paint-over burst frames.
   * @param h264_fc H.264 full color (I444) flag.
   * @param h264_ff H.264 full frame encoding flag.
   * @param h264_sm H.264 streaming mode flag.
   * @param capture_cursor Capture cursor flag.
   * @param wm_path Watermark image file path.
   * @param wm_loc Watermark location enum.
   * @param vaapi_idx VA-API render node index.
   * @param use_cpu_flag Force CPU encoding flag.
   * @param debug_log Enable debug logging flag.
   * @param cbr_mode H.264 CBR mode flag.
   * @param bitrate_kbps H.264 CBR bitrate in kbps.
   * @param vbv_buffer_size_kb H.264 VBV buffer size in kb.
   * @param adjust_size Auto-adjust screen capture size flag.
   */
  CaptureSettings(int cw, int ch, int cx, int cy, double fps, int jq,
                  int pojq, bool upoq, int potf, int dbt, int dbd,
                  OutputMode om = OutputMode::JPEG, int crf = 25, int h264_po_crf = 18, int h264_po_burst = 5,
                  bool h264_fc = false, bool h264_ff = false, bool h264_sm = false,
                  bool capture_cursor = false,
                  const char* wm_path = nullptr,
                  WatermarkLocation wm_loc = WatermarkLocation::NONE,
                  int vaapi_idx = -1, bool use_cpu_flag = false, bool debug_log = false,
                  bool cbr_mode = false, int bitrate_kbps = 4000, int vbv_buffer_size_kb = 0, bool adjust_size = false)
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
      h264_paintover_crf(h264_po_crf),
      h264_paintover_burst_frames(h264_po_burst),
      h264_fullcolor(h264_fc),
      h264_fullframe(h264_ff),
      h264_streaming_mode(h264_sm),
      capture_cursor(capture_cursor),
      watermark_path(wm_path),
      watermark_location_enum(wm_loc),
      vaapi_render_node_index(vaapi_idx),
      use_cpu(use_cpu_flag),
      debug_logging(debug_log),
      h264_cbr_mode(cbr_mode),
      h264_bitrate_kbps(bitrate_kbps),
      h264_vbv_buffer_size_kb(vbv_buffer_size_kb),
      auto_adjust_screen_capture_size(adjust_size),
      omit_stripe_headers(false),
      deferred_free(false),
      vaapi_render_node_path(nullptr) {}
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
    // Prefer the v2 ABI of cuCtxCreate/cuCtxDestroy (same signature). A v1 context is NOT
    // valid for the v2 CUDA mem ops NVENC uses for H.264 4:4:4 (-> INVALID_CONTEXT / encode
    // error 20); _v2 fixes 4:4:4. Fall back to v1 so older drivers still load.
    g_cuda_funcs.pfn_cuCtxCreate = (tcuCtxCreate)dlsym(g_cuda_lib_handle, "cuCtxCreate_v2");
    if (!g_cuda_funcs.pfn_cuCtxCreate)
        g_cuda_funcs.pfn_cuCtxCreate = (tcuCtxCreate)dlsym(g_cuda_lib_handle, "cuCtxCreate");
    g_cuda_funcs.pfn_cuCtxDestroy = (tcuCtxDestroy)dlsym(g_cuda_lib_handle, "cuCtxDestroy_v2");
    if (!g_cuda_funcs.pfn_cuCtxDestroy)
        g_cuda_funcs.pfn_cuCtxDestroy = (tcuCtxDestroy)dlsym(g_cuda_lib_handle, "cuCtxDestroy");
    // Optional context push/pop/set for the device-input path; best-effort, NOT in the gate below.
    g_cuda_funcs.pfn_cuCtxPushCurrent = (tcuCtxPushCurrent)dlsym(g_cuda_lib_handle, "cuCtxPushCurrent_v2");
    if (!g_cuda_funcs.pfn_cuCtxPushCurrent)
        g_cuda_funcs.pfn_cuCtxPushCurrent = (tcuCtxPushCurrent)dlsym(g_cuda_lib_handle, "cuCtxPushCurrent");
    g_cuda_funcs.pfn_cuCtxPopCurrent = (tcuCtxPopCurrent)dlsym(g_cuda_lib_handle, "cuCtxPopCurrent_v2");
    if (!g_cuda_funcs.pfn_cuCtxPopCurrent)
        g_cuda_funcs.pfn_cuCtxPopCurrent = (tcuCtxPopCurrent)dlsym(g_cuda_lib_handle, "cuCtxPopCurrent");
    g_cuda_funcs.pfn_cuCtxSetCurrent = (tcuCtxSetCurrent)dlsym(g_cuda_lib_handle, "cuCtxSetCurrent");

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
 * @brief No-op: libcuda / g_cuda_funcs are deliberately process-lifetime.
 *
 * libcuda and g_cuda_funcs are process-global; with multiple ScreenCapture instances per
 * process, dlclose'ing/zeroing them from one instance's stop would be a use-after-free while
 * another is mid-encode. So they are never freed here (mirrors g_nvrtc_handle); per-instance
 * GPU resources are still released by reset_nvenc_encoder(). Kept callable as an explicit hook.
 */
void UnloadCudaApi() {
    // Intentionally a no-op; see above.
}

/**
 * @brief Dynamically loads the NVIDIA Encoder (NVENC) library and initializes the API function list.
 *
 * This function checks if the API is already loaded. If not, it attempts to load
 * `libnvidia-encode.so.1` or `libnvidia-encode.so` using `dlopen`. It then uses
 * `dlsym` to get the `NvEncodeAPICreateInstance` function and calls it to populate
 * the provided function list struct.
 *
 * @return true if the library was loaded and the function list was successfully
 *         populated, false otherwise.
 */
bool LoadNvencApi(NV_ENCODE_API_FUNCTION_LIST& nvenc_funcs) {
  if (nvenc_funcs.nvEncOpenEncodeSessionEx != nullptr) {
    return true;
  }
  if (!g_nvenc_lib_handle) {
      const char* lib_names[] = {"libnvidia-encode.so.1", "libnvidia-encode.so"};
      for (const char* name : lib_names) {
        g_nvenc_lib_handle = dlopen(name, RTLD_LAZY | RTLD_GLOBAL);
        if (g_nvenc_lib_handle) {
          break;
        }
      }
  }

  if (!g_nvenc_lib_handle) {
    return false;
  }

  memset(&nvenc_funcs, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
  nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;

  PFN_NvEncodeAPICreateInstance NvEncodeAPICreateInstance_func_ptr =
    (PFN_NvEncodeAPICreateInstance)dlsym(g_nvenc_lib_handle, "NvEncodeAPICreateInstance");

  if (!NvEncodeAPICreateInstance_func_ptr) {
    return false;
  }

  NVENCSTATUS status = NvEncodeAPICreateInstance_func_ptr(&nvenc_funcs);
  if (status != NV_ENC_SUCCESS) {
    memset(&nvenc_funcs, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    return false;
  }
  if (!nvenc_funcs.nvEncOpenEncodeSessionEx) {
    memset(&nvenc_funcs, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    return false;
  }
  return true;
}

// --- Multi-GPU NVENC fix (issue 8): GET_ATTACHED_IDS filter ----------------
// On driver 570-595, libnvidia-encode enumerates every host GPU via the RM
// GET_ATTACHED_IDS ioctl and peer-inits each; GPUs whose /dev/nvidiaX is absent
// from the container make nvEncOpenEncodeSessionEx fail with UNSUPPORTED_DEVICE.
// We GOT-patch ioctl in the NVIDIA libraries (kept inside PixelFlux, no separate
// LD_PRELOAD object) and drop the unreachable GPUs from the response.
namespace {
constexpr unsigned long kNvEscRmControl = 0x2A;   // ioctl NR for NV_ESC_RM_CONTROL
constexpr uint32_t kGpuGetAttachedIds = 0x0201;   // NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS
constexpr int kMaxAttachedGpus = 32;
constexpr uint32_t kInvalidGpuId = 0xFFFFFFFFu;

struct NvRmControlParams {                         // NVOS54_PARAMETERS (32 bytes)
  uint32_t hClient, hObject, cmd, flags;
  uint64_t params;
  uint32_t paramsSize, status;
};

bool nv_node_present(unsigned minor) {
  char path[64];
  snprintf(path, sizeof(path), "/dev/nvidia%u", minor);
  return access(path, F_OK) == 0;
}

// Resolve a gpuId to its /dev/nvidia minor via /proc (the PCI bus is encoded in
// gpuId >> 8). Returns -1 when no match is found.
int nv_gpuid_to_minor(uint32_t gpu_id) {
  unsigned want_full = gpu_id >> 8;   // encodes (domain << 8) | bus
  DIR* dir = opendir("/proc/driver/nvidia/gpus");
  if (!dir) return -1;
  int minor = -1;
  struct dirent* ent;
  while ((ent = readdir(dir)) != nullptr) {
    unsigned dom, bus, slot, fn;
    if (sscanf(ent->d_name, "%x:%x:%x.%x", &dom, &bus, &slot, &fn) != 4) continue;
    if (((dom << 8) | bus) != want_full) continue;
    char info[512];
    snprintf(info, sizeof(info), "/proc/driver/nvidia/gpus/%s/information", ent->d_name);
    if (FILE* f = fopen(info, "r")) {
      char line[256];
      while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "Device Minor: %d", &minor) == 1) break;
      }
      fclose(f);
    }
    break;
  }
  closedir(dir);
  return minor;
}

// ioctl wrapper installed into the NVIDIA libraries' GOT. Our own object's GOT
// is left untouched, so the inner ioctl() call reaches libc normally.
int nv_filtered_ioctl(int fd, unsigned long req, ...) {
  va_list ap;
  va_start(ap, req);
  void* arg = va_arg(ap, void*);
  va_end(ap);
  int rc = ioctl(fd, req, arg);
  if (rc != 0 || _IOC_NR(req) != kNvEscRmControl || !arg) return rc;
  NvRmControlParams* ctrl = static_cast<NvRmControlParams*>(arg);
  if (ctrl->cmd != kGpuGetAttachedIds || ctrl->status != 0 || !ctrl->params) return rc;
  // Guard the params layout before treating it as the id array: require the EXACT size
  // (NV0000_CTRL_GPU_GET_ATTACHED_IDS_PARAMS = kMaxAttachedGpus ids) so a lying paramsSize
  // can't make us rewrite memory under a different layout. A future struct change must update this.
  if (ctrl->paramsSize != sizeof(uint32_t) * kMaxAttachedGpus) return rc;
  uint32_t* ids = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(ctrl->params));
  uint32_t kept[kMaxAttachedGpus];
  int total = 0, nkept = 0;
  for (int i = 0; i < kMaxAttachedGpus && ids[i] != kInvalidGpuId; i++) {
    total++;
    int minor = nv_gpuid_to_minor(ids[i]);
    if (minor >= 0 && nv_node_present(static_cast<unsigned>(minor))) kept[nkept++] = ids[i];
  }
  if (nkept > 0 && nkept < total) {
    for (int i = 0; i < nkept; i++) ids[i] = kept[i];
    for (int i = nkept; i < kMaxAttachedGpus; i++) ids[i] = kInvalidGpuId;
  }
  return rc;
}

// Current VM protection of the page holding addr, from /proc/self/maps (-1 if
// unknown). The NVIDIA libs ship partial RELRO + lazy binding, so the PLT GOT
// stays writable for the linker's lazy resolver; we restore that exact prot
// after patching. (Forcing it read-only crashed the next lazy resolve.)
int nv_page_prot(uintptr_t addr) {
  FILE* f = fopen("/proc/self/maps", "r");
  if (!f) return -1;
  char line[512];
  int prot = -1;
  while (fgets(line, sizeof(line), f)) {
    uintptr_t lo, hi; char perm[5] = {0};
    if (sscanf(line, "%lx-%lx %4s", &lo, &hi, perm) != 3) continue;
    if (addr >= lo && addr < hi) {
      prot = ((perm[0] == 'r') ? PROT_READ : 0) |
             ((perm[1] == 'w') ? PROT_WRITE : 0) |
             ((perm[2] == 'x') ? PROT_EXEC : 0);
      break;
    }
  }
  fclose(f);
  return prot;
}

void nv_patch_ioctl_got(uintptr_t base, const ElfW(Dyn)* dyn) {
  const ElfW(Sym)* symtab = nullptr;
  const char* strtab = nullptr;
  ElfW(Rela)* jmprel = nullptr;
  size_t pltrelsz = 0;
  for (const ElfW(Dyn)* d = dyn; d->d_tag != DT_NULL; d++) {
    if (d->d_tag == DT_SYMTAB) symtab = reinterpret_cast<const ElfW(Sym)*>(d->d_un.d_ptr);
    else if (d->d_tag == DT_STRTAB) strtab = reinterpret_cast<const char*>(d->d_un.d_ptr);
    else if (d->d_tag == DT_JMPREL) jmprel = reinterpret_cast<ElfW(Rela)*>(d->d_un.d_ptr);
    else if (d->d_tag == DT_PLTRELSZ) pltrelsz = d->d_un.d_val;
  }
  if (!symtab || !strtab || !jmprel || !pltrelsz) return;
  long page = sysconf(_SC_PAGESIZE);
  for (size_t i = 0; i < pltrelsz / sizeof(ElfW(Rela)); i++) {
    ElfW(Rela)* r = &jmprel[i];
    if (strcmp(strtab + symtab[ELF64_R_SYM(r->r_info)].st_name, "ioctl") != 0) continue;
    void** slot = reinterpret_cast<void**>(base + r->r_offset);
    void* pg = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(slot) & ~(page - 1));
    // Restore the slot's original protection (writable under partial RELRO), not a
    // hardcoded read-only: these libs lazily bind through this page, so leaving it
    // read-only faults the next resolve. Default to writable if maps is unreadable.
    int orig = nv_page_prot(reinterpret_cast<uintptr_t>(slot));
    if (orig < 0) orig = PROT_READ | PROT_WRITE;
    if (mprotect(pg, page, PROT_READ | PROT_WRITE) == 0) {
      *slot = reinterpret_cast<void*>(nv_filtered_ioctl);
      mprotect(pg, page, orig);
    }
  }
}

int nv_patch_phdr_cb(struct dl_phdr_info* info, size_t, void*) {
  if (!info->dlpi_name || !*info->dlpi_name) return 0;
  // GET_ATTACHED_IDS is issued by libcuda and libnvcuvid (libnvidia-encode has no
  // ioctl of its own and calls through libnvcuvid), so all three must be patched.
  if (!strstr(info->dlpi_name, "libnvidia") && !strstr(info->dlpi_name, "libcuda") &&
      !strstr(info->dlpi_name, "libnvcuvid")) return 0;
  for (int i = 0; i < info->dlpi_phnum; i++) {
    if (info->dlpi_phdr[i].p_type == PT_DYNAMIC) {
      nv_patch_ioctl_got(info->dlpi_addr,
                         reinterpret_cast<const ElfW(Dyn)*>(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr));
    }
  }
  return 0;
}

// True when at least one host GPU is hidden from the container (the only case
// the peer-init bug can trigger), so the filter is a no-op everywhere else.
bool nv_has_hidden_gpus() {
  int host = 0, visible = 0;
  if (DIR* d = opendir("/proc/driver/nvidia/gpus")) {
    for (struct dirent* e; (e = readdir(d)) != nullptr; ) {
      if (e->d_name[0] != '.') host++;
    }
    closedir(d);
  }
  for (int m = 0; m < kMaxAttachedGpus; m++) {
    if (nv_node_present(static_cast<unsigned>(m))) visible++;
  }
  return host > visible && visible > 0;
}

void InstallNvencGpuFilter() {
  std::lock_guard<std::mutex> lock(g_nv_filter_mutex);
  if (g_nv_filter_installed) return;
  if (nv_has_hidden_gpus()) dl_iterate_phdr(nv_patch_phdr_cb, nullptr);
  g_nv_filter_installed = true;
}
}  // namespace

// --- Issue 10: optional CUDA (NVRTC) ARGB->NV12 color conversion -----------
// When libnvrtc is present, offload NVENC-mode color conversion from CPU
// (libyuv) to the GPU: upload ARGB, run a runtime-compiled BT.601-limited
// kernel (matching libyuv exactly), download NV12. Returns false on absence or
// any error so the caller falls back to libyuv -> it can never regress.
// Verified: kernel NV12 == libyuv (Y exact, UV +/-1) and NVENC-encode->decode
// PSNR ~55 dB. Disable with env PIXELFLUX_NO_CUDA_CONVERT=1.
namespace cuda_convert {
typedef unsigned long long CUdptr;
typedef struct CUmod_st* CUmod;
typedef struct CUfunc_st* CUfunc;
typedef int (*tAlloc)(CUdptr*, size_t);
typedef int (*tFree)(CUdptr);
typedef int (*tH2D)(CUdptr, const void*, size_t);
typedef int (*tD2H)(void*, CUdptr, size_t);
typedef int (*tLaunch)(CUfunc, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**);
typedef int (*tModLoad)(CUmod*, const void*, unsigned, int*, void**);
typedef int (*tModFn)(CUfunc*, CUmod, const char*);
typedef int (*tSetCtx)(CUcontext);
typedef int (*tSync)();
typedef int (*tNvCreate)(void**, const char*, const char*, int, const char**, const char**);
typedef int (*tNvCompile)(void*, int, const char**);
typedef int (*tNvPtxSz)(void*, size_t*);
typedef int (*tNvPtx)(void*, char*);
typedef int (*tNvDestroy)(void**);
// Device-compute-capability query so the kernel is compiled for the ACTUAL device's
// virtual arch (PTX runs only on its own virtual arch and newer -- a fixed compute_52
// PTX cannot JIT on a Kepler sm_35). Resolved best-effort via dlsym; null -> fallback list.
typedef int (*tDevAttr)(int*, int, CUdevice);     // cuDeviceGetAttribute
typedef int (*tCtxGetDev)(CUdevice*);             // cuCtxGetDevice
// CUdevice attribute ids (stable CUDA-driver enum values).
static const int CU_DEV_ATTR_CC_MAJOR = 75;       // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
static const int CU_DEV_ATTR_CC_MINOR = 76;       // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

static std::mutex g_mutex;
static bool g_tried = false, g_ready = false;
static CUcontext g_init_ctx = nullptr;   // context the cache below is bound to
static CUfunc g_fn = nullptr;
static CUdptr g_d_argb = 0, g_d_y = 0, g_d_uv = 0;
static size_t g_cap_argb = 0, g_cap_y = 0, g_cap_uv = 0;
static CUdptr g_d_nv12 = 0;       // contiguous pitched NV12 for gated device-input NVENC (E2)
static size_t g_cap_nv12 = 0;
static tAlloc g_alloc; static tFree g_free; static tH2D g_h2d; static tD2H g_d2h;
static tLaunch g_launch; static tSetCtx g_setctx; static tSync g_sync;
// libnvrtc compilation is context-independent, so cache the compiled PTX once for
// the whole process. Re-initializing against a new context (multi-display NVENC)
// then only re-runs cuModuleLoadDataEx instead of a full NVRTC compile every frame.
static void* g_nvrtc_handle = nullptr;    // persistent dlopen handle (never dlclose'd)
static std::vector<char> g_ptx;           // compiled PTX, reused across contexts
// Virtual arch the cached g_ptx was compiled for (e.g. "compute_89"). Part of the cache
// key: a context switch to a device with a different compute capability recompiles instead
// of JIT-ing wrong-arch PTX. Empty until the first successful compile.
static std::string g_ptx_arch;

static const char* KERNEL = R"(extern "C" __global__ void argb_to_nv12(
  const unsigned char* a,int aw,unsigned char* yp,int yw,unsigned char* uv,int uvw,int W,int H){
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y; if(x>=W||y>=H)return;
  const unsigned char* p=a+y*aw+x*4; int B=p[0],G=p[1],R=p[2];
  yp[y*yw+x]=(unsigned char)((66*R+129*G+25*B+0x1080)>>8);
  if((x&1)==0&&(y&1)==0){const unsigned char* q=p+aw;
    #define AV(s,t) (((s)+(t)+1)>>1)
    int ab=AV(AV(p[0],q[0]),AV(p[4],q[4])),ag=AV(AV(p[1],q[1]),AV(p[5],q[5])),ar=AV(AV(p[2],q[2]),AV(p[6],q[6]));
    unsigned char* o=uv+(y/2)*uvw+(x/2)*2;
    o[0]=(unsigned char)((112*ab-74*ag-38*ar+0x8080)>>8);
    o[1]=(unsigned char)((112*ar-94*ag-18*ab+0x8080)>>8);}})";

// Resolve the active device's compute capability via the CUDA driver API and return its
// virtual arch string (e.g. Kepler->"compute_35", Maxwell->"compute_52", Ada L40S->
// "compute_89"). Best-effort: empty string on any failure so the caller uses its fallback
// list. ctx must be current (init_locked does g_setctx(ctx) before calling). Uses
// cuCtxGetDevice to get the device of the in-use context, then cuDeviceGetAttribute for the
// major/minor cc. This is correct across CUDA versions because the NVRTC paired with a given
// driver always supports that driver's own devices.
static std::string device_compute_arch() {
  if (!g_cuda_lib_handle) return std::string();
  auto getDev = (tCtxGetDev)dlsym(g_cuda_lib_handle, "cuCtxGetDevice");
  auto devAttr = (tDevAttr)dlsym(g_cuda_lib_handle, "cuDeviceGetAttribute");
  if (!getDev || !devAttr) return std::string();
  CUdevice dev = 0;
  if (getDev(&dev) != 0) {
    // Fall back to device 0 (the device NVENC's cuCtxCreate used) if no current ctx.
    if (g_cuda_funcs.pfn_cuDeviceGet && g_cuda_funcs.pfn_cuDeviceGet(&dev, 0) != CUDA_SUCCESS)
      return std::string();
  }
  int major = 0, minor = 0;
  if (devAttr(&major, CU_DEV_ATTR_CC_MAJOR, dev) != 0) return std::string();
  if (devAttr(&minor, CU_DEV_ATTR_CC_MINOR, dev) != 0) return std::string();
  if (major <= 0) return std::string();
  return "compute_" + std::to_string(major) + std::to_string(minor);
}

// Compile the kernel to PTX for the given virtual arch (e.g. "compute_35") and cache it in
// g_ptx, keyed by arch in g_ptx_arch. Reuses the cache when the requested arch is unchanged
// (PTX is context-independent), but recompiles when a context switch targets a different-cc
// device so we never JIT wrong-arch PTX. Returns true with g_ptx/g_ptx_arch populated.
static bool compile_ptx_for_arch(const std::string& arch) {
  if (!g_ptx.empty() && g_ptx_arch == arch) return true;
  if (!g_nvrtc_handle) {
    g_nvrtc_handle=dlopen("libnvrtc.so.12",RTLD_NOW|RTLD_GLOBAL);
    if(!g_nvrtc_handle) g_nvrtc_handle=dlopen("libnvrtc.so",RTLD_NOW|RTLD_GLOBAL);
  }
  void* rt=g_nvrtc_handle;
  if(!rt) return false;
  auto nvCreate=(tNvCreate)dlsym(rt,"nvrtcCreateProgram");
  auto nvCompile=(tNvCompile)dlsym(rt,"nvrtcCompileProgram");
  auto nvPtxSz=(tNvPtxSz)dlsym(rt,"nvrtcGetPTXSize");
  auto nvPtx=(tNvPtx)dlsym(rt,"nvrtcGetPTX");
  auto nvDestroy=(tNvDestroy)dlsym(rt,"nvrtcDestroyProgram");
  if(!nvCreate||!nvCompile||!nvPtxSz||!nvPtx) return false;
  void* prog=nullptr; if(nvCreate(&prog,KERNEL,"cc.cu",0,nullptr,nullptr)!=0) return false;
  // Destroy the NVRTC program on every exit path below, not just success.
  bool ok=false;
  std::string opt = "--gpu-architecture=" + arch;
  const char* opts[]={opt.c_str()};
  if(nvCompile(prog,1,opts)==0) {
    size_t psz=0;
    if(nvPtxSz(prog,&psz)==0 && psz!=0) {
      std::vector<char> ptx(psz);
      if(nvPtx(prog,ptx.data())==0) { g_ptx=std::move(ptx); g_ptx_arch=arch; ok=true; }
    }
  }
  if(nvDestroy) nvDestroy(&prog);
  return ok && !g_ptx.empty();
}

// Compile the kernel for the device in use, choosing a device-compute-capability-aware
// virtual arch so the resulting PTX JIT-compiles on THIS exact GPU (Kepler sm_35 cannot run
// compute_52 PTX). Order: the queried device cc first, then a small descending fallback
// (compute_89 -> compute_52 -> compute_35) so a missing/rejected arch (e.g. a CUDA 12/13
// NVRTC rejecting compute_35, or an unusually new device) still finds a usable target.
// Caller (init_locked) has already made ctx current, so device_compute_arch() sees it.
static bool compile_ptx_once() {
  std::vector<std::string> archs;
  std::string dev_arch = device_compute_arch();
  if (!dev_arch.empty()) archs.push_back(dev_arch);
  // Descending fallbacks (skip a duplicate of the queried arch). compute_35 last so a
  // Kepler target is always attempted; modern archs first so a modern device still gets
  // an arch NVRTC accepts even if the cc query failed.
  for (const char* fb : {"compute_89", "compute_52", "compute_35"}) {
    if (dev_arch != fb) archs.push_back(fb);
  }
  for (const std::string& a : archs) {
    if (compile_ptx_for_arch(a)) return true;
  }
  return false;
}

static bool init_locked(CUcontext ctx) {
  if (!g_cuda_lib_handle) return false;
  g_alloc=(tAlloc)dlsym(g_cuda_lib_handle,"cuMemAlloc_v2");
  g_free=(tFree)dlsym(g_cuda_lib_handle,"cuMemFree_v2");
  g_h2d=(tH2D)dlsym(g_cuda_lib_handle,"cuMemcpyHtoD_v2");
  g_d2h=(tD2H)dlsym(g_cuda_lib_handle,"cuMemcpyDtoH_v2");
  g_launch=(tLaunch)dlsym(g_cuda_lib_handle,"cuLaunchKernel");
  g_setctx=(tSetCtx)dlsym(g_cuda_lib_handle,"cuCtxSetCurrent");
  g_sync=(tSync)dlsym(g_cuda_lib_handle,"cuCtxSynchronize");
  auto modLoad=(tModLoad)dlsym(g_cuda_lib_handle,"cuModuleLoadDataEx");
  auto modFn=(tModFn)dlsym(g_cuda_lib_handle,"cuModuleGetFunction");
  if(!g_alloc||!g_free||!g_h2d||!g_d2h||!g_launch||!g_setctx||!g_sync||!modLoad||!modFn) return false;
  // Make ctx current BEFORE compiling so the device-cc query (cuCtxGetDevice in
  // compile_ptx_once) targets THIS context's device and picks its exact virtual arch.
  g_setctx(ctx);
  if(!compile_ptx_once()) return false;
  // The CUmod is never cuModuleUnload'd: it is context-scoped and dies with its CUcontext
  // (teardown invalidates before cuCtxDestroy), so there is no per-module process leak.
  CUmod mod=nullptr; if(modLoad(&mod,g_ptx.data(),0,nullptr,nullptr)!=0) return false;
  if(modFn(&g_fn,mod,"argb_to_nv12")!=0||!g_fn) return false;
  return true;
}

static bool ensure_buf(CUdptr& d, size_t& cap, size_t need) {
  if (cap >= need) return true;
  if (d) { g_free(d); d = 0; cap = 0; }
  if (g_alloc(&d, need) != 0) { d = 0; return false; }
  cap = need; return true;
}

// NV12 -> y/uv on success; false means the caller should use libyuv.
bool argb_to_nv12(CUcontext ctx, const uint8_t* argb, int argb_stride,
                  uint8_t* y, int y_stride, uint8_t* uv, int uv_stride, int w, int h) {
  if (!ctx || w <= 0 || h <= 0 || (w & 1) || (h & 1)) return false;
  if (y_stride != w || uv_stride != w) return false;   // padded dst -> libyuv
  static const bool disabled = (std::getenv("PIXELFLUX_NO_CUDA_CONVERT") != nullptr);
  if (disabled) return false;
  std::lock_guard<std::mutex> lk(g_mutex);
  if (g_tried && ctx != g_init_ctx) {
    // NVENC was re-initialized (resolution/format change, error recovery): the
    // old context was destroyed along with its module and device buffers, so
    // drop the stale handles (do NOT free them - cuCtxDestroy already did) and
    // re-init against the new context.
    g_tried = false; g_ready = false; g_fn = nullptr;
    g_d_argb = g_d_y = g_d_uv = 0; g_cap_argb = g_cap_y = g_cap_uv = 0;
  }
  if (!g_tried) { g_tried = true; g_init_ctx = ctx; g_ready = init_locked(ctx); }
  if (!g_ready) return false;
  if (g_setctx(ctx) != 0) return false;
  size_t argb_sz = (size_t)argb_stride * h, y_sz = (size_t)w * h, uv_sz = (size_t)w * (h / 2);
  if (!ensure_buf(g_d_argb, g_cap_argb, argb_sz) || !ensure_buf(g_d_y, g_cap_y, y_sz) || !ensure_buf(g_d_uv, g_cap_uv, uv_sz)) return false;
  if (g_h2d(g_d_argb, argb, argb_sz) != 0) return false;
  int aw = argb_stride, yw = w, uvw = w, ww = w, hh = h;
  void* args[] = {&g_d_argb, &aw, &g_d_y, &yw, &g_d_uv, &uvw, &ww, &hh};
  if (g_launch(g_fn, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, nullptr, args, nullptr) != 0) return false;
  if (g_sync() != 0) return false;
  if (g_d2h(y, g_d_y, y_sz) != 0) return false;
  if (g_d2h(uv, g_d_uv, uv_sz) != 0) return false;
  return true;
}

// Gated (PIXELFLUX_NVENC_DEVICE_INPUT) device-input variant. Same conversion, but writes
// a CONTIGUOUS 256-pitch NV12 into g_d_nv12 (Y at base, UV at base+pitch*h) so NVENC can
// register it directly (nvEncRegisterResource), avoiding the host re-upload. The host
// y/uv planes are still filled (per-row, since the device buffer is pitched) because the
// change-detection hash reads them. Returns the device base + pitch. Default ON
// (disable with PIXELFLUX_NVENC_DEVICE_INPUT=0).
bool nvenc_device_input_enabled() {
  // Default ON; disable with PIXELFLUX_NVENC_DEVICE_INPUT=0 (or false/off/no).
  static const bool on = []() {
    const char* v = std::getenv("PIXELFLUX_NVENC_DEVICE_INPUT");
    if (!v) return true;
    std::string s(v);
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return !(s == "0" || s == "false" || s == "off" || s == "no");
  }();
  return on;
}
bool argb_to_nv12_device(CUcontext ctx, const uint8_t* argb, int argb_stride,
                         uint8_t* y, int y_stride, uint8_t* uv, int uv_stride,
                         int w, int h, unsigned long long* out_base, int* out_pitch) {
  if (!ctx || w <= 0 || h <= 0 || (w & 1) || (h & 1)) return false;
  if (y_stride != w || uv_stride != w) return false;
  static const bool disabled = (std::getenv("PIXELFLUX_NO_CUDA_CONVERT") != nullptr);
  if (disabled) return false;
  std::lock_guard<std::mutex> lk(g_mutex);
  if (g_tried && ctx != g_init_ctx) {
    g_tried = false; g_ready = false; g_fn = nullptr;
    g_d_argb = g_d_y = g_d_uv = g_d_nv12 = 0;
    g_cap_argb = g_cap_y = g_cap_uv = g_cap_nv12 = 0;
  }
  if (!g_tried) { g_tried = true; g_init_ctx = ctx; g_ready = init_locked(ctx); }
  if (!g_ready) return false;
  if (g_setctx(ctx) != 0) return false;
  int pitch = (w + 255) & ~255;
  size_t argb_sz = (size_t)argb_stride * h;
  size_t nv12_sz = (size_t)pitch * h + (size_t)pitch * (h / 2);
  if (!ensure_buf(g_d_argb, g_cap_argb, argb_sz) || !ensure_buf(g_d_nv12, g_cap_nv12, nv12_sz)) return false;
  if (g_h2d(g_d_argb, argb, argb_sz) != 0) return false;
  CUdptr y_base = g_d_nv12;
  CUdptr uv_base = g_d_nv12 + (CUdptr)((size_t)pitch * h);
  int aw = argb_stride, yw = pitch, uvw = pitch, ww = w, hh = h;
  void* args[] = {&g_d_argb, &aw, &y_base, &yw, &uv_base, &uvw, &ww, &hh};
  if (g_launch(g_fn, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, nullptr, args, nullptr) != 0) return false;
  if (g_sync() != 0) return false;
  // Per-row download to the tight host planes for the change-detection hash.
  for (int r = 0; r < h; ++r)
    if (g_d2h(y + (size_t)r * y_stride, y_base + (CUdptr)((size_t)r * pitch), (size_t)w) != 0) return false;
  for (int r = 0; r < h / 2; ++r)
    if (g_d2h(uv + (size_t)r * uv_stride, uv_base + (CUdptr)((size_t)r * pitch), (size_t)w) != 0) return false;
  *out_base = (unsigned long long)g_d_nv12;
  *out_pitch = pitch;
  return true;
}


// Clear the cached module/device buffers (assumes g_mutex already held). The handles are
// only dropped, never freed: the underlying GPU allocations and module die with the owning
// CUcontext (freed by cuCtxDestroy), so freeing them here would double-free.
static void clear_cache_locked() {
  g_tried = false; g_ready = false; g_fn = nullptr; g_init_ctx = nullptr;
  g_d_argb = g_d_y = g_d_uv = g_d_nv12 = 0;
  g_cap_argb = g_cap_y = g_cap_uv = g_cap_nv12 = 0;
}

// Drop the cached module/device buffers bound to the current CUDA context. MUST be called
// by NVENC teardown BEFORE cuCtxDestroy: otherwise a later cuCtxCreate can hand back the
// same CUcontext address (pointer-identity ABA), the staleness check in argb_to_nv12
// (ctx == g_init_ctx) is skipped, and the kernel launches against freed GPU memory. Like
// that staleness branch we only drop the handles -- cuCtxDestroy frees the underlying
// allocations -- and taking g_mutex also serializes against any in-flight argb_to_nv12 so
// the context cannot be destroyed mid-conversion.
void invalidate() {
  std::lock_guard<std::mutex> lk(g_mutex);
  clear_cache_locked();
}

// Destroy a per-instance CUDA context safely w.r.t. the SHARED color-convert cache. Holding
// g_mutex across the destroy serializes it against any in-flight convert on any context
// (avoids an intermittent exit crash), and we clear the cache only if it is bound to THIS
// context, so one instance's teardown doesn't nuke another's live cache. Buffers/module are
// freed by cuCtxDestroy, never here.
void destroy_context(CUcontext ctx) {
  std::lock_guard<std::mutex> lk(g_mutex);
  if (g_init_ctx == ctx) clear_cache_locked();
  if (ctx && g_cuda_funcs.pfn_cuCtxDestroy) g_cuda_funcs.pfn_cuCtxDestroy(ctx);
}
}  // namespace cuda_convert

/**
 * @brief Scans the system for available VA-API compatible DRM render nodes.
 * This function searches the `/dev/dri/` directory for device files named
 * `renderD*`, which represent GPU render nodes that can be used for
 * hardware-accelerated computation like video encoding without needing a
 * graphical display server.
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

// True when GPU auto-selection is requested. SELKIES_AUTO_GPU takes precedence
// over AUTO_GPU (matches the Wayland backend).
static bool auto_gpu_enabled() {
    const char* v = getenv("SELKIES_AUTO_GPU");
    if (v == nullptr) v = getenv("AUTO_GPU");
    if (v == nullptr) return false;
    std::string s(v);
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s == "true";
}

// Walk /sys/class/drm cards in numeric order and return the first card's renderD*
// node present in /dev/dri, skipping non-GPU cards (IPMI/VGA) a bare /dev/dri scan
// would include. Returns "" if none. Mirrors the Wayland backend's auto-selection.
static std::string auto_select_render_node() {
    std::vector<std::string> cards;
    if (DIR* d = opendir("/sys/class/drm")) {
        struct dirent* e;
        while ((e = readdir(d)) != nullptr) {
            std::string n = e->d_name;
            if (n.rfind("card", 0) == 0 && n.size() > 4 &&
                n.find_first_not_of("0123456789", 4) == std::string::npos) {
                cards.push_back(n);
            }
        }
        closedir(d);
    }
    std::sort(cards.begin(), cards.end(), [](const std::string& a, const std::string& b) {
        return atoi(a.c_str() + 4) < atoi(b.c_str() + 4);
    });
    for (const std::string& card : cards) {
        std::string drm_dir = "/sys/class/drm/" + card + "/device/drm";
        std::vector<std::string> renders;
        if (DIR* dd = opendir(drm_dir.c_str())) {
            struct dirent* e;
            while ((e = readdir(dd)) != nullptr) {
                std::string en = e->d_name;
                if (en.rfind("renderD", 0) == 0) renders.push_back(en);
            }
            closedir(dd);
        }
        std::sort(renders.begin(), renders.end());
        for (const std::string& en : renders) {
            std::string dev = "/dev/dri/" + en;
            if (access(dev.c_str(), F_OK) == 0) return dev;
        }
    }
    return std::string();
}

// Kernel driver bound to a /dev/dri/renderD* node (basename of the
// /sys/class/drm/<node>/device/driver symlink, e.g. "nvidia"/"i915"/"amdgpu"; "" if unknown).
// AUTO_GPU uses this to route NVIDIA nodes to NVENC and Intel/AMD nodes to VAAPI.
static std::string render_node_kernel_driver(const std::string& dev_path) {
    if (dev_path.empty()) return std::string();
    // dev_path is like "/dev/dri/renderD128"; the matching sysfs node name is the basename.
    size_t slash = dev_path.find_last_of('/');
    std::string node = (slash == std::string::npos) ? dev_path : dev_path.substr(slash + 1);
    if (node.empty()) return std::string();
    std::string link = "/sys/class/drm/" + node + "/device/driver";
    char buf[1024];
    ssize_t n = readlink(link.c_str(), buf, sizeof(buf) - 1);
    if (n <= 0) return std::string();
    buf[n] = '\0';
    std::string target(buf);
    size_t sl = target.find_last_of('/');
    return (sl == std::string::npos) ? target : target.substr(sl + 1);
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
  int frame_counter,
  bool emit_header);

/**
 * @brief Encodes a horizontal stripe of YUV data into H.264 format using x264.
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
 * @param h264_streaming_mode True to enable streaming mode optimizations.
 * @param force_idr True to force the encoder to generate an IDR (key) frame.
 * @param is_cbr True to enable Constant Bitrate (CBR) mode, false for CRF mode.
 * @param bitrate_kbps Target bitrate in kbps when CBR mode is enabled.
 * @param vbv_buffer_size_kb VBV buffer size in kb for CBR mode (0 for auto/default).
 * @return A StripeEncodeResult containing the H.264 NAL units, or an empty result on failure.
 *         The result data includes a custom 10-byte header: type tag (0x04), frame type,
 *         frame_id (uint16_t), stripe_y_start (uint16_t), width (uint16_t), height (uint16_t),
 *         all multi-byte fields in network byte order.
 */
StripeEncodeResult encode_stripe_h264(
  MinimalEncoderStore& h264_minimal_store,
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
  bool use_full_range,
  bool h264_streaming_mode,
  bool force_id,
  bool is_cbr,
  int bitrate_kbps,
  int vbv_buffer_size_kb,
  bool emit_header);

/**
 * @brief Calculates a 64-bit XXH3 hash for a stripe of YUV data.
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
 * @param use_fullframe_hashing True to use full-frame hashing (samples every 12th row),
 *                              false to hash every row.
 * @return A 64-bit hash value of the stripe data, or 0 on error.
 */
uint64_t calculate_yuv_stripe_hash(const uint8_t* y_plane_stripe_start, int y_stride,
                                   const uint8_t* u_plane_stripe_start, int u_stride,
                                   const uint8_t* v_plane_stripe_start, int v_stride,
                                   int width, int height, bool is_i420, bool use_fullframe_hashing);

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
  int h264_paintover_crf = 18;
  int h264_paintover_burst_frames = 5;
  bool h264_fullcolor = false;
  bool h264_fullframe = false;
  bool h264_streaming_mode = false;
  bool capture_cursor = false;
  OutputMode output_mode = OutputMode::H264;
  std::string watermark_path_internal;
  WatermarkLocation watermark_location_internal;
  bool use_cpu = false;
  bool debug_logging = false;
  bool h264_cbr_mode = false;
  int h264_bitrate_kbps = 4000;
  int h264_vbv_buffer_size_kb = 0;
  bool auto_adjust_screen_capture_size = false;

  std::atomic<bool> stop_requested;
  std::thread capture_thread;
  // Per-module wire/ownership toggles (see CaptureSettings). Kept per-instance so
  // multiple modules in one process don't clobber each other's settings; atomic so
  // the capture loop and encode worker threads observe modify_settings consistently.
  std::atomic<bool> emit_stripe_headers_{true};
  std::atomic<bool> deferred_free_{false};
  StripeCallback stripe_callback = nullptr;
  void* user_data = nullptr;
  int frame_counter = 0;
  int encoded_frame_count = 0;
  int total_stripes_encoded_this_interval = 0;
  mutable std::mutex settings_mutex;
  bool is_nvidia_system_detected = false;
  bool nvenc_operational = false;
  int vaapi_render_node_index = -1;
  std::string vaapi_render_node_path_;  // resolved render-node path (AUTO_GPU / explicit / index)
  bool vaapi_operational = false;

private:
    MinimalEncoderStore h264_minimal_store_;
    NvencEncoderState nvenc_state_;
    std::mutex nvenc_mutex_;
    std::atomic<bool> nvenc_force_next_idr_{true};
    VaapiEncoderState vaapi_state_;
    std::mutex vaapi_mutex_;
    std::atomic<bool> vaapi_force_next_idr_{true};
    std::atomic<bool> force_next_idr_{false};

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

    void reset_nvenc_encoder();
    bool initialize_nvenc_encoder(int width, int height, int target_qp, double fps, bool use_yuv444, bool is_cbr, int bitrate_kbps, int vbv_buffer_size_kb);
    StripeEncodeResult encode_fullframe_nvenc(int width, int height, const uint8_t* y_plane, int y_stride, const uint8_t* u_plane, int u_stride, const uint8_t* v_plane, int v_stride, bool is_i444, int frame_counter, bool force_idr_frame);
    void reset_vaapi_encoder();
    bool initialize_vaapi_encoder(const std::string& render_node_path, int width, int height, int qp, bool use_yuv444, bool is_cbr, int bitrate_kbps, int vbv_buffer_size_kb);
    StripeEncodeResult encode_fullframe_vaapi(int width, int height, double fps, const uint8_t* y_plane, int y_stride, const uint8_t* u_plane, int u_stride, const uint8_t* v_plane, int v_stride, bool is_i444, int frame_counter, bool force_idr_frame);

    void load_watermark_image();
    void capture_loop();
    void overlay_image(int image_height, int image_width, const uint32_t *image_ptr,
                     int image_x, int image_y, int frame_height, int frame_width,
                     unsigned char *frame_ptr, int frame_stride_bytes, int frame_bytes_per_pixel);

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
  void start_capture();

  /**
   * @brief Stops the screen capture process.
   * Sets the stop_requested flag and waits for the capture thread to join.
   * This is a blocking call.
   */
  void stop_capture();

  /**
   * @brief Modifies the capture and encoding settings.
   * This function is thread-safe. The new settings will be picked up by
   * the capture loop at the beginning of its next iteration.
   * If dimensions or H.264 color format change, XShm and encoders may be reinitialized.
   * @param new_settings A CaptureSettings struct containing the new settings.
   */
  void modify_settings(const CaptureSettings& new_settings) {
    // Resolve the render-node PATH (opendir/readlink) BEFORE the lock; it depends only on
    // new_settings, so doing it lock-free keeps that I/O off the hot lock (we lock only to
    // store). Priority: AUTO_GPU > explicit path > legacy index. vaapi_render_node_index is
    // the VAAPI-vs-NVENC gate (>=0 => VAAPI node; -1 => NVENC/CPU).
    std::string resolved_node_path;
    int resolved_node_index = new_settings.vaapi_render_node_index;
    {
        std::string resolved;
        if (auto_gpu_enabled()) {
            resolved = auto_select_render_node();
            // NVIDIA does H.264 via NVENC, not VAAPI: only adopt the resolved node for the
            // VAAPI path when its driver is NOT nvidia. An nvidia node is left empty (index -1)
            // so the NVENC path is taken instead of falling back to CPU x264.
            if (!resolved.empty()) {
                std::string drv = render_node_kernel_driver(resolved);
                if (drv == "nvidia") {
                    resolved.clear();  // NVENC-capable: do not engage VAAPI
                }
            }
        } else if (new_settings.vaapi_render_node_path && new_settings.vaapi_render_node_path[0]) {
            resolved = new_settings.vaapi_render_node_path;
        } else if (resolved_node_index >= 0) {
            std::vector<std::string> nodes = find_vaapi_render_nodes();
            if (resolved_node_index < (int)nodes.size()) resolved = nodes[resolved_node_index];
            else if (!nodes.empty()) resolved = nodes[0];
        }
        resolved_node_path = resolved;
        resolved_node_index = resolved.empty() ? -1 : (resolved_node_index >= 0 ? resolved_node_index : 0);
    }

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
    h264_paintover_crf = new_settings.h264_paintover_crf;
    h264_paintover_burst_frames = new_settings.h264_paintover_burst_frames;
    h264_fullcolor = new_settings.h264_fullcolor;
    h264_fullframe = new_settings.h264_fullframe;
    h264_streaming_mode = new_settings.h264_streaming_mode;
    capture_cursor = new_settings.capture_cursor;
    emit_stripe_headers_.store(!new_settings.omit_stripe_headers, std::memory_order_relaxed);
    deferred_free_.store(new_settings.deferred_free, std::memory_order_relaxed);
    // Store the node resolved lock-free above (path is authoritative; index is the gate).
    vaapi_render_node_path_ = resolved_node_path;
    vaapi_render_node_index = resolved_node_index;
    if (!resolved_node_path.empty()) {
        std::cout << "Render node resolved: " << resolved_node_path << std::endl;
    }
    use_cpu = new_settings.use_cpu;
    debug_logging = new_settings.debug_logging;
    std::string new_wm_path_str = new_settings.watermark_path ? new_settings.watermark_path : "";
    bool path_actually_changed_in_settings = (watermark_path_internal != new_wm_path_str);
  
    watermark_path_internal = new_wm_path_str;
    watermark_location_internal = new_settings.watermark_location_enum;

    if (path_actually_changed_in_settings) {
        std::lock_guard<std::mutex> data_lock(watermark_data_mutex_);
        watermark_loaded_ = false;
    }

    h264_cbr_mode = new_settings.h264_cbr_mode;
    h264_bitrate_kbps = new_settings.h264_bitrate_kbps;
    h264_vbv_buffer_size_kb = new_settings.h264_vbv_buffer_size_kb;
    auto_adjust_screen_capture_size = new_settings.auto_adjust_screen_capture_size;
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
      damage_block_duration, output_mode, h264_crf, h264_paintover_crf,
      h264_paintover_burst_frames, h264_fullcolor, h264_fullframe, h264_streaming_mode,
      capture_cursor, watermark_path_internal.c_str(), watermark_location_internal,
      vaapi_render_node_index, use_cpu, debug_logging, h264_cbr_mode, h264_bitrate_kbps,
      h264_vbv_buffer_size_kb, auto_adjust_screen_capture_size);
  }

  /**
   * @brief Requests the next encoded frame to be an IDR (key) frame.
   * This function is thread-safe. It sets flags that will cause the next
   * encoded frame to be an IDR frame in the appropriate encoder backend.
   */
  void request_idr() {
    std::lock_guard<std::mutex> lock(settings_mutex);
    if (debug_logging) {
      const char* backend = use_cpu ? "CPU" : (nvenc_operational ? "NVENC" : (vaapi_operational ? "VAAPI" : "None"));
      std::cout << "[pixelflux] Request IDR -> " << backend << std::endl;
    }

    if (use_cpu) force_next_idr_ = true;
    else if (nvenc_operational) nvenc_force_next_idr_ = true;
    else if (vaapi_operational) vaapi_force_next_idr_ = true;
  }

  /**
   * @brief Updates the target video bitrate for H.264 CBR mode.
   * This function is thread-safe. If CBR mode is not enabled, it does nothing.
   * @param bitrate The new target bitrate in kbps.
   */
  void update_video_bitrate(int bitrate) {
    std::lock_guard<std::mutex> lock(settings_mutex);
    if (!h264_cbr_mode) return;

    if (debug_logging) {
      std::cout << "[pixelflux] Updating video bitrate from " << h264_bitrate_kbps << " to " << bitrate << std::endl;
    }

    h264_bitrate_kbps = static_cast<int>(std::abs(bitrate));
  }

  /**
   * @brief Updates the VBV buffer size for H.264 CBR mode.
   * This function is thread-safe. If CBR mode is not enabled, it does nothing.
   * @param vbv_buffer_size_kb The new VBV buffer size in kb
   */
  void update_vbv_buffer_size(int vbv_buffer_size_kb) {
    std::lock_guard<std::mutex> lock(settings_mutex);
    if (!h264_cbr_mode) return;

    if (debug_logging) {
      std::cout << "[pixelflux] Updating VBV buffer size from " << h264_vbv_buffer_size_kb << " to " << vbv_buffer_size_kb << std::endl;
    }

    h264_vbv_buffer_size_kb = static_cast<int>(std::abs(vbv_buffer_size_kb));
  }

  /**
   * @brief Updates the target framerate for video encoding.
   * This function is thread-safe.
   * @param fps The new target frames per second.
   */
  void update_framerate(double fps) {
    std::lock_guard<std::mutex> lock(settings_mutex);
    if (debug_logging) {
      std::cout << "[pixelflux] Updating video framerate from " << target_fps << " to " << fps << std::endl;
    }

    target_fps = static_cast<double>(std::abs(fps));
  }
};

/**
 * @brief Starts the screen capture process in a new thread.
 * If a capture thread is already running, this function will stop it first.
 * It resets all encoder states (CPU, NVENC, VAAPI) and frame counters to
 * ensure a clean start. It also probes for hardware encoder availability and
 * pre-loads any configured watermark image. The capture itself runs in the
 * background. The settings for the capture must be set via `modify_settings`
 * prior to calling this function.
 */
void ScreenCaptureModule::start_capture() {
    if (capture_thread.joinable()) {
      stop_capture();
    }
    if (LoadNvencApi(nvenc_state_.nvenc_funcs)) {
      is_nvidia_system_detected = true;
    } else {
      is_nvidia_system_detected = false;
    }
    h264_minimal_store_.reset();
    nvenc_operational = false;
    nvenc_force_next_idr_ = true;
    vaapi_operational = false;
    vaapi_force_next_idr_ = true;
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
 * @brief Stops the screen capture process and releases resources.
 * This function signals the background capture thread to stop. It is a
 * blocking call that waits for the thread to finish its current work and
 * join. After the thread has terminated, it cleans up this instance's hardware
 * encoder sessions (NVENC, VAAPI) and their per-instance GPU resources.
 */
void ScreenCaptureModule::stop_capture() {
    stop_requested = true;
    if (capture_thread.joinable()) {
      capture_thread.join();
    }
    if (nvenc_state_.initialized) {
      reset_nvenc_encoder();  // destroys THIS instance's CUDA context + NVENC session
    }
    if (vaapi_state_.initialized) {
      reset_vaapi_encoder();
    }
    // Deliberately do NOT unload the process-global CUDA driver here: with multiple
    // instances per process, that would corrupt another instance still mid-encode.
    // libcuda is process-lifetime (see UnloadCudaApi); per-instance state is freed above.
}

/**
 * @brief Resets and tears down the current NVENC encoder session.
 * This function is thread-safe. It destroys all allocated input and output
 * buffers, destroys the encoder session, and releases the CUDA context.
 * It ensures that all GPU resources associated with the encoder are freed.
 */
void ScreenCaptureModule::reset_nvenc_encoder() {
  std::lock_guard<std::mutex> lock(nvenc_mutex_);

  if (!nvenc_state_.initialized) {
    return;
  }

  if (nvenc_state_.encoder_session && nvenc_state_.nvenc_funcs.nvEncDestroyEncoder) {
    // E2: unregister the cached device-input resource BEFORE destroying the session.
    if (nvenc_state_.registered_resource && nvenc_state_.nvenc_funcs.nvEncUnregisterResource) {
        nvenc_state_.nvenc_funcs.nvEncUnregisterResource(nvenc_state_.encoder_session, nvenc_state_.registered_resource);
    }
    nvenc_state_.registered_resource = nullptr;
    nvenc_state_.registered_base = 0;
    nvenc_state_.registered_w = nvenc_state_.registered_h = nvenc_state_.registered_pitch = 0;
    for (NV_ENC_INPUT_PTR& ptr : nvenc_state_.input_buffers) {
        if (ptr && nvenc_state_.nvenc_funcs.nvEncDestroyInputBuffer)
            nvenc_state_.nvenc_funcs.nvEncDestroyInputBuffer(nvenc_state_.encoder_session, ptr);
        ptr = nullptr;
    }
    nvenc_state_.input_buffers.clear();

    for (NV_ENC_OUTPUT_PTR& ptr : nvenc_state_.output_buffers) {
        if (ptr && nvenc_state_.nvenc_funcs.nvEncDestroyBitstreamBuffer)
            nvenc_state_.nvenc_funcs.nvEncDestroyBitstreamBuffer(nvenc_state_.encoder_session, ptr);
        ptr = nullptr;
    }
    nvenc_state_.output_buffers.clear();

    nvenc_state_.nvenc_funcs.nvEncDestroyEncoder(nvenc_state_.encoder_session);
    nvenc_state_.encoder_session = nullptr;
  }

  if (nvenc_state_.cuda_context && g_cuda_funcs.pfn_cuCtxDestroy) {
    // Destroy under cuda_convert's g_mutex (serializes vs in-flight convert; clears the
    // shared cache only if bound to THIS context). See cuda_convert::destroy_context.
    cuda_convert::destroy_context(nvenc_state_.cuda_context);
    nvenc_state_.cuda_context = nullptr;
  }

  nvenc_state_.initialized = false;
}

/**
 * @brief Initializes or reconfigures the NVENC H.264 encoder.
 * This function is thread-safe. It sets up a new encoder session if one
 * is not already active. If an active session exists with the correct
 * dimensions and color format, it attempts a lightweight reconfiguration for
 * the target QP (quality) or bitrate. If dimensions or color format have changed, it
 * performs a full teardown and re-initialization.
 * @param width The target encoding width.
 * @param height The target encoding height.
 * @param target_qp The target Quantization Parameter (lower is higher quality) for CRF mode.
 * @param fps The target frames per second, used for rate control hints.
 * @param use_yuv444 True to configure for YUV 4:4:4, false for NV12 (4:2:0).
 * @param is_cbr True to enable Constant Bitrate (CBR) mode, false for CRF mode.
 * @param bitrate_kbps Target bitrate in kbps when CBR mode is enabled.
 * @param vbv_buffer_size_kb VBV buffer size in kb for CBR mode (0 for auto/default).
 * @return True if the encoder is successfully initialized or reconfigured, false otherwise.
 */
bool ScreenCaptureModule::initialize_nvenc_encoder(int width,
                              int height,
                              int target_qp,
                              double fps,
                              bool use_yuv444,
                              bool is_cbr,
                              int bitrate_kbps,
                              int vbv_buffer_size_kb) {
  std::lock_guard<std::mutex> lock(nvenc_mutex_);

  NV_ENC_BUFFER_FORMAT target_buffer_format =
    use_yuv444 ? NV_ENC_BUFFER_FORMAT_YUV444 : NV_ENC_BUFFER_FORMAT_NV12;

  if (nvenc_state_.initialized && nvenc_state_.initialized_width == width &&
      nvenc_state_.initialized_height == height &&
      nvenc_state_.initialized_qp == target_qp &&
      nvenc_state_.initialized_buffer_format == target_buffer_format &&
      nvenc_state_.cbr_mode == is_cbr &&
      nvenc_state_.initialized_bitrate_kbps == bitrate_kbps) {
    return true;
  }

  if (nvenc_state_.initialized && nvenc_state_.initialized_width == width &&
      nvenc_state_.initialized_height == height &&
      nvenc_state_.initialized_buffer_format == target_buffer_format &&
      nvenc_state_.cbr_mode == is_cbr) {

    bool reconfig_needed = false;
    if (is_cbr) {
        if (nvenc_state_.initialized_bitrate_kbps != bitrate_kbps) reconfig_needed = true;
    } else {
        if (nvenc_state_.initialized_qp != target_qp) reconfig_needed = true;
    }

    if (reconfig_needed) {
      NV_ENC_RECONFIGURE_PARAMS reconfigure_params = {0};
      NV_ENC_CONFIG new_config = nvenc_state_.encode_config;

      reconfigure_params.version = NV_ENC_RECONFIGURE_PARAMS_VER;
      reconfigure_params.reInitEncodeParams = nvenc_state_.init_params;
      reconfigure_params.reInitEncodeParams.encodeConfig = &new_config;

      if (is_cbr) {
        new_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
        uint32_t bps = static_cast<uint32_t>(bitrate_kbps * 1000);
        new_config.rcParams.averageBitRate = bps;
        new_config.rcParams.maxBitRate = bps;
        if (vbv_buffer_size_kb > 0) {
          new_config.rcParams.vbvBufferSize = static_cast<uint32_t>(vbv_buffer_size_kb * 1000);
        } else {
          new_config.rcParams.vbvBufferSize = (bps + 9) / 10;
        }
        reconfigure_params.forceIDR = false;
      } else {
        new_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
        new_config.rcParams.constQP.qpInterP = target_qp;
        new_config.rcParams.constQP.qpIntra = target_qp;
        new_config.rcParams.constQP.qpInterB = target_qp;
        
        bool is_quality_increasing = (target_qp < nvenc_state_.initialized_qp);
        reconfigure_params.forceIDR = is_quality_increasing;
      }

      NVENCSTATUS status = nvenc_state_.nvenc_funcs.nvEncReconfigureEncoder(
        nvenc_state_.encoder_session, &reconfigure_params);

      if (status == NV_ENC_SUCCESS) {
        nvenc_state_.initialized_qp = target_qp;
        nvenc_state_.cbr_mode = is_cbr;
        nvenc_state_.initialized_bitrate_kbps = bitrate_kbps;
        nvenc_state_.encode_config = new_config;
        return true;
      }
    } else {
      return true;
    }
  }

  if (nvenc_state_.initialized) {
    // Manually unlock before recursive call to reset, then re-lock.
    nvenc_mutex_.unlock();
    reset_nvenc_encoder();
    nvenc_mutex_.lock();
  }

  if (!LoadCudaApi()) {
    std::cerr << "NVENC_INIT_FATAL: Failed to load CUDA driver API." << std::endl;
    return false;
  }
  
  if (!LoadNvencApi(nvenc_state_.nvenc_funcs)) {
      nvenc_state_.initialized = false;
      return false;
  }

  if (!nvenc_state_.nvenc_funcs.nvEncOpenEncodeSessionEx) {
    nvenc_state_.initialized = false;
    return false;
  }

  // libcuda and libnvidia-encode are now loaded; filter hidden GPUs out of the
  // RM enumeration before opening the session (see InstallNvencGpuFilter).
  InstallNvencGpuFilter();

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
  cu_status = g_cuda_funcs.pfn_cuCtxCreate(&nvenc_state_.cuda_context, 0, cu_device);
  if (cu_status != CUDA_SUCCESS) {
      std::cerr << "NVENC_INIT_ERROR: cuCtxCreate failed with code " << cu_status << std::endl;
      return false;
  }

  NVENCSTATUS status;
  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS session_params = {0};
  session_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
  session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
  session_params.device = nvenc_state_.cuda_context;
  session_params.apiVersion = NVENCAPI_VERSION;

  status = nvenc_state_.nvenc_funcs.nvEncOpenEncodeSessionEx(
    &session_params, &nvenc_state_.encoder_session);

  if (status != NV_ENC_SUCCESS) {
    std::string error_str = "NVENC_INIT_ERROR: nvEncOpenEncodeSessionEx (CUDA Path) FAILED: " + std::to_string(status);
    std::cerr << error_str << std::endl;
    nvenc_state_.encoder_session = nullptr;
    nvenc_mutex_.unlock();
    reset_nvenc_encoder();
    nvenc_mutex_.lock();
    return false;
  }
  if (!nvenc_state_.encoder_session) {
    nvenc_mutex_.unlock();
    reset_nvenc_encoder();
    nvenc_mutex_.lock();
    return false;
  }

  memset(&nvenc_state_.init_params, 0, sizeof(nvenc_state_.init_params));
  nvenc_state_.init_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
  nvenc_state_.init_params.encodeGUID = NV_ENC_CODEC_H264_GUID;
  nvenc_state_.init_params.presetGUID = NV_ENC_PRESET_P1_GUID;
  nvenc_state_.init_params.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
  nvenc_state_.init_params.encodeWidth = width;
  nvenc_state_.init_params.encodeHeight = height;
  nvenc_state_.init_params.darWidth = width;
  nvenc_state_.init_params.darHeight = height;
  nvenc_state_.init_params.frameRateNum = static_cast<uint32_t>(fps < 1.0 ? 30 : fps);
  nvenc_state_.init_params.frameRateDen = 1;
  nvenc_state_.init_params.enablePTD = 1;

  NV_ENC_PRESET_CONFIG preset_config = {0};
  preset_config.version = NV_ENC_PRESET_CONFIG_VER;
  preset_config.presetCfg.version = NV_ENC_CONFIG_VER;

  if (nvenc_state_.nvenc_funcs.nvEncGetEncodePresetConfigEx) {
    status = nvenc_state_.nvenc_funcs.nvEncGetEncodePresetConfigEx(
      nvenc_state_.encoder_session,
      nvenc_state_.init_params.encodeGUID,
      nvenc_state_.init_params.presetGUID,
      nvenc_state_.init_params.tuningInfo,
      &preset_config);

    if (status != NV_ENC_SUCCESS) {
      std::cerr << "NVENC_INIT_WARN: nvEncGetEncodePresetConfigEx FAILED: " << status
                << ". Falling back to manual config." << std::endl;
      memset(&nvenc_state_.encode_config, 0, sizeof(nvenc_state_.encode_config));
      nvenc_state_.encode_config.version = NV_ENC_CONFIG_VER;
    } else {
      nvenc_state_.encode_config = preset_config.presetCfg;
      nvenc_state_.encode_config.version = NV_ENC_CONFIG_VER;
    }
  } else {
    std::cerr << "NVENC_INIT_WARN: nvEncGetEncodePresetConfigEx not available. Using manual "
                 "config."
              << std::endl;
    memset(&nvenc_state_.encode_config, 0, sizeof(nvenc_state_.encode_config));
    nvenc_state_.encode_config.version = NV_ENC_CONFIG_VER;
  }

  nvenc_state_.encode_config.profileGUID =
    use_yuv444 ? NV_ENC_H264_PROFILE_HIGH_444_GUID : NV_ENC_H264_PROFILE_HIGH_GUID;
  
  if (is_cbr) {
     nvenc_state_.encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
     uint32_t bps = static_cast<uint32_t>(bitrate_kbps * 1000);
     nvenc_state_.encode_config.rcParams.averageBitRate = bps;
     nvenc_state_.encode_config.rcParams.maxBitRate = bps;
     if (vbv_buffer_size_kb > 0) {
       nvenc_state_.encode_config.rcParams.vbvBufferSize = static_cast<uint32_t>(vbv_buffer_size_kb * 1000);
     } else {
       nvenc_state_.encode_config.rcParams.vbvBufferSize = bps * 0.1;
     }
  } else {
      nvenc_state_.encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
      nvenc_state_.encode_config.rcParams.constQP.qpInterP = target_qp;
      nvenc_state_.encode_config.rcParams.constQP.qpIntra = target_qp;
      nvenc_state_.encode_config.rcParams.constQP.qpInterB = target_qp;
  }
  nvenc_state_.encode_config.gopLength = NVENC_INFINITE_GOPLENGTH;
  nvenc_state_.encode_config.frameIntervalP = 1;

  NV_ENC_CONFIG_H264* h264_cfg = &nvenc_state_.encode_config.encodeCodecConfig.h264Config;
  // Decodable High 4:4:4 = chromaFormatIDC=3, separate_colour_plane_flag=0 (leave the default;
  // 1 yields a stream decoders reject). 4:4:4 also requires the cuCtxCreate_v2 context (see
  // LoadCudaApi); the v1 context makes NVENC's 4:4:4 CUDA ops fail.
  h264_cfg->chromaFormatIDC = use_yuv444 ? 3 : 1;
  h264_cfg->h264VUIParameters.videoFullRangeFlag = use_yuv444 ? 1 : 0;
  h264_cfg->repeatSPSPPS = 1;
  nvenc_state_.init_params.encodeConfig = &nvenc_state_.encode_config;

  status = nvenc_state_.nvenc_funcs.nvEncInitializeEncoder(nvenc_state_.encoder_session,
                                                            &nvenc_state_.init_params);
  if (status != NV_ENC_SUCCESS) {
    std::string error_str =
      "NVENC_INIT_ERROR: nvEncInitializeEncoder FAILED: " + std::to_string(status);
    if (nvenc_state_.nvenc_funcs.nvEncGetLastErrorString) {
      const char* api_err =
        nvenc_state_.nvenc_funcs.nvEncGetLastErrorString(nvenc_state_.encoder_session);
      if (api_err)
        error_str += " - API Error: " + std::string(api_err);
    }
    std::cerr << error_str << std::endl;

    nvenc_mutex_.unlock();
    reset_nvenc_encoder();
    nvenc_mutex_.lock();
    return false;
  }

  nvenc_state_.input_buffers.resize(nvenc_state_.buffer_pool_size);
  nvenc_state_.output_buffers.resize(nvenc_state_.buffer_pool_size);
  for (int i = 0; i < nvenc_state_.buffer_pool_size; ++i) {
    NV_ENC_CREATE_INPUT_BUFFER icp = {0};
    icp.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
    icp.width = width;
    icp.height = height;
    icp.bufferFmt = target_buffer_format;
    status = nvenc_state_.nvenc_funcs.nvEncCreateInputBuffer(nvenc_state_.encoder_session,
                                                              &icp);
    if (status != NV_ENC_SUCCESS) {
      nvenc_mutex_.unlock();
      reset_nvenc_encoder();
      nvenc_mutex_.lock();
      return false;
    }
    nvenc_state_.input_buffers[i] = icp.inputBuffer;
    NV_ENC_CREATE_BITSTREAM_BUFFER ocp = {0};
    ocp.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    status = nvenc_state_.nvenc_funcs.nvEncCreateBitstreamBuffer(
      nvenc_state_.encoder_session, &ocp);
    if (status != NV_ENC_SUCCESS) {
      nvenc_mutex_.unlock();
      reset_nvenc_encoder();
      nvenc_mutex_.lock();
      return false;
    }
    nvenc_state_.output_buffers[i] = ocp.bitstreamBuffer;
  }
  nvenc_state_.initialized_width = width;
  nvenc_state_.initialized_height = height;
  nvenc_state_.initialized_qp = target_qp;
  nvenc_state_.initialized_buffer_format = target_buffer_format;
  nvenc_state_.cbr_mode = is_cbr;
  nvenc_state_.initialized_bitrate_kbps = bitrate_kbps;
  nvenc_state_.initialized = true;
  return true;
}

/**
 * @brief Encodes a full YUV frame using the initialized NVENC session.
 * This function is thread-safe. It takes YUV plane data, copies it into a
 * locked NVENC input buffer, and submits it for encoding. It then retrieves
 * the resulting H.264 bitstream, prepends a custom 10-byte header, and
 * returns it.
 * @param width The width of the frame.
 * @param height The height of the frame.
 * @param y_plane Pointer to the Y plane data.
 * @param y_stride Stride of the Y plane.
 * @param u_plane Pointer to the U plane (or interleaved UV plane for NV12).
 * @param u_stride Stride of the U/UV plane.
 * @param v_plane Pointer to the V plane (used for I444, null for NV12).
 * @param v_stride Stride of the V plane.
 * @param is_i444 True if the input is YUV 4:4:4, false if NV12.
 * @param frame_counter The current frame ID.
 * @param force_idr_frame True to force the encoder to generate an IDR (key) frame.
 * @return A StripeEncodeResult containing the encoded H.264 data.
 * @throws std::runtime_error if any NVENC API call fails during the process.
 */
StripeEncodeResult ScreenCaptureModule::encode_fullframe_nvenc(int width,
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

  std::lock_guard<std::mutex> lock(nvenc_mutex_);

  if (!nvenc_state_.initialized) {
    throw std::runtime_error("NVENC_ENCODE_FATAL: Not initialized.");
  }

  NV_ENC_INPUT_PTR in_ptr =
    nvenc_state_.input_buffers[nvenc_state_.current_input_buffer_idx];
  NV_ENC_OUTPUT_PTR out_ptr =
    nvenc_state_.output_buffers[nvenc_state_.current_output_buffer_idx];

  // E2 gated device-input: register+map the contiguous device NV12 the conversion site
  // produced (dev_input_base) and feed it directly, skipping the host lock+copy. Falls
  // back to the host path on any failure. Off by default (dev_input_base == 0).
  NVENCSTATUS status = NV_ENC_SUCCESS;
  NV_ENC_INPUT_PTR pic_input = nullptr;
  int pic_pitch = 0;
  NV_ENC_INPUT_PTR mapped_resource = nullptr;
  // This runs on a fresh per-frame thread with NO current CUDA context, but the device-input
  // path's register/map/encode/unmap needs the encoder's context current on THIS thread. Push
  // it for that span and pop after (best-effort: fall back to cuCtxSetCurrent, else unchanged).
  // The host-buffer path below doesn't need it. 4:4:4 stays on the host path (gated !is_i444).
  bool nvenc_ctx_pushed = false;
  bool nvenc_ctx_set = false;
  if (cuda_convert::nvenc_device_input_enabled() && nvenc_state_.dev_input_base != 0 && !is_i444 &&
      nvenc_state_.nvenc_funcs.nvEncRegisterResource && nvenc_state_.nvenc_funcs.nvEncMapInputResource) {
    if (nvenc_state_.cuda_context) {
      if (g_cuda_funcs.pfn_cuCtxPushCurrent &&
          g_cuda_funcs.pfn_cuCtxPushCurrent(nvenc_state_.cuda_context) == CUDA_SUCCESS) {
        nvenc_ctx_pushed = true;
      } else if (g_cuda_funcs.pfn_cuCtxSetCurrent &&
                 g_cuda_funcs.pfn_cuCtxSetCurrent(nvenc_state_.cuda_context) == CUDA_SUCCESS) {
        nvenc_ctx_set = true;
      }
    }
    if (nvenc_state_.registered_resource &&
        (nvenc_state_.registered_base != nvenc_state_.dev_input_base ||
         nvenc_state_.registered_w != width || nvenc_state_.registered_h != height ||
         nvenc_state_.registered_pitch != nvenc_state_.dev_input_pitch)) {
      if (nvenc_state_.nvenc_funcs.nvEncUnregisterResource)
        nvenc_state_.nvenc_funcs.nvEncUnregisterResource(nvenc_state_.encoder_session, nvenc_state_.registered_resource);
      nvenc_state_.registered_resource = nullptr;
    }
    if (!nvenc_state_.registered_resource) {
      NV_ENC_REGISTER_RESOURCE rr = {0};
      rr.version = NV_ENC_REGISTER_RESOURCE_VER;
      rr.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
      rr.width = width; rr.height = height; rr.pitch = nvenc_state_.dev_input_pitch;
      rr.resourceToRegister = reinterpret_cast<void*>(static_cast<uintptr_t>(nvenc_state_.dev_input_base));
      rr.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
      rr.bufferUsage = NV_ENC_INPUT_IMAGE;
      if (nvenc_state_.nvenc_funcs.nvEncRegisterResource(nvenc_state_.encoder_session, &rr) == NV_ENC_SUCCESS) {
        nvenc_state_.registered_resource = rr.registeredResource;
        nvenc_state_.registered_base = nvenc_state_.dev_input_base;
        nvenc_state_.registered_w = width; nvenc_state_.registered_h = height;
        nvenc_state_.registered_pitch = nvenc_state_.dev_input_pitch;
      }
    }
    if (nvenc_state_.registered_resource) {
      NV_ENC_MAP_INPUT_RESOURCE mr = {0};
      mr.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
      mr.registeredResource = nvenc_state_.registered_resource;
      if (nvenc_state_.nvenc_funcs.nvEncMapInputResource(nvenc_state_.encoder_session, &mr) == NV_ENC_SUCCESS) {
        mapped_resource = mr.mappedResource;
        pic_input = mapped_resource;
        pic_pitch = nvenc_state_.dev_input_pitch;
      }
    }
  }

  if (!pic_input) {
    NV_ENC_LOCK_INPUT_BUFFER lip = {0};
    lip.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
    lip.inputBuffer = in_ptr;
    status =
      nvenc_state_.nvenc_funcs.nvEncLockInputBuffer(nvenc_state_.encoder_session, &lip);
    if (status != NV_ENC_SUCCESS)
      throw std::runtime_error("NVENC_ENCODE_ERROR: nvEncLockInputBuffer FAILED: " +
                               std::to_string(status));

    unsigned char* locked_buffer = static_cast<unsigned char*>(lip.bufferDataPtr);
    int locked_pitch = lip.pitch;

    uint8_t* y_dst = locked_buffer;
    uint8_t* uv_or_u_dst = locked_buffer + static_cast<size_t>(locked_pitch) * height;

    if (is_i444) {
      uint8_t* v_dst = uv_or_u_dst + static_cast<size_t>(locked_pitch) * height;
      libyuv::CopyPlane(y_plane, y_stride, y_dst, locked_pitch, width, height);
      libyuv::CopyPlane(u_plane, u_stride, uv_or_u_dst, locked_pitch, width, height);
      libyuv::CopyPlane(v_plane, v_stride, v_dst, locked_pitch, width, height);
    } else {
      if (v_plane) {
          libyuv::I420ToNV12(y_plane, y_stride, u_plane, u_stride, v_plane, v_stride,
                              y_dst, locked_pitch, uv_or_u_dst, locked_pitch, width, height);
      } else {
          libyuv::CopyPlane(y_plane, y_stride, y_dst, locked_pitch, width, height);
          libyuv::CopyPlane(u_plane, u_stride, uv_or_u_dst, locked_pitch, width, height / 2);
      }
    }

    nvenc_state_.nvenc_funcs.nvEncUnlockInputBuffer(nvenc_state_.encoder_session, in_ptr);
    pic_input = in_ptr;
    pic_pitch = locked_pitch;
  }

  NV_ENC_PIC_PARAMS pp = {0};
  pp.version = NV_ENC_PIC_PARAMS_VER;
  pp.inputBuffer = pic_input;
  pp.outputBitstream = out_ptr;
  pp.bufferFmt = nvenc_state_.initialized_buffer_format;
  pp.inputWidth = width;
  pp.inputHeight = height;
  pp.inputPitch = pic_pitch;
  pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
  pp.inputTimeStamp = frame_counter;
  pp.frameIdx = frame_counter;
  if (force_idr_frame) {
    pp.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;
  }

  status =
    nvenc_state_.nvenc_funcs.nvEncEncodePicture(nvenc_state_.encoder_session, &pp);
  if (status != NV_ENC_SUCCESS) {
    std::string err_msg = "NVENC_ENCODE_ERROR: nvEncEncodePicture FAILED: " + std::to_string(status);
    throw std::runtime_error(err_msg);
  }

  // Device-input was consumed by the (synchronous) encode; unmap it. A throw above
  // leaves it mapped, but the encoder reset on that error path destroys the session.
  if (mapped_resource && nvenc_state_.nvenc_funcs.nvEncUnmapInputResource) {
    nvenc_state_.nvenc_funcs.nvEncUnmapInputResource(nvenc_state_.encoder_session, mapped_resource);
    mapped_resource = nullptr;
  }

  // Restore the thread's prior context now the device-input span is done (pop balances push,
  // else unset SetCurrent). A throw before here leaks the push, but only on this ending thread.
  if (nvenc_ctx_pushed && g_cuda_funcs.pfn_cuCtxPopCurrent) {
    CUcontext popped = nullptr;
    g_cuda_funcs.pfn_cuCtxPopCurrent(&popped);
  } else if (nvenc_ctx_set && g_cuda_funcs.pfn_cuCtxSetCurrent) {
    g_cuda_funcs.pfn_cuCtxSetCurrent(nullptr);
  }

  NV_ENC_LOCK_BITSTREAM lbs = {0};
  lbs.version = NV_ENC_LOCK_BITSTREAM_VER;
  lbs.outputBitstream = out_ptr;
  status =
    nvenc_state_.nvenc_funcs.nvEncLockBitstream(nvenc_state_.encoder_session, &lbs);
  if (status != NV_ENC_SUCCESS) {
    throw std::runtime_error("NVENC_ENCODE_ERROR: nvEncLockBitstream FAILED: " + std::to_string(status));
  }

  if (lbs.bitstreamSizeInBytes > 0) {
    const unsigned char TAG = 0x04;
    unsigned char type_hdr = 0x00;
    if (lbs.pictureType == NV_ENC_PIC_TYPE_IDR) type_hdr = 0x01;
    else if (lbs.pictureType == NV_ENC_PIC_TYPE_I) type_hdr = 0x02;

    int header_sz = emit_stripe_headers_.load(std::memory_order_relaxed) ? 10 : 0;
    result.data = new unsigned char[lbs.bitstreamSizeInBytes + header_sz];
    result.size = lbs.bitstreamSizeInBytes + header_sz;
    if (header_sz) {
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
    }
    std::memcpy(result.data + header_sz, lbs.bitstreamBufferPtr, lbs.bitstreamSizeInBytes);
  } else {
    result.size = 0;
    result.data = nullptr;
  }

  nvenc_state_.nvenc_funcs.nvEncUnlockBitstream(nvenc_state_.encoder_session, out_ptr);

  nvenc_state_.current_input_buffer_idx = (nvenc_state_.current_input_buffer_idx + 1) % nvenc_state_.buffer_pool_size;
  nvenc_state_.current_output_buffer_idx = (nvenc_state_.current_output_buffer_idx + 1) % nvenc_state_.buffer_pool_size;

  return result;
}

/**
 * @brief Releases all resources associated with the VA-API encoder session.
 * This function is thread-safe. It frees all allocated libav objects,
 * including the codec context, hardware device and frame contexts, and reusable
 * frame and packet structures. It resets the state to uninitialized.
 */
void ScreenCaptureModule::reset_vaapi_encoder() {
    std::lock_guard<std::mutex> lock(vaapi_mutex_);
    if (!vaapi_state_.initialized) {
        return;
    }
    if (vaapi_state_.codec_ctx) {
        avcodec_free_context(&vaapi_state_.codec_ctx);
    }
    if (vaapi_state_.hw_frames_ctx) {
        av_buffer_unref(&vaapi_state_.hw_frames_ctx);
    }
    if (vaapi_state_.hw_device_ctx) {
        av_buffer_unref(&vaapi_state_.hw_device_ctx);
    }
    if (vaapi_state_.sw_frame) {
        av_frame_free(&vaapi_state_.sw_frame);
    }
    if (vaapi_state_.hw_frame) {
        av_frame_free(&vaapi_state_.hw_frame);
    }
    if (vaapi_state_.packet) {
        av_packet_free(&vaapi_state_.packet);
    }
    vaapi_state_ = {};
    if (debug_logging) {
        std::cout << "VAAPI: Encoder resources released." << std::endl;
    }
}

/**
 * @brief Initializes a VA-API H.264 hardware encoder using libavcodec.
 * This function is thread-safe. It configures and opens the 'h264_vaapi'
 * encoder. This involves creating a VA-API hardware device context for a
 * specific DRM render node, setting up a hardware frame context for GPU
 * surface management, and configuring the encoder with the specified
 * dimensions, quality (QP), and pixel format.
 * @param render_node_idx The index of the /dev/dri/renderD node to use.
 * @param width The target encoding width.
 * @param height The target encoding height.
 * @param qp The target Quantization Parameter for Constant QP (CQP) rate control.
 * @param use_yuv444 If true, configures the encoder for YUV 4:4:4 input;
 *                   otherwise, configures for YUV 4:2:0 (NV12).
 * @param is_cbr True to enable Constant Bitrate (CBR) mode, false for CQP mode.
 * @param bitrate_kbps Target bitrate in kbps when CBR mode is enabled.
 * @param vbv_buffer_size_kb VBV buffer size in kb for CBR mode (0 for auto/default).
 * @return True if the encoder was successfully initialized, false otherwise.
 */
bool ScreenCaptureModule::initialize_vaapi_encoder(const std::string& render_node_path, int width, int height, int qp, bool use_yuv444, bool is_cbr, int bitrate_kbps, int vbv_buffer_size_kb) {
    std::unique_lock<std::mutex> lock(vaapi_mutex_);
    if (vaapi_state_.initialized && vaapi_state_.initialized_width == width &&
        vaapi_state_.initialized_height == height &&
        vaapi_state_.initialized_is_444 == use_yuv444 &&
        vaapi_state_.initialized_cbr == is_cbr) {
        if (is_cbr) {
            if (vaapi_state_.initialized_bitrate_kbps == bitrate_kbps) return true;
        } else {
            if (vaapi_state_.initialized_qp == qp) return true;
        }
    }
    if (vaapi_state_.initialized) {
        lock.unlock();
        reset_vaapi_encoder();
        lock.lock();
    }
    int ret = 0;
    const AVCodec *codec = avcodec_find_encoder_by_name("h264_vaapi");
    if (!codec) {
        std::cerr << "VAAPI_INIT: Codec 'h264_vaapi' not found." << std::endl;
        return false;
    }
    std::vector<std::string> nodes = find_vaapi_render_nodes();
    if (nodes.empty()) {
        std::cerr << "VAAPI_INIT: No /dev/dri/renderD nodes found." << std::endl;
        return false;
    }
    std::string node_to_use = !render_node_path.empty() ? render_node_path : nodes[0];
    if (debug_logging) {
        std::cout << "VAAPI_INIT: Using render node: " << node_to_use << std::endl;
    }
    ret = av_hwdevice_ctx_create(&vaapi_state_.hw_device_ctx, AV_HWDEVICE_TYPE_VAAPI, node_to_use.c_str(), NULL, 0);
    if (ret < 0) {
        std::cerr << "VAAPI_INIT: Failed to create VAAPI hardware device context: " << ret << std::endl;
        return false;
    }
    vaapi_state_.codec_ctx = avcodec_alloc_context3(codec);
    if (!vaapi_state_.codec_ctx) {
        std::cerr << "VAAPI_INIT: Failed to allocate codec context." << std::endl;
        return false;
    }
    vaapi_state_.codec_ctx->width = width;
    vaapi_state_.codec_ctx->height = height;
    vaapi_state_.codec_ctx->time_base = {1, (int)target_fps};
    vaapi_state_.codec_ctx->framerate = {(int)target_fps, 1};
    vaapi_state_.codec_ctx->pix_fmt = AV_PIX_FMT_VAAPI;
    vaapi_state_.codec_ctx->gop_size = INT_MAX;
    vaapi_state_.codec_ctx->max_b_frames = 0;
    av_opt_set(vaapi_state_.codec_ctx->priv_data, "tune", "zerolatency", 0);
    av_opt_set(vaapi_state_.codec_ctx->priv_data, "preset", "ultrafast", 0);
    if (use_yuv444) {
        av_opt_set_int(vaapi_state_.codec_ctx, "profile", AV_PROFILE_H264_HIGH_444_PREDICTIVE, 0);
    } else {
        av_opt_set_int(vaapi_state_.codec_ctx, "profile", AV_PROFILE_H264_HIGH, 0);
    }

     if (is_cbr) {
        av_opt_set(vaapi_state_.codec_ctx->priv_data, "rc_mode", "CBR", 0);
        int64_t bps = static_cast<int64_t>(bitrate_kbps) * 1000;
        vaapi_state_.codec_ctx->bit_rate = bps;
        vaapi_state_.codec_ctx->rc_max_rate = bps;
        if (vbv_buffer_size_kb > 0) {
          vaapi_state_.codec_ctx->rc_buffer_size = static_cast<int64_t>(vbv_buffer_size_kb) * 1000;
        } else {
          vaapi_state_.codec_ctx->rc_buffer_size = (bps + 9) / 10;
        }
        vaapi_state_.codec_ctx->rc_min_rate = bps;
     } else {
         av_opt_set(vaapi_state_.codec_ctx->priv_data, "rc_mode", "CQP", 0);
         av_opt_set_int(vaapi_state_.codec_ctx->priv_data, "qp", qp, 0);
      }

    vaapi_state_.hw_frames_ctx = av_hwframe_ctx_alloc(vaapi_state_.hw_device_ctx);
    if (!vaapi_state_.hw_frames_ctx) {
        std::cerr << "VAAPI_INIT: Failed to create hardware frames context." << std::endl;
        return false;
    }
    AVHWFramesContext *frames_ctx = (AVHWFramesContext *)(vaapi_state_.hw_frames_ctx->data);
    frames_ctx->format = AV_PIX_FMT_VAAPI;
    frames_ctx->sw_format = use_yuv444 ? AV_PIX_FMT_YUV444P : AV_PIX_FMT_NV12;
    frames_ctx->width = width;
    frames_ctx->height = height;
    frames_ctx->initial_pool_size = 20;
    ret = av_hwframe_ctx_init(vaapi_state_.hw_frames_ctx);
    if (ret < 0) {
        std::cerr << "VAAPI_INIT: Failed to initialize hardware frames context: " << ret << std::endl;
        return false;
    }
    vaapi_state_.codec_ctx->hw_frames_ctx = av_buffer_ref(vaapi_state_.hw_frames_ctx);
    if (!vaapi_state_.codec_ctx->hw_frames_ctx) {
        std::cerr << "VAAPI_INIT: Failed to link hardware frames context." << std::endl;
        return false;
    }
    ret = avcodec_open2(vaapi_state_.codec_ctx, codec, NULL);
    if (ret < 0) {
        std::cerr << "VAAPI_INIT: Failed to open codec: " << ret << std::endl;
        return false;
    }
    vaapi_state_.sw_frame = av_frame_alloc();
    vaapi_state_.hw_frame = av_frame_alloc();
    vaapi_state_.packet = av_packet_alloc();
    if (!vaapi_state_.sw_frame || !vaapi_state_.hw_frame || !vaapi_state_.packet) {
        std::cerr << "VAAPI_INIT: Failed to allocate reusable frame/packet objects." << std::endl;
        return false;
    }
    vaapi_state_.initialized = true;
    vaapi_state_.initialized_width = width;
    vaapi_state_.initialized_height = height;
    vaapi_state_.initialized_qp = qp;
    vaapi_state_.initialized_is_444 = use_yuv444;
    vaapi_state_.frame_count = 0;
    vaapi_state_.initialized_cbr = is_cbr;
    vaapi_state_.initialized_bitrate_kbps = bitrate_kbps;
    if (debug_logging) {
        std::cout << "VAAPI_INIT: Encoder initialized successfully via FFmpeg for "
                  << width << "x" << height << " " << (use_yuv444 ? "YUV444P" : "NV12")
                  << (is_cbr ? "with CBR:" + std::to_string(bitrate_kbps) : " with QP: " + std::to_string(qp)) << "." << std::endl;
    }
    return true;
}

/**
 * @brief Encodes a full YUV frame using the initialized VA-API session.
 * This function is thread-safe. It takes YUV plane data, transfers it from
 * system memory to a hardware surface on the GPU, submits it to the encoder,
 * and retrieves the resulting H.264 bitstream packet. The encoded data is
 * packaged into a StripeEncodeResult with a prepended 10-byte custom header.
 * @param width The width of the input frame.
 * @param height The height of the input frame.
 * @param fps The target frames per second (used for PTS calculation).
 * @param y_plane Pointer to the start of the Y plane data.
 * @param y_stride Stride in bytes for the Y plane.
 * @param u_plane Pointer to the start of the U plane (for I444) or interleaved
 *                UV plane (for NV12).
 * @param u_stride Stride in bytes for the U or UV plane.
 * @param v_plane Pointer to the start of the V plane (for I444); should be
 *                nullptr for NV12.
 * @param v_stride Stride in bytes for the V plane.
 * @param is_i444 True if the input format is YUV444P, false for NV12.
 * @param frame_counter The unique identifier for the current frame.
 * @param force_idr_frame If true, flags the frame as a keyframe (IDR).
 * @return A StripeEncodeResult containing the encoded H.264 data. On failure
 *         or if no packet is output, the result may be empty.
 * @throws std::runtime_error if a critical libav API call fails.
 */
StripeEncodeResult ScreenCaptureModule::encode_fullframe_vaapi(int width, int height, double fps,
                                          const uint8_t* y_plane, int y_stride,
                                          const uint8_t* u_plane, int u_stride,
                                          const uint8_t* v_plane, int v_stride,
                                          bool is_i444,
                                          int frame_counter,
                                          bool force_idr_frame) {
    std::lock_guard<std::mutex> lock(vaapi_mutex_);
    if (!vaapi_state_.initialized) {
        throw std::runtime_error("VAAPI_ENCODE_FATAL: Not initialized.");
    }
    int ret = av_hwframe_get_buffer(vaapi_state_.hw_frames_ctx, vaapi_state_.hw_frame, 0);
    if (ret < 0) {
        throw std::runtime_error("VAAPI_ENCODE_ERROR: Failed to get hardware frame from pool: " + std::to_string(ret));
    }
    AVFrame *tmp_sw_frame = av_frame_alloc();
    if (!tmp_sw_frame) {
        av_frame_unref(vaapi_state_.hw_frame);
        throw std::runtime_error("VAAPI_ENCODE_ERROR: Failed to allocate temporary mapping frame.");
    }
    ret = av_hwframe_map(tmp_sw_frame, vaapi_state_.hw_frame, AV_HWFRAME_MAP_WRITE);
    if (ret < 0) {
        av_frame_free(&tmp_sw_frame);
        av_frame_unref(vaapi_state_.hw_frame);
        throw std::runtime_error("VAAPI_ENCODE_ERROR: Failed to map hardware frame for writing: " + std::to_string(ret));
    }
    if (is_i444) {
        libyuv::CopyPlane(y_plane, y_stride, tmp_sw_frame->data[0], tmp_sw_frame->linesize[0], width, height);
        libyuv::CopyPlane(u_plane, u_stride, tmp_sw_frame->data[1], tmp_sw_frame->linesize[1], width, height);
        libyuv::CopyPlane(v_plane, v_stride, tmp_sw_frame->data[2], tmp_sw_frame->linesize[2], width, height);
    } else {
        libyuv::CopyPlane(y_plane, y_stride, tmp_sw_frame->data[0], tmp_sw_frame->linesize[0], width, height);
        libyuv::CopyPlane(u_plane, u_stride, tmp_sw_frame->data[1], tmp_sw_frame->linesize[1], width, height / 2);
    }
    av_frame_unref(tmp_sw_frame);
    av_frame_free(&tmp_sw_frame);
    vaapi_state_.hw_frame->pts = vaapi_state_.frame_count++;
    if (force_idr_frame) {
        vaapi_state_.hw_frame->pict_type = AV_PICTURE_TYPE_I;
    } else {
        vaapi_state_.hw_frame->pict_type = AV_PICTURE_TYPE_NONE;
    }
    ret = avcodec_send_frame(vaapi_state_.codec_ctx, vaapi_state_.hw_frame);
    av_frame_unref(vaapi_state_.hw_frame);
    if (ret < 0) {
        throw std::runtime_error("VAAPI_ENCODE_ERROR: Failed to send frame to encoder: " + std::to_string(ret));
    }
    while (true) {
        ret = avcodec_receive_packet(vaapi_state_.codec_ctx, vaapi_state_.packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            return {};
        } else if (ret < 0) {
            throw std::runtime_error("VAAPI_ENCODE_ERROR: Failed to receive packet from encoder: " + std::to_string(ret));
        }
        StripeEncodeResult result;
        result.type = StripeDataType::H264;
        result.stripe_y_start = 0;
        result.stripe_height = height;
        result.frame_id = frame_counter;
        if (vaapi_state_.packet->size > 0) {
            const unsigned char TAG = 0x04;
            unsigned char type_hdr = (vaapi_state_.packet->flags & AV_PKT_FLAG_KEY) ? 0x01 : 0x00;
            int header_sz = emit_stripe_headers_.load(std::memory_order_relaxed) ? 10 : 0;
            result.data = new unsigned char[vaapi_state_.packet->size + header_sz];
            result.size = vaapi_state_.packet->size + header_sz;
            if (header_sz) {
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
            }
            std::memcpy(result.data + header_sz, vaapi_state_.packet->data, vaapi_state_.packet->size);
        }
        av_packet_unref(vaapi_state_.packet);
        return result;
    }
}

/**
 * @brief Loads a watermark image from disk into memory.
 * This function is thread-safe. It reads the image file specified by the
 * internal watermark path setting using the stb_image library. It then
 * converts the pixel data to a 32-bit ARGB format suitable for fast
 * alpha blending in the `overlay_image` function.
 */
void ScreenCaptureModule::load_watermark_image() {
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
 * @brief Overlays a source image onto a destination frame with alpha blending.
 * This function iterates through the pixels of the source image and blends
 * them onto the destination frame buffer at the specified coordinates. It
 * handles transparency based on the alpha channel of the source image.
 * @param image_height Height of the source image to overlay.
 * @param image_width Width of the source image to overlay.
 * @param image_ptr Pointer to the source image data (32-bit ARGB format).
 * @param image_x The X-coordinate on the destination frame to place the top-left of the source image.
 * @param image_y The Y-coordinate on the destination frame to place the top-left of the source image.
 * @param frame_height Height of the destination frame buffer.
 * @param frame_width Width of the destination frame buffer.
 * @param frame_ptr Pointer to the destination frame buffer data (BGR or BGRX format).
 * @param frame_stride_bytes The stride (bytes per row) of the destination frame.
 * @param frame_bytes_per_pixel The bytes per pixel of the destination frame.
 */
void ScreenCaptureModule::overlay_image(int image_height, int image_width, const uint32_t *image_ptr,
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

/**
 * @brief The main function for the screen capture thread.
 * This loop continuously captures the screen at the target FPS. It handles
 * settings changes, screen capture via XShm, optional cursor and watermark
 * overlaying, color space conversion (BGRX to YUV), damage detection via
 * hashing, and dispatching encoding tasks to a thread pool. It then collects
 * the encoded results and invokes the user-provided callback.
 */
void ScreenCaptureModule::capture_loop() {
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
    int local_current_h264_paintover_crf;
    int local_current_h264_paintover_burst_frames;
    bool local_current_h264_fullcolor;
    bool local_current_h264_fullframe;
    bool local_current_h264_streaming_mode;
    OutputMode local_current_output_mode;
    bool local_current_capture_cursor;
    int local_vaapi_render_node_index;
    std::string local_vaapi_render_node_path;
    int xfixes_event_base = 0;
    int xfixes_error_base = 0;
    std::string local_watermark_path_setting;
    WatermarkLocation local_watermark_location_setting;
    bool local_use_cpu;
    bool local_debug_logging;
    bool local_current_h264_cbr_mode;
    int local_current_h264_bitrate_kbps;
    int local_current_h264_vbv_buffer_size_kb;
    bool local_current_auto_adjust_screen_capture_size;

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
      local_current_h264_paintover_crf = h264_paintover_crf;
      local_current_h264_paintover_burst_frames = h264_paintover_burst_frames;
      local_current_h264_fullcolor = h264_fullcolor;
      local_current_h264_fullframe = h264_fullframe;
      local_current_h264_streaming_mode = h264_streaming_mode;
      local_current_capture_cursor = capture_cursor;
      local_vaapi_render_node_index = vaapi_render_node_index;
      local_vaapi_render_node_path = vaapi_render_node_path_;
      local_use_cpu = use_cpu;
      local_debug_logging = debug_logging;
      local_watermark_path_setting = watermark_path_internal;
      local_watermark_location_setting = watermark_location_internal;
      local_current_h264_cbr_mode = h264_cbr_mode;
      local_current_h264_bitrate_kbps = h264_bitrate_kbps;
      local_current_h264_vbv_buffer_size_kb = h264_vbv_buffer_size_kb;
      local_current_auto_adjust_screen_capture_size = auto_adjust_screen_capture_size;
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

    this->vaapi_operational = false;
    this->nvenc_operational = false;

    if (!local_use_cpu && local_vaapi_render_node_index >= 0 &&
        local_current_output_mode == OutputMode::H264 && local_current_h264_fullframe) {
        if (this->initialize_vaapi_encoder(local_vaapi_render_node_path,
                                      local_capture_width_actual,
                                      local_capture_height_actual,
                                      local_current_h264_crf,
                                      local_current_h264_fullcolor,
                                      local_current_h264_cbr_mode,
                                      local_current_h264_bitrate_kbps,
                                      local_current_h264_vbv_buffer_size_kb)) {
            this->vaapi_operational = true;
            this->vaapi_force_next_idr_ = true;
            std::cout << "VAAPI Encoder Initialized successfully." << std::endl;
        } else {
            std::cerr << "VAAPI Encoder initialization failed. Falling back to CPU." << std::endl;
            local_use_cpu = true;
            std::lock_guard<std::mutex> lock(settings_mutex);
            this->use_cpu = true;
        }
    } else {
      if (!local_use_cpu && this->is_nvidia_system_detected &&
          local_current_output_mode == OutputMode::H264 && local_current_h264_fullframe) {
        if (this->initialize_nvenc_encoder(local_capture_width_actual,
                                     local_capture_height_actual,
                                     local_current_h264_crf,
                                     local_current_target_fps,
                                     local_current_h264_fullcolor,
                                     local_current_h264_cbr_mode,
                                     local_current_h264_bitrate_kbps,
                                     local_current_h264_vbv_buffer_size_kb)) {
          this->nvenc_operational = true;
          this->nvenc_force_next_idr_ = true;
          std::cout << "NVENC Encoder Initialized successfully." << std::endl;
        } else {
          std::cerr << "NVENC Encoder initialization failed. Falling back to x264." << std::endl;
          local_use_cpu = true;
          std::lock_guard<std::mutex> lock(settings_mutex);
          this->use_cpu = true;
        }
      } else {
          if (!this->nvenc_operational && this->nvenc_state_.initialized) {
            this->reset_nvenc_encoder();
          }
      }
    }

    std::chrono::duration < double > target_frame_duration_seconds =
      std::chrono::duration < double > (1.0 / (local_current_target_fps > 0.0 ? local_current_target_fps : 1.0));

    auto next_frame_time =
      std::chrono::high_resolution_clock::now() + target_frame_duration_seconds;

    const int MAX_ATTACH_ATTEMPTS = 5;
    const int RETRY_BACKOFF_MS = 500;
    char* display_env = std::getenv("DISPLAY");
    const char* display_name = display_env ? display_env : ":0";
    Display* display = XOpenDisplay(display_name);

    if (!display) {
      std::cerr << "Error: Failed to open X display " << display_name << std::endl;
      return;
    }

    Window root_window = DefaultRootWindow(display);
    int screen = DefaultScreen(display);
    XWindowAttributes attributes;

    if (XGetWindowAttributes(display, root_window, &attributes)) {
      if (local_current_auto_adjust_screen_capture_size) {
          std::cout << "[pixelflux] auto_adjust_screen_capture_size is enabled, ignoring requested capture size "
                    << local_capture_width_actual << "x" << local_capture_height_actual
                    << " and resetting x and y offset to 0" << std::endl;
          
          local_capture_width_actual = attributes.width;
          local_capture_height_actual = attributes.height;
          local_capture_x_offset = 0;
          local_capture_y_offset = 0;

          std::lock_guard<std::mutex> lock(settings_mutex);
          this->capture_width = attributes.width;
          this->capture_height = attributes.height;
      } else {
          if (local_capture_width_actual > attributes.width) {
          local_capture_width_actual = attributes.width;
          local_capture_x_offset = 0;
        }
        if (local_capture_height_actual > attributes.height) {
            local_capture_height_actual = attributes.height;
            local_capture_y_offset = 0;
        }
        if (local_capture_x_offset + local_capture_width_actual > attributes.width) {
            local_capture_x_offset = attributes.width - local_capture_width_actual;
        }
        if (local_capture_y_offset + local_capture_height_actual > attributes.height) {
            local_capture_y_offset = attributes.height - local_capture_height_actual;
        }
        if (local_capture_x_offset < 0) local_capture_x_offset = 0;
        if (local_capture_y_offset < 0) local_capture_y_offset = 0;
      }
    }

    this->yuv_planes_are_i444_ = local_current_h264_fullcolor;
    if (local_current_output_mode == OutputMode::H264) {
        bool use_nv12_planes = !local_use_cpu && local_current_h264_fullframe && !local_current_h264_fullcolor &&
                       ((this->is_nvidia_system_detected && local_vaapi_render_node_index < 0) || (local_vaapi_render_node_index >= 0));
        size_t y_plane_size = static_cast<size_t>(local_capture_width_actual) *
                              local_capture_height_actual;
        full_frame_y_plane_.assign(y_plane_size, 0);
        full_frame_y_stride_ = local_capture_width_actual;

        if (this->yuv_planes_are_i444_) {
            full_frame_u_plane_.assign(y_plane_size, 0);
            full_frame_v_plane_.assign(y_plane_size, 0);
            full_frame_u_stride_ = local_capture_width_actual;
            full_frame_v_stride_ = local_capture_width_actual;
        } else if (use_nv12_planes) { 
            size_t uv_plane_size = static_cast<size_t>(local_capture_width_actual) * (static_cast<size_t>(local_capture_height_actual) / 2);
            full_frame_u_plane_.assign(uv_plane_size, 0);
            full_frame_u_stride_ = local_capture_width_actual;
            full_frame_v_plane_.clear();
            full_frame_v_stride_ = 0;
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

    if (!local_watermark_path_setting.empty() && local_watermark_location_setting != WatermarkLocation::NONE) {
        load_watermark_image();
    }

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
    XImage* shm_image = nullptr;
    bool shm_setup_complete = false;

    for (int attempt = 1; attempt <= MAX_ATTACH_ATTEMPTS; ++attempt) {
        memset(&shminfo, 0, sizeof(shminfo));
        shm_image = XShmCreateImage(display, DefaultVisual(display, screen), DefaultDepth(display, screen),
                                    ZPixmap, nullptr, &shminfo, local_capture_width_actual,
                                    local_capture_height_actual);
        if (!shm_image) {
            std::cerr << "Attempt " << attempt << ": XShmCreateImage failed." << std::endl;
            if (attempt < MAX_ATTACH_ATTEMPTS) std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_BACKOFF_MS));
            continue;
        }

        shminfo.shmid = shmget(IPC_PRIVATE, static_cast<size_t>(shm_image->bytes_per_line) * shm_image->height, IPC_CREAT | 0600);
        if (shminfo.shmid < 0) {
            perror("shmget");
            XDestroyImage(shm_image);
            if (attempt < MAX_ATTACH_ATTEMPTS) std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_BACKOFF_MS));
            continue;
        }

        shminfo.shmaddr = (char*)shmat(shminfo.shmid, nullptr, 0);
        if (shminfo.shmaddr == (char*)-1) {
            perror("shmat");
            shmctl(shminfo.shmid, IPC_RMID, 0);
            XDestroyImage(shm_image);
            if (attempt < MAX_ATTACH_ATTEMPTS) std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_BACKOFF_MS));
            continue;
        }

        shminfo.readOnly = False;
        shm_image->data = shminfo.shmaddr;
        g_shm_attach_failed = false;
        XErrorHandler old_handler = XSetErrorHandler(shm_attach_error_handler);
        XShmAttach(display, &shminfo);
        XSync(display, False);
        XSetErrorHandler(old_handler);

        if (g_shm_attach_failed) {
            std::cerr << "Attempt " << attempt << "/" << MAX_ATTACH_ATTEMPTS << ": XShmAttach failed with an X server error." << std::endl;
            shmdt(shminfo.shmaddr);
            shmctl(shminfo.shmid, IPC_RMID, 0);
            XDestroyImage(shm_image);
            if (attempt < MAX_ATTACH_ATTEMPTS) {
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_BACKOFF_MS));
            }
            continue;
        }
        
        shm_setup_complete = true;
        break;
    }

    if (!shm_setup_complete) {
        std::cerr << "ERROR: Failed to set up XShm after " << MAX_ATTACH_ATTEMPTS << " attempts. Exiting capture thread." << std::endl;
        if (display) {
            XCloseDisplay(display);
            display = nullptr;
        }
        return;
    }

    std::cout << "XShm setup complete for " << local_capture_width_actual
              << "x" << local_capture_height_actual << "." << std::endl;

    int num_cores = std::max(1, (int)std::thread::hardware_concurrency());
    std::cout << "CPU cores available: " << num_cores << std::endl;
    int num_stripes_config = num_cores;

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
    std::stringstream settings_ss;
    settings_ss << "Stream settings active -> Res: " << local_capture_width_actual << "x"
                << local_capture_height_actual
                << " | FPS: " << std::fixed << std::setprecision(1) << local_current_target_fps
                << " | Stripes: " << N_processing_stripes;
    if (local_current_output_mode == OutputMode::JPEG) {
        settings_ss << " | Mode: JPEG";
        settings_ss << " | Quality: " << local_current_jpeg_quality;
        if (local_current_use_paint_over_quality) {
            settings_ss << " | PaintOver Q: " << local_current_paint_over_jpeg_quality
                        << " (Trigger: " << local_current_paint_over_trigger_frames << "f)";
        }
    } else {
        std::string encoder_type = "CPU";
        if (this->vaapi_operational) encoder_type = "VAAPI";
        else if (this->nvenc_operational) encoder_type = "NVENC";
        settings_ss << " | Mode: H264 (" << encoder_type << ")";
        settings_ss << (local_current_h264_fullframe ? " FullFrame" : " Striped");
        if (local_current_h264_streaming_mode) settings_ss << " Streaming";
        if (!local_current_h264_cbr_mode) settings_ss << " | CRF: " << local_current_h264_crf;
        else settings_ss << " | CBR: " << local_current_h264_bitrate_kbps;
        if (local_current_use_paint_over_quality) {
            settings_ss << " | PaintOver CRF: " << local_current_h264_paintover_crf
                        << " (Burst: " << local_current_h264_paintover_burst_frames << "f)";
        }
        settings_ss << " | Colorspace: " << (local_current_h264_fullcolor ? "I444 (Full Range)" : "I420 (Limited Range)");
    }
    settings_ss << " | Damage Thresh: " << local_current_damage_block_threshold << "f"
                << " | Damage Dur: " << local_current_damage_block_duration << "f";
    std::cout << settings_ss.str() << std::endl;

    std::vector<uint64_t> previous_hashes(num_stripes_config, 0);
    std::vector<int> no_motion_frame_counts(num_stripes_config, 0);
    std::vector<bool> paint_over_sent(num_stripes_config, false);
    std::vector<int> current_jpeg_qualities(num_stripes_config);
    std::vector<int> consecutive_stripe_changes(num_stripes_config, 0);
    std::vector<bool> stripe_is_in_damage_block(num_stripes_config, false);
    std::vector<int> stripe_damage_block_frames_remaining(num_stripes_config, 0);
    std::vector<uint64_t> stripe_hash_at_damage_block_start(num_stripes_config, 0);
    std::vector<int> h264_paintover_burst_frames_remaining(num_stripes_config, 0);

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

      // Re-anchor if we fell behind, so lateness can't accumulate into a burst.
      if (next_frame_time < current_loop_iter_start_time) {
        next_frame_time = current_loop_iter_start_time + target_frame_duration_seconds;
      }

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
          target_frame_duration_seconds = std::chrono::duration < double > (1.0 / (local_current_target_fps > 0.0 ? local_current_target_fps : 1.0));
          next_frame_time = intended_current_frame_time + target_frame_duration_seconds;
          // intended_current_frame_time predates the lateness re-anchor above, so on a
          // behind-schedule frame this could rewind next_frame_time into the past and
          // trigger a catch-up burst. Re-apply the same re-anchor to preserve it.
          if (next_frame_time < current_loop_iter_start_time) {
            next_frame_time = current_loop_iter_start_time + target_frame_duration_seconds;
          }
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
        local_current_h264_paintover_crf = h264_paintover_crf;
        local_current_h264_paintover_burst_frames = h264_paintover_burst_frames;
        local_current_h264_fullcolor = h264_fullcolor;
        local_current_h264_fullframe = h264_fullframe;
        local_current_h264_streaming_mode = h264_streaming_mode;
        local_current_capture_cursor = capture_cursor;
        local_vaapi_render_node_index = vaapi_render_node_index;
      local_vaapi_render_node_path = vaapi_render_node_path_;
        local_watermark_path_setting = watermark_path_internal;
        local_watermark_location_setting = watermark_location_internal;
        local_use_cpu = use_cpu;
        local_debug_logging = debug_logging;
        local_current_h264_cbr_mode = h264_cbr_mode;
        local_current_h264_bitrate_kbps = h264_bitrate_kbps;
        local_current_h264_vbv_buffer_size_kb = h264_vbv_buffer_size_kb;
        local_current_auto_adjust_screen_capture_size = auto_adjust_screen_capture_size;
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

      if (local_current_auto_adjust_screen_capture_size) {
        XWindowAttributes attributes;
        if (XGetWindowAttributes(display, root_window, &attributes)) {
          if (local_capture_width_actual != attributes.width ||
              local_capture_height_actual != attributes.height) {
                if (debug_logging) {
                    std::cout << "[pixelflux] Auto-adjusting capture size from "
                              << local_capture_width_actual << "x"
                              << local_capture_height_actual << " to "
                              << attributes.width << "x"
                              << attributes.height << std::endl;
                }
            local_capture_width_actual = attributes.width;
            local_capture_height_actual = attributes.height;
            local_capture_x_offset = 0;
            local_capture_y_offset = 0;

            std::lock_guard<std::mutex> lock(settings_mutex);
            capture_width = local_capture_width_actual;
            capture_height = local_capture_height_actual;
            capture_x = 0;
            capture_y = 0;
          }
        }
      }

      // H.264 4:2:0 needs even dimensions; clamp before SHM/plane (re)allocation.
      if (local_current_output_mode == OutputMode::H264) {
        local_capture_width_actual &= ~1;
        local_capture_height_actual &= ~1;
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
            bool use_nv12_planes = !local_use_cpu && local_current_h264_fullframe && !local_current_h264_fullcolor &&
                           ((this->is_nvidia_system_detected && local_vaapi_render_node_index < 0) || (local_vaapi_render_node_index >= 0));

            size_t y_plane_size = static_cast<size_t>(local_capture_width_actual) *
                                  local_capture_height_actual;
            full_frame_y_plane_.assign(y_plane_size, 0);
            full_frame_y_stride_ = local_capture_width_actual;

            if (this->yuv_planes_are_i444_) {
                full_frame_u_plane_.assign(y_plane_size, 0);
                full_frame_v_plane_.assign(y_plane_size, 0);
                full_frame_u_stride_ = local_capture_width_actual;
                full_frame_v_stride_ = local_capture_width_actual;
            } else if (use_nv12_planes) {
                size_t uv_plane_size = static_cast<size_t>(local_capture_width_actual) * (static_cast<size_t>(local_capture_height_actual) / 2);
                full_frame_u_plane_.assign(uv_plane_size, 0);
                full_frame_u_stride_ = local_capture_width_actual;
                full_frame_v_plane_.clear();
                full_frame_v_stride_ = 0;
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
        h264_minimal_store_.reset();
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
            bool use_nv12_for_hw_encoder = (this->nvenc_operational || this->vaapi_operational) && !this->yuv_planes_are_i444_;

            if (use_nv12_for_hw_encoder) {
                // Default to GPU (CUDA/NVRTC) color conversion in NVENC mode when
                // libnvrtc is present; fall back to libyuv on absence or any error.
                bool converted = false;
                nvenc_state_.dev_input_base = 0;  // host path unless device conversion sets it
                if (this->nvenc_operational) {
                    if (cuda_convert::nvenc_device_input_enabled()) {
                        unsigned long long dev_base = 0; int dev_pitch = 0;
                        converted = cuda_convert::argb_to_nv12_device(
                            nvenc_state_.cuda_context, shm_data_ptr, shm_stride_bytes,
                            full_frame_y_plane_.data(), full_frame_y_stride_,
                            full_frame_u_plane_.data(), full_frame_u_stride_,
                            local_capture_width_actual, local_capture_height_actual, &dev_base, &dev_pitch);
                        if (converted) { nvenc_state_.dev_input_base = dev_base; nvenc_state_.dev_input_pitch = dev_pitch; }
                    } else {
                        converted = cuda_convert::argb_to_nv12(
                            nvenc_state_.cuda_context, shm_data_ptr, shm_stride_bytes,
                            full_frame_y_plane_.data(), full_frame_y_stride_,
                            full_frame_u_plane_.data(), full_frame_u_stride_,
                            local_capture_width_actual, local_capture_height_actual);
                    }
                }
                if (!converted) {
                    libyuv::ARGBToNV12(shm_data_ptr, shm_stride_bytes,
                                       full_frame_y_plane_.data(), full_frame_y_stride_,
                                       full_frame_u_plane_.data(), full_frame_u_stride_,
                                       local_capture_width_actual, local_capture_height_actual);
                }
            } else if (this->yuv_planes_are_i444_) {
                libyuv::ARGBToI444(shm_data_ptr, shm_stride_bytes,
                                   full_frame_y_plane_.data(), full_frame_y_stride_,
                                   full_frame_u_plane_.data(), full_frame_u_stride_,
                                   full_frame_v_plane_.data(), full_frame_v_stride_,
                                   local_capture_width_actual, local_capture_height_actual);
            } else {
                // A runtime HW-encoder fallback can leave NV12-sized planes (empty V)
                // while we take the I420 path; re-size to I420 before converting.
                if (full_frame_v_plane_.empty()) {
                    size_t chroma_plane_size =
                        (static_cast<size_t>(local_capture_width_actual) / 2) *
                        (static_cast<size_t>(local_capture_height_actual) / 2);
                    full_frame_u_plane_.assign(chroma_plane_size, 0);
                    full_frame_v_plane_.assign(chroma_plane_size, 0);
                    full_frame_u_stride_ = local_capture_width_actual / 2;
                    full_frame_v_stride_ = local_capture_width_actual / 2;
                }
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
            h264_paintover_burst_frames_remaining.assign(N_processing_stripes, 0);

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

          auto calculate_current_hash = [&]() {
              if (local_current_output_mode == OutputMode::H264) {
                  const uint8_t* y_plane_stripe_ptr = full_frame_y_plane_.data() +
                      static_cast<size_t>(start_y) * full_frame_y_stride_;
                  const uint8_t* u_plane_stripe_ptr = full_frame_u_plane_.data() +
                      (static_cast<size_t>(this->yuv_planes_are_i444_ ?
                      start_y : (start_y / 2)) * full_frame_u_stride_);

                  bool use_nv12_path = this->nvenc_operational && !this->yuv_planes_are_i444_;

                  const uint8_t* v_plane_stripe_ptr = use_nv12_path ? nullptr :
                      (full_frame_v_plane_.empty() ? nullptr : full_frame_v_plane_.data() +
                      (static_cast<size_t>(this->yuv_planes_are_i444_ ?
                      start_y : (start_y / 2)) * full_frame_v_stride_));

                  int v_stride = use_nv12_path ? 0 : full_frame_v_stride_;

                  return calculate_yuv_stripe_hash(
                      y_plane_stripe_ptr, full_frame_y_stride_,
                      u_plane_stripe_ptr, full_frame_u_stride_,
                      v_plane_stripe_ptr, v_stride,
                      local_capture_width_actual, current_stripe_height,
                      !this->yuv_planes_are_i444_, local_current_h264_fullframe);
              } else {
                  const unsigned char* shm_stripe_start_ptr = shm_data_ptr +
                      static_cast<size_t>(start_y) * shm_stride_bytes;
                  return calculate_bgr_stripe_hash_from_shm(
                      shm_stripe_start_ptr, shm_stride_bytes,
                      local_capture_width_actual, current_stripe_height,
                      shm_bytes_per_pixel);
              }
          };

          uint64_t current_hash = 0;
          bool hash_calculated_this_iteration = false;
          bool send_this_stripe = false;
          bool force_idr_for_paintover = false;
          int crf_for_encode = local_current_h264_crf;
          if (local_current_output_mode == OutputMode::H264 && h264_paintover_burst_frames_remaining[i] > 0) {
              send_this_stripe = true;
              crf_for_encode = local_current_h264_paintover_crf;
              h264_paintover_burst_frames_remaining[i]--;
              current_hash = calculate_current_hash();
              hash_calculated_this_iteration = true;
              if (current_hash != previous_hashes[i]) {
                  h264_paintover_burst_frames_remaining[i] = 0;
                  paint_over_sent[i] = false;
                  crf_for_encode = local_current_h264_crf;
                  consecutive_stripe_changes[i] = 1;
              }
          }
          else if (local_current_output_mode == OutputMode::H264 && local_current_h264_streaming_mode) {
              send_this_stripe = true;
          }
          else if (stripe_is_in_damage_block[i]) {
              send_this_stripe = true;
              stripe_damage_block_frames_remaining[i]--;
              if (stripe_damage_block_frames_remaining[i] <= 0) {
                  current_hash = calculate_current_hash();
                  hash_calculated_this_iteration = true;

                  if (current_hash != stripe_hash_at_damage_block_start[i]) {
                      stripe_damage_block_frames_remaining[i] = local_current_damage_block_duration;
                      stripe_hash_at_damage_block_start[i] = current_hash;
                  } else {
                      stripe_is_in_damage_block[i] = false;
                      consecutive_stripe_changes[i] = 0;
                      no_motion_frame_counts[i] = 1;
                  }
              }
          }
          else {
              current_hash = calculate_current_hash();
              hash_calculated_this_iteration = true;
              if (current_hash != previous_hashes[i]) {
                  send_this_stripe = true;
                  no_motion_frame_counts[i] = 0;
                  paint_over_sent[i] = false;
                  consecutive_stripe_changes[i]++;
                  current_jpeg_qualities[i] = local_current_jpeg_quality;
                  h264_paintover_burst_frames_remaining[i] = 0;
                  if (consecutive_stripe_changes[i] >= local_current_damage_block_threshold) {
                      stripe_is_in_damage_block[i] = true;
                      stripe_damage_block_frames_remaining[i] = local_current_damage_block_duration;
                      stripe_hash_at_damage_block_start[i] = current_hash;
                  }
              } else {
                  send_this_stripe = false;
                  consecutive_stripe_changes[i] = 0;
                  no_motion_frame_counts[i]++;
                  if (no_motion_frame_counts[i] >= local_current_paint_over_trigger_frames && !paint_over_sent[i]) {
                      if (local_current_output_mode == OutputMode::JPEG && 
                          local_current_use_paint_over_quality &&
                          local_current_paint_over_jpeg_quality > local_current_jpeg_quality) {
                          send_this_stripe = true;
                          current_jpeg_qualities[i] = local_current_paint_over_jpeg_quality;
                          paint_over_sent[i] = true;
                      } else if (local_current_output_mode == OutputMode::H264) {
                          if (local_current_use_paint_over_quality && local_current_h264_paintover_crf < local_current_h264_crf) {
                              send_this_stripe = true;
                              paint_over_sent[i] = true;
                              if (this->nvenc_operational) {
                                  this->nvenc_force_next_idr_ = true;
                                  h264_paintover_burst_frames_remaining[i] = local_current_h264_paintover_burst_frames - 1;
                              } else if (this->vaapi_operational) {
                                  this->vaapi_force_next_idr_ = true;
                                  h264_paintover_burst_frames_remaining[i] = 0;
                              } else {
                                  force_idr_for_paintover = true;
                                  crf_for_encode = local_current_h264_paintover_crf;
                                  h264_paintover_burst_frames_remaining[i] = local_current_h264_paintover_burst_frames - 1;
                              }
                          }
                      }
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
                int, int, int, int, const unsigned char*, int, int, int, int, bool)>
                task(encode_stripe_jpeg);
              futures.push_back(task.get_future());
              threads.push_back(std::thread(
                std::move(task), i, start_y, current_stripe_height,
                local_capture_width_actual,
                shm_data_ptr,
                shm_stride_bytes,
                shm_bytes_per_pixel,
                quality_to_use,
                this->frame_counter,
                this->emit_stripe_headers_.load(std::memory_order_relaxed)));
            } else {
              if (this->vaapi_operational) {
                std::packaged_task<StripeEncodeResult()> task([=, this]() {
                    bool force_idr = this->vaapi_force_next_idr_.exchange(false);
                    return this->encode_fullframe_vaapi(
                        local_capture_width_actual, local_capture_height_actual, local_current_target_fps,
                        full_frame_y_plane_.data(), full_frame_y_stride_,
                        full_frame_u_plane_.data(), full_frame_u_stride_,
                        this->yuv_planes_are_i444_ ? full_frame_v_plane_.data() : nullptr,
                        this->yuv_planes_are_i444_ ? full_frame_v_stride_ : 0,
                        this->yuv_planes_are_i444_,
                        this->frame_counter, force_idr
                    );
                });
                futures.push_back(task.get_future());
                threads.push_back(std::thread(std::move(task)));
              } else if (this->nvenc_operational) {
                int target_qp_for_frame = crf_for_encode;
                if (local_current_use_paint_over_quality && this->nvenc_force_next_idr_) {
                    target_qp_for_frame = local_current_h264_paintover_crf;
                }
                if (!this->initialize_nvenc_encoder(local_capture_width_actual,
                                              local_capture_height_actual,
                                              target_qp_for_frame,
                                              local_current_target_fps,
                                              local_current_h264_fullcolor,
                                              local_current_h264_cbr_mode,
                                              local_current_h264_bitrate_kbps,
                                              local_current_h264_vbv_buffer_size_kb)) {
                    std::cerr << "NVENC: Re-initialization for QP change failed. Disabling NVENC." << std::endl;
                    this->nvenc_operational = false;
                    this->reset_nvenc_encoder();
                    continue;
                }
                std::packaged_task<StripeEncodeResult()> task([=, this]() {
                    bool force_idr = this->nvenc_force_next_idr_.exchange(false);
                    return this->encode_fullframe_nvenc(
                        local_capture_width_actual, local_capture_height_actual,
                        full_frame_y_plane_.data(), full_frame_y_stride_,
                        full_frame_u_plane_.data(), full_frame_u_stride_,
                        this->yuv_planes_are_i444_ ? full_frame_v_plane_.data() : nullptr,
                        this->yuv_planes_are_i444_ ? full_frame_v_stride_ : 0,
                        this->yuv_planes_are_i444_, this->frame_counter, force_idr
                    );
                });
                futures.push_back(task.get_future());
                threads.push_back(std::thread(std::move(task)));
              } else {
                if (force_idr_for_paintover) {
                  std::lock_guard<std::mutex> lock(h264_minimal_store_.store_mutex);
                  h264_minimal_store_.ensure_size(i);
                  if (i < static_cast<int>(h264_minimal_store_.force_idr_flags.size())) {
                      h264_minimal_store_.force_idr_flags[i] = true;
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
                
                bool force_idr = this->force_next_idr_.exchange(false);
                std::packaged_task<StripeEncodeResult(
                  MinimalEncoderStore&, int, int, int, int, const uint8_t*, int,
                  const uint8_t*, int, const uint8_t*, int, bool, int, int, int,
                  bool, int, bool, bool, int, int, bool)>
                  task(encode_stripe_h264);
                futures.push_back(task.get_future());
                threads.push_back(std::thread(
                  std::move(task), std::ref(h264_minimal_store_), i, start_y, current_stripe_height,
                  local_capture_width_actual,
                  y_plane_for_thread, full_frame_y_stride_,
                  u_plane_for_thread, full_frame_u_stride_,
                  v_plane_for_thread, full_frame_v_stride_,
                  this->yuv_planes_are_i444_,
                  this->frame_counter,
                  crf_for_encode,
                  derived_h264_colorspace_setting,
                  derived_h264_use_full_range,
                  local_current_h264_streaming_mode,
                  force_idr,
                  local_current_h264_cbr_mode,
                  local_current_h264_bitrate_kbps,
                  local_current_h264_vbv_buffer_size_kb,
                  this->emit_stripe_headers_.load(std::memory_order_relaxed)
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
                this->reset_nvenc_encoder();
                this->nvenc_force_next_idr_ = true;
            } else if (std::string(e.what()).find("VAAPI_") != std::string::npos) {
                std::cerr << "ENCODE_THREAD_ERROR: " << e.what() << std::endl;
                std::cerr << "Disabling VAAPI for this session due to runtime error." << std::endl;
                this->vaapi_operational = false;
                this->reset_vaapi_encoder();
                this->vaapi_force_next_idr_ = true;
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
            if (deferred_free_.load(std::memory_order_relaxed)) {
              // Ownership handed to Python, which frees the buffer after its async
              // send completes. Detach defensively; clear() below does not free
              // data anyway (StripeEncodeResult has no data-freeing destructor).
              result.data = nullptr;
              result.size = 0;
            }
          } else if (result.data != nullptr) {
            // No callback consumed this buffer (callback==nullptr, or empty result):
            // StripeEncodeResult has no data-freeing destructor and clear() below
            // won't release it, so free the new[]'d buffer here to avoid a leak.
            delete[] result.data;
            result.data = nullptr;
            result.size = 0;
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

        if (local_debug_logging && output_elapsed_time_log.count() >= 1) {
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
                    << (local_current_output_mode == OutputMode::H264 && local_current_h264_cbr_mode ?
                        " CBR:" + std::to_string(local_current_h264_bitrate_kbps): 
                        (local_current_output_mode == OutputMode::H264
                        ? " CRF" + std::to_string(local_current_h264_crf)
                        : " Q:" + std::to_string(local_current_jpeg_quality)))
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
        XSync(display, False);
        XCloseDisplay(display);
        display = nullptr;
    }
    std::cout << "Capture loop stopped. X resources released." << std::endl;
}

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
  int frame_counter,
  bool emit_header) {
  StripeEncodeResult result;
  result.type = StripeDataType::JPEG;
  result.stripe_y_start = stripe_y_start;
  result.stripe_height = stripe_height;
  result.frame_id = frame_counter;

  if (!shm_data_base || stripe_height <= 0 || capture_width_actual <= 0 ||
      (shm_bytes_per_pixel != 3 && shm_bytes_per_pixel != 4)) {
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

  if (shm_bytes_per_pixel == 4) {
    cinfo.input_components = 4;
    cinfo.in_color_space = JCS_EXT_BGRX;
  } else {
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_EXT_BGR;
  }

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, jpeg_quality, TRUE);

  unsigned char* jpeg_buffer = nullptr;
  unsigned long jpeg_size_temp = 0;
  jpeg_mem_dest(&cinfo, &jpeg_buffer, &jpeg_size_temp);

  jpeg_start_compress(&cinfo, TRUE);

  JSAMPROW row_pointer[1];
  for (int y_in_stripe = 0; y_in_stripe < stripe_height; ++y_in_stripe) {
    const unsigned char* shm_current_row_in_full_frame_ptr =
        shm_data_base + static_cast<size_t>(stripe_y_start + y_in_stripe) * shm_stride_bytes;
    row_pointer[0] = (JSAMPROW)shm_current_row_in_full_frame_ptr;
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);

  if (jpeg_size_temp > 0 && jpeg_buffer) {
    int padding_size = emit_header ? 4 : 0;
    result.data = new (std::nothrow) unsigned char[jpeg_size_temp + padding_size];
    if (!result.data) {
      std::cerr << "JPEG T" << thread_id
                << ": Failed to allocate memory for JPEG output." << std::endl;
      jpeg_destroy_compress(&cinfo);
      if (jpeg_buffer)
        free(jpeg_buffer);
      result.type = StripeDataType::UNKNOWN;
      return result;
    }

    if (padding_size) {
      uint16_t frame_counter_net = htons(static_cast<uint16_t>(frame_counter % 65536));
      uint16_t stripe_y_start_net = htons(static_cast<uint16_t>(stripe_y_start));
      std::memcpy(result.data, &frame_counter_net, 2);
      std::memcpy(result.data + 2, &stripe_y_start_net, 2);
    }
    std::memcpy(result.data + padding_size, jpeg_buffer, jpeg_size_temp);
    result.size = static_cast<int>(jpeg_size_temp) + padding_size;
  } else {
    result.size = 0;
    result.data = nullptr;
  }

  jpeg_destroy_compress(&cinfo);
  if (jpeg_buffer) {
    free(jpeg_buffer);
  }
  return result;
}

/**
 * @brief Encodes a horizontal YUV stripe into an H.264 bitstream using x264.
 *
 * Manages a thread-specific x264 encoder instance from the provided encoder store.
 * The encoder is re-initialized if input parameters
 * such as resolution or colorspace change. The CRF or bitrate can be reconfigured
 * between frames without a full re-initialization.
 *
 * The output NAL units are packaged into a StripeEncodeResult with a custom
 * 10-byte header.
 *
 * @param h264_minimal_store A reference to the encoder store for this instance.
 * @param thread_id         Identifier for the calling thread, used to select a
 *                          dedicated encoder instance.
 * @param stripe_y_start    The Y-coordinate of the stripe's top edge.
 * @param stripe_height     Height of the stripe in pixels. Must be an even value.
 * @param capture_width_actual Width of the stripe in pixels. Must be an even value.
 * @param y_plane_stripe_start Pointer to the start of the Y plane data for this stripe.
 * @param y_stride          Stride in bytes for the Y plane.
 * @param u_plane_stripe_start Pointer to the start of the U plane data for this stripe.
 * @param u_stride          Stride in bytes for the U plane.
 * @param v_plane_stripe_start Pointer to the start of the V plane data for this stripe.
 * @param v_stride          Stride in bytes for the V plane.
 * @param is_i444_input     `true` for I444 colorspace, `false` for I420.
 * @param frame_counter     The frame number, used to set the picture's PTS.
 * @param current_crf_setting The target Constant Rate Factor (CRF) for CRF mode.
 * @param colorspace_setting Integer representing the colorspace (444 or 420).
 * @param use_full_range    If `true`, signals full-range color in the VUI.
 * @param h264_streaming_mode If `true`, enables streaming mode optimizations.
 * @param force_idr         If `true`, forces the encoder to generate an IDR frame.
 * @param is_cbr            If `true`, enables Constant Bitrate (CBR) mode.
 * @param h264_bitrate_kbps Target bitrate in kbps for CBR mode.
 * @param vbv_buffer_size_kb VBV buffer size in kb for CBR mode (0 for auto/default).
 * @return                  A `StripeEncodeResult` containing the encoded bitstream.
 *                          The `data` buffer is dynamically allocated and must be
 *                          freed by the caller.
 */
StripeEncodeResult encode_stripe_h264(
  MinimalEncoderStore& h264_minimal_store,
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
  bool use_full_range,
  bool h264_streaming_mode,
  bool force_idr,
  bool is_cbr,
  int h264_bitrate_kbps,
  int vbv_buffer_size_kb,
  bool emit_header) {

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
    std::lock_guard<std::mutex> lock(h264_minimal_store.store_mutex);
    h264_minimal_store.ensure_size(thread_id);

    bool is_first_init = !h264_minimal_store.initialized_flags[thread_id];
    bool dims_changed = !is_first_init &&
                        (h264_minimal_store.initialized_widths[thread_id] !=
                            capture_width_actual ||
                         h264_minimal_store.initialized_heights[thread_id] !=
                            stripe_height);
    bool cs_or_fr_changed = !is_first_init &&
                            (h264_minimal_store.initialized_csps[thread_id] !=
                                target_x264_csp ||
                             h264_minimal_store.initialized_colorspaces[thread_id] !=
                                colorspace_setting ||
                             h264_minimal_store.initialized_full_range_flags[thread_id] !=
                                use_full_range);
    bool rc_changed = !is_first_init && h264_minimal_store.initialized_cbr_flags[thread_id] != is_cbr;


    bool needs_crf_reinit = false;
    if (!is_first_init &&
        h264_minimal_store.initialized_crfs[thread_id] != current_crf_setting) {
        needs_crf_reinit = true;
    }
    
    bool needs_cbr_reinit = false;
    if (!is_first_init && is_cbr && h264_minimal_store.initialized_bitrates[thread_id] != h264_bitrate_kbps) {
        needs_cbr_reinit = true;
    }

    bool perform_full_reinit = is_first_init || dims_changed || cs_or_fr_changed || rc_changed;

    if (perform_full_reinit) {
      if (h264_minimal_store.encoders[thread_id]) {
        x264_encoder_close(h264_minimal_store.encoders[thread_id]);
        h264_minimal_store.encoders[thread_id] = nullptr;
      }
      h264_minimal_store.initialized_flags[thread_id] = false;

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
        param.b_repeat_headers = 1;
        param.b_annexb = 1;
        param.i_sync_lookahead = 0;
        param.i_bframe = 0;
        param.i_threads = h264_streaming_mode ? 0 : 1;
        param.i_log_level = X264_LOG_ERROR;
        param.vui.b_fullrange = 0;
        param.vui.i_sar_width = 1;
        param.vui.i_sar_height = 1;
        if (is_cbr) {
            param.rc.i_rc_method = X264_RC_ABR;
            int abs_bitrate = static_cast<int>(std::abs(h264_bitrate_kbps));
            param.rc.i_bitrate = abs_bitrate;
            param.rc.i_vbv_max_bitrate = abs_bitrate;
            if (vbv_buffer_size_kb > 0) {
              param.rc.i_vbv_buffer_size = vbv_buffer_size_kb;
            } else {
              param.rc.i_vbv_buffer_size = (abs_bitrate + 9) / 10;
            }
            param.rc.b_filler = 0;
        } else {
            param.rc.i_rc_method = X264_RC_CRF;
            param.rc.f_rf_constant = static_cast<float>(std::max(0, std::min(51, current_crf_setting)));
        }
        if (param.i_csp == X264_CSP_I444) {
             param.vui.i_colorprim = 1;
             param.vui.i_transfer = 1;
             param.vui.i_colmatrix = 6;
             x264_param_apply_profile(&param, "high444");
        } else {
           param.vui.i_colorprim = 1;
           param.vui.i_transfer  = 1;
           param.vui.i_colmatrix = 6;
           x264_param_apply_profile(&param, "baseline");
        }
        param.b_aud = 0;

        h264_minimal_store.encoders[thread_id] = x264_encoder_open(&param);
        if (!h264_minimal_store.encoders[thread_id]) {
          std::cerr << "H264 T" << thread_id << ": x264_encoder_open FAILED." << std::endl;
          result.type = StripeDataType::UNKNOWN;
        } else {
          h264_minimal_store.initialized_flags[thread_id] = true;
          h264_minimal_store.initialized_widths[thread_id] = param.i_width;
          h264_minimal_store.initialized_heights[thread_id] = param.i_height;
          h264_minimal_store.initialized_crfs[thread_id] = current_crf_setting;
          h264_minimal_store.initialized_csps[thread_id] = param.i_csp;
          h264_minimal_store.initialized_colorspaces[thread_id] = colorspace_setting;
          h264_minimal_store.initialized_full_range_flags[thread_id] = use_full_range;
          h264_minimal_store.force_idr_flags[thread_id] = true;
          h264_minimal_store.initialized_cbr_flags[thread_id] = is_cbr;
          h264_minimal_store.initialized_bitrates[thread_id] = static_cast<int>(std::abs(h264_bitrate_kbps));
        }
      }
    } else if (needs_crf_reinit) {
      x264_t* encoder_to_reconfig = h264_minimal_store.encoders[thread_id];
      if (encoder_to_reconfig) {
        x264_param_t params_for_reconfig;
        x264_encoder_parameters(encoder_to_reconfig, &params_for_reconfig);
        params_for_reconfig.rc.f_rf_constant =
          static_cast<float>(std::max(0, std::min(51, current_crf_setting)));
        if (x264_encoder_reconfig(encoder_to_reconfig, &params_for_reconfig) == 0) {
          h264_minimal_store.initialized_crfs[thread_id] = current_crf_setting;
        } else {
          std::cerr << "H264 T" << thread_id
                    << ": x264_encoder_reconfig for CRF FAILED. Old CRF "
                    << h264_minimal_store.initialized_crfs[thread_id]
                    << " may persist." << std::endl;
        }
      }
    } else if (needs_cbr_reinit) {
      x264_t* encoder_to_reconfig = h264_minimal_store.encoders[thread_id];
      if (encoder_to_reconfig) {
        x264_param_t params_for_reconfig;
        x264_encoder_parameters(encoder_to_reconfig, &params_for_reconfig);
        int abs_bitrate = static_cast<int>(std::abs(h264_bitrate_kbps));
        params_for_reconfig.rc.i_bitrate = abs_bitrate;
        params_for_reconfig.rc.i_vbv_max_bitrate = abs_bitrate;
        if (vbv_buffer_size_kb > 0) {
          params_for_reconfig.rc.i_vbv_buffer_size = vbv_buffer_size_kb;
        } else {
          params_for_reconfig.rc.i_vbv_buffer_size = static_cast<int>(abs_bitrate * 0.1);
        }
        if (x264_encoder_reconfig(encoder_to_reconfig, &params_for_reconfig) == 0) {
          h264_minimal_store.initialized_bitrates[thread_id] = abs_bitrate;
        } else {
          std::cerr << "H264 T" << thread_id
                    << ": x264_encoder_reconfig for CBR FAILED. Old CBR "
                    << h264_minimal_store.initialized_bitrates[thread_id]
                    << " may persist." << std::endl;
        }
      }
    }

    if (h264_minimal_store.initialized_flags[thread_id]) {
      current_encoder = h264_minimal_store.encoders[thread_id];
    }
  }

  if (result.type == StripeDataType::UNKNOWN) return result;
  if (!current_encoder) {
    std::cerr << "H264 T" << thread_id << ": Encoder not ready post-init for Y"
              << stripe_y_start << "." << std::endl;
    result.type = StripeDataType::UNKNOWN; return result;
  }

  x264_picture_t pic_in;
  x264_picture_init(&pic_in);
  pic_in.i_pts = static_cast<int64_t>(frame_counter);
  pic_in.img.i_csp = target_x264_csp;

  pic_in.img.plane[0] = (uint8_t*)y_plane_stripe_start;
  pic_in.img.plane[1] = (uint8_t*)u_plane_stripe_start;
  pic_in.img.plane[2] = (uint8_t*)v_plane_stripe_start;
  pic_in.img.i_stride[0] = y_stride;
  pic_in.img.i_stride[1] = u_stride;
  pic_in.img.i_stride[2] = v_stride;

  bool force_idr_now = force_idr ? true :  false;
  {
    std::lock_guard<std::mutex> lock(h264_minimal_store.store_mutex);
    h264_minimal_store.ensure_size(thread_id);
    if (h264_minimal_store.initialized_flags[thread_id] &&
        thread_id < static_cast<int>(h264_minimal_store.force_idr_flags.size()) &&
        h264_minimal_store.force_idr_flags[thread_id]) {
      force_idr_now = true;
    }
  }
  pic_in.i_type = force_idr_now ? X264_TYPE_IDR : X264_TYPE_AUTO;

  x264_nal_t* nals = nullptr;
  int i_nals = 0;
  x264_picture_t pic_out;
  x264_picture_init(&pic_out);

  int frame_size = x264_encoder_encode(current_encoder, &nals, &i_nals,
                                       &pic_in, &pic_out);

  if (frame_size < 0) {
    std::cerr << "H264 T" << thread_id << ": x264_encoder_encode FAILED: " << frame_size
              << " (Y" << stripe_y_start << ")" << std::endl;
    result.type = StripeDataType::UNKNOWN; return result;
  }

  if (frame_size > 0) {
    if (force_idr_now && pic_out.b_keyframe &&
        (pic_out.i_type == X264_TYPE_IDR || pic_out.i_type == X264_TYPE_I)) {
      std::lock_guard<std::mutex> lock(h264_minimal_store.store_mutex);
      if (thread_id < static_cast<int>(h264_minimal_store.force_idr_flags.size())) {
        h264_minimal_store.force_idr_flags[thread_id] = false;
      }
    }

    const unsigned char DATA_TYPE_H264_STRIPED_TAG = 0x04;
    unsigned char frame_type_header_byte = 0x00;
    if (pic_out.i_type == X264_TYPE_IDR) frame_type_header_byte = 0x01;
    else if (pic_out.i_type == X264_TYPE_I) frame_type_header_byte = 0x02;

    int header_sz = emit_header ? 10 : 0;
    int total_sz = frame_size + header_sz;
    result.data = new (std::nothrow) unsigned char[total_sz];
    if (!result.data) {
      std::cerr << "H264 T" << thread_id << ": new result.data FAILED (Y"
                << stripe_y_start << ")" << std::endl;
      result.type = StripeDataType::UNKNOWN; return result;
    }

    if (header_sz) {
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
    }

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
 * @param use_fullframe_hashing True to use full-frame hashing (samples every 12th row),
 *                              false to hash every row.
 * @return A 64-bit hash value representing the content of the YUV stripe.
 *         Returns 0 if input parameters are invalid (e.g., null pointers,
 *         non-positive dimensions).
 */
uint64_t calculate_yuv_stripe_hash(const uint8_t* y_plane_stripe_start, int y_stride,
                                   const uint8_t* u_plane_stripe_start, int u_stride,
                                   const uint8_t* v_plane_stripe_start, int v_stride,
                                   int width, int height, bool is_i420, bool use_fullframe_hashing) {
    if (!y_plane_stripe_start || !u_plane_stripe_start || width <= 0 || height <= 0) {
        return 0;
    }

    const int row_step = use_fullframe_hashing ? 12 : 1;
    XXH3_state_t hash_state;
    XXH3_64bits_reset(&hash_state);

    for (int r = 0; r < height; r += row_step) {
        XXH3_64bits_update(&hash_state, y_plane_stripe_start +
                           static_cast<size_t>(r) * y_stride, width);
    }

    if (v_plane_stripe_start) {
        int chroma_width = is_i420 ? (width / 2) : width;
        int chroma_height = is_i420 ? (height / 2) : height;

        if (chroma_width > 0 && chroma_height > 0) {
            for (int r = 0; r < chroma_height; r += row_step) {
                XXH3_64bits_update(&hash_state, u_plane_stripe_start +
                                   static_cast<size_t>(r) * u_stride, chroma_width);
            }
            for (int r = 0; r < chroma_height; r += row_step) {
                XXH3_64bits_update(&hash_state, v_plane_stripe_start +
                                   static_cast<size_t>(r) * v_stride, chroma_width);
            }
        }
    } else {
        int uv_plane_height = height / 2;
        int uv_plane_width_bytes = width;

        if (uv_plane_height > 0) {
             for (int r = 0; r < uv_plane_height; r += row_step) {
                XXH3_64bits_update(&hash_state, u_plane_stripe_start +
                                   static_cast<size_t>(r) * u_stride, uv_plane_width_bytes);
            }
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
uint64_t calculate_bgr_stripe_hash_from_shm(const unsigned char* shm_start_ptr,
                                            int stride_bytes,
                                            int width, int height,
                                            int bytes_per_pixel) {
    XXH64_state_t* const state = XXH64_createState();
    if (state==NULL) abort();
    XXH64_reset(state, 0);

    for (int y = 0; y < height; ++y) {
        const unsigned char* row_ptr = shm_start_ptr + static_cast<size_t>(y) * stride_bytes;
        XXH64_update(state, row_ptr, static_cast<size_t>(width) * bytes_per_pixel);
    }

    uint64_t const hash = XXH64_digest(state);
    XXH64_freeState(state);
    return hash;
}

extern "C" {

  typedef void* ScreenCaptureModuleHandle;

  // Exposes the C++ struct size so Python can assert ctypes/C ABI agreement.
  int pixelflux_capture_settings_size() {
    return static_cast<int>(sizeof(CaptureSettings));
  }

  // FNV-1a-64 over each field's (name, offset, size) in declaration order, so
  // Python can catch a same-size field reorder/rename the size-only guard misses.
  // Declared metadata only (not a memory dump) and LE-explicit, so it matches the
  // pure-stdlib Python computation in __init__.py byte-for-byte.
  uint64_t pixelflux_capture_settings_layout_hash() {
    const uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    const uint64_t FNV_PRIME  = 0x100000001b3ULL;
    uint64_t h = FNV_OFFSET;
    auto mix = [&](const unsigned char* p, size_t n) {
      for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= FNV_PRIME; }
    };
    auto mix_str = [&](const char* s) {
      size_t n = 0; while (s[n]) ++n;
      mix(reinterpret_cast<const unsigned char*>(s), n);
      unsigned char z = 0; mix(&z, 1);  // 0x00 delimiter between name and numbers
    };
    auto mix_u32 = [&](uint32_t v) {     // little-endian, matches Python struct.pack("<I")
      unsigned char b[4] = {
        static_cast<unsigned char>(v & 0xFF),
        static_cast<unsigned char>((v >> 8) & 0xFF),
        static_cast<unsigned char>((v >> 16) & 0xFF),
        static_cast<unsigned char>((v >> 24) & 0xFF),
      };
      mix(b, 4);
    };
    // #NAME makes the hashed string literal and the offsetof/sizeof target the
    // SAME token, so a field rename can never desync the literal from the field.
    #define PF_FIELD(NAME) do {                                                   \
        mix_str(#NAME);                                                           \
        mix_u32(static_cast<uint32_t>(offsetof(CaptureSettings, NAME)));          \
        mix_u32(static_cast<uint32_t>(sizeof(static_cast<CaptureSettings*>(nullptr)->NAME))); \
      } while (0)
    PF_FIELD(capture_width);
    PF_FIELD(capture_height);
    PF_FIELD(scale);
    PF_FIELD(capture_x);
    PF_FIELD(capture_y);
    PF_FIELD(target_fps);
    PF_FIELD(jpeg_quality);
    PF_FIELD(paint_over_jpeg_quality);
    PF_FIELD(use_paint_over_quality);
    PF_FIELD(paint_over_trigger_frames);
    PF_FIELD(damage_block_threshold);
    PF_FIELD(damage_block_duration);
    PF_FIELD(output_mode);
    PF_FIELD(h264_crf);
    PF_FIELD(h264_paintover_crf);
    PF_FIELD(h264_paintover_burst_frames);
    PF_FIELD(h264_fullcolor);
    PF_FIELD(h264_fullframe);
    PF_FIELD(h264_streaming_mode);
    PF_FIELD(capture_cursor);
    PF_FIELD(watermark_path);
    PF_FIELD(watermark_location_enum);
    PF_FIELD(vaapi_render_node_index);
    PF_FIELD(use_cpu);
    PF_FIELD(debug_logging);
    PF_FIELD(h264_cbr_mode);
    PF_FIELD(h264_bitrate_kbps);
    PF_FIELD(h264_vbv_buffer_size_kb);
    PF_FIELD(auto_adjust_screen_capture_size);
    PF_FIELD(omit_stripe_headers);
    PF_FIELD(deferred_free);
    PF_FIELD(vaapi_render_node_path);
    #undef PF_FIELD
    return h;
  }

  /**
   * @brief Creates a new instance of the ScreenCaptureModule.
   * @return A handle to the created ScreenCaptureModule instance.
   */
  ScreenCaptureModuleHandle create_screen_capture_module() {
    return static_cast<ScreenCaptureModuleHandle>(new ScreenCaptureModule());
  }

  /**
   * @brief Destroys a ScreenCaptureModule instance.
   * @param module_handle Handle to the ScreenCaptureModule instance to destroy.
   */
  void destroy_screen_capture_module(ScreenCaptureModuleHandle module_handle) {
    if (module_handle) {
      delete static_cast<ScreenCaptureModule*>(module_handle);
    }
  }

  /**
   * @brief Starts the screen capture process with the given settings and callback.
   * @param module_handle Handle to the ScreenCaptureModule instance.
   * @param settings The initial capture and encoding settings.
   * @param callback A function pointer to be called when an encoded stripe is ready.
   * @param user_data User-defined data to be passed to the callback function.
   */
  void start_screen_capture(ScreenCaptureModuleHandle module_handle,
                            CaptureSettings settings,
                            StripeCallback callback,
                            void* user_data) {
    if (module_handle) {
      ScreenCaptureModule* module = static_cast<ScreenCaptureModule*>(module_handle);
      module->modify_settings(settings);

      {
        std::lock_guard<std::mutex> lock(module->settings_mutex);
        module->stripe_callback = callback;
        module->user_data = user_data;
      }

      module->start_capture();
    }
  }

  /**
   * @brief Requests an IDR frame from an encoder
   * @param moduel_handle Handle to the ScreenCaptureModule instance.
  */
  void request_idr(ScreenCaptureModuleHandle module_handle) {
		if (module_handle) {
			ScreenCaptureModule* module = static_cast<ScreenCaptureModule*>(module_handle);
			if (module) {
				module->request_idr();
			}
		}
  }

  /**
   * @brief Update video bitrate of an encoder
   * @param moduel_handle Handle to the ScreenCaptureModule instance.
   * @param bitrate video bitrate to set 
   */
  void update_video_bitrate(ScreenCaptureModuleHandle module_handle, int bitrate) {
    if (module_handle) {
      ScreenCaptureModule* module = static_cast<ScreenCaptureModule*>(module_handle);
      if (module) module->update_video_bitrate(bitrate);
    }
  }

  /**
   * @brief Updates the framerate of the screen capture process.
   * @param module_handle Handle to the ScreenCaptureModule instance.
   * @param fps The new framerate to set.
   */
  void update_framerate(ScreenCaptureModuleHandle module_handle, double fps) {
    if (module_handle) {
      ScreenCaptureModule* module = static_cast<ScreenCaptureModule*>(module_handle);
      if (module) module->update_framerate(fps);
    }
  }

    /**
   * @brief Updates the VBV buffer size for H.264 CBR mode.
   * @param module_handle Handle to the ScreenCaptureModule instance.
   * @param vbv_buffer_size_kbps The new VBV buffer size in kb
   */
  void update_vbv_buffer_size(ScreenCaptureModuleHandle module_handle, int vbv_buffer_size_kb) {
    if (module_handle) {
      ScreenCaptureModule* module = static_cast<ScreenCaptureModule*>(module_handle);
      if (module) module->update_vbv_buffer_size(vbv_buffer_size_kb);
    }
  }

  /**
   * @brief Stops the screen capture process.
   * @param module_handle Handle to the ScreenCaptureModule instance.
   */
  void stop_screen_capture(ScreenCaptureModuleHandle module_handle) {
    if (module_handle) {
      static_cast<ScreenCaptureModule*>(module_handle)->stop_capture();
    }
  }

  /**
   * @brief Frees the data buffer within a StripeEncodeResult.
   * This is called from Python via ctypes to prevent memory leaks.
   * @param result Pointer to the StripeEncodeResult whose data needs freeing.
   */
  void free_stripe_encode_result_data(StripeEncodeResult* result) {
    if (result && result->data) {
      delete[] result->data;
      result->data = nullptr;
    }
  }

}
