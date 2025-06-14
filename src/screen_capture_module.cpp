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

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <GL/glx.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <future>
#include <iomanip>
#include <iostream>
#include <jpeglib.h>
#include <libyuv/convert.h>
#include <libyuv/convert_from.h>
#include <libyuv/planar_functions.h>
#include <list>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <thread>
#include <vector>
#include <x264.h>
#include <xxhash.h>
#include "nvEncodeAPI.h"

// Manages a pool of x264 encoders and their associated picture structures
// for multi-threaded H.264 encoding.
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

  // Ensures that internal vectors are large enough to accommodate a given thread_id.
  // Input: thread_id - The ID of the thread, used as an index.
  // Output: None.
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

  // Resets the store by closing all encoders, cleaning pictures, and clearing vectors.
  // Input: None.
  // Output: None.
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

  // Destructor for MinimalEncoderStore, ensures resources are released by calling reset.
  // Input: None.
  // Output: None.
  ~MinimalEncoderStore() { reset(); }
};

MinimalEncoderStore g_h264_minimal_store;

// Holds the state and resources for an NVIDIA NVENC hardware encoder session.
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

  // Constructor for NvencEncoderState, initializes version fields for NVENC structures.
  // Input: None.
  // Output: None.
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

enum class OutputMode { JPEG = 0, H264 = 1 };

enum class StripeDataType { UNKNOWN = 0, JPEG = 1, H264 = 2 };

// Stores settings for screen capture and encoding.
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

  // Default constructor for CaptureSettings, initializes with default capture parameters.
  // Input: None.
  // Output: None.
  CaptureSettings()
    : capture_width(1920)
    , capture_height(1080)
    , capture_x(0)
    , capture_y(0)
    , target_fps(60.0)
    , jpeg_quality(85)
    , paint_over_jpeg_quality(95)
    , use_paint_over_quality(false)
    , paint_over_trigger_frames(10)
    , damage_block_threshold(15)
    , damage_block_duration(30)
    , output_mode(OutputMode::JPEG)
    , h264_crf(25)
    , h264_fullcolor(false)
    , h264_fullframe(false) {}

  // Parameterized constructor for CaptureSettings.
  // Input: cw - capture width, ch - capture height, cx - capture x offset, cy - capture y offset,
  //        fps - target frames per second, jq - JPEG quality, pojq - paint over JPEG quality,
  //        upoq - use paint over quality flag, potf - paint over trigger frames,
  //        dbt - damage block threshold, dbd - damage block duration,
  //        om - output mode, crf - H.264 CRF value, h264_fc - H.264 full color flag,
  //        h264_ff - H.264 full frame flag.
  // Output: None.
  CaptureSettings(int cw,
                  int ch,
                  int cx,
                  int cy,
                  double fps,
                  int jq,
                  int pojq,
                  bool upoq,
                  int potf,
                  int dbt,
                  int dbd,
                  OutputMode om = OutputMode::JPEG,
                  int crf = 25,
                  bool h264_fc = false,
                  bool h264_ff = false)
    : capture_width(cw)
    , capture_height(ch)
    , capture_x(cx)
    , capture_y(cy)
    , target_fps(fps)
    , jpeg_quality(jq)
    , paint_over_jpeg_quality(pojq)
    , use_paint_over_quality(upoq)
    , paint_over_trigger_frames(potf)
    , damage_block_threshold(dbt)
    , damage_block_duration(dbd)
    , output_mode(om)
    , h264_crf(crf)
    , h264_fullcolor(h264_fc)
    , h264_fullframe(h264_ff) {}
};

// Represents the result of encoding a single stripe (or full frame) of video data.
struct StripeEncodeResult {
  StripeDataType type;
  int stripe_y_start;
  int stripe_height;
  int size;
  unsigned char* data;
  int frame_id;

  // Default constructor for StripeEncodeResult, initializes members to default/null values.
  // Input: None.
  // Output: None.
  StripeEncodeResult()
    : type(StripeDataType::UNKNOWN)
    , stripe_y_start(0)
    , stripe_height(0)
    , size(0)
    , data(nullptr)
    , frame_id(-1) {}

  StripeEncodeResult(StripeEncodeResult&& other) noexcept;
  StripeEncodeResult& operator=(StripeEncodeResult&& other) noexcept;

private:
  StripeEncodeResult(const StripeEncodeResult&) = delete;
  StripeEncodeResult& operator=(const StripeEncodeResult&) = delete;
};

// Move constructor for StripeEncodeResult.
// Input: other - Rvalue reference to another StripeEncodeResult to move from.
// Output: None.
StripeEncodeResult::StripeEncodeResult(StripeEncodeResult&& other) noexcept
  : type(other.type)
  , stripe_y_start(other.stripe_y_start)
  , stripe_height(other.stripe_height)
  , size(other.size)
  , data(other.data)
  , frame_id(other.frame_id) {
  other.type = StripeDataType::UNKNOWN;
  other.stripe_y_start = 0;
  other.stripe_height = 0;
  other.size = 0;
  other.data = nullptr;
  other.frame_id = -1;
}

// Move assignment operator for StripeEncodeResult.
// Input: other - Rvalue reference to another StripeEncodeResult to move assign from.
// Output: Reference to this StripeEncodeResult.
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

typedef void (*StripeCallback)(StripeEncodeResult* result, void* user_data);

extern "C" {
void free_stripe_encode_result_data(StripeEncodeResult* result);
}

// Loads the NVIDIA NVENC library and initializes its API function pointers.
// Input: None.
// Output: True if the API was loaded successfully, false otherwise.
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

// Resets the current NVENC encoder session, releasing its resources.
// Input: None.
// Output: None.
void reset_nvenc_encoder() {
  std::lock_guard<std::mutex> lock(g_nvenc_mutex);

  if (!g_nvenc_state.nvenc_funcs.nvEncDestroyEncoder ||
      !g_nvenc_state.nvenc_funcs.nvEncDestroyInputBuffer ||
      !g_nvenc_state.nvenc_funcs.nvEncDestroyBitstreamBuffer) {
    g_nvenc_state.encoder_session = nullptr;
    g_nvenc_state.input_buffers.clear();
    g_nvenc_state.output_buffers.clear();
    g_nvenc_state.initialized = false;
    return;
  }

  if (!g_nvenc_state.encoder_session) {
    g_nvenc_state.input_buffers.clear();
    g_nvenc_state.output_buffers.clear();
    g_nvenc_state.initialized = false;
    return;
  }

  for (NV_ENC_INPUT_PTR& ptr : g_nvenc_state.input_buffers) {
    if (ptr)
      g_nvenc_state.nvenc_funcs.nvEncDestroyInputBuffer(g_nvenc_state.encoder_session, ptr);
    ptr = nullptr;
  }
  g_nvenc_state.input_buffers.clear();

  for (NV_ENC_OUTPUT_PTR& ptr : g_nvenc_state.output_buffers) {
    if (ptr)
      g_nvenc_state.nvenc_funcs.nvEncDestroyBitstreamBuffer(g_nvenc_state.encoder_session,
                                                            ptr);
    ptr = nullptr;
  }
  g_nvenc_state.output_buffers.clear();

  g_nvenc_state.nvenc_funcs.nvEncDestroyEncoder(g_nvenc_state.encoder_session);
  g_nvenc_state.encoder_session = nullptr;
  g_nvenc_state.initialized = false;
}

// Unloads the NVENC library if it was previously loaded and an encoder was initialized.
// This also resets any active encoder session.
// Input: None.
// Output: None.
void unload_nvenc_library_if_loaded() {
  std::lock_guard<std::mutex> lock(g_nvenc_mutex);
  if (g_nvenc_state.initialized) {
    if (g_nvenc_state.encoder_session && g_nvenc_state.nvenc_funcs.nvEncDestroyEncoder) {
      for (NV_ENC_INPUT_PTR& ptr : g_nvenc_state.input_buffers) {
        if (ptr && g_nvenc_state.nvenc_funcs.nvEncDestroyInputBuffer)
          g_nvenc_state.nvenc_funcs.nvEncDestroyInputBuffer(g_nvenc_state.encoder_session,
                                                            ptr);
        ptr = nullptr;
      }
      g_nvenc_state.input_buffers.clear();
      for (NV_ENC_OUTPUT_PTR& ptr : g_nvenc_state.output_buffers) {
        if (ptr && g_nvenc_state.nvenc_funcs.nvEncDestroyBitstreamBuffer)
          g_nvenc_state.nvenc_funcs.nvEncDestroyBitstreamBuffer(
            g_nvenc_state.encoder_session, ptr);
        ptr = nullptr;
      }
      g_nvenc_state.output_buffers.clear();
      g_nvenc_state.nvenc_funcs.nvEncDestroyEncoder(g_nvenc_state.encoder_session);
      g_nvenc_state.encoder_session = nullptr;
    }
    g_nvenc_state.initialized = false;
  }

  if (g_nvenc_lib_handle) {
    dlclose(g_nvenc_lib_handle);
    g_nvenc_lib_handle = nullptr;
    memset(&g_nvenc_state.nvenc_funcs, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    g_nvenc_state.nvenc_funcs.version = NV_ENCODE_API_FUNCTION_LIST_VER;
  }
  g_nvenc_state.initialized = false;
}

// Initializes the NVENC encoder with specified parameters.
// If already initialized with the same parameters, it returns true.
// Otherwise, it resets any existing encoder and initializes a new one.
// Input: p_display - X11 Display pointer for OpenGL context.
//        width - Encoding width.
//        height - Encoding height.
//        target_qp - Target quantization parameter for constant QP mode.
//        fps - Target frames per second.
//        use_yuv444 - True to use YUV444 color format, false for NV12 (YUV420).
// Output: True if initialization was successful, false otherwise.
bool initialize_nvenc_encoder(Display* p_display,
                              int width,
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

  if (!g_nvenc_state.nvenc_funcs.nvEncOpenEncodeSessionEx) {
    g_nvenc_state.initialized = false;
    return false;
  }

  NVENCSTATUS status;
  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS session_params = {0};
  session_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
  session_params.deviceType = NV_ENC_DEVICE_TYPE_OPENGL;
  session_params.device = p_display;
  session_params.apiVersion = NVENCAPI_VERSION;

  status = g_nvenc_state.nvenc_funcs.nvEncOpenEncodeSessionEx(
    &session_params, &g_nvenc_state.encoder_session);

  if (status != NV_ENC_SUCCESS) {
    g_nvenc_state.encoder_session = nullptr;
    g_nvenc_state.initialized = false;
    return false;
  }
  if (!g_nvenc_state.encoder_session) {
    g_nvenc_state.initialized = false;
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

// Encodes a full BGR24 frame using NVENC into H.264 format.
// Input: width - Width of the input frame.
//        height - Height of the input frame.
//        full_bgr24_data - Pointer to the BGR24 pixel data.
//        frame_counter - Current frame number.
//        force_idr_frame - True to force an IDR frame, false otherwise.
// Output: StripeEncodeResult containing the H.264 encoded data. Throws std::runtime_error on failure.
StripeEncodeResult encode_fullframe_nvenc(int width,
                                          int height,
                                          const unsigned char* full_bgr24_data,
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
  if (!g_nvenc_state.nvenc_funcs.nvEncLockInputBuffer ||
      !g_nvenc_state.nvenc_funcs.nvEncEncodePicture) {
    throw std::runtime_error("NVENC_ENCODE_FATAL: Core NVENC functions not loaded.");
  }

  const NV_ENC_BUFFER_FORMAT current_buffer_format = g_nvenc_state.initialized_buffer_format;
  const bool is_yuv444_mode = (current_buffer_format == NV_ENC_BUFFER_FORMAT_YUV444 ||
                               current_buffer_format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT);

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

  unsigned char* const locked_buffer_data_start =
    static_cast<unsigned char*>(lip.bufferDataPtr);
  const int locked_buffer_pitch_for_planes = lip.pitch;
  int conversion_result_code = -1;

  if (is_yuv444_mode) {
    unsigned char* y_plane_target = locked_buffer_data_start;
    unsigned char* u_plane_target =
      locked_buffer_data_start + static_cast<size_t>(locked_buffer_pitch_for_planes) * height;
    unsigned char* v_plane_target =
      u_plane_target + static_cast<size_t>(locked_buffer_pitch_for_planes) * height;

    conversion_result_code = libyuv::RAWToI444(full_bgr24_data,
                                               width * 3,
                                               y_plane_target,
                                               locked_buffer_pitch_for_planes,
                                               u_plane_target,
                                               locked_buffer_pitch_for_planes,
                                               v_plane_target,
                                               locked_buffer_pitch_for_planes,
                                               width,
                                               height);
  } else {
    size_t y_plane_size_i420 = static_cast<size_t>(width) * height;
    size_t uv_plane_size_i420 = static_cast<size_t>(width / 2) * (height / 2);
    std::vector<uint8_t> i420_intermediate_buffer(y_plane_size_i420 +
                                                  2 * uv_plane_size_i420);
    uint8_t* i420_y = i420_intermediate_buffer.data();
    uint8_t* i420_u = i420_y + y_plane_size_i420;
    uint8_t* i420_v = i420_u + uv_plane_size_i420;

    conversion_result_code = libyuv::RAWToI420(full_bgr24_data,
                                               width * 3,
                                               i420_y,
                                               width,
                                               i420_u,
                                               width / 2,
                                               i420_v,
                                               width / 2,
                                               width,
                                               height);
    if (conversion_result_code == 0) {
      unsigned char* nv12_y_target = locked_buffer_data_start;
      unsigned char* nv12_uv_target =
        locked_buffer_data_start + static_cast<size_t>(locked_buffer_pitch_for_planes) * height;
      conversion_result_code = libyuv::I420ToNV12(i420_y,
                                                  width,
                                                  i420_u,
                                                  width / 2,
                                                  i420_v,
                                                  width / 2,
                                                  nv12_y_target,
                                                  locked_buffer_pitch_for_planes,
                                                  nv12_uv_target,
                                                  locked_buffer_pitch_for_planes,
                                                  width,
                                                  height);
    }
  }

  if (conversion_result_code != 0) {
    g_nvenc_state.nvenc_funcs.nvEncUnlockInputBuffer(g_nvenc_state.encoder_session, in_ptr);
    throw std::runtime_error(
      "NVENC_ENCODE_ERROR: libyuv conversion failed with code " +
      std::to_string(conversion_result_code) + (is_yuv444_mode ? " (for YUV444)" : " (for NV12)"));
  }

  g_nvenc_state.nvenc_funcs.nvEncUnlockInputBuffer(g_nvenc_state.encoder_session, in_ptr);

  NV_ENC_PIC_PARAMS pp = {0};
  pp.version = NV_ENC_PIC_PARAMS_VER;
  pp.inputBuffer = in_ptr;
  pp.outputBitstream = out_ptr;
  pp.bufferFmt = current_buffer_format;
  pp.inputWidth = width;
  pp.inputHeight = height;
  pp.inputPitch = locked_buffer_pitch_for_planes;
  pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
  pp.inputTimeStamp = frame_counter;
  pp.frameIdx = frame_counter;
  if (force_idr_frame) {
    pp.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;
  }

  status =
    g_nvenc_state.nvenc_funcs.nvEncEncodePicture(g_nvenc_state.encoder_session, &pp);
  if (status != NV_ENC_SUCCESS) {
    std::string err_msg =
      "NVENC_ENCODE_ERROR: nvEncEncodePicture FAILED: " + std::to_string(status);
    const char* api_err =
      g_nvenc_state.nvenc_funcs.nvEncGetLastErrorString(g_nvenc_state.encoder_session);
    if (api_err)
      err_msg += " - API Error: " + std::string(api_err);
    g_nvenc_state.current_input_buffer_idx =
      (g_nvenc_state.current_input_buffer_idx + 1) % g_nvenc_state.buffer_pool_size;
    throw std::runtime_error(err_msg);
  }

  NV_ENC_LOCK_BITSTREAM lbs = {0};
  lbs.version = NV_ENC_LOCK_BITSTREAM_VER;
  lbs.outputBitstream = out_ptr;
  status =
    g_nvenc_state.nvenc_funcs.nvEncLockBitstream(g_nvenc_state.encoder_session, &lbs);
  if (status != NV_ENC_SUCCESS) {
    g_nvenc_state.current_input_buffer_idx =
      (g_nvenc_state.current_input_buffer_idx + 1) % g_nvenc_state.buffer_pool_size;
    g_nvenc_state.current_output_buffer_idx =
      (g_nvenc_state.current_output_buffer_idx + 1) % g_nvenc_state.buffer_pool_size;
    throw std::runtime_error("NVENC_ENCODE_ERROR: nvEncLockBitstream FAILED: " +
                             std::to_string(status));
  }

  if (lbs.bitstreamSizeInBytes > 0) {
    const unsigned char TAG = 0x04;
    unsigned char type_hdr = 0x00;
    if (lbs.pictureType == NV_ENC_PIC_TYPE_IDR)
      type_hdr = 0x01;
    else if (lbs.pictureType == NV_ENC_PIC_TYPE_I)
      type_hdr = 0x02;

    int header_sz = 10;
    int total_sz = lbs.bitstreamSizeInBytes + header_sz;
    result.data = new (std::nothrow) unsigned char[total_sz];
    if (!result.data) {
      g_nvenc_state.nvenc_funcs.nvEncUnlockBitstream(g_nvenc_state.encoder_session, out_ptr);
      g_nvenc_state.current_input_buffer_idx =
        (g_nvenc_state.current_input_buffer_idx + 1) % g_nvenc_state.buffer_pool_size;
      g_nvenc_state.current_output_buffer_idx =
        (g_nvenc_state.current_output_buffer_idx + 1) % g_nvenc_state.buffer_pool_size;
      throw std::runtime_error("NVENC_ENCODE_ERROR: Malloc failed for output bitstream.");
    }
    result.data[0] = TAG;
    result.data[1] = type_hdr;
    uint16_t net_val;
    net_val = htons(static_cast<uint16_t>(result.frame_id % 65536));
    std::memcpy(result.data + 2, &net_val, 2);
    net_val = htons(static_cast<uint16_t>(result.stripe_y_start));
    std::memcpy(result.data + 4, &net_val, 2);
    net_val = htons(static_cast<uint16_t>(width));
    std::memcpy(result.data + 6, &net_val, 2);
    net_val = htons(static_cast<uint16_t>(height));
    std::memcpy(result.data + 8, &net_val, 2);
    std::memcpy(
      result.data + header_sz, lbs.bitstreamBufferPtr, lbs.bitstreamSizeInBytes);
    result.size = total_sz;
  } else {
    result.size = 0;
    result.data = nullptr;
  }

  g_nvenc_state.nvenc_funcs.nvEncUnlockBitstream(g_nvenc_state.encoder_session, out_ptr);
  g_nvenc_state.current_input_buffer_idx =
    (g_nvenc_state.current_input_buffer_idx + 1) % g_nvenc_state.buffer_pool_size;
  g_nvenc_state.current_output_buffer_idx =
    (g_nvenc_state.current_output_buffer_idx + 1) % g_nvenc_state.buffer_pool_size;

  return result;
}

StripeEncodeResult encode_stripe_jpeg(int thread_id,
                                      int stripe_y_start,
                                      int stripe_height,
                                      int width,
                                      int height,
                                      int capture_width_actual,
                                      const unsigned char* rgb_data,
                                      int rgb_data_len,
                                      int jpeg_quality,
                                      int frame_counter);

StripeEncodeResult encode_stripe_h264(int thread_id,
                                      int stripe_y_start,
                                      int stripe_height,
                                      int capture_width_actual,
                                      const unsigned char* stripe_rgb24_data,
                                      int frame_counter,
                                      int current_crf_setting,
                                      int colorspace_setting,
                                      bool use_full_range);

uint64_t calculate_stripe_hash(const std::vector<unsigned char>& rgb_data);

// Manages the screen capture process, including X11 interaction,
// frame grabbing, and dispatching encoding tasks.
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
  OutputMode output_mode = OutputMode::H264;

  std::atomic<bool> stop_requested;
  std::thread capture_thread;
  StripeCallback stripe_callback = nullptr;
  void* user_data = nullptr;
  int frame_counter = 0;
  int encoded_frame_count = 0;
  int total_stripes_encoded_this_interval = 0;
  int total_nvenc_frames_encoded_this_interval = 0;
  mutable std::mutex settings_mutex;

  bool is_nvidia_system_detected = false;
  bool nvenc_operational = false;
  Display* glx_display_for_nvenc = nullptr;
  Window glx_window_for_nvenc = 0;
  GLXContext glx_context_for_nvenc = nullptr;
  GLXFBConfig* glx_fbconfigs_for_nvenc = nullptr;
  XVisualInfo* glx_visual_info_for_nvenc = nullptr;
  Colormap glx_colormap_for_nvenc = 0;

public:
  // Constructor for ScreenCaptureModule. Attempts to load NVENC API.
  // Input: None.
  // Output: None.
  ScreenCaptureModule() : stop_requested(false) {
    std::lock_guard<std::mutex> lock(g_nvenc_mutex);
    if (LoadNvencApi()) {
      is_nvidia_system_detected = true;
    } else {
      is_nvidia_system_detected = false;
    }
  }

  // Destructor for ScreenCaptureModule. Ensures capture is stopped.
  // Input: None.
  // Output: None.
  ~ScreenCaptureModule() { stop_capture(); }

  // Starts the screen capture thread. If a capture is already running, it's stopped first.
  // Initializes or resets encoder states.
  // Input: None.
  // Output: None.
  void start_capture() {
    if (capture_thread.joinable()) {
      stop_capture();
    }
    g_h264_minimal_store.reset();

    nvenc_operational = false;
    g_nvenc_force_next_idr_global = true;

    stop_requested = false;
    frame_counter = 0;
    encoded_frame_count = 0;
    total_stripes_encoded_this_interval = 0;
    total_nvenc_frames_encoded_this_interval = 0;

    capture_thread = std::thread(&ScreenCaptureModule::capture_loop, this);
  }

  // Stops the screen capture thread and cleans up NVENC resources.
  // Input: None.
  // Output: None.
  void stop_capture() {
    stop_requested = true;
    if (capture_thread.joinable()) {
      capture_thread.join();
    }
    if (g_nvenc_state.initialized) {
      reset_nvenc_encoder();
    }
    unload_nvenc_library_if_loaded();
  }

  // Retrieves the current capture settings.
  // Input: None.
  // Output: A CaptureSettings struct with the current settings.
  CaptureSettings get_current_settings() const {
    std::lock_guard<std::mutex> lock(settings_mutex);
    return CaptureSettings(capture_width,
                           capture_height,
                           capture_x,
                           capture_y,
                           target_fps,
                           jpeg_quality,
                           paint_over_jpeg_quality,
                           use_paint_over_quality,
                           paint_over_trigger_frames,
                           damage_block_threshold,
                           damage_block_duration,
                           output_mode,
                           h264_crf,
                           h264_fullcolor,
                           h264_fullframe);
  }

private:
  // Main loop for capturing screen frames, processing, and encoding them.
  // Runs in a separate thread.
  // Input: None.
  // Output: None.
  void capture_loop() {
    Display* display = nullptr;
    Window root_window = 0;
    int screen_num = 0;

    XShmSegmentInfo shminfo_loop;
    memset(&shminfo_loop, 0, sizeof(shminfo_loop));
    XImage* shm_image = nullptr;

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

    try {
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
      }

      if (local_current_output_mode == OutputMode::H264) {
        if (local_capture_width_actual % 2 != 0 && local_capture_width_actual > 0)
          local_capture_width_actual--;
        if (local_capture_height_actual % 2 != 0 && local_capture_height_actual > 0)
          local_capture_height_actual--;
      }
      if (local_capture_width_actual <= 0 || local_capture_height_actual <= 0) {
        throw std::runtime_error(
          "CAPTURE_FATAL: Invalid capture dimensions after initial adjustment.");
      }

      std::chrono::duration<double> target_frame_duration_seconds =
        std::chrono::duration<double>(
          1.0 / (local_current_target_fps < 1.0 ? 30.0 : local_current_target_fps));
      auto next_frame_time =
        std::chrono::high_resolution_clock::now() + target_frame_duration_seconds;

      char* display_env = std::getenv("DISPLAY");
      const char* display_name = display_env ? display_env : ":0";
      display = XOpenDisplay(display_name);
      if (!display) {
        throw std::runtime_error("CAPTURE_FATAL: Failed to open X display " +
                                 std::string(display_name));
      }
      this->glx_display_for_nvenc = display;
      root_window = DefaultRootWindow(display);
      screen_num = DefaultScreen(display);

      if (!XShmQueryExtension(display)) {
        throw std::runtime_error("CAPTURE_FATAL: X Shared Memory Extension not available!");
      }

      if (this->is_nvidia_system_detected) {
        int glx_major, glx_minor;
        if (!glXQueryVersion(display, &glx_major, &glx_minor)) {
          throw std::runtime_error("GLX_FATAL: glXQueryVersion failed.");
        }
        static int fb_attributes[] = {GLX_X_RENDERABLE,
                                      True,
                                      GLX_DRAWABLE_TYPE,
                                      GLX_WINDOW_BIT,
                                      GLX_RENDER_TYPE,
                                      GLX_RGBA_BIT,
                                      GLX_X_VISUAL_TYPE,
                                      GLX_TRUE_COLOR,
                                      GLX_RED_SIZE,
                                      8,
                                      GLX_GREEN_SIZE,
                                      8,
                                      GLX_BLUE_SIZE,
                                      8,
                                      GLX_ALPHA_SIZE,
                                      8,
                                      GLX_DEPTH_SIZE,
                                      24,
                                      GLX_STENCIL_SIZE,
                                      8,
                                      GLX_DOUBLEBUFFER,
                                      True,
                                      None};
        int fbcount;
        this->glx_fbconfigs_for_nvenc =
          glXChooseFBConfig(display, screen_num, fb_attributes, &fbcount);
        if (!this->glx_fbconfigs_for_nvenc || fbcount == 0) {
          if (this->glx_fbconfigs_for_nvenc)
            XFree(this->glx_fbconfigs_for_nvenc);
          this->glx_fbconfigs_for_nvenc = nullptr;
          throw std::runtime_error("GLX_FATAL: glXChooseFBConfig failed.");
        }
        this->glx_visual_info_for_nvenc =
          glXGetVisualFromFBConfig(display, this->glx_fbconfigs_for_nvenc[0]);
        if (!this->glx_visual_info_for_nvenc) {
          throw std::runtime_error("GLX_FATAL: glXGetVisualFromFBConfig failed.");
        }
        XSetWindowAttributes swa;
        this->glx_colormap_for_nvenc =
          XCreateColormap(display,
                          root_window,
                          this->glx_visual_info_for_nvenc->visual,
                          AllocNone);
        swa.colormap = this->glx_colormap_for_nvenc;
        swa.background_pixmap = None;
        swa.border_pixel = 0;
        swa.event_mask = StructureNotifyMask;
        this->glx_window_for_nvenc =
          XCreateWindow(display,
                        root_window,
                        0,
                        0,
                        1,
                        1,
                        0,
                        this->glx_visual_info_for_nvenc->depth,
                        InputOutput,
                        this->glx_visual_info_for_nvenc->visual,
                        CWBorderPixel | CWColormap | CWEventMask,
                        &swa);
        if (!this->glx_window_for_nvenc) {
          if (this->glx_colormap_for_nvenc)
            XFreeColormap(display, this->glx_colormap_for_nvenc);
          throw std::runtime_error("GLX_FATAL: XCreateWindow failed for GLX window.");
        }
        this->glx_context_for_nvenc =
          glXCreateContext(display, this->glx_visual_info_for_nvenc, NULL, GL_TRUE);
        if (!this->glx_context_for_nvenc) {
          if (this->glx_window_for_nvenc)
            XDestroyWindow(display, this->glx_window_for_nvenc);
          if (this->glx_colormap_for_nvenc)
            XFreeColormap(display, this->glx_colormap_for_nvenc);
          throw std::runtime_error("GLX_FATAL: glXCreateContext failed.");
        }
        if (!glXMakeCurrent(
              display, this->glx_window_for_nvenc, this->glx_context_for_nvenc)) {
          if (this->glx_context_for_nvenc)
            glXDestroyContext(display, this->glx_context_for_nvenc);
          if (this->glx_window_for_nvenc)
            XDestroyWindow(display, this->glx_window_for_nvenc);
          if (this->glx_colormap_for_nvenc)
            XFreeColormap(display, this->glx_colormap_for_nvenc);
          throw std::runtime_error("GLX_FATAL: glXMakeCurrent failed.");
        }
      }

      shm_image = XShmCreateImage(display,
                                  DefaultVisual(display, screen_num),
                                  DefaultDepth(display, screen_num),
                                  ZPixmap,
                                  nullptr,
                                  &shminfo_loop,
                                  local_capture_width_actual,
                                  local_capture_height_actual);
      if (!shm_image) {
        throw std::runtime_error(
          "CAPTURE_FATAL: XShmCreateImage failed for " +
          std::to_string(local_capture_width_actual) + "x" +
          std::to_string(local_capture_height_actual));
      }
      shminfo_loop.shmid = shmget(
        IPC_PRIVATE, static_cast<size_t>(shm_image->bytes_per_line) * shm_image->height, IPC_CREAT | 0600);
      if (shminfo_loop.shmid < 0) {
        XDestroyImage(shm_image);
        shm_image = nullptr;
        perror("shmget");
        throw std::runtime_error("CAPTURE_FATAL: shmget failed.");
      }
      shminfo_loop.shmaddr = (char*)shmat(shminfo_loop.shmid, nullptr, 0);
      if (shminfo_loop.shmaddr == (char*)-1) {
        shmctl(shminfo_loop.shmid, IPC_RMID, 0);
        XDestroyImage(shm_image);
        shm_image = nullptr;
        perror("shmat");
        throw std::runtime_error("CAPTURE_FATAL: shmat failed.");
      }
      shminfo_loop.readOnly = False;
      shm_image->data = shminfo_loop.shmaddr;
      if (!XShmAttach(display, &shminfo_loop)) {
        shmdt(shminfo_loop.shmaddr);
        shmctl(shminfo_loop.shmid, IPC_RMID, 0);
        XDestroyImage(shm_image);
        shm_image = nullptr;
        throw std::runtime_error("CAPTURE_FATAL: XShmAttach failed.");
      }

      this->nvenc_operational = false;
      if (this->is_nvidia_system_detected &&
          local_current_output_mode == OutputMode::H264 && local_current_h264_fullframe) {
        if (this->glx_display_for_nvenc && this->glx_context_for_nvenc) {
          if (initialize_nvenc_encoder(this->glx_display_for_nvenc,
                                       local_capture_width_actual,
                                       local_capture_height_actual,
                                       local_current_h264_crf,
                                       local_current_target_fps,
                                       local_current_h264_fullcolor)) {
            this->nvenc_operational = true;
            g_nvenc_force_next_idr_global = true;
          }
        }
      }
      if (!this->nvenc_operational && g_nvenc_state.initialized) {
        reset_nvenc_encoder();
      }

      int num_cores = std::max(1, (int)std::thread::hardware_concurrency());
      int num_stripes_config = num_cores;

      int N_processing_stripes;
      if (local_capture_height_actual <= 0)
        N_processing_stripes = 0;
      else {
        if (local_current_output_mode == OutputMode::H264) {
          if (local_current_h264_fullframe)
            N_processing_stripes = 1;
          else {
            const int MIN_H264_STRIPE_HEIGHT_PX = 64;
            if (local_capture_height_actual < MIN_H264_STRIPE_HEIGHT_PX)
              N_processing_stripes = 1;
            else {
              int max_stripes_by_min_height =
                local_capture_height_actual / MIN_H264_STRIPE_HEIGHT_PX;
              N_processing_stripes = std::min(num_stripes_config, max_stripes_by_min_height);
              if (N_processing_stripes == 0)
                N_processing_stripes = 1;
            }
          }
        } else {
          N_processing_stripes = std::min(num_stripes_config, local_capture_height_actual);
          if (N_processing_stripes == 0 && local_capture_height_actual > 0)
            N_processing_stripes = 1;
        }
      }
      if (N_processing_stripes == 0 && local_capture_height_actual > 0)
        N_processing_stripes = 1;

      std::vector<uint64_t> stripe_previous_hashes(num_stripes_config, 0);
      std::vector<int> stripe_consecutive_same_hashes(num_stripes_config, 0);
      std::vector<bool> stripe_paint_over_sent(num_stripes_config, false);
      std::vector<int> stripe_consecutive_diff_hashes(num_stripes_config, 0);
      std::vector<int> current_jpeg_qualities(num_stripes_config);

      enum class StripeOperationalMode { CHECKING, STALE_BLOCKED, ACTIVE_BLOCKED };
      std::vector<StripeOperationalMode> stripe_op_modes(num_stripes_config,
                                                         StripeOperationalMode::CHECKING);
      std::vector<int> stripe_block_timers(num_stripes_config, 0);
      std::vector<uint64_t> stripe_hash_at_active_block_start(num_stripes_config, 0);

      for (int k_init = 0; k_init < num_stripes_config; ++k_init) {
        current_jpeg_qualities[k_init] = local_current_use_paint_over_quality
                                           ? local_current_paint_over_jpeg_quality
                                           : local_current_jpeg_quality;
      }
      auto last_output_time = std::chrono::high_resolution_clock::now();

      while (!stop_requested) {
        auto current_loop_iter_start_time = std::chrono::high_resolution_clock::now();
        if (current_loop_iter_start_time < next_frame_time) {
          auto time_to_sleep = next_frame_time - current_loop_iter_start_time;
          if (time_to_sleep > std::chrono::milliseconds(0))
            std::this_thread::sleep_for(time_to_sleep);
        }
        next_frame_time += target_frame_duration_seconds;

        if (!XShmGetImage(display,
                          root_window,
                          shm_image,
                          local_capture_x_offset,
                          local_capture_y_offset,
                          AllPlanes)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }

        std::vector<unsigned char> full_bgr_data(
          static_cast<size_t>(local_capture_width_actual) * local_capture_height_actual * 3);
        unsigned char* shm_data_ptr = (unsigned char*)shm_image->data;
        int bytes_per_pixel_shm = shm_image->bits_per_pixel / 8;
        int bytes_per_line_shm = shm_image->bytes_per_line;
        for (int y = 0; y < local_capture_height_actual; ++y) {
          for (int x = 0; x < local_capture_width_actual; ++x) {
            unsigned char* pixel_ptr = shm_data_ptr +
                                       (static_cast<size_t>(y) * bytes_per_line_shm) +
                                       (static_cast<size_t>(x) * bytes_per_pixel_shm);
            size_t base_idx = (static_cast<size_t>(y) * local_capture_width_actual + x) * 3;
            full_bgr_data[base_idx + 0] = pixel_ptr[2];
            full_bgr_data[base_idx + 1] = pixel_ptr[1];
            full_bgr_data[base_idx + 2] = pixel_ptr[0];
          }
        }

        std::vector<std::future<StripeEncodeResult>> futures;
        std::vector<std::thread> threads;

        int h264_base_even_height = 0;
        int h264_num_stripes_with_extra_pair = 0;
        int current_y_start_for_stripe = 0;
        if (local_current_output_mode == OutputMode::H264 && N_processing_stripes > 0 &&
            local_capture_height_actual > 0 && !local_current_h264_fullframe) {
          int H = local_capture_height_actual, N = N_processing_stripes, base_h = H / N;
          h264_base_even_height = (base_h > 0) ? (base_h - (base_h % 2)) : 0;
          if (h264_base_even_height == 0 && H >= 2)
            h264_base_even_height = 2;
          if (h264_base_even_height > 0) {
            int H_base_covered = h264_base_even_height * N, H_remaining = H - H_base_covered;
            if (H_remaining < 0)
              H_remaining = 0;
            h264_num_stripes_with_extra_pair = std::min(H_remaining / 2, N);
          }
        }
        bool any_content_encoded_this_frame = false;
        int derived_h264_colorspace_setting;
        bool derived_h264_use_full_range;
        if (local_current_h264_fullcolor) {
          derived_h264_colorspace_setting = 444;
          derived_h264_use_full_range = true;
        } else {
          derived_h264_colorspace_setting = 420;
          derived_h264_use_full_range = false;
        }

        bool use_advanced_damage_blocking_this_frame = (N_processing_stripes == 1);

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
                if (i < h264_num_stripes_with_extra_pair)
                  current_stripe_height += 2;
              } else if (N_processing_stripes == 1) {
                current_stripe_height = local_capture_height_actual;
              } else {
                current_stripe_height = 0;
              }
            }
          } else {
            if (N_processing_stripes > 0) {
              int base_stripe_height_jpeg =
                local_capture_height_actual / N_processing_stripes;
              int remainder_height_jpeg =
                local_capture_height_actual % N_processing_stripes;
              start_y =
                i * base_stripe_height_jpeg + std::min(i, remainder_height_jpeg);
              current_stripe_height =
                base_stripe_height_jpeg + (i < remainder_height_jpeg ? 1 : 0);
            } else {
              current_stripe_height = 0;
            }
          }

          if (current_stripe_height <= 0)
            continue;
          if (start_y + current_stripe_height > local_capture_height_actual) {
            current_stripe_height = local_capture_height_actual - start_y;
            if (current_stripe_height <= 0)
              continue;
            if (local_current_output_mode == OutputMode::H264 &&
                !local_current_h264_fullframe && current_stripe_height % 2 != 0 &&
                current_stripe_height > 0) {
              current_stripe_height--;
            }
            if (current_stripe_height <= 0)
              continue;
          }

          if (local_current_output_mode == OutputMode::H264 &&
              !local_current_h264_fullframe) {
            current_y_start_for_stripe += current_stripe_height;
          }

          bool send_this_stripe_final = false;
          bool is_h264_idr_paintover_this_stripe = false;
          const unsigned char* data_for_processing_ptr = nullptr;
          std::vector<unsigned char> stripe_bgr_data_storage_iter;
          bool data_was_extracted_this_iteration = false;

          if (use_advanced_damage_blocking_this_frame) {
            StripeOperationalMode current_op_mode = stripe_op_modes[i];
            bool should_calculate_hash_adv = false;
            bool should_send_stripe_preliminary_adv = false;
            uint64_t current_hash_adv = stripe_previous_hashes[i];

            if (current_op_mode == StripeOperationalMode::STALE_BLOCKED) {
              stripe_block_timers[i]--;
              stripe_consecutive_same_hashes[i]++;
              if ((local_current_use_paint_over_quality &&
                   local_current_output_mode == OutputMode::JPEG) ||
                  local_current_output_mode == OutputMode::H264) {
                if (stripe_consecutive_same_hashes[i] >=
                      local_current_paint_over_trigger_frames &&
                    !stripe_paint_over_sent[i]) {
                  should_send_stripe_preliminary_adv = true;
                }
              }
              if (stripe_block_timers[i] <= 0)
                should_calculate_hash_adv = true;
            } else if (current_op_mode == StripeOperationalMode::ACTIVE_BLOCKED) {
              should_send_stripe_preliminary_adv = true;
              stripe_paint_over_sent[i] = false;
              stripe_consecutive_same_hashes[i] = 0;
              stripe_block_timers[i]--;
              if (stripe_block_timers[i] <= 0)
                should_calculate_hash_adv = true;
            } else {
              should_calculate_hash_adv = true;
            }

            if (should_calculate_hash_adv || should_send_stripe_preliminary_adv) {
              data_for_processing_ptr = full_bgr_data.data();
              data_was_extracted_this_iteration = true;
              if (should_calculate_hash_adv) {
                current_hash_adv = calculate_stripe_hash(full_bgr_data);
              }
            }

            if (current_op_mode == StripeOperationalMode::STALE_BLOCKED) {
              if (should_calculate_hash_adv) {
                if (current_hash_adv == stripe_previous_hashes[i]) {
                  stripe_op_modes[i] = StripeOperationalMode::STALE_BLOCKED;
                  stripe_block_timers[i] = local_current_damage_block_duration;
                  if (should_send_stripe_preliminary_adv) {
                    send_this_stripe_final = true;
                    stripe_paint_over_sent[i] = true;
                    if (local_current_output_mode == OutputMode::H264)
                      is_h264_idr_paintover_this_stripe = true;
                  }
                } else {
                  stripe_previous_hashes[i] = current_hash_adv;
                  stripe_consecutive_same_hashes[i] = 0;
                  stripe_consecutive_diff_hashes[i] = 1;
                  stripe_op_modes[i] = StripeOperationalMode::CHECKING;
                  send_this_stripe_final = true;
                  stripe_paint_over_sent[i] = false;
                }
              } else {
                if (should_send_stripe_preliminary_adv) {
                  send_this_stripe_final = true;
                  stripe_paint_over_sent[i] = true;
                  if (local_current_output_mode == OutputMode::H264)
                    is_h264_idr_paintover_this_stripe = true;
                }
              }
            } else if (current_op_mode == StripeOperationalMode::ACTIVE_BLOCKED) {
              send_this_stripe_final = true;
              if (should_calculate_hash_adv) {
                if (current_hash_adv == stripe_hash_at_active_block_start[i] ||
                    current_hash_adv == stripe_previous_hashes[i]) {
                  stripe_previous_hashes[i] = current_hash_adv;
                  stripe_consecutive_same_hashes[i] = 1;
                  stripe_consecutive_diff_hashes[i] = 0;
                  stripe_op_modes[i] = StripeOperationalMode::CHECKING;
                  current_jpeg_qualities[i] = local_current_use_paint_over_quality
                                                ? local_current_paint_over_jpeg_quality
                                                : local_current_jpeg_quality;
                } else {
                  stripe_previous_hashes[i] = current_hash_adv;
                  stripe_consecutive_same_hashes[i] = 0;
                  stripe_consecutive_diff_hashes[i] = 1;
                  stripe_op_modes[i] = StripeOperationalMode::ACTIVE_BLOCKED;
                  stripe_block_timers[i] = local_current_damage_block_duration;
                  stripe_hash_at_active_block_start[i] = current_hash_adv;
                }
              }
              if (should_calculate_hash_adv)
                stripe_previous_hashes[i] = current_hash_adv;

            } else {
              if (current_hash_adv == stripe_previous_hashes[i]) {
                stripe_consecutive_same_hashes[i]++;
                stripe_consecutive_diff_hashes[i] = 0;
                bool paint_over_cond =
                  (local_current_output_mode == OutputMode::JPEG &&
                   local_current_use_paint_over_quality) ||
                  local_current_output_mode == OutputMode::H264;
                if (paint_over_cond &&
                    stripe_consecutive_same_hashes[i] >=
                      local_current_paint_over_trigger_frames &&
                    !stripe_paint_over_sent[i]) {
                  send_this_stripe_final = true;
                  stripe_paint_over_sent[i] = true;
                  if (local_current_output_mode == OutputMode::H264)
                    is_h264_idr_paintover_this_stripe = true;
                }
                if (stripe_consecutive_same_hashes[i] >= local_current_damage_block_threshold) {
                  stripe_op_modes[i] = StripeOperationalMode::STALE_BLOCKED;
                  stripe_block_timers[i] = local_current_damage_block_duration;
                }
              } else {
                stripe_previous_hashes[i] = current_hash_adv;
                stripe_consecutive_same_hashes[i] = 0;
                stripe_consecutive_diff_hashes[i]++;
                send_this_stripe_final = true;
                stripe_paint_over_sent[i] = false;
                if (local_current_output_mode == OutputMode::JPEG)
                  current_jpeg_qualities[i] =
                    std::max(current_jpeg_qualities[i] - 1, local_current_jpeg_quality);
                if (stripe_consecutive_diff_hashes[i] >=
                    local_current_damage_block_threshold) {
                  stripe_op_modes[i] = StripeOperationalMode::ACTIVE_BLOCKED;
                  stripe_block_timers[i] = local_current_damage_block_duration;
                  stripe_hash_at_active_block_start[i] = current_hash_adv;
                }
              }
            }
          } else {
            stripe_bgr_data_storage_iter.resize(
              static_cast<size_t>(local_capture_width_actual) * current_stripe_height * 3);
            int row_stride_bgr = local_capture_width_actual * 3;
            for (int y_offset = 0; y_offset < current_stripe_height; ++y_offset) {
              int global_y = start_y + y_offset;
              size_t dest_offset = static_cast<size_t>(y_offset) * row_stride_bgr;
              size_t src_offset = static_cast<size_t>(global_y) * row_stride_bgr;
              if (global_y < local_capture_height_actual &&
                  (src_offset + row_stride_bgr) <= full_bgr_data.size())
                std::memcpy(stripe_bgr_data_storage_iter.data() + dest_offset,
                            &full_bgr_data[src_offset],
                            row_stride_bgr);
              else
                std::memset(
                  stripe_bgr_data_storage_iter.data() + dest_offset, 0, row_stride_bgr);
            }
            data_for_processing_ptr = stripe_bgr_data_storage_iter.data();
            data_was_extracted_this_iteration = true;

            uint64_t current_hash_simple =
              calculate_stripe_hash(stripe_bgr_data_storage_iter);

            if (current_hash_simple == stripe_previous_hashes[i]) {
              stripe_consecutive_same_hashes[i]++;
              stripe_consecutive_diff_hashes[i] = 0;
              bool paint_over_cond =
                (local_current_output_mode == OutputMode::JPEG &&
                 local_current_use_paint_over_quality) ||
                local_current_output_mode == OutputMode::H264;
              if (paint_over_cond &&
                  stripe_consecutive_same_hashes[i] >=
                    local_current_paint_over_trigger_frames &&
                  !stripe_paint_over_sent[i]) {
                send_this_stripe_final = true;
                stripe_paint_over_sent[i] = true;
                if (local_current_output_mode == OutputMode::H264)
                  is_h264_idr_paintover_this_stripe = true;
              }
            } else {
              stripe_previous_hashes[i] = current_hash_simple;
              stripe_consecutive_same_hashes[i] = 0;
              stripe_consecutive_diff_hashes[i]++;
              send_this_stripe_final = true;
              stripe_paint_over_sent[i] = false;
              if (local_current_output_mode == OutputMode::JPEG) {
                current_jpeg_qualities[i] =
                  std::max(current_jpeg_qualities[i] - 1, local_current_jpeg_quality);
              }
            }
          }

          if (send_this_stripe_final) {
            any_content_encoded_this_frame = true;
            if (!data_was_extracted_this_iteration) {
              if (N_processing_stripes == 1) {
                data_for_processing_ptr = full_bgr_data.data();
              } else {
                stripe_bgr_data_storage_iter.resize(
                  static_cast<size_t>(local_capture_width_actual) * current_stripe_height *
                  3);
                int row_stride_bgr = local_capture_width_actual * 3;
                for (int y_offset = 0; y_offset < current_stripe_height; ++y_offset) {
                  int global_y = start_y + y_offset;
                  size_t dest_offset = static_cast<size_t>(y_offset) * row_stride_bgr;
                  size_t src_offset = static_cast<size_t>(global_y) * row_stride_bgr;
                  if (global_y < local_capture_height_actual &&
                      (src_offset + row_stride_bgr) <= full_bgr_data.size())
                    std::memcpy(stripe_bgr_data_storage_iter.data() + dest_offset,
                                &full_bgr_data[src_offset],
                                row_stride_bgr);
                  else
                    std::memset(stripe_bgr_data_storage_iter.data() + dest_offset,
                                0,
                                row_stride_bgr);
                }
                data_for_processing_ptr = stripe_bgr_data_storage_iter.data();
              }
            }
            if (!data_for_processing_ptr) {
              std::cerr << "CRITICAL_WARN: Stripe " << i
                        << " trying to send with NULL data_for_processing_ptr." << std::endl;
              continue;
            }

            if (local_current_output_mode == OutputMode::H264 &&
                local_current_h264_fullframe && this->is_nvidia_system_detected &&
                this->nvenc_operational) {
              total_nvenc_frames_encoded_this_interval++;
              bool force_idr_for_nvenc =
                g_nvenc_force_next_idr_global.exchange(false) ||
                is_h264_idr_paintover_this_stripe;
              StripeEncodeResult nvenc_res;
              try {
                nvenc_res = encode_fullframe_nvenc(local_capture_width_actual,
                                                   local_capture_height_actual,
                                                   data_for_processing_ptr,
                                                   this->frame_counter,
                                                   force_idr_for_nvenc);
                if (stripe_callback && nvenc_res.data && nvenc_res.size > 0)
                  stripe_callback(&nvenc_res, user_data);
                else if (nvenc_res.data)
                  free_stripe_encode_result_data(&nvenc_res);
              } catch (const std::runtime_error& enc_err) {
                std::cerr << "NVENC_ENCODE_RUNTIME_ERROR: " << enc_err.what()
                          << ". Disabling NVENC for this session." << std::endl;
                this->nvenc_operational = false;
                reset_nvenc_encoder();
                if (force_idr_for_nvenc)
                  g_nvenc_force_next_idr_global = true;
              }
            } else if (local_current_output_mode == OutputMode::JPEG) {
              total_stripes_encoded_this_interval++;
              int quality_to_use_jpeg;
              if (stripe_paint_over_sent[i] && local_current_use_paint_over_quality)
                quality_to_use_jpeg = local_current_paint_over_jpeg_quality;
              else if (use_advanced_damage_blocking_this_frame &&
                       stripe_op_modes[i] == StripeOperationalMode::ACTIVE_BLOCKED)
                quality_to_use_jpeg = local_current_jpeg_quality;
              else
                quality_to_use_jpeg = current_jpeg_qualities[i];

              std::packaged_task<StripeEncodeResult(
                int, int, int, int, int, int, const unsigned char*, int, int, int)>
                task(encode_stripe_jpeg);
              futures.push_back(task.get_future());
              threads.push_back(std::thread(std::move(task),
                                            i,
                                            start_y,
                                            current_stripe_height,
                                            DisplayWidth(display, screen_num),
                                            local_capture_height_actual,
                                            local_capture_width_actual,
                                            full_bgr_data.data(),
                                            static_cast<int>(full_bgr_data.size()),
                                            quality_to_use_jpeg,
                                            this->frame_counter));

            } else {
              total_stripes_encoded_this_interval++;
              int crf_for_encode = local_current_h264_crf;
              if (is_h264_idr_paintover_this_stripe && local_current_h264_crf > 10) {
                crf_for_encode = 10;
                std::lock_guard<std::mutex> lock(g_h264_minimal_store.store_mutex);
                g_h264_minimal_store.ensure_size(i);
                g_h264_minimal_store.force_idr_flags[i] = true;
              }

              std::packaged_task<StripeEncodeResult(
                int, int, int, int, const unsigned char*, int, int, int, bool)>
                task(encode_stripe_h264);
              futures.push_back(task.get_future());
              std::vector<unsigned char> data_copy_for_thread(
                data_for_processing_ptr,
                data_for_processing_ptr +
                  (static_cast<size_t>(local_capture_width_actual) *
                   current_stripe_height * 3));
              threads.push_back(std::thread(
                [task_moved = std::move(task),
                 i,
                 start_y_thr = start_y,
                 stripe_h_thr = current_stripe_height,
                 cap_w_thr = local_capture_width_actual,
                 data_c = std::move(data_copy_for_thread),
                 fc_thr = this->frame_counter,
                 crf_val_thr = crf_for_encode,
                 cs_thr = derived_h264_colorspace_setting,
                 fr_thr = derived_h264_use_full_range]() mutable {
                  task_moved(i,
                             start_y_thr,
                             stripe_h_thr,
                             cap_w_thr,
                             data_c.data(),
                             fc_thr,
                             crf_val_thr,
                             cs_thr,
                             fr_thr);
                }));
            }
          }
        }

        std::vector<StripeEncodeResult> stripe_results;
        stripe_results.reserve(futures.size());
        for (auto& future : futures)
          stripe_results.push_back(future.get());
        futures.clear();
        for (StripeEncodeResult& result : stripe_results) {
          if (stripe_callback && result.data && result.size > 0)
            stripe_callback(&result, user_data);
          else if (result.data)
            free_stripe_encode_result_data(&result);
        }
        stripe_results.clear();
        for (auto& thread : threads)
          if (thread.joinable())
            thread.join();
        threads.clear();

        this->frame_counter++;
        if (any_content_encoded_this_frame)
          encoded_frame_count++;

        auto current_output_time_log = std::chrono::high_resolution_clock::now();
        auto output_elapsed_time_log = std::chrono::duration_cast<std::chrono::seconds>(
          current_output_time_log - last_output_time);
        if (output_elapsed_time_log.count() >= 1) {
          double actual_fps_val =
            (encoded_frame_count > 0 && output_elapsed_time_log.count() > 0)
              ? static_cast<double>(encoded_frame_count) / output_elapsed_time_log.count()
              : 0.0;
          double total_stripes_per_second_val =
            (total_stripes_encoded_this_interval > 0 &&
             output_elapsed_time_log.count() > 0)
              ? static_cast<double>(total_stripes_encoded_this_interval) /
                  output_elapsed_time_log.count()
              : 0.0;
          std::string mode_str, quality_str;
          if (local_current_output_mode == OutputMode::JPEG) {
            mode_str = "JPEG";
            quality_str = " Q:" + std::to_string(local_current_jpeg_quality);
          } else {
            if (local_current_h264_fullframe && this->is_nvidia_system_detected &&
                this->nvenc_operational) {
              mode_str = "H264 NVENC (FullFrame ";
              mode_str += (local_current_h264_fullcolor ? "YUV444" : "NV12");
              mode_str += ")";

              std::lock_guard<std::mutex> nvenc_lock(g_nvenc_mutex);
              quality_str = " QP:" + (g_nvenc_state.initialized
                                        ? std::to_string(g_nvenc_state.initialized_qp)
                                        : "N/A");
            } else {
              mode_str = "H264 x264";
              mode_str += (local_current_h264_fullcolor ? " CS:444 FR" : " CS:420 LR");
              mode_str += (local_current_h264_fullframe ? " FF" : " Striped");
              quality_str = " CRF:" + std::to_string(local_current_h264_crf);
            }
          }
          std::cout
            << "Res: " << local_capture_width_actual << "x" << local_capture_height_actual
            << " Mode: " << mode_str << " Stripes: "
            << (local_current_h264_fullframe && local_current_output_mode == OutputMode::H264
                  ? 1
                  : N_processing_stripes)
            << quality_str << " EncFPS: " << std::fixed << std::setprecision(2)
            << actual_fps_val
            << ((local_current_output_mode == OutputMode::H264 &&
                 local_current_h264_fullframe && this->is_nvidia_system_detected &&
                 this->nvenc_operational)
                  ? ""
                  : (" EncStripes/s: " +
                     std::to_string(static_cast<int>(total_stripes_per_second_val))))
            << std::endl;
          encoded_frame_count = 0;
          total_stripes_encoded_this_interval = 0;
          total_nvenc_frames_encoded_this_interval = 0;
          last_output_time = std::chrono::high_resolution_clock::now();
        }
      }
    } catch (const std::runtime_error& e) {
      std::cerr << "CAPTURE_FATAL_EXCEPTION: " << e.what() << std::endl;
      if (this->nvenc_operational || g_nvenc_state.initialized) {
        reset_nvenc_encoder();
        this->nvenc_operational = false;
      }
    } catch (...) {
      std::cerr << "CAPTURE_FATAL_EXCEPTION: Unknown." << std::endl;
      if (this->nvenc_operational || g_nvenc_state.initialized) {
        reset_nvenc_encoder();
        this->nvenc_operational = false;
      }
    }

    if (display) {
      if (this->is_nvidia_system_detected) {
        if (this->glx_context_for_nvenc) {
          if (glXGetCurrentDisplay() == display &&
              glXGetCurrentContext() == this->glx_context_for_nvenc) {
            glXMakeCurrent(display, None, NULL);
          }
          glXDestroyContext(display, this->glx_context_for_nvenc);
          this->glx_context_for_nvenc = nullptr;
        }
        if (this->glx_window_for_nvenc) {
          XDestroyWindow(display, this->glx_window_for_nvenc);
          this->glx_window_for_nvenc = 0;
        }
        if (this->glx_colormap_for_nvenc) {
          XFreeColormap(display, this->glx_colormap_for_nvenc);
          this->glx_colormap_for_nvenc = 0;
        }
        if (this->glx_visual_info_for_nvenc) {
          XFree(this->glx_visual_info_for_nvenc);
          this->glx_visual_info_for_nvenc = nullptr;
        }
        if (this->glx_fbconfigs_for_nvenc) {
          XFree(this->glx_fbconfigs_for_nvenc);
          this->glx_fbconfigs_for_nvenc = nullptr;
        }
      }
      if (shm_image) {
        XShmDetach(display, &shminfo_loop);
        if (shminfo_loop.shmaddr && shminfo_loop.shmaddr != (char*)-1)
          shmdt(shminfo_loop.shmaddr);
        if (shminfo_loop.shmid != -1 && shminfo_loop.shmid != 0)
          shmctl(shminfo_loop.shmid, IPC_RMID, 0);
        XDestroyImage(shm_image);
        shm_image = nullptr;
      }
      XCloseDisplay(display);
      display = nullptr;
      this->glx_display_for_nvenc = nullptr;
    }
    if (g_nvenc_state.initialized)
      reset_nvenc_encoder();
  }
};

extern "C" {
typedef void* ScreenCaptureModuleHandle;
// Creates a new ScreenCaptureModule instance.
// Input: None.
// Output: A handle (ScreenCaptureModuleHandle) to the created module, or nullptr on failure.
ScreenCaptureModuleHandle create_screen_capture_module() {
  try {
    return static_cast<ScreenCaptureModuleHandle>(new ScreenCaptureModule());
  } catch (...) {
    return nullptr;
  }
}
// Destroys a ScreenCaptureModule instance.
// Input: module_handle - Handle to the module to be destroyed.
// Output: None.
void destroy_screen_capture_module(ScreenCaptureModuleHandle module_handle) {
  if (module_handle)
    delete static_cast<ScreenCaptureModule*>(module_handle);
}
// Starts screen capture on the specified module with given settings and callback.
// Input: module_handle - Handle to the ScreenCaptureModule.
//        settings - CaptureSettings struct with desired parameters.
//        callback - Function pointer to call with encoded stripe data.
//        user_data - User-defined data to pass to the callback.
// Output: None.
void start_screen_capture(ScreenCaptureModuleHandle module_handle,
                          CaptureSettings settings,
                          StripeCallback callback,
                          void* user_data) {
  if (module_handle) {
    ScreenCaptureModule* module = static_cast<ScreenCaptureModule*>(module_handle);

    {
      std::lock_guard<std::mutex> lock(module->settings_mutex);
      module->capture_width = settings.capture_width;
      module->capture_height = settings.capture_height;
      module->capture_x = settings.capture_x;
      module->capture_y = settings.capture_y;
      module->target_fps = settings.target_fps;
      module->jpeg_quality = settings.jpeg_quality;
      module->paint_over_jpeg_quality = settings.paint_over_jpeg_quality;
      module->use_paint_over_quality = settings.use_paint_over_quality;
      module->paint_over_trigger_frames = settings.paint_over_trigger_frames;
      module->damage_block_threshold = settings.damage_block_threshold;
      module->damage_block_duration = settings.damage_block_duration;
      module->output_mode = settings.output_mode;
      module->h264_crf = settings.h264_crf;
      module->h264_fullcolor = settings.h264_fullcolor;
      module->h264_fullframe = settings.h264_fullframe;

      module->stripe_callback = callback;
      module->user_data = user_data;
    }
    module->start_capture();
  }
}
// Stops screen capture on the specified module.
// Input: module_handle - Handle to the ScreenCaptureModule.
// Output: None.
void stop_screen_capture(ScreenCaptureModuleHandle module_handle) {
  if (module_handle)
    static_cast<ScreenCaptureModule*>(module_handle)->stop_capture();
}
// Retrieves the current capture settings from the specified module.
// Input: module_handle - Handle to the ScreenCaptureModule.
// Output: A CaptureSettings struct with current settings, or default settings if handle is invalid.
CaptureSettings get_screen_capture_settings(ScreenCaptureModuleHandle module_handle) {
  if (module_handle)
    return static_cast<ScreenCaptureModule*>(module_handle)->get_current_settings();
  else
    return CaptureSettings{};
}
// Frees the data buffer within a StripeEncodeResult struct.
// Input: result - Pointer to the StripeEncodeResult whose data buffer is to be freed.
// Output: None.
void free_stripe_encode_result_data(StripeEncodeResult* result) {
  if (result && result->data) {
    delete[] result->data;
    result->data = nullptr;
    result->size = 0;
  }
}
}

// Encodes a stripe of RGB data into JPEG format.
// Input: thread_id - Identifier for the calling thread (for logging).
//        stripe_y_start - Starting Y coordinate of the stripe in the full image.
//        stripe_height - Height of the stripe to encode.
//        width - Width of the full image (used for context, actual stripe width is capture_width_actual).
//        height - Height of the full image (used for boundary checks).
//        capture_width_actual - Actual width of the stripe to encode.
//        rgb_data - Pointer to the full RGB image data.
//        rgb_data_len - Length of the rgb_data buffer (not directly used, implied by dimensions).
//        jpeg_quality - Quality setting for JPEG compression (0-100).
//        frame_counter - Current frame number.
// Output: StripeEncodeResult containing the JPEG encoded data.
StripeEncodeResult encode_stripe_jpeg(int thread_id,
                                      int stripe_y_start,
                                      int stripe_height,
                                      int width,
                                      int height,
                                      int capture_width_actual,
                                      const unsigned char* rgb_data,
                                      int rgb_data_len,
                                      int jpeg_quality,
                                      int frame_counter) {
  StripeEncodeResult result;
  result.type = StripeDataType::JPEG;
  result.stripe_y_start = stripe_y_start;
  result.stripe_height = stripe_height;
  result.frame_id = frame_counter;

  if (!rgb_data || stripe_height <= 0 || capture_width_actual <= 0) {
    std::cerr << "JPEG T" << thread_id << ": Invalid input for JPEG encoding." << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }
  if (stripe_y_start < 0 || stripe_y_start + stripe_height > height) {
    std::cerr << "JPEG T" << thread_id << ": Stripe Y coordinates out of bounds. Y_start="
              << stripe_y_start << ", StripeH=" << stripe_height << ", ImgH=" << height
              << std::endl;
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

  JSAMPROW row_pointer[1];
  int full_image_row_stride = capture_width_actual * 3;

  for (int y_in_stripe = 0; y_in_stripe < stripe_height; ++y_in_stripe) {
    int global_y = stripe_y_start + y_in_stripe;
    row_pointer[0] =
      const_cast<unsigned char*>(rgb_data + (static_cast<size_t>(global_y) * full_image_row_stride));
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);

  if (jpeg_size_temp > 0 && jpeg_buffer) {
    int padding_size = 4;
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
    free(jpeg_buffer);
  }
  return result;
}

// Encodes a stripe of RGB24 data into H.264 format using x264.
// Input: thread_id - Identifier for the calling thread (used for per-thread encoder instances).
//        stripe_y_start - Starting Y coordinate of the stripe.
//        stripe_height - Height of the stripe to encode (must be even).
//        capture_width_actual - Width of the stripe to encode (must be even).
//        stripe_rgb24_data - Pointer to the RGB24 pixel data for this stripe.
//        frame_counter - Current frame number.
//        current_crf_setting - CRF value for H.264 encoding.
//        colorspace_setting - Target colorspace (e.g., 420 or 444).
//        use_full_range - Boolean indicating if full range color should be used.
// Output: StripeEncodeResult containing the H.264 encoded data.
StripeEncodeResult encode_stripe_h264(int thread_id,
                                      int stripe_y_start,
                                      int stripe_height,
                                      int capture_width_actual,
                                      const unsigned char* stripe_rgb24_data,
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

  if (!stripe_rgb24_data) {
    std::cerr << "H264 T" << thread_id << ": Error - null rgb_data for stripe Y"
              << stripe_y_start << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }
  if (stripe_height <= 0 || capture_width_actual <= 0) {
    std::cerr << "H264 T" << thread_id << ": Invalid dimensions (" << capture_width_actual
              << "x" << stripe_height << ") for stripe Y" << stripe_y_start << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }
  if (capture_width_actual % 2 != 0 || stripe_height % 2 != 0) {
    std::cerr << "H264 T" << thread_id << ": Error - Odd dimensions ("
              << capture_width_actual << "x" << stripe_height << ") for stripe Y"
              << stripe_y_start << ". H264 requires even dimensions." << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }

  x264_t* current_encoder = nullptr;
  x264_picture_t* current_pic_in_ptr = nullptr;
  int target_x264_csp;
  int actual_colorspace_setting_for_reinit = colorspace_setting;

  switch (colorspace_setting) {
    case 444:
      target_x264_csp = X264_CSP_I444;
      break;
    case 420:
    default:
      target_x264_csp = X264_CSP_I420;
      actual_colorspace_setting_for_reinit = 420;
      break;
  }

  {
    std::lock_guard<std::mutex> lock(g_h264_minimal_store.store_mutex);
    g_h264_minimal_store.ensure_size(thread_id);

    bool is_first_init = !g_h264_minimal_store.initialized_flags[thread_id];
    bool dims_changed =
      !is_first_init &&
      (g_h264_minimal_store.initialized_widths[thread_id] != capture_width_actual ||
       g_h264_minimal_store.initialized_heights[thread_id] != stripe_height);
    bool cs_or_fr_changed =
      !is_first_init &&
      (g_h264_minimal_store.initialized_csps[thread_id] != target_x264_csp ||
       g_h264_minimal_store.initialized_colorspaces[thread_id] !=
         actual_colorspace_setting_for_reinit ||
       g_h264_minimal_store.initialized_full_range_flags[thread_id] != use_full_range);

    bool needs_crf_reinit_due_to_paint_over_or_change = false;
    if (!is_first_init &&
        g_h264_minimal_store.initialized_crfs[thread_id] != current_crf_setting) {
      needs_crf_reinit_due_to_paint_over_or_change = true;
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
        std::cerr << "H264 T" << thread_id << ": x264_param_default_preset FAILED."
                  << std::endl;
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
        param.i_threads = -1;
        param.i_log_level = X264_LOG_ERROR;
        param.vui.b_fullrange = use_full_range ? 1 : 0;
        param.vui.i_sar_width = 1;
        param.vui.i_sar_height = 1;
        param.vui.i_colorprim = use_full_range ? 1 : 9;
        param.vui.i_transfer = use_full_range ? 1 : 13;
        param.vui.i_colmatrix = use_full_range ? 1 : 9;
        param.b_aud = 0;

        if (param.i_csp == X264_CSP_I444) {
          x264_param_apply_profile(&param, "high444");
        } else {
          x264_param_apply_profile(&param, "baseline");
        }

        g_h264_minimal_store.encoders[thread_id] = x264_encoder_open(&param);
        if (!g_h264_minimal_store.encoders[thread_id]) {
          std::cerr << "H264 T" << thread_id << ": x264_encoder_open FAILED." << std::endl;
          result.type = StripeDataType::UNKNOWN;
        } else {
          g_h264_minimal_store.pics_in_ptrs[thread_id] = new (std::nothrow) x264_picture_t();
          if (!g_h264_minimal_store.pics_in_ptrs[thread_id]) {
            std::cerr << "H264 T" << thread_id << ": FAILED to new x264_picture_t."
                      << std::endl;
            x264_encoder_close(g_h264_minimal_store.encoders[thread_id]);
            g_h264_minimal_store.encoders[thread_id] = nullptr;
            result.type = StripeDataType::UNKNOWN;
          } else {
            x264_picture_init(g_h264_minimal_store.pics_in_ptrs[thread_id]);
            if (x264_picture_alloc(g_h264_minimal_store.pics_in_ptrs[thread_id],
                                   param.i_csp,
                                   param.i_width,
                                   param.i_height) < 0) {
              std::cerr << "H264 T" << thread_id << ": x264_picture_alloc FAILED for CSP "
                        << param.i_csp << " (" << param.i_width << "x" << param.i_height
                        << ")." << std::endl;
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
              g_h264_minimal_store.initialized_colorspaces[thread_id] =
                actual_colorspace_setting_for_reinit;
              g_h264_minimal_store.initialized_full_range_flags[thread_id] =
                use_full_range;
              g_h264_minimal_store.force_idr_flags[thread_id] = true;
            }
          }
        }
      }
    } else if (needs_crf_reinit_due_to_paint_over_or_change) {
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
                    << " may persist for current frame." << std::endl;
        }
      }
    }

    if (g_h264_minimal_store.initialized_flags[thread_id]) {
      current_encoder = g_h264_minimal_store.encoders[thread_id];
      current_pic_in_ptr = g_h264_minimal_store.pics_in_ptrs[thread_id];
    }
  }

  if (result.type == StripeDataType::UNKNOWN)
    return result;
  if (!current_encoder || !current_pic_in_ptr) {
    std::cerr << "H264 T" << thread_id << ": Encoder/Pic not ready post-init for Y"
              << stripe_y_start << "." << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }

  bool planes_ok =
    current_pic_in_ptr->img.plane[0] && current_pic_in_ptr->img.plane[1];
  if (target_x264_csp == X264_CSP_I420 || target_x264_csp == X264_CSP_I444) {
    planes_ok = planes_ok && current_pic_in_ptr->img.plane[2];
  }
  if (!planes_ok) {
    std::cerr << "H264 T" << thread_id << ": Pic planes NULL for CSP " << target_x264_csp
              << " (Y" << stripe_y_start << "). Critical error." << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }

  int src_stride_rgb24 = capture_width_actual * 3;
  int conversion_status = -1;

  if (target_x264_csp == X264_CSP_I444) {
    conversion_status =
      libyuv::RAWToI444(stripe_rgb24_data,
                        src_stride_rgb24,
                        current_pic_in_ptr->img.plane[0],
                        current_pic_in_ptr->img.i_stride[0],
                        current_pic_in_ptr->img.plane[1],
                        current_pic_in_ptr->img.i_stride[1],
                        current_pic_in_ptr->img.plane[2],
                        current_pic_in_ptr->img.i_stride[2],
                        capture_width_actual,
                        stripe_height);
  } else {
    conversion_status =
      libyuv::RAWToI420(stripe_rgb24_data,
                        src_stride_rgb24,
                        current_pic_in_ptr->img.plane[0],
                        current_pic_in_ptr->img.i_stride[0],
                        current_pic_in_ptr->img.plane[1],
                        current_pic_in_ptr->img.i_stride[1],
                        current_pic_in_ptr->img.plane[2],
                        current_pic_in_ptr->img.i_stride[2],
                        capture_width_actual,
                        stripe_height);
  }

  if (conversion_status != 0) {
    std::cerr << "H264 T" << thread_id << ": libyuv conversion to CSP " << target_x264_csp
              << " FAILED code " << conversion_status << " (Y" << stripe_y_start << ")"
              << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }

  current_pic_in_ptr->i_pts = static_cast<int64_t>(frame_counter);

  bool force_idr_now = false;
  {
    std::lock_guard<std::mutex> lock(g_h264_minimal_store.store_mutex);
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

  int frame_size =
    x264_encoder_encode(current_encoder, &nals, &i_nals, current_pic_in_ptr, &pic_out);

  if (frame_size < 0) {
    std::cerr << "H264 T" << thread_id << ": x264_encoder_encode FAILED: " << frame_size
              << " (Y" << stripe_y_start << ")" << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
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
    if (pic_out.i_type == X264_TYPE_IDR)
      frame_type_header_byte = 0x01;
    else if (pic_out.i_type == X264_TYPE_I)
      frame_type_header_byte = 0x02;

    int header_sz = 10;
    int total_sz = frame_size + header_sz;
    result.data = new (std::nothrow) unsigned char[total_sz];
    if (!result.data) {
      std::cerr << "H264 T" << thread_id << ": new result.data FAILED (Y" << stripe_y_start
                << ")" << std::endl;
      result.type = StripeDataType::UNKNOWN;
      return result;
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
        std::cerr << "H264 T" << thread_id << ": NAL copy overflow detected (Y"
                  << stripe_y_start << ")" << std::endl;
        delete[] result.data;
        result.data = nullptr;
        result.size = 0;
        result.type = StripeDataType::UNKNOWN;
        return result;
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

// Calculates a 64-bit hash (XXH3) of the provided RGB data.
// Input: rgb_data - A vector of unsigned char containing the RGB pixel data.
// Output: A 64-bit hash value. Returns 0 if the input data is empty.
uint64_t calculate_stripe_hash(const std::vector<unsigned char>& rgb_data) {
  if (rgb_data.empty())
    return 0;
  return XXH3_64bits(rgb_data.data(), rgb_data.size());
}
