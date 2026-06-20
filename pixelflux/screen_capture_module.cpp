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

// Python.h must be first so its feature macros win over libc/libstdc++ headers.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

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
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
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
// One-time, thread-safe resolution of the process-global CUDA driver handle/func-pointers
// (LoadCudaApi): concurrent multi-instance NVENC init otherwise races g_cuda_funcs (a thread
// could observe a torn struct and call a null/garbage fn-ptr). call_once publishes the fully
// populated struct before any waiter returns.
static std::once_flag g_cuda_load_once;
static bool g_cuda_load_ok = false;
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

// NVENC API version negotiated against the DRIVER (NVENCAPI_VERSION format: major|(minor<<24)).
// Following GStreamer's nvcodec approach: a build against a newer SDK header runs on older
// drivers (down to NVENC 10.0 / ~driver R445, covering CUDA 11 / 470) by (a) probing a list of
// API versions with nvEncodeAPICreateInstance until one <= the driver's max succeeds, and (b)
// tagging every struct with a DELIBERATELY-OLD, widely-supported revision + the negotiated
// version (NVENC structs are append-only, so an old-rev struct is the prefix an old driver
// expects). Defaults to the compiled version so pre-negotiation uses are unchanged.
static std::atomic<uint32_t> g_nvenc_api_version{NVENCAPI_VERSION};
// .version = negotiated api | (old struct revision << 16) | (0x7<<28) | extra-bits.
static inline uint32_t nvenc_struct_ver(uint32_t rev, uint32_t extra) {
  return g_nvenc_api_version.load(std::memory_order_relaxed) | (rev << 16) | (0x7u << 28) | extra;
}
// NV_ENC_CONFIG gained fields in SDK 12.0 (rev 7 -> 8). Use rev 8 only when the negotiated
// major is >= 12 (matches GStreamer; needed for AV1 later), else the universally-accepted rev 7.
static inline uint32_t nvenc_config_ver() {
  uint32_t major = g_nvenc_api_version.load(std::memory_order_relaxed) & 0xFFu;
  return nvenc_struct_ver(major >= 12 ? 8 : 7, (1u << 31));
}

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

  // Gated device-input: cached registered CUDA device resource + the per-frame
  // device buffer the conversion site produced. All zero/null when the gate is off.
  NV_ENC_REGISTERED_PTR registered_resource = nullptr;
  unsigned long long registered_base = 0;
  int registered_w = 0, registered_h = 0, registered_pitch = 0;
  unsigned long long dev_input_base = 0;
  int dev_input_pitch = 0;

  NvencEncoderState() {
    nvenc_funcs.version = nvenc_struct_ver(2, 0);
    init_params.version = nvenc_struct_ver(5, (1u << 31));
  }
};

static void* g_nvenc_lib_handle = nullptr;
typedef NVENCSTATUS(NVENCAPI* PFN_NvEncodeAPICreateInstance)(
  NV_ENCODE_API_FUNCTION_LIST*);
// One-time, thread-safe resolution of the process-global NVENC handle + API-version
// negotiation: concurrent multi-instance init otherwise races g_nvenc_lib_handle and the
// version probe. The per-instance function list is still populated per call (each instance
// owns its own), but only AFTER the negotiated version is published.
static std::once_flag g_nvenc_load_once;
static bool g_nvenc_load_ok = false;
static PFN_NvEncodeAPICreateInstance g_nvenc_create_instance = nullptr;

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
// Reader/writer lock serializing X Display connection setup/teardown across instances while
// keeping per-frame grabs concurrent. Each ScreenCapture instance opens/closes its own Display
// (XOpenDisplay/XCloseDisplay) from its capture thread; libxcb's global connection state is not
// safe against concurrent open/close even with per-Display locking enabled, so multi-instance
// churn corrupts the glibc heap ("tcache_thread_shutdown unaligned" / "[xcb] Aborting"). A plain
// mutex around the per-frame XShmGetImage would serialize every instance's grab (throughput cost);
// a shared_mutex lets grabs run in parallel yet still excludes them while any open/close runs.
//
//   WRITER (exclusive, std::unique_lock): connection setup/teardown. The initial-setup writer and
//     the reinit (resolution-change) setup writer each span the WHOLE setup phase as ONE exclusive
//     critical section -- XOpenDisplay, XGetWindowAttributes, XShmQueryExtension,
//     XFixesQueryExtension, the XShmCreateImage/shmget/shmat/XShmAttach+XSync retry, and the
//     XSetErrorHandler swap + g_shm_attach_failed read. Every libxcb-touching setup call (not just
//     the attach) is thus exclusive vs other instances' open/close AND vs grabs, because the
//     post-open setup-phase calls also mutate libxcb global state and otherwise race. Holding the
//     process-global writer across the whole setup also serializes the XSetErrorHandler swap, so
//     g_xshm_setup_mutex is removed and only ONE lock governs connection lifecycle. Teardown
//     (XShmDetach, XSync, XCloseDisplay) likewise runs under the writer.
//   READER (shared, std::shared_lock): the per-frame XShmGetImage only. Concurrent capture threads
//     grab in parallel, but every grab is blocked while any instance holds the writer.
//
// Lock-ordering invariant: this lock is NEVER acquired while any other lock is held, and no other
// lock is acquired while this is held -- so it can never participate in a deadlock cycle. The
// setup writer keeps this invariant by holding ONLY stack-local work inside the scope: the
// auto-adjust settings_mutex publish and load_watermark_image()/YUV-plane allocation are done
// AFTER the writer releases. CRITICAL: neither the reader nor the writer is ever held across a
// blocking encode, a capture-thread join, the per-frame loop, or the frame-pacing sleep:
//   - the reader scope is exactly the XShmGetImage call (result captured into a bool); all decode/
//     encode/cursor work happens AFTER the reader is released;
//   - the teardown/reinit writer scopes are exactly the bare open/close/attach/detach call(s); the
//     setup writer adds the contiguous setup-phase X calls + flag-read, no encode and no join.
//     Because the setup writer is held across the whole setup, every failure-path XCloseDisplay
//     inside it is a BARE call (the non-recursive mutex is already held) and the RAII lock
//     releases exactly once on every path.
// Anti-starvation: glibc's shared_mutex has no writer priority, so the grab loop yields on any
// iteration where the per-frame pacing sleep does not fire -- otherwise saturating readers could
// starve a concurrent stop_capture writer (XCloseDisplay) indefinitely.
// XInitThreads() (called once in PyInit__capture before any XOpenDisplay) adds the per-Display
// locking; this lock covers the connection-lifecycle gap XInitThreads alone does not.
static std::shared_mutex g_x_display_lifecycle_mutex;
// Writer-preference layer over g_x_display_lifecycle_mutex: glibc's std::shared_mutex is
// reader-preference, so a continuous stream of per-frame grabs (readers) at high/unbounded fps
// can starve a stopping instance's XCloseDisplay (writer) indefinitely, hanging stop_capture.
// Every writer increments this flag BEFORE blocking on the unique_lock (signaling intent) and
// decrements it the instant it acquires; each per-frame reader spins yield() while this is
// nonzero before taking its shared_lock, so readers back off the moment any writer is waiting
// and the writer acquires within one reader-drain. Steady state (no writer) costs readers one
// relaxed-fenced atomic load -- grab-vs-grab parallelism is preserved.
static std::atomic<unsigned> g_x_display_writers_waiting{0};
// Process-global serialization of libx264 session lifecycle. x264_encoder_open()/
// x264_encoder_close() touch libx264-internal process-global state (e.g. its CPU/thread
// setup); with multiple ScreenCapture instances churning CPU H264 (use_cpu=True) start/stop
// concurrently, unserialized open/close from different instances corrupts the glibc heap.
// Held (lock_guard) ONLY around the open/close calls -- never around x264_encoder_encode --
// so per-frame encoding stays fully parallel. Mirrors g_nv_filter_mutex / the connection-lifecycle
// writer (g_x_display_lifecycle_mutex): serialize lifecycle, leave the per-frame hot path parallel.
// Lock ordering: always acquired INSIDE MinimalEncoderStore::store_mutex (store_mutex first,
// then this global), which is consistent at every site so no deadlock is possible.
static std::mutex g_x264_lifecycle_mutex;

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
        // Serialize the close against any other instance's concurrent open/close
        // (process-global libx264 heap-corruption fix). Tight scope: close only.
        {
          std::lock_guard<std::mutex> x264_lock(g_x264_lifecycle_mutex);
          x264_encoder_close(encoders[i]);
        }
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
  // Resolve once (process-global); waiters block until the struct is fully published.
  std::call_once(g_cuda_load_once, []() {
    // libcuda.so.1 is the driver's runtime soname (present with any driver, 470-595);
    // libcuda.so (unversioned) is the dev symlink and is absent on runtime-only installs.
    g_cuda_lib_handle = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!g_cuda_lib_handle) g_cuda_lib_handle = dlopen("libcuda.so", RTLD_LAZY);
    if (!g_cuda_lib_handle) {
        std::cerr << "CUDA_API_LOAD: dlopen failed for libcuda.so.1 / libcuda.so" << std::endl;
        return;
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
        g_cuda_funcs = CudaFunctions{};  // value-init; CudaFunctions is non-trivial (no memset)
        return;
    }
    g_cuda_load_ok = true;
  });
  return g_cuda_load_ok;
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
  // Resolve the handle + negotiate the API version ONCE for the whole process; concurrent
  // multi-instance init would otherwise race g_nvenc_lib_handle / g_nvenc_api_version. The
  // per-instance function list (nvenc_funcs) is filled per call AFTER negotiation completes.
  std::call_once(g_nvenc_load_once, []() {
    const char* lib_names[] = {"libnvidia-encode.so.1", "libnvidia-encode.so"};
    for (const char* name : lib_names) {
      g_nvenc_lib_handle = dlopen(name, RTLD_LAZY | RTLD_GLOBAL);
      if (g_nvenc_lib_handle) {
        break;
      }
    }
    if (!g_nvenc_lib_handle) {
      return;
    }

    PFN_NvEncodeAPICreateInstance create_instance =
      (PFN_NvEncodeAPICreateInstance)dlsym(g_nvenc_lib_handle, "NvEncodeAPICreateInstance");
    if (!create_instance) {
      return;
    }

    // Negotiate the API version against the driver (GStreamer nvcodec approach): probe known API
    // versions newest-first and use the highest the driver accepts, so a newer-SDK build still runs
    // on older drivers down to NVENC 10.0 (~R445), covering CUDA 11 / driver 470. The struct
    // .version helpers tag every struct with old, widely-supported revisions + the chosen version.
    typedef NVENCSTATUS(NVENCAPI * PFN_GetMaxVer)(uint32_t*);
    auto get_max = (PFN_GetMaxVer)dlsym(g_nvenc_lib_handle, "NvEncodeAPIGetMaxSupportedVersion");
    uint32_t drv_max = 0;
    if (get_max) get_max(&drv_max);   // (major<<4)|minor; 0 -> rely on createInstance probing
    // Optional cap for testing/pinning a lower version, e.g. PIXELFLUX_NVENC_MAX_API="11.0".
    if (const char* cap = std::getenv("PIXELFLUX_NVENC_MAX_API")) {
      unsigned cmaj = 0, cmin = 0;
      if (sscanf(cap, "%u.%u", &cmaj, &cmin) == 2) {
        uint32_t capv = (cmaj << 4) | (cmin & 0xf);
        if (capv && (drv_max == 0 || capv < drv_max)) drv_max = capv;
      }
    }
    static const uint32_t kVersions[][2] = {   // {major, minor}, newest-first, floor NVENC 10.0
      {NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION},
      {12, 1}, {12, 0}, {11, 1}, {11, 0}, {10, 0},
    };
    NV_ENCODE_API_FUNCTION_LIST probe;   // local probe list; per-instance lists filled below
    NVENCSTATUS status = NV_ENC_ERR_INVALID_VERSION;
    for (const auto& v : kVersions) {
      uint32_t vv = (v[0] << 4) | v[1];
      if (drv_max != 0 && vv > drv_max) continue;   // driver can't support this version; skip
      g_nvenc_api_version.store(v[0] | (v[1] << 24), std::memory_order_relaxed);
      memset(&probe, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
      probe.version = nvenc_struct_ver(2, 0);
      status = create_instance(&probe);
      if (status == NV_ENC_SUCCESS && probe.nvEncOpenEncodeSessionEx) break;
      status = NV_ENC_ERR_INVALID_VERSION;
    }
    if (status != NV_ENC_SUCCESS) {
      g_nvenc_api_version.store(NVENCAPI_VERSION, std::memory_order_relaxed);
      return;
    }
    g_nvenc_create_instance = create_instance;
    g_nvenc_load_ok = true;
    uint32_t a = g_nvenc_api_version.load(std::memory_order_relaxed);
    std::cerr << "[pixelflux] NVENC API version negotiated: " << (a & 0xFFu) << "."
              << ((a >> 24) & 0xFFu) << std::endl;
  });

  if (!g_nvenc_load_ok || !g_nvenc_create_instance) {
    return false;
  }
  // Populate THIS instance's function list with the negotiated version (set-once above).
  memset(&nvenc_funcs, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
  nvenc_funcs.version = nvenc_struct_ver(2, 0);
  NVENCSTATUS status = g_nvenc_create_instance(&nvenc_funcs);
  if (status != NV_ENC_SUCCESS || !nvenc_funcs.nvEncOpenEncodeSessionEx) {
    memset(&nvenc_funcs, 0, sizeof(NV_ENCODE_API_FUNCTION_LIST));
    return false;
  }
  return true;
}

// --- Multi-GPU NVENC fix: GET_ATTACHED_IDS filter ----------------
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
  unsigned want_bus = (gpu_id >> 8) & 0xFFu;  // bus number (most common encoding)
  unsigned want_full = gpu_id >> 8;           // some gpuIds encode (domain << 8) | bus
  DIR* dir = opendir("/proc/driver/nvidia/gpus");
  if (!dir) return -1;
  int minor = -1;
  struct dirent* ent;
  while ((ent = readdir(dir)) != nullptr) {
    unsigned dom, bus, slot, fn;
    if (sscanf(ent->d_name, "%x:%x:%x.%x", &dom, &bus, &slot, &fn) != 4) continue;
    // Match the bus number (most common case) OR the domain:bus combined value (larger
    // gpuIds), mirroring the reference interposer's dual strategy so a gpuId whose high bits
    // don't cleanly carry the domain still resolves instead of dropping a valid GPU.
    if (bus != want_bus && ((dom << 8) | bus) != want_full) continue;
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

// Resolve a DT_ d_ptr to an absolute address. On glibc the loader rewrites these to
// absolute; on musl/Alpine they stay FILE-RELATIVE -> add dlpi_addr (base). Heuristic:
// values below base are relative offsets; values at/above base are already absolute.
static inline uintptr_t nv_dyn_addr(uintptr_t base, ElfW(Addr) v) {
  return (static_cast<uintptr_t>(v) < base) ? base + static_cast<uintptr_t>(v)
                                            : static_cast<uintptr_t>(v);
}

void nv_patch_ioctl_got(uintptr_t base, const ElfW(Dyn)* dyn) {
  const ElfW(Sym)* symtab = nullptr;
  const char* strtab = nullptr;
  ElfW(Rela)* jmprel = nullptr;
  size_t pltrelsz = 0;
  for (const ElfW(Dyn)* d = dyn; d->d_tag != DT_NULL; d++) {
    if (d->d_tag == DT_SYMTAB) symtab = reinterpret_cast<const ElfW(Sym)*>(nv_dyn_addr(base, d->d_un.d_ptr));
    else if (d->d_tag == DT_STRTAB) strtab = reinterpret_cast<const char*>(nv_dyn_addr(base, d->d_un.d_ptr));
    else if (d->d_tag == DT_JMPREL) jmprel = reinterpret_cast<ElfW(Rela)*>(nv_dyn_addr(base, d->d_un.d_ptr));
    else if (d->d_tag == DT_PLTRELSZ) pltrelsz = d->d_un.d_val;
  }
  if (!symtab || !strtab || !jmprel || !pltrelsz) return;
  // Best-effort: bail if any resolved table isn't in a readable mapping (a bad
  // relative/absolute guess would otherwise fault on the first dereference below).
  auto readable = [](const void* p) {
    int prot = nv_page_prot(reinterpret_cast<uintptr_t>(p));
    return prot < 0 || (prot & PROT_READ);
  };
  if (!readable(symtab) || !readable(strtab) || !readable(jmprel)) return;
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

// --- optional CUDA (NVRTC) ARGB->NV12 color conversion -----------
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
// Strided 2D copy (cuMemcpy2D_v2) to download the pitched device NV12 to tight host planes
// in one call per plane instead of a per-row loop. Layout matches the CUDA driver ABI
// (CUDA_MEMCPY2D, stable since CUDA 3.x); CUarray is never used here (left null).
typedef struct CUarray_st* CUarray;
enum { CU_MEMORYTYPE_HOST_ = 1, CU_MEMORYTYPE_DEVICE_ = 2 };
struct CUDA_MEMCPY2D {
  size_t srcXInBytes, srcY;
  unsigned int srcMemoryType; const void* srcHost; CUdptr srcDevice; CUarray srcArray; size_t srcPitch;
  size_t dstXInBytes, dstY;
  unsigned int dstMemoryType; void* dstHost; CUdptr dstDevice; CUarray dstArray; size_t dstPitch;
  size_t WidthInBytes, Height;
};
typedef int (*tMemcpy2D)(const CUDA_MEMCPY2D*);   // cuMemcpy2D_v2
// Device-compute-capability query so the kernel is compiled for the ACTUAL device's
// virtual arch (PTX runs only on its own virtual arch and newer -- a fixed compute_52
// PTX cannot JIT on a Kepler sm_35). Resolved best-effort via dlsym; null -> fallback list.
typedef int (*tDevAttr)(int*, int, CUdevice);     // cuDeviceGetAttribute
typedef int (*tCtxGetDev)(CUdevice*);             // cuCtxGetDevice
// CUdevice attribute ids (stable CUDA-driver enum values).
static const int CU_DEV_ATTR_CC_MAJOR = 75;       // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
static const int CU_DEV_ATTR_CC_MINOR = 76;       // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

static std::mutex g_mutex;
// Per-CUcontext cache. Each ScreenCaptureModule instance owns its OWN CUcontext, so the
// compiled module function and device buffers must be keyed by context: a single global
// cache shared across instances leaked GPU memory (it zeroed the outgoing context's live
// handles on every context flip without freeing them) and thrashed the module every frame
// with 2+ concurrent NVENC instances. The entry's GPU allocations + module are reclaimed
// by cuCtxDestroy when the owning context is destroyed (destroy_context erases the entry).
struct CacheEntry {
  bool ready = false;
  CUfunc fn = nullptr;
  CUdptr d_argb = 0, d_y = 0, d_uv = 0;
  size_t cap_argb = 0, cap_y = 0, cap_uv = 0;
  CUdptr d_nv12 = 0;              // contiguous pitched NV12 for gated device-input NVENC
  size_t cap_nv12 = 0;
};
static std::map<CUcontext, CacheEntry> g_cache;
static tAlloc g_alloc; static tFree g_free; static tH2D g_h2d; static tD2H g_d2h;
static tLaunch g_launch; static tSetCtx g_setctx; static tSync g_sync;
static tMemcpy2D g_memcpy2d = nullptr;  // optional fast path; null -> per-row download fallback
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
    // Try the NVRTC sonames across CUDA 11/12/13 (+ the unversioned dev symlink). The wheel's
    // libnvrtc (any version) is RTLD_GLOBAL-preloaded by __init__._preload_wheel_libnvrtc, so a
    // matching soname here returns that handle; system/conda installs resolve via ldconfig.
    const char* nvrtc_names[] = {"libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2",
                                 "libnvrtc.so.11", "libnvrtc.so"};
    for (const char* n : nvrtc_names) {
      g_nvrtc_handle = dlopen(n, RTLD_NOW | RTLD_GLOBAL);
      if (g_nvrtc_handle) break;
    }
  }
  void* rt=g_nvrtc_handle;
  // If no handle (e.g. a preloaded NVRTC whose soname isn't in the list), resolve the symbols
  // from the global namespace so the RTLD_GLOBAL preload still satisfies us.
  auto nsym=[&](const char* s)->void*{ return rt ? dlsym(rt,s) : dlsym(RTLD_DEFAULT,s); };
  auto nvCreate=(tNvCreate)nsym("nvrtcCreateProgram");
  auto nvCompile=(tNvCompile)nsym("nvrtcCompileProgram");
  auto nvPtxSz=(tNvPtxSz)nsym("nvrtcGetPTXSize");
  auto nvPtx=(tNvPtx)nsym("nvrtcGetPTX");
  auto nvDestroy=(tNvDestroy)nsym("nvrtcDestroyProgram");
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

static bool init_locked(CUcontext ctx, CacheEntry& e) {
  if (!g_cuda_lib_handle) return false;
  // Resolve the process-global driver entry points ONCE: the per-context GPU pipeline reads
  // them lock-free, so re-writing them on every new-context init (while another context's
  // convert reads them) would be a data race. Set-once under g_mutex (init_locked is only
  // reached via get_entry_locked, which holds g_mutex).
  static bool funcs_loaded = false;
  if (!funcs_loaded) {
    g_alloc=(tAlloc)dlsym(g_cuda_lib_handle,"cuMemAlloc_v2");
    g_free=(tFree)dlsym(g_cuda_lib_handle,"cuMemFree_v2");
    g_h2d=(tH2D)dlsym(g_cuda_lib_handle,"cuMemcpyHtoD_v2");
    g_d2h=(tD2H)dlsym(g_cuda_lib_handle,"cuMemcpyDtoH_v2");
    g_launch=(tLaunch)dlsym(g_cuda_lib_handle,"cuLaunchKernel");
    g_setctx=(tSetCtx)dlsym(g_cuda_lib_handle,"cuCtxSetCurrent");
    g_sync=(tSync)dlsym(g_cuda_lib_handle,"cuCtxSynchronize");
    g_memcpy2d=(tMemcpy2D)dlsym(g_cuda_lib_handle,"cuMemcpy2D_v2");  // optional fast path
    if(!g_alloc||!g_free||!g_h2d||!g_d2h||!g_launch||!g_setctx||!g_sync) return false;
    funcs_loaded = true;
  }
  auto modLoad=(tModLoad)dlsym(g_cuda_lib_handle,"cuModuleLoadDataEx");
  auto modFn=(tModFn)dlsym(g_cuda_lib_handle,"cuModuleGetFunction");
  if(!modLoad||!modFn) return false;
  // Make ctx current BEFORE compiling so the device-cc query (cuCtxGetDevice in
  // compile_ptx_once) targets THIS context's device and picks its exact virtual arch.
  g_setctx(ctx);
  if(!compile_ptx_once()) return false;
  // The CUmod is never cuModuleUnload'd: it is context-scoped and dies with its CUcontext
  // (destroy_context erases the cache entry before cuCtxDestroy), so there is no per-module process leak.
  CUmod mod=nullptr; if(modLoad(&mod,g_ptx.data(),0,nullptr,nullptr)!=0) return false;
  if(modFn(&e.fn,mod,"argb_to_nv12")!=0||!e.fn) return false;
  return true;
}

// Look up (or lazily create+init) the cache entry for ctx. Returns nullptr if init failed;
// a failed entry is left in the map (ready=false) so we don't retry the full compile every
// frame -- matching the old single-context g_tried/g_ready latch, but per context. Assumes
// g_mutex held.
static CacheEntry* get_entry_locked(CUcontext ctx) {
  auto it = g_cache.find(ctx);
  if (it == g_cache.end()) {
    CacheEntry e;
    e.ready = init_locked(ctx, e);
    it = g_cache.emplace(ctx, e).first;
  }
  return it->second.ready ? &it->second : nullptr;
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
  // Look up (or lazily build) this context's cache entry under g_mutex (map access + one-time
  // init only). The GPU pipeline below then runs WITHOUT the lock: each CUcontext is driven by
  // a single capture thread (one per instance), so no two threads touch the same entry, and
  // converts on different contexts (multi-display) overlap instead of serializing on the
  // blocking cuCtxSynchronize. The entry pointer stays valid (destroy_context only erases a
  // context after its capture thread is joined; std::map nodes are address-stable).
  CacheEntry* e;
  { std::lock_guard<std::mutex> lk(g_mutex); e = get_entry_locked(ctx); }
  if (!e) return false;
  if (g_setctx(ctx) != 0) return false;
  size_t argb_sz = (size_t)argb_stride * h, y_sz = (size_t)w * h, uv_sz = (size_t)w * (h / 2);
  if (!ensure_buf(e->d_argb, e->cap_argb, argb_sz) || !ensure_buf(e->d_y, e->cap_y, y_sz) || !ensure_buf(e->d_uv, e->cap_uv, uv_sz)) return false;
  if (g_h2d(e->d_argb, argb, argb_sz) != 0) return false;
  int aw = argb_stride, yw = w, uvw = w, ww = w, hh = h;
  void* args[] = {&e->d_argb, &aw, &e->d_y, &yw, &e->d_uv, &uvw, &ww, &hh};
  if (g_launch(e->fn, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, nullptr, args, nullptr) != 0) return false;
  if (g_sync() != 0) return false;
  if (g_d2h(y, e->d_y, y_sz) != 0) return false;
  if (g_d2h(uv, e->d_uv, uv_sz) != 0) return false;
  return true;
}

// Gated (PIXELFLUX_NVENC_DEVICE_INPUT) device-input variant. Same conversion, but writes
// a CONTIGUOUS 256-pitch NV12 into the per-context d_nv12 (Y at base, UV at base+pitch*h) so NVENC can
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
// skip_host_download: in H.264 FullFrame STREAMING mode the change-detection hash is never
// computed (the loop sends every stripe) and NVENC consumes the device buffer directly, so the
// per-frame device->host NV12 copy is pure overhead -> skip it. The host planes then go stale;
// encode_fullframe_nvenc refreshes them from this same device buffer (download_nv12) ONLY on the
// rare nvEncMapInputResource-failure fallback, so non-streaming hashing and that fallback stay correct.
bool argb_to_nv12_device(CUcontext ctx, const uint8_t* argb, int argb_stride,
                         uint8_t* y, int y_stride, uint8_t* uv, int uv_stride,
                         int w, int h, unsigned long long* out_base, int* out_pitch,
                         bool skip_host_download) {
  if (!ctx || w <= 0 || h <= 0 || (w & 1) || (h & 1)) return false;
  if (y_stride != w || uv_stride != w) return false;
  static const bool disabled = (std::getenv("PIXELFLUX_NO_CUDA_CONVERT") != nullptr);
  if (disabled) return false;
  // g_mutex guards only the map lookup/init; the GPU pipeline runs lock-free (see argb_to_nv12).
  CacheEntry* e;
  { std::lock_guard<std::mutex> lk(g_mutex); e = get_entry_locked(ctx); }
  if (!e) return false;
  if (g_setctx(ctx) != 0) return false;
  int pitch = (w + 255) & ~255;
  size_t argb_sz = (size_t)argb_stride * h;
  size_t nv12_sz = (size_t)pitch * h + (size_t)pitch * (h / 2);
  if (!ensure_buf(e->d_argb, e->cap_argb, argb_sz) || !ensure_buf(e->d_nv12, e->cap_nv12, nv12_sz)) return false;
  if (g_h2d(e->d_argb, argb, argb_sz) != 0) return false;
  CUdptr y_base = e->d_nv12;
  CUdptr uv_base = e->d_nv12 + (CUdptr)((size_t)pitch * h);
  int aw = argb_stride, yw = pitch, uvw = pitch, ww = w, hh = h;
  void* args[] = {&e->d_argb, &aw, &y_base, &yw, &uv_base, &uvw, &ww, &hh};
  if (g_launch(e->fn, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, nullptr, args, nullptr) != 0) return false;
  if (g_sync() != 0) return false;
  // Download the pitched device NV12 to the tight host planes for the change-detection hash.
  // One strided 2D copy per plane (cuMemcpy2D) instead of h + h/2 single-row copies (1620 per
  // frame at 1080p); falls back to the per-row loop if cuMemcpy2D_v2 wasn't resolved. Skipped
  // in streaming mode (host planes refreshed on demand on the map-failure fallback only).
  if (!skip_host_download) {
    if (g_memcpy2d) {
      CUDA_MEMCPY2D cp = {};
      cp.srcMemoryType = CU_MEMORYTYPE_DEVICE_; cp.dstMemoryType = CU_MEMORYTYPE_HOST_;
      cp.WidthInBytes = (size_t)w;
      cp.srcDevice = y_base; cp.srcPitch = (size_t)pitch;
      cp.dstHost = y; cp.dstPitch = (size_t)y_stride; cp.Height = (size_t)h;
      if (g_memcpy2d(&cp) != 0) return false;
      cp.srcDevice = uv_base;
      cp.dstHost = uv; cp.dstPitch = (size_t)uv_stride; cp.Height = (size_t)(h / 2);
      if (g_memcpy2d(&cp) != 0) return false;
    } else {
      for (int r = 0; r < h; ++r)
        if (g_d2h(y + (size_t)r * y_stride, y_base + (CUdptr)((size_t)r * pitch), (size_t)w) != 0) return false;
      for (int r = 0; r < h / 2; ++r)
        if (g_d2h(uv + (size_t)r * uv_stride, uv_base + (CUdptr)((size_t)r * pitch), (size_t)w) != 0) return false;
    }
  }
  *out_base = (unsigned long long)e->d_nv12;
  *out_pitch = pitch;
  return true;
}

// Copy a pitched device NV12 (dev_base: Y at base, UV at base + dev_pitch*h) into NV12
// destinations (e.g. the locked NVENC input buffer), both pitched. Used only as the streaming
// map-failure fallback so the device buffer (the source of truth) still reaches the encoder
// when argb_to_nv12_device skipped the host download. ctx must be current on the calling
// thread (encode_fullframe_nvenc pushes it). Returns false on any error.
bool download_nv12(CUcontext ctx, unsigned long long dev_base, int dev_pitch,
                   uint8_t* y_dst, int y_dst_pitch, uint8_t* uv_dst, int uv_dst_pitch,
                   int w, int h) {
  if (!ctx || !dev_base || dev_pitch <= 0 || w <= 0 || h <= 0) return false;
  if (g_setctx(ctx) != 0) return false;
  CUdptr y_base = (CUdptr)dev_base;
  CUdptr uv_base = (CUdptr)dev_base + (CUdptr)((size_t)dev_pitch * h);
  if (g_memcpy2d) {
    CUDA_MEMCPY2D cp = {};
    cp.srcMemoryType = CU_MEMORYTYPE_DEVICE_; cp.dstMemoryType = CU_MEMORYTYPE_HOST_;
    cp.WidthInBytes = (size_t)w;
    cp.srcDevice = y_base; cp.srcPitch = (size_t)dev_pitch;
    cp.dstHost = y_dst; cp.dstPitch = (size_t)y_dst_pitch; cp.Height = (size_t)h;
    if (g_memcpy2d(&cp) != 0) return false;
    cp.srcDevice = uv_base;
    cp.dstHost = uv_dst; cp.dstPitch = (size_t)uv_dst_pitch; cp.Height = (size_t)(h / 2);
    if (g_memcpy2d(&cp) != 0) return false;
  } else {
    for (int r = 0; r < h; ++r)
      if (g_d2h(y_dst + (size_t)r * y_dst_pitch, y_base + (CUdptr)((size_t)r * dev_pitch), (size_t)w) != 0) return false;
    for (int r = 0; r < h / 2; ++r)
      if (g_d2h(uv_dst + (size_t)r * uv_dst_pitch, uv_base + (CUdptr)((size_t)r * dev_pitch), (size_t)w) != 0) return false;
  }
  return true;
}


// Destroy a per-instance CUDA context safely w.r.t. the per-context color-convert cache.
// Holding g_mutex across the destroy serializes it against any in-flight convert on any
// context (avoids an intermittent exit crash), and we erase ONLY this context's entry, so
// one instance's teardown can't drop another's live cache. This context's entry buffers +
// module are reclaimed by cuCtxDestroy, so erasing the entry (dropping the handles) here is
// not a leak and not a double-free.
void destroy_context(CUcontext ctx) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_cache.erase(ctx);
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
  // Set in start_capture / cleared in stop_capture; is_capturing reads this instead
  // of capture_thread.joinable() (which races stop_capture's GIL-released join()).
  std::atomic<bool> running{false};
  std::thread capture_thread;
  // Serializes start_capture()/stop_capture() on ONE instance: with the GIL released,
  // concurrent calls otherwise race the non-atomic capture_thread (joinable-check vs
  // join vs reassignment) -> std::terminate. Distinct from settings_mutex/nvenc_mutex_;
  // those are NOT held across join() (deadlock risk). The same-thread re-entrancy guard
  // (capture_thread == this_thread) is checked via capture_thread_id_ BEFORE this lock.
  std::mutex thread_lifecycle_mutex_;
  // Snapshot of capture_thread's id, published under thread_lifecycle_mutex_ right
  // after the thread is spawned and cleared after it is joined. Lets a re-entrant
  // stop_capture()/start_capture() from the capture thread itself (e.g. a Python
  // stripe callback calling sc.stop_capture()) detect itself and short-circuit
  // WITHOUT reading the non-atomic capture_thread or blocking on the lifecycle
  // lock — a blocking self-join would deadlock another thread already joining this
  // one in stop_capture_locked(). Mirrors pcmflux's validated pattern.
  std::atomic<std::thread::id> capture_thread_id_{};
  // Set when a Python stripe callback drops the LAST ref to its OWN ScreenCapture
  // (ScreenCapture_dealloc then runs re-entrantly ON this capture thread, with
  // capture_loop still on the stack below). The module cannot be deleted inline there
  // -- instead ownership is transferred to the capture thread, which deletes itself as
  // its very last act (capture_thread_main, after capture_loop fully returns). The
  // thread is detached in that same handoff so its own std::thread member can be
  // destroyed without join()/std::terminate. Only ever set from the capture thread.
  std::atomic<bool> delete_on_exit_{false};
  // True iff the caller is running ON the capture thread (e.g. a re-entrant
  // dealloc/clear from a stripe callback). Reads only the atomic id snapshot.
  bool on_capture_thread() const {
    return capture_thread_id_.load(std::memory_order_acquire) ==
           std::this_thread::get_id();
  }
  // Called from ScreenCapture_dealloc ONLY when on_capture_thread() is true: hand this
  // module off to its own capture thread for self-delete-on-exit. Sequenced entirely on
  // the capture thread (synchronous up-stack from capture_loop's stripe dispatch), so the
  // member writes here race with nothing. Clearing stripe_callback/user_data stops the
  // current frame's remaining stripe dispatches from re-entering the trampoline with the
  // now-dangling user_data (the dispatch loop checks stripe_callback != nullptr and frees
  // the buffers itself instead). Detach makes the running thread's own std::thread member
  // safe to destroy during the later delete this. No external ref to the Python object
  // can exist here (its refcount just hit 0 under the GIL), so no concurrent
  // start/stop_capture can touch capture_thread.
  void request_self_delete() {
    stop_requested = true;
    stripe_callback = nullptr;
    user_data = nullptr;
    delete_on_exit_.store(true, std::memory_order_release);
    if (capture_thread.joinable()) {
      capture_thread.detach();
    }
  }
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
    // Thread entry: runs capture_loop() to completion, then -- if a re-entrant
    // self-teardown handed this module off via request_self_delete() -- releases the
    // per-instance hardware encoders and `delete this` as the thread's final action.
    // Doing the delete out here (not inside capture_loop) guarantees all of
    // capture_loop's member-referencing RAII (e.g. its running-flag guard) has already
    // run before the object is freed.
    void capture_thread_main();
    void capture_loop();
    // Teardown body of stop_capture(); caller MUST hold thread_lifecycle_mutex_ (lets
    // start_capture() stop a prior run without re-locking the non-recursive lifecycle mutex).
    void stop_capture_locked();
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

  // get_current_settings() was removed: it was dead (never called) and returned a
  // CaptureSettings whose const char* fields borrowed .c_str() into mutable members
  // (dangling once settings_mutex was released).

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
    // Re-entrant start from the capture thread itself (a stripe callback calling
    // sc.start_capture()): it can't join/recreate the thread it's running on, so it
    // just keeps the thread alive and undoes any stop_requested a nested
    // stop_capture() set. Short-circuit BEFORE the lifecycle lock so we never block
    // on (or self-join under) a lock another thread already holds while joining us.
    if (capture_thread_id_.load(std::memory_order_acquire) ==
        std::this_thread::get_id()) {
      stop_requested = false;
      return;
    }
    // Hold the lifecycle lock across the joinable-check + stop + thread reassignment so a
    // concurrent stop_capture() can't race the non-atomic capture_thread. Use the *_locked
    // teardown (we already hold the lock; the mutex is non-recursive).
    std::lock_guard<std::mutex> lifecycle_lock(thread_lifecycle_mutex_);
    if (capture_thread.joinable()) {
      stop_capture_locked();
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
    // Spawn the capture thread via capture_thread_main (which runs capture_loop and then
    // performs the re-entrant self-delete-on-exit handoff, if requested). The thread's id
    // is the same whichever member it starts in, so the snapshot below is still correct.
    capture_thread = std::thread(&ScreenCaptureModule::capture_thread_main, this);
    // Publish the new thread's id (under the lifecycle lock) so a re-entrant
    // stop/start from within the capture thread can recognize itself and short-circuit.
    capture_thread_id_.store(capture_thread.get_id(), std::memory_order_release);
    // running is set inside capture_loop AFTER init succeeds (and cleared via RAII on any
    // exit), so is_capturing reflects real thread health, not merely "thread spawned".
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
    // Re-entrant stop from the capture thread itself (a Python stripe callback ->
    // sc.stop_capture(), or dealloc/GC on that thread): it must NOT join itself
    // (join() on the current thread throws) and never touches the non-atomic
    // capture_thread, so it returns here WITHOUT taking the lifecycle lock. Taking it
    // would deadlock if another thread already holds it while blocked in
    // capture_thread.join() waiting for this very thread to exit. The atomic
    // stop_requested set above suffices to wind the capture loop down; this reads only
    // the atomic id snapshot, no non-atomic state. (This is the check that breaks the
    // re-entrant-stop deadlock; the old guard lived inside stop_capture_locked, AFTER the lock.)
    if (capture_thread_id_.load(std::memory_order_acquire) ==
        std::this_thread::get_id()) {
      return;
    }
    // Serialize against a concurrent start_capture()/stop_capture() on this instance.
    std::lock_guard<std::mutex> lifecycle_lock(thread_lifecycle_mutex_);
    stop_capture_locked();
}

// Teardown body; caller holds thread_lifecycle_mutex_. No other lock is held across join().
// Both callers (stop_capture/start_capture) short-circuit a re-entrant capture-thread
// caller BEFORE the lock via capture_thread_id_, so the thread joined here is never the
// current thread (join() on self would throw); no same-thread guard is needed here.
void ScreenCaptureModule::stop_capture_locked() {
    stop_requested = true;
    if (capture_thread.joinable()) {
      capture_thread.join();
      // Clear the published id now that the thread is gone, so a later self-check can't
      // alias a future thread that happens to reuse this id.
      capture_thread_id_.store(std::thread::id{}, std::memory_order_release);
    }
    running = false;
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
    // Unregister the cached device-input resource BEFORE destroying the session.
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

      reconfigure_params.version = nvenc_struct_ver(1, (1u << 31));
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
  session_params.version = nvenc_struct_ver(1, 0);
  session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
  session_params.device = nvenc_state_.cuda_context;
  session_params.apiVersion = g_nvenc_api_version.load(std::memory_order_relaxed);

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
  nvenc_state_.init_params.version = nvenc_struct_ver(5, (1u << 31));
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
  preset_config.version = nvenc_struct_ver(4, (1u << 31));
  preset_config.presetCfg.version = nvenc_config_ver();

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
      nvenc_state_.encode_config.version = nvenc_config_ver();
    } else {
      nvenc_state_.encode_config = preset_config.presetCfg;
      nvenc_state_.encode_config.version = nvenc_config_ver();
    }
  } else {
    std::cerr << "NVENC_INIT_WARN: nvEncGetEncodePresetConfigEx not available. Using manual "
                 "config."
              << std::endl;
    memset(&nvenc_state_.encode_config, 0, sizeof(nvenc_state_.encode_config));
    nvenc_state_.encode_config.version = nvenc_config_ver();
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
  // Infinite GOP (no auto keyframes; IDRs are forced explicitly). idrPeriod set to match
  // below so the streamed profile/level is deterministic from frame 1.
  nvenc_state_.encode_config.gopLength = NVENC_INFINITE_GOPLENGTH;
  nvenc_state_.encode_config.frameIntervalP = 1;

  NV_ENC_CONFIG_H264* h264_cfg = &nvenc_state_.encode_config.encodeCodecConfig.h264Config;
  // Decodable High 4:4:4 = chromaFormatIDC=3, separate_colour_plane_flag=0 (leave the default;
  // 1 yields a stream decoders reject). 4:4:4 also requires the cuCtxCreate_v2 context (see
  // LoadCudaApi); the v1 context makes NVENC's 4:4:4 CUDA ops fail.
  h264_cfg->chromaFormatIDC = use_yuv444 ? 3 : 1;
  // Pin an explicit level/idrPeriod so the FIRST access unit declares the final High
  // profile + level (the client parses it from the SPS). Leaving level at AUTOSELECT lets
  // the first frames advertise a lower level, forcing WebCodecs/D3D11 to re-init when it
  // later bumps. Compute the MINIMUM level whose Annex-A MaxFS (frame size in macroblocks)
  // and MaxMBPS (macroblock rate) both fit this resolution+fps, with a 5.2 floor so <=4K
  // stays at the prior deterministic High@5.2; only >4K (MBs > 36864) computes 6.x.
  // NOTE: current NVENC H.264 on this HW (e.g. L40S) caps at level 5.2 / ~4096 width and
  // will REJECT a computed 6.x level ('Invalid Level'), at which point the existing x264
  // fallback takes over and encodes the >4K stream. The computation itself is still the
  // correct minimum level and is honored by HW/codecs that DO support levels >5.2; we do
  // not clamp it here so those paths stay accurate. Computed once from the fixed
  // width/height/fps -> deterministic from frame 1.
  {
    uint64_t mbs = (uint64_t)((width + 15) / 16) * ((height + 15) / 16);
    uint64_t mbps = (uint64_t)(mbs * (fps > 0.0 ? fps : 1.0));  // macroblocks/sec
    struct LevelCap { uint32_t level; uint64_t max_fs; uint64_t max_mbps; };
    static const LevelCap kLevels[] = {   // floor 5.2; ascending; Annex-A MaxFS/MaxMBPS
      { NV_ENC_LEVEL_H264_52,  36864ull,  2073600ull },
      { NV_ENC_LEVEL_H264_60, 139264ull,  4177920ull },
      { NV_ENC_LEVEL_H264_61, 139264ull,  8355840ull },
      { NV_ENC_LEVEL_H264_62, 139264ull, 16711680ull },
    };
    uint32_t chosen = NV_ENC_LEVEL_H264_62;   // highest if nothing fits (caller width cap still applies)
    for (const auto& lc : kLevels) {
      if (mbs <= lc.max_fs && mbps <= lc.max_mbps) { chosen = lc.level; break; }
    }
    h264_cfg->level = chosen;
  }
  h264_cfg->idrPeriod = NVENC_INFINITE_GOPLENGTH;
  // VUI: signal the colour description consistently so the decoder never re-derives it.
  // Match the x264 path / libyuv + CUDA conversion math exactly: BT.709 primaries+transfer,
  // BT.601 (SMPTE170M) matrix coeffs, LIMITED range. The 4:4:4 path feeds libyuv ARGBToI444
  // (studio/limited range) just like the 4:2:0 path feeds ARGBToNV12/the limited-range CUDA
  // kernel, so the flag must be limited for both -- signalling full range for 4:4:4 mismatched
  // the actual limited-range samples and washed out the decode.
  h264_cfg->h264VUIParameters.videoSignalTypePresentFlag = 1;
  h264_cfg->h264VUIParameters.videoFullRangeFlag = 0;
  h264_cfg->h264VUIParameters.colourDescriptionPresentFlag = 1;
  h264_cfg->h264VUIParameters.colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES_BT709;
  h264_cfg->h264VUIParameters.transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
  h264_cfg->h264VUIParameters.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS_SMPTE170M;
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
    icp.version = nvenc_struct_ver(1, 0);
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
    ocp.version = nvenc_struct_ver(1, 0);
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

  // Gated device-input: register+map the contiguous device NV12 the conversion site
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
  // RAII: if anything throws while the device-input resource is mapped and/or the encoder
  // context is pushed on this thread, unmap and pop/clear before unwinding. The success path
  // below nulls mapped_resource and clears the ctx flags, so the guard is then a no-op.
  struct DeviceInputGuard {
    ScreenCaptureModule* self;
    NV_ENC_INPUT_PTR& mapped;
    bool& pushed;
    bool& set;
    ~DeviceInputGuard() {
      if (mapped && self->nvenc_state_.nvenc_funcs.nvEncUnmapInputResource) {
        self->nvenc_state_.nvenc_funcs.nvEncUnmapInputResource(
            self->nvenc_state_.encoder_session, mapped);
        mapped = nullptr;
      }
      if (pushed && g_cuda_funcs.pfn_cuCtxPopCurrent) {
        CUcontext popped = nullptr;
        g_cuda_funcs.pfn_cuCtxPopCurrent(&popped);
        pushed = false;
      } else if (set && g_cuda_funcs.pfn_cuCtxSetCurrent) {
        g_cuda_funcs.pfn_cuCtxSetCurrent(nullptr);
        set = false;
      }
    }
  } device_input_guard{this, mapped_resource, nvenc_ctx_pushed, nvenc_ctx_set};
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
      rr.version = nvenc_struct_ver(3, 0);
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
      mr.version = nvenc_struct_ver(4, 0);
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
    lip.version = nvenc_struct_ver(1, 0);
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

    // Map-failure fallback when device-input conversion ran (dev_input_base != 0, always NV12):
    // the host planes may be stale (streaming skipped the device->host download), so refresh the
    // locked buffer directly from the authoritative device NV12 instead of the host planes.
    bool filled_from_device = false;
    if (!is_i444 && cuda_convert::nvenc_device_input_enabled() && nvenc_state_.dev_input_base != 0) {
      filled_from_device = cuda_convert::download_nv12(
          nvenc_state_.cuda_context, nvenc_state_.dev_input_base, nvenc_state_.dev_input_pitch,
          y_dst, locked_pitch, uv_or_u_dst, locked_pitch, width, height);
    }

    if (filled_from_device) {
      // locked buffer already populated from the device NV12; nothing further to copy.
    } else if (is_i444) {
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
  pp.version = nvenc_struct_ver(4, (1u << 31));
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

  // Device-input was consumed by the (synchronous) encode; unmap it. A throw above is
  // handled by device_input_guard, which unmaps and pops/clears on unwind.
  if (mapped_resource && nvenc_state_.nvenc_funcs.nvEncUnmapInputResource) {
    nvenc_state_.nvenc_funcs.nvEncUnmapInputResource(nvenc_state_.encoder_session, mapped_resource);
    mapped_resource = nullptr;
  }

  // Restore the thread's prior context now the device-input span is done (pop balances push,
  // else unset SetCurrent). Clear the flags so device_input_guard becomes a no-op; a throw
  // before here is handled by that guard.
  if (nvenc_ctx_pushed && g_cuda_funcs.pfn_cuCtxPopCurrent) {
    CUcontext popped = nullptr;
    g_cuda_funcs.pfn_cuCtxPopCurrent(&popped);
    nvenc_ctx_pushed = false;
  } else if (nvenc_ctx_set && g_cuda_funcs.pfn_cuCtxSetCurrent) {
    g_cuda_funcs.pfn_cuCtxSetCurrent(nullptr);
    nvenc_ctx_set = false;
  }

  NV_ENC_LOCK_BITSTREAM lbs = {0};
  lbs.version = nvenc_struct_ver(1, 0);
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

// Thread entry. Runs the capture loop, then performs self-delete-on-exit if a re-entrant
// teardown (a stripe callback dropping the last ref to its own ScreenCapture) handed this
// module off via request_self_delete(). capture_loop() has fully returned here, so all of
// its RAII (the running-flag guard etc.) has already touched members for the last time;
// only then is it safe to free the object. The capture thread was detached in
// request_self_delete(), so destroying its own std::thread member during `delete this`
// (inside ~ScreenCaptureModule) does not std::terminate, and the destructor's stop_capture()
// short-circuits because it is still the capture thread. NOTHING may touch `this` after the
// delete; capture_thread_main returns immediately, ending the detached thread.
void ScreenCaptureModule::capture_thread_main() {
    capture_loop();
    if (delete_on_exit_.load(std::memory_order_acquire)) {
        // Release this instance's hardware encoder resources first: ~ScreenCaptureModule's
        // stop_capture() self-short-circuits (we are the capture thread) and so would skip
        // the usual reset_*(); both resets are self-locking and idempotent (no-op if not
        // initialized), so calling them here avoids leaking GPU/VAAPI resources.
        reset_nvenc_encoder();
        reset_vaapi_encoder();
        delete this;
        return;
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
    // running reflects "thread alive AND past init". Clear it on EVERY exit (the many early
    // returns + natural end) via RAII so is_capturing can't stay True after the thread dies
    // on its own (XOpenDisplay/XShm/encoder-init failure). Set true only after init succeeds.
    struct RunningGuard { std::atomic<bool>& r; ~RunningGuard() { r.store(false); } } running_guard{running};
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
    Display* display = nullptr;
    Window root_window = 0;
    int screen = 0;
    XShmSegmentInfo shminfo;
    XImage* shm_image = nullptr;
    bool shm_setup_complete = false;
    // Carry the auto-adjust dimension publish out of the setup writer scope: we must not acquire
    // settings_mutex while the lifecycle writer is held (lock-ordering invariant), so set the
    // stack locals inside the scope and publish to this->capture_* after the writer releases.
    bool publish_auto_adjust_dims = false;
    int publish_capture_width = 0;
    int publish_capture_height = 0;
    {
      // SETUP WRITER: one exclusive critical section spanning the ENTIRE connection-setup phase --
      // XOpenDisplay, XGetWindowAttributes, XShmQueryExtension, XFixesQueryExtension, the
      // XShmCreateImage/shmget/shmat/XShmAttach+XSync retry loop, and the success/failure
      // determination. Every libxcb-touching setup call runs under this single writer so it can
      // never race another instance's open/close (or any in-flight grab) on libxcb global state;
      // it also serializes the process-global XSetErrorHandler swap + g_shm_attach_failed read.
      // The mutex is non-recursive, so every failure-path XCloseDisplay WITHIN this scope is a
      // BARE call (the lock is already held); the RAII lock releases exactly once on every exit
      // path (success and each failure). No encode, no capture-thread join here. The only
      // non-X work kept inside is stack-local dimension math; YUV-plane allocation and
      // load_watermark_image() (which take other mutexes / do file I/O) run AFTER the scope.
      // Writer-preference: signal intent BEFORE blocking so readers back off; clear once acquired.
      g_x_display_writers_waiting.fetch_add(1, std::memory_order_release);
      std::unique_lock<std::shared_mutex> x_lifecycle_lock(g_x_display_lifecycle_mutex);
      g_x_display_writers_waiting.fetch_sub(1, std::memory_order_relaxed);
      display = XOpenDisplay(display_name);

      if (!display) {
        std::cerr << "Error: Failed to open X display " << display_name << std::endl;
        return;
      }

      root_window = DefaultRootWindow(display);
      screen = DefaultScreen(display);
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

            // Defer the settings_mutex publish until after the writer releases.
            publish_auto_adjust_dims = true;
            publish_capture_width = attributes.width;
            publish_capture_height = attributes.height;
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

      if (!XShmQueryExtension(display)) {
        std::cerr << "Error: X Shared Memory Extension not available!" << std::endl;
        XCloseDisplay(display);  // BARE: setup writer already held.
        display = nullptr;
        return;
      }

      std::cout << "X Shared Memory Extension available." << std::endl;

      if (local_current_capture_cursor) {
        if (!XFixesQueryExtension(display, &xfixes_event_base, &xfixes_error_base)) {
          std::cerr << "Error: XFixes extension not available!" << std::endl;
          XCloseDisplay(display);  // BARE: setup writer already held.
          display = nullptr;
          return;
        }
        std::cout << "XFixes Extension available." << std::endl;
      }

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
              shm_image = nullptr;
              if (attempt < MAX_ATTACH_ATTEMPTS) std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_BACKOFF_MS));
              continue;
          }

          shminfo.shmaddr = (char*)shmat(shminfo.shmid, nullptr, 0);
          if (shminfo.shmaddr == (char*)-1) {
              perror("shmat");
              shmctl(shminfo.shmid, IPC_RMID, 0);
              XDestroyImage(shm_image);
              shm_image = nullptr;
              if (attempt < MAX_ATTACH_ATTEMPTS) std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_BACKOFF_MS));
              continue;
          }

          shminfo.readOnly = False;
          shm_image->data = shminfo.shmaddr;
          // Already under the setup writer: the XSetErrorHandler swap + XShmAttach + XSync + flag
          // read are exclusive vs other instances' setup (folds in the old g_xshm_setup_mutex).
          g_shm_attach_failed = false;
          XErrorHandler old_handler = XSetErrorHandler(shm_attach_error_handler);
          XShmAttach(display, &shminfo);
          XSync(display, False);
          XSetErrorHandler(old_handler);
          bool attach_failed = g_shm_attach_failed;

          if (attach_failed) {
              std::cerr << "Attempt " << attempt << "/" << MAX_ATTACH_ATTEMPTS << ": XShmAttach failed with an X server error." << std::endl;
              shmdt(shminfo.shmaddr);
              shmctl(shminfo.shmid, IPC_RMID, 0);
              XDestroyImage(shm_image);
              shm_image = nullptr;
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
          XCloseDisplay(display);  // BARE: setup writer already held.
          display = nullptr;
          return;
      }
    }  // SETUP WRITER released here: all libxcb-touching setup is done.

    // Publish the auto-adjusted dimensions now that the lifecycle writer is released (avoids
    // nesting settings_mutex inside the writer).
    if (publish_auto_adjust_dims) {
      std::lock_guard<std::mutex> lock(settings_mutex);
      this->capture_width = publish_capture_width;
      this->capture_height = publish_capture_height;
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

    // Init (display / XShm / encoder selection) succeeded -> capture is live.
    running.store(true);
    while (!stop_requested) {
      auto current_loop_iter_start_time = std::chrono::high_resolution_clock::now();

      bool paced_this_iter = false;
      if (current_loop_iter_start_time < next_frame_time) {
        auto time_to_sleep = next_frame_time - current_loop_iter_start_time;
        if (time_to_sleep > std::chrono::milliseconds(0)) {
          std::this_thread::sleep_for(time_to_sleep);
          paced_this_iter = true;
        }
      }
      // Anti-starvation: when target_fps is so high the per-frame pacing sleep does not fire,
      // the grab loop spins acquiring the shared_lock READER back-to-back. glibc's shared_mutex
      // has no writer priority, so a concurrent stop_capture (XCloseDisplay, WRITER) could be
      // starved indefinitely by saturating readers. Yield on the unpaced path so the writer
      // gets a turn. (No effect when a real pacing sleep already ran.)
      if (!paced_this_iter) {
        std::this_thread::yield();
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

        // REINIT SETUP WRITER: one exclusive critical section spanning the whole resolution-change
        // setup -- XShmDetach of the old segment through XShmCreateImage/shmget/shmat/XShmAttach of
        // the new one. Mirrors the initial-setup writer: every libxcb-touching call is exclusive vs
        // other instances' open/close (and vs grabs), failure-path XCloseDisplay is BARE (writer
        // already held), and the RAII lock releases exactly once on every path. YUV-plane
        // (re)allocation runs AFTER this scope. Every failure path returns from the thread, so
        // falling through past the scope means the reinit succeeded.
        {
          // Writer-preference: signal intent BEFORE blocking so readers back off; clear once acquired.
          g_x_display_writers_waiting.fetch_add(1, std::memory_order_release);
          std::unique_lock<std::shared_mutex> x_lifecycle_lock(g_x_display_lifecycle_mutex);
          g_x_display_writers_waiting.fetch_sub(1, std::memory_order_relaxed);

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
            XCloseDisplay(display); display = nullptr; return;  // BARE: reinit writer already held.
          }
          shminfo.shmid = shmget(
            IPC_PRIVATE, static_cast<size_t>(shm_image->bytes_per_line) * shm_image->height,
            IPC_CREAT | 0600);
          if (shminfo.shmid < 0) {
            perror("shmget re-init"); XDestroyImage(shm_image); shm_image = nullptr;
            XCloseDisplay(display); display = nullptr; return;  // BARE: reinit writer already held.
          }
          shminfo.shmaddr = (char*)shmat(shminfo.shmid, nullptr, 0);
          if (shminfo.shmaddr == (char*)-1) {
            perror("shmat re-init");
            shmctl(shminfo.shmid, IPC_RMID, 0); shminfo.shmid = -1;
            XDestroyImage(shm_image); shm_image = nullptr;
            XCloseDisplay(display); display = nullptr; return;  // BARE: reinit writer already held.
          }
          shminfo.readOnly = False;
          shm_image->data = shminfo.shmaddr;
          Bool reinit_attach_ok = XShmAttach(display, &shminfo);
          if (!reinit_attach_ok) {
            shmdt(shminfo.shmaddr); shminfo.shmaddr = (char*)-1;
            shmctl(shminfo.shmid, IPC_RMID, 0); shminfo.shmid = -1;
            XDestroyImage(shm_image); shm_image = nullptr;
            XCloseDisplay(display); display = nullptr; return;  // BARE: reinit writer already held.
          }
        }  // REINIT SETUP WRITER released here.

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

      // READER (shared): only the per-frame grab touches the connection here, so concurrent
      // capture threads grab in parallel; they are all blocked only while some instance holds the
      // writer (open/close/attach/detach). The lock is released the instant the grab returns -- all
      // the decode/encode work below runs UNLOCKED (it reads the process-private SHM segment, not
      // the connection), so a slow encode never blocks another instance's open/close.
      bool grab_ok;
      {
        // Writer-preference back-off: defer to any waiting writer (XCloseDisplay during stop) so a
        // saturating reader stream can't starve it. Steady state (no writer) = one relaxed load.
        while (g_x_display_writers_waiting.load(std::memory_order_acquire) != 0) { std::this_thread::yield(); }
        std::shared_lock<std::shared_mutex> x_lifecycle_lock(g_x_display_lifecycle_mutex);
        grab_ok = XShmGetImage(display, root_window, shm_image, local_capture_x_offset, local_capture_y_offset, AllPlanes);
      }
      if (grab_ok) {
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
                        // FullFrame streaming: the hash is never computed and NVENC consumes the
                        // device buffer directly, so skip the device->host NV12 download. Any
                        // other mode (striped / non-streaming) still needs the host planes.
                        bool skip_host_dl = local_current_h264_streaming_mode && local_current_h264_fullframe;
                        converted = cuda_convert::argb_to_nv12_device(
                            nvenc_state_.cuda_context, shm_data_ptr, shm_stride_bytes,
                            full_frame_y_plane_.data(), full_frame_y_stride_,
                            full_frame_u_plane_.data(), full_frame_u_stride_,
                            local_capture_width_actual, local_capture_height_actual, &dev_base, &dev_pitch,
                            skip_host_dl);
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
                 {
                     // Serialize detach against concurrent instance connection setup/teardown.
                     // WRITER: exclusive vs grabs and other instances' open/close.
                     // Writer-preference: signal intent BEFORE blocking so readers back off; clear once acquired.
                     g_x_display_writers_waiting.fetch_add(1, std::memory_order_release);
                     std::unique_lock<std::shared_mutex> x_lifecycle_lock(g_x_display_lifecycle_mutex);
                     g_x_display_writers_waiting.fetch_sub(1, std::memory_order_relaxed);
                     XShmDetach(display, &shminfo);
                 }
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
        {
            // Serialize connection teardown so it can't race another instance's XOpenDisplay/
            // XCloseDisplay on libxcb global state. XSync flushes immediately before close.
            // WRITER: exclusive vs grabs and other instances' open/close.
            // Writer-preference: signal intent BEFORE blocking so readers back off; clear once acquired.
            g_x_display_writers_waiting.fetch_add(1, std::memory_order_release);
            std::unique_lock<std::shared_mutex> x_lifecycle_lock(g_x_display_lifecycle_mutex);
            g_x_display_writers_waiting.fetch_sub(1, std::memory_order_relaxed);
            XSync(display, False);
            XCloseDisplay(display);
        }
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
 *         The caller owns `result.data`: the capture loop frees it via `delete[]`
 *         unless ownership is transferred to Python (StripeFrame), which then frees it.
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
        // Process-global serialization vs other instances' libx264 open/close
        // (heap-corruption fix). Tight scope: close only, not the encode.
        {
          std::lock_guard<std::mutex> x264_lock(g_x264_lifecycle_mutex);
          x264_encoder_close(h264_minimal_store.encoders[thread_id]);
        }
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

        // Process-global serialization vs other instances' libx264 open/close
        // (heap-corruption fix). Tight scope: open only, not the encode.
        {
          std::lock_guard<std::mutex> x264_lock(g_x264_lifecycle_mutex);
          h264_minimal_store.encoders[thread_id] = x264_encoder_open(&param);
        }
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

// ============================================================================
// CPython C-API extension (full API, not Limited/abi3). Built with -std=c++17,
// so no C++20 designated initializers. All types via PyType_FromSpec.
// ============================================================================

// Heap type for StripeFrame; created once in PyInit, never freed (immortal).
static PyObject* g_StripeFrameType = nullptr;

// ----------------------------------------------------------------------------
// StripeFrame: zero-copy, refcount-owned view over one encoded stripe. Owns the
// new[]'d buffer; a memoryview/slice keeps it alive via Py_buffer.obj refcount
// (this replaces the old OwnedFrame ctypes pin).
// ----------------------------------------------------------------------------
typedef struct {
  PyObject_HEAD
  unsigned char* data;  // new[]-allocated; freed in dealloc
  int size;
  int data_type;        // StripeDataType: 1=JPEG, 2=H264
  int stripe_y_start;
  int stripe_height;
  int frame_id;
} StripeFrameObject;

static int StripeFrame_getbuffer(PyObject* self, Py_buffer* view, int flags) {
  StripeFrameObject* f = (StripeFrameObject*)self;
  if (f->data == nullptr) {
    PyErr_SetString(PyExc_ValueError, "StripeFrame buffer already released");
    view->obj = nullptr;
    return -1;
  }
  // readonly=1; FillInfo INCREFs self into view->obj, pinning the buffer.
  return PyBuffer_FillInfo(view, self, f->data, (Py_ssize_t)f->size, 1, flags);
}

static void StripeFrame_releasebuffer(PyObject* self, Py_buffer* view) {
  (void)self;
  (void)view;  // FillInfo set view->obj; CPython DECREFs it. Nothing to do.
}

static Py_ssize_t StripeFrame_length(PyObject* self) {
  return (Py_ssize_t)((StripeFrameObject*)self)->size;
}

static PyObject* StripeFrame_get_data_type(PyObject* self, void* closure) {
  (void)closure;
  return PyLong_FromLong(((StripeFrameObject*)self)->data_type);
}

static PyObject* StripeFrame_get_stripe_y_start(PyObject* self, void* closure) {
  (void)closure;
  return PyLong_FromLong(((StripeFrameObject*)self)->stripe_y_start);
}

static PyObject* StripeFrame_get_stripe_height(PyObject* self, void* closure) {
  (void)closure;
  return PyLong_FromLong(((StripeFrameObject*)self)->stripe_height);
}

static PyObject* StripeFrame_get_frame_id(PyObject* self, void* closure) {
  (void)closure;
  return PyLong_FromLong(((StripeFrameObject*)self)->frame_id);
}

static void StripeFrame_dealloc(PyObject* self) {
  StripeFrameObject* f = (StripeFrameObject*)self;
  delete[] f->data;
  f->data = nullptr;
  // Heap-type teardown: call tp_free, then drop the type ref tp_alloc took.
  PyTypeObject* tp = Py_TYPE(self);
  freefunc free_fn = (freefunc)PyType_GetSlot(tp, Py_tp_free);
  free_fn(self);
  Py_DECREF(tp);
}

static PyGetSetDef StripeFrame_getset[] = {
    {"data_type", StripeFrame_get_data_type, nullptr, PyDoc_STR("Encoded type: 1=JPEG, 2=H264."), nullptr},
    {"stripe_y_start", StripeFrame_get_stripe_y_start, nullptr, PyDoc_STR("Stripe top Y coordinate."), nullptr},
    {"stripe_height", StripeFrame_get_stripe_height, nullptr, PyDoc_STR("Stripe height in pixels."), nullptr},
    {"frame_id", StripeFrame_get_frame_id, nullptr, PyDoc_STR("Frame identifier."), nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

static PyType_Slot StripeFrame_slots[] = {
    {Py_tp_dealloc, (void*)StripeFrame_dealloc},
    {Py_tp_getset, (void*)StripeFrame_getset},
    {Py_bf_getbuffer, (void*)StripeFrame_getbuffer},
    {Py_bf_releasebuffer, (void*)StripeFrame_releasebuffer},
    {Py_sq_length, (void*)StripeFrame_length},
    {Py_mp_length, (void*)StripeFrame_length},
    {Py_tp_doc, (void*)PyDoc_STR("Zero-copy buffer-protocol view over an encoded stripe.")},
    {0, nullptr},
};

static PyType_Spec StripeFrame_spec = {
    "pixelflux._capture.StripeFrame",
    sizeof(StripeFrameObject),
    0,
    Py_TPFLAGS_DEFAULT,
    StripeFrame_slots,
};

// Allocate a StripeFrame and take ownership of `data` (size bytes). Returns a
// new ref, or nullptr (exception set; caller still owns `data`).
static StripeFrameObject* StripeFrame_new_owning(unsigned char* data, int size,
                                                 int data_type, int stripe_y_start,
                                                 int stripe_height, int frame_id) {
  PyTypeObject* ft = (PyTypeObject*)g_StripeFrameType;
  StripeFrameObject* f = (StripeFrameObject*)ft->tp_alloc(ft, 0);
  if (!f) return nullptr;
  f->data = data;
  f->size = size;
  f->data_type = data_type;
  f->stripe_y_start = stripe_y_start;
  f->stripe_height = stripe_height;
  f->frame_id = frame_id;
  return f;
}

// Module function: copy any buffer-protocol object into a fresh StripeFrame.
// Interim path the Wayland bridge uses (Rust still returns bytes) and the
// GPU-free cross-version test vehicle for StripeFrame's buffer protocol.
static PyObject* py_stripe_frame_from_buffer(PyObject* self, PyObject* args) {
  (void)self;
  Py_buffer buf;
  int data_type, stripe_y_start, stripe_height, frame_id;
  if (!PyArg_ParseTuple(args, "y*iiii:stripe_frame_from_buffer", &buf,
                        &data_type, &stripe_y_start, &stripe_height, &frame_id)) {
    return nullptr;
  }
  int size = (int)buf.len;
  unsigned char* copy = new (std::nothrow) unsigned char[size > 0 ? size : 1];
  if (!copy) {
    PyBuffer_Release(&buf);
    PyErr_NoMemory();
    return nullptr;
  }
  if (size > 0) std::memcpy(copy, buf.buf, (size_t)size);
  PyBuffer_Release(&buf);
  StripeFrameObject* f = StripeFrame_new_owning(copy, size, data_type,
                                                stripe_y_start, stripe_height, frame_id);
  if (!f) { delete[] copy; return nullptr; }
  return (PyObject*)f;
}

// ----------------------------------------------------------------------------
// ScreenCapture: owns a ScreenCaptureModule and the Python callback. The C++
// capture threads invoke capture_trampoline per stripe.
// ----------------------------------------------------------------------------
typedef struct {
  PyObject_HEAD
  ScreenCaptureModule* module;
  PyObject* callback;
  PyObject* watermark_bytes;    // keeps cset.watermark_path's bytes alive for the thread
  PyObject* render_node_bytes;  // keeps cset.vaapi_render_node_path's bytes alive
} ScreenCaptureObject;

// Called from the C++ capture/encode threads. Builds a StripeFrame that takes
// ownership of result->data and dispatches to the Python callback. Never lets a
// Python exception escape into C++.
//
// Leak-safety: deferred_free is forced on by the Python ScreenCapture_start_capture
// wrapper (cset.deferred_free = true), so after this returns the C++ loop sets
// result.data=nullptr WITHOUT delete[] -- it never reclaims an un-taken buffer
// (StripeEncodeResult has no data-freeing dtor). So the trampoline is the sole
// owner: on alloc failure or a NULL Python callback it must delete[] here.
static void capture_trampoline(StripeEncodeResult* result, void* user_data) {
  ScreenCaptureObject* cap = (ScreenCaptureObject*)user_data;
  PyGILState_STATE g = PyGILState_Ensure();
  // Own a strong ref to the capture object itself across the call. A callback that
  // runs gc.collect() (or otherwise drops the last external ref to this very
  // ScreenCapture) could let cyclic GC reclaim/finalize the object graph while the
  // capture thread is mid-stripe -> heap corruption. Holding our own external ref
  // here keeps `cap` (and, transitively, its module) alive across the call. The
  // matching Py_DECREF below is what actually triggers dealloc on the re-entrant
  // self-teardown path (see ScreenCapture_dealloc's on-capture-thread branch), which
  // is why it must run only AFTER the callback has fully returned.
  Py_INCREF((PyObject*)cap);
  // Own a strong ref to the callback across the call: a re-entrant
  // cap.stop_capture()/clear/dealloc (a documented path -- the Python callback
  // may stop its own capture) runs Py_CLEAR(cap->callback) up this same C stack.
  // Reading cap->callback once and INCREF'ing it here guarantees that Py_CLEAR
  // only drops ITS ref, never the last one, so the in-flight callable stays
  // valid through PyObject_CallFunctionObjArgs (was use-after-free / heap
  // corruption). Read once under the GIL; if NULL, skip the call.
  PyObject* cb = cap->callback;
  Py_XINCREF(cb);
  if (cb && result && result->size > 0 && result->data) {
    StripeFrameObject* f = StripeFrame_new_owning(
        result->data, result->size, (int)result->type,
        result->stripe_y_start, result->stripe_height, result->frame_id);
    if (f) {
      result->data = nullptr;  // ownership transferred to Python; C++ won't free
      PyObject* r = PyObject_CallFunctionObjArgs(cb, (PyObject*)f, nullptr);
      if (!r) {
        PyErr_WriteUnraisable(cb);
      } else {
        Py_DECREF(r);
      }
      Py_DECREF(f);
    } else {
      // Alloc failed: C++ deferred path won't reclaim, so free here ourselves.
      PyErr_WriteUnraisable(cb);
      delete[] result->data;
      result->data = nullptr;
    }
  } else if (result && result->data) {
    // No Python callback (cap->callback == NULL) but the buffer was allocated. The
    // deferred-free loop detaches result.data WITHOUT delete, so as sole owner we
    // must reclaim it here to avoid a per-frame leak.
    delete[] result->data;
    result->data = nullptr;
  }
  Py_XDECREF(cb);  // release the strong ref taken across the call
  // Release our strong ref on the capture object. If this drops the last ref it runs
  // ScreenCapture_dealloc HERE, on the capture thread, with capture_loop still on the
  // stack below; that path must NOT delete the module inline (it hands the module off
  // to the capture thread for self-delete-on-exit -- see ScreenCapture_dealloc). Do
  // not touch `cap` after this point.
  Py_DECREF((PyObject*)cap);
  PyGILState_Release(g);
}

static PyObject* ScreenCapture_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  (void)args;
  (void)kwds;
  ScreenCaptureObject* self = (ScreenCaptureObject*)type->tp_alloc(type, 0);
  if (!self) return nullptr;
  self->module = nullptr;
  self->callback = nullptr;
  self->watermark_bytes = nullptr;
  self->render_node_bytes = nullptr;
  self->module = new (std::nothrow) ScreenCaptureModule();
  if (!self->module) {
    Py_DECREF(self);
    PyErr_NoMemory();
    return nullptr;
  }
  // Note: tp_alloc (PyType_GenericAlloc) already GC-tracks the object for a
  // Py_TPFLAGS_HAVE_GC type, so do NOT call PyObject_GC_Track here (double-track aborts).
  return (PyObject*)self;
}

// GC support: holds a Python callback (+ watermark/render-node bytes) that can form a
// reference cycle (e.g. a bound-method callback whose self owns this ScreenCapture).
static int ScreenCapture_traverse(PyObject* self, visitproc visit, void* arg) {
  ScreenCaptureObject* cap = (ScreenCaptureObject*)self;
  Py_VISIT(Py_TYPE(self));  // heap type: must visit its type
  Py_VISIT(cap->callback);
  Py_VISIT(cap->watermark_bytes);
  Py_VISIT(cap->render_node_bytes);
  return 0;
}

// Break the cycle. Stop the capture FIRST (GIL released so a final trampoline can run)
// so the thread can't touch callback after we clear it -- same ordering as stop_capture.
static int ScreenCapture_clear(PyObject* self) {
  ScreenCaptureObject* cap = (ScreenCaptureObject*)self;
  if (cap->module) {
    Py_BEGIN_ALLOW_THREADS
    try { cap->module->stop_capture(); } catch (...) {}  // teardown must not throw across ALLOW_THREADS
    Py_END_ALLOW_THREADS
  }
  Py_CLEAR(cap->callback);
  Py_CLEAR(cap->watermark_bytes);
  Py_CLEAR(cap->render_node_bytes);
  return 0;
}

static void ScreenCapture_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);  // GC type: untrack before teardown
  ScreenCaptureObject* cap = (ScreenCaptureObject*)self;
  if (cap->module) {
    if (cap->module->on_capture_thread()) {
      // Re-entrant dealloc: a Python stripe callback dropped the LAST ref to its own
      // ScreenCapture, so we are running ON the capture thread with capture_loop still
      // on the C stack below us. Deleting the module here would free an object whose
      // member function is still executing (UAF) and destroy a joinable std::thread
      // from within itself (std::terminate). Instead transfer module ownership to the
      // capture thread: it stops, drains the current frame WITHOUT re-entering this
      // (now-dying) callback, and `delete this`-es as its final act. We detach inside
      // request_self_delete so no join is ever attempted on the running thread.
      cap->module->request_self_delete();
      cap->module = nullptr;  // Python object no longer owns the (self-deleting) module
    } else {
      Py_BEGIN_ALLOW_THREADS  // stop_capture joins the thread; release GIL so the
      try { cap->module->stop_capture(); } catch (...) {}  // trampoline finishes draining; never throw in dealloc
      Py_END_ALLOW_THREADS
      delete cap->module;
      cap->module = nullptr;
    }
  }
  Py_CLEAR(cap->callback);
  Py_CLEAR(cap->watermark_bytes);
  Py_CLEAR(cap->render_node_bytes);
  PyTypeObject* tp = Py_TYPE(self);
  freefunc free_fn = (freefunc)PyType_GetSlot(tp, Py_tp_free);
  free_fn(self);
  Py_DECREF(tp);
}

// Read one bool-ish attribute (0/1) from a settings object; -1 + exception set.
static int read_bool_attr(PyObject* obj, const char* name, bool* out) {
  PyObject* v = PyObject_GetAttrString(obj, name);
  if (!v) return -1;
  int b = PyObject_IsTrue(v);
  Py_DECREF(v);
  if (b < 0) return -1;
  *out = (b != 0);
  return 0;
}

// Read one int attribute; -1 + exception set.
static int read_int_attr(PyObject* obj, const char* name, long* out) {
  PyObject* v = PyObject_GetAttrString(obj, name);
  if (!v) return -1;
  long n = PyLong_AsLong(v);
  Py_DECREF(v);
  if (n == -1 && PyErr_Occurred()) return -1;
  *out = n;
  return 0;
}

// Read one double attribute; -1 + exception set.
static int read_double_attr(PyObject* obj, const char* name, double* out) {
  PyObject* v = PyObject_GetAttrString(obj, name);
  if (!v) return -1;
  double d = PyFloat_AsDouble(v);
  Py_DECREF(v);
  if (d == -1.0 && PyErr_Occurred()) return -1;
  *out = d;
  return 0;
}

// Read a str/bytes/None C-string attribute. On success *holder owns a new bytes
// ref (or nullptr for None) and *out points into it (or nullptr); -1 on error.
static int read_cstr_attr(PyObject* obj, const char* name, const char** out,
                          PyObject** holder) {
  *out = nullptr;
  *holder = nullptr;
  PyObject* v = PyObject_GetAttrString(obj, name);
  if (!v) return -1;
  if (v == Py_None) {
    Py_DECREF(v);
    return 0;
  }
  PyObject* b = nullptr;
  if (PyUnicode_Check(v)) {
    b = PyUnicode_AsUTF8String(v);  // new ref
    Py_DECREF(v);
    if (!b) return -1;
  } else if (PyBytes_Check(v)) {
    b = v;  // steal the ref we already hold
  } else {
    Py_DECREF(v);
    PyErr_Format(PyExc_TypeError, "%s must be str, bytes, or None", name);
    return -1;
  }
  *holder = b;
  *out = PyBytes_AS_STRING(b);  // valid while b lives
  return 0;
}

static PyObject* ScreenCapture_start_capture(PyObject* self, PyObject* args) {
  ScreenCaptureObject* cap = (ScreenCaptureObject*)self;
  PyObject* settings;
  PyObject* callback;
  if (!PyArg_ParseTuple(args, "OO:start_capture", &settings, &callback)) {
    return nullptr;
  }
  if (!PyCallable_Check(callback)) {
    PyErr_SetString(PyExc_TypeError, "callback must be callable");
    return nullptr;
  }
  if (!cap->module) {
    PyErr_SetString(PyExc_RuntimeError, "capture module not initialized");
    return nullptr;
  }

  CaptureSettings cset;  // C++ defaults; overridden below
  PyObject* wm_bytes = nullptr;
  PyObject* node_bytes = nullptr;
  long lv;
  double dv;
  bool bv;

  // Two C-string fields: held alive in wm_bytes/node_bytes for the thread.
  if (read_cstr_attr(settings, "watermark_path", &cset.watermark_path, &wm_bytes) < 0)
    goto err;
  if (read_cstr_attr(settings, "vaapi_render_node_path", &cset.vaapi_render_node_path, &node_bytes) < 0)
    goto err;

  if (read_int_attr(settings, "capture_width", &lv) < 0) goto err;
  cset.capture_width = (int)lv;
  if (read_int_attr(settings, "capture_height", &lv) < 0) goto err;
  cset.capture_height = (int)lv;
  if (read_double_attr(settings, "scale", &dv) < 0) goto err;
  cset.scale = dv;
  if (read_int_attr(settings, "capture_x", &lv) < 0) goto err;
  cset.capture_x = (int)lv;
  if (read_int_attr(settings, "capture_y", &lv) < 0) goto err;
  cset.capture_y = (int)lv;
  if (read_double_attr(settings, "target_fps", &dv) < 0) goto err;
  cset.target_fps = dv;
  if (read_int_attr(settings, "jpeg_quality", &lv) < 0) goto err;
  cset.jpeg_quality = (int)lv;
  if (read_int_attr(settings, "paint_over_jpeg_quality", &lv) < 0) goto err;
  cset.paint_over_jpeg_quality = (int)lv;
  if (read_bool_attr(settings, "use_paint_over_quality", &bv) < 0) goto err;
  cset.use_paint_over_quality = bv;
  if (read_int_attr(settings, "paint_over_trigger_frames", &lv) < 0) goto err;
  cset.paint_over_trigger_frames = (int)lv;
  if (read_int_attr(settings, "damage_block_threshold", &lv) < 0) goto err;
  cset.damage_block_threshold = (int)lv;
  if (read_int_attr(settings, "damage_block_duration", &lv) < 0) goto err;
  cset.damage_block_duration = (int)lv;
  if (read_int_attr(settings, "output_mode", &lv) < 0) goto err;
  cset.output_mode = (OutputMode)lv;
  if (read_int_attr(settings, "h264_crf", &lv) < 0) goto err;
  cset.h264_crf = (int)lv;
  if (read_int_attr(settings, "h264_paintover_crf", &lv) < 0) goto err;
  cset.h264_paintover_crf = (int)lv;
  if (read_int_attr(settings, "h264_paintover_burst_frames", &lv) < 0) goto err;
  cset.h264_paintover_burst_frames = (int)lv;
  if (read_bool_attr(settings, "h264_fullcolor", &bv) < 0) goto err;
  cset.h264_fullcolor = bv;
  if (read_bool_attr(settings, "h264_fullframe", &bv) < 0) goto err;
  cset.h264_fullframe = bv;
  if (read_bool_attr(settings, "h264_streaming_mode", &bv) < 0) goto err;
  cset.h264_streaming_mode = bv;
  if (read_bool_attr(settings, "capture_cursor", &bv) < 0) goto err;
  cset.capture_cursor = bv;
  if (read_int_attr(settings, "watermark_location_enum", &lv) < 0) goto err;
  cset.watermark_location_enum = (WatermarkLocation)lv;
  if (read_int_attr(settings, "vaapi_render_node_index", &lv) < 0) goto err;
  cset.vaapi_render_node_index = (int)lv;
  if (read_bool_attr(settings, "use_cpu", &bv) < 0) goto err;
  cset.use_cpu = bv;
  if (read_bool_attr(settings, "debug_logging", &bv) < 0) goto err;
  cset.debug_logging = bv;
  if (read_bool_attr(settings, "h264_cbr_mode", &bv) < 0) goto err;
  cset.h264_cbr_mode = bv;
  if (read_int_attr(settings, "h264_bitrate_kbps", &lv) < 0) goto err;
  cset.h264_bitrate_kbps = (int)lv;
  if (read_int_attr(settings, "h264_vbv_buffer_size_kb", &lv) < 0) goto err;
  cset.h264_vbv_buffer_size_kb = (int)lv;
  if (read_bool_attr(settings, "auto_adjust_screen_capture_size", &bv) < 0) goto err;
  cset.auto_adjust_screen_capture_size = bv;
  if (read_bool_attr(settings, "omit_stripe_headers", &bv) < 0) goto err;
  cset.omit_stripe_headers = bv;
  // deferred_free is forced on: the trampoline always takes ownership. The
  // Python value is ignored (kept on the settings object only for API parity).
  cset.deferred_free = true;

  // Commit: retain C-string bytes + callback, wire up module, start.
  Py_XSETREF(cap->watermark_bytes, wm_bytes);    // steals our ref; clears prior
  Py_XSETREF(cap->render_node_bytes, node_bytes);
  wm_bytes = nullptr;
  node_bytes = nullptr;
  Py_INCREF(callback);
  Py_XSETREF(cap->callback, callback);
  cap->module->modify_settings(cset);
  {
    std::lock_guard<std::mutex> lock(cap->module->settings_mutex);
    cap->module->stripe_callback = capture_trampoline;
    cap->module->user_data = cap;
  }
  // start_capture() stops any prior run, probes encoders, and spawns the thread.
  // std::thread creation can throw std::system_error (thread limit/OOM); a C++ exception
  // must not unwind through ALLOW_THREADS (GIL released) into CPython. Catch, restore the
  // GIL, then raise a Python error.
  {  // scoped so the `goto err` paths above don't jump across start_err's initialization
    std::string start_err;
    Py_BEGIN_ALLOW_THREADS
    try {
      cap->module->start_capture();
    } catch (const std::exception& e) {
      start_err = e.what();
    } catch (...) {
      start_err = "unknown C++ exception";
    }
    Py_END_ALLOW_THREADS
    if (!start_err.empty()) {
      Py_CLEAR(cap->callback);
      PyErr_Format(PyExc_RuntimeError, "start_capture failed: %s", start_err.c_str());
      return nullptr;
    }
  }
  Py_RETURN_NONE;

err:
  Py_XDECREF(wm_bytes);
  Py_XDECREF(node_bytes);
  return nullptr;
}

static PyObject* ScreenCapture_stop_capture(PyObject* self, PyObject* Py_UNUSED(ignored)) {
  ScreenCaptureObject* cap = (ScreenCaptureObject*)self;
  if (cap->module) {
    Py_BEGIN_ALLOW_THREADS  // joins the thread; release GIL so a final
    try { cap->module->stop_capture(); } catch (...) {}  // trampoline runs; never throw across the C boundary
    Py_END_ALLOW_THREADS
  }
  Py_CLEAR(cap->callback);
  Py_RETURN_NONE;
}

static PyObject* ScreenCapture_request_idr_frame(PyObject* self, PyObject* Py_UNUSED(ignored)) {
  ScreenCaptureObject* cap = (ScreenCaptureObject*)self;
  if (cap->module) cap->module->request_idr();
  Py_RETURN_NONE;
}

static PyObject* ScreenCapture_update_video_bitrate(PyObject* self, PyObject* arg) {
  ScreenCaptureObject* cap = (ScreenCaptureObject*)self;
  long n = PyLong_AsLong(arg);
  if (n == -1 && PyErr_Occurred()) return nullptr;
  if (cap->module) cap->module->update_video_bitrate((int)n);
  Py_RETURN_NONE;
}

static PyObject* ScreenCapture_update_framerate(PyObject* self, PyObject* arg) {
  ScreenCaptureObject* cap = (ScreenCaptureObject*)self;
  double fps = PyFloat_AsDouble(arg);
  if (fps == -1.0 && PyErr_Occurred()) return nullptr;
  if (cap->module) cap->module->update_framerate(fps);
  Py_RETURN_NONE;
}

static PyObject* ScreenCapture_update_vbv_buffer_size(PyObject* self, PyObject* arg) {
  ScreenCaptureObject* cap = (ScreenCaptureObject*)self;
  long n = PyLong_AsLong(arg);
  if (n == -1 && PyErr_Occurred()) return nullptr;
  if (cap->module) cap->module->update_vbv_buffer_size((int)n);
  Py_RETURN_NONE;
}

static PyObject* ScreenCapture_get_is_capturing(PyObject* self, void* closure) {
  (void)closure;
  ScreenCaptureObject* cap = (ScreenCaptureObject*)self;
  bool running = cap->module && cap->module->running.load() &&
                 !cap->module->stop_requested.load();
  return PyBool_FromLong(running ? 1 : 0);
}

static PyMethodDef ScreenCapture_methods[] = {
    {"start_capture", ScreenCapture_start_capture, METH_VARARGS,
     PyDoc_STR("start_capture(settings, callback): begin capture; callback(frame) per stripe.")},
    {"stop_capture", ScreenCapture_stop_capture, METH_NOARGS,
     PyDoc_STR("Stop capture and join the capture thread.")},
    {"request_idr_frame", ScreenCapture_request_idr_frame, METH_NOARGS,
     PyDoc_STR("Request the next encoded frame be an IDR (key) frame.")},
    {"update_video_bitrate", ScreenCapture_update_video_bitrate, METH_O,
     PyDoc_STR("update_video_bitrate(kbps): set the H.264 CBR bitrate.")},
    {"update_framerate", ScreenCapture_update_framerate, METH_O,
     PyDoc_STR("update_framerate(fps): set the target framerate.")},
    {"update_vbv_buffer_size", ScreenCapture_update_vbv_buffer_size, METH_O,
     PyDoc_STR("update_vbv_buffer_size(kb): set the H.264 VBV buffer size.")},
    {nullptr, nullptr, 0, nullptr},
};

static PyGetSetDef ScreenCapture_getset[] = {
    {"is_capturing", ScreenCapture_get_is_capturing, nullptr,
     PyDoc_STR("True while the capture thread is running."), nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

static PyType_Slot ScreenCapture_slots[] = {
    {Py_tp_new, (void*)ScreenCapture_new},
    {Py_tp_dealloc, (void*)ScreenCapture_dealloc},
    {Py_tp_traverse, (void*)ScreenCapture_traverse},
    {Py_tp_clear, (void*)ScreenCapture_clear},
    {Py_tp_methods, (void*)ScreenCapture_methods},
    {Py_tp_getset, (void*)ScreenCapture_getset},
    {Py_tp_doc, (void*)PyDoc_STR("X11 screen capture + JPEG/H.264 encoder.")},
    {0, nullptr},
};

static PyType_Spec ScreenCapture_spec = {
    "pixelflux._capture.ScreenCapture",
    sizeof(ScreenCaptureObject),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    ScreenCapture_slots,
};

// ----------------------------------------------------------------------------
// Module definition
// ----------------------------------------------------------------------------
static PyMethodDef capture_module_methods[] = {
    {"stripe_frame_from_buffer", py_stripe_frame_from_buffer, METH_VARARGS,
     PyDoc_STR("stripe_frame_from_buffer(buf, data_type, stripe_y_start, stripe_height, frame_id): "
               "copy a buffer into an owning StripeFrame.")},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef capture_module = {
    PyModuleDef_HEAD_INIT,
    "pixelflux._capture",
    PyDoc_STR("Native X11 screen capture -> JPEG/H.264 (full C-API, zero-copy)."),
    -1,
    capture_module_methods,
    nullptr, nullptr, nullptr, nullptr,
};

PyMODINIT_FUNC PyInit__capture(void) {
  // Make Xlib/libxcb multithread-safe BEFORE any other Xlib call in this process. Each
  // ScreenCapture instance opens/uses/closes its own Display from its own capture thread;
  // without this, concurrent multi-instance Xlib use corrupts the glibc heap. Module import
  // runs before any ScreenCapture exists (hence before the first XOpenDisplay), so this is the
  // earliest, canonical entry point. call_once guards against any interpreter re-init.
  static std::once_flag g_xinitthreads_once;
  std::call_once(g_xinitthreads_once, []() { XInitThreads(); });

  PyObject* m = PyModule_Create(&capture_module);
  if (!m) return nullptr;

  g_StripeFrameType = PyType_FromSpec(&StripeFrame_spec);
  if (!g_StripeFrameType) { Py_DECREF(m); return nullptr; }
  Py_INCREF(g_StripeFrameType);  // module-global keeps it alive forever
  if (PyModule_AddObject(m, "StripeFrame", g_StripeFrameType) < 0) {
    Py_DECREF(g_StripeFrameType);  // undo the AddObject ref we tried to give
    Py_DECREF(g_StripeFrameType);  // undo the global ref
    Py_DECREF(m);
    return nullptr;
  }

  PyObject* capture_type = PyType_FromSpec(&ScreenCapture_spec);
  if (!capture_type) { Py_DECREF(m); return nullptr; }
  if (PyModule_AddObject(m, "ScreenCapture", capture_type) < 0) {
    Py_DECREF(capture_type);
    Py_DECREF(m);
    return nullptr;
  }

  return m;
}
