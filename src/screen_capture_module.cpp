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
#include <X11/Xlib.h>
#include <X11/extensions/XShm.h>
#include <X11/Xutil.h>
#include <jpeglib.h>
#include <netinet/in.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#define XXH_STATIC_LINKING_ONLY
#include <xxhash.h>
#include <libyuv/convert.h>
#include <libyuv/convert_from.h>
#include <libyuv/convert_from_argb.h>
#include <libyuv/planar_functions.h>
#include <x264.h>

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
      h264_fullframe(false) {}


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
   */
  CaptureSettings(int cw, int ch, int cx, int cy, double fps, int jq,
                  int pojq, bool upoq, int potf, int dbt, int dbd,
                  OutputMode om = OutputMode::JPEG, int crf = 25,
                  bool h264_fc = false, bool h264_ff = false)
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
      h264_fullframe(h264_ff) {}
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
 * @brief Callback function type for processing encoded stripes.
 * @param result Pointer to the StripeEncodeResult containing the encoded data.
 * @param user_data User-defined data passed to the callback.
 */
typedef void (*StripeCallback)(StripeEncodeResult* result, void* user_data);

extern "C" {
  /**
   * @brief Frees the data buffer within a StripeEncodeResult.
   * This function is intended to be called by the consumer of the
   * StripeEncodeResult once the data is no longer needed.
   * @param result Pointer to the StripeEncodeResult whose data needs freeing.
   */
  void free_stripe_encode_result_data(StripeEncodeResult* result);
}

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
  OutputMode output_mode = OutputMode::H264;

  std::atomic<bool> stop_requested;
  std::thread capture_thread;
  StripeCallback stripe_callback = nullptr;
  void* user_data = nullptr;
  int frame_counter = 0;
  int encoded_frame_count = 0;
  int total_stripes_encoded_this_interval = 0;
  mutable std::mutex settings_mutex;

private:
    std::vector<uint8_t> full_frame_y_plane_;
    std::vector<uint8_t> full_frame_u_plane_;
    std::vector<uint8_t> full_frame_v_plane_;
    int full_frame_y_stride_;
    int full_frame_u_stride_;
    int full_frame_v_stride_;
    bool yuv_planes_are_i444_;

public:
  /**
   * @brief Default constructor for ScreenCaptureModule.
   * Initializes stop_requested to false and YUV plane strides to 0.
   */
  ScreenCaptureModule() : stop_requested(false),
                          full_frame_y_stride_(0), full_frame_u_stride_(0), full_frame_v_stride_(0),
                          yuv_planes_are_i444_(false) {}

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
    g_h264_minimal_store.reset();
    stop_requested = false;
    frame_counter = 0;
    encoded_frame_count = 0;
    total_stripes_encoded_this_interval = 0;
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
      h264_fullcolor, h264_fullframe);
  }

private:
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
          if(display) XCloseDisplay(display); display = nullptr; return;
        }
        shminfo.shmid = shmget(
          IPC_PRIVATE, static_cast<size_t>(shm_image->bytes_per_line) * shm_image->height,
          IPC_CREAT | 0600);
        if (shminfo.shmid < 0) {
          perror("shmget re-init"); if(shm_image) XDestroyImage(shm_image); shm_image = nullptr;
          if(display) XCloseDisplay(display); display = nullptr; return;
        }
        shminfo.shmaddr = (char*)shmat(shminfo.shmid, nullptr, 0);
        if (shminfo.shmaddr == (char*)-1) {
          perror("shmat re-init"); if(shminfo.shmid != -1) shmctl(shminfo.shmid, IPC_RMID, 0);
          shminfo.shmid = -1;
          if(shm_image) XDestroyImage(shm_image); shm_image = nullptr;
          if(display) XCloseDisplay(display); display = nullptr; return;
        }
        shminfo.readOnly = False;
        shm_image->data = shminfo.shmaddr;
        if (!XShmAttach(display, &shminfo)) {
          std::cerr << "Error: XShmAttach failed during re-init." << std::endl;
          if(shminfo.shmaddr != (char*)-1) shmdt(shminfo.shmaddr); shminfo.shmaddr = (char*)-1;
          if(shminfo.shmid != -1) shmctl(shminfo.shmid, IPC_RMID, 0); shminfo.shmid = -1;
          if(shm_image) XDestroyImage(shm_image); shm_image = nullptr;
          if(display) XCloseDisplay(display); display = nullptr; return;
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

      if (XShmGetImage(display, root_window, shm_image,
                       local_capture_x_offset, local_capture_y_offset, AllPlanes)) {

        unsigned char* shm_data_ptr = (unsigned char*)shm_image->data;
        int shm_stride_bytes = shm_image->bytes_per_line;
        int shm_bytes_per_pixel = shm_image->bits_per_pixel / 8;

        if (local_current_output_mode == OutputMode::H264) {
            if (this->yuv_planes_are_i444_) {
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

        std::vector<StripeEncodeResult> stripe_results;
        stripe_results.reserve(futures.size());
        for (auto& future : futures) {
          stripe_results.push_back(future.get());
        }
        futures.clear();

        for (StripeEncodeResult& result : stripe_results) {
          if (stripe_callback != nullptr && result.data != nullptr && result.size > 0) {
            stripe_callback(&result, user_data);
          } else {
             if (result.data) {
                free_stripe_encode_result_data(&result);
             }
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
                    << (local_current_output_mode == OutputMode::JPEG ? "JPEG" : "H264")
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
};

extern "C" {

  typedef void* ScreenCaptureModuleHandle;

  /**
   * @brief Creates a new instance of the ScreenCaptureModule.
   * @return A handle to the created ScreenCaptureModule instance.
   *         The caller is responsible for destroying this instance using
   *         destroy_screen_capture_module.
   */
  ScreenCaptureModuleHandle create_screen_capture_module() {
    return static_cast<ScreenCaptureModuleHandle>(new ScreenCaptureModule());
  }

  /**
   * @brief Destroys a ScreenCaptureModule instance.
   * This will also stop the capture if it is running.
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

      std::lock_guard<std::mutex> lock(module->settings_mutex);
      module->stripe_callback = callback;
      module->user_data = user_data;
      module->start_capture();
    }
  }

  /**
   * @brief Stops the screen capture process.
   * This is a blocking call that waits for the capture thread to terminate.
   * @param module_handle Handle to the ScreenCaptureModule instance.
   */
  void stop_screen_capture(ScreenCaptureModuleHandle module_handle) {
    if (module_handle) {
      static_cast<ScreenCaptureModule*>(module_handle)->stop_capture();
    }
  }

  /**
   * @brief Modifies the settings of an active screen capture.
   * The changes will be applied by the capture loop.
   * @param module_handle Handle to the ScreenCaptureModule instance.
   * @param settings The new capture and encoding settings.
   */
  void modify_screen_capture(ScreenCaptureModuleHandle module_handle,
                             CaptureSettings settings) {
    if (module_handle) {
      static_cast<ScreenCaptureModule*>(module_handle)->modify_settings(settings);
    }
  }

  /**
   * @brief Retrieves the current settings of the screen capture module.
   * @param module_handle Handle to the ScreenCaptureModule instance.
   * @return The current CaptureSettings. If module_handle is null, returns default settings.
   */
  CaptureSettings get_screen_capture_settings(ScreenCaptureModuleHandle module_handle) {
    if (module_handle) {
      return static_cast<ScreenCaptureModule*>(module_handle)->get_current_settings();
    } else {
      return CaptureSettings{};
    }
  }

  /**
   * @brief Frees the data buffer within a StripeEncodeResult.
   * This function is intended to be called by the consumer of the
   * StripeEncodeResult once the data is no longer needed.
   * @param result Pointer to the StripeEncodeResult whose data needs freeing.
   *               If result or result->data is null, the function does nothing.
   */
  void free_stripe_encode_result_data(StripeEncodeResult* result) {
    if (result && result->data) {
      delete[] result->data;
      result->data = nullptr;
    }
  }

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

    for (int r = 0; r < height; ++r) {
        XXH3_64bits_update(&hash_state, y_plane_stripe_start +
                           static_cast<size_t>(r) * y_stride, width);
    }

    int chroma_width = is_i420 ? (width / 2) : width;
    int chroma_height = is_i420 ? (height / 2) : height;

    if (is_i420) {
        if (width % 2 != 0 && chroma_width > 0) chroma_width = (width-1)/2;
        if (height % 2 != 0 && chroma_height > 0) chroma_height = (height-1)/2;
    }

    if (chroma_width <=0 || chroma_height <=0) {
    } else {
        for (int r = 0; r < chroma_height; ++r) {
            XXH3_64bits_update(&hash_state, u_plane_stripe_start +
                               static_cast<size_t>(r) * u_stride, chroma_width);
        }
        for (int r = 0; r < chroma_height; ++r) {
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
