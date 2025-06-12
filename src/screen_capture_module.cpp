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
#include <xxhash.h>
#include <libyuv/convert.h>
#include <libyuv/convert_from.h>
#include <libyuv/planar_functions.h>
#include <x264.h>

// --- Global Store for H.264 Encoders ---

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

// Global instance of the H.264 encoder store.
MinimalEncoderStore g_h264_minimal_store;

/**
 * @brief Enumerates the possible output modes for encoding.
 */
enum class OutputMode {
  JPEG = 0, // Output frames as JPEG images.
  H264 = 1  // Output frames as H.264 video.
};

/**
 * @brief Enumerates the data types for encoded stripes.
 */
enum class StripeDataType {
  UNKNOWN = 0, // Unknown or uninitialized data type.
  JPEG    = 1, // Data is JPEG encoded.
  H264    = 2  // Data is H.264 encoded.
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
  // Disallow copy construction and assignment to prevent unintended data copies.
  StripeEncodeResult(const StripeEncodeResult&) = delete;
  StripeEncodeResult& operator=(const StripeEncodeResult&) = delete;
};

/**
 * @brief Move constructor implementation for StripeEncodeResult.
 */
StripeEncodeResult::StripeEncodeResult(StripeEncodeResult&& other) noexcept
  : type(other.type),
    stripe_y_start(other.stripe_y_start),
    stripe_height(other.stripe_height),
    size(other.size),
    data(other.data),
    frame_id(other.frame_id) {
  // Reset other to a valid, empty state.
  other.type = StripeDataType::UNKNOWN;
  other.stripe_y_start = 0;
  other.stripe_height = 0;
  other.size = 0;
  other.data = nullptr;
  other.frame_id = -1;
}

/**
 * @brief Move assignment operator implementation for StripeEncodeResult.
 */
StripeEncodeResult& StripeEncodeResult::operator=(StripeEncodeResult&& other) noexcept {
  if (this != &other) {
    // Free existing data if any.
    if (data) {
      delete[] data;
      data = nullptr;
    }
    // Move data from other.
    type = other.type;
    stripe_y_start = other.stripe_y_start;
    stripe_height = other.stripe_height;
    size = other.size;
    data = other.data;
    frame_id = other.frame_id;

    // Reset other to a valid, empty state.
    other.type = StripeDataType::UNKNOWN;
    other.stripe_y_start = 0;
    other.stripe_height = 0;
    other.size = 0;
    other.data = nullptr;
    other.frame_id = -1;
  }
  return *this;
}

// --- Function Pointer and Extern Declarations ---

/**
 * @brief Callback function type for processing encoded stripes.
 * @param result Pointer to the StripeEncodeResult containing the encoded data.
 * @param user_data User-defined data passed to the callback.
 */
typedef void (*StripeCallback)(StripeEncodeResult* result, void* user_data);

extern "C" {
  /**
   * @brief Frees the data buffer within a StripeEncodeResult.
   * @param result Pointer to the StripeEncodeResult whose data needs freeing.
   */
  void free_stripe_encode_result_data(StripeEncodeResult* result);
}

/**
 * @brief Encodes a stripe of an image into JPEG format.
 * @param thread_id Identifier for the calling thread, used for encoder management.
 * @param stripe_y_start The Y-coordinate of the top of the stripe.
 * @param stripe_height The height of the stripe to encode.
 * @param width The width of the full image (not necessarily capture_width_actual).
 * @param height The height of the full image.
 * @param capture_width_actual The actual width of the stripe being encoded.
 * @param rgb_data Pointer to the full RGB data of the frame.
 * @param rgb_data_len Length of the rgb_data buffer.
 * @param jpeg_quality The quality setting for JPEG compression (0-100).
 * @param frame_counter The current frame number.
 * @return A StripeEncodeResult containing the JPEG data or an error state.
 */
StripeEncodeResult encode_stripe_jpeg(
  int thread_id,
  int stripe_y_start,
  int stripe_height,
  int width,
  int height,
  int capture_width_actual,
  const unsigned char* rgb_data,
  int rgb_data_len,
  int jpeg_quality,
  int frame_counter);

/**
 * @brief Encodes a stripe of an image into H.264 format.
 * @param thread_id Identifier for the calling thread, used for encoder management.
 * @param stripe_y_start The Y-coordinate of the top of the stripe.
 * @param stripe_height The height of the stripe to encode.
 * @param capture_width_actual The actual width of the stripe being encoded.
 * @param stripe_rgb24_data Pointer to the RGB data for this specific stripe.
 * @param frame_counter The current frame number.
 * @param current_crf_setting The CRF value for H.264 encoding.
 * @param colorspace_setting The target colorspace (e.g., 420, 444).
 * @param use_full_range Boolean indicating if full color range should be used.
 * @return A StripeEncodeResult containing the H.264 data or an error state.
 */
StripeEncodeResult encode_stripe_h264(
  int thread_id,
  int stripe_y_start,
  int stripe_height,
  int capture_width_actual,
  const unsigned char* stripe_rgb24_data,
  int frame_counter,
  int current_crf_setting,
  int colorspace_setting,
  bool use_full_range);

/**
 * @brief Calculates a hash value for a given stripe's RGB data.
 * Used for detecting changes between frames to optimize encoding.
 * @param rgb_data A vector containing the RGB data of the stripe.
 * @return A 64-bit hash value.
 */
uint64_t calculate_stripe_hash(const std::vector<unsigned char>& rgb_data);

// --- Screen Capture Module Class ---

/**
 * @brief Manages the screen capture process, including settings and threading.
 * This class encapsulates the logic for capturing screen content,
 * dividing it into stripes, encoding these stripes (JPEG or H.264),
 * and invoking a callback with the encoded data.
 */
class ScreenCaptureModule {
public:
  // Capture and encoding settings (mirrored from CaptureSettings for direct access)
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

  // Control and state variables
  std::atomic<bool> stop_requested;
  std::thread capture_thread;
  StripeCallback stripe_callback = nullptr;
  void* user_data = nullptr;
  int frame_counter = 0;
  int encoded_frame_count = 0;
  int total_stripes_encoded_this_interval = 0;
  mutable std::mutex settings_mutex; // Protects access to settings

public:
  /**
   * @brief Default constructor for ScreenCaptureModule.
   * Initializes stop_requested to false.
   */
  ScreenCaptureModule() : stop_requested(false) {}

  /**
   * @brief Destructor for ScreenCaptureModule.
   * Ensures that the capture process is stopped and resources are released.
   */
  ~ScreenCaptureModule() {
    stop_capture();
  }

  /**
   * @brief Starts the screen capture process in a new thread.
   * If a capture thread is already running, it is stopped first.
   * Resets encoder stores and frame counters.
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
   */
  void stop_capture() {
    stop_requested = true;
    if (capture_thread.joinable()) {
      capture_thread.join();
    }
  }

  /**
   * @brief Modifies the capture and encoding settings.
   * This function is thread-safe.
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
   * @return A CaptureSettings struct containing the current settings.
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
   * Handles X11 setup, screen grabbing using XShm, frame processing,
   * stripe division, encoding, and invoking callbacks.
   * Runs until stop_requested is true.
   */
  void capture_loop() {
    auto start_time_loop = std::chrono::high_resolution_clock::now();
    int frame_count_loop = 0;

    // Local copies of settings for the capture loop to minimize lock contention
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

    // Initial settings load from shared members
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

    // Ensure dimensions are even for H.264 if it's the selected output mode
    if (local_current_output_mode == OutputMode::H264) {
      if (local_capture_width_actual % 2 != 0 && local_capture_width_actual > 0) {
        local_capture_width_actual--;
      }
      if (local_capture_height_actual % 2 != 0 && local_capture_height_actual > 0) {
        local_capture_height_actual--;
      }
    }

    // Timing setup for achieving target FPS
    std::chrono::duration<double> target_frame_duration_seconds =
      std::chrono::duration<double>(1.0 / local_current_target_fps);
    auto next_frame_time =
      std::chrono::high_resolution_clock::now() + target_frame_duration_seconds;

    // X11 Display Setup
    char* display_env = std::getenv("DISPLAY");
    const char* display_name = display_env ? display_env : ":0";
    Display* display = XOpenDisplay(display_name);
    if (!display) {
      std::cerr << "Error: Failed to open X display " << display_name << std::endl;
      return;
    }
    Window root_window = DefaultRootWindow(display);
    int screen = DefaultScreen(display);

    // X Shared Memory (XShm) Extension Check
    if (!XShmQueryExtension(display)) {
      std::cerr << "Error: X Shared Memory Extension not available!" << std::endl;
      XCloseDisplay(display);
      return;
    }
    std::cout << "X Shared Memory Extension available." << std::endl;

    // XShm Image and Segment Info Setup
    XShmSegmentInfo shminfo;
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
                           shm_image->bytes_per_line * shm_image->height,
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

    // Determine number of stripes based on CPU cores
    int num_cores = std::max(1, (int)std::thread::hardware_concurrency());
    std::cout << "CPU cores available: " << num_cores << std::endl;
    int num_stripes_config = num_cores;

    // Per-stripe state variables for change detection and optimization
    std::vector<uint64_t> previous_hashes(num_stripes_config, 0);
    std::vector<int> no_motion_frame_counts(num_stripes_config, 0);
    std::vector<bool> paint_over_sent(num_stripes_config, false);
    std::vector<int> damage_block_counts(num_stripes_config, 0);
    std::vector<bool> damage_blocked(num_stripes_config, false);
    std::vector<int> damage_block_timer(num_stripes_config, 0);
    std::vector<int> current_jpeg_qualities(num_stripes_config);

    for (int i = 0; i < num_stripes_config; ++i) {
      current_jpeg_qualities[i] =
        local_current_use_paint_over_quality
          ? local_current_paint_over_jpeg_quality
          : local_current_jpeg_quality;
    }

    auto last_output_time = std::chrono::high_resolution_clock::now();

    // --- Main Capture Loop ---
    while (!stop_requested) {
      auto current_loop_iter_start_time = std::chrono::high_resolution_clock::now();

      // Frame Pacing: Sleep if ahead of schedule for the next frame
      if (current_loop_iter_start_time < next_frame_time) {
        auto time_to_sleep = next_frame_time - current_loop_iter_start_time;
        if (time_to_sleep > std::chrono::milliseconds(0)) {
          std::this_thread::sleep_for(time_to_sleep);
        }
      }
      auto intended_current_frame_time = next_frame_time;
      next_frame_time += target_frame_duration_seconds;

      // Periodically update local settings from shared members
      int old_w = local_capture_width_actual;
      int old_h = local_capture_height_actual;
      {
        std::lock_guard<std::mutex> lock(settings_mutex);
        local_capture_width_actual = capture_width;
        local_capture_height_actual = capture_height;
        local_capture_x_offset = capture_x;
        local_capture_y_offset = capture_y;

        if (local_current_target_fps != target_fps) {
          local_current_target_fps = target_fps;
          target_frame_duration_seconds =
            std::chrono::duration<double>(1.0 / local_current_target_fps);
          next_frame_time = intended_current_frame_time + target_frame_duration_seconds;
        }
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

      // Adjust dimensions for H.264 if mode or dimensions changed
      if (local_current_output_mode == OutputMode::H264) {
        if (local_capture_width_actual % 2 != 0 && local_capture_width_actual > 0) {
          local_capture_width_actual--;
        }
        if (local_capture_height_actual % 2 != 0 && local_capture_height_actual > 0) {
          local_capture_height_actual--;
        }
      }
      
      // Handle capture dimension changes: Re-initialize XShm
      if (old_w != local_capture_width_actual || old_h != local_capture_height_actual) {
        std::cout << "Capture dimensions changed from " << old_w << "x" << old_h 
                  << " to " << local_capture_width_actual << "x"
                  << local_capture_height_actual 
                  << ". Re-initializing XShm." << std::endl;
        XShmDetach(display, &shminfo);
        shmdt(shminfo.shmaddr);
        shmctl(shminfo.shmid, IPC_RMID, 0);
        if (shm_image) XDestroyImage(shm_image);
        shm_image = nullptr;

        shm_image = XShmCreateImage(
          display, DefaultVisual(display, screen), DefaultDepth(display, screen),
          ZPixmap, nullptr, &shminfo, local_capture_width_actual,
          local_capture_height_actual);
        if (!shm_image) {
          std::cerr << "Error: XShmCreateImage failed during re-init." << std::endl;
          XCloseDisplay(display); return;
        }
        shminfo.shmid = shmget(
          IPC_PRIVATE, shm_image->bytes_per_line * shm_image->height,
          IPC_CREAT | 0600);
        if (shminfo.shmid < 0) {
          perror("shmget re-init"); XDestroyImage(shm_image);
          XCloseDisplay(display); return;
        }
        shminfo.shmaddr = (char*)shmat(shminfo.shmid, nullptr, 0);
        if (shminfo.shmaddr == (char*)-1) {
          perror("shmat re-init"); shmctl(shminfo.shmid, IPC_RMID, 0);
          XDestroyImage(shm_image); XCloseDisplay(display); return;
        }
        shminfo.readOnly = False;
        shm_image->data = shminfo.shmaddr;
        if (!XShmAttach(display, &shminfo)) {
          std::cerr << "Error: XShmAttach failed during re-init." << std::endl;
          shmdt(shminfo.shmaddr); shmctl(shminfo.shmid, IPC_RMID, 0);
          XDestroyImage(shm_image); XCloseDisplay(display); return;
        }
        std::cout << "XShm re-initialization complete." << std::endl;
        g_h264_minimal_store.reset(); // Reset encoders due to dimension change
      }

      // Capture screen image using XShm
      if (XShmGetImage(display, root_window, shm_image,
                       local_capture_x_offset, local_capture_y_offset, AllPlanes)) {
        // Convert XImage (BGRA or similar) to RGB24 format
        std::vector<unsigned char> full_rgb_data(
          static_cast<size_t>(local_capture_width_actual) *
          local_capture_height_actual * 3);
        unsigned char* shm_data_ptr = (unsigned char*)shm_image->data;
        int bytes_per_pixel_shm = shm_image->bits_per_pixel / 8;
        int bytes_per_line_shm = shm_image->bytes_per_line;

        for (int y = 0; y < local_capture_height_actual; ++y) {
          for (int x = 0; x < local_capture_width_actual; ++x) {
            unsigned char* pixel_ptr =
              shm_data_ptr + (static_cast<size_t>(y) * bytes_per_line_shm) +
              (static_cast<size_t>(x) * bytes_per_pixel_shm);
            size_t base_idx = (static_cast<size_t>(y) * local_capture_width_actual + x) * 3;
            full_rgb_data[base_idx + 0] = pixel_ptr[2]; // R
            full_rgb_data[base_idx + 1] = pixel_ptr[1]; // G
            full_rgb_data[base_idx + 2] = pixel_ptr[0]; // B
          }
        }

        std::vector<std::future<StripeEncodeResult>> futures;
        std::vector<std::thread> threads;
        
        // Determine the number of processing stripes based on mode and height
        int N_processing_stripes = num_stripes_config;
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
          } else { // JPEG mode
            N_processing_stripes =
              std::min(num_stripes_config, local_capture_height_actual);
            if (N_processing_stripes == 0 && local_capture_height_actual > 0) {
              N_processing_stripes = 1;
            }
          }
        }
        if (N_processing_stripes == 0 && local_capture_height_actual > 0) {
           N_processing_stripes = 1; // Ensure at least one stripe if height > 0
        }

        // Calculate H.264 stripe heights to ensure they are even
        int h264_base_even_height = 0;
        int h264_num_stripes_with_extra_pair = 0;
        int current_y_start_for_stripe = 0; 

        if (local_current_output_mode == OutputMode::H264 &&
            N_processing_stripes > 0 && local_capture_height_actual > 0) {
          int H = local_capture_height_actual;
          int N = N_processing_stripes;
          int base_h = H / N;
          // Ensure base height is even and positive
          h264_base_even_height = (base_h > 0) ? (base_h - (base_h % 2)) : 0;
          if (h264_base_even_height == 0 && H >= 2) { 
            h264_base_even_height = 2; // Smallest even height
          } else if (h264_base_even_height == 0 && H > 0) {
             std::cerr << "Warning: H.264 stripe height calculation error for H="
                       << H << std::endl;
             N_processing_stripes = 0; // Cannot proceed with 0-height stripes
          }

          if (h264_base_even_height > 0) {
            int H_base_covered = h264_base_even_height * N;
            int H_remaining = H - H_base_covered;
            if (H_remaining < 0) H_remaining = 0; // Should not happen with correct logic
            // Distribute remaining height (must be in pairs of 2 pixels)
            h264_num_stripes_with_extra_pair = H_remaining / 2;
            h264_num_stripes_with_extra_pair =
              std::min(h264_num_stripes_with_extra_pair, N);
          } else if (H > 0) { // Error case if base height couldn't be determined
            std::cerr << "Warning: Could not calculate a positive even base "
                      << "height for H.264 stripes (H=" << H << ", N=" << N
                      << "). Stripe heights may be zero or invalid." << std::endl;
             N_processing_stripes = 0;
          }
        }
        bool any_stripe_encoded_this_frame = false;

        // Determine H.264 colorspace and range based on h264_fullcolor setting
        int derived_h264_colorspace_setting;
        bool derived_h264_use_full_range;
        if (local_current_h264_fullcolor) {
          derived_h264_colorspace_setting = 444; // YUV444
          derived_h264_use_full_range = true;
        } else {
          derived_h264_colorspace_setting = 420; // YUV420
          derived_h264_use_full_range = false;
        }

        // --- Stripe Processing Loop ---
        for (int i = 0; i < N_processing_stripes; ++i) {
          int start_y = current_y_start_for_stripe; 
          int current_stripe_height = 0;

          // Calculate stripe height based on output mode
          if (local_current_output_mode == OutputMode::H264) {
            if (h264_base_even_height > 0) {
              current_stripe_height = h264_base_even_height;
              if (i < h264_num_stripes_with_extra_pair) {
                current_stripe_height += 2; // Add extra 2 pixels
              }
            } else if (N_processing_stripes == 1) { // Single stripe takes full height
                current_stripe_height = local_capture_height_actual;
            } else {
                current_stripe_height = 0; // Error or no height
            }
          } else { // JPEG mode
            if (N_processing_stripes > 0) {
                int base_stripe_height_jpeg =
                  local_capture_height_actual / N_processing_stripes;
                int remainder_height_jpeg =
                  local_capture_height_actual % N_processing_stripes;
                // Distribute remainder pixels
                start_y = i * base_stripe_height_jpeg + std::min(i, remainder_height_jpeg);
                current_stripe_height =
                  base_stripe_height_jpeg + (i < remainder_height_jpeg ? 1 : 0);
            } else {
                current_stripe_height = 0;
            }
          }

          if (current_stripe_height <= 0) {
            continue; // Skip if stripe has no height
          }

          // Adjust last stripe's height if it exceeds total capture height
          if (start_y + current_stripe_height > local_capture_height_actual) {
             current_stripe_height = local_capture_height_actual - start_y;
             if (current_stripe_height <= 0) continue;
             // Ensure H.264 stripe height remains even
             if (local_current_output_mode == OutputMode::H264 &&
                 current_stripe_height % 2 != 0 && current_stripe_height > 0) {
                 current_stripe_height--;
             }
             if (current_stripe_height <= 0) continue;
          }

          // For H.264, update Y start for the next stripe based on current one
          if (local_current_output_mode == OutputMode::H264) {
            current_y_start_for_stripe = start_y + current_stripe_height;
          }
          
          // Extract RGB data for the current stripe
          std::vector<unsigned char> stripe_rgb_data_for_processing(
            static_cast<size_t>(local_capture_width_actual) * current_stripe_height * 3);
          int row_stride_rgb = local_capture_width_actual * 3;

          for (int y_offset = 0; y_offset < current_stripe_height; ++y_offset) {
            int global_y = start_y + y_offset;
            size_t dest_offset = static_cast<size_t>(y_offset) * row_stride_rgb;
            size_t src_offset = static_cast<size_t>(global_y) * row_stride_rgb;
            if (global_y < local_capture_height_actual && 
                (src_offset + row_stride_rgb) <= full_rgb_data.size()) {
              std::memcpy(&stripe_rgb_data_for_processing[dest_offset], 
                          &full_rgb_data[src_offset], 
                          row_stride_rgb);
            } else { // Should not happen with correct height calculation
              std::memset(&stripe_rgb_data_for_processing[dest_offset], 0, row_stride_rgb);
            }
          }

          // --- Change Detection and Encoding Logic ---
          uint64_t current_hash = calculate_stripe_hash(stripe_rgb_data_for_processing);
          bool send_this_stripe = false;
          bool is_h264_idr_paintover_on_undamaged_this_stripe = false;

          if (current_hash == previous_hashes[i]) { // Stripe content unchanged
            no_motion_frame_counts[i]++;
            // Trigger paint-over if no motion for a certain number of frames
            if (no_motion_frame_counts[i] >= local_current_paint_over_trigger_frames &&
                !paint_over_sent[i] && !damage_blocked[i]) {
              if (local_current_output_mode == OutputMode::JPEG) {
                if (local_current_use_paint_over_quality) {
                  send_this_stripe = true; // Send high-quality JPEG
                }
              } else { // H264 mode
                send_this_stripe = true; // Send an IDR frame for paint-over
                is_h264_idr_paintover_on_undamaged_this_stripe = true;
                { // Set force_idr_flag for this encoder instance
                  std::lock_guard<std::mutex> lock(g_h264_minimal_store.store_mutex);
                  if (i < static_cast<int>(g_h264_minimal_store.force_idr_flags.size())) {
                    g_h264_minimal_store.force_idr_flags[i] = true;
                  }
                }
              }
              if (send_this_stripe) paint_over_sent[i] = true;
            }
          } else { // Stripe content changed
            no_motion_frame_counts[i] = 0;
            paint_over_sent[i] = false;
            send_this_stripe = true;
            previous_hashes[i] = current_hash;
            // Damage blocking: If too many changes, temporarily stop sending
            damage_block_counts[i]++;
            if (damage_block_counts[i] >= local_current_damage_block_threshold) {
              damage_blocked[i] = true;
              damage_block_timer[i] = local_current_damage_block_duration;
            }
          }

          // If stripe needs to be sent, launch encoding task
          if (send_this_stripe) {
            any_stripe_encoded_this_frame = true;
            total_stripes_encoded_this_interval++;
            if (local_current_output_mode == OutputMode::JPEG) {
              int quality_to_use =
                (paint_over_sent[i] && local_current_use_paint_over_quality)
                  ? local_current_paint_over_jpeg_quality
                  : current_jpeg_qualities[i];
              if (current_hash != previous_hashes[i]) { // If changed, slightly reduce quality
                current_jpeg_qualities[i] =
                  std::max(current_jpeg_qualities[i] - 1, local_current_jpeg_quality);
              }
              std::packaged_task<StripeEncodeResult(
                int, int, int, int, int, int, const unsigned char*, int, int, int)>
                task(encode_stripe_jpeg);
              futures.push_back(task.get_future());
              threads.push_back(std::thread(
                std::move(task), i, start_y, current_stripe_height,
                DisplayWidth(display, screen), // Full screen width (context)
                local_capture_height_actual,   // Full capture height (context)
                local_capture_width_actual,    // Actual stripe width
                full_rgb_data.data(),          // Full frame data
                static_cast<int>(full_rgb_data.size()),
                quality_to_use,
                this->frame_counter));
            } else { // H264 mode
              int crf_for_encode = local_current_h264_crf;
              // Use lower CRF (higher quality) for IDR paint-over frames
              if (is_h264_idr_paintover_on_undamaged_this_stripe &&
                  local_current_h264_crf > 10) {
                crf_for_encode = 10;
              }
              std::packaged_task<StripeEncodeResult(
                int, int, int, int, const unsigned char*, int, int, int, bool)>
                task(encode_stripe_h264);
              futures.push_back(task.get_future());
              // Lambda to capture necessary data for the H.264 encoding thread
              threads.push_back(std::thread(
                [task_moved = std::move(task), i, start_y, current_stripe_height,
                 local_capture_width_actual,
                 data_copy = stripe_rgb_data_for_processing, // Copy stripe data
                 fc = this->frame_counter,
                 crf_val = crf_for_encode,
                 cs = derived_h264_colorspace_setting,
                 fr = derived_h264_use_full_range]() mutable {
                  task_moved(i, start_y, current_stripe_height,
                             local_capture_width_actual,
                             data_copy.data(), fc, crf_val, cs, fr);
                }));
            }
          }

          // Damage block timer countdown
          if (damage_block_timer[i] > 0) {
            damage_block_timer[i]--;
            if (damage_block_timer[i] == 0) {
              damage_blocked[i] = false;
              damage_block_counts[i] = 0;
              // Reset JPEG quality when damage block ends
              if (local_current_output_mode == OutputMode::JPEG) {
                current_jpeg_qualities[i] =
                  local_current_use_paint_over_quality
                    ? local_current_paint_over_jpeg_quality
                    : local_current_jpeg_quality;
              }
            }
          }
        } // End of stripe processing loop

        // --- Collect Encoding Results ---
        std::vector<StripeEncodeResult> stripe_results;
        stripe_results.reserve(futures.size());
        for (auto& future : futures) {
          stripe_results.push_back(future.get()); // Wait for and get result
        }
        futures.clear();

        // Process results and invoke callback
        for (StripeEncodeResult& result : stripe_results) {
          if (stripe_callback != nullptr && result.data != nullptr && result.size > 0) {
            stripe_callback(&result, user_data);
          } else {
             if (result.data) { // If data exists but callback is null or size is 0
                free_stripe_encode_result_data(&result);
             }
          }
        }
        stripe_results.clear();

        // Join all encoding threads
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

        // --- Logging Performance Metrics ---
        auto current_time_for_fps_log = std::chrono::high_resolution_clock::now();
        auto elapsed_time_for_fps_log =
          std::chrono::duration_cast<std::chrono::seconds>(
            current_time_for_fps_log - start_time_loop);

        // Reset loop-specific frame counter periodically (e.g., every second)
        if (elapsed_time_for_fps_log.count() >= 1) {
          frame_count_loop = 0;
          start_time_loop = std::chrono::high_resolution_clock::now();
        }

        auto current_output_time_log = std::chrono::high_resolution_clock::now();
        auto output_elapsed_time_log =
          std::chrono::duration_cast<std::chrono::seconds>(
            current_output_time_log - last_output_time);

        // Log stats every second
        if (output_elapsed_time_log.count() >= 1) {
          double actual_fps_val =
            (encoded_frame_count > 0 && output_elapsed_time_log.count() > 0)
            ? static_cast<double>(encoded_frame_count) / output_elapsed_time_log.count()
            : 0.0;
          double total_stripes_per_second_val =
            (total_stripes_encoded_this_interval > 0 && output_elapsed_time_log.count() > 0)
            ? static_cast<double>(total_stripes_encoded_this_interval) / output_elapsed_time_log.count()
            : 0.0;

          std::cout << "Res: " << local_capture_width_actual << "x"
                    << local_capture_height_actual
                    << " Mode: " << (local_current_output_mode == OutputMode::JPEG ? "JPEG" : "H264")
                    << (local_current_output_mode == OutputMode::H264
                        ? (std::string(local_current_h264_fullcolor ? " CS:444 FR" : " CS:420 LR") +
                           (local_current_h264_fullframe ? " FF" : ""))
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

      } else { // XShmGetImage failed
        std::cerr << "Failed to capture XImage using XShmGetImage" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Wait before retrying
      }
    } // End of main capture while loop

    // --- Cleanup X11 Resources ---
    XShmDetach(display, &shminfo);
    shmdt(shminfo.shmaddr);
    shmctl(shminfo.shmid, IPC_RMID, 0);
    if (shm_image) XDestroyImage(shm_image);
    XCloseDisplay(display);
    std::cout << "Capture loop stopped. X resources released." << std::endl;
  } // End of capture_loop method
}; // End of ScreenCaptureModule class

// --- C ABI for External Usage ---
extern "C" {

  // Opaque handle type for ScreenCaptureModule
  typedef void* ScreenCaptureModuleHandle;

  /**
   * @brief Creates a new ScreenCaptureModule instance.
   * @return Handle to the created module, or nullptr on failure.
   */
  ScreenCaptureModuleHandle create_screen_capture_module() {
    return static_cast<ScreenCaptureModuleHandle>(new ScreenCaptureModule());
  }

  /**
   * @brief Destroys a ScreenCaptureModule instance.
   * @param module_handle Handle to the module to be destroyed.
   */
  void destroy_screen_capture_module(ScreenCaptureModuleHandle module_handle) {
    if (module_handle) {
      delete static_cast<ScreenCaptureModule*>(module_handle);
    }
  }

  /**
   * @brief Starts the screen capture process with specified settings and callback.
   * @param module_handle Handle to the ScreenCaptureModule.
   * @param settings CaptureSettings to apply.
   * @param callback Function to call with encoded stripe data.
   * @param user_data User-defined data to pass to the callback.
   */
  void start_screen_capture(ScreenCaptureModuleHandle module_handle,
                            CaptureSettings settings,
                            StripeCallback callback,
                            void* user_data) {
    if (module_handle) {
      ScreenCaptureModule* module = static_cast<ScreenCaptureModule*>(module_handle);
      module->modify_settings(settings); 

      std::lock_guard<std::mutex> lock(module->settings_mutex); // Protect callback/user_data
      module->stripe_callback = callback;
      module->user_data = user_data;
      module->start_capture();
    }
  }

  /**
   * @brief Stops the screen capture process for the given module.
   * @param module_handle Handle to the ScreenCaptureModule.
   */
  void stop_screen_capture(ScreenCaptureModuleHandle module_handle) {
    if (module_handle) {
      static_cast<ScreenCaptureModule*>(module_handle)->stop_capture();
    }
  }

  /**
   * @brief Modifies the settings of an active or inactive ScreenCaptureModule.
   * @param module_handle Handle to the ScreenCaptureModule.
   * @param settings New CaptureSettings to apply.
   */
  void modify_screen_capture(ScreenCaptureModuleHandle module_handle,
                             CaptureSettings settings) {
    if (module_handle) {
      static_cast<ScreenCaptureModule*>(module_handle)->modify_settings(settings);
    }
  }

  /**
   * @brief Retrieves the current settings of a ScreenCaptureModule.
   * @param module_handle Handle to the ScreenCaptureModule.
   * @return Current CaptureSettings, or default settings if handle is invalid.
   */
  CaptureSettings get_screen_capture_settings(ScreenCaptureModuleHandle module_handle) {
    if (module_handle) {
      return static_cast<ScreenCaptureModule*>(module_handle)->get_current_settings();
    } else {
      return CaptureSettings{}; // Return default settings on error
    }
  }

  /**
   * @brief Frees the dynamically allocated data buffer within a StripeEncodeResult.
   * This function is typically called by the consumer of the stripe data
   * after processing it.
   * @param result Pointer to the StripeEncodeResult whose data buffer is to be freed.
   */
  void free_stripe_encode_result_data(StripeEncodeResult* result) {
    if (result && result->data) {
      delete[] result->data;
      result->data = nullptr;
      // result->size is not reset here as the struct might be reused,
      // but data pointer being null indicates no valid data.
    }
  }

} // extern "C"

// --- Encoder Implementations ---

/**
 * @brief Encodes a stripe of image data into JPEG format.
 * This function takes a portion of a larger image (defined by stripe_y_start
 * and stripe_height within the full rgb_data) and compresses it as a JPEG.
 * A small header (frame counter, stripe Y start) is prepended to the JPEG data.
 * @param thread_id An identifier for the thread, mostly for logging.
 * @param stripe_y_start The Y-coordinate of the top edge of this stripe in the full image.
 * @param stripe_height The height of this stripe.
 * @param width The width of the full source image (unused, capture_width_actual is used).
 * @param height The height of the full source image.
 * @param capture_width_actual The actual width of the stripe to be encoded.
 * @param rgb_data Pointer to the beginning of the full RGB image data.
 * @param rgb_data_len Total length of the rgb_data buffer (unused).
 * @param jpeg_quality Quality setting for JPEG compression (0-100).
 * @param frame_counter Identifier for the current frame.
 * @return StripeEncodeResult containing the encoded JPEG data and metadata.
 *         If an error occurs, type is UNKNOWN and data is nullptr.
 */
StripeEncodeResult encode_stripe_jpeg(
  int thread_id,
  int stripe_y_start,
  int stripe_height,
  int width, // Unused in current logic, capture_width_actual is primary
  int height,
  int capture_width_actual,
  const unsigned char* rgb_data,
  int rgb_data_len, // Unused in current logic
  int jpeg_quality,
  int frame_counter) {
  StripeEncodeResult result;
  result.type = StripeDataType::JPEG;
  result.stripe_y_start = stripe_y_start;
  result.stripe_height = stripe_height;
  result.frame_id = frame_counter;

  // Input validation
  if (!rgb_data || stripe_height <= 0 || capture_width_actual <= 0) {
    std::cerr << "JPEG T" << thread_id 
              << ": Invalid input for JPEG encoding." << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }

  // Initialize JPEG compression structures
  jpeg_compress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  // Set JPEG parameters
  cinfo.image_width = capture_width_actual;
  cinfo.image_height = stripe_height;
  cinfo.input_components = 3; // RGB
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, jpeg_quality, TRUE); // TRUE for_baseline_JPEG

  // Set up memory destination for JPEG output
  unsigned char* jpeg_buffer = nullptr;
  unsigned long jpeg_size_temp = 0; // libjpeg uses unsigned long
  jpeg_mem_dest(&cinfo, &jpeg_buffer, &jpeg_size_temp);

  // Start compression
  jpeg_start_compress(&cinfo, TRUE);

  // Write scanlines
  JSAMPROW row_pointer[1];
  int row_stride = capture_width_actual * 3;
  
  for (int y_in_stripe = 0; y_in_stripe < stripe_height; ++y_in_stripe) {
    int global_y = stripe_y_start + y_in_stripe; // Y-coordinate in the full image
    if (global_y < height) { // Ensure we are within bounds of the source image
      // Point to the correct row in the full RGB data buffer
      row_pointer[0] = const_cast<unsigned char*>(
        rgb_data + (static_cast<size_t>(global_y) * row_stride));
      jpeg_write_scanlines(&cinfo, row_pointer, 1);
    } else {
      // If somehow y_in_stripe goes beyond source image height (e.g. bad input height),
      // write a black row to avoid reading out of bounds.
      std::vector<unsigned char> black_row(row_stride, 0);
      row_pointer[0] = black_row.data();
      jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }
  }

  // Finish compression
  jpeg_finish_compress(&cinfo);

  // Prepare result data with custom header
  if (jpeg_size_temp > 0 && jpeg_buffer) {
    int padding_size = 4; // For frame_counter (2 bytes) and stripe_y_start (2 bytes)
    result.data = new (std::nothrow) unsigned char[jpeg_size_temp + padding_size];
    if (!result.data) {
      std::cerr << "JPEG T" << thread_id 
                << ": Failed to allocate memory for JPEG output." << std::endl;
      jpeg_destroy_compress(&cinfo);
      if (jpeg_buffer) free(jpeg_buffer); // jpeg_mem_dest might use malloc
      result.type = StripeDataType::UNKNOWN;
      return result;
    }

    // Prepare header data (network byte order)
    uint16_t frame_counter_net = htons(static_cast<uint16_t>(frame_counter % 65536));
    uint16_t stripe_y_start_net = htons(static_cast<uint16_t>(stripe_y_start));

    // Copy header and JPEG data
    std::memcpy(result.data, &frame_counter_net, 2);
    std::memcpy(result.data + 2, &stripe_y_start_net, 2);
    std::memcpy(result.data + padding_size, jpeg_buffer, jpeg_size_temp);
    result.size = static_cast<int>(jpeg_size_temp) + padding_size;
  } else {
    result.size = 0;
    result.data = nullptr;
    if (jpeg_size_temp == 0) {
      std::cerr << "JPEG T" << thread_id 
                << ": Compression resulted in 0 size." << std::endl;
    }
  }

  // Clean up JPEG resources
  jpeg_destroy_compress(&cinfo);
  if (jpeg_buffer) {
    free(jpeg_buffer); // Free buffer allocated by jpeg_mem_dest
  }
  return result;
}

/**
 * @brief Encodes a stripe of RGB data into H.264 format using x264.
 * Manages x264 encoder instances per thread, re-initializing them if parameters
 * (dimensions, colorspace, CRF) change. Converts RGB to YUV (I420 or I444)
 * before encoding. Prepends a custom header to the H.264 NAL units.
 * @param thread_id Identifier for the calling thread, used for encoder management.
 * @param stripe_y_start The Y-coordinate of the top of the stripe.
 * @param stripe_height The height of the stripe (must be even).
 * @param capture_width_actual The width of the stripe (must be even).
 * @param stripe_rgb24_data Pointer to the RGB24 data for this specific stripe.
 * @param frame_counter The current frame number (used for PTS).
 * @param current_crf_setting The CRF value for H.264 encoding.
 * @param colorspace_setting Target colorspace: 444 for I444, 420 for I420.
 * @param use_full_range True for full range video, false for limited range.
 * @return StripeEncodeResult containing H.264 NAL units and metadata.
 *         If an error occurs, type is UNKNOWN and data is nullptr.
 */
StripeEncodeResult encode_stripe_h264(
  int thread_id,
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

  // Input validation
  if (!stripe_rgb24_data) {
    std::cerr << "H264 T" << thread_id << ": Error - null rgb_data for stripe Y"
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
  // H.264 typically requires even dimensions for chroma subsampling compatibility
  if (capture_width_actual % 2 != 0 || stripe_height % 2 != 0) {
    std::cerr << "H264 T" << thread_id << ": Warning - Odd dimensions ("
              << capture_width_actual << "x" << stripe_height 
              << ") for stripe Y" << stripe_y_start
              << ". Encoder might behave unexpectedly." << std::endl;
  }

  x264_t* current_encoder = nullptr;
  x264_picture_t* current_pic_in_ptr = nullptr;
  int target_x264_csp;
  int actual_colorspace_setting_for_reinit = colorspace_setting; 

  // Determine x264 colorspace based on input setting
  switch (colorspace_setting) {
    case 444: 
      target_x264_csp = X264_CSP_I444; 
      break;
    case 420:
    default:  // Default to I420 if an unsupported value is given
      target_x264_csp = X264_CSP_I420; 
      actual_colorspace_setting_for_reinit = 420; // Standardize for reinit check
      break;
  }
  
  // --- Encoder Management and Re-initialization ---
  { // Scope for store_mutex to protect g_h264_minimal_store
    std::lock_guard<std::mutex> lock(g_h264_minimal_store.store_mutex);
    g_h264_minimal_store.ensure_size(thread_id); // Ensure vectors are large enough

    // Determine if encoder re-initialization is needed
    bool is_first_init = !g_h264_minimal_store.initialized_flags[thread_id];
    bool dims_changed = !is_first_init && 
                        (g_h264_minimal_store.initialized_widths[thread_id] != capture_width_actual ||
                         g_h264_minimal_store.initialized_heights[thread_id] != stripe_height);
    bool cs_or_fr_changed = !is_first_init && 
                            (g_h264_minimal_store.initialized_csps[thread_id] != target_x264_csp ||
                             g_h264_minimal_store.initialized_colorspaces[thread_id] != actual_colorspace_setting_for_reinit || 
                             g_h264_minimal_store.initialized_full_range_flags[thread_id] != use_full_range);
    
    // Special case: if forcing IDR for paint-over with CRF 10, and current CRF is higher,
    // re-init to ensure CRF 10 is applied correctly for this IDR frame.
    bool needs_crf10_reinit = false;
    if (!is_first_init && !dims_changed && !cs_or_fr_changed &&
        g_h264_minimal_store.force_idr_flags[thread_id] &&
        current_crf_setting == 10 &&
        g_h264_minimal_store.initialized_crfs[thread_id] > 10) {
      needs_crf10_reinit = true;
    }

    bool perform_full_reinit = is_first_init || dims_changed || cs_or_fr_changed || needs_crf10_reinit;
    bool log_x264_info = is_first_init || dims_changed || cs_or_fr_changed; // Log more on significant changes

    if (perform_full_reinit) {
      // Clean up existing encoder and picture if they exist
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

      // Initialize x264 parameters
      x264_param_t param;
      if (x264_param_default_preset(&param, "ultrafast", "zerolatency") < 0) {
        std::cerr << "H264 T" << thread_id 
                  << ": x264_param_default_preset FAILED." << std::endl;
        result.type = StripeDataType::UNKNOWN; 
      } else {
        param.i_width = capture_width_actual;
        param.i_height = stripe_height;
        param.i_csp = target_x264_csp; 
        param.i_fps_num = 60; // Assuming target 60 FPS for encoder settings
        param.i_fps_den = 1;
        param.i_keyint_max = X264_KEYINT_MAX_INFINITE; // No automatic IDR frames
        param.rc.f_rf_constant = static_cast<float>(std::max(0, std::min(51, current_crf_setting)));
        param.rc.i_rc_method = X264_RC_CRF;
        param.b_repeat_headers = 1; // Repeat SPS/PPS before IDR frames
        param.b_annexb = 1;         // Output Annex B NAL units
        param.i_sync_lookahead = 0; // Required by zerolatency
        param.i_bframe = 0;         // No B-frames for low latency
        param.i_threads = -1;
        param.i_log_level = log_x264_info ? X264_LOG_INFO : X264_LOG_ERROR;
        param.vui.b_fullrange = use_full_range ? 1 : 0;
        param.vui.i_sar_width = 1;   // Assuming 1:1 SAR for screen content
        param.vui.i_sar_height = 1;
        param.vui.i_colorprim = 1;
        param.vui.i_transfer = 13;   // sRGB transfer characteristics
        param.vui.i_colmatrix = 1;   // BT.709 matrix (common for sRGB-like content) 
        param.b_aud = 1;

        // Apply profile based on colorspace
        if (param.i_csp == X264_CSP_I444) {
          if (x264_param_apply_profile(&param, "high444") < 0) {
            if (log_x264_info) { 
              std::cerr << "H264 T" << thread_id 
                        << ": Warning - Failed to apply 'high444' profile for 4:4:4. "
                        << "x264 might use a different profile or operate without one." << std::endl;
            }
          } else {
            if (log_x264_info) {
              std::cout << "H264 T" << thread_id 
                        << ": Applied 'high444' profile for 4:4:4 encoding." << std::endl;
            }
          }
        } else { // For X264_CSP_I420
          if (x264_param_apply_profile(&param, "baseline") < 0 && log_x264_info) {
             std::cerr << "H264 T" << thread_id 
                       << ": Warning - Failed to apply 'baseline' profile (non-fatal)." << std::endl;
          }
        }

        // Open encoder
        g_h264_minimal_store.encoders[thread_id] = x264_encoder_open(&param);
        if (!g_h264_minimal_store.encoders[thread_id]) {
          std::cerr << "H264 T" << thread_id << ": x264_encoder_open FAILED." << std::endl;
          result.type = StripeDataType::UNKNOWN;
        } else {
          // Allocate picture structure for input frames
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
              // Successfully initialized
              g_h264_minimal_store.initialized_flags[thread_id] = true;
              g_h264_minimal_store.initialized_widths[thread_id] = param.i_width;
              g_h264_minimal_store.initialized_heights[thread_id] = param.i_height;
              g_h264_minimal_store.initialized_crfs[thread_id] = current_crf_setting;
              g_h264_minimal_store.initialized_csps[thread_id] = param.i_csp;
              g_h264_minimal_store.initialized_colorspaces[thread_id] = actual_colorspace_setting_for_reinit; 
              g_h264_minimal_store.initialized_full_range_flags[thread_id] = use_full_range;
              g_h264_minimal_store.force_idr_flags[thread_id] = true; // Force IDR on first frame after init
            }
          }
        }
      }
    } else if (g_h264_minimal_store.initialized_crfs[thread_id] != current_crf_setting) {
      // Only CRF changed, try to reconfigure encoder
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

    // Assign current encoder and picture if initialization was successful
    if (g_h264_minimal_store.initialized_flags[thread_id]) { 
      current_encoder = g_h264_minimal_store.encoders[thread_id];
      current_pic_in_ptr = g_h264_minimal_store.pics_in_ptrs[thread_id];
    }
  } // End of store_mutex scope

  if (result.type == StripeDataType::UNKNOWN) return result; // Exit if re-init failed
  if (!current_encoder || !current_pic_in_ptr) {
    std::cerr << "H264 T" << thread_id << ": Encoder/Pic not ready post-init for Y"
              << stripe_y_start << "." << std::endl;
    result.type = StripeDataType::UNKNOWN; return result;
  }
  
  // Verify that picture planes are allocated (sanity check)
  bool planes_ok = current_pic_in_ptr->img.plane[0] && current_pic_in_ptr->img.plane[1];
  if (target_x264_csp == X264_CSP_I420 || target_x264_csp == X264_CSP_I444) {
    planes_ok = planes_ok && current_pic_in_ptr->img.plane[2]; 
  }
  if (!planes_ok) {
    std::cerr << "H264 T" << thread_id << ": Pic planes NULL for CSP " << target_x264_csp 
              << " (Y" << stripe_y_start << "). Bug." << std::endl;
    result.type = StripeDataType::UNKNOWN; return result;
  }

  // --- Color Conversion (RGB to YUV) using libyuv ---
  int src_stride_rgb24 = capture_width_actual * 3;
  int conversion_status = -1;

  if (target_x264_csp == X264_CSP_I444) { // RGB to I444 (YUV 4:4:4)
    conversion_status = libyuv::RAWToI444( 
      stripe_rgb24_data, src_stride_rgb24,
      current_pic_in_ptr->img.plane[0], current_pic_in_ptr->img.i_stride[0], // Y plane
      current_pic_in_ptr->img.plane[1], current_pic_in_ptr->img.i_stride[1], // U plane
      current_pic_in_ptr->img.plane[2], current_pic_in_ptr->img.i_stride[2], // V plane
      capture_width_actual, stripe_height);
  } else { // RGB to I420 (YUV 4:2:0)
    conversion_status = libyuv::RAWToI420( 
      stripe_rgb24_data, src_stride_rgb24,
      current_pic_in_ptr->img.plane[0], current_pic_in_ptr->img.i_stride[0], // Y plane
      current_pic_in_ptr->img.plane[1], current_pic_in_ptr->img.i_stride[1], // U plane
      current_pic_in_ptr->img.plane[2], current_pic_in_ptr->img.i_stride[2], // V plane
      capture_width_actual, stripe_height);
  }

  if (conversion_status != 0) {
    std::cerr << "H264 T" << thread_id << ": libyuv conversion to CSP " << target_x264_csp 
              << " FAILED code " << conversion_status << " (Y" << stripe_y_start << ")"
              << std::endl;
    result.type = StripeDataType::UNKNOWN; return result;
  }

  // --- Encode Frame ---
  current_pic_in_ptr->i_pts = static_cast<int64_t>(frame_counter); // Presentation timestamp

  // Check if an IDR frame should be forced for this stripe
  bool force_idr_now = false;
  { // Scope for store_mutex
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
  x264_picture_t pic_out; // Output picture from encoder
  x264_picture_init(&pic_out); 
  
  int frame_size = x264_encoder_encode(current_encoder, &nals, &i_nals,
                                       current_pic_in_ptr, &pic_out);

  if (frame_size < 0) {
    std::cerr << "H264 T" << thread_id << ": x264_encoder_encode FAILED: " << frame_size
              << " (Y" << stripe_y_start << ")" << std::endl;
    result.type = StripeDataType::UNKNOWN; return result;
  }

  if (frame_size > 0) {
    // If an IDR frame was forced and successfully encoded, reset the flag
    if (force_idr_now && pic_out.b_keyframe && pic_out.i_type == X264_TYPE_IDR) {
      std::lock_guard<std::mutex> lock(g_h264_minimal_store.store_mutex);
      if (thread_id < static_cast<int>(g_h264_minimal_store.force_idr_flags.size())) { 
        g_h264_minimal_store.force_idr_flags[thread_id] = false; 
      }
    }

    // --- Prepare Output Data with Custom Header ---
    const unsigned char DATA_TYPE_H264_STRIPED_TAG = 0x04;
    unsigned char frame_type_header_byte = 0x00; // P-frame or other
    if (pic_out.i_type == X264_TYPE_IDR) frame_type_header_byte = 0x01; // IDR-frame
    else if (pic_out.i_type == X264_TYPE_I) frame_type_header_byte = 0x02; // I-frame

    int header_sz = 10; // Tag(1) + Type(1) + FrameID(2) + YStart(2) + Width(2) + Height(2)
    int total_sz = frame_size + header_sz;
    result.data = new (std::nothrow) unsigned char[total_sz];
    if (!result.data) {
      std::cerr << "H264 T" << thread_id << ": new result.data FAILED (Y"
                << stripe_y_start << ")" << std::endl;
      result.type = StripeDataType::UNKNOWN; return result;
    }

    // Fill header (network byte order for multi-byte fields)
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

    // Copy NAL units to payload
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
  } else { // frame_size == 0 (no output from encoder for this frame)
    result.data = nullptr;
    result.size = 0;
  }
  return result;
}

/**
 * @brief Calculates a 64-bit hash of the provided RGB data using XXH3.
 * This is used to detect changes in stripe content between frames.
 * @param rgb_data A vector of unsigned char containing the RGB pixel data.
 * @return A 64-bit hash value. Returns 0 if the input data is empty.
 */
uint64_t calculate_stripe_hash(const std::vector<unsigned char>& rgb_data) {
  if (rgb_data.empty()) return 0;
  return XXH3_64bits(rgb_data.data(), rgb_data.size());
}
