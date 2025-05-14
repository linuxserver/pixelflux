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
struct MinimalEncoderStore {
  std::vector < x264_t * > encoders;
  std::vector < x264_picture_t * > pics_in_ptrs;
  std::vector < bool > initialized_flags;
  std::vector < int > initialized_widths;
  std::vector < int > initialized_heights;
  std::vector < bool > force_idr_flags;
  std::mutex store_mutex;

  void ensure_size(int thread_id) {
    if (thread_id >= static_cast < int > (encoders.size())) {
      size_t new_size = static_cast < size_t > (thread_id) + 1;
      encoders.resize(new_size, nullptr);
      pics_in_ptrs.resize(new_size, nullptr);
      initialized_flags.resize(new_size, false);
      initialized_widths.resize(new_size, 0);
      initialized_heights.resize(new_size, 0);
      force_idr_flags.resize(new_size, false);
    }
  }

  void reset() {
      std::lock_guard < std::mutex > lock(store_mutex);
      for (size_t i = 0; i < encoders.size(); ++i) {
        if (encoders[i]) {
          x264_encoder_close(encoders[i]);
          encoders[i] = nullptr;
        }
        if (pics_in_ptrs[i]) {
          // Add bounds check for safety in case vectors got cleared differently
          if (i < initialized_flags.size() && initialized_flags[i]) {
            x264_picture_clean(pics_in_ptrs[i]);
          }
          delete pics_in_ptrs[i];
          pics_in_ptrs[i] = nullptr;
        }
      }
      // Clear all vectors to fully reset the state
      encoders.clear();
      pics_in_ptrs.clear();
      initialized_flags.clear();
      initialized_widths.clear();
      initialized_heights.clear();
      force_idr_flags.clear();
    }

    ~MinimalEncoderStore() {
      reset();
    }
};

MinimalEncoderStore g_h264_minimal_store; // Global instance

enum class OutputMode {
  JPEG = 0,
    H264 = 1
};

enum class StripeDataType {
  UNKNOWN = 0,
    JPEG = 1,
    H264 = 2
};

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

  CaptureSettings(): capture_width(1920),
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
    h264_crf(25) {}

  CaptureSettings(int cw, int ch, int cx, int cy, double fps, int jq, int pojq, bool upoq,
      int potf, int dbt, int dbd, OutputMode om = OutputMode::JPEG, int crf = 25): capture_width(cw),
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
    h264_crf(crf) {}
};

struct StripeEncodeResult {
  StripeDataType type;
  int stripe_y_start;
  int stripe_height;
  int size;
  unsigned char * data;
  int frame_id;

  StripeEncodeResult(): type(StripeDataType::UNKNOWN), stripe_y_start(0), stripe_height(0), size(0), data(nullptr), frame_id(-1) {}
  StripeEncodeResult(StripeEncodeResult && other) noexcept;
  StripeEncodeResult & operator = (StripeEncodeResult && other) noexcept;

  private:
    StripeEncodeResult(const StripeEncodeResult & ) = delete;
  StripeEncodeResult & operator = (const StripeEncodeResult & ) = delete;
};

StripeEncodeResult::StripeEncodeResult(StripeEncodeResult && other) noexcept: type(other.type),
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

StripeEncodeResult & StripeEncodeResult::operator = (StripeEncodeResult && other) noexcept {
  if (this != & other) {
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
  return * this;
}

typedef void( * StripeCallback)(StripeEncodeResult * result, void * user_data);
extern "C"
void free_stripe_encode_result_data(StripeEncodeResult * result);

StripeEncodeResult encode_stripe_jpeg(int thread_id, int stripe_y_start, int stripe_height,
  int width, int height, int capture_width_actual,
  const unsigned char * rgb_data, int rgb_data_len,
    int jpeg_quality, int frame_counter);

StripeEncodeResult encode_stripe_h264(int thread_id, int stripe_y_start, int stripe_height,
  int capture_width_actual,
  const unsigned char * stripe_rgb24_data,
    int frame_counter, int current_crf_setting);

uint64_t calculate_stripe_hash(const std::vector < unsigned char > & rgb_data);

class ScreenCaptureModule {
  public: int capture_width = 1024;
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
  OutputMode output_mode = OutputMode::H264;

  std::atomic < bool > stop_requested;
  std::thread capture_thread;
  StripeCallback stripe_callback = nullptr;
  void * user_data = nullptr;
  int frame_counter = 0;
  int encoded_frame_count = 0;
  int total_stripes_encoded_this_interval = 0;
  mutable std::mutex settings_mutex;

  public: ScreenCaptureModule(): stop_requested(false) {}

    ~ScreenCaptureModule() {
      stop_capture();
    }

  void start_capture() {
    if (capture_thread.joinable()) {
      stop_capture();
    }
    g_h264_minimal_store.reset();
    stop_requested = false;
    frame_counter = 0;
    encoded_frame_count = 0;
    capture_thread = std::thread( & ScreenCaptureModule::capture_loop, this);
  }

  void stop_capture() {
    stop_requested = true;
    if (capture_thread.joinable()) {
      capture_thread.join();
    }
  }

  void modify_settings(const CaptureSettings & new_settings) {
    std::lock_guard < std::mutex > lock(settings_mutex);
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
  }

  CaptureSettings get_current_settings() const {
    std::lock_guard < std::mutex > lock(settings_mutex);
    return CaptureSettings(capture_width, capture_height, capture_x, capture_y, target_fps,
      jpeg_quality, paint_over_jpeg_quality, use_paint_over_quality,
      paint_over_trigger_frames, damage_block_threshold,
      damage_block_duration, output_mode, h264_crf);
  }

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
    OutputMode local_current_output_mode;

    {
      std::lock_guard < std::mutex > lock(settings_mutex);
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
    }
    // Ensure overall capture dimensions are even if encoding to H.264
    if (local_current_output_mode == OutputMode::H264) {
      if (local_capture_width_actual % 2 != 0) {
        local_capture_width_actual--; // Shave 1px from the right
      }
      if (local_capture_height_actual % 2 != 0) {
        local_capture_height_actual--; // Shave 1px from the bottom
      }
    }
    std::chrono::duration < double > target_frame_duration_seconds =
      std::chrono::duration < double > (1.0 / local_current_target_fps);
    auto next_frame_time = std::chrono::high_resolution_clock::now() + target_frame_duration_seconds;

    char * display_env = std::getenv("DISPLAY");
    const char * display_name = display_env ? display_env : ":0";

    Display * display = XOpenDisplay(display_name);
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
    XImage * shm_image = nullptr;

    shm_image = XShmCreateImage(display, DefaultVisual(display, screen), DefaultDepth(display, screen),
      ZPixmap, nullptr, & shminfo, local_capture_width_actual,
      local_capture_height_actual);
    if (!shm_image) {
      std::cerr << "Error: XShmCreateImage failed" << std::endl;
      XCloseDisplay(display);
      return;
    }

    int shmid = shmget(IPC_PRIVATE, shm_image -> bytes_per_line * shm_image -> height, IPC_CREAT | 0600);
    if (shmid < 0) {
      perror("shmget");
      std::cerr << "Error: shmget failed" << std::endl;
      XDestroyImage(shm_image);
      XCloseDisplay(display);
      return;
    }

    shminfo.shmid = shmid;
    shminfo.shmaddr = (char * ) shmat(shmid, nullptr, 0);
    if (shminfo.shmaddr == (char * ) - 1) {
      perror("shmat");
      std::cerr << "Error: shmat failed" << std::endl;
      shmctl(shmid, IPC_RMID, 0);
      XDestroyImage(shm_image);
      XCloseDisplay(display);
      return;
    }
    shminfo.readOnly = False;
    shm_image -> data = shminfo.shmaddr;

    if (!XShmAttach(display, & shminfo)) {
      std::cerr << "Error: XShmAttach failed" << std::endl;
      shmdt(shminfo.shmaddr);
      shmctl(shmid, IPC_RMID, 0);
      XDestroyImage(shm_image);
      XCloseDisplay(display);
      return;
    }
    std::cout << "XShm setup complete." << std::endl;

    int num_cores = std::max(1, (int) std::thread::hardware_concurrency());
    std::cout << "CPU cores available: " << num_cores << std::endl;
    int num_stripes = num_cores; //num_cores;

    std::vector < uint64_t > previous_hashes(num_stripes, 0);
    std::vector < int > no_motion_frame_counts(num_stripes, 0);
    std::vector < uint64_t > last_paint_over_hashes(num_stripes, 0);
    std::vector < bool > paint_over_sent(num_stripes, false);
    std::vector < int > damage_block_counts(num_stripes, 0);
    std::vector < bool > damage_blocked(num_stripes, false);
    std::vector < int > damage_block_timer(num_stripes, 0);
    std::vector < int > current_jpeg_qualities(num_stripes);

    for (int i = 0; i < num_stripes; ++i) {
      current_jpeg_qualities[i] = local_current_use_paint_over_quality ? local_current_paint_over_jpeg_quality : local_current_jpeg_quality;
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

      {
        std::lock_guard < std::mutex > lock(settings_mutex);
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
        local_current_output_mode = output_mode;
        local_current_h264_crf = h264_crf;
      }

      // Ensure capture width is even when encoding H.264
      if (local_current_output_mode == OutputMode::H264) {
        if (local_capture_width_actual % 2 != 0) {
          local_capture_width_actual--; // Shave 1px from the right if odd
        }
      }

      if (XShmGetImage(display, root_window, shm_image, local_capture_x_offset, local_capture_y_offset, AllPlanes)) {
        std::vector < unsigned char > full_rgb_data(local_capture_width_actual * local_capture_height_actual * 3);
        unsigned char * shm_data_ptr = (unsigned char * ) shm_image -> data;
        int bytes_per_pixel_shm = shm_image -> bits_per_pixel / 8;
        int bytes_per_line_shm = shm_image -> bytes_per_line;

        for (int y = 0; y < local_capture_height_actual; ++y) {
          for (int x = 0; x < local_capture_width_actual; ++x) {
            unsigned char * pixel_ptr =
              shm_data_ptr + (y * bytes_per_line_shm) + (x * bytes_per_pixel_shm);
            full_rgb_data[(y * local_capture_width_actual + x) * 3 + 0] = pixel_ptr[2]; // R
            full_rgb_data[(y * local_capture_width_actual + x) * 3 + 1] = pixel_ptr[1]; // G
            full_rgb_data[(y * local_capture_width_actual + x) * 3 + 2] = pixel_ptr[0]; // B
          }
        }

        std::vector < std::future < StripeEncodeResult >> futures;
        std::vector < std::thread > threads;
        
        int N_processing_stripes = num_stripes;
        if (local_capture_height_actual <= 0) {
            N_processing_stripes = 0;
        } else {
            if (local_current_output_mode == OutputMode::H264) {
                const int MIN_H264_STRIPE_HEIGHT_PX = 64;
                if (local_capture_height_actual < MIN_H264_STRIPE_HEIGHT_PX) {
                    N_processing_stripes = 1;
                } else {
                    int max_stripes_by_min_height = local_capture_height_actual / MIN_H264_STRIPE_HEIGHT_PX;
                    N_processing_stripes = std::min(num_stripes, max_stripes_by_min_height);
                    if (N_processing_stripes == 0) N_processing_stripes = 1; // Ensure at least one if height allows
                }
            } else { // JPEG
                N_processing_stripes = std::min(num_stripes, local_capture_height_actual);
                if (N_processing_stripes == 0 && local_capture_height_actual > 0) N_processing_stripes = 1;
            }
        }
        if (N_processing_stripes == 0 && local_capture_height_actual > 0) { // Final safety
             N_processing_stripes = 1;
        }


        int h264_base_even = 0;
        int h264_num_stripes_with_extra_pair = 0;
        int current_y_offset = 0; 

        if (local_current_output_mode == OutputMode::H264 && N_processing_stripes > 0) {
          int H = local_capture_height_actual;
          int N = N_processing_stripes;

          int base_h = H / N;
          h264_base_even = (base_h > 0) ? (base_h - (base_h % 2)) : 0;
          if (h264_base_even <= 0 && H >= 2) {
            h264_base_even = 2;
          }

          if (h264_base_even > 0) {
            int H_base_covered = h264_base_even * N;
            int H_remaining = H - H_base_covered;
            if (H_remaining < 0) H_remaining = 0;

            h264_num_stripes_with_extra_pair = H_remaining / 2;
            h264_num_stripes_with_extra_pair = std::min(h264_num_stripes_with_extra_pair, N);
          } else {
            std::cerr << "Warning: Could not calculate a positive even base height for H.264 stripes (H=" <<
              H << ", N=" << N << "). Stripe heights may be zero." << std::endl;
            h264_num_stripes_with_extra_pair = 0; 
          }
        }
        bool any_stripe_encoded_this_frame = false;

        for (int i = 0; i < N_processing_stripes; ++i) {
          int start_y = current_y_offset; 
          int current_stripe_height = 0;

          if (local_current_output_mode == OutputMode::H264) {
            // Check if a valid base height was calculated
            if (h264_base_even > 0) {
              if (i < h264_num_stripes_with_extra_pair) {
                current_stripe_height = h264_base_even + 2;
              } else {
                current_stripe_height = h264_base_even;
              }
            } else { // h264_base_even is 0 or less
                if (N_processing_stripes == 1) { // Special case: single stripe for entire (small) height
                    current_stripe_height = local_capture_height_actual;
                    if (current_stripe_height % 2 != 0 && current_stripe_height > 0) { // Ensure even for H264
                        current_stripe_height--;
                    }
                } else {
                    current_stripe_height = 0; 
                }
            }
          } else { // JPEG
            if (N_processing_stripes > 0) {
                int base_stripe_height_jpeg = local_capture_height_actual / N_processing_stripes;
                int remainder_height_jpeg = local_capture_height_actual % N_processing_stripes;
                start_y = i * base_stripe_height_jpeg + std::min(i, remainder_height_jpeg);
                current_stripe_height = base_stripe_height_jpeg + (i < remainder_height_jpeg ? 1 : 0);
            } else {
                current_stripe_height = 0;
            }
          }

          if (local_current_output_mode == OutputMode::H264) { // For H264, current_y_offset is cumulative
            current_y_offset = start_y + current_stripe_height;
          } // For JPEG, start_y is calculated independently, current_y_offset is not used for its start_y.

          // Skip processing immediately if stripe height is zero or less
          if (current_stripe_height <= 0) {
            continue;
          }

          // Calculate height to actually copy
          int effective_height_for_copy = std::min(current_stripe_height, local_capture_height_actual - start_y);
          if (effective_height_for_copy <= 0) {
            continue;
          }

          std::vector < unsigned char > stripe_rgb_data_for_hash_and_h264(
            local_capture_width_actual * current_stripe_height * 3);
          int row_stride_rgb = local_capture_width_actual * 3;

          for (int y_offset = 0; y_offset < current_stripe_height; ++y_offset) {
            int global_y = start_y + y_offset;
            if (global_y < local_capture_height_actual && (global_y * row_stride_rgb + row_stride_rgb) <= full_rgb_data.size()) {
              std::memcpy( & stripe_rgb_data_for_hash_and_h264[y_offset * row_stride_rgb], &
                full_rgb_data[global_y * row_stride_rgb], row_stride_rgb);
            } else {
              std::memset( & stripe_rgb_data_for_hash_and_h264[y_offset * row_stride_rgb], 0, row_stride_rgb);
            }
          }

          uint64_t current_hash = calculate_stripe_hash(stripe_rgb_data_for_hash_and_h264);
          bool send_this_stripe = false;

          if (current_hash == previous_hashes[i]) {
            no_motion_frame_counts[i]++;
            if (no_motion_frame_counts[i] >= local_current_paint_over_trigger_frames &&
              !paint_over_sent[i] && !damage_blocked[i]) {

              if (local_current_output_mode == OutputMode::JPEG) {
                if (local_current_use_paint_over_quality) {
                  send_this_stripe = true;
                  last_paint_over_hashes[i] = current_hash;
                }
              } else { // H264 mode
                send_this_stripe = true;
                {
                  std::lock_guard < std::mutex > lock(g_h264_minimal_store.store_mutex);
                  if (i < static_cast < int > (g_h264_minimal_store.force_idr_flags.size())) {
                    g_h264_minimal_store.force_idr_flags[i] = true;
                  }
                }
              }
              if (send_this_stripe) paint_over_sent[i] = true;
            }
          } else {
            no_motion_frame_counts[i] = 0;
            paint_over_sent[i] = false;
            send_this_stripe = true;
            previous_hashes[i] = current_hash;

            damage_block_counts[i]++;
            if (damage_block_counts[i] >= local_current_damage_block_threshold) {
              damage_blocked[i] = true;
              damage_block_timer[i] = local_current_damage_block_duration;
            }
          }

          if (send_this_stripe) {
            any_stripe_encoded_this_frame = true;
            total_stripes_encoded_this_interval++;
            if (local_current_output_mode == OutputMode::JPEG) {
              int quality_to_use = (current_hash == previous_hashes[i] && local_current_use_paint_over_quality) ?
                local_current_paint_over_jpeg_quality : current_jpeg_qualities[i];
              if (current_hash != previous_hashes[i]) {
                current_jpeg_qualities[i] = std::max(current_jpeg_qualities[i] - 1, local_current_jpeg_quality);
              }

              std::packaged_task < StripeEncodeResult(int, int, int, int, int, int,
                  const unsigned char * , int, int, int) >
                task(encode_stripe_jpeg);
              futures.push_back(task.get_future());
              threads.push_back(std::thread(
                std::move(task), i, start_y, current_stripe_height,
                DisplayWidth(display, screen),
                local_capture_height_actual,
                local_capture_width_actual,
                full_rgb_data.data(),
                static_cast < int > (full_rgb_data.size()),
                quality_to_use,
                frame_counter));
            } else { // H264 Mode
              std::packaged_task < StripeEncodeResult(int, int, int, int,
                  const unsigned char * , int, int) >
                task(encode_stripe_h264);
              futures.push_back(task.get_future());
              threads.push_back(std::thread(
                [task_moved = std::move(task), i, start_y, current_stripe_height,
                  local_capture_width_actual,
                  data_copy = stripe_rgb_data_for_hash_and_h264, fc = frame_counter,
                  crf_val = local_current_h264_crf
                ]
                () mutable {
                  task_moved(i, start_y, current_stripe_height,
                    local_capture_width_actual,
                    data_copy.data(), fc, crf_val);
                }));
            }
          }

          if (damage_block_timer[i] > 0) {
            damage_block_timer[i]--;
            if (damage_block_timer[i] == 0) {
              damage_blocked[i] = false;
              damage_block_counts[i] = 0;
              if (local_current_output_mode == OutputMode::JPEG) {
                current_jpeg_qualities[i] = local_current_use_paint_over_quality ? local_current_paint_over_jpeg_quality : local_current_jpeg_quality;
              }
            }
          }
        }

        std::vector < StripeEncodeResult > stripe_results;
        stripe_results.reserve(futures.size());
        for (auto & future: futures) {
          stripe_results.push_back(future.get());
        }
        futures.clear();

        for (StripeEncodeResult & result: stripe_results) {
          if (stripe_callback != nullptr && result.data != nullptr) {
            stripe_callback( & result, user_data);
          } else if (result.data == nullptr && result.type != StripeDataType::UNKNOWN) {
            free_stripe_encode_result_data( & result);
          }
        }

        for (auto & thread: threads) {
          if (thread.joinable()) {
            thread.join();
          }
        }
        threads.clear();

        frame_counter++;
        if (any_stripe_encoded_this_frame) {
          encoded_frame_count++;
        }
        frame_count_loop++;

        auto current_time_for_fps_log = std::chrono::high_resolution_clock::now();
        auto elapsed_time_for_fps_log =
          std::chrono::duration_cast < std::chrono::seconds > (current_time_for_fps_log - start_time_loop);

        if (elapsed_time_for_fps_log.count() >= 1) {
          frame_count_loop = 0;
          start_time_loop = std::chrono::high_resolution_clock::now();
        }

        auto current_output_time_log = std::chrono::high_resolution_clock::now();
        auto output_elapsed_time_log =
          std::chrono::duration_cast < std::chrono::seconds > (current_output_time_log - last_output_time);
        if (output_elapsed_time_log.count() >= 1) {
          double actual_fps_val = (encoded_frame_count > 0 && output_elapsed_time_log.count() > 0) ?
            static_cast < double > (encoded_frame_count) / output_elapsed_time_log.count() : 0.0;

          double total_stripes_per_second_val = (total_stripes_encoded_this_interval > 0 && output_elapsed_time_log.count() > 0) ?
            static_cast < double > (total_stripes_encoded_this_interval) / output_elapsed_time_log.count() : 0.0;

          std::cout << "Resolution: " << local_capture_width_actual << "x" <<
            local_capture_height_actual << " Mode: " << (local_current_output_mode == OutputMode::JPEG ? "JPEG" : "H264") <<
            " Stripes: " << N_processing_stripes <<
            " CRF: " << local_current_h264_crf <<
            " Frames/s: " << std::fixed << std::setprecision(2) << actual_fps_val <<
            " Stripes/s: " << std::fixed << std::setprecision(2) << total_stripes_per_second_val <<
            std::endl;

          encoded_frame_count = 0;
          total_stripes_encoded_this_interval = 0;
          last_output_time = std::chrono::high_resolution_clock::now();
        }

      } else {
        std::cerr << "Failed to capture XImage using XShmGetImage" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }

    XShmDetach(display, & shminfo);
    shmdt(shminfo.shmaddr);
    shmctl(shmid, IPC_RMID, 0);
    if (shm_image) XDestroyImage(shm_image);
    XCloseDisplay(display);
    std::cout << "Capture loop stopped." << std::endl;
  }
};

extern "C" {

  typedef void * ScreenCaptureModuleHandle;

  ScreenCaptureModuleHandle create_screen_capture_module() {
    return static_cast < ScreenCaptureModuleHandle > (new ScreenCaptureModule());
  }

  void destroy_screen_capture_module(ScreenCaptureModuleHandle module_handle) {
    if (module_handle) {
      delete static_cast < ScreenCaptureModule * > (module_handle);
    }
  }

  void start_screen_capture(ScreenCaptureModuleHandle module_handle, CaptureSettings settings,
    StripeCallback callback, void * user_data) {
    if (module_handle) {
      ScreenCaptureModule * module = static_cast < ScreenCaptureModule * > (module_handle);
      module -> modify_settings(settings);

      std::lock_guard < std::mutex > lock(module -> settings_mutex);
      module -> stripe_callback = callback;
      module -> user_data = user_data;
      module -> start_capture();
    }
  }

  void stop_screen_capture(ScreenCaptureModuleHandle module_handle) {
    if (module_handle) {
      static_cast < ScreenCaptureModule * > (module_handle) -> stop_capture();
    }
  }

  void modify_screen_capture(ScreenCaptureModuleHandle module_handle, CaptureSettings settings) {
    if (module_handle) {
      static_cast < ScreenCaptureModule * > (module_handle) -> modify_settings(settings);
    }
  }

  CaptureSettings get_screen_capture_settings(ScreenCaptureModuleHandle module_handle) {
    if (module_handle) {
      return static_cast < ScreenCaptureModule * > (module_handle) -> get_current_settings();
    } else {
      return CaptureSettings {};
    }
  }

  void free_stripe_encode_result_data(StripeEncodeResult * result) {
    if (result && result -> data) {
      delete[] result -> data;
      result -> data = nullptr;
      result -> size = 0;
      result -> type = StripeDataType::UNKNOWN;
    }
  }

} // extern "C"

// --- Encoder Implementations ---
StripeEncodeResult encode_stripe_jpeg(int thread_id, int stripe_y_start, int stripe_height,
  int width, int height, int capture_width_actual,
  const unsigned char * rgb_data, int rgb_data_len,
    int jpeg_quality, int frame_counter) {
  StripeEncodeResult result;
  result.type = StripeDataType::JPEG;
  result.stripe_y_start = stripe_y_start;
  result.stripe_height = stripe_height;
  result.frame_id = frame_counter;

  jpeg_compress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error( & jerr);
  jpeg_create_compress( & cinfo);

  cinfo.image_width = capture_width_actual;
  cinfo.image_height = stripe_height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults( & cinfo);
  jpeg_set_quality( & cinfo, jpeg_quality, TRUE);

  unsigned char * jpeg_buffer = nullptr;
  unsigned long jpeg_size_temp = 0;
  jpeg_mem_dest( & cinfo, & jpeg_buffer, & jpeg_size_temp);
  jpeg_start_compress( & cinfo, TRUE);

  JSAMPROW row_pointer[1];
  int row_stride = capture_width_actual * 3;
  std::vector < unsigned char > stripe_rgb_data_internal(capture_width_actual * stripe_height * 3);

  for (int y_offset = 0; y_offset < stripe_height; ++y_offset) {
    int global_y = stripe_y_start + y_offset;
    if (global_y < height) {
      if (rgb_data != nullptr) {
        std::memcpy( & stripe_rgb_data_internal[y_offset * row_stride], & rgb_data[global_y * row_stride],
          row_stride);
      } else {
        std::memset( & stripe_rgb_data_internal[y_offset * row_stride], 0, row_stride);
      }
    } else {
      std::memset( & stripe_rgb_data_internal[y_offset * row_stride], 0, row_stride);
    }
    row_pointer[0] = & stripe_rgb_data_internal[y_offset * row_stride];
    jpeg_write_scanlines( & cinfo, row_pointer, 1);
  }

  jpeg_finish_compress( & cinfo);

  int padding_size = 4; // For frame_counter and stripe_y_start
  unsigned char * padded_jpeg_buffer = new unsigned char[jpeg_size_temp + padding_size];
  uint16_t frame_counter_net = htons(static_cast < uint16_t > (frame_counter % 65536));
  uint16_t stripe_y_start_net = htons(static_cast < uint16_t > (stripe_y_start));

  std::memcpy(padded_jpeg_buffer, & frame_counter_net, 2);
  std::memcpy(padded_jpeg_buffer + 2, & stripe_y_start_net, 2);
  std::memcpy(padded_jpeg_buffer + padding_size, jpeg_buffer, jpeg_size_temp);

  result.size = static_cast < int > (jpeg_size_temp) + padding_size;
  result.data = padded_jpeg_buffer;
  jpeg_destroy_compress( & cinfo);
  if (jpeg_buffer) {
    free(jpeg_buffer);
  }
  return result;
}

// H.264 encoder
StripeEncodeResult encode_stripe_h264(
  int thread_id,
  int stripe_y_start,
  int stripe_height,
  int capture_width_actual,
  const unsigned char * stripe_rgb24_data,
    int frame_counter,
    int current_crf_setting) {

  StripeEncodeResult result;
  result.type = StripeDataType::H264;
  result.stripe_y_start = stripe_y_start;
  result.stripe_height = stripe_height;
  result.frame_id = frame_counter;
  result.data = nullptr;
  result.size = 0;

  if (!stripe_rgb24_data) {
    std::cerr << "H264 T" << thread_id << ": Error - null rgb_data" << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }

  if (stripe_height <= 0 || capture_width_actual <= 0) {
    std::cerr << "H264 T" << thread_id << ": Invalid dimensions for encoding (" <<
      capture_width_actual << "x" << stripe_height << ")" << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }
  // Ensure width and height are even for I420
  x264_t * current_encoder = nullptr;
  x264_picture_t * current_pic_in_ptr = nullptr;
  {
    std::lock_guard < std::mutex > lock(g_h264_minimal_store.store_mutex);
    g_h264_minimal_store.ensure_size(thread_id);

    bool needs_reinit = false;
    if (!g_h264_minimal_store.initialized_flags[thread_id]) {
      needs_reinit = true;
    } else if (g_h264_minimal_store.initialized_widths[thread_id] != capture_width_actual ||
      g_h264_minimal_store.initialized_heights[thread_id] != stripe_height) {

      needs_reinit = true;

      if (g_h264_minimal_store.encoders[thread_id]) {
        x264_encoder_close(g_h264_minimal_store.encoders[thread_id]);
        g_h264_minimal_store.encoders[thread_id] = nullptr;
      }
      if (g_h264_minimal_store.pics_in_ptrs[thread_id]) {
        x264_picture_clean(g_h264_minimal_store.pics_in_ptrs[thread_id]);
        delete g_h264_minimal_store.pics_in_ptrs[thread_id];
        g_h264_minimal_store.pics_in_ptrs[thread_id] = nullptr;
      }
      g_h264_minimal_store.initialized_flags[thread_id] = false;
    }

    if (needs_reinit) {
      x264_param_t param;
      if (x264_param_default_preset( & param, "ultrafast", "zerolatency") < 0) {
        std::cerr << "H264 T" << thread_id << ": x264_param_default_preset FAILED." << std::endl;
        result.type = StripeDataType::UNKNOWN;
      } else {
        param.i_width = capture_width_actual;
        param.i_height = stripe_height;
        param.i_csp = X264_CSP_I420; // Explicitly set, though default preset might do it.
        param.i_fps_num = 60;
        param.i_fps_den = 1;
        param.i_keyint_max = 12000;
        param.rc.f_rf_constant = std::max(0, std::min(51, current_crf_setting)); // adjust for quality/bitrate.
        param.rc.i_rc_method = X264_RC_CRF;
        param.b_repeat_headers = 1; // Crucial: Embed SPS/PPS with IDR frames.
        param.b_annexb = 1; // NALUs are prefixed with 00 00 00 01 (or 00 00 01)
        param.i_sync_lookahead = 0;
        param.i_bframe = 0;
        param.i_threads = 0;

        if (x264_param_apply_profile( & param, "baseline") < 0) { // Baseline for broad compatibility
          std::cerr << "H264 T" << thread_id << ": Failed to apply baseline profile (non-fatal)." << std::endl;
        }

        g_h264_minimal_store.encoders[thread_id] = x264_encoder_open( & param);
        if (!g_h264_minimal_store.encoders[thread_id]) {
          std::cerr << "H264 T" << thread_id << ": x264_encoder_open FAILED." << std::endl;
          result.type = StripeDataType::UNKNOWN;
        } else {
          g_h264_minimal_store.pics_in_ptrs[thread_id] = new(std::nothrow) x264_picture_t();
          if (!g_h264_minimal_store.pics_in_ptrs[thread_id]) {
            std::cerr << "H264 T" << thread_id << ": FAILED to new x264_picture_t." << std::endl;
            x264_encoder_close(g_h264_minimal_store.encoders[thread_id]);
            g_h264_minimal_store.encoders[thread_id] = nullptr;
            result.type = StripeDataType::UNKNOWN;
          } else {
            x264_picture_init(g_h264_minimal_store.pics_in_ptrs[thread_id]);
            if (x264_picture_alloc(g_h264_minimal_store.pics_in_ptrs[thread_id], param.i_csp, param.i_width, param.i_height) < 0) {
              std::cerr << "H264 T" << thread_id << ": x264_picture_alloc FAILED." << std::endl;
              delete g_h264_minimal_store.pics_in_ptrs[thread_id];
              g_h264_minimal_store.pics_in_ptrs[thread_id] = nullptr;
              x264_encoder_close(g_h264_minimal_store.encoders[thread_id]);
              g_h264_minimal_store.encoders[thread_id] = nullptr;
              result.type = StripeDataType::UNKNOWN;
            } else {
              g_h264_minimal_store.initialized_flags[thread_id] = true;
              g_h264_minimal_store.initialized_widths[thread_id] = param.i_width;
              g_h264_minimal_store.initialized_heights[thread_id] = param.i_height;
              g_h264_minimal_store.force_idr_flags[thread_id] = true; // <<< SET FLAG: Force IDR for the next frame
            }
          }
        }
      }
    }

    if (g_h264_minimal_store.initialized_flags[thread_id]) {
      current_encoder = g_h264_minimal_store.encoders[thread_id];
      current_pic_in_ptr = g_h264_minimal_store.pics_in_ptrs[thread_id];
    }
  }

  if (!current_encoder || !current_pic_in_ptr) {
    if (result.type != StripeDataType::UNKNOWN) {
      std::cerr << "H264 T" << thread_id << ": Encoder/Picture not available post-init. Skipping frame." << std::endl;
    }
    result.type = StripeDataType::UNKNOWN;
    return result;
  }

  if (!current_pic_in_ptr -> img.plane[0] || !current_pic_in_ptr -> img.plane[1] || !current_pic_in_ptr -> img.plane[2]) {
    std::cerr << "H264 T" << thread_id << ": Input picture planes are NULL after successful alloc. This is a bug. Skipping frame." << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }

  int src_stride_rgb24 = capture_width_actual * 3;
  // libyuv::RAWToI420 expects B,G,R order for the "RAW" format.
  // x264_picture_t.img.plane order for X264_CSP_I420: [0]=Y, [1]=U (Cb), [2]=V (Cr)
  int conversion_status = libyuv::RAWToI420(
    stripe_rgb24_data, src_stride_rgb24, // BGR input
    current_pic_in_ptr -> img.plane[0], current_pic_in_ptr -> img.i_stride[0], // Y
    current_pic_in_ptr -> img.plane[1], current_pic_in_ptr -> img.i_stride[1], // U (Cb)
    current_pic_in_ptr -> img.plane[2], current_pic_in_ptr -> img.i_stride[2], // V (Cr)
    capture_width_actual, stripe_height);

  if (conversion_status != 0) {
    std::cerr << "H264 T" << thread_id << ": libyuv::RAWToI420 (BGR to I420) FAILED code " << conversion_status << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }

  current_pic_in_ptr -> i_pts = static_cast < int64_t > (frame_counter);

  bool is_forcing_idr_this_call = false;
  {
    std::lock_guard < std::mutex > lock(g_h264_minimal_store.store_mutex);
    if (g_h264_minimal_store.initialized_flags[thread_id] &&
      thread_id < static_cast < int > (g_h264_minimal_store.force_idr_flags.size()) && // Bounds check for safety
      g_h264_minimal_store.force_idr_flags[thread_id]) {
      is_forcing_idr_this_call = true;
    }
  }

  if (is_forcing_idr_this_call) {
    current_pic_in_ptr -> i_type = X264_TYPE_IDR; // Request an IDR frame
  } else {
    current_pic_in_ptr -> i_type = X264_TYPE_AUTO; // Let x264 decide
  }

  x264_nal_t * nals = nullptr;
  int i_nals = 0;
  x264_picture_t pic_out;
  x264_picture_init( & pic_out);

  int frame_size = x264_encoder_encode(current_encoder, & nals, & i_nals, current_pic_in_ptr, & pic_out);

  if (frame_size < 0) {
    std::cerr << "H264 T" << thread_id << ": x264_encoder_encode FAILED error " << frame_size << std::endl;
    result.type = StripeDataType::UNKNOWN;
    return result;
  }

  if (frame_size > 0) {
    // If we forced an IDR, and it was successful, clear the flag.
    if (is_forcing_idr_this_call) {
      if (pic_out.b_keyframe && pic_out.i_type == X264_TYPE_IDR) {
        std::lock_guard < std::mutex > lock(g_h264_minimal_store.store_mutex);
        if (thread_id < static_cast < int > (g_h264_minimal_store.force_idr_flags.size())) { // Bounds check
          g_h264_minimal_store.force_idr_flags[thread_id] = false;
        }
      }
    }

    const unsigned char DATA_TYPE_H264_STRIPED = 0x04;
    unsigned char current_frame_type_for_header = 0x00; // Default: P-frame / non-IDR

    // For the header, 0x01 means keyframe with SPS/PPS (i.e., an IDR frame because b_repeat_headers=1)
    if (pic_out.i_type == X264_TYPE_IDR) {
      current_frame_type_for_header = 0x01; // IDR Keyframe
    }

    // Header: dataType(1) + frameType(1) + frameID(2) + stripeYStart(2) + stripeWidth(2) + stripeHeight(2)
    int full_header_size = 10;
    int total_output_size = frame_size + full_header_size;

    result.data = new(std::nothrow) unsigned char[total_output_size];
    if (!result.data) {
      std::cerr << "H264 T" << thread_id << ": FAILED to new result.data for H.264 output." << std::endl;
      result.type = StripeDataType::UNKNOWN;
      return result;
    }

    result.data[0] = DATA_TYPE_H264_STRIPED;
    result.data[1] = current_frame_type_for_header;

    uint16_t frame_id_net = htons(static_cast < uint16_t > (result.frame_id % 65536));
    uint16_t stripe_y_start_net = htons(static_cast < uint16_t > (result.stripe_y_start));
    uint16_t stripe_width_net = htons(static_cast < uint16_t > (capture_width_actual));
    uint16_t stripe_height_net = htons(static_cast < uint16_t > (result.stripe_height));

    std::memcpy(result.data + 2, & frame_id_net, 2);
    std::memcpy(result.data + 4, & stripe_y_start_net, 2);
    std::memcpy(result.data + 6, & stripe_width_net, 2);
    std::memcpy(result.data + 8, & stripe_height_net, 2);

    unsigned char * p_payload_destination = result.data + full_header_size;
    unsigned char * p_current_nal_copy_target = p_payload_destination;

    for (int k = 0; k < i_nals; ++k) {
      // nals[k].p_payload already includes the Annex B start code (00 00 01 or 00 00 00 01)
      // because param.b_annexb = 1 was set.
      if ((p_current_nal_copy_target - p_payload_destination + nals[k].i_payload) <= frame_size) {
        std::memcpy(p_current_nal_copy_target, nals[k].p_payload, nals[k].i_payload);
        p_current_nal_copy_target += nals[k].i_payload;
      } else {
        std::cerr << "H264 T" << thread_id << ": Buffer overflow during NAL copy." << std::endl;
        delete[] result.data;
        result.data = nullptr;
        result.size = 0;
        result.type = StripeDataType::UNKNOWN;
        return result;
      }
    }
    result.size = total_output_size;
  } else {
    // No output NALs from encoder for this frame (e.g. due to B-frame delay, but B-frames are off)
    // Or if forcing IDR and it couldn't produce one immediately.
    result.data = nullptr;
    result.size = 0;
  }

  return result;
}

uint64_t calculate_stripe_hash(const std::vector < unsigned char > & rgb_data) {
  if (rgb_data.empty()) return 0;
  return XXH3_64bits(rgb_data.data(), rgb_data.size());
}
