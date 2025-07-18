# distutils: language = c++
# distutils: extra_compile_args = -std=c++17
# distutils: include_dirs = pixelflux/include

from cpython cimport PyObject, Py_INCREF, Py_DECREF

cdef extern from "screen_capture_module.cpp":
    cpdef enum OutputMode:
        JPEG = 0,
        H264 = 1

    cpdef enum WatermarkLocation:
        NONE = 0,
        TL = 1,
        TR = 2,
        BL = 3,
        BR = 4,
        MI = 5,
        AN = 6

    cpdef enum StripeDataType:
        UNKNOWN = 0,
        JPEG = 1,
        H264 = 2

    cdef cppclass CaptureSettingsStruct:
        CaptureSettingsStruct()  # Default constructor declaration
        int capture_width
        int capture_height
        int capture_x
        int capture_y
        double target_fps
        int jpeg_quality
        int paint_over_jpeg_quality
        bint use_paint_over_quality
        int paint_over_trigger_frames
        int damage_block_threshold
        int damage_block_duration
        OutputMode output_mode
        int h264_crf
        bint h264_fullcolor
        bint h264_fullframe
        bint capture_cursor
        const char* watermark_path
        WatermarkLocation watermark_location_enum
        int vaapi_render_node_index

    cdef cppclass StripeEncodeResultStruct:
        StripeDataType type
        int stripe_y_start
        int stripe_height
        int size
        unsigned char* data
        int frame_id

