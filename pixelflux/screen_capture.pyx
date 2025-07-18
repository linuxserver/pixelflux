# distutils: language = c++
# distutils: extra_compile_args = -std=c++17
# distutils: include_dirs = pixelflux/include

from cpython cimport PyObject, Py_INCREF, Py_DECREF
from libc.stdlib cimport malloc, free
from cpython cimport PyCapsule_New, PyCapsule_GetPointer

cdef extern from "screen_capture_module.cpp":
    cdef enum class OutputMode:
        JPEG = 0,
        H264 = 1

    cdef enum class WatermarkLocation:
        NONE = 0,
        TL = 1,
        TR = 2,
        BL = 3,
        BR = 4,
        MI = 5,
        AN = 6

    cdef enum class StripeDataType:
        UNKNOWN = 0,
        JPEG = 1,
        H264 = 2
    
    cdef cppclass CaptureSettingsStruct:
        CaptureSettingsStruct()
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
    
    void* create_screen_capture_module()
    void destroy_screen_capture_module(void* module)
    void start_screen_capture(void* module, CaptureSettingsStruct& settings, void (*callback)(StripeEncodeResultStruct*, void*), void* user_data)
    void stop_screen_capture(void* module)
    void free_stripe_encode_result_data(StripeEncodeResultStruct* result)

# Python interface classes
cdef class CaptureSettings:
    cdef CaptureSettingsStruct _c_obj
    
    property capture_width:
        def __get__(self): return self._c_obj.capture_width
        def __set__(self, value): self._c_obj.capture_width = value
    
    property capture_height:
        def __get__(self): return self._c_obj.capture_height
        def __set__(self, value): self._c_obj.capture_height = value
    
    property capture_x:
        def __get__(self): return self._c_obj.capture_x
        def __set__(self, value): self._c_obj.capture_x = value
    
    property capture_y:
        def __get__(self): return self._c_obj.capture_y
        def __set__(self, value): self._c_obj.capture_y = value
    
    property target_fps:
        def __get__(self): return self._c_obj.target_fps
        def __set__(self, value): self._c_obj.target_fps = value
    
    property jpeg_quality:
        def __get__(self): return self._c_obj.jpeg_quality
        def __set__(self, value): self._c_obj.jpeg_quality = value
    
    property paint_over_jpeg_quality:
        def __get__(self): return self._c_obj.paint_over_jpeg_quality
        def __set__(self, value): self._c_obj.paint_over_jpeg_quality = value
    
    property use_paint_over_quality:
        def __get__(self): return self._c_obj.use_paint_over_quality
        def __set__(self, value): self._c_obj.use_paint_over_quality = value
    
    property paint_over_trigger_frames:
        def __get__(self): return self._c_obj.paint_over_trigger_frames
        def __set__(self, value): self._c_obj.paint_over_trigger_frames = value
    
    property damage_block_threshold:
        def __get__(self): return self._c_obj.damage_block_threshold
        def __set__(self, value): self._c_obj.damage_block_threshold = value
    
    property damage_block_duration:
        def __get__(self): return self._c_obj.damage_block_duration
        def __set__(self, value): self._c_obj.damage_block_duration = value
    
    property output_mode:
        def __get__(self): return self._c_obj.output_mode
        def __set__(self, value): self._c_obj.output_mode = value
    
    property h264_crf:
        def __get__(self): return self._c_obj.h264_crf
        def __set__(self, value): self._c_obj.h264_crf = value
    
    property h264_fullcolor:
        def __get__(self): return self._c_obj.h264_fullcolor
        def __set__(self, value): self._c_obj.h264_fullcolor = value
    
    property h264_fullframe:
        def __get__(self): return self._c_obj.h264_fullframe
        def __set__(self, value): self._c_obj.h264_fullframe = value
    
    property capture_cursor:
        def __get__(self): return self._c_obj.capture_cursor
        def __set__(self, value): self._c_obj.capture_cursor = value
    
    property watermark_path:
        def __get__(self):
            if self._c_obj.watermark_path == NULL:
                return None
            return <bytes>self._c_obj.watermark_path
        def __set__(self, bytes value):
            self._watermark_path_ref = value  # Keep reference
            self._c_obj.watermark_path = self._watermark_path_ref
    
    property watermark_location_enum:
        def __get__(self): return self._c_obj.watermark_location_enum
        def __set__(self, value): self._c_obj.watermark_location_enum = value
    
    property vaapi_render_node_index:
        def __get__(self): return self._c_obj.vaapi_render_node_index
        def __set__(self, value): self._c_obj.vaapi_render_node_index = value
    
    cdef object _watermark_path_ref  # To keep bytes reference

cdef class StripeEncodeResult:
    cdef StripeEncodeResultStruct* _ptr
    cdef unsigned char[:] _data_view
    
    def __cinit__(self):
        self._ptr = NULL
    
    def __dealloc__(self):
        if self._ptr != NULL:
            free_stripe_encode_result_data(self._ptr)
            free(self._ptr)
    
    property type:
        def __get__(self):
            if self._ptr == NULL: return 0
            return self._ptr.type
        def __set__(self, value):
            if self._ptr != NULL: self._ptr.type = value
    
    property stripe_y_start:
        def __get__(self):
            if self._ptr == NULL: return 0
            return self._ptr.stripe_y_start
        def __set__(self, value):
            if self._ptr != NULL: self._ptr.stripe_y_start = value
    
    property stripe_height:
        def __get__(self):
            if self._ptr == NULL: return 0
            return self._ptr.stripe_height
        def __set__(self, value):
            if self._ptr != NULL: self._ptr.stripe_height = value
    
    property size:
        def __get__(self):
            if self._ptr == NULL: return 0
            return self._ptr.size
        def __set__(self, value):
            if self._ptr != NULL: self._ptr.size = value
    
    property data:
        def __get__(self):
            if self._ptr == NULL or self._ptr.size == 0 or self._ptr.data == NULL:
                return None
            self._data_view = <unsigned char[:self._ptr.size]> self._ptr.data
            return self._data_view
    
    property frame_id:
        def __get__(self):
            if self._ptr == NULL: return 0
            return self._ptr.frame_id
        def __set__(self, value):
            if self._ptr != NULL: self._ptr.frame_id = value

# Callback handling
cdef void _stripe_callback(StripeEncodeResultStruct* c_result, void* user_data) noexcept with gil:
    cdef object capsule = <object>user_data
    cdef void* tuple_ptr = PyCapsule_GetPointer(capsule, NULL)
    if tuple_ptr == NULL:
        return
    
    cdef object callback_tuple = <object>tuple_ptr
    cdef object py_callback, py_user_data
    try:
        py_callback, py_user_data = callback_tuple
    except (TypeError, ValueError):
        return
    
    # Create Python object from C result
    # Create Python wrapper with ownership
    cdef StripeEncodeResult py_result = StripeEncodeResult()
    py_result._ptr = <StripeEncodeResultStruct*>malloc(sizeof(StripeEncodeResultStruct))
    if py_result._ptr == NULL:
        free_stripe_encode_result_data(c_result)
        return
    
    # Manually copy each field since copy assignment is deleted
    py_result._ptr.type = c_result.type
    py_result._ptr.stripe_y_start = c_result.stripe_y_start
    py_result._ptr.stripe_height = c_result.stripe_height
    py_result._ptr.size = c_result.size
    py_result._ptr.data = c_result.data
    py_result._ptr.frame_id = c_result.frame_id
    
    try:
        (<object>py_callback)(py_result)
    finally:
        # Prevent double-free - Python object now owns memory
        c_result.data = NULL
        c_result.size = 0

# ScreenCapture class
cdef class ScreenCapture:
    cdef void* _module
    cdef object _python_stripe_callback
    cdef object _c_callback_capsule
    cdef object _py_capsule
    
    def __cinit__(self):
        self._module = create_screen_capture_module()
        if not self._module:
            raise Exception("Failed to create screen capture module")
        self._python_stripe_callback = None
        self._c_callback_capsule = None
        self._py_capsule = None
    
    def __dealloc__(self):
        if self._module:
            self.stop_capture()
            destroy_screen_capture_module(self._module)
    
    def start_capture(self, CaptureSettings settings, stripe_callback, user_data=None):
        if self._python_stripe_callback is not None:
            raise ValueError("Capture already started")
        if not callable(stripe_callback):
            raise TypeError("stripe_callback must be callable")
        
        self._python_stripe_callback = stripe_callback
        self._c_callback_capsule = (stripe_callback, user_data)
        
        # Create capsule to pass to C callback
        cdef object py_capsule = PyCapsule_New(
            <void*>self._c_callback_capsule, NULL, NULL
        )
        cdef void* callback_ptr = <void*>py_capsule
        
        start_screen_capture(
            self._module,
            settings._c_obj,
            <void (*)(StripeEncodeResultStruct*, void*) noexcept>_stripe_callback,
            callback_ptr
        )
        
        # Keep reference to prevent garbage collection
        self._py_capsule = py_capsule
    
    def stop_capture(self):
        if self._python_stripe_callback is None:
            return
        
        stop_screen_capture(self._module)
        self._python_stripe_callback = None
        self._c_callback_capsule = None
