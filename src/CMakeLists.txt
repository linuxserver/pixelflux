cmake_minimum_required(VERSION 3.14)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(screen_capture_module_cpp)
include(FetchContent)

set(FC_COMMON_CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
)

FetchContent_Declare(
    libyuv_dep
    GIT_REPOSITORY https://chromium.googlesource.com/libyuv/libyuv
    GIT_TAG origin/main
    CMAKE_ARGS ${FC_COMMON_CMAKE_ARGS} -DBUILD_SHARED_LIBS=OFF -DTEST=OFF
    FETCHCONTENT_QUIET FALSE
)

FetchContent_Declare(
    xxhash_dep
    GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
    GIT_TAG v0.8.3
    SOURCE_SUBDIR cmake_unofficial
    CMAKE_ARGS ${FC_COMMON_CMAKE_ARGS} -DBUILD_SHARED_LIBS=OFF -DXXHASH_BUILD_XXHSUM=OFF
    FETCHCONTENT_QUIET FALSE
)
FetchContent_MakeAvailable(libyuv_dep xxhash_dep)

set(STB_IMAGE_HEADER_URL "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h")
set(STB_IMAGE_HEADER_PARENT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/stb_image_downloaded/include")
set(STB_IMAGE_HEADER_DOWNLOAD_DIR "${STB_IMAGE_HEADER_PARENT_INCLUDE_DIR}/")
set(STB_IMAGE_HEADER_FILE "${STB_IMAGE_HEADER_DOWNLOAD_DIR}/stb_image.h")
file(MAKE_DIRECTORY "${STB_IMAGE_HEADER_DOWNLOAD_DIR}")
if(NOT EXISTS "${STB_IMAGE_HEADER_FILE}")
    message(STATUS "Downloading stb_image.h from ${STB_IMAGE_HEADER_URL}")
    file(DOWNLOAD "${STB_IMAGE_HEADER_URL}" "${STB_IMAGE_HEADER_FILE}" SHOW_PROGRESS STATUS download_status LOG download_log TIMEOUT 60)
    list(GET download_status 0 download_error_code)
    list(GET download_status 1 download_error_string)
    if(NOT download_error_code EQUAL 0)
        message(FATAL_ERROR "Failed to download ${STB_IMAGE_HEADER_URL}: [${download_error_code}] ${download_error_string}\nFull Log:\n${download_log}")
    endif()
endif()
if(NOT TARGET stb_image_header)
    add_library(stb_image_header INTERFACE)
    target_include_directories(stb_image_header INTERFACE "${STB_IMAGE_HEADER_PARENT_INCLUDE_DIR}")
endif()

set(CUDA_STUB_HEADER_URL "https://raw.githubusercontent.com/GStreamer/gstreamer/188a9fa7a282df8b142f992194ceadc724744397/subprojects/gst-plugins-bad/sys/hip/stub/cuda.h")
set(CUDA_STUB_HEADER_PARENT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/cuda_stub_downloaded/include")
set(CUDA_STUB_HEADER_DOWNLOAD_DIR "${CUDA_STUB_HEADER_PARENT_INCLUDE_DIR}/")
set(CUDA_STUB_HEADER_FILE "${CUDA_STUB_HEADER_DOWNLOAD_DIR}/cuda.h")
file(MAKE_DIRECTORY "${CUDA_STUB_HEADER_DOWNLOAD_DIR}")
if(NOT EXISTS "${CUDA_STUB_HEADER_FILE}")
    message(STATUS "Downloading cuda.h stub from ${CUDA_STUB_HEADER_URL}")
    file(DOWNLOAD "${CUDA_STUB_HEADER_URL}" "${CUDA_STUB_HEADER_FILE}" SHOW_PROGRESS STATUS download_status LOG download_log TIMEOUT 60)
    list(GET download_status 0 download_error_code)
    list(GET download_status 1 download_error_string)
    if(NOT download_error_code EQUAL 0)
        message(FATAL_ERROR "Failed to download ${CUDA_STUB_HEADER_URL}: [${download_error_code}] ${download_error_string}\nFull Log:\n${download_log}")
    endif()
endif()
if(NOT TARGET cuda_stub_header)
    add_library(cuda_stub_header INTERFACE)
    target_include_directories(cuda_stub_header INTERFACE "${CUDA_STUB_HEADER_PARENT_INCLUDE_DIR}")
endif()

set(NVENC_HEADER_URL "https://raw.githubusercontent.com/FFmpeg/nv-codec-headers/refs/tags/n12.1.14.0/include/ffnvcodec/nvEncodeAPI.h")
set(NVENC_HEADER_PARENT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/nvenc_header_downloaded/include")
set(NVENC_HEADER_DOWNLOAD_DIR "${NVENC_HEADER_PARENT_INCLUDE_DIR}/")
set(NVENC_HEADER_FILE "${NVENC_HEADER_DOWNLOAD_DIR}/nvEncodeAPI.h")
file(MAKE_DIRECTORY "${NVENC_HEADER_DOWNLOAD_DIR}")
if(NOT EXISTS "${NVENC_HEADER_FILE}")
    message(STATUS "Downloading nvEncodeAPI.h from ${NVENC_HEADER_URL}")
    file(DOWNLOAD "${NVENC_HEADER_URL}" "${NVENC_HEADER_FILE}" SHOW_PROGRESS STATUS download_status LOG download_log TIMEOUT 60)
    list(GET download_status 0 download_error_code)
    list(GET download_status 1 download_error_string)
    if(NOT download_error_code EQUAL 0)
        message(FATAL_ERROR "Failed to download ${NVENC_HEADER_URL}: [${download_error_code}] ${download_error_string}\nFull Log:\n${download_log}")
    endif()
endif()
if(NOT TARGET nvenc_header)
    add_library(nvenc_header INTERFACE)
    target_include_directories(nvenc_header INTERFACE "${NVENC_HEADER_PARENT_INCLUDE_DIR}")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(X264 REQUIRED x264)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(X11 REQUIRED)
find_package(Threads REQUIRED)
find_package(JPEG REQUIRED)
find_library(XFIXES_LIB Xfixes REQUIRED)

add_library(screen_capture_module SHARED
    screen_capture_module.cpp
)
target_include_directories(screen_capture_module PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${X11_INCLUDE_DIR}
    "${libyuv_dep_SOURCE_DIR}/include"
    "${xxhash_dep_SOURCE_DIR}/lib"
    ${X264_INCLUDE_DIRS}
    ${JPEG_INCLUDE_DIRS}
)
target_link_libraries(screen_capture_module PRIVATE
    Python3::Python
    Threads::Threads
    ${X11_LIBRARIES}
    yuv
    xxhash
    stb_image_header
    cuda_stub_header
    nvenc_header
    ${X264_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${XFIXES_LIB}
    dl
)

set_target_properties(screen_capture_module PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    OUTPUT_NAME "screen_capture_module"
)
install(TARGETS screen_capture_module
    LIBRARY DESTINATION screen_capture
    COMPONENT screen_capture_runtime
)
