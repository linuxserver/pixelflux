cmake_minimum_required(VERSION 3.14)
project(screen_capture_module_cpp)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

set(NVENC_HEADER_URL "https://raw.githubusercontent.com/FFmpeg/nv-codec-headers/refs/tags/n12.1.14.0/include/ffnvcodec/nvEncodeAPI.h")
set(NVENC_HEADER_PARENT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/nv_codec_headers_downloaded/include")
set(NVENC_HEADER_DOWNLOAD_DIR "${NVENC_HEADER_PARENT_INCLUDE_DIR}/")
set(NVENC_HEADER_FILE "${NVENC_HEADER_DOWNLOAD_DIR}/nvEncodeAPI.h")

file(MAKE_DIRECTORY "${NVENC_HEADER_DOWNLOAD_DIR}")

if(NOT EXISTS "${NVENC_HEADER_FILE}")
    message(STATUS "Downloading nvEncodeAPI.h from ${NVENC_HEADER_URL}")
    file(DOWNLOAD "${NVENC_HEADER_URL}" "${NVENC_HEADER_FILE}"
         SHOW_PROGRESS
         STATUS download_status
         LOG download_log
         TIMEOUT 60
    )
    list(GET download_status 0 download_error_code)
    list(GET download_status 1 download_error_string)

    if(NOT download_error_code EQUAL 0)
        message(FATAL_ERROR "Failed to download ${NVENC_HEADER_URL}: [${download_error_code}] ${download_error_string}\nFull Log:\n${download_log}")
    elseif(NOT EXISTS "${NVENC_HEADER_FILE}")
        message(FATAL_ERROR "Download reported success, but file ${NVENC_HEADER_FILE} does not exist.\nFull Log:\n${download_log}")
    else()
        message(STATUS "nvEncodeAPI.h downloaded successfully to ${NVENC_HEADER_FILE}")
    endif()
else()
    message(STATUS "nvEncodeAPI.h already exists at ${NVENC_HEADER_FILE}. Skipping download.")
endif()

if(NOT TARGET nv_codec_headers)
    add_library(nv_codec_headers INTERFACE)
    target_include_directories(nv_codec_headers INTERFACE
        "${NVENC_HEADER_PARENT_INCLUDE_DIR}"
    )
    message(STATUS "nv_codec_headers INTERFACE target created for downloaded header.")
else()
    message(STATUS "nv_codec_headers INTERFACE target already exists.")
endif()

if(NOT EXISTS "${NVENC_HEADER_FILE}")
    message(FATAL_ERROR "Critical error: nvEncodeAPI.h is missing at ${NVENC_HEADER_FILE}. Cannot build.")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(X264 REQUIRED x264)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(X11 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
find_package(JPEG REQUIRED)

if(IS_DIRECTORY "/usr/lib/x86_64-linux-gnu/nvidia/current")
    link_directories("/usr/lib/x86_64-linux-gnu/nvidia/current")
    message(STATUS "Added /usr/lib/x86_64-linux-gnu/nvidia/current to linker search paths.")
elseif(IS_DIRECTORY "/usr/lib/nvidia-current")
    link_directories("/usr/lib/nvidia-current")
    message(STATUS "Added /usr/lib/nvidia-current to linker search paths.")
elseif(IS_DIRECTORY "/usr/lib/nvidia")
    link_directories("/usr/lib/nvidia")
    message(STATUS "Added /usr/lib/nvidia to linker search paths.")
elseif(IS_DIRECTORY "/usr/lib64/nvidia")
    link_directories("/usr/lib64/nvidia")
    message(STATUS "Added /usr/lib64/nvidia to linker search paths.")
else()
    message(WARNING "Could not find a common NVIDIA driver directory. Runtime loading of libnvidia-encode might require LD_LIBRARY_PATH or other system configuration.")
endif()

add_library(screen_capture_module SHARED
    screen_capture_module.cpp
)

target_include_directories(screen_capture_module PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${X11_INCLUDE_DIR}
    ${OpenGL_INCLUDE_DIRS}
    "${libyuv_dep_SOURCE_DIR}/include"
    "${xxhash_dep_SOURCE_DIR}/lib"
    ${X264_INCLUDE_DIRS}
    ${JPEG_INCLUDE_DIRS}
)

target_link_libraries(screen_capture_module PRIVATE
    Python3::Python
    Threads::Threads
    ${X11_LIBRARIES}
    OpenGL::GL
    OpenGL::GLX
    yuv
    xxhash
    ${X264_LIBRARIES}
    ${JPEG_LIBRARIES}
    nv_codec_headers
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
