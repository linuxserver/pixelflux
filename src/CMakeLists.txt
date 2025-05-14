cmake_minimum_required(VERSION 3.14)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) 
project(screen_capture_module_cpp)
include(FetchContent)
# --- Common Args for FetchContent ---
set(FC_COMMON_CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON 
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
)

# --- libyuv (FetchContent - uses CMake) ---
FetchContent_Declare(
    libyuv_dep
    GIT_REPOSITORY https://chromium.googlesource.com/libyuv/libyuv
    GIT_TAG origin/main
    CMAKE_ARGS ${FC_COMMON_CMAKE_ARGS} -DBUILD_SHARED_LIBS=OFF -DTEST=OFF
    FETCHCONTENT_QUIET FALSE
)

# --- xxHash (FetchContent - uses CMake) ---
FetchContent_Declare(
    xxhash_dep
    GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
    GIT_TAG v0.8.3
    SOURCE_SUBDIR cmake_unofficial 
    CMAKE_ARGS ${FC_COMMON_CMAKE_ARGS} -DBUILD_SHARED_LIBS=OFF -DXXHASH_BUILD_XXHSUM=OFF
    FETCHCONTENT_QUIET FALSE
)
FetchContent_MakeAvailable(libyuv_dep xxhash_dep)

# --- Other (System Libraries) ---
find_package(PkgConfig REQUIRED)
pkg_check_modules(X264 REQUIRED x264)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(X11 REQUIRED) 
find_package(Threads REQUIRED)
find_package(JPEG REQUIRED) 

# --- Output ---
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
    ${X264_LIBRARIES}
    ${JPEG_LIBRARIES} 
)

# Output
set_target_properties(screen_capture_module PROPERTIES
    PREFIX "" 
    SUFFIX ".so" 
    OUTPUT_NAME "screen_capture_module" 
)
install(TARGETS screen_capture_module
    LIBRARY DESTINATION screen_capture 
    COMPONENT screen_capture_runtime 
)
