cmake_minimum_required(VERSION 3.30)
project(VisionNative
        VERSION 1.0.0
        DESCRIPTION "🚀 Advanced C++ Algorithms for Computer Vision & Image Processing"
        LANGUAGES C CXX
)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add cmake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(BUILD_TESTS "Build Tests" OFF)
option(ENABLE_CUDA "Enable CUDA Support" OFF)
option(ENABLE_OPENCL "Enable OpenCL Support" OFF)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

# Include utilities
include(cmake/Dependencies.cmake)
include(cmake/CompilerFlags.cmake)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(apps)

# Installation
include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Print configuration summary
message(STATUS "+-------------------- VISION CPP CONFIGURATION --------------------+")
message(STATUS "| Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "| C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "| OpenCV version: ${OpenCV_VERSION}")
message(STATUS "| CUDA support: ${ENABLE_CUDA}")
message(STATUS "| OpenCL support: ${ENABLE_OPENCL}")
message(STATUS "| Build tests: ${BUILD_TESTS}")
message(STATUS "+------------------------------------------------------------------+")
