cmake_minimum_required(VERSION 3.18)
project(benchmark VERSION 1.0.0)

# set cpp standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME}
        benchmark.hpp
        benchmark.cpp
)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        core
        ${OpenCV_LIBS}
        ${OPENMP_LIBRARIES}
        Threads::Threads

        # For OS X
        "-framework CoreGraphics"
        "-framework Foundation"
)

# Include directories
target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/include
)

# Set compiler features
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Add missing header for std::function
target_compile_definitions(${PROJECT_NAME} PRIVATE)
