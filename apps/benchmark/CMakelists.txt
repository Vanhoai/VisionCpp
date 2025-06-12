cmake_minimum_required(VERSION 3.18)
project(benchmark VERSION 1.0.0)

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
