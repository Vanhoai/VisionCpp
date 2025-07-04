cmake_minimum_required(VERSION 3.18)
project(training VERSION 1.0.0)

file(GLOB_RECURSE TRAINING_HEADERS "include/training/*.hpp")
file(GLOB_RECURSE TRAINING_SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} main.cpp ${TRAINING_HEADERS} ${TRAINING_SOURCES})

# Set include directories
target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        core
        nn
        datasets

        ${OpenCV_LIBS}
        ${OPENMP_LIBRARIES}
        Threads::Threads
        Eigen3::Eigen
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

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE EIGEN_DONT_PARALLELIZE=0)
    message(STATUS "Enable multi-threading with OpenMP")
endif ()
