cmake_minimum_required(VERSION 3.18)
project(study VERSION 1.0.0)

file(GLOB_RECURSE STUDY_HEADERS "include/study/*.hpp")
file(GLOB_RECURSE STUDY_SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} main.cpp ${STUDY_HEADERS} ${STUDY_SOURCES})

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
        processing

        ${OpenCV_LIBS}
        ${OPENMP_LIBRARIES}
        Threads::Threads
        Eigen3::Eigen

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
