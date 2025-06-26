cmake_minimum_required(VERSION 3.30)
project(processing LANGUAGES C CXX)

make_library(
        ${PROJECT_NAME}
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        PROCESSINGTargets
        PROCESSING_EXPORT
)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        core
        ${OPENMP_LIBRARIES}
        Threads::Threads
        ${MATH_LIBRARIES}
        PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
)
