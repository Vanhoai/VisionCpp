cmake_minimum_required(VERSION 3.30)
project(src LANGUAGES C CXX)

add_subdirectory(core)
add_subdirectory(backends)
add_subdirectory(datasets)
add_subdirectory(nn)
add_subdirectory(vision)
add_subdirectory(processing)

add_library(src INTERFACE)

target_link_libraries(src
        INTERFACE
        core
        backends
        datasets
        nn
        vision
        processing
)

set_target_properties(src
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}"
)
