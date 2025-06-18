cmake_minimum_required(VERSION 3.30)
project(preprocessing LANGUAGES C CXX)

# Collect source files
file(GLOB_RECURSE PREPROCESSING_HEADERS "include/preprocessing/*.hpp")
file(GLOB_RECURSE PREPROCESSING_SOURCES "src/*.cpp")

# Create the preprocessing library
add_library(${PROJECT_NAME} ${PREPROCESSING_HEADERS} ${PREPROCESSING_SOURCES})

# Set include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link dependencies for use in the preprocessing library
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

# Set compiler features
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)


# Add target definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE PREPROCESSING_EXPORTS=1)

# Set properties
set_target_properties(${PROJECT_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

# Install rules (optional)
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(GNUInstallDirs)
    install(TARGETS ${PROJECT_NAME}
            EXPORT PREPROCESSINGTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(DIRECTORY include/DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif ()
