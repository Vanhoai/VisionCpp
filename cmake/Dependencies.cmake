# Dependencies management for the project
include(FetchContent)

function(setup_dependencies)
    # Set FetchContent properties
    set(FETCHCONTENT OFF)

    # External dependencies
    FetchContent_Declare(
            spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG v1.12.0
    )

    if (BUILD_TESTS)
        # Install libraries for testing
    endif ()

    # Make dependencies available
    FetchContent_MakeAvailable(spdlog)

    if (BUILD_TESTS)
        # Make sure available for tests
    endif ()

    # OpenMP - Optional but recommended
    find_package(OpenMP QUIET)

    # Find system packages
    find_package(PkgConfig QUIET)
    find_package(OpenCV REQUIRED)
    find_package(Eigen3 REQUIRED)

    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "Eigen3 version: ${Eigen3_VERSION}")

endfunction()
