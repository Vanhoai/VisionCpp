# Set default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Common compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# GCC/Clang specific flags
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

    # Vectorization and optimization flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -ffast-math -funroll-loops")

    # Additional optimization flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -funroll-loops")
    message(STATUS "GCC/Clang compiler detected: ${CMAKE_CXX_COMPILER_ID}")
endif ()

# Optional: define Eigen behavior (optional and use-case specific)
add_definitions(-DEIGEN_NO_DEBUG)              # Disable Eigen assertions
add_definitions(-DEIGEN_DONT_PARALLELIZE=0)    # Let Eigen parallelize OR
# add_definitions(-DEIGEN_USE_THREADS=0)       # If you want to fully control threading (e.g., only OpenMP)

# Position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile commands for tools like clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
