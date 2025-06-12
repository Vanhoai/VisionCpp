# Find required dependencies
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# OpenCL - Optional if you want to use OpenCL features
if (ENABLE_OPENCL)
    find_package(OpenCL REQUIRED)
endif ()

# OpenMP for parallel processing
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(OPENMP_LIBRARIES OpenMP::OpenMP_CXX)
endif ()

# Cuda - Optional if you want to use CUDA features
if (ENABLE_CUDA)
    find_package(CUDA QUIET)
    if (CUDA_FOUND)
        enable_language(CUDA)
        message(STATUS "CUDA found: ${CUDA_VERSION}")
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    else ()
        message(WARNING "CUDA not found, disabling CUDA support")
        set(ENABLE_CUDA OFF)
    endif ()
endif ()