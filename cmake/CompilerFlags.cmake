if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# GCC/Clang flags
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

    # Enable OpenMP if available
    find_package(OpenMP QUIET)
    if (OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else ()
        message(STATUS "OpenMP not found, proceeding without it.")
    endif ()
endif ()

# Function to apply common compile options
function(apply_common_compile_options target)
    target_compile_features(${target} PUBLIC cxx_std_17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
                -Wall -Wextra -Wpedantic
                $<$<CONFIG:Release>:-O3 -march=native>
                $<$<CONFIG:Debug>:-g -O0>
        )
    endif ()
endfunction()