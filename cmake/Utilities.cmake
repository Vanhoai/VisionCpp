# Function to create a library target with common settings
function(create_library target_name)
    cmake_parse_arguments(LIB
            "SHARED;STATIC;OBJECT"
            ""
            "SOURCES;HEADERS;DEPENDENCIES;INCLUDE_DIRS"
            ${ARGN}
    )

    # Determine library type
    if (LIB_SHARED)
        set(lib_type SHARED)
    elseif (LIB_STATIC)
        set(lib_type STATIC)
    elseif (LIB_OBJECT)
        set(lib_type OBJECT)
    else ()
        set(lib_type "")
    endif ()

    # Create library
    add_library(${target_name} ${lib_type} ${LIB_SOURCES})

    # Set include directories
    if (LIB_INCLUDE_DIRS)
        target_include_directories(${target_name}
                PUBLIC ${LIB_INCLUDE_DIRS}
        )
    endif ()

    # Link dependencies
    if (LIB_DEPENDENCIES)
        target_link_libraries(${target_name} ${LIB_DEPENDENCIES})
    endif ()

    # Apply common compile options
    apply_common_compile_options(${target_name})

endfunction()