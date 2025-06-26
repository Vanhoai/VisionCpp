function(make_library PROJECT_NAME INCLUDE_DIR SRC_DIR EXPORT_NAME DEFINES)

    # Collect source files
    file(GLOB_RECURSE HEADERS "${INCLUDE_DIR}/*.hpp")
    file(GLOB_RECURSE SOURCES "${SRC_DIR}/*.cpp")

    # Create the library
    add_library(${PROJECT_NAME} ${HEADERS} ${SOURCES})

    # Set include directories
    target_include_directories(${PROJECT_NAME}
            PRIVATE ${SRC_DIR}
            PUBLIC ${INCLUDE_DIR}
    )

    # Set compiler features
    target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

    # Add definitions
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${DEFINES}=1)

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
                EXPORT ${EXPORT_NAME}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )

        install(DIRECTORY ${INCLUDE_DIR}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    endif ()

    message(STATUS "Library ${PROJECT_NAME} created with sources: ${SOURCES} and headers: ${HEADERS}")
    message(STATUS "Include directories: ${INCLUDE_DIR}")
    message(STATUS "Export name: ${EXPORT_NAME}")
    message(STATUS "Definitions: ${DEFINES}")

endfunction()
