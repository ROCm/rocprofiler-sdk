#
#
#
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "core")
add_subdirectory(common)
if(ROCPROFILER_BUILD_TOOLS)
    add_subdirectory(output)
endif()
add_subdirectory(rocprofiler-sdk)
add_subdirectory(rocprofiler-sdk-roctx)

if(ROCPROFILER_BUILD_TOOLS)
    set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "tools")
    add_subdirectory(rocprofiler-sdk-codeobj)
    add_subdirectory(rocprofiler-sdk-att)
    add_subdirectory(rocprofiler-sdk-tool)
endif()

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
