add_library(rocprofiler-sdk-tool-output-lttng STATIC rocprofiler_sdk_trace_provider.cpp
                                                     rocprofiler_sdk_trace_provider.hpp)

target_include_directories(rocprofiler-sdk-tool-output-lttng
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    rocprofiler-sdk-tool-output-lttng
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-lttng
            dl)

target_link_libraries(rocprofiler-sdk-output-library
                      PRIVATE rocprofiler-sdk-tool-output-lttng)
