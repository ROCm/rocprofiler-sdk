add_library(rocprofiler-sdk-tool-output-lttng STATIC rocprofiler_sdk_trace_provider.cpp
                                                     rocprofiler_sdk_trace_provider.hpp)

target_include_directories(rocprofiler-sdk-tool-output-lttng
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(rocprofiler-sdk-tool-output-lttng rocprofiler-sdk::rocprofiler-sdk-lttng dl)

target_link_libraries(
    rocprofiler-sdk-output-library
    PRIVATE rocprofiler-sdk-tool-output-lttng)
