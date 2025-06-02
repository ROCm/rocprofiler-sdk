find_package(LTTngUST REQUIRED)

add_library(rocprofiler-sdk-tool-output-lttng STATIC rocprofiler_sdk_trace_provider.cpp
                                                     rocprofiler_sdk_trace_provider.hpp)

target_include_directories(rocprofiler-sdk-tool-output-lttng
                           PUBLIC ${LTTNGUST_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(rocprofiler-sdk-tool-output-lttng ${LTTNGUST_LIBRARIES} dl)
