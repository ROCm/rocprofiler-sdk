#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rocprofv3-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

add_test(
    NAME rocprofv3-test-trace-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hsa-trace --kernel-trace
        --memory-copy-trace -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace -o out
        $<TARGET_FILE:simple-transpose>)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(tracing-env "${PRELOAD_ENV}"
                "HSA_TOOLS_LIB=$<TARGET_FILE:rocprofiler::rocprofiler-shared-library>")

set_tests_properties(rocprofv3-test-trace-execute PROPERTIES LABELS "integration-tests"
                                                             ENVIRONMENT "${tracing-env}")

foreach(FILENAME validate.py conftest.py)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                   ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()

add_test(
    NAME rocprofv3-test-trace-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --hsa-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_hsa_api_trace.csv
        --kernel-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_kernel_trace.csv
        --memory-copy-input
        ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-trace/out_memory_copy_trace.csv)

set_tests_properties(
    rocprofv3-test-trace-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS rocprofv3-test-trace-execute
               FAIL_REGULAR_EXPRESSION "AssertionError")