#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(rocprofiler-sdk-samples-counter-collection LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)

add_library(counter-collection-buffer-client SHARED)
target_sources(counter-collection-buffer-client PRIVATE client.cpp client.hpp)
target_link_libraries(
    counter-collection-buffer-client
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(counter-collection-buffer)
target_sources(counter-collection-buffer PRIVATE main.cpp)
target_link_libraries(counter-collection-buffer PRIVATE counter-collection-buffer-client
                                                        Threads::Threads)

rocprofiler_samples_get_ld_library_path_env(LIBRARY_PATH_ENV)
rocprofiler_samples_get_preload_env(PRELOAD_ENV counter-collection-buffer-client)

set(counter-collection-buffer-env "${PRELOAD_ENV}" "${LIBRARY_PATH_ENV}")

add_test(NAME counter-collection-buffer COMMAND $<TARGET_FILE:counter-collection-buffer>)

set_tests_properties(
    counter-collection-buffer
    PROPERTIES TIMEOUT 120 LABELS "samples" ENVIRONMENT
               "${counter-collection-buffer-env}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_library(counter-collection-callback-client SHARED)
target_sources(counter-collection-callback-client PRIVATE callback_client.cpp client.hpp)
target_link_libraries(
    counter-collection-callback-client
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(counter-collection-callback)
target_sources(counter-collection-callback PRIVATE main.cpp)
target_link_libraries(counter-collection-callback
                      PRIVATE counter-collection-callback-client Threads::Threads)

rocprofiler_samples_get_preload_env(PRELOAD_ENV counter-collection-callback-client)

set(counter-collection-callback-env "${PRELOAD_ENV}" "${LIBRARY_PATH_ENV}")

add_test(NAME counter-collection-callback
         COMMAND $<TARGET_FILE:counter-collection-callback>)

set_tests_properties(
    counter-collection-callback
    PROPERTIES TIMEOUT 120 LABELS "samples" ENVIRONMENT
               "${counter-collection-callback-env}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_library(counter-collection-functional-counter-client SHARED)
target_sources(counter-collection-functional-counter-client
               PRIVATE print_functional_counters.cpp client.hpp)
target_link_libraries(
    counter-collection-functional-counter-client
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library)

add_executable(counter-collection-print-functional-counters)
target_sources(counter-collection-print-functional-counters PRIVATE main.cpp)
target_link_libraries(
    counter-collection-print-functional-counters
    PRIVATE counter-collection-functional-counter-client Threads::Threads)

rocprofiler_samples_get_preload_env(PRELOAD_ENV
                                    counter-collection-functional-counter-client)

set(counter-collection-functional-counter-env "${PRELOAD_ENV}" "${LIBRARY_PATH_ENV}")

add_test(NAME counter-collection-print-functional-counters
         COMMAND $<TARGET_FILE:counter-collection-print-functional-counters>)

set_tests_properties(
    counter-collection-print-functional-counters
    PROPERTIES TIMEOUT 120 LABELS "samples" ENVIRONMENT
               "${counter-collection-functional-counter-env}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_library(counter-collection-agent-profiling-client SHARED)
target_sources(counter-collection-agent-profiling-client PRIVATE agent_profiling.cpp
                                                                 client.hpp)
target_link_libraries(
    counter-collection-agent-profiling-client
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library)

add_executable(counter-collection-agent-profiling)
target_sources(counter-collection-agent-profiling PRIVATE main.cpp)
target_link_libraries(counter-collection-agent-profiling
                      PRIVATE counter-collection-agent-profiling-client Threads::Threads)

rocprofiler_samples_get_preload_env(PRELOAD_ENV counter-collection-agent-profiling-client)

set(counter-collection-functional-counter-env "${PRELOAD_ENV}" "${LIBRARY_PATH_ENV}")

add_test(NAME counter-collection-agent-profiling
         COMMAND $<TARGET_FILE:counter-collection-agent-profiling>)

set_tests_properties(
    counter-collection-agent-profiling
    PROPERTIES TIMEOUT 120 LABELS "samples" ENVIRONMENT
               "${counter-collection-functional-counter-env}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
