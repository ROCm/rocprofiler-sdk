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

project(rocprofiler-samples-code-object-tracing LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

if(NOT TARGET rocprofiler::rocprofiler)
    find_package(rocprofiler REQUIRED)
endif()

add_library(code-object-tracing-client SHARED)
target_sources(code-object-tracing-client PRIVATE client.cpp)
target_link_libraries(
    code-object-tracing-client
    PRIVATE rocprofiler::rocprofiler
            $<TARGET_NAME_IF_EXISTS:rocprofiler::samples-build-flags>)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
find_package(Threads REQUIRED)

add_executable(code-object-tracing)
target_sources(code-object-tracing PRIVATE main.cpp)
target_link_libraries(
    code-object-tracing PRIVATE code-object-tracing-client Threads::Threads
                                $<TARGET_NAME_IF_EXISTS:rocprofiler::samples-build-flags>)

add_test(NAME code-object-tracing COMMAND $<TARGET_FILE:code-object-tracing>)

set_tests_properties(
    code-object-tracing
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "samples"
        ENVIRONMENT
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV};HSA_TOOLS_LIB=$<TARGET_FILE:rocprofiler::rocprofiler>"
        FAIL_REGULAR_EXPRESSION
        "threw an exception")