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

project(rocprofiler-sdk-samples-code-object-isa-decode LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)

add_executable(code-object-isa-decode)
target_sources(code-object-isa-decode PRIVATE main.cpp client.cpp)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS "-g")

target_link_libraries(
    code-object-isa-decode PRIVATE Threads::Threads rocprofiler::samples-common-library
                                   rocprofiler-sdk-codeobj rocprofiler::rocprofiler)

add_test(NAME code-object-isa-decode COMMAND $<TARGET_FILE:code-object-isa-decode>)

set_tests_properties(
    code-object-isa-decode
    PROPERTIES TIMEOUT 45 LABELS "samples" ENVIRONMENT
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}" FAIL_REGULAR_EXPRESSION
               "threw an exception")