#
#   Builds common utilities into a static library
#
rocprofiler_activate_clang_tidy()

set(common_sources environment.cpp demangle.cpp logging.cpp static_object.cpp utility.cpp
                   xml.cpp)
set(common_headers
    defines.hpp
    environment.hpp
    demangle.hpp
    logging.hpp
    mpl.hpp
    scope_destructor.hpp
    static_object.hpp
    stringize_arg.hpp
    synchronized.hpp
    utility.hpp
    xml.hpp)

add_library(rocprofiler-common-library STATIC)
add_library(rocprofiler-sdk::rocprofiler-common-library ALIAS rocprofiler-common-library)

add_subdirectory(container)
add_subdirectory(memory)

target_sources(rocprofiler-common-library PRIVATE ${common_sources} ${common_headers})
target_include_directories(rocprofiler-common-library
                           PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)

target_link_libraries(
    rocprofiler-common-library
    PUBLIC $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-headers>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-build-flags>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-threading>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-memcheck>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-cxx-filesystem>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-glog>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-fmt>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-dl>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-ptl>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-atomic>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-hsakmt-nolink>)

set_target_properties(rocprofiler-common-library PROPERTIES OUTPUT_NAME
                                                            rocprofiler-common)
