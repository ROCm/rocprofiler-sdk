#
#
#
set(ROCPROFILER_LIB_HSA_DETAILS_SOURCES)
set(ROCPROFILER_LIB_HSA_DETAILS_HEADERS ostream.hpp fmt.hpp)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_HSA_DETAILS_SOURCES}
                                                  ${ROCPROFILER_LIB_HSA_DETAILS_HEADERS})
