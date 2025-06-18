#
# libpyrocpd python binding sources
#

set(libpyrocpd_source_headers
    common.hpp
    functions.hpp
    interop.hpp
    perfetto.hpp
    csv.hpp
    otf2.hpp
    ctf2.hpp
    sql_generator.hpp
    pysqlite_Connection.h
    types.hpp
    ctf/barectf.h
    ctf/barectf-bitfield.h)

set(libpyrocpd_source_sources csv.cpp functions.cpp interop.cpp otf2.cpp ctf2.cpp perfetto.cpp
                              types.cpp ctf/barectf.c)

foreach(_PYTHON_VERSION ${ROCPROFILER_PYTHON_VERSIONS})
    rocprofiler_rocpd_python_bindings_target_sources(
        ${_PYTHON_VERSION} PRIVATE ${libpyrocpd_source_sources}
        ${libpyrocpd_source_headers})
endforeach()

add_subdirectory(serialization)
