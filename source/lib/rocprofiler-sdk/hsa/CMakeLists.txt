#
set(ROCPROFILER_LIB_HSA_SOURCES
    agent_cache.cpp
    aql_packet.cpp
    async_copy.cpp
    hsa_barrier.cpp
    hsa.cpp
    pc_sampling.hpp
    profile_serializer.cpp
    queue_controller.cpp
    queue.cpp
    scratch_memory.cpp)

set(ROCPROFILER_LIB_HSA_HEADERS
    agent_cache.hpp
    aql_packet.hpp
    async_copy.hpp
    defines.hpp
    hsa_barrier.hpp
    hsa.hpp
    pc_sampling.cpp
    profile_serializer.hpp
    queue_controller.hpp
    queue.hpp
    queue_info_session.hpp
    rocprofiler_packet.hpp
    scratch_memory.hpp
    types.hpp
    utils.hpp)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_HSA_SOURCES}
                                                  ${ROCPROFILER_LIB_HSA_HEADERS})

add_subdirectory(details)
