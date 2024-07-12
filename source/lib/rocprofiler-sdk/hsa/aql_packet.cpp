// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include <cstdlib>
#include <iostream>
#include "lib/common/logging.hpp"

#define CHECK_HSA(fn, message)                                                                     \
    if((fn) != HSA_STATUS_SUCCESS)                                                                 \
    {                                                                                              \
        ROCP_ERROR << message;                                                                     \
        exit(1);                                                                                   \
    }

namespace rocprofiler
{
namespace hsa
{
hsa_status_t
CounterAQLPacket::CounterMemoryPool::Alloc(void** ptr, size_t size, desc_t flags, void* data)
{
    if(size == 0)
    {
        if(ptr != nullptr) *ptr = nullptr;
        return HSA_STATUS_SUCCESS;
    }
    if(!data) return HSA_STATUS_ERROR;
    auto& pool = *reinterpret_cast<CounterAQLPacket::CounterMemoryPool*>(data);

    if(!pool.allocate_fn || !pool.free_fn || !pool.allow_access_fn) return HSA_STATUS_ERROR;
    if(!flags.host_access || pool.kernarg_pool_.handle == 0 || !pool.fill_fn)
        return HSA_STATUS_ERROR;

    hsa_status_t status;
    if(!pool.bIgnoreKernArg && flags.memory_hint == AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED)
        status = pool.allocate_fn(pool.kernarg_pool_, size, 0, ptr);
    else
        status = pool.allocate_fn(pool.cpu_pool_, size, 0, ptr);

    if(status != HSA_STATUS_SUCCESS) return status;

    status = pool.fill_fn(*ptr, 0u, size / sizeof(uint32_t));
    if(status != HSA_STATUS_SUCCESS) return status;

    status = pool.allow_access_fn(1, &pool.gpu_agent, nullptr, *ptr);
    return status;
}

void
CounterAQLPacket::CounterMemoryPool::Free(void* ptr, void* data)
{
    if(ptr == nullptr) return;

    assert(data);
    auto& pool = *reinterpret_cast<CounterAQLPacket::CounterMemoryPool*>(data);
    assert(pool.free_fn);
    pool.free_fn(ptr);
}

hsa_status_t
CounterAQLPacket::CounterMemoryPool::Copy(void* dst, const void* src, size_t size, void* data)
{
    if(size == 0) return HSA_STATUS_SUCCESS;
    if(!data) return HSA_STATUS_ERROR;
    auto& pool = *reinterpret_cast<CounterAQLPacket::CounterMemoryPool*>(data);

    if(!pool.api_copy_fn) return HSA_STATUS_ERROR;

    return pool.api_copy_fn(dst, src, size);
}

CounterAQLPacket::CounterAQLPacket(aqlprofile_agent_handle_t                  agent,
                                   CounterAQLPacket::CounterMemoryPool        _pool,
                                   const std::vector<aqlprofile_pmc_event_t>& events)
: pool(_pool)
{
    if(events.empty()) return;

    packets.start_packet = null_amd_aql_pm4_packet;
    packets.stop_packet  = null_amd_aql_pm4_packet;
    packets.read_packet  = null_amd_aql_pm4_packet;

    aqlprofile_pmc_profile_t profile{};
    profile.agent       = agent;
    profile.events      = events.data();
    profile.event_count = static_cast<uint32_t>(events.size());

    hsa_status_t status = aqlprofile_pmc_create_packets(&this->handle,
                                                        &this->packets,
                                                        profile,
                                                        &CounterMemoryPool::Alloc,
                                                        &CounterMemoryPool::Free,
                                                        &CounterMemoryPool::Copy,
                                                        reinterpret_cast<void*>(&pool));
    if(status != HSA_STATUS_SUCCESS) ROCP_FATAL << "Could not create PMC packets!";

    auto header                 = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packets.start_packet.header = header;
    packets.stop_packet.header  = header;
    packets.read_packet.header  = header;
    empty                       = false;
}

hsa_status_t
TraceMemoryPool::Alloc(void** ptr, size_t size, desc_t flags, void* data)
{
    if(!data) return HSA_STATUS_ERROR;
    auto& pool = *reinterpret_cast<TraceMemoryPool*>(data);

    if(!pool.allocate_fn || !pool.free_fn || !pool.allow_access_fn) return HSA_STATUS_ERROR;

    hsa_status_t status = HSA_STATUS_ERROR;
    if(flags.host_access)
    {
        status = pool.allocate_fn(pool.cpu_pool_, size, 0, ptr);

        if(status == HSA_STATUS_SUCCESS)
            status = pool.allow_access_fn(1, &pool.gpu_agent, nullptr, *ptr);
    }
    else
    {
        // Return page aligned data to avoid cache flush overlap
        status = pool.allocate_fn(pool.gpu_pool_, size + 0x2000, 0, ptr);
        *ptr   = (void*) ((uintptr_t(*ptr) + 0xFFF) & ~0xFFFul);  // NOLINT
    }
    return status;
}

void
TraceMemoryPool::Free(void* ptr, void* data)
{
    assert(data);
    auto& pool = *reinterpret_cast<TraceMemoryPool*>(data);

    if(pool.free_fn) pool.free_fn(ptr);
}

hsa_status_t
TraceMemoryPool::Copy(void* dst, const void* src, size_t size, void* data)
{
    if(!data) return HSA_STATUS_ERROR;
    auto& pool = *reinterpret_cast<TraceMemoryPool*>(data);

    if(!pool.api_copy_fn) return HSA_STATUS_ERROR;

    return pool.api_copy_fn(dst, src, size);
}

TraceControlAQLPacket::TraceControlAQLPacket(const TraceMemoryPool&          _tracepool,
                                             const aqlprofile_att_profile_t& p)
: tracepool(std::make_shared<TraceMemoryPool>(_tracepool))
{
    auto status = aqlprofile_att_create_packets(&tracepool->handle,
                                                &packets,
                                                p,
                                                &TraceMemoryPool::Alloc,
                                                &TraceMemoryPool::Free,
                                                &TraceMemoryPool::Copy,
                                                tracepool.get());
    CHECK_HSA(status, "failed to create ATT packet");

    packets.start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packets.stop_packet.header  = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packets.start_packet.completion_signal = hsa_signal_t{.handle = 0};
    packets.stop_packet.completion_signal  = hsa_signal_t{.handle = 0};
    this->empty                            = false;

    clear();
};

CodeobjMarkerAQLPacket::CodeobjMarkerAQLPacket(const TraceMemoryPool& _tracepool,
                                               uint64_t               id,
                                               uint64_t               addr,
                                               uint64_t               size,
                                               bool                   bFromStart,
                                               bool                   bIsUnload)
: tracepool(_tracepool)
{
    aqlprofile_att_codeobj_data_t codeobj{};
    codeobj.id        = id;
    codeobj.addr      = addr;
    codeobj.size      = size;
    codeobj.agent     = tracepool.gpu_agent;
    codeobj.isUnload  = bIsUnload;
    codeobj.fromStart = bFromStart;

    auto status = aqlprofile_att_codeobj_marker(&packet,
                                                &tracepool.handle,
                                                codeobj,
                                                &TraceMemoryPool::Alloc,
                                                &TraceMemoryPool::Free,
                                                &tracepool);
    CHECK_HSA(status, "failed to create ATT marker");

    packet.header            = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packet.completion_signal = hsa_signal_t{.handle = 0};
    this->empty              = false;

    clear();
}

}  // namespace hsa
}  // namespace rocprofiler
