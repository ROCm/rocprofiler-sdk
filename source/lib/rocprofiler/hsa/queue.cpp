/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "lib/rocprofiler/hsa/queue.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler/buffer.hpp"
#include "lib/rocprofiler/context/context.hpp"

#include <glog/logging.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <rocprofiler/fwd.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace rocprofiler
{
namespace hsa
{
namespace
{
common::active_capacity_gate&
signal_limiter()
{
    // Limit the maximun number of HSA signals created.
    // There is a hard limit to the maximum that can exist.
    static common::active_capacity_gate _gate(1024);
    return _gate;
}

bool
AsyncSignalHandler(hsa_signal_value_t, void* data)
{
    if(!data) return true;
    auto& queue_info_session = *static_cast<Queue::queue_info_session_t*>(data);

    // Calls our internal callbacks to callers who need to be notified post
    // kernel execution.
    queue_info_session.queue.signal_callback([&](const auto& map) {
        for(const auto& [client_id, cb_pair] : map)
        {
            // If this is the client that gave us the AQLPacket,
            // return it to that client otherwise notify.
            if(queue_info_session.inst_pkt_id == client_id)
            {
                cb_pair.second(queue_info_session.queue,
                               client_id,
                               queue_info_session.kernel_pkt,
                               std::move(queue_info_session.inst_pkt));
            }
            else
            {
                cb_pair.second(
                    queue_info_session.queue, client_id, queue_info_session.kernel_pkt, nullptr);
            }
        }
    });

    size_t signals_to_remove = 0;
    // Delete signals and packets, signal we have completed.
    if(queue_info_session.interrupt_signal.handle != 0u)
    {
        signals_to_remove++;
        queue_info_session.queue.core_api().hsa_signal_destroy_fn(
            queue_info_session.interrupt_signal);
    }
    if(queue_info_session.kernel_pkt.ext_amd_aql_pm4.completion_signal.handle != 0u)
    {
        signals_to_remove++;
        queue_info_session.queue.core_api().hsa_signal_destroy_fn(
            queue_info_session.kernel_pkt.ext_amd_aql_pm4.completion_signal);
    }
    if(signals_to_remove > 0)
    {
        signal_limiter().remove_active(signals_to_remove);
    }
    queue_info_session.queue.async_complete();

    delete static_cast<Queue::queue_info_session_t*>(data);
    return false;
}

template <typename Integral = uint64_t>
constexpr Integral
bit_mask(int first, int last)
{
    assert(last >= first && "Error: hsa_support::bit_mask -> invalid argument");
    size_t num_bits = last - first + 1;
    return ((num_bits >= sizeof(Integral) * 8) ? ~Integral{0}
                                               /* num_bits exceed the size of Integral */
                                               : ((Integral{1} << num_bits) - 1))
           << first;
}

/* Extract bits [last:first] from t.  */
template <typename Integral>
constexpr Integral
bit_extract(Integral x, int first, int last)
{
    return (x >> first) & bit_mask<Integral>(0, last - first);
}

/**
 * @brief This function is a queue write interceptor. It intercepts the
 * packet write function. Creates an instance of packet class with the raw
 * pointer. invoke the populate function of the packet class which returns a
 * pointer to the packet. This packet is written into the queue by this
 * interceptor by invoking the writer function.
 */
void
WriteInterceptor(const void* packets,
                 uint64_t    pkt_count,
                 uint64_t,
                 void*                                 data,
                 hsa_amd_queue_intercept_packet_writer writer)
{
    auto&& AddVendorSpecificPacket = [](hsa_ext_amd_aql_pm4_packet_t     _packet,
                                        hsa_signal_t                     _signal,
                                        std::vector<rocprofiler_packet>& _packets) {
        _packets.emplace_back(_packet).ext_amd_aql_pm4.completion_signal = _signal;
    };

    auto&& CreateBarrierPacket = [](hsa_signal_t                     _signal,
                                    std::vector<rocprofiler_packet>& _packets) {
        hsa_barrier_and_packet_t barrier{};
        barrier.header        = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        barrier.dep_signal[0] = _signal;
        _packets.emplace_back(barrier);
    };

    LOG_IF(FATAL, data == nullptr) << "WriteInterceptor was not passed a pointer to the queue";

    auto& queue   = *static_cast<Queue*>(data);
    auto  thr_id  = common::get_tid();
    auto* corr_id = context::get_latest_correlation_id();
    // increase the reference count to denote that this correlation id is being used in a kernel
    if(corr_id) corr_id->ref_count.fetch_add(1);

    // We have no packets or no one who needs to be notified, do nothing.
    if(pkt_count == 0 || queue.get_notifiers() == 0)
    {
        writer(packets, pkt_count);
        return;
    }

    // hsa_ext_amd_aql_pm4_packet_t
    const auto* packets_arr         = static_cast<const rocprofiler_packet*>(packets);
    auto        transformed_packets = std::vector<rocprofiler_packet>{};

    // Searching accross all the packets given during this write
    for(size_t i = 0; i < pkt_count; ++i)
    {
        const auto& original_packet = packets_arr[i].kernel_dispatch;
        auto        packet_type     = bit_extract(original_packet.header,
                                       HSA_PACKET_HEADER_TYPE,
                                       HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1);
        if(packet_type != HSA_PACKET_TYPE_KERNEL_DISPATCH)
        {
            transformed_packets.emplace_back(packets_arr[i]);
            continue;
        }

        // Copy kernel pkt, copy is to allow for signal to be modified
        rocprofiler_packet kernel_pkt = packets_arr[i];
        queue.create_signal(HSA_AMD_SIGNAL_AMD_GPU_ONLY,
                            &kernel_pkt.ext_amd_aql_pm4.completion_signal);

        // Stores the instrumentation pkt (i.e. AQL packets for counter collection)
        // along with an ID of the client we got the packet from (this will be returned via
        // completed_cb_t)
        ClientID                   inst_pkt_id = -1;
        std::unique_ptr<AQLPacket> inst_pkt;

        // Signal callbacks that a kernel_pkt is being enqueued
        queue.signal_callback([&](const auto& map) {
            for(const auto& [client_id, cb_pair] : map)
            {
                if(auto maybe_pkt = cb_pair.first(queue, client_id, kernel_pkt))
                {
                    LOG_IF(FATAL, inst_pkt)
                        << "We do not support two injections into the HSA queue";
                    inst_pkt    = std::move(maybe_pkt);
                    inst_pkt_id = client_id;
                }
            }
        });

        constexpr auto dummy_signal = hsa_signal_t{.handle = 0};

        // Write instrumentation start packet (if one exists)
        if(inst_pkt)
        {
            inst_pkt->start.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
            AddVendorSpecificPacket(inst_pkt->start, dummy_signal, transformed_packets);
            CreateBarrierPacket(inst_pkt->start.completion_signal, transformed_packets);
        }

        transformed_packets.emplace_back(kernel_pkt);

        // Make a copy of the original packet, adding its signal to a barrier
        // packet and create a new signal for it to get timestamps
        if(original_packet.completion_signal.handle != 0u)
        {
            hsa_barrier_and_packet_t barrier{};
            barrier.header            = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
            barrier.completion_signal = original_packet.completion_signal;
            transformed_packets.emplace_back(barrier);
        }

        hsa_signal_t interrupt_signal{};
        // Adding a barrier packet with the original packet's completion signal.
        queue.create_signal(0, &interrupt_signal);

        if(inst_pkt)
        {
            inst_pkt->stop.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
            AddVendorSpecificPacket(inst_pkt->stop, dummy_signal, transformed_packets);
            inst_pkt->read.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
            AddVendorSpecificPacket(inst_pkt->read, interrupt_signal, transformed_packets);

            // Added Interrupt Signal with barrier and provided handler for it
            CreateBarrierPacket(interrupt_signal, transformed_packets);
        }
        else
        {
            hsa_barrier_and_packet_t barrier{};
            barrier.header            = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
            barrier.completion_signal = interrupt_signal;
            transformed_packets.emplace_back(barrier);
        }

        // TODO(jrmadsen): fetch kernel identifier from code object loading
        uint64_t kernel_id = 0;

        // Enqueue the signal into the handler. Will call completed_cb when
        // signal completes.
        queue.async_started();
        queue.signal_async_handler(
            interrupt_signal,
            new Queue::queue_info_session_t{.queue            = queue,
                                            .inst_pkt         = std::move(inst_pkt),
                                            .inst_pkt_id      = inst_pkt_id,
                                            .interrupt_signal = interrupt_signal,
                                            .tid              = thr_id,
                                            .kernel_id        = kernel_id,
                                            .correlation_id   = corr_id,
                                            .kernel_pkt       = kernel_pkt});
    }

    writer(transformed_packets.data(), transformed_packets.size());
}
}  // namespace

Queue::~Queue()
{
    // Potentially replace with condition variable at some point
    // but performance may not matter here.
    while(_active_async_packets.load(std::memory_order_relaxed) > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
}

void
Queue::signal_async_handler(const hsa_signal_t& signal, Queue::queue_info_session_t* data) const
{
    hsa_status_t status = _ext_api.hsa_amd_signal_async_handler_fn(
        signal, HSA_SIGNAL_CONDITION_EQ, 0, AsyncSignalHandler, static_cast<void*>(data));
    LOG_IF(FATAL, status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: hsa_amd_signal_async_handler failed";
}

void
Queue::create_signal(uint32_t attribute, hsa_signal_t* signal) const
{
    signal_limiter().add_active(1);
    hsa_status_t status = _ext_api.hsa_amd_signal_create_fn(1, 0, nullptr, attribute, signal);
    LOG_IF(FATAL, status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: hsa_amd_signal_create failed";
}

Queue::Queue(const AgentCache&  agent,
             uint32_t           size,
             hsa_queue_type32_t type,
             void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
             void*         data,
             uint32_t      private_segment_size,
             uint32_t      group_segment_size,
             CoreApiTable  core_api,
             AmdExtTable   ext_api,
             hsa_queue_t** queue)
: _core_api(core_api)
, _ext_api(ext_api)
, _agent(agent)

{
    LOG_IF(FATAL,
           _ext_api.hsa_amd_queue_intercept_create_fn(_agent.get_hsa_agent(),
                                                      size,
                                                      type,
                                                      callback,
                                                      data,
                                                      private_segment_size,
                                                      group_segment_size,
                                                      &_intercept_queue) != HSA_STATUS_SUCCESS)
        << "Could not create intercept queue";

    LOG_IF(FATAL,
           _ext_api.hsa_amd_profiling_set_profiler_enabled_fn(_intercept_queue, true) !=
               HSA_STATUS_SUCCESS)
        << "Could not setup intercept profiler";

    LOG_IF(FATAL,
           _ext_api.hsa_amd_queue_intercept_register_fn(_intercept_queue, WriteInterceptor, this))
        << "Could not register interceptor";
    *queue = _intercept_queue;
}

void
Queue::register_callback(ClientID id, queue_cb_t enqueue_cb, completed_cb_t complete_cb)
{
    _callbacks.wlock([&](auto& map) {
        LOG_IF(FATAL, rocprofiler::common::get_val(map, id)) << "ID already exists!";
        _notifiers++;
        map[id] = std::make_pair(enqueue_cb, complete_cb);
    });
}

void
Queue::remove_callback(ClientID id)
{
    _callbacks.wlock([&](auto& map) {
        if(map.erase(id) == 1) _notifiers--;
    });
}
}  // namespace hsa
}  // namespace rocprofiler
