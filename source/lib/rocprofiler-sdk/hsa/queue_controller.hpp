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

#pragma once

#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/rocprofiler-sdk/hsa/profile_serializer.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
// Tracks and manages HSA queues
class QueueController
{
public:
    using agent_callback_tuple_t =
        std::tuple<rocprofiler_agent_t, Queue::queue_cb_t, Queue::completed_cb_t>;
    using queue_iterator_cb_t    = std::function<void(const Queue*)>;
    using callback_iterator_cb_t = std::function<void(ClientID, const agent_callback_tuple_t&)>;
    using queue_map_t            = std::unordered_map<hsa_queue_t*, std::unique_ptr<Queue>>;

    QueueController() = default;
    // Initializes the QueueInterceptor. This must be delayed until
    // HSA has been inited.
    void init(CoreApiTable& core_table, AmdExtTable& ext_table);

    // Called to add a queue that was created by the user program
    void add_queue(hsa_queue_t*, std::unique_ptr<Queue>);
    void destroy_queue(hsa_queue_t*);

    // Add callback to queues associated with the agent. Returns a client
    // id that can be used by callers to remove the callback. If no agent
    // is specified, callback will be applied to all agents.
    ClientID add_callback(std::optional<rocprofiler_agent_t>,
                          Queue::queue_cb_t,
                          Queue::completed_cb_t);
    void     remove_callback(ClientID);

    const CoreApiTable& get_core_table() const { return _core_table; }
    const AmdExtTable&  get_ext_table() const { return _ext_table; }

    // Gets the list of supported HSA agents that can be Pintercepted
    const auto& get_supported_agents() const { return _supported_agents; }
    auto&       get_supported_agents() { return _supported_agents; }

    const Queue* get_queue(const hsa_queue_t&) const;

    void iterate_queues(const queue_iterator_cb_t&) const;
    void set_queue_state(queue_state state, hsa_queue_t* hsa_queue);

    void add_dispatch_ready(const Queue* queue);

    void iterate_callbacks(const callback_iterator_cb_t&) const;

    common::Synchronized<hsa::profiler_serializer>& serializer() { return _profiler_serializer; }

    /**
     * Disable serialization for QueueController, has no effect if counter collection
     * is not in use (which defaults to no serialization mechanism). Should only be used for
     * testing.
     */
    void enable_serialization();
    void disable_serialization();

    // Prints current state of signals for queues, used for debugging. Only prints
    // serialization related signals if not compiled in debug mode.
    void print_debug_signals() const;

#if !defined(NDEBUG)
    // Tracks the creation of all signals in queues, used for debugging and disabled
    // in release mode (adds locking around signal creation).
    common::Synchronized<std::unordered_map<uint64_t, hsa_signal_t>> _debug_signals;
#endif

private:
    using client_id_map_t   = std::unordered_map<ClientID, agent_callback_tuple_t>;
    using agent_cache_map_t = std::unordered_map<uint32_t, AgentCache>;

    CoreApiTable                                   _core_table       = {};
    AmdExtTable                                    _ext_table        = {};
    common::Synchronized<queue_map_t>              _queues           = {};
    common::Synchronized<client_id_map_t>          _callback_cache   = {};
    agent_cache_map_t                              _supported_agents = {};
    common::Synchronized<hsa::profiler_serializer> _profiler_serializer;
};

QueueController*
get_queue_controller();

void
queue_controller_init(HsaApiTable* table);

void
queue_controller_fini();

void
profiler_serializer_kernel_completion_signal(hsa_signal_t queue_block_signal);

}  // namespace hsa
}  // namespace rocprofiler
