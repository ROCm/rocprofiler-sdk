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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/counters/core.hpp"

#include "lib/common/container/small_vector.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

namespace rocprofiler
{
namespace counters
{
class CounterController
{
public:
    CounterController()
    {
        // Pre-read metrics map file to catch faliures during initial setup.
        rocprofiler::counters::getMetricIdMap();
    }

    // Adds a counter collection profile to our global cache.
    // Note: these profiles can be used across multiple contexts
    //       and are independent of the context.
    uint64_t add_profile(std::shared_ptr<profile_config>&& config)
    {
        static std::atomic<uint64_t> profile_val = 1;
        uint64_t                     ret         = 0;
        _configs.wlock([&](auto& data) {
            config->id = rocprofiler_profile_config_id_t{.handle = profile_val};
            data.emplace(profile_val, std::move(config));
            ret = profile_val;
            profile_val++;
        });
        return ret;
    }

    void destroy_profile(uint64_t id)
    {
        _configs.wlock([&](auto& data) { data.erase(id); });
    }

    // Setup the counter collection service. counter_callback_info is created here
    // to contain the counters that need to be collected (specified in profile_id) and
    // the AQL packet generator for injecting packets. Note: the service is created
    // in the stop state.
    static bool configure_dispatch(rocprofiler_context_id_t                         context_id,
                                   rocprofiler_buffer_id_t                          buffer,
                                   rocprofiler_profile_counting_dispatch_callback_t callback,
                                   void*                                            callback_args,
                                   rocprofiler_profile_counting_record_callback_t   record_callback,
                                   void* record_callback_args)
    {
        auto* ctx_p = rocprofiler::context::get_mutable_registered_context(context_id);
        if(!ctx_p) return false;

        auto& ctx = *ctx_p;

        if(!ctx.counter_collection)
        {
            ctx.counter_collection =
                std::make_unique<rocprofiler::context::counter_collection_service>();
        }

        auto& cb = *ctx.counter_collection->callbacks.emplace_back(
            std::make_shared<counter_callback_info>());

        cb.user_cb       = callback;
        cb.callback_args = callback_args;
        cb.context       = context_id;
        if(buffer.handle != 0)
        {
            cb.buffer = buffer;
        }
        cb.internal_context     = ctx_p;
        cb.record_callback      = record_callback;
        cb.record_callback_args = record_callback_args;

        return true;
    }

    std::shared_ptr<profile_config> get_profile_cfg(rocprofiler_profile_config_id_t id)
    {
        std::shared_ptr<profile_config> cfg;
        _configs.rlock([&](const auto& map) { cfg = map.at(id.handle); });
        return cfg;
    }

private:
    rocprofiler::common::Synchronized<std::unordered_map<uint64_t, std::shared_ptr<profile_config>>>
        _configs;
};

CounterController&
get_controller()
{
    static CounterController controller;
    return controller;
}

uint64_t
create_counter_profile(std::shared_ptr<profile_config>&& config)
{
    return get_controller().add_profile(std::move(config));
}

void
destroy_counter_profile(uint64_t id)
{
    get_controller().destroy_profile(id);
}

std::shared_ptr<profile_config>
get_profile_config(rocprofiler_profile_config_id_t id)
{
    try
    {
        return get_controller().get_profile_cfg(id);
    } catch(std::out_of_range&)
    {
        return nullptr;
    }
}

rocprofiler_status_t
counter_callback_info::setup_profile_config(const hsa::AgentCache&           agent,
                                            std::shared_ptr<profile_config>& profile)
{
    if(profile->pkt_generator || !profile->reqired_hw_counters.empty())
    {
        return ROCPROFILER_STATUS_SUCCESS;
    }

    // Sets up the packet generator for the profile. This must be delayed until after HSA is loaded.
    // This call needs to be thread protected in that only one thread must be setting up profile at
    // the same time.

    auto& config     = *profile;
    auto  agent_name = std::string(config.agent->name);
    for(const auto& metric : config.metrics)
    {
        auto req_counters = get_required_hardware_counters(get_ast_map(), agent_name, metric);

        if(!req_counters)
        {
            LOG(ERROR) << fmt::format("Could not find counter {}", metric.name());
            return ROCPROFILER_STATUS_ERROR_PROFILE_COUNTER_NOT_FOUND;
        }

        // Special metrics are those that are not hw counters but other
        // constants like MAX_WAVE_SIZE
        for(const auto& req_metric : *req_counters)
        {
            if(req_metric.special().empty())
            {
                config.reqired_hw_counters.insert(req_metric);
            }
            else
            {
                config.required_special_counters.insert(req_metric);
            }
        }

        const auto& asts      = get_ast_map();
        const auto* agent_map = rocprofiler::common::get_val(asts, agent_name);
        if(!agent_map)
        {
            LOG(ERROR) << fmt::format("Coult not build AST for {}", agent_name);
            return ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED;
        }

        const auto* counter_ast = rocprofiler::common::get_val(*agent_map, metric.name());
        if(!counter_ast)
        {
            LOG(ERROR) << fmt::format("Coult not find AST for {}", metric.name());
            return ROCPROFILER_STATUS_ERROR_AST_NOT_FOUND;
        }
        config.asts.push_back(*counter_ast);
        config.asts.back().set_dimensions();
    }

    profile->pkt_generator = std::make_unique<rocprofiler::aql::AQLPacketConstruct>(
        agent,
        std::vector<counters::Metric>{profile->reqired_hw_counters.begin(),
                                      profile->reqired_hw_counters.end()});
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
counter_callback_info::get_packet(std::unique_ptr<rocprofiler::hsa::AQLPacket>& ret_pkt,
                                  const hsa::AgentCache&                        agent,
                                  std::shared_ptr<profile_config>&              profile)
{
    rocprofiler_status_t status;
    // Check packet cache
    profile->packets.wlock([&](auto& pkt_vector) {
        status = counter_callback_info::setup_profile_config(agent, profile);
        if(!pkt_vector.empty() && status == ROCPROFILER_STATUS_SUCCESS)
        {
            ret_pkt = std::move(pkt_vector.back());
            pkt_vector.pop_back();
        }
    });

    if(status != ROCPROFILER_STATUS_SUCCESS) return status;
    if(!ret_pkt)
    {
        // If we do not have a packet in the cache, create one.
        ret_pkt =
            profile->pkt_generator->construct_packet(hsa::get_queue_controller().get_ext_table());
    }

    ret_pkt->before_krn_pkt.clear();
    ret_pkt->after_krn_pkt.clear();
    packet_return_map.wlock([&](auto& data) { data.emplace(ret_pkt.get(), profile); });

    return ROCPROFILER_STATUS_SUCCESS;
}

/**
 * Callback we get from HSA interceptor when a kernel packet is being enqueued.
 *
 * We return an AQLPacket containing the start/stop/read packets for injection.
 */
std::unique_ptr<rocprofiler::hsa::AQLPacket>
queue_cb(const std::shared_ptr<counter_callback_info>&                   info,
         const hsa::Queue&                                               queue,
         const hsa::rocprofiler_packet&                                  pkt,
         uint64_t                                                        kernel_id,
         rocprofiler_user_data_t*                                        user_data,
         const hsa::Queue::queue_info_session_t::external_corr_id_map_t& extern_corr_ids,
         const context::correlation_id*                                  correlation_id)
{
    if(!info || !info->user_cb) return nullptr;

    auto _corr_id_v =
        rocprofiler_correlation_id_t{.internal = 0, .external = context::null_user_data};
    if(const auto* _corr_id = correlation_id)
    {
        _corr_id_v.internal = _corr_id->internal;
        if(const auto* external =
               rocprofiler::common::get_val(extern_corr_ids, info->internal_context))
        {
            _corr_id_v.external = *external;
        }
    }

    auto req_profile = rocprofiler_profile_config_id_t{.handle = 0};
    auto dispatch_data =
        common::init_public_api_struct(rocprofiler_profile_counting_dispatch_data_t{});

    dispatch_data.kernel_id            = kernel_id;
    dispatch_data.agent_id             = CHECK_NOTNULL(queue.get_agent().get_rocp_agent())->id;
    dispatch_data.queue_id             = queue.get_id();
    dispatch_data.correlation_id       = _corr_id_v;
    dispatch_data.private_segment_size = pkt.kernel_dispatch.private_segment_size;
    dispatch_data.group_segment_size   = pkt.kernel_dispatch.group_segment_size;
    dispatch_data.workgroup_size       = {pkt.kernel_dispatch.workgroup_size_x,
                                    pkt.kernel_dispatch.workgroup_size_y,
                                    pkt.kernel_dispatch.workgroup_size_z};
    dispatch_data.grid_size            = {pkt.kernel_dispatch.grid_size_x,
                               pkt.kernel_dispatch.grid_size_y,
                               pkt.kernel_dispatch.grid_size_z};

    info->user_cb(dispatch_data, &req_profile, user_data, info->callback_args);

    if(req_profile.handle == 0) return nullptr;

    auto prof_config = get_controller().get_profile_cfg(req_profile);
    CHECK(prof_config);

    std::unique_ptr<rocprofiler::hsa::AQLPacket> ret_pkt;
    auto status = info->get_packet(ret_pkt, queue.get_agent(), prof_config);
    CHECK_EQ(status, ROCPROFILER_STATUS_SUCCESS) << rocprofiler_get_status_string(status);

    if(ret_pkt->empty) return ret_pkt;

    auto&& CreateBarrierPacket =
        [](hsa_signal_t*                                                     dependency_signal,
           hsa_signal_t*                                                     completion_signal,
           common::container::small_vector<hsa_ext_amd_aql_pm4_packet_t, 3>& _packets) {
            hsa::rocprofiler_packet barrier{};
            barrier.barrier_and.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
            if(dependency_signal != nullptr) barrier.barrier_and.dep_signal[0] = *dependency_signal;
            if(completion_signal != nullptr)
                barrier.barrier_and.completion_signal = *completion_signal;
            _packets.emplace_back(barrier.ext_amd_aql_pm4);
        };

    hsa_signal_t ready_signal = queue.ready_signal;
    hsa_signal_t block_signal = queue.block_signal;
    CreateBarrierPacket(nullptr, &ready_signal, ret_pkt->before_krn_pkt);
    CreateBarrierPacket(&block_signal, &block_signal, ret_pkt->before_krn_pkt);

    ret_pkt->before_krn_pkt.push_back(ret_pkt->start);
    ret_pkt->before_krn_pkt.end()->completion_signal.handle = 0;
    ret_pkt->after_krn_pkt.push_back(ret_pkt->stop);
    ret_pkt->after_krn_pkt.push_back(ret_pkt->read);
    for(auto& aql_pkt : ret_pkt->after_krn_pkt)
    {
        aql_pkt.completion_signal.handle = 0;
    }

    return ret_pkt;
}

/**
 * Callback called by HSA interceptor when the kernel has completed processing.
 */
void
completed_cb(const std::shared_ptr<counter_callback_info>& info,
             const hsa::Queue&                             queue,
             hsa::rocprofiler_packet,
             const hsa::Queue::queue_info_session_t& session,
             inst_pkt_t&                             pkts)
{
    if(!info || pkts.empty()) return;

    std::shared_ptr<profile_config> prof_config;
    // Get the Profile Config
    std::unique_ptr<rocprofiler::hsa::AQLPacket> pkt = nullptr;
    info->packet_return_map.wlock([&](auto& data) {
        for(auto& [aql_pkt, _] : pkts)
        {
            const auto* profile = rocprofiler::common::get_val(data, aql_pkt.get());
            if(profile)
            {
                prof_config = *profile;
                data.erase(aql_pkt.get());
                pkt = std::move(aql_pkt);
                return;
            }
        }
    });

    if(!pkt) return;

    if(!pkt->empty)
    {
        hsa::profiler_serializer_kernel_completion_signal(session.queue.block_signal);
    }

    auto decoded_pkt = EvaluateAST::read_pkt(prof_config->pkt_generator.get(), *pkt);
    EvaluateAST::read_special_counters(
        *prof_config->agent, prof_config->required_special_counters, decoded_pkt);

    prof_config->packets.wlock([&](auto& pkt_vector) {
        if(pkt)
        {
            pkt_vector.emplace_back(std::move(pkt));
        }
    });

    common::container::small_vector<rocprofiler_record_counter_t, 128> out;
    rocprofiler::buffer::instance*                                     buf = nullptr;

    if(info->buffer)
    {
        buf = CHECK_NOTNULL(buffer::get_buffer(info->buffer->handle));
    }

    auto _corr_id_v =
        rocprofiler_correlation_id_t{.internal = 0, .external = context::null_user_data};
    if(const auto* _corr_id = session.correlation_id)
    {
        _corr_id_v.internal = _corr_id->internal;
        if(const auto* external =
               rocprofiler::common::get_val(session.extern_corr_ids, info->internal_context))
        {
            _corr_id_v.external = *external;
        }
    }

    for(auto& ast : prof_config->asts)
    {
        std::vector<std::unique_ptr<std::vector<rocprofiler_record_counter_t>>> cache;
        auto* ret = ast.evaluate(decoded_pkt, cache);
        CHECK(ret);
        ast.set_out_id(*ret);

        for(auto& val : *ret)
        {
            val.correlation_id = _corr_id_v;
            if(buf)
                buf->emplace(ROCPROFILER_BUFFER_CATEGORY_COUNTERS, 0, val);
            else
                out.push_back(val);
        }
    }

    if(!out.empty())
    {
        CHECK(info->record_callback);

        auto dispatch_data =
            common::init_public_api_struct(rocprofiler_profile_counting_dispatch_data_t{});

        const auto& kernel_dispatch_pkt = session.kernel_pkt.kernel_dispatch;

        dispatch_data.kernel_id            = session.kernel_id;
        dispatch_data.agent_id             = CHECK_NOTNULL(queue.get_agent().get_rocp_agent())->id;
        dispatch_data.queue_id             = queue.get_id();
        dispatch_data.correlation_id       = _corr_id_v;
        dispatch_data.private_segment_size = kernel_dispatch_pkt.private_segment_size;
        dispatch_data.group_segment_size   = kernel_dispatch_pkt.group_segment_size;
        dispatch_data.workgroup_size       = {kernel_dispatch_pkt.workgroup_size_x,
                                        kernel_dispatch_pkt.workgroup_size_y,
                                        kernel_dispatch_pkt.workgroup_size_z};
        dispatch_data.grid_size            = {kernel_dispatch_pkt.grid_size_x,
                                   kernel_dispatch_pkt.grid_size_y,
                                   kernel_dispatch_pkt.grid_size_z};

        info->record_callback(
            dispatch_data, out.data(), out.size(), session.user_data, info->record_callback_args);
    }
}

void
start_context(const context::context* ctx)
{
    if(!ctx || !ctx->counter_collection) return;

    auto& controller = hsa::get_queue_controller();

    // Only one thread should be attempting to enable/disable this context
    ctx->counter_collection->enabled.wlock([&](auto& enabled) {
        if(enabled) return;
        for(auto& cb : ctx->counter_collection->callbacks)
        {
            // Insert our callbacks into HSA Interceptor. This
            // turns on counter instrumentation.
            cb->queue_id = controller.add_callback(
                std::nullopt,
                [=](const hsa::Queue&                                               q,
                    const hsa::rocprofiler_packet&                                  kern_pkt,
                    uint64_t                                                        kernel_id,
                    rocprofiler_user_data_t*                                        user_data,
                    const hsa::Queue::queue_info_session_t::external_corr_id_map_t& extern_corr_ids,
                    const context::correlation_id* correlation_id) {
                    return queue_cb(
                        cb, q, kern_pkt, kernel_id, user_data, extern_corr_ids, correlation_id);
                },
                // Completion CB
                [=](const hsa::Queue&                       q,
                    hsa::rocprofiler_packet                 kern_pkt,
                    const hsa::Queue::queue_info_session_t& session,
                    inst_pkt_t& aql) { completed_cb(cb, q, kern_pkt, session, aql); });
        }
        enabled = true;
    });
}

void
stop_context(const context::context* ctx)
{
    if(!ctx || !ctx->counter_collection) return;

    auto& controller = hsa::get_queue_controller();

    ctx->counter_collection->enabled.wlock([&](auto& enabled) {
        if(!enabled) return;
        for(auto& cb : ctx->counter_collection->callbacks)
        {
            // Remove our callbacks from HSA's queue controller
            controller.remove_callback(cb->queue_id);
            cb->queue_id = -1;
        }
        enabled = false;
    });
}

bool
configure_buffered_dispatch(rocprofiler_context_id_t                         context_id,
                            rocprofiler_buffer_id_t                          buffer,
                            rocprofiler_profile_counting_dispatch_callback_t callback,
                            void*                                            callback_args)
{
    CHECK_NE(buffer.handle, 0);
    return get_controller().configure_dispatch(
        context_id, buffer, callback, callback_args, nullptr, nullptr);
}

bool
configure_callback_dispatch(rocprofiler_context_id_t                         context_id,
                            rocprofiler_profile_counting_dispatch_callback_t callback,
                            void*                                            callback_data_args,
                            rocprofiler_profile_counting_record_callback_t   record_callback,
                            void*                                            record_callback_args)
{
    return get_controller().configure_dispatch(context_id,
                                               {.handle = 0},
                                               callback,
                                               callback_data_args,
                                               record_callback,
                                               record_callback_args);
}

}  // namespace counters
}  // namespace rocprofiler
