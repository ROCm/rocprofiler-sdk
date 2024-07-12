// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "pcs.hpp"
#include "utils.hpp"

#include "common/defines.hpp"

#include <cassert>
#include <cstdio>
#include <iomanip>
#include <memory>
#include <sstream>

namespace client
{
namespace pcs
{
// TODO: Since this is used only within the `tool_init`,
// we are safe using static constructor.
// It would be nice to make this consistent with the `buffer_ids`.
tool_agent_info_vec_t gpu_agents;
// The reason for using raw pointers is the following.
// Sometimes, statically created objects of the client::pcs
// namespace might be freed prior to the `tool_fini`,
// meaning `buffer_ids` become unusable inside `tool_fini`.
// Instead, use raw pointers to control objects deallocation time.
// TODO: The approach with exporting raw pointers outside of the
// `pcs` namespace is a temporary solution.
// Instead, it would be better to encapsulate `buffer_ids` inside the
// `pcs` namespace and export functions for registering/flushing/destroying buffers.
pc_sampling_buffer_id_vec_t* buffer_ids = nullptr;

void
init()
{
    buffer_ids = new pc_sampling_buffer_id_vec_t();
}

void
fini()
{
    // Clear the data
    buffer_ids->clear();
    delete buffer_ids;
}

pc_sampling_buffer_id_vec_t*
get_pc_sampling_buffer_ids()
{
    return buffer_ids;
}

rocprofiler_status_t
find_all_gpu_agents_supporting_pc_sampling_impl(rocprofiler_agent_version_t version,
                                                const void**                agents,
                                                size_t                      num_agents,
                                                void*                       user_data)
{
    assert(version == ROCPROFILER_AGENT_INFO_VERSION_0);
    // user_data represent the pointer to the array where gpu_agent will be stored
    if(!user_data) return ROCPROFILER_STATUS_ERROR;

    std::stringstream ss;

    auto* _out_agents = static_cast<tool_agent_info_vec_t*>(user_data);
    auto* _agents     = reinterpret_cast<const rocprofiler_agent_t**>(agents);
    for(size_t i = 0; i < num_agents; i++)
    {
        if(_agents[i]->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            // Instantiate the tool_agent_info.
            // Store pointer to the rocprofiler_agent_t and instatiate a vector of
            // available configurations.
            // Move the ownership to the _out_agents
            auto tool_gpu_agent           = std::make_unique<tool_agent_info>();
            tool_gpu_agent->agent_id      = _agents[i]->id;
            tool_gpu_agent->avail_configs = std::make_unique<avail_configs_vec_t>();
            tool_gpu_agent->agent         = _agents[i];
            // Check if the GPU agent supports PC sampling. If so, add it to the
            // output list `_out_agents`.
            if(query_avail_configs_for_agent(tool_gpu_agent.get()))
                _out_agents->push_back(std::move(tool_gpu_agent));
        }

        ss << "[" << __FUNCTION__ << "] " << _agents[i]->name << " :: "
           << "id=" << _agents[i]->id.handle << ", "
           << "type=" << _agents[i]->type << "\n";
    }

    *utils::get_output_stream() << ss.str() << std::endl;

    return ROCPROFILER_STATUS_SUCCESS;
}

void
find_all_gpu_agents_supporting_pc_sampling()
{
    // This function returns the all gpu agents supporting some kind of PC sampling
    ROCPROFILER_CHECK(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           &find_all_gpu_agents_supporting_pc_sampling_impl,
                                           sizeof(rocprofiler_agent_t),
                                           static_cast<void*>(&gpu_agents)));
}

/**
 * @brief The function queries available PC sampling configurations.
 * If there is at least one available configuration, it returns true.
 * Otherwise, this function returns false to indicate the agent does
 * not support PC sampling.
 */
bool
query_avail_configs_for_agent(tool_agent_info* agent_info)
{
    // Clear the available configurations vector
    agent_info->avail_configs->clear();

    auto cb = [](const rocprofiler_pc_sampling_configuration_t* configs,
                 size_t                                         num_config,
                 void*                                          user_data) {
        auto* avail_configs = static_cast<avail_configs_vec_t*>(user_data);
        for(size_t i = 0; i < num_config; i++)
        {
            avail_configs->emplace_back(configs[i]);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    auto status = rocprofiler_query_pc_sampling_agent_configurations(
        agent_info->agent_id, cb, agent_info->avail_configs.get());

    std::stringstream ss;

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        // The query operation failed, so consider the PC sampling is unsupported at the agent.
        // This can happen if the PC sampling service is invoked within the ROCgdb.
        ss << "Querying PC sampling capabilities failed with status=" << status
           << " :: " << rocprofiler_get_status_string(status) << std::endl;
        *utils::get_output_stream() << ss.str() << std::endl;
        return false;
    }
    else if(agent_info->avail_configs->empty())
    {
        // No available configuration at the moment, so mark the PC sampling as unsupported.
        return false;
    }

    ss << "The agent with the id: " << agent_info->agent_id.handle << " supports the "
       << agent_info->avail_configs->size() << " configurations: " << std::endl;
    size_t ind = 0;
    for(auto& cfg : *agent_info->avail_configs)
    {
        ss << "(" << ++ind << ".) "
           << "method: " << cfg.method << ", "
           << "unit: " << cfg.unit << ", "
           << "min_interval: " << cfg.min_interval << ", "
           << "max_interval: " << cfg.max_interval << ", "
           << "flags: " << std::hex << cfg.flags << std::dec << std::endl;
    }

    *utils::get_output_stream() << ss.str() << std::flush;

    return true;
}

void
configure_pc_sampling_prefer_stochastic(tool_agent_info*         agent_info,
                                        rocprofiler_context_id_t context_id,
                                        rocprofiler_buffer_id_t  buffer_id)
{
    int    failures = 10;
    size_t interval = 0;
    do
    {
        // Update the list of available configurations
        auto success = query_avail_configs_for_agent(agent_info);
        if(!success)
        {
            // An error occured while querying PC sampling capabilities,
            // so avoid trying configuring PC sampling service.
            // Instead return false to indicated a failure.
            ROCPROFILER_CHECK(ROCPROFILER_STATUS_ERROR);
        }

        const rocprofiler_pc_sampling_configuration_t* first_host_trap_config  = nullptr;
        const rocprofiler_pc_sampling_configuration_t* first_stochastic_config = nullptr;
        // Search until encountering on the stochastic configuration, if any.
        // Otherwise, use the host trap config
        for(auto const& cfg : *agent_info->avail_configs)
        {
            if(cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
            {
                first_stochastic_config = &cfg;
                break;
            }
            else if(!first_host_trap_config &&
                    cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
            {
                first_host_trap_config = &cfg;
            }
        }

        // Check if the stochastic config is found. Use host trap config otherwise.
        const rocprofiler_pc_sampling_configuration_t* picked_cfg =
            (first_stochastic_config != nullptr) ? first_stochastic_config : first_host_trap_config;

        if(picked_cfg->min_interval == picked_cfg->max_interval)
        {
            // Another process already configured PC sampling, so use the intreval it set up.
            interval = picked_cfg->min_interval;
        }
        else
        {
            interval = 10000;
        }

        auto status = rocprofiler_configure_pc_sampling_service(context_id,
                                                                agent_info->agent_id,
                                                                picked_cfg->method,
                                                                picked_cfg->unit,
                                                                interval,
                                                                buffer_id);
        if(status == ROCPROFILER_STATUS_SUCCESS)
        {
            *utils::get_output_stream()
                << ">>> We chose PC sampling interval: " << interval
                << " on the agent: " << agent_info->agent->id.handle << std::endl;
            return;
        }
        else if(status != ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE)
        {
            ROCPROFILER_CHECK(status);
        }
        // status ==  ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE
        // means another process P2 already configured PC sampling.
        // Query available configurations again and receive the configurations picked by P2.
        // However, if P2 destroys PC sampling service after query function finished,
        // but before the `rocprofiler_configure_pc_sampling_service` is called,
        // then the `rocprofiler_configure_pc_sampling_service` will fail again.
        // The process P1 executing this loop can spin wait (starve) if it is unlucky enough
        // to always be interuppted by some other process P2 that creates/destroys
        // PC sampling service on the same device while P1 is executing the code
        // after the `query_avail_configs_for_agent` and
        // before the `rocprofiler_configure_pc_sampling_service`.
        // This should happen very rarely, but just to be sure, we introduce a counter `failures`
        // that will allow certain amount of failures to process P1.
    } while(--failures);

    // The process failed too many times configuring PC sampling,
    // report this to user;
    ROCPROFILER_CHECK(ROCPROFILER_STATUS_ERROR);
}

void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t /*context_id*/,
                                 rocprofiler_buffer_id_t /*buffer_id*/,
                                 rocprofiler_record_header_t** headers,
                                 size_t                        num_headers,
                                 void* /*data*/,
                                 uint64_t drop_count)
{
    std::stringstream ss;
    ss << "The number of delivered samples is: " << num_headers << ", "
       << "while the number of dropped samples is: " << drop_count << std::endl;

    for(size_t i = 0; i < num_headers; i++)
    {
        auto* cur_header = headers[i];

        if(cur_header == nullptr)
        {
            throw std::runtime_error{
                "rocprofiler provided a null pointer to header. this should never happen"};
        }
        else if(cur_header->hash !=
                rocprofiler_record_header_compute_hash(cur_header->category, cur_header->kind))
        {
            throw std::runtime_error{"rocprofiler_record_header_t (category | kind) != hash"};
        }
        else if(cur_header->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
        {
            if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_SAMPLE)
            {
                auto* pc_sample =
                    static_cast<rocprofiler_pc_sampling_record_t*>(cur_header->payload);

                ss << "pc: " << std::hex << pc_sample->pc << ", "
                   << "timestamp: " << std::dec << pc_sample->timestamp << ", "
                   << "exec: " << std::hex << std::setw(16) << pc_sample->exec_mask << ", "
                   << "workgroup_id_(x=" << std::dec << std::setw(5) << pc_sample->workgroup_id.x
                   << ", "
                   << "y=" << std::setw(5) << pc_sample->workgroup_id.y << ", "
                   << "z=" << std::setw(5) << pc_sample->workgroup_id.z << "), "
                   << "wave_id: " << std::setw(2) << static_cast<unsigned int>(pc_sample->wave_id)
                   << ", "
                   << "cu_id: " << pc_sample->hw_id << ", "
                   << "correlation: {internal=" << std::setw(7)
                   << pc_sample->correlation_id.internal << ", "
                   << "external=" << std::setw(5) << pc_sample->correlation_id.external.value << "}"
                   << std::endl;
            }
            else if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_CODE_OBJECT_LOAD_MARKER)
            {
                auto* marker = static_cast<rocprofiler_pc_sampling_code_object_load_marker_t*>(
                    cur_header->payload);
                ss << "code object loading: " << marker->code_object_id << std::endl;
            }
            else if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_CODE_OBJECT_UNLOAD_MARKER)
            {
                auto* marker = static_cast<rocprofiler_pc_sampling_code_object_unload_marker_t*>(
                    cur_header->payload);
                ss << "code object unloading: " << marker->code_object_id << std::endl;
            }
        }
        else
        {
            throw std::runtime_error{"unexpected rocprofiler_record_header_t category + kind"};
        }
    }

    *utils::get_output_stream() << ss.str() << std::endl;
}
}  // namespace pcs
}  // namespace client
