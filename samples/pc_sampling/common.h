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

#pragma once

#include <rocprofiler/rocprofiler.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

constexpr size_t       BUFFER_SIZE_BYTES = 4096;
constexpr size_t       WATERMARK         = (BUFFER_SIZE_BYTES / 2);
const std::string_view MI200_NAME        = "gfx90a";

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            puts(#result " failed");                                                               \
        }                                                                                          \
    }

// We might want to test the calls that fails
// e.g. calling `rocprofiler_configure_pc_sampling_service `
// after previous initialization.
#define ROCPROFILER_CALL_FAILS(result, msg)                                                        \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS == ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            puts(#result " succeeded");                                                            \
        }                                                                                          \
    }

static rocprofiler_status_t
find_first_gpu_agent_impl(const rocprofiler_agent_t** agents, size_t num_agents, void* data)
{
    // data is required
    if(!data) return ROCPROFILER_STATUS_ERROR;

    auto* _out_agent = static_cast<rocprofiler_agent_t*>(data);
    // find the first GPU agent
    for(size_t i = 0; i < num_agents; i++)
    {
        if(agents[i]->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            *_out_agent = *agents[i];
            printf("[%s] %s :: id=%u, type=%i, num pc sample configs=%zu\n",
                   __FUNCTION__,
                   _out_agent->name,
                   _out_agent->node_id,
                   _out_agent->type,
                   _out_agent->num_pc_sampling_configs);
            return ROCPROFILER_STATUS_SUCCESS;
        }
        else
        {
            printf("[%s] %s :: id=%u, type=%i, num pc sample configs=%zu\n",
                   __FUNCTION__,
                   agents[i]->name,
                   agents[i]->node_id,
                   agents[i]->type,
                   agents[i]->num_pc_sampling_configs);
        }
    }
    return ROCPROFILER_STATUS_ERROR;
}

static std::optional<rocprofiler_agent_t>
find_first_gpu_agent()
{
    // This function returns the first gpu agent it encounters.
    // TODO: write the better function querying information about the agent,
    //   and return if the agent is MI200.
    rocprofiler_agent_t gpu_agent;

    auto status = rocprofiler_query_available_agents(
        &find_first_gpu_agent_impl, sizeof(rocprofiler_agent_t), static_cast<void*>(&gpu_agent));

    if(status != ROCPROFILER_STATUS_SUCCESS) return std::nullopt;

    return gpu_agent;
}

static void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t /*context_id*/,
                                 rocprofiler_buffer_id_t /*buffer_id*/,
                                 rocprofiler_record_header_t** headers,
                                 size_t                        num_headers,
                                 void* /*data*/,
                                 uint64_t drop_count)
{
    // Vladimir: I am not sure if this is the right way of iterating over PC sampling records.
    printf(
        "The number of delivered samples is: %zu, while the number of dropped samples is: %lu.\n",
        num_headers,
        drop_count);

    for(size_t i = 0; i < num_headers; i++)
    {
        auto* cur_header = headers[i];
        if(cur_header->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
        {
            auto* pc_sample = static_cast<rocprofiler_pc_sampling_record_t*>(cur_header->payload);
            printf("--- pc: %lx, dispatch_id: %lx, timestamp: %lu, hardware_id: %lu\n",
                   pc_sample->pc,
                   pc_sample->dispatch_id,
                   pc_sample->timestamp,
                   pc_sample->hardware_id);
            // Vladimir: How to parse the remaining part of the `rocprofiler_pc_sampling_record_t`
            // struct?
        }
    }
    // Vladimr: We might want to add somewhere in the documentation that headars actually contain PC
    // samples.
}

static void
run_HIP_app()
{
    // TODO: provide the simple HIP app
}
