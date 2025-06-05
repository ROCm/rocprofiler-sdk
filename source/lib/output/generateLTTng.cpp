// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "generateLTTng.hpp"
#include "output_stream.hpp"
#include "statistics.hpp"
#include "timestamps.hpp"

#include "lttng/rocprofiler_sdk_trace_provider.hpp"

#include "lib/common/string_entry.hpp"
#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>

#include <utility>

namespace rocprofiler
{
namespace tool
{
struct args_info
{
    std::string type  = {};
    std::string value = {};
};
int
iterate_args_callback(rocprofiler_buffer_tracing_kind_t /*kind*/,
                      rocprofiler_tracing_operation_t /*operation*/,
                      uint32_t /*arg_number*/,
                      const void* const /*arg_value_addr*/,
                      int32_t /*arg_indirection_count*/,
                      const char* arg_type,
                      const char* arg_name,
                      const char* arg_value_str,
                      void*       data)
{
    ROCP_FATAL_IF(data == nullptr) << "nullptr to data for iterate_args_callback";

    auto* _data = static_cast<std::vector<args_info>*>(data);
    if(arg_type && arg_name && arg_value_str)
        _data->emplace_back(args_info{arg_name, arg_value_str});
    return 0;
}

void
write_lttng(const output_config& /*cfg*/,
            const metadata&                                                 tool_metadata,
            uint64_t                                                        pid,
            const std::vector<agent_info>&                                  agent_data,
            std::deque<rocprofiler_buffer_tracing_hip_api_ext_record_t>*    hip_api_data,
            std::deque<rocprofiler_buffer_tracing_hsa_api_record_t>*        hsa_api_data,
            std::deque<tool_buffer_tracing_kernel_dispatch_ext_record_t>*   kernel_dispatch_data,
            std::deque<tool_buffer_tracing_memory_copy_ext_record_t>*       memory_copy_data,
            std::deque<rocprofiler_buffer_tracing_marker_api_record_t>*     marker_api_data,
            std::deque<rocprofiler_buffer_tracing_scratch_memory_record_t>* scratch_memory_data,
            std::deque<rocprofiler_buffer_tracing_rccl_api_record_t>*       rccl_api_data,
            std::deque<tool_buffer_tracing_memory_allocation_ext_record_t>* memory_allocation_data,
            std::deque<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>* rocdecode_api_data,
            std::deque<rocprofiler_buffer_tracing_rocjpeg_api_record_t>*       rocjpeg_api_data)
{
    auto buffer_names     = sdk::get_buffer_tracing_names();
    auto callbk_name_info = sdk::get_callback_tracing_names();

    for(const auto& _agent_info : agent_data)
    {
        tracepoint(rocprofv3_trace,
                   agents_info,
                   pid,
                   _agent_info.type,
                   _agent_info.node_id,
                   _agent_info.logical_node_id,
                   _agent_info.logical_node_type_id,
                   _agent_info.gpu_id,
                   _agent_info.name,
                   _agent_info.vendor_name,
                   _agent_info.product_name,
                   _agent_info.model_name);
    }
    // Loop over each deque and call write_event for each record
    if(hip_api_data)
    {
        for(auto hip_api_record : *hip_api_data)
        {
            auto api_name = buffer_names.at(hip_api_record.kind, hip_api_record.operation);

            std::vector<const char*> args_types  = {};
            std::vector<const char*> args_values = {};
            std::vector<args_info>   args        = {};
            {
                auto _record = rocprofiler_record_header_t{
                    .hash = rocprofiler_record_header_compute_hash(
                        ROCPROFILER_BUFFER_CATEGORY_TRACING, hip_api_record.kind),
                    .payload = &hip_api_record};

                rocprofiler_iterate_buffer_tracing_record_args(
                    _record, iterate_args_callback, &args);

                for(const auto& arg : args)
                {
                    args_types.push_back(arg.type.c_str());
                    args_values.push_back(arg.value.c_str());
                }
            }

            tracepoint(rocprofv3_trace,
                       hip_api,
                       pid,
                       api_name.data(),
                       hip_api_record.correlation_id.ancestor,
                       hip_api_record.correlation_id.internal,
                       hip_api_record.start_timestamp,
                       hip_api_record.end_timestamp,
                       hip_api_record.thread_id,
                       args_types.data(),
                       args_values.data(),
                       args.size());
        }
    }
    if(hsa_api_data)
    {
        for(auto hsa_api_record : *hsa_api_data)
        {
            auto api_name = buffer_names.at(hsa_api_record.kind, hsa_api_record.operation);

            tracepoint(rocprofv3_trace,
                       hsa_api,
                       pid,
                       api_name.data(),
                       hsa_api_record.correlation_id.ancestor,
                       hsa_api_record.correlation_id.internal,
                       hsa_api_record.start_timestamp,
                       hsa_api_record.end_timestamp,
                       hsa_api_record.thread_id);
        }
    }
    if(kernel_dispatch_data)
    {
        for(auto kernel_dispatch_record : *kernel_dispatch_data)
        {
            auto name =
                tool_metadata.get_kernel_name(kernel_dispatch_record.dispatch_info.kernel_id,
                                              kernel_dispatch_record.correlation_id.external.value);
            const auto& agent =
                tool_metadata.get_agent(kernel_dispatch_record.dispatch_info.agent_id);

            tracepoint(rocprofv3_trace,
                       kernel_dispatch,
                       pid,
                       kernel_dispatch_record.correlation_id.internal,
                       kernel_dispatch_record.start_timestamp,
                       kernel_dispatch_record.end_timestamp,
                       kernel_dispatch_record.thread_id,
                       agent->node_id,
                       kernel_dispatch_record.dispatch_info.queue_id.handle,
                       kernel_dispatch_record.stream_id.handle,
                       name.data());
        }
    }
    if(memory_copy_data)
    {
        for(auto memory_copy_record : *memory_copy_data)
        {
            const auto& src_agent = tool_metadata.get_agent(memory_copy_record.src_agent_id);
            const auto& dst_agent = tool_metadata.get_agent(memory_copy_record.dst_agent_id);

            tracepoint(rocprofv3_trace,
                       memory_copy,
                       pid,
                       memory_copy_record.operation,
                       memory_copy_record.correlation_id.internal,
                       memory_copy_record.start_timestamp,
                       memory_copy_record.end_timestamp,
                       memory_copy_record.thread_id,
                       src_agent->node_id,
                       dst_agent->node_id,
                       memory_copy_record.stream_id.handle,
                       memory_copy_record.bytes);
        }
    }
    if(marker_api_data)
    {
        for(auto marker_api_record : *marker_api_data)
        {
            auto name =
                (marker_api_record.kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API &&
                 marker_api_record.operation != ROCPROFILER_MARKER_CORE_API_ID_roctxGetThreadId)
                    ? tool_metadata.get_marker_message(marker_api_record.correlation_id.internal)
                    : buffer_names.at(marker_api_record.kind, marker_api_record.operation);
            tracepoint(rocprofv3_trace,
                       marker_api,
                       pid,
                       marker_api_record.operation,
                       marker_api_record.correlation_id.ancestor,
                       marker_api_record.correlation_id.internal,
                       marker_api_record.start_timestamp,
                       marker_api_record.end_timestamp,
                       marker_api_record.thread_id,
                       name.data());
        }
    }
    if(scratch_memory_data)
    {
        for(auto scratch_memory_record : *scratch_memory_data)
        {
            const auto& agent = tool_metadata.get_agent(scratch_memory_record.agent_id);

            tracepoint(rocprofv3_trace,
                       scratch_memory,
                       pid,
                       scratch_memory_record.operation,
                       scratch_memory_record.correlation_id.ancestor,
                       scratch_memory_record.correlation_id.internal,
                       scratch_memory_record.start_timestamp,
                       scratch_memory_record.end_timestamp,
                       scratch_memory_record.thread_id,
                       agent->node_id,
                       scratch_memory_record.queue_id.handle,
                       scratch_memory_record.flags);
        }
    }
    if(rccl_api_data)
    {
        for(auto rccl_api_record : *rccl_api_data)
        {
            auto name = buffer_names.at(rccl_api_record.kind, rccl_api_record.operation);
            tracepoint(rocprofv3_trace,
                       rccl_api,
                       pid,
                       rccl_api_record.correlation_id.ancestor,
                       rccl_api_record.correlation_id.internal,
                       rccl_api_record.start_timestamp,
                       rccl_api_record.end_timestamp,
                       rccl_api_record.thread_id,
                       name.data());
        }
    }
    if(memory_allocation_data)
    {
        for(auto memory_allocation_record : *memory_allocation_data)
        {
            const auto& agent = tool_metadata.get_agent(memory_allocation_record.agent_id);

            tracepoint(rocprofv3_trace,
                       memory_allocation,
                       pid,
                       memory_allocation_record.operation,
                       memory_allocation_record.correlation_id.internal,
                       memory_allocation_record.start_timestamp,
                       memory_allocation_record.end_timestamp,
                       memory_allocation_record.thread_id,
                       agent->node_id,
                       memory_allocation_record.stream_id.handle,
                       memory_allocation_record.address,
                       memory_allocation_record.allocation_size);
        }
    }
    if(rocdecode_api_data)
    {
        for(auto rocdecode_api_record : *rocdecode_api_data)
        {
            auto name = buffer_names.at(rocdecode_api_record.kind, rocdecode_api_record.operation);

            std::vector<const char*> args_types  = {};
            std::vector<const char*> args_values = {};
            std::vector<args_info>   args        = {};
            {
                auto _record = rocprofiler_record_header_t{
                    .hash = rocprofiler_record_header_compute_hash(
                        ROCPROFILER_BUFFER_CATEGORY_TRACING, rocdecode_api_record.kind),
                    .payload = &rocdecode_api_record};

                rocprofiler_iterate_buffer_tracing_record_args(
                    _record, iterate_args_callback, &args);

                for(const auto& arg : args)
                {
                    args_types.push_back(arg.type.c_str());
                    args_values.push_back(arg.value.c_str());
                }
            }

            tracepoint(rocprofv3_trace,
                       rocdecode_api,
                       pid,
                       rocdecode_api_record.correlation_id.ancestor,
                       rocdecode_api_record.correlation_id.internal,
                       rocdecode_api_record.start_timestamp,
                       rocdecode_api_record.end_timestamp,
                       rocdecode_api_record.thread_id,
                       name.data(),
                       args_types.data(),
                       args_values.data(),
                       args.size());
        }
    }
    if(rocjpeg_api_data)
    {
        for(auto rocjpeg_api_record : *rocjpeg_api_data)
        {
            auto name = buffer_names.at(rocjpeg_api_record.kind, rocjpeg_api_record.operation);

            tracepoint(rocprofv3_trace,
                       rocjpeg_api,
                       pid,
                       rocjpeg_api_record.correlation_id.ancestor,
                       rocjpeg_api_record.correlation_id.internal,
                       rocjpeg_api_record.start_timestamp,
                       rocjpeg_api_record.end_timestamp,
                       rocjpeg_api_record.thread_id,
                       name.data());
        }
    }
}

}  // namespace tool
}  // namespace rocprofiler
