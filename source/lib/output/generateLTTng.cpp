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
void
write_lttng(const output_config&                                            /*cfg*/,
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

    for(auto& _agent_info : agent_data)
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
        for (auto hip_api_record : *hip_api_data)
        {
            // --- Prepare raw args data ---
            const uint8_t* raw_args_ptr = reinterpret_cast<const uint8_t*>(&hip_api_record.args);
            uint64_t       raw_args_size_val = static_cast<uint64_t>(
                sizeof(hip_api_record.args));  // Or sizeof(rocprofiler_hip_api_args_t)

            auto api_name = buffer_names.at(hip_api_record.kind, hip_api_record.operation);

            tracepoint(rocprofv3_trace,
                       hip_api,
                       pid,
                       api_name.data(),
                       hip_api_record.correlation_id.ancestor,
                       hip_api_record.correlation_id.internal,
                       hip_api_record.start_timestamp,
                       hip_api_record.end_timestamp,
                       hip_api_record.thread_id,
                       raw_args_ptr,
                       raw_args_size_val  // Sending args as bytes
            );
        }
    }
    if(hsa_api_data)
    {
        for (auto hsa_api_record : *hsa_api_data)
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
                       hsa_api_record.thread_id
            );
        }
    }
    if(kernel_dispatch_data)
    {
        for(auto kernel_dispatch_record : *kernel_dispatch_data) {
            auto name =
                tool_metadata.get_kernel_name(kernel_dispatch_record.dispatch_info.kernel_id, kernel_dispatch_record.correlation_id.external.value);

            tracepoint(rocprofv3_trace,
                       kernel_dispatch,
                       pid,
                       kernel_dispatch_record.correlation_id.internal,
                       kernel_dispatch_record.start_timestamp,
                       kernel_dispatch_record.end_timestamp,
                       kernel_dispatch_record.thread_id,
                       kernel_dispatch_record.dispatch_info.agent_id.handle,
                       kernel_dispatch_record.dispatch_info.queue_id.handle,
                       kernel_dispatch_record.stream_id.handle,
                       name.data()
            );
        }
    }
    if(memory_copy_data)
    {
        for(auto memory_copy_record : *memory_copy_data) {
            tracepoint(rocprofv3_trace,
                       memory_copy,
                       pid,
                       memory_copy_record.operation,
                       memory_copy_record.correlation_id.internal,
                       memory_copy_record.start_timestamp,
                       memory_copy_record.end_timestamp,
                       memory_copy_record.thread_id,
                       memory_copy_record.src_agent_id.handle,
                       memory_copy_record.dst_agent_id.handle,
                       memory_copy_record.stream_id.handle,
                       memory_copy_record.bytes
            );
        }
    }
    if(marker_api_data)
    {
        // ctf_out.write_event_source_component("marker_api_source",
        //                              marker_api_source_init,
        //                              marker_api_source_finalize,
        //                              marker_api_source_next,
        //                              marker_api_data);
    }
    if(scratch_memory_data)
    {
        // ctf_out.write_event_source_component("scratch_memory_source",
        //                              scratch_memory_source_init,
        //                              scratch_memory_source_finalize,
        //                              scratch_memory_source_next,
        //                              scratch_memory_data);
    }
    if(rccl_api_data)
    {
        // ctf_out.write_event_source_component("rccl_api_source",
        //                              rccl_api_source_init,
        //                              rccl_api_source_finalize,
        //                              rccl_api_source_next,
        //                              rccl_api_data);
    }
    if(memory_allocation_data)
    {
        // ctf_out.write_event_source_component("memory_allocation_source",
        //                              memory_allocation_source_init,
        //                              memory_allocation_source_finalize,
        //                              memory_allocation_source_next,
        //                              memory_allocation_data);
    }
    if(rocdecode_api_data)
    {
        // ctf_out.write_event_source_component("rocdecode_api_source",
        //                              rocdecode_api_source_init,
        //                              rocdecode_api_source_finalize,
        //                              rocdecode_api_source_next,
        //                              rocdecode_api_data);
    }
    if(rocjpeg_api_data)
    {
        // ctf_out.write_event_source_component("rocjpeg_api_source",
        //                              rocjpeg_api_source_init,
        //                              rocjpeg_api_source_finalize,
        //                              rocjpeg_api_source_next,
        //                              rocjpeg_api_data);
    }
}

}  // namespace tool
}  // namespace rocprofiler
