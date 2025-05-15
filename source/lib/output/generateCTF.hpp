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

#include <babeltrace2/babeltrace.h>
#include <babeltrace2/graph/component-class.h>
#include <babeltrace2/graph/component.h>
#include <babeltrace2/graph/port.h>

#pragma once

#include "agent_info.hpp"
#include "buffered_output.hpp"
#include "metadata.hpp"
#include "output_config.hpp"
#include "output_stream.hpp"
#include "statistics.hpp"

namespace rocprofiler
{
namespace tool
{

    enum record_type_t {
        HIP_API_EXT = 0,

    };
struct ctf_output
{
    ctf_output(const output_config& cfg);
    ~ctf_output();

    void write_event_source_component(const char* source_name,
                                      bt_component_class_initialize_method_status (*init_method)(
                                          bt_self_component_source*,
                                          bt_self_component_source_configuration*,
                                          const bt_value*,
                                          void*),
                                      void (*finalize_method)(bt_self_component_source*),
                                      bt_message_iterator_class_next_method_status (*next_method)(
                                          bt_self_message_iterator* self_message_iterator,
                                          bt_message_array_const    msgs,
                                          uint64_t                  capacity,
                                          uint64_t*                 count), void* data);
    void close();

    bool hip_api_ext_initialized{false};

    std::deque<rocprofiler_buffer_tracing_hip_api_ext_record_t>* hip_api_data = nullptr;

private:
    bt_graph*            graph;
    bt_component*        ctf_writer_comp;
    bt_message_iterator* msg_iter;
    bt_event_class*      event_class;
};

ctf_output
open_ctf_stream(const output_config& cfg);

void
close_ctf_stream(ctf_output& ctf_out);

void
write_ctf(ctf_output&                                                        ctf_out,
          const output_config&                                               cfg,
          const metadata&                                                    tool_metadata,
          uint64_t                                                           pid,
          const std::vector<agent_info>&                                     agent_data,
          std::deque<rocprofiler_buffer_tracing_hip_api_ext_record_t>*       hip_api_data,
          std::deque<rocprofiler_buffer_tracing_hsa_api_record_t>*           hsa_api_data,
          std::deque<tool_buffer_tracing_kernel_dispatch_ext_record_t>*      kernel_dispatch_data,
          std::deque<tool_buffer_tracing_memory_copy_ext_record_t>*          memory_copy_data,
          std::deque<rocprofiler_buffer_tracing_marker_api_record_t>*        marker_api_data,
          std::deque<rocprofiler_buffer_tracing_scratch_memory_record_t>*    scratch_memory_data,
          std::deque<rocprofiler_buffer_tracing_rccl_api_record_t>*          rccl_api_data,
          std::deque<rocprofiler_buffer_tracing_memory_allocation_record_t>* memory_allocation_data,
          std::deque<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>* rocdecode_api_data,
          std::deque<rocprofiler_buffer_tracing_rocjpeg_api_record_t>*       rocjpeg_api_data);

}  // namespace tool
}  // namespace rocprofiler
