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

#include "generateCTF.hpp"
#include "output_stream.hpp"
#include "statistics.hpp"
#include "timestamps.hpp"

#include "lib/common/filesystem.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>

#include <utility>

namespace rocprofiler
{
namespace tool
{
namespace fs = common::filesystem;

ctf_output::ctf_output(const output_config& cfg)
{
    auto _filename = get_output_filename(cfg, "results", std::string_view{});
    auto _filepath = fs::path{_filename};

    // Initialize Babeltrace 2 logging
    bt_logging_set_global_level(BT_LOGGING_LEVEL_DEBUG);

    // Create a Babeltrace 2 component graph
    graph = bt_graph_create(0);  // Pass 0 for default MIP version
    if(!graph)
    {
        fprintf(stderr, "Failed to create Babeltrace 2 graph\n");
        return;
    }

    // Prepare parameters for the CTF sink
    bt_value* params = bt_value_map_create();
    if(!params)
    {
        fprintf(stderr, "Failed to create parameters map\n");
        bt_graph_put_ref(graph);
        return;
    }
    bt_value* out_path = bt_value_string_create_init(_filepath.c_str());
    bt_value_map_insert_entry(params, "path", out_path);

    // Add the CTF writer sink component from the ctf plugin
    bt_component* ctf_writer_comp = nullptr;
    // Discover the plugin and component class
    const bt_plugin_set*      plugin_set  = nullptr;
    bt_plugin_find_all_status find_status = bt_plugin_find_all(BT_TRUE,  // find_in_std_env_var
                                                               BT_TRUE,  // find_in_user_dir
                                                               BT_TRUE,  // find_in_system_dir
                                                               BT_TRUE,  // find_in_static
                                                               BT_TRUE,  // find_in_static_linked
                                                               &plugin_set);

    if(find_status != BT_PLUGIN_FIND_ALL_STATUS_OK || !plugin_set)
    {
        fprintf(stderr, "Failed to find plugins\n");
        bt_graph_put_ref(graph);
        return;
    }

    const bt_plugin* ctf_plugin   = nullptr;
    uint64_t         plugin_count = bt_plugin_set_get_plugin_count(plugin_set);
    for(uint64_t i = 0; i < plugin_count; ++i)
    {
        const bt_plugin* plugin = bt_plugin_set_borrow_plugin_by_index_const(plugin_set, i);
        if(plugin && strcmp(bt_plugin_get_name(plugin), "ctf") == 0)
        {
            ctf_plugin = plugin;
            break;
        }
    }
    if(!ctf_plugin)
    {
        fprintf(stderr, "Failed to find 'ctf' plugin\n");
        bt_graph_put_ref(graph);
        return;
    }

    const bt_component_class_sink* ctf_sink_class =
        bt_plugin_borrow_sink_component_class_by_name_const(ctf_plugin, "fs");
    if(!ctf_sink_class)
    {
        fprintf(stderr, "Failed to find 'fs' sink class in 'ctf' plugin\n");
        bt_graph_put_ref(graph);
        return;
    }

    // Now add the sink component
    bt_graph_add_component_status status =
        bt_graph_add_sink_component(graph,
                                    ctf_sink_class,
                                    "ctf_writer",
                                    params,
                                    BT_LOGGING_LEVEL_NONE,
                                    (const bt_component_sink**) &ctf_writer_comp);
    bt_plugin_set_put_ref(plugin_set);
    bt_value_put_ref(params);

    if(status != BT_GRAPH_ADD_COMPONENT_STATUS_OK || !ctf_writer_comp)
    {
        fprintf(stderr, "Failed to add CTF writer sink component to the graph\n");
        bt_graph_put_ref(graph);
        return;
    }
}

ctf_output::~ctf_output()
{
    close();
}

void ctf_output::close() {
    // Cleanup
    bt_graph_put_ref(graph);
}

template <typename record_type>
void ctf_output::write_event(const record_type& event) {
    write_event_impl(event, ctf_event_tag<record_type>{});
}

// ---- HIP API EXT ----
void ctf_output::write_event_impl(
    const rocprofiler_buffer_tracing_hip_api_ext_record_t& event,
    ctf_event_tag<rocprofiler_buffer_tracing_hip_api_ext_record_t>)
{
    // Example: Extract fields
    auto kind = event.kind;
    auto op   = event.operation;
    auto corr = event.correlation_id;
    auto start = event.start_timestamp;
    auto end   = event.end_timestamp;
    auto tid   = event.thread_id;
    // event.args, event.retval

    // TODO: Use Babeltrace 2 API to create a message/event and set these fields
}

// ---- HSA API ----
void ctf_output::write_event_impl(
    const rocprofiler_buffer_tracing_hsa_api_record_t& event,
    ctf_event_tag<rocprofiler_buffer_tracing_hsa_api_record_t>)
{
    auto kind = event.kind;
    auto op   = event.operation;
    auto corr = event.correlation_id;
    auto start = event.start_timestamp;
    auto end   = event.end_timestamp;
    auto tid   = event.thread_id;

    // TODO: Use Babeltrace 2 API to create a message/event and set these fields
}

// ---- Kernel Dispatch ----
void ctf_output::write_event_impl(
    const tool_buffer_tracing_kernel_dispatch_ext_record_t& event,
    ctf_event_tag<tool_buffer_tracing_kernel_dispatch_ext_record_t>)
{
    auto& base = static_cast<const rocprofiler_buffer_tracing_kernel_dispatch_record_t&>(event);
    auto kind = base.kind;
    auto op   = base.operation;
    auto corr = base.correlation_id;
    auto tid  = base.thread_id;
    auto start = base.start_timestamp;
    auto end   = base.end_timestamp;
    auto dispatch_info = base.dispatch_info;
    auto stream_id = event.stream_id;
    auto kernel_rename_val = event.kernel_rename_val;

    // TODO: Use Babeltrace 2 API to create a message/event and set these fields
}

// ---- Memory Copy ----
void ctf_output::write_event_impl(
    const tool_buffer_tracing_memory_copy_ext_record_t& event,
    ctf_event_tag<tool_buffer_tracing_memory_copy_ext_record_t>)
{
    auto& base = static_cast<const rocprofiler_buffer_tracing_memory_copy_record_t&>(event);
    auto kind = base.kind;
    auto op   = base.operation;
    auto corr = base.correlation_id;
    auto tid  = base.thread_id;
    auto start = base.start_timestamp;
    auto end   = base.end_timestamp;
    auto dst_agent_id = base.dst_agent_id;
    auto src_agent_id = base.src_agent_id;
    auto bytes = base.bytes;
    auto dst_addr = base.dst_address;
    auto src_addr = base.src_address;
    auto stream_id = event.stream_id;

    // TODO: Use Babeltrace 2 API to create a message/event and set these fields
}

// ---- Marker API ----
void ctf_output::write_event_impl(
    const rocprofiler_buffer_tracing_marker_api_record_t& event,
    ctf_event_tag<rocprofiler_buffer_tracing_marker_api_record_t>)
{
    auto kind = event.kind;
    auto op   = event.operation;
    auto corr = event.correlation_id;
    auto start = event.start_timestamp;
    auto end   = event.end_timestamp;
    auto tid   = event.thread_id;

    // TODO: Use Babeltrace 2 API to create a message/event and set these fields
}

// ---- Scratch Memory ----
void ctf_output::write_event_impl(
    const rocprofiler_buffer_tracing_scratch_memory_record_t& event,
    ctf_event_tag<rocprofiler_buffer_tracing_scratch_memory_record_t>)
{
    auto kind = event.kind;
    auto op   = event.operation;
    auto corr = event.correlation_id;
    auto agent_id = event.agent_id;
    auto queue_id = event.queue_id;
    auto tid = event.thread_id;
    auto start = event.start_timestamp;
    auto end = event.end_timestamp;
    auto flags = event.flags;

    // TODO: Use Babeltrace 2 API to create a message/event and set these fields
}

// ---- RCCL API ----
void ctf_output::write_event_impl(
    const rocprofiler_buffer_tracing_rccl_api_record_t& event,
    ctf_event_tag<rocprofiler_buffer_tracing_rccl_api_record_t>)
{
    auto kind = event.kind;
    auto op   = event.operation;
    auto corr = event.correlation_id;
    auto start = event.start_timestamp;
    auto end   = event.end_timestamp;
    auto tid   = event.thread_id;

    // TODO: Use Babeltrace 2 API to create a message/event and set these fields
}

// ---- Memory Allocation ----
void ctf_output::write_event_impl(
    const rocprofiler_buffer_tracing_memory_allocation_record_t& event,
    ctf_event_tag<rocprofiler_buffer_tracing_memory_allocation_record_t>)
{
    auto kind = event.kind;
    auto op   = event.operation;
    auto corr = event.correlation_id;
    auto tid  = event.thread_id;
    auto start = event.start_timestamp;
    auto end   = event.end_timestamp;
    auto agent_id = event.agent_id;
    auto address = event.address;
    auto alloc_size = event.allocation_size;

    // TODO: Use Babeltrace 2 API to create a message/event and set these fields
}

// ---- rocDecode API EXT ----
void ctf_output::write_event_impl(
    const rocprofiler_buffer_tracing_rocdecode_api_ext_record_t& event,
    ctf_event_tag<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t>)
{
    auto kind = event.kind;
    auto op   = event.operation;
    auto corr = event.correlation_id;
    auto start = event.start_timestamp;
    auto end   = event.end_timestamp;
    auto tid   = event.thread_id;
    // event.args, event.retval

    // TODO: Use Babeltrace 2 API to create a message/event and set these fields
}

// ---- rocJPEG API ----
void ctf_output::write_event_impl(
    const rocprofiler_buffer_tracing_rocjpeg_api_record_t& event,
    ctf_event_tag<rocprofiler_buffer_tracing_rocjpeg_api_record_t>)
{
    auto kind = event.kind;
    auto op   = event.operation;
    auto corr = event.correlation_id;
    auto start = event.start_timestamp;
    auto end   = event.end_timestamp;
    auto tid   = event.thread_id;

    // TODO: Use Babeltrace 2 API to create a message/event and set these fields
}

void
setup(const output_config& cfg)
{
    auto _filename = get_output_filename(cfg, "results", std::string_view{});
    auto _filepath = fs::path{_filename};
    auto _name     = _filepath.filename().string();
    auto _path     = _filepath.parent_path().string();

    if(fs::exists(_filepath)) fs::remove_all(_filepath);

    fs::create_directories(_filepath);

    ROCP_ERROR << "Opened result file: " << _filename;
}

ctf_output open_ctf_stream(const output_config& cfg) {
    setup(cfg);
    return ctf_output{cfg};
}

void close_ctf_stream(ctf_output& ctf_out) {
    ctf_out.close();
}

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
          std::deque<rocprofiler_buffer_tracing_rocjpeg_api_record_t>*       rocjpeg_api_data)
{
    // Loop over each deque and call write_event for each record
    if (hip_api_data) {
        for (const auto& rec : *hip_api_data) {
            ctf_out.write_event(rec);
        }
    }
    if (hsa_api_data) {
        for (const auto& rec : *hsa_api_data) {
            ctf_out.write_event(rec);
        }
    }
    if (kernel_dispatch_data) {
        for (const auto& rec : *kernel_dispatch_data) {
            ctf_out.write_event(rec);
        }
    }
    if (memory_copy_data) {
        for (const auto& rec : *memory_copy_data) {
            ctf_out.write_event(rec);
        }
    }
    if (marker_api_data) {
        for (const auto& rec : *marker_api_data) {
            ctf_out.write_event(rec);
        }
    }
    if (scratch_memory_data) {
        for (const auto& rec : *scratch_memory_data) {
            ctf_out.write_event(rec);
        }
    }
    if (rccl_api_data) {
        for (const auto& rec : *rccl_api_data) {
            ctf_out.write_event(rec);
        }
    }
    if (memory_allocation_data) {
        for (const auto& rec : *memory_allocation_data) {
            ctf_out.write_event(rec);
        }
    }
    if (rocdecode_api_data) {
        for (const auto& rec : *rocdecode_api_data) {
            ctf_out.write_event(rec);
        }
    }
    if (rocjpeg_api_data) {
        for (const auto& rec : *rocjpeg_api_data) {
            ctf_out.write_event(rec);
        }
    }
}

}  // namespace tool
}  // namespace rocprofiler
