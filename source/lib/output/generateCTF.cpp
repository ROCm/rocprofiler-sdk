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

#include "ctf/hip_api_ext.hpp"

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

ctf_output::~ctf_output() { close(); }

void
ctf_output::close()
{
    // Cleanup
    bt_graph_put_ref(graph);
}

void
ctf_output::write_event_source_component(
    const char* source_name,
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
        uint64_t*                 count),
    void* data)
{
    bt_message_iterator_class* msg_iter_cls = bt_message_iterator_class_create(next_method);

    // 2. Optionally set msg_iter init/finalize
    // bt_message_iterator_class_set_initialize_method(msg_iter_cls, hip_api_source_msg_iter_init);
    // bt_message_iterator_class_set_finalize_method(msg_iter_cls,
    // hip_api_source_msg_iter_finalize);

    // 3. Create source component class
    bt_component_class_source* src_class =
        bt_component_class_source_create(source_name, msg_iter_cls);

    // 4. Set component class init/finalize
    bt_component_class_source_set_initialize_method(src_class, init_method);
    bt_component_class_source_set_finalize_method(src_class, finalize_method);

    // 5. Add the source component to the graph
    const bt_component_source* src_comp = nullptr;
    bt_graph_add_source_component_with_initialize_method_data(
        graph, src_class, source_name, NULL, data, BT_LOGGING_LEVEL_NONE, &src_comp);

    bt_port_output* src_out_port =
        (bt_port_output*) bt_self_component_source_borrow_output_port_by_index(
            (bt_self_component_source*) src_comp, 0);
    bt_port_input* sink_in_port =
        (bt_port_input*) bt_self_component_sink_borrow_input_port_by_index(
            (bt_self_component_sink*) ctf_writer_comp, 0);
    bt_graph_connect_ports(graph, src_out_port, sink_in_port, NULL);

    bt_graph_run(graph);
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

ctf_output
open_ctf_stream(const output_config& cfg)
{
    setup(cfg);
    return ctf_output{cfg};
}

void
close_ctf_stream(ctf_output& ctf_out)
{
    ctf_out.close();
}

void
write_ctf(ctf_output& ctf_out,
          const output_config& /*cfg*/,
          const metadata& /*tool_metadata*/,
          uint64_t /*pid*/,
          const std::vector<agent_info>& /*agent_data*/,
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
    if(hip_api_data)
    {
        if(!ctf_out.hip_api_ext_initialized)
        {
            ctf_out.hip_api_data =
                new std::deque<rocprofiler_buffer_tracing_hip_api_ext_record_t>();
            ctf_out.write_event_source_component("hip_api_source",
                                                 hip_api_source_init,
                                                 hip_api_source_finalize,
                                                 hip_api_source_next,
                                                 (void*) ctf_out.hip_api_data);
            ctf_out.hip_api_ext_initialized = true;
        }

        while(!hip_api_data->empty()) {
            ctf_out.hip_api_data->push_front(hip_api_data->back());
            hip_api_data->pop_back();
        }
    }
    if(hsa_api_data)
    {
        // ctf_out.write_event_source_component("hsa_api_source",
        //                              hsa_api_source_init,
        //                              hsa_api_source_finalize,
        //                              hsa_api_source_next,
        //                              hsa_api_data);
    }
    if(kernel_dispatch_data)
    {
        // ctf_out.write_event_source_component("kernel_dispatch_source",
        //                              kernel_dispatch_source_init,
        //                              kernel_dispatch_source_finalize,
        //                              kernel_dispatch_source_next,
        //                              kernel_dispatch_data);
    }
    if(memory_copy_data)
    {
        // ctf_out.write_event_source_component("memory_copy_source",
        //                              memory_copy_source_init,
        //                              memory_copy_source_finalize,
        //                              memory_copy_source_next,
        //                              memory_copy_data);
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
