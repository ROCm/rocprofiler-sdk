// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/python/rocpd/source/csv.hpp"

#include "lib/common/defines.hpp"
#include "lib/common/hasher.hpp"
#include "lib/output/csv_output_file.hpp"
#include "lib/output/generator.hpp"
#include "lib/output/metadata.hpp"
#include "lib/output/node_info.hpp"
#include "lib/output/output_config.hpp"
#include "lib/output/output_stream.hpp"
#include "lib/output/sql/common.hpp"
#include "lib/output/stream_info.hpp"
#include "lib/rocprofiler-sdk-tool/config.hpp"

#include <fmt/format.h>

#include <cereal/archives/json.hpp>
#include <sstream>

#include <atomic>
#include <filesystem>
#include <future>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <vector>

namespace rocpd
{
namespace output
{
std::string
extract_field_from_extdata(const std::string& extdata, const std::string& field)
{
    try
    {
        std::stringstream        ss(extdata);
        cereal::JSONInputArchive archive(ss);
        std::string              value;
        archive(cereal::make_nvp(field.c_str(), value));
        return value;
    } catch(...)
    {
        return {};
    }
}

rocprofiler::tool::csv_output_file
generate_agent_ofs(const rocprofiler::tool::output_config& cfg)
{
    return rocprofiler::tool::csv_output_file{cfg,
                                              "agent_info",
                                              rocprofiler::tool::csv::agent_info_csv_encoder_guid{},
                                              {"Guid",
                                               "Node_Id",
                                               "Logical_Node_Id",
                                               "Agent_Type",
                                               "Cpu_Cores_Count",
                                               "Simd_Count",
                                               "Cpu_Core_Id_Base",
                                               "Simd_Id_Base",
                                               "Max_Waves_Per_Simd",
                                               "Lds_Size_In_Kb",
                                               "Gds_Size_In_Kb",
                                               "Num_Gws",
                                               "Wave_Front_Size",
                                               "Num_Xcc",
                                               "Cu_Count",
                                               "Array_Count",
                                               "Num_Shader_Banks",
                                               "Simd_Arrays_Per_Engine",
                                               "Cu_Per_Simd_Array",
                                               "Simd_Per_Cu",
                                               "Max_Slots_Scratch_Cu",
                                               "Gfx_Target_Version",
                                               "Vendor_Id",
                                               "Device_Id",
                                               "Location_Id",
                                               "Domain",
                                               "Drm_Render_Minor",
                                               "Num_Sdma_Engines",
                                               "Num_Sdma_Xgmi_Engines",
                                               "Num_Sdma_Queues_Per_Engine",
                                               "Num_Cp_Queues",
                                               "Max_Engine_Clk_Ccompute",
                                               "Max_Engine_Clk_Fcompute",
                                               "Sdma_Fw_Version",
                                               "Fw_Version",
                                               "Capability",
                                               "Cu_Per_Engine",
                                               "Max_Waves_Per_Cu",
                                               "Family_Id",
                                               "Workgroup_Max_Size",
                                               "Grid_Max_Size",
                                               "Local_Mem_Size",
                                               "Hive_Id",
                                               "Gpu_Id",
                                               "Workgroup_Max_Dim_X",
                                               "Workgroup_Max_Dim_Y",
                                               "Workgroup_Max_Dim_Z",
                                               "Grid_Max_Dim_X",
                                               "Grid_Max_Dim_Y",
                                               "Grid_Max_Dim_Z",
                                               "Name",
                                               "Vendor_Name",
                                               "Product_Name",
                                               "Model_Name"}};
}

rocprofiler::tool::csv_output_file
generate_kernel_ofs(const rocprofiler::tool::output_config& cfg)
{
    return rocprofiler::tool::csv_output_file{
        cfg,
        domain_type::KERNEL_DISPATCH,
        rocprofiler::tool::csv::kernel_trace_with_stream_csv_encoder_guid{},
        {"Guid",
         "Kind",
         "Agent_Id",
         "Queue_Id",
         "Stream_Id",
         "Thread_Id",
         "Dispatch_Id",
         "Kernel_Id",
         "Kernel_Name",
         "Correlation_Id",
         "Start_Timestamp",
         "End_Timestamp",
         "LDS_Block_Size",
         "Scratch_Size",
         "VGPR_Count",
         "Accum_VGPR_Count",
         "SGPR_Count",
         "Workgroup_Size_X",
         "Workgroup_Size_Y",
         "Workgroup_Size_Z",
         "Grid_Size_X",
         "Grid_Size_Y",
         "Grid_Size_Z"}};
}

rocprofiler::tool::csv_output_file
generate_memory_copy_ofs(const rocprofiler::tool::output_config& cfg)
{
    return rocprofiler::tool::csv_output_file{
        cfg,
        domain_type::MEMORY_COPY,
        rocprofiler::tool::csv::memory_copy_with_stream_csv_encoder_guid{},
        {"Guid",
         "Kind",
         "Direction",
         "Stream_Id",
         "Source_Agent_Id",
         "Destination_Agent_Id",
         "Correlation_Id",
         "Start_Timestamp",
         "End_Timestamp"}};
}

rocprofiler::tool::csv_output_file
generate_memory_allocation_ofs(const rocprofiler::tool::output_config& cfg)
{
    return rocprofiler::tool::csv_output_file{
        cfg,
        domain_type::MEMORY_ALLOCATION,
        rocprofiler::tool::csv::memory_allocation_csv_encoder_guid{},
        {"Guid",
         "Kind",
         "Operation",
         "Agent_Id",
         "Allocation_Size",
         "Address",
         "Correlation_Id",
         "Start_Timestamp",
         "End_Timestamp"}};
}

rocprofiler::tool::csv_output_file
generate_scratch_memory_ofs(const rocprofiler::tool::output_config& cfg)
{
    return rocprofiler::tool::csv_output_file{cfg,
                                              domain_type::SCRATCH_MEMORY,
                                              rocprofiler::tool::csv::scratch_memory_encoder_guid{},
                                              {
                                                  "Guid",
                                                  "Kind",
                                                  "Operation",
                                                  "Agent_Id",
                                                  "Allocation_Size",
                                                  "Queue_Id",
                                                  "Thread_Id",
                                                  "Alloc_Flags",
                                                  "Start_Timestamp",
                                                  "End_Timestamp",
                                              }};
}

rocprofiler::tool::csv_output_file
generate_counter_ofs(const rocprofiler::tool::output_config& cfg)
{
    return rocprofiler::tool::csv_output_file{
        cfg,
        domain_type::COUNTER_COLLECTION,
        rocprofiler::tool::csv::counter_collection_csv_encoder_guid{},
        {"Guid",           "Correlation_Id", "Dispatch_Id",   "Agent_Id",        "Queue_Id",
         "Process_Id",     "Thread_Id",      "Grid_Size",     "Kernel_Id",       "Kernel_Name",
         "Workgroup_Size", "LDS_Block_Size", "Scratch_Size",  "VGPR_Count",      "Accum_VGPR_Count",
         "SGPR_Count",     "Counter_Name",   "Counter_Value", "Start_Timestamp", "End_Timestamp"}};
}
rocprofiler::tool::csv_output_file
generate_region_ofs(const rocprofiler::tool::output_config& cfg, domain_type domain)
{
    return rocprofiler::tool::csv_output_file{cfg,
                                              domain,
                                              rocprofiler::tool::csv::api_csv_encoder_guid{},
                                              {"Guid",
                                               "Domain",
                                               "Function",
                                               "Process_Id",
                                               "Thread_Id",
                                               "Correlation_Id",
                                               "Start_Timestamp",
                                               "End_Timestamp"}};
}
void
generate_csv(rocprofiler::tool::csv_output_file& ofs, const std::vector<rocpd::types::agent>& data)
{
    if(data.empty()) return;

    auto sorted_data = data;
    std::sort(sorted_data.begin(),
              sorted_data.end(),
              [](const rocpd::types::agent& lhs, const rocpd::types::agent& rhs) {
                  return lhs.node_id < rhs.node_id;
              });

    for(auto& itr : sorted_data)
    {
        auto row_ss = std::stringstream{};
        rocprofiler::tool::csv::agent_info_csv_encoder_guid::write_row(
            row_ss,
            itr.guid,
            itr.node_id,
            itr.logical_node_id,
            itr.type,
            itr.cpu_cores_count,
            itr.simd_count,
            itr.cpu_core_id_base,
            itr.simd_id_base,
            itr.max_waves_per_simd,
            itr.lds_size_in_kb,
            itr.gds_size_in_kb,
            itr.num_gws,
            itr.wave_front_size,
            itr.num_xcc,
            itr.cu_count,
            itr.array_count,
            itr.num_shader_banks,
            itr.simd_arrays_per_engine,
            itr.cu_per_simd_array,
            itr.simd_per_cu,
            itr.max_slots_scratch_cu,
            itr.gfx_target_version,
            itr.vendor_id,
            itr.device_id,
            itr.location_id,
            itr.domain,
            itr.drm_render_minor,
            itr.num_sdma_engines,
            itr.num_sdma_xgmi_engines,
            itr.num_sdma_queues_per_engine,
            itr.num_cp_queues,
            itr.max_engine_clk_ccompute,
            itr.max_engine_clk_fcompute,
            itr.sdma_fw_version.Value,
            itr.fw_version.Value,
            itr.capability.Value,
            itr.cu_per_engine,
            itr.max_waves_per_cu,
            itr.family_id,
            itr.workgroup_max_size,
            itr.grid_max_size,
            itr.local_mem_size,
            itr.hive_id,
            itr.gpu_id,
            itr.workgroup_max_dim.x,
            itr.workgroup_max_dim.y,
            itr.workgroup_max_dim.z,
            itr.grid_max_dim.x,
            itr.grid_max_dim.y,
            itr.grid_max_dim.z,
            itr.name,
            itr.vendor_name,
            itr.product_name,
            itr.model_name);
        ofs << row_ss.str();
    }
}

void
generate_csv(const rocprofiler::tool::output_config&                            cfg,
             rocprofiler::tool::csv_output_file&                                ofs,
             const rocprofiler::tool::generator<rocpd::types::kernel_dispatch>& data)
{
    if(data.empty()) return;

    for(const auto& ditr : data)
    {
        for(const auto& record : data.get(ditr))
        {
            std::string kernel_identifier = cfg.kernel_rename ? record.region : record.name;
            std::string agent_identifier  = create_agent_index(cfg.agent_index_value,
                                                              record.agent_abs_index,
                                                              record.agent_log_index,
                                                              record.agent_type_index,
                                                              std::string_view(record.agent_type))
                                               .as_string();

            auto row_ss = std::stringstream{};
            rocprofiler::tool::csv::kernel_trace_with_stream_csv_encoder_guid::write_row(
                row_ss,
                record.guid,
                record.category,
                agent_identifier,
                record.queue_id,
                record.stream_id,
                record.tid,
                record.dispatch_id,
                record.kernel_id,
                kernel_identifier,
                record.stack_id,
                record.start,
                record.end,
                record.lds_size,
                record.scratch_size,
                record.arch_vgpr_count,
                record.accum_vgpr_count,
                record.sgpr_count,
                record.workgroup_size.x,
                record.workgroup_size.y,
                record.workgroup_size.z,
                record.grid_size.x,
                record.grid_size.y,
                record.grid_size.z);
            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const rocprofiler::tool::output_config&                          cfg,
             rocprofiler::tool::csv_output_file&                              ofs,
             const rocprofiler::tool::generator<rocpd::types::memory_copies>& data)
{
    if(data.empty()) return;

    for(const auto& ditr : data)
    {
        for(const auto& record : data.get(ditr))
        {
            auto row_ss = std::stringstream{};

            std::string src_agent_identifier =
                create_agent_index(cfg.agent_index_value,
                                   record.src_agent_abs_index,
                                   record.src_agent_log_index,
                                   record.src_agent_type_index,
                                   std::string_view(record.src_agent_type))
                    .as_string();

            std::string dst_agent_identifier =
                create_agent_index(cfg.agent_index_value,
                                   record.dst_agent_abs_index,
                                   record.dst_agent_log_index,
                                   record.dst_agent_type_index,
                                   std::string_view(record.dst_agent_type))
                    .as_string();

            rocprofiler::tool::csv::memory_copy_with_stream_csv_encoder_guid::write_row(
                row_ss,
                record.guid,
                record.category,
                record.name,
                record.stream_id,
                src_agent_identifier,
                dst_agent_identifier,
                record.stack_id,
                record.start,
                record.end);
            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const rocprofiler::tool::output_config&                              cfg,
             rocprofiler::tool::csv_output_file&                                  ofs,
             const rocprofiler::tool::generator<rocpd::types::memory_allocation>& data)
{
    if(data.empty()) return;

    for(const auto& ditr : data)
    {
        for(const auto& record : data.get(ditr))
        {
            std::string agent_identifier = create_agent_index(cfg.agent_index_value,
                                                              record.agent_abs_index,
                                                              record.agent_log_index,
                                                              record.agent_type_index,
                                                              std::string_view(record.agent_type))
                                               .as_string();

            std::string agent_id = record.type != "FREE" ? agent_identifier : "";

            auto row_ss = std::stringstream{};

            rocprofiler::tool::csv::memory_allocation_csv_encoder_guid::write_row(
                row_ss,
                record.guid,
                record.category,
                record.type,
                agent_identifier,
                record.size,
                rocprofiler::sdk::utility::as_hex(record.address, 16),
                record.stack_id,
                record.start,
                record.end);

            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const rocprofiler::tool::output_config&                           cfg,
             rocprofiler::tool::csv_output_file&                               ofs,
             const rocprofiler::tool::generator<rocpd::types::scratch_memory>& data)
{
    if(data.empty()) return;

    for(const auto& ditr : data)
    {
        for(const auto& record : data.get(ditr))
        {
            std::string agent_identifier = create_agent_index(cfg.agent_index_value,
                                                              record.agent_abs_index,
                                                              record.agent_log_index,
                                                              record.agent_type_index,
                                                              std::string_view(record.agent_type))
                                               .as_string();

            auto row_ss = std::stringstream{};

            rocprofiler::tool::csv::scratch_memory_encoder_guid::write_row(row_ss,
                                                                           record.guid,
                                                                           record.category,
                                                                           record.operation,
                                                                           agent_identifier,
                                                                           record.size,
                                                                           record.queue_id,
                                                                           record.tid,
                                                                           record.alloc_flags,
                                                                           record.start,
                                                                           record.end);

            ofs << row_ss.str();
        }
    }
}

void
generate_csv(const rocprofiler::tool::output_config&                    cfg,
             rocprofiler::tool::csv_output_file&                        ofs,
             const rocprofiler::tool::generator<rocpd::types::counter>& data)
{
    if(data.empty()) return;

    for(const auto& ditr : data)
    {
        for(const auto& record : data.get(ditr))
        {
            std::string agent_identifier = create_agent_index(cfg.agent_index_value,
                                                              record.agent_abs_index,
                                                              record.agent_log_index,
                                                              record.agent_type_index,
                                                              std::string_view(record.agent_type))
                                               .as_string();

            auto row_ss = std::stringstream{};
            rocprofiler::tool::csv::counter_collection_csv_encoder_guid::write_row(
                row_ss,
                record.guid,
                record.stack_id,
                record.dispatch_id,
                agent_identifier,
                record.queue_id,
                record.pid,
                record.tid,
                record.grid_size,
                record.kernel_id,
                record.kernel_name,
                record.workgroup_size,
                record.lds_block_size,
                record.scratch_size,
                record.vgpr_count,
                record.accum_vgpr_count,
                record.sgpr_count,
                record.counter_name,
                record.value,
                record.start,
                record.end);

            ofs << row_ss.str();
        }
    }
}

void
generate_csv(rocprofiler::tool::csv_output_file&                       ofs,
             const rocprofiler::tool::generator<rocpd::types::region>& data,
             const domain_type                                         domain)
{
    // namespace sdk = ::rocprofiler::sdk;

    if(data.empty()) return;

    for(const auto& ditr : data)
    {
        for(const auto& record : data.get(ditr))
        {
            auto row_ss = std::stringstream{};

            auto name = record.name;

            if(domain == domain_type::MARKER)
            {
                std::string message = extract_field_from_extdata(record.extdata, "message");
                if(!message.empty()) name = message;
            }

            rocprofiler::tool::csv::api_csv_encoder_guid::write_row(row_ss,
                                                                    record.guid,
                                                                    record.category,
                                                                    name,
                                                                    record.pid,
                                                                    record.tid,
                                                                    record.stack_id,
                                                                    record.start,
                                                                    record.end);

            ofs << row_ss.str();
        }
    }
}

}  // namespace output
}  // namespace rocpd