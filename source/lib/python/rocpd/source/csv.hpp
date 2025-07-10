
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

#pragma once

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>

#include "lib/python/rocpd/source/types.hpp"

#include "lib/common/defines.hpp"
#include "lib/output/csv_output_file.hpp"
#include "lib/output/generateStats.hpp"
#include "lib/output/generator.hpp"
#include "lib/output/metadata.hpp"
#include "lib/output/node_info.hpp"
#include "lib/output/output_config.hpp"
#include "lib/output/sql/common.hpp"
#include "lib/output/stream_info.hpp"
#include "lib/rocprofiler-sdk-tool/config.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

namespace rocpd
{
namespace output
{
rocprofiler::tool::csv_output_file
generate_agent_ofs(const rocprofiler::tool::output_config& cfg);

rocprofiler::tool::csv_output_file
generate_kernel_ofs(const rocprofiler::tool::output_config& cfg);

rocprofiler::tool::csv_output_file
generate_memory_copy_ofs(const rocprofiler::tool::output_config& cfg);

rocprofiler::tool::csv_output_file
generate_memory_allocation_ofs(const rocprofiler::tool::output_config& cfg);

rocprofiler::tool::csv_output_file
generate_scratch_memory_ofs(const rocprofiler::tool::output_config& cfg);

rocprofiler::tool::csv_output_file
generate_counter_ofs(const rocprofiler::tool::output_config& cfg);

rocprofiler::tool::csv_output_file
generate_region_ofs(const rocprofiler::tool::output_config& cfg, domain_type domain);

void
generate_csv(rocprofiler::tool::csv_output_file& ofs, const std::vector<rocpd::types::agent>& data);

void
generate_csv(const rocprofiler::tool::output_config&                            cfg,
             rocprofiler::tool::csv_output_file&                                ofs,
             const rocprofiler::tool::generator<rocpd::types::kernel_dispatch>& data);

void
generate_csv(const rocprofiler::tool::output_config&                          cfg,
             rocprofiler::tool::csv_output_file&                              ofs,
             const rocprofiler::tool::generator<rocpd::types::memory_copies>& data);

void
generate_csv(const rocprofiler::tool::output_config&                              cfg,
             rocprofiler::tool::csv_output_file&                                  ofs,
             const rocprofiler::tool::generator<rocpd::types::memory_allocation>& data);

void
generate_csv(const rocprofiler::tool::output_config&                           cfg,
             rocprofiler::tool::csv_output_file&                               ofs,
             const rocprofiler::tool::generator<rocpd::types::scratch_memory>& data);

void
generate_csv(const rocprofiler::tool::output_config&                    cfg,
             rocprofiler::tool::csv_output_file&                        ofs,
             const rocprofiler::tool::generator<rocpd::types::counter>& data);

void
generate_csv(rocprofiler::tool::csv_output_file&                       ofs,
             const rocprofiler::tool::generator<rocpd::types::region>& data,
             const domain_type                                         domain);

}  // namespace output
}  // namespace rocpd
