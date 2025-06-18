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

#include "lib/python/rocpd/source/ctf2.hpp"

#include "lib/common/defines.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/hasher.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/units.hpp"
#include "lib/common/utility.hpp"
#include "lib/output/generator.hpp"
#include "lib/output/metadata.hpp"
#include "lib/output/node_info.hpp"
#include "lib/output/output_config.hpp"
#include "lib/output/output_stream.hpp"
#include "lib/output/sql/common.hpp"
#include "lib/output/stream_info.hpp"
#include "lib/output/timestamps.hpp"
#include "lib/rocprofiler-sdk-tool/config.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/cxx/perfetto.hpp>

#include <fmt/format.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <future>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#define CTF2_CHECK(result)                                                                         \
    {                                                                                              \
        CTF2_ErrorCode ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result;                       \
        if(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != CTF2_SUCCESS)                            \
        {                                                                                          \
            auto _err_name = CTF2_Error_GetName(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));      \
            auto _err_msg =                                                                        \
                CTF2_Error_GetDescription(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));            \
            ROCP_FATAL << #result << " failed with error code " << _err_name                       \
                       << " (code=" << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__)                 \
                       << ") :: " << _err_msg;                                                     \
        }                                                                                          \
    }

#define BUFFER_SIZE (1024 * 10)  // 10 KiB

namespace rocpd
{
namespace output
{
namespace
{
// --- Platform callbacks for barectf ---
// Simple clock_gettime for nanoseconds
static uint64_t
get_time(void* const data)
{
    (void) data;  // Unused
    auto _ts = rocprofiler_timestamp_t{};
    rocprofiler_get_timestamp(&_ts);
    return (uint64_t) _ts;
}

// Dummy "is backend full" - for this example, we assume it's never full
static int
is_backend_full(void* const data)
{
    (void) data; /* Optional */
    return 0;    // Never full
}

static void
write_packet(const struct my_platform_ctx* const platform_ctx)
{
    /* Append current packet to data stream file */
    const size_t nmemb = fwrite(barectf_packet_buf_addr(platform_ctx->ctx),
                                barectf_packet_buf_size(platform_ctx->ctx),
                                1,
                                platform_ctx->fh);

    assert(nmemb == 1);
}

static void
open_packet(void* const data)
{
    struct my_platform_ctx* const platform_ctx =
        reinterpret_cast<struct my_platform_ctx* const>(data);

    barectf_default_open_packet(platform_ctx->ctx);
}
static void
close_packet(void* const data)
{
    struct my_platform_ctx* const platform_ctx =
        reinterpret_cast<struct my_platform_ctx* const>(data);

    /* Close packet now */
    barectf_default_close_packet(platform_ctx->ctx);

    /* Write packet to file */
    write_packet(platform_ctx);
}

}  // namespace

void
CTF2Session::add_event(const extended_event_data& event_data, const types::process& process) const
{
    if(!platform_ctx || !platform_ctx->ctx)
    {
        // ROCP_FATAL << "Platform context is not initialized";
        return;
    }
    switch(event_data.event_type)
    {
        case ctf2_event_type::kernel_dispatch:
        {
            auto kernel_dispatch =
                static_cast<const types::kernel_dispatch*>(event_data.kernel_dispatch);
            if(!kernel_dispatch)
            {
                ROCP_FATAL << "Kernel dispatch event data is null";
                return;
            }
            barectf_default_trace_kernel_dispatch(
                platform_ctx->ctx,
                (event_data.event_phase == ctf2_event_phase::start ? 0 : 1),
                process.pid,
                kernel_dispatch->corr_id,
                event_data.timestamp,
                kernel_dispatch->tid,
                kernel_dispatch->agent_type_index,
                kernel_dispatch->queue_id,
                0,  // Assuming stream_id is not used in this example
                event_data.name.c_str());
            break;
        }
        case ctf2_event_type::api:
        {
            auto api_region = static_cast<const types::region*>(event_data.api_region);
            if(!api_region)
            {
                ROCP_FATAL << "API region event data is null";
                return;
            }
            barectf_default_trace_api(platform_ctx->ctx,
                                      (event_data.event_phase == ctf2_event_phase::start ? 0 : 1),
                                      process.pid,
                                      event_data.name.c_str(),
                                      api_region->parent_stack_id,
                                      api_region->corr_id,
                                      event_data.timestamp,
                                      api_region->tid);
            break;
        }
        case ctf2_event_type::memory_copy:
        {
            auto memory_copy = static_cast<const types::memory_copies*>(event_data.memory_copy);
            if(!memory_copy)
            {
                ROCP_FATAL << "Memory copy event data is null";
                return;
            }
            barectf_default_trace_memory_copy(
                platform_ctx->ctx,
                (event_data.event_phase == ctf2_event_phase::start ? 0 : 1),
                process.pid,
                memory_copy->category.c_str(),
                memory_copy->corr_id,
                event_data.timestamp,
                memory_copy->tid,
                memory_copy->src_agent_type_index,
                memory_copy->dst_agent_type_index,
                0,
                memory_copy->size);
            break;
        }
        // Add other event types as needed
        default:
            ROCP_WARNING << "Unsupported event type: " << static_cast<int>(event_data.event_type);
    }
}

CTF2Session::CTF2Session(const tool::output_config& output_cfg,
                         uint64_t                   min_start,
                         uint64_t                   max_fini)
: config{output_cfg}
{
    namespace fs = rocprofiler::common::filesystem;

    auto _filename =
        rocprofiler::tool::get_output_filename(output_cfg, "results", std::string_view{});
    auto _filepath = fs::path{_filename};

    if(fs::exists(_filepath)) fs::remove_all(_filepath);

    struct barectf_platform_callbacks cbs;

    /* Set platform callback functions */
    cbs.default_clock_get_value = get_time;
    cbs.is_backend_full         = is_backend_full;
    cbs.open_packet             = open_packet;
    cbs.close_packet            = close_packet;

    /* Allocate platform context (which contains a barectf context) */
    platform_ctx = reinterpret_cast<my_platform_ctx*>(malloc(sizeof(*platform_ctx)));

    if(!platform_ctx)
    {
        // goto error;
    }

    /* Allocate packet buffer */
    ctf_buffer = reinterpret_cast<uint8_t*>(malloc(BUFFER_SIZE));

    if(!ctf_buffer)
    {
        // goto error;
    }

    /* Open data stream file */
    platform_ctx->fh = fopen(_filepath.c_str(), "wb");

    if(!platform_ctx->fh)
    {
        // goto error;
    }

    /* Initialize barectf context */
    barectf_init(platform_ctx->ctx, ctf_buffer, BUFFER_SIZE, cbs, platform_ctx);

    /* Open the first packet */
    open_packet(platform_ctx);

    std::cout << "CTF2 session initialized with output file: " << _filepath << std::endl;
}

CTF2Session::~CTF2Session()
{
    /* Close current packet if needed */
    if(barectf_packet_is_open(platform_ctx->ctx) && !barectf_packet_is_empty(platform_ctx->ctx))
    {
        close_packet(platform_ctx);
    }

    /* Close data stream file */
    fclose(platform_ctx->fh);

    /* Deallocate packet buffer */
    free(ctf_buffer);

    /* Deallocate platform context */
    if(platform_ctx) free(platform_ctx);

    std::cout << "CTF2 session finalized and output file closed." << std::endl;
}

void
write_ctf2(const CTF2Session&                                      ctf2_session,
           const types::process&                                   process,
           const uint16_t                                          tree_node_id,
           const std::unordered_map<uint64_t, extended_agent_ctf>& agent_data,
           const tool::generator<types::thread>&                   thread_gen,
           const tool::generator<types::region>&                   api_gen,
           const tool::generator<types::kernel_dispatch>&          kernel_dispatch_gen,
           const tool::generator<types::memory_copies>&            memory_copy_gen,
           const tool::generator<types::memory_allocation>&        memory_allocation_gen)
{
    const auto& ocfg = ctf2_session.config;

    auto _app_ts = rocprofiler::tool::timestamps_t{process.start, process.fini};

    auto _data = std::deque<extended_event_data>{};

    for(auto ditr : kernel_dispatch_gen)
        for(const auto& itr : kernel_dispatch_gen.get(ditr))
        {
            auto _name = fmt::format(
                "{}", (ocfg.kernel_rename && !itr.region.empty()) ? itr.region : itr.name);

            _data.emplace_back(extended_event_data{
                itr.start, ctf2_event_type::kernel_dispatch, ctf2_event_phase::start, &itr, _name});

            _data.emplace_back(extended_event_data{
                itr.end, ctf2_event_type::kernel_dispatch, ctf2_event_phase::end, &itr, _name});
        }

    for(auto ditr : memory_copy_gen)
        for(const auto& itr : memory_copy_gen.get(ditr))
        {
            std::string _name = itr.name;

            _data.emplace_back(extended_event_data{itr.start,
                                                   ctf2_event_type::kernel_dispatch,
                                                   ctf2_event_phase::start,
                                                   .memory_copy = &itr,
                                                   _name});

            _data.emplace_back(extended_event_data{itr.end,
                                                   ctf2_event_type::kernel_dispatch,
                                                   ctf2_event_phase::end,
                                                   .memory_copy = &itr,
                                                   _name});
        }

    for(auto ditr : api_gen)
        for(const auto& itr : api_gen.get(ditr))
        {
            std::string _name = itr.name;

            _data.emplace_back(extended_event_data{itr.start,
                                                   ctf2_event_type::api,
                                                   ctf2_event_phase::start,
                                                   .api_region = &itr,
                                                   _name});

            _data.emplace_back(extended_event_data{
                itr.end, ctf2_event_type::api, ctf2_event_phase::end, .api_region = &itr, _name});
        }

    std::sort(_data.begin(),
              _data.end(),
              [](const extended_event_data& lhs, const extended_event_data& rhs) {
                  if(lhs.timestamp != rhs.timestamp) return (lhs.timestamp < rhs.timestamp);
                  return (lhs.event_phase >= rhs.event_phase);
              });

    uint64_t largest_timestamp = 0;
    for(const auto& itr : _data)
    {
        ROCP_ERROR_IF(itr.timestamp < largest_timestamp)
            << "event found with timestamp < last event timestamp by "
            << (largest_timestamp - itr.timestamp) << " nsec";

        ctf2_session.add_event(itr, process);

        largest_timestamp = itr.timestamp;

        ROCP_ERROR_IF(itr.timestamp < _app_ts.app_start_time)
            << "event found with timestamp < app start time by "
            << (_app_ts.app_start_time - itr.timestamp) << " nsec";
        ROCP_ERROR_IF(itr.timestamp > _app_ts.app_end_time)
            << "event found with timestamp > app end time by "
            << (itr.timestamp - _app_ts.app_end_time) << " nsec";
    }
}

}  // namespace output
}  // namespace rocpd
