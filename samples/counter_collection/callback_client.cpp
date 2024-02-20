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

#include "client.hpp"

#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

int
start()
{
    return 1;
}

namespace
{
rocprofiler_context_id_t&
get_client_ctx()
{
    static rocprofiler_context_id_t ctx;
    return ctx;
}

void
record_callback(rocprofiler_queue_id_t,
                rocprofiler_agent_id_t,
                rocprofiler_correlation_id_t,
                uint64_t,
                void*                         callback_data_args,
                size_t                        record_count,
                rocprofiler_record_counter_t* record_data)
{
    std::stringstream ss;
    for(size_t i = 0; i < record_count; ++i)
    {
        ss << "(Id: " << record_data[i].id << " Value [D]: " << record_data[i].counter_value
           << " Corr_Id: " << record_data[i].corr_id.internal << "),";
    }
    auto* output_stream = static_cast<std::ostream*>(callback_data_args);
    if(!output_stream) throw std::runtime_error{"nullptr to output stream"};
    *output_stream << "[" << __FUNCTION__ << "] " << ss.str() << "\n";
}

/**
 * Callback from rocprofiler when an kernel dispatch is enqueued into the HSA queue.
 * rocprofiler_profile_config_id_t* is a return to specify what counters to collect
 * for this dispatch (dispatch_packet). This example function creates a profile
 * to collect the counter SQ_WAVES for all kernel dispatch packets.
 */
void
dispatch_callback(rocprofiler_queue_id_t /*queue_id*/,
                  const rocprofiler_agent_t* agent,
                  rocprofiler_correlation_id_t /*correlation_id*/,
                  const hsa_kernel_dispatch_packet_t* /*dispatch_packet*/,
                  uint64_t /*kernel_id*/,
                  void* /*callback_data_args*/,
                  rocprofiler_profile_config_id_t* config)
{
    /**
     * This simple example uses the same profile counter set for all agents.
     * We store this in a cache to prevent constructing many identical profile counter
     * sets. We first check the cache to see if we have already constructed a counter"
     * set for the agent. If we have, return it. Otherwise, construct a new profile counter
     * set.
     */
    static std::shared_mutex                                             m_mutex       = {};
    static std::unordered_map<uint64_t, rocprofiler_profile_config_id_t> profile_cache = {};

    auto search_cache = [&]() {
        if(auto pos = profile_cache.find(agent->id.handle); pos != profile_cache.end())
        {
            *config = pos->second;
            return true;
        }
        return false;
    };

    {
        auto rlock = std::shared_lock{m_mutex};
        if(search_cache()) return;
    }

    auto wlock = std::unique_lock{m_mutex};
    if(search_cache()) return;

    // Counters we want to collect (here its SQ_WAVES)
    std::set<std::string> counters_to_collect = {"SQ_WAVES"};
    // GPU Counter IDs
    std::vector<rocprofiler_counter_id_t> gpu_counters;

    // Iterate through the agents and get the counters available on that agent
    ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
                         agent->id,
                         [](rocprofiler_agent_id_t,
                            rocprofiler_counter_id_t* counters,
                            size_t                    num_counters,
                            void*                     user_data) {
                             std::vector<rocprofiler_counter_id_t>* vec =
                                 static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                             for(size_t i = 0; i < num_counters; i++)
                             {
                                 vec->push_back(counters[i]);
                             }
                             return ROCPROFILER_STATUS_SUCCESS;
                         },
                         static_cast<void*>(&gpu_counters)),
                     "Could not fetch supported counters");

    std::vector<rocprofiler_counter_id_t> collect_counters;
    // Look for the counters contained in counters_to_collect in gpu_counters
    for(auto& counter : gpu_counters)
    {
        rocprofiler_counter_info_v0_t version;
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&version)),
            "Could not query info");
        if(counters_to_collect.count(std::string(version.name)) > 0)
        {
            std::clog << "Counter: " << counter.handle << " " << version.name << "\n";
            collect_counters.push_back(counter);
        }
    }

    // Create a colleciton profile for the counters
    rocprofiler_profile_config_id_t profile;
    ROCPROFILER_CALL(rocprofiler_create_profile_config(
                         agent->id, collect_counters.data(), collect_counters.size(), &profile),
                     "Could not construct profile cfg");

    profile_cache.emplace(agent->id.handle, profile);
    // Return the profile to collect those counters for this dispatch
    *config = profile;
}

int
tool_init(rocprofiler_client_finalize_t, void* user_data)
{
    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_configure_callback_dispatch_profile_counting_service(
                         get_client_ctx(), dispatch_callback, nullptr, record_callback, user_data),
                     "Could not setup counting service");
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context");

    // no errors
    return 0;
}

void
tool_fini(void* user_data)
{
    std::clog << "In tool fini\n";
    rocprofiler_stop_context(get_client_ctx());

    auto* output_stream = static_cast<std::ostream*>(user_data);
    *output_stream << std::flush;
    if(output_stream != &std::cout && output_stream != &std::cerr) delete output_stream;
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t    version,
                      const char* runtime_version,
                      uint32_t,
                      rocprofiler_client_id_t* id)
{
    // set the client name
    id->name = "CounterClientSample";

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " is using rocprofiler-sdk v" << major << "." << minor << "." << patch
         << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    std::ostream* output_stream = nullptr;
    std::string   filename      = "counter_collection.log";
    if(auto* outfile = getenv("ROCPROFILER_SAMPLE_OUTPUT_FILE"); outfile) filename = outfile;
    if(filename == "stdout")
        output_stream = &std::cout;
    else if(filename == "stderr")
        output_stream = &std::cerr;
    else
        output_stream = new std::ofstream{filename};

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &tool_init,
                                            &tool_fini,
                                            static_cast<void*>(output_stream)};

    // return pointer to configure data
    return &cfg;
}
