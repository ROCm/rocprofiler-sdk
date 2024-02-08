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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <fmt/core.h>

#include "lib/common/container/small_vector.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/counters/dimensions.hpp"
#include "lib/rocprofiler-sdk/counters/evaluate_ast.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

extern "C" {
/**
 * @brief Query Counter name.
 *
 * @param [in] counter_id
 * @param [out] name if nullptr, size will be returned
 * @param [out] size
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_query_counter_name(rocprofiler_counter_id_t counter_id, const char** name, size_t* size)
{
    const auto& id_map = *CHECK_NOTNULL(rocprofiler::counters::getMetricIdMap());

    if(const auto* metric_ptr = rocprofiler::common::get_val(id_map, counter_id.handle))
    {
        *name = metric_ptr->name().c_str();
        *size = metric_ptr->name().size();
        return ROCPROFILER_STATUS_SUCCESS;
    }

    return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
}

/**
 * @brief This call returns the number of instances specific counter contains.
 *        WARNING: There is a restriction on this call in the alpha/beta release
 *        of rocprof. This call will not return correct instance information in
 *        tool_init and must be called as part of the dispatch callback for accurate
 *        instance counting information. The reason for this restriction is that HSA
 *        is not yet loaded on tool_init.
 *
 * @param [in] agent rocprofiler agent
 * @param [in] counter_id counter id (obtained from iterate_agent_supported_counters)
 * @param [out] instance_count number of instances the counter has
 * @return rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_query_counter_instance_count(rocprofiler_agent_id_t,
                                         rocprofiler_counter_id_t counter_id,
                                         size_t*                  instance_count)
{
    *instance_count = 0;

    if(rocprofiler::counters::get_dimension_cache().empty())
    {
        return ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED;
    }

    const auto* dims = rocprofiler::common::get_val(rocprofiler::counters::get_dimension_cache(),
                                                    counter_id.handle);
    if(!dims) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;

    for(const auto& metric_dim : *dims)
    {
        if(*instance_count == 0)
            *instance_count = metric_dim.size();
        else if(metric_dim.size() > 0)
            *instance_count = metric_dim.size() * *instance_count;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}
/**
 * @brief Query Agent Counters Availability.
 *
 * @param [in] agent
 * @param [out] counters_list
 * @param [out] counters_count
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_iterate_agent_supported_counters(rocprofiler_agent_id_t              agent_id,
                                             rocprofiler_available_counters_cb_t cb,
                                             void*                               user_data)
{
    const auto* agent = rocprofiler::agent::get_agent(agent_id);
    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    auto metrics = rocprofiler::counters::getMetricsForAgent(agent->name);
    std::vector<rocprofiler_counter_id_t> ids;
    ids.reserve(metrics.size());
    for(const auto& metric : metrics)
    {
        ids.push_back({.handle = metric.id()});
    }

    return cb(agent_id, ids.data(), ids.size(), user_data);
}

/**
 * @brief Query counter id information from record_id
 *
 * @param [in] id record id from rocprofiler_record_counter_t
 * @param [out] counter_id counter id associated with the record
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                    rocprofiler_counter_id_t*         counter_id)
{
    // Get counter id from record
    *counter_id = rocprofiler::counters::rec_to_counter_id(id);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_query_record_dimension_position(rocprofiler_counter_instance_id_t  id,
                                            rocprofiler_counter_dimension_id_t dim,
                                            size_t*                            pos)
{
    *pos = rocprofiler::counters::rec_to_dim_pos(
        id, static_cast<rocprofiler::counters::rocprofiler_profile_counter_instance_types>(dim));
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_iterate_counter_dimensions(rocprofiler_counter_id_t              id,
                                       rocprofiler_available_dimensions_cb_t info_cb,
                                       void*                                 user_data)
{
    if(rocprofiler::counters::get_dimension_cache().empty())
    {
        return ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED;
    }

    const auto* dims =
        rocprofiler::common::get_val(rocprofiler::counters::get_dimension_cache(), id.handle);
    if(!dims) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;

    // This is likely faster than a map lookup given the limited number of dims.
    rocprofiler::common::container::small_vector<rocprofiler_record_dimension_info_t, 6> user_dims;
    for(const auto& internal_dim : *dims)
    {
        auto& dim         = user_dims.emplace_back();
        dim.name          = internal_dim.name().c_str();
        dim.instance_size = internal_dim.size();
        dim.id            = static_cast<rocprofiler_counter_dimension_id_t>(internal_dim.type());
    }

    if(user_dims.empty())
    {
        return ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND;
    }

    info_cb(id, user_dims.data(), user_dims.size(), user_data);

    return ROCPROFILER_STATUS_SUCCESS;
}
}
