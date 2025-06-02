// clang-format off
// rocprofiler_sdk_trace_provider.hpp

#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER rocprofv3_trace  // APP Provider Name

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "./rocprofiler_sdk_trace_provider.hpp"  // Must point to itself

#if !defined(_ROCPROFILER_SDK_TRACE_PROVIDER_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define _ROCPROFILER_SDK_TRACE_PROVIDER_H

#include <lttng/tracepoint.h>

// Define the tracepoint event for recording the agents
TRACEPOINT_EVENT(
    rocprofv3_trace,
    agents_info,
    TP_ARGS(
        uint64_t, rec_pid,

        uint32_t, rec_type,

        uint32_t, rec_node_id,
        int32_t, rec_logical_node_id,
        int32_t, rec_logical_node_type_id,
        uint64_t, rec_gpu_id,

        const char*, rec_name,
        const char*, rec_vendor_name,
        const char*, rec_product_name,
        const char*, rec_model_name
        ),
    TP_FIELDS(
        ctf_integer(uint64_t, pid, rec_pid)

        ctf_integer(uint32_t, type, rec_type)

        ctf_integer(uint32_t, node_id, rec_node_id)
        ctf_integer(int32_t, logical_node_id, rec_logical_node_id)
        ctf_integer(int32_t, logical_node_type_id, rec_logical_node_type_id)
        ctf_integer(uint64_t, gpu_id, rec_gpu_id)

        ctf_string(name, rec_name)
        ctf_string(vendor_name, rec_vendor_name)
        ctf_string(product_name, rec_product_name)
        ctf_string(model_name, rec_model_name)
    )
)

// Define the tracepoint event for recording HIP API calls
TRACEPOINT_EVENT(
    rocprofv3_trace,
    hip_api,
    TP_ARGS(
        uint64_t, rec_pid,

        const char*, rec_api_name,

        uint64_t, rec_ancestor_correlation_id,
        uint64_t, rec_internal_correlation_id,

        uint64_t, rec_start_ts,
        uint64_t, rec_end_ts,

        uint64_t, rec_thread_id,

        /* Raw dump of the rocprofiler_hip_api_args_t union */
        const uint8_t*, raw_args_data,  // Pointer to the start of the args union
        uint64_t, raw_args_size  // Size of the args union (sizeof)
        ),
    TP_FIELDS(
        ctf_integer(uint64_t, pid, rec_pid)

        ctf_string(api_name, rec_api_name)

        ctf_integer(uint64_t, ancestor_correlation_id, rec_ancestor_correlation_id)
        ctf_integer(uint64_t, internal_correlation_id, rec_internal_correlation_id)

        ctf_integer(uint64_t, start_timestamp, rec_start_ts)
        ctf_integer(uint64_t, end_timestamp, rec_end_ts)

        ctf_integer(uint64_t, thread_id, rec_thread_id)

        /* Raw dump of the args union */
        ctf_sequence(uint8_t, args_raw_payload, raw_args_data, uint64_t, raw_args_size)
    )
)

// Define the tracepoint event for recording HSA API calls
TRACEPOINT_EVENT(
    rocprofv3_trace,
    hsa_api,
    TP_ARGS(
        uint64_t, rec_pid,

        const char*, rec_api_name,

        uint64_t, rec_ancestor_correlation_id,
        uint64_t, rec_internal_correlation_id,

        uint64_t, rec_start_ts,
        uint64_t, rec_end_ts,

        uint64_t, rec_thread_id
        ),
    TP_FIELDS(
        ctf_integer(uint64_t, pid, rec_pid)

        ctf_string(api_name, rec_api_name)

        ctf_integer(uint64_t, ancestor_correlation_id, rec_ancestor_correlation_id)
        ctf_integer(uint64_t, internal_correlation_id, rec_internal_correlation_id)

        ctf_integer(uint64_t, start_timestamp, rec_start_ts)
        ctf_integer(uint64_t, end_timestamp, rec_end_ts)

        ctf_integer(uint64_t, thread_id, rec_thread_id)
    )
)

// Define the tracepoint event for recording Kernel Dispatchs
TRACEPOINT_EVENT(
    rocprofv3_trace,
    kernel_dispatch,
    TP_ARGS(
        uint64_t, rec_pid,

        uint64_t, rec_internal_correlation_id,

        uint64_t, rec_start_ts,
        uint64_t, rec_end_ts,

        uint64_t, rec_thread_id,
        uint64_t, rec_agent_id,
        uint64_t, rec_queue_id,
        uint64_t, rec_stream_id,

        // uint32_t, rec_private_segment_size,
        // uint32_t, rec_group_segment_size,

        // uint32_t, rec_workgroup_size_x,
        // uint32_t, rec_workgroup_size_y,
        // uint32_t, rec_workgroup_size_z,

        // uint32_t, rec_grid_size_x,
        // uint32_t, rec_grid_size_y,
        // uint32_t, rec_grid_size_z,

        const char*, rec_kernel_name
        ),
    TP_FIELDS(
        ctf_integer(uint64_t, pid, rec_pid)

        ctf_integer(uint64_t, internal_correlation_id, rec_internal_correlation_id)

        ctf_integer(uint64_t, start_timestamp, rec_start_ts)
        ctf_integer(uint64_t, end_timestamp, rec_end_ts)

        ctf_integer(uint64_t, thread_id, rec_thread_id)
        ctf_integer(uint64_t, agent_id, rec_agent_id)
        ctf_integer(uint64_t, queue_id, rec_queue_id)
        ctf_integer(uint64_t, stream_id, rec_stream_id)

        // ctf_integer(uint32_t, private_segment_size, rec_private_segment_size)
        // ctf_integer(uint32_t, group_segment_size, rec_group_segment_size)

        // ctf_integer(uint32_t, workgroup_size_x, rec_workgroup_size_x)
        // ctf_integer(uint32_t, workgroup_size_y, rec_workgroup_size_y)
        // ctf_integer(uint32_t, workgroup_size_z, rec_workgroup_size_z)

        // ctf_integer(uint32_t, grid_size_x, rec_grid_size_x)
        // ctf_integer(uint32_t, grid_size_y, rec_grid_size_y)
        // ctf_integer(uint32_t, grid_size_z, rec_grid_size_z)

        ctf_string(kernel_name, rec_kernel_name)
    )
)

// Define the tracepoint event for recording Memory Copy Traces
TRACEPOINT_EVENT(
    rocprofv3_trace,
    memory_copy,
    TP_ARGS(
        uint64_t, rec_pid,

        uint32_t, rec_operation,

        uint64_t, rec_internal_correlation_id,

        uint64_t, rec_start_ts,
        uint64_t, rec_end_ts,

        uint64_t, rec_thread_id,
        uint64_t, rec_src_agent_id,
        uint64_t, rec_dst_agent_id,
        uint64_t, rec_stream_id,

        uint64_t, rec_size
        ),
    TP_FIELDS(
        ctf_integer(uint64_t, pid, rec_pid)

        ctf_integer(uint32_t, operation, rec_operation)

        ctf_integer(uint64_t, internal_correlation_id, rec_internal_correlation_id)

        ctf_integer(uint64_t, start_timestamp, rec_start_ts)
        ctf_integer(uint64_t, end_timestamp, rec_end_ts)

        ctf_integer(uint64_t, thread_id, rec_thread_id)
        ctf_integer(uint64_t, src_agent_id, rec_src_agent_id)
        ctf_integer(uint64_t, dst_agent_id, rec_dst_agent_id)
        ctf_integer(uint64_t, stream_id, rec_stream_id)

        ctf_integer(uint64_t, size, rec_size)
    )
)

#endif /* _ROCPROFILER_SDK_TRACE_PROVIDER_H */

#include <lttng/tracepoint-event.h>  // Must be last

// clang-format on