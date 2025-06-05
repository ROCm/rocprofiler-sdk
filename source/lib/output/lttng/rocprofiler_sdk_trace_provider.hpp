// clang-format off
// rocprofiler_sdk_trace_provider.hpp

#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER rocprofv3_trace  // APP Provider Name

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "./rocprofiler_sdk_trace_provider.hpp"  // Must point to itself

#if !defined(_ROCPROFILER_SDK_TRACE_PROVIDER_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define _ROCPROFILER_SDK_TRACE_PROVIDER_H

#include <rocprofiler-sdk/fwd.h>

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

        const char**, rec_args_types,
        const char**, rec_args_values,
        size_t, rec_args_len
        ),
    TP_FIELDS(
        ctf_integer(uint64_t, pid, rec_pid)

        ctf_string(api_name, rec_api_name)

        ctf_integer(uint64_t, ancestor_correlation_id, rec_ancestor_correlation_id)
        ctf_integer(uint64_t, internal_correlation_id, rec_internal_correlation_id)

        ctf_integer(uint64_t, start_timestamp, rec_start_ts)
        ctf_integer(uint64_t, duration, rec_end_ts-rec_start_ts)

        ctf_integer(uint64_t, thread_id, rec_thread_id)

        ctf_sequence(uint8_t, args_types_payload, (const uint8_t*)&rec_args_types, uint64_t, rec_args_len*sizeof(const char*))
        ctf_sequence(uint8_t, args_values_payload, (const uint8_t*)&rec_args_values, uint64_t, rec_args_len*sizeof(const char*))
        ctf_integer(size_t, args_count, rec_args_len)
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
        ctf_integer(uint64_t, duration, rec_end_ts-rec_start_ts)

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
        ctf_integer(uint64_t, duration, rec_end_ts-rec_start_ts)

        ctf_integer(uint64_t, thread_id, rec_thread_id)
        ctf_integer(uint64_t, agent_node_id, rec_agent_id)
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
        ctf_integer(uint64_t, duration, rec_end_ts-rec_start_ts)

        ctf_integer(uint64_t, thread_id, rec_thread_id)
        ctf_integer(uint64_t, src_agent_node_id, rec_src_agent_id)
        ctf_integer(uint64_t, dst_agent_node_id, rec_dst_agent_id)
        ctf_integer(uint64_t, stream_id, rec_stream_id)

        ctf_integer(uint64_t, size, rec_size)
    )
)

// Define the tracepoint event for recording Marker API Traces
TRACEPOINT_EVENT(
    rocprofv3_trace,
    marker_api,
    TP_ARGS(
        uint64_t, rec_pid,

        uint32_t, rec_operation,

        uint64_t, rec_ancestor_correlation_id,
        uint64_t, rec_internal_correlation_id,

        uint64_t, rec_start_ts,
        uint64_t, rec_end_ts,

        uint64_t, rec_thread_id,

        const char*, rec_name
        ),
    TP_FIELDS(
        ctf_integer(uint64_t, pid, rec_pid)

        ctf_integer(uint32_t, operation, rec_operation)

        ctf_integer(uint64_t, ancestor_correlation_id, rec_ancestor_correlation_id)
        ctf_integer(uint64_t, internal_correlation_id, rec_internal_correlation_id)

        ctf_integer(uint64_t, start_timestamp, rec_start_ts)
        ctf_integer(uint64_t, duration, rec_end_ts-rec_start_ts)

        ctf_integer(uint64_t, thread_id, rec_thread_id)

        ctf_string(name, rec_name)
    )
)

// Define the tracepoint event for recording Scratch Memory Copy Traces
TRACEPOINT_EVENT(
    rocprofv3_trace,
    scratch_memory,
    TP_ARGS(
        uint64_t, rec_pid,

        uint32_t, rec_operation,

        uint64_t, rec_ancestor_correlation_id,
        uint64_t, rec_internal_correlation_id,

        uint64_t, rec_start_ts,
        uint64_t, rec_end_ts,

        uint64_t, rec_thread_id,
        uint64_t, rec_agent_id,
        uint64_t, rec_queue_id,

        uint32_t, rec_flags
        ),
    TP_FIELDS(
        ctf_integer(uint64_t, pid, rec_pid)

        ctf_integer(uint32_t, operation, rec_operation)

        ctf_integer(uint64_t, ancestor_correlation_id, rec_ancestor_correlation_id)
        ctf_integer(uint64_t, internal_correlation_id, rec_internal_correlation_id)

        ctf_integer(uint64_t, start_timestamp, rec_start_ts)
        ctf_integer(uint64_t, duration, rec_end_ts-rec_start_ts)

        ctf_integer(uint64_t, thread_id, rec_thread_id)
        ctf_integer(uint64_t, agent_node_id, rec_agent_id)
        ctf_integer(uint64_t, dst_queue_id, rec_queue_id)

        ctf_integer(uint64_t, flags, rec_flags)
    )
)

// Define the tracepoint event for recording RCCL API Traces
TRACEPOINT_EVENT(
    rocprofv3_trace,
    rccl_api,
    TP_ARGS(
        uint64_t, rec_pid,

        uint64_t, rec_ancestor_correlation_id,
        uint64_t, rec_internal_correlation_id,

        uint64_t, rec_start_ts,
        uint64_t, rec_end_ts,

        uint64_t, rec_thread_id,

        const char*, rec_name
        ),
    TP_FIELDS(
        ctf_integer(uint64_t, pid, rec_pid)

        ctf_integer(uint64_t, ancestor_correlation_id, rec_ancestor_correlation_id)
        ctf_integer(uint64_t, internal_correlation_id, rec_internal_correlation_id)

        ctf_integer(uint64_t, start_timestamp, rec_start_ts)
        ctf_integer(uint64_t, duration, rec_end_ts-rec_start_ts)

        ctf_integer(uint64_t, thread_id, rec_thread_id)

        ctf_string(name, rec_name)
    )
)

// Define the tracepoint event for recording Memory Allocation Traces
TRACEPOINT_EVENT(
    rocprofv3_trace,
    memory_allocation,
    TP_ARGS(
        uint64_t, rec_pid,

        uint64_t, rec_operation,

        uint64_t, rec_internal_correlation_id,

        uint64_t, rec_start_ts,
        uint64_t, rec_end_ts,

        uint64_t, rec_thread_id,
        uint64_t, rec_agent_id,
        uint64_t, rec_stream_id,

        // uint64_t, rec_ptr_address,
        rocprofiler_address_t, rec_address,
        uint64_t, rec_allocation_size
        ),
    TP_FIELDS(
        ctf_integer(uint64_t, pid, rec_pid)

        ctf_integer(uint64_t, operation, rec_operation)

        ctf_integer(uint64_t, internal_correlation_id, rec_internal_correlation_id)

        ctf_integer(uint64_t, start_timestamp, rec_start_ts)
        ctf_integer(uint64_t, duration, rec_end_ts-rec_start_ts)

        ctf_integer(uint64_t, thread_id, rec_thread_id)
        ctf_integer(uint64_t, agent_node_id, rec_agent_id)
        ctf_integer(uint64_t, stream_id, rec_stream_id)

        ctf_integer(uint64_t, ptr_value, rec_address.value)
        ctf_sequence(uint8_t, ptr, (const uint8_t*)(rec_address.ptr), uint64_t, sizeof(rec_address.ptr))
        ctf_integer(uint64_t, allocation_size, rec_allocation_size)
    )
)

// Define the tracepoint event for recording ROCDecode API Traces
TRACEPOINT_EVENT(
    rocprofv3_trace,
    rocdecode_api,
    TP_ARGS(
        uint64_t, rec_pid,

        uint64_t, rec_ancestor_correlation_id,
        uint64_t, rec_internal_correlation_id,

        uint64_t, rec_start_ts,
        uint64_t, rec_end_ts,

        uint64_t, rec_thread_id,

        const char*, rec_name,

        const char**, rec_args_types,
        const char**, rec_args_values,
        size_t, rec_args_len
        ),
    TP_FIELDS(
        ctf_integer(uint64_t, pid, rec_pid)

        ctf_integer(uint64_t, internal_correlation_id, rec_internal_correlation_id)

        ctf_integer(uint64_t, start_timestamp, rec_start_ts)
        ctf_integer(uint64_t, duration, rec_end_ts-rec_start_ts)

        ctf_integer(uint64_t, thread_id, rec_thread_id)

        ctf_string(name, rec_name)

        ctf_sequence(uint8_t, args_types_payload, (const uint8_t*)&rec_args_types, uint64_t, rec_args_len*sizeof(const char*))
        ctf_sequence(uint8_t, args_values_payload, (const uint8_t*)&rec_args_values, uint64_t, rec_args_len*sizeof(const char*))
        ctf_integer(size_t, args_count, rec_args_len)
    )
)

// Define the tracepoint event for recording ROCJPEG API Traces
TRACEPOINT_EVENT(
    rocprofv3_trace,
    rocjpeg_api,
    TP_ARGS(
        uint64_t, rec_pid,

        uint64_t, rec_ancestor_correlation_id,
        uint64_t, rec_internal_correlation_id,

        uint64_t, rec_start_ts,
        uint64_t, rec_end_ts,

        uint64_t, rec_thread_id,

        const char*, rec_name
        ),
    TP_FIELDS(
        ctf_integer(uint64_t, pid, rec_pid)

        ctf_integer(uint64_t, ancestor_correlation_id, rec_ancestor_correlation_id)
        ctf_integer(uint64_t, internal_correlation_id, rec_internal_correlation_id)

        ctf_integer(uint64_t, start_timestamp, rec_start_ts)
        ctf_integer(uint64_t, duration, rec_end_ts-rec_start_ts)

        ctf_integer(uint64_t, thread_id, rec_thread_id)

        ctf_string(name, rec_name)
    )
)

#endif /* _ROCPROFILER_SDK_TRACE_PROVIDER_H */

#include <lttng/tracepoint-event.h>  // Must be last

// clang-format on