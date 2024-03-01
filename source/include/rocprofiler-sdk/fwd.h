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

#pragma once

#include <rocprofiler-sdk/defines.h>

#include <stddef.h>
#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

//--------------------------------------------------------------------------------------//
//
//                                      ENUMERATIONS
//
//--------------------------------------------------------------------------------------//

/**
 * @defgroup BASIC_DATA_TYPES Basic data types
 * @brief Basic data types and typedefs
 *
 * @{
 */

// TODO(aelwazir): Do we need to add a null (way) for every handle?
// TODO(aelwazir): Remove API Data args from the doxygen?
// TODO(aelwazir): Not everything in bin needs to be installed bin, use libexec or share?

/**
 * @brief Status codes.
 */
typedef enum  // NOLINT(performance-enum-size)
{
    ROCPROFILER_STATUS_SUCCESS = 0,                ///< No error occurred
    ROCPROFILER_STATUS_ERROR,                      ///< Generalized error
    ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND,    ///< No valid context for given context id
    ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND,     ///< No valid buffer for given buffer id
    ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND,       ///< Kind identifier is invalid
    ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND,  ///< Operation identifier is invalid for domain
    ROCPROFILER_STATUS_ERROR_THREAD_NOT_FOUND,     ///< No valid thread for given thread id
    ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND,      ///< Agent identifier not found
    ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND,    ///< Counter identifier does not exist
    ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR,        ///< Generalized context error
    ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID,      ///< Context configuration is not valid
    ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED,  ///< Context was not started (e.g., atomic swap
                                                   ///< into active array failed)
    ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT,  ///< Context operation failed due to a conflict with
                                                ///< another context
    ROCPROFILER_STATUS_ERROR_BUFFER_BUSY,  ///< buffer operation failed because it currently busy
                                           ///< handling another request (e.g. flushing)
    ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED,  ///< service has already been configured
                                                          ///< in context
    ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED,        ///< Function call is not valid outside of
                                                          ///< rocprofiler configuration (i.e.
                                                          ///< function called post-initialization)
    ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED,             ///< Function is not implemented
    ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI,  ///< Data structure provided by user is incompatible
                                                ///< with current version of rocprofiler
    ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT,  ///< Function invoked with one or more invalid
                                                ///< arguments
    ROCPROFILER_STATUS_ERROR_METRIC_NOT_VALID_FOR_AGENT,  ///< Invalid metric supplied to agent.
    ROCPROFILER_STATUS_ERROR_FINALIZED,       ///< invalid because rocprofiler has been finalized
    ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED,  ///< Call requires HSA to be loaded before performed
    ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND,   ///< Dimension is not found for counter
    ROCPROFILER_STATUS_ERROR_PROFILE_COUNTER_NOT_FOUND,  ///< Profile could not find counter for GPU
                                                         ///< agent
    ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED,      ///< AST could not be generated correctly
    ROCPROFILER_STATUS_ERROR_AST_NOT_FOUND,              ///< AST was not found
    ROCPROFILER_STATUS_ERROR_AQL_NO_EVENT_COORD,  ///< Event coordinate was not found by AQL profile
    ROCPROFILER_STATUS_LAST,
} rocprofiler_status_t;

/**
 * @brief Buffer record categories. This enumeration type is encoded in @ref
 * rocprofiler_record_header_t category field
 */
typedef enum  // NOLINT(performance-enum-size)
{
    ROCPROFILER_BUFFER_CATEGORY_NONE = 0,
    ROCPROFILER_BUFFER_CATEGORY_TRACING,
    ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING,
    ROCPROFILER_BUFFER_CATEGORY_COUNTERS,
    ROCPROFILER_BUFFER_CATEGORY_LAST,
} rocprofiler_buffer_category_t;

/**
 * @brief Agent type.
 */
typedef enum  // NOLINT(performance-enum-size)
{
    ROCPROFILER_AGENT_TYPE_NONE = 0,  ///< Agent type is unknown
    ROCPROFILER_AGENT_TYPE_CPU,       ///< Agent type is a CPU
    ROCPROFILER_AGENT_TYPE_GPU,       ///< Agent type is a GPU
    ROCPROFILER_AGENT_TYPE_LAST,
} rocprofiler_agent_type_t;

/**
 * @brief Service Callback Phase.
 */
typedef enum  // NOLINT(performance-enum-size)
{
    ROCPROFILER_CALLBACK_PHASE_NONE = 0,  ///< Callback has no phase
    ROCPROFILER_CALLBACK_PHASE_ENTER,     ///< Callback invoked prior to function execution
    ROCPROFILER_CALLBACK_PHASE_LOAD =
        ROCPROFILER_CALLBACK_PHASE_ENTER,  ///< Callback invoked prior to code object loading
    ROCPROFILER_CALLBACK_PHASE_EXIT,       ///< Callback invoked after to function execution
    ROCPROFILER_CALLBACK_PHASE_UNLOAD =
        ROCPROFILER_CALLBACK_PHASE_EXIT,  ///< Callback invoked prior to code object unloading
    ROCPROFILER_CALLBACK_PHASE_LAST,
} rocprofiler_callback_phase_t;

/**
 * @brief Service Callback Tracing Kind. @see rocprofiler_configure_callback_tracing_service.
 */
typedef enum  // NOLINT(performance-enum-size)
{
    ROCPROFILER_CALLBACK_TRACING_NONE = 0,
    ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,       ///< @see ::rocprofiler_hsa_core_api_id_t
    ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,    ///< @see ::rocprofiler_hsa_amd_ext_api_id_t
    ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API,  ///< @see ::rocprofiler_hsa_image_ext_api_id_t
    ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API,  ///< @see
                                                        ///< ::rocprofiler_hsa_finalize_ext_api_id_t
    ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,       ///< @see ::rocprofiler_hip_runtime_api_id_t
    ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,    ///< @see ::rocprofiler_hip_compiler_api_id_t
    ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,     ///< @see ::rocprofiler_marker_core_api_id_t
    ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,  ///< @see
                                                      ///< ::rocprofiler_marker_control_api_id_t
    ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API,     ///< @see ::rocprofiler_marker_name_api_id_t
    ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,      ///< @see ::rocprofiler_code_object_operation_t
    ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,  ///< Callbacks for kernel dispatches
    ROCPROFILER_CALLBACK_TRACING_LAST,
} rocprofiler_callback_tracing_kind_t;

/**
 * @brief Service Buffer Tracing Kind. @see rocprofiler_configure_buffer_tracing_service.
 */
typedef enum  // NOLINT(performance-enum-size)
{
    ROCPROFILER_BUFFER_TRACING_NONE = 0,
    ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,          ///< @see ::rocprofiler_hsa_core_api_id_t
    ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,       ///< @see ::rocprofiler_hsa_amd_ext_api_id_t
    ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API,     ///< @see ::rocprofiler_hsa_image_ext_api_id_t
    ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API,  ///< @see
                                                      ///< ::rocprofiler_hsa_finalize_ext_api_id_t
    ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,       ///< @see ::rocprofiler_hip_runtime_api_id_t
    ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API,      ///< @see ::rocprofiler_hip_compiler_api_id_t
    ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API,       ///< @see ::rocprofiler_marker_core_api_id_t
    ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API,  ///< @see ::rocprofiler_marker_control_api_id_t
    ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API,     ///< @see ::rocprofiler_marker_name_api_id_t
    ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,         ///< @see ::rocprofiler_memory_copy_operation_t
    ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,     ///< Buffer kernel dispatch info
    ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION,      ///< Buffer page migration info
    ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY,      ///< Buffer scratch memory reclaimation info
    ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT,  ///< Correlation ID in no longer in use
    ROCPROFILER_BUFFER_TRACING_LAST,
} rocprofiler_buffer_tracing_kind_t;

/**
 * @brief ROCProfiler Code Object Tracer Operation.
 */
typedef enum  // NOLINT(performance-enum-size)
{
    ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_NONE = 0,  ///< Unknown code object operation
    ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LOAD,      ///< Code object containing kernel symbols
    ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER,  ///< Kernel symbols
    ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LAST,
} rocprofiler_code_object_operation_t;

/**
 * @brief Memory Copy Operation.
 */
typedef enum  // NOLINT(performance-enum-size)
{
    ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_NONE = 0,          ///< Unknown memory copy direction
    ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_HOST_TO_HOST,      ///< Memory copy from host to host
    ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_HOST_TO_DEVICE,    ///< Memory copy from host to device
    ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_DEVICE_TO_HOST,    ///< Memory copy from device to host
    ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_DEVICE_TO_DEVICE,  ///< Memory copy from device to device
    ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_LAST,
} rocprofiler_memory_copy_operation_t;

/**
 * @brief PC Sampling Method.
 */
typedef enum  // NOLINT(performance-enum-size)
{
    ROCPROFILER_PC_SAMPLING_METHOD_NONE = 0,    ///< Unknown sampling type
    ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC,  ///< Stochastic sampling (MI300+)
    ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,   ///< Interval sampling (MI200+)
    ROCPROFILER_PC_SAMPLING_METHOD_LAST,
} rocprofiler_pc_sampling_method_t;

/**
 * @brief PC Sampling Unit.
 */
typedef enum  // NOLINT(performance-enum-size)
{
    ROCPROFILER_PC_SAMPLING_UNIT_NONE = 0,      ///< Sample interval has unspecified units
    ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS,  ///< Sample interval is in instructions
    ROCPROFILER_PC_SAMPLING_UNIT_CYCLES,        ///< Sample interval is in cycles
    ROCPROFILER_PC_SAMPLING_UNIT_TIME,          ///< Sample internval is in nanoseconds
    ROCPROFILER_PC_SAMPLING_UNIT_LAST,
} rocprofiler_pc_sampling_unit_t;

/**
 * @brief Actions when Buffer is full.
 */
typedef enum  // NOLINT(performance-enum-size)
{
    ROCPROFILER_BUFFER_POLICY_NONE = 0,  ///< No policy has been set
    ROCPROFILER_BUFFER_POLICY_DISCARD,   ///< Drop records when buffer is full
    ROCPROFILER_BUFFER_POLICY_LOSSLESS,  ///< Block when buffer is full
    ROCPROFILER_BUFFER_POLICY_LAST,
} rocprofiler_buffer_policy_t;

/**
 * @brief Enumeration for specifying runtime libraries supported by rocprofiler. This enumeration is
 * used for thread creation callbacks. @see INTERNAL_THREADING.
 */
typedef enum
{
    ROCPROFILER_LIBRARY        = (1 << 0),
    ROCPROFILER_HSA_LIBRARY    = (1 << 1),
    ROCPROFILER_HIP_LIBRARY    = (1 << 2),
    ROCPROFILER_MARKER_LIBRARY = (1 << 3),
    ROCPROFILER_LIBRARY_LAST   = ROCPROFILER_MARKER_LIBRARY,
} rocprofiler_runtime_library_t;

/**
 * @brief Enumeration for specifying intercept tables supported by rocprofiler. This enumeration is
 * used for intercept tables. @see INTERCEPT_TABLE.
 */
typedef enum
{
    ROCPROFILER_HSA_TABLE            = (1 << 0),
    ROCPROFILER_HIP_RUNTIME_TABLE    = (1 << 1),
    ROCPROFILER_HIP_COMPILER_TABLE   = (1 << 2),
    ROCPROFILER_MARKER_CORE_TABLE    = (1 << 3),
    ROCPROFILER_MARKER_CONTROL_TABLE = (1 << 4),
    ROCPROFILER_MARKER_NAME_TABLE    = (1 << 5),
    ROCPROFILER_TABLE_LAST           = ROCPROFILER_MARKER_NAME_TABLE,
} rocprofiler_intercept_table_t;

/**
 * @brief Enumeration for specifying the data type contained within the union.
 */
typedef enum
{
    ROCPROFILER_UNION_TYPE_NONE = 0,  ///< No union type
    ROCPROFILER_UNION_TYPE_STRING,    ///< String Type set
    ROCPROFILER_UNION_TYPE_INT,       ///< Integer Type Set
    ROCPROFILER_UNION_TYPE_LAST,
} rocprofiler_union_type_t;

/**
 * @brief Enumeration for specifying the counter info struct version you want.
 */
typedef enum
{
    ROCPROFILER_COUNTER_INFO_VERSION_NONE,
    ROCPROFILER_COUNTER_INFO_VERSION_0,  ///< @see rocprofiler_counter_info_v0_t
    ROCPROFILER_COUNTER_INFO_VERSION_LAST,
} rocprofiler_counter_info_version_id_t;
//--------------------------------------------------------------------------------------//
//
//                                      ALIASES
//
//--------------------------------------------------------------------------------------//

/**
 * @brief ROCProfiler Timestamp.
 */
typedef uint64_t rocprofiler_timestamp_t;

/**
 * @brief ROCProfiler Address.
 */
typedef uint64_t rocprofiler_address_t;

/**
 * @brief Thread ID. Value will be equivalent to `syscall(__NR_gettid)`
 */
typedef uint64_t rocprofiler_thread_id_t;

/**
 * @brief Tracing Operation ID. Depending on the kind, operations can be determined.
 * If the value is equal to zero that means all operations will be considered
 * for tracing.
 */
typedef uint32_t rocprofiler_tracing_operation_t;

/**
 * @brief Kernel identifier type
 *
 */
typedef uint64_t rocprofiler_kernel_id_t;

// forward declaration of struct
typedef struct rocprofiler_pc_sampling_configuration_s rocprofiler_pc_sampling_configuration_t;
typedef struct rocprofiler_pc_sampling_record_s        rocprofiler_pc_sampling_record_t;

/**
 * @brief Unique record id encoding both the counter
 *        and dimensional values (positions) for the record.
 */
typedef uint64_t rocprofiler_counter_instance_id_t;

/**
 * @brief A dimension for counter instances. Some example
 *        dimensions include XCC, SM (Shader), etc. This
 *        value represents the dimension beind described
 *        or queried about.
 */
typedef uint64_t rocprofiler_counter_dimension_id_t;

//--------------------------------------------------------------------------------------//
//
//                                      UNIONS
//
//--------------------------------------------------------------------------------------//

/**
 * @brief User-assignable data type
 *
 */
typedef union rocprofiler_user_data_t
{
    uint64_t value;  ///< usage example: set to process id, thread id, etc.
    void*    ptr;    ///< usage example: set to address of data allocation
} rocprofiler_user_data_t;

//--------------------------------------------------------------------------------------//
//
//                                      STRUCTS
//
//--------------------------------------------------------------------------------------//

/**
 * @brief Context ID.
 */
typedef struct
{
    uint64_t handle;
} rocprofiler_context_id_t;

/**
 * @brief Queue ID.
 */
typedef struct
{
    uint64_t handle;
} rocprofiler_queue_id_t;

/**
 * @brief ROCProfiler Record Correlation ID.
 */
typedef struct
{
    uint64_t                internal;
    rocprofiler_user_data_t external;
} rocprofiler_correlation_id_t;

/**
 * @struct rocprofiler_buffer_id_t
 * @brief Buffer ID.
 */
typedef struct
{
    uint64_t handle;
} rocprofiler_buffer_id_t;

/**
 * @brief Agent Identifier
 */
typedef struct
{
    uint64_t handle;
} rocprofiler_agent_id_t;

/**
 * @brief Counter ID.
 */
typedef struct
{
    uint64_t handle;
} rocprofiler_counter_id_t;

/**
 * @brief Profile Configurations
 * @see rocprofiler_create_profile_config for how to create.
 */
typedef struct
{
    uint64_t handle;  // Opaque handle
} rocprofiler_profile_config_id_t;

/**
 * @brief Multi-dimensional struct of data used to describe GPU workgroup and grid sizes
 */
typedef struct rocprofiler_dim3_t
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
} rocprofiler_dim3_t;

/**
 * @brief Tracing record
 *
 */
typedef struct rocprofiler_callback_tracing_record_t
{
    rocprofiler_context_id_t            context_id;
    rocprofiler_thread_id_t             thread_id;
    rocprofiler_correlation_id_t        correlation_id;
    rocprofiler_callback_tracing_kind_t kind;
    uint32_t                            operation;
    rocprofiler_callback_phase_t        phase;
    void*                               payload;
} rocprofiler_callback_tracing_record_t;

/**
 * @brief Generic record with type identifier(s) and a pointer to data. This data type is used with
 * buffered data.
 *
 * @code{.cpp}
 * void
 * tool_tracing_callback(rocprofiler_record_header_t** headers,
 *                       size_t                        num_headers)
 * {
 *     for(size_t i = 0; i < num_headers; ++i)
 *     {
 *         rocprofiler_record_header_t* header = headers[i];
 *
 *         if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
 *            header->kind == ROCPROFILER_BUFFER_TRACING_HSA_API)
 *         {
 *             // cast to rocprofiler_buffer_tracing_hsa_api_record_t which
 *             // is type associated with this category + kind
 *             auto* record =
 *                 static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload);
 *
 *             // trivial test
 *             assert(record->start_timestamp <= record->end_timestamp);
 *         }
 *     }
 * }
 *
 * @endcode
 */
typedef struct
{
    union
    {
        struct
        {
            uint32_t category;  ///< rocprofiler_buffer_category_t
            uint32_t kind;      ///< domain
        };
        uint64_t hash;  ///< generic identifier. You can compute this via: `uint64_t hash = category
                        ///< | ((uint64_t)(kind) << 32)`, e.g.
    };
    void* payload;
} rocprofiler_record_header_t;

/**
 * @brief Function for computing the unsigned 64-bit hash value in @ref rocprofiler_record_header_t
 * from a category and kind (two unsigned 32-bit values)
 *
 * @param [in] category a value from @ref rocprofiler_buffer_category_t
 * @param [in] kind depending on the category, this is the domain value, e.g., @ref
 * rocprofiler_buffer_tracing_kind_t value
 * @return uint64_t hash value of category and kind
 */
static inline uint64_t
rocprofiler_record_header_compute_hash(uint32_t category, uint32_t kind)
{
    uint64_t value = category;
    value |= ((uint64_t)(kind)) << 32;
    return value;
}

/**
 * @brief Details for the dimension, including its size, for a counter record.
 */
typedef struct
{
    const char*                        name;
    size_t                             instance_size;
    rocprofiler_counter_dimension_id_t id;

    /// @var id
    /// @brief Id for this dimension used by @ref rocprofiler_query_record_dimension_position
} rocprofiler_record_dimension_info_t;

/**
 * @brief ROCProfiler Profile Counting Counter Record per instance.
 */
typedef struct
{
    rocprofiler_counter_instance_id_t id;             ///< counter identifier
    double                            counter_value;  ///< counter value
    rocprofiler_correlation_id_t      correlation_id;

    /// @var correlation_id
    /// @brief Used to correlate the kernel data to an API call
} rocprofiler_record_counter_t;

/**
 * @brief Counter info struct version 0
 */
typedef struct
{
    int         is_derived;   ///< If this counter is a derived counter
    const char* name;         ///< Name of the counter
    const char* description;  ///< Description of the counter
    const char* block;        ///< Block of the counter (non-derived only)
    const char* expression;   ///< Counter expression (derived counters only)
} rocprofiler_counter_info_v0_t;

/**
 * @brief ROCProfiler SPM Record.
 *
 */
typedef struct
{
    /**
     * Counters, including identifiers to get counter information and Counters
     * values
     */
    rocprofiler_record_counter_t* counters;
    uint64_t                      counters_count;
} rocprofiler_spm_record_t;

/** @} */

ROCPROFILER_EXTERN_C_FINI
