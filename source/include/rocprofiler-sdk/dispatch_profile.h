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

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/profile_config.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup DISPATCH_PROFILE_COUNTING_SERVICE Dispatch Profile Counting Service
 * @brief Per-dispatch hardware counter collection service
 *
 * @{
 */

/**
 * @brief Kernel dispatch data for profile counting callbacks
 *
 */
typedef struct rocprofiler_profile_counting_dispatch_data_t
{
    uint64_t                     size;            ///< Size of this struct
    rocprofiler_kernel_id_t      kernel_id;       ///< Kernel identifier
    rocprofiler_agent_id_t       agent_id;        ///< Agent ID where kernel is launched
    rocprofiler_queue_id_t       queue_id;        ///< Queue ID where kernel packet is enqueued
    rocprofiler_correlation_id_t correlation_id;  ///< Correlation ID for this dispatch
    uint32_t                     private_segment_size;  /// runtime private memory segment size
    uint32_t                     group_segment_size;    /// runtime group memory segment size
    rocprofiler_dim3_t           workgroup_size;        /// runtime workgroup size (grid * threads)
    rocprofiler_dim3_t           grid_size;             /// runtime grid size
} rocprofiler_profile_counting_dispatch_data_t;

/**
 * @brief Kernel Dispatch Callback. This is a callback that is invoked before the kernel
 *        is enqueued into the HSA queue. What counters to collect for a kernel are set
 *        via passing back a profile config (config) in this callback. These counters
 *        will be collected and emplaced in the buffer with @ref rocprofiler_buffer_id_t used when
 *        setting up this callback.
 *
 * @param [in] dispatch_data      @see ::rocprofiler_profile_counting_dispatch_data_t
 * @param [out] config            Profile config detailing the counters to collect for this kernel
 * @param [out] user_data         User data unique to this dispatch. Returned in record callback
 * @param [in] callback_data_args Callback supplied via buffered_dispatch_profile_counting_service
 */
typedef void (*rocprofiler_profile_counting_dispatch_callback_t)(
    rocprofiler_profile_counting_dispatch_data_t dispatch_data,
    rocprofiler_profile_config_id_t*             config,
    rocprofiler_user_data_t*                     user_data,
    void*                                        callback_data_args);

/**
 * @brief Counting record callback. This is a callback is invoked when the kernel
 *        execution is complete and contains the counter profile data requested in
 *        @ref rocprofiler_profile_counting_dispatch_callback_t. Only used with
 *        @ref rocprofiler_configure_callback_dispatch_profile_counting_service.
 *
 * @param [in] dispatch_data      @see ::rocprofiler_profile_counting_dispatch_data_t
 * @param [in] record_data        Counter record data.
 * @param [in] record_count       Number of counter records.
 * @param [in] user_data          User data instance from dispatch callback
 * @param [in] callback_data_args Callback supplied via buffered_dispatch_profile_counting_service
 */
typedef void (*rocprofiler_profile_counting_record_callback_t)(
    rocprofiler_profile_counting_dispatch_data_t dispatch_data,
    rocprofiler_record_counter_t*                record_data,
    size_t                                       record_count,
    rocprofiler_user_data_t                      user_data,
    void*                                        callback_data_args);

/**
 * @brief Configure buffered dispatch profile Counting Service.
 *        Collects the counters in dispatch packets and stores them
 *        in a buffer with @p buffer_id. The buffer may contain packets from more than
 *        one dispatch (denoted by correlation id). Will trigger the
 *        callback based on the parameters setup in buffer_id_t.
 *
 *        NOTE: Interface is up for comment as to whether restrictions
 *        on agent should be made here (limiting the CB based on agent)
 *        or if the restriction should be performed by the tool in
 *        @ref rocprofiler_profile_counting_dispatch_callback_t (i.e.
 *        tool code checking the agent param to see if they want to profile
 *        it).
 *
 *        Interface is up for comment as to whether restrictions
 *        on agent should be made here (limiting the CB based on agent)
 *        or if the restriction should be performed by the tool in
 *        @ref rocprofiler_profile_counting_dispatch_callback_t (i.e.
 *        tool code checking the agent param to see if they want to profile
 *        it).
 *
 * @param [in] context_id context id
 * @param [in] buffer_id id of the buffer to use for the counting service
 * @param [in] callback callback to perform when dispatch is enqueued
 * @param [in] callback_data_args callback data
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_buffered_dispatch_profile_counting_service(
    rocprofiler_context_id_t                         context_id,
    rocprofiler_buffer_id_t                          buffer_id,
    rocprofiler_profile_counting_dispatch_callback_t callback,
    void*                                            callback_data_args);

/**
 * @brief Configure buffered dispatch profile Counting Service.
 *        Collects the counters in dispatch packets and calls a callback
 *        with the counters collected during that dispatch.
 *
 * @param [in] context_id context id
 * @param [in] dispatch_callback callback to perform when dispatch is enqueued
 * @param [in] dispatch_callback_args callback data for dispatch callback
 * @param [in] record_callback  Record callback for completed profile data
 * @param [in] record_callback_args Callback args for record callback
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_callback_dispatch_profile_counting_service(
    rocprofiler_context_id_t                         context_id,
    rocprofiler_profile_counting_dispatch_callback_t dispatch_callback,
    void*                                            dispatch_callback_args,
    rocprofiler_profile_counting_record_callback_t   record_callback,
    void*                                            record_callback_args);
/** @} */

ROCPROFILER_EXTERN_C_FINI
