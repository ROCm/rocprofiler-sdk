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
#include <rocprofiler-sdk/fwd.h>

/**
 * @defgroup AGENT_PROFILE_COUNTING_SERVICE Agent Profile Counting Service
 * @brief needs brief description
 *
 * @{
 */
ROCPROFILER_EXTERN_C_INIT

/**
 * @brief Callback to set the profile config for the agent.
 *
 * @param [in] context_id context id
 * @param [in] config_id Profile config detailing the counters to collect for this kernel
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_PROFILE_NOT_FOUND Returned if the config_id is not found
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID Returned if the ctx is not valid
 * @retval ::ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED Returned if attempting to make this
 * call outside of context startup.
 * @retval ::ROCPROFILER_STATUS_ERROR_AGENT_MISMATCH Agent of profile does not match agent of the
 * context.
 * @retval ::ROCPROFILER_STATUS_SUCCESS Returned if succesfully configured
 */
typedef rocprofiler_status_t (*rocprofiler_agent_set_profile_callback_t)(
    rocprofiler_context_id_t        context_id,
    rocprofiler_profile_config_id_t config_id);

/**
 * @brief Configure Profile Counting Service for agent. Called when the context is started.
 *        Selects the counters to be used for agent profiling.
 *
 * @param [in]  context_id context id
 * @param [in]  agent_id agent id
 * @param [in]  set_config Function to call to set the profile config (see
 * rocprofiler_agent_set_profile_callback_t)
 * @param [in]  user_data Data supplied to rocprofiler_configure_agent_profile_counting_service
 */
typedef void (*rocprofiler_agent_profile_callback_t)(
    rocprofiler_context_id_t                 context_id,
    rocprofiler_agent_id_t                   agent_id,
    rocprofiler_agent_set_profile_callback_t set_config,
    void*                                    user_data);

/**
 * @brief Configure Profile Counting Service for agent. There may only be one agent profile
 * configured per context and can be only one active context that is profiling a single agent
 * at a time. Multiple agent contexts can be started at the same time if they are profiling
 * different agents.
 *
 * @param [in] context_id context id
 * @param [in] buffer_id id of the buffer to use for the counting service. When
 * rocprofiler_sample_agent_profile_counting_service is called, counter data will be written
 * to this buffer.
 * @param [in] agent_id agent to configure profiling on.
 * @param [in] cb Callback called when the context is started for the tool to specify what
 * counters to collect (rocprofiler_profile_config_id_t).
 * @param [in] user_data User supplied data to be passed to the callback cb when triggered
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID Returned if the context does not exist.
 * @retval ::ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND Returned if the buffer is not found.
 * @retval ::ROCPROFILER_STATUS_SUCCESS Returned if succesfully configured
 */
rocprofiler_status_t
rocprofiler_configure_agent_profile_counting_service(rocprofiler_context_id_t context_id,
                                                     rocprofiler_buffer_id_t  buffer_id,
                                                     rocprofiler_agent_id_t   agent_id,
                                                     rocprofiler_agent_profile_callback_t cb,
                                                     void*                                user_data)
    ROCPROFILER_NONNULL(4) ROCPROFILER_API;

/**
 * @brief Trigger a read of the counter data for the agent profile. The counter data will be
 * written to the buffer specified in rocprofiler_configure_agent_profile_counting_service.
 * The data in rocprofiler_user_data_t will be written to the buffer along with the counter data.
 * flags can be used to specify if this call should be performed asynchronously (default is
 * synchronous).
 *
 * @param [in] context_id context id
 * @param [in] user_data User supplied data, included in records outputted to buffer.
 * @param [in] flags Flags to specify how the counter data should be collected (defaults to sync).
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID Returned if the context does not exist or
 * the context is not configured for agent profiling.
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR Returned if another operation is in progress (
 * start/stop ctx or another read).
 * @retval ::ROCPROFILER_STATUS_ERROR Returned if HSA has not been initialized yet.
 * @retval ::ROCPROFILER_STATUS_SUCCESS Returned if read request was successful.
 */
rocprofiler_status_t
rocprofiler_sample_agent_profile_counting_service(rocprofiler_context_id_t   context_id,
                                                  rocprofiler_user_data_t    user_data,
                                                  rocprofiler_counter_flag_t flags) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
