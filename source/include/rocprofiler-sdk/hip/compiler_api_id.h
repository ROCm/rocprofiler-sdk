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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/version.h>

#include <hip/amd_detail/hip_api_trace.hpp>

/**
 * @brief ROCProfiler enumeration of HIP Compiler API tracing operations
 */
typedef enum  // NOLINT(performance-enum-size)
{
    ROCPROFILER_HIP_COMPILER_API_ID_NONE                      = -1,
    ROCPROFILER_HIP_COMPILER_API_ID___hipPopCallConfiguration = 0,
    ROCPROFILER_HIP_COMPILER_API_ID___hipPushCallConfiguration,
    ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterFatBinary,
    ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterFunction,
    ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterManagedVar,
    ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterSurface,
    ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterTexture,
    ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterVar,
    ROCPROFILER_HIP_COMPILER_API_ID___hipUnregisterFatBinary,
    ROCPROFILER_HIP_COMPILER_API_ID_LAST,
} rocprofiler_hip_compiler_api_id_t;
