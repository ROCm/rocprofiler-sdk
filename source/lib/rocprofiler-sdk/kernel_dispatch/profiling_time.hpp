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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <hsa/hsa.h>

#include <cstdint>
#include <optional>

namespace rocprofiler
{
namespace kernel_dispatch
{
struct profiling_time
{
    hsa_status_t status = HSA_STATUS_ERROR_INVALID_SIGNAL;
    uint64_t     start  = 0;
    uint64_t     end    = 0;

    profiling_time& operator+=(uint64_t offset);
    profiling_time& operator-=(uint64_t offset);
    profiling_time& operator*=(uint64_t scale);
};

// get the profiling time for a signal on an agent, if start time is less than baseline, correct to
// start at baseline. If kernel_id is provided, it will be included in error log message if there is
// an issue with
profiling_time
get_dispatch_time(hsa_agent_t             agent,
                  hsa_signal_t            signal,
                  rocprofiler_kernel_id_t kernel_id,
                  std::optional<uint64_t> baseline = {});
}  // namespace kernel_dispatch
}  // namespace rocprofiler
