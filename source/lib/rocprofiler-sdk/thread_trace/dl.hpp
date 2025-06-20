// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/thread_trace/trace_decoder_api.h"

#include <memory>

namespace rocprofiler
{
namespace thread_trace
{
class DL
{
    using parse_fn_t  = decltype(rocprof_trace_decoder_parse_data);
    using info_fn_t   = decltype(rocprof_trace_decoder_get_info_string);
    using status_fn_t = decltype(rocprof_trace_decoder_get_status_string);

public:
    DL(const char* libpath);
    ~DL();
    DL(DL&)        = delete;
    DL(DL&& other) = delete;

    bool valid() const
    {
        return handle != nullptr && parse != nullptr && info != nullptr && status != nullptr;
    };

    parse_fn_t*  parse  = nullptr;
    info_fn_t*   info   = nullptr;
    status_fn_t* status = nullptr;
    void*        handle = nullptr;

    uint32_t version_major = 0;
    uint32_t version_minor = 0;
    uint32_t version_patch = 0;
    uint64_t version       = 0;
};

}  // namespace thread_trace
}  // namespace rocprofiler
