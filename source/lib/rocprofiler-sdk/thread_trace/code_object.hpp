// MIT License
//
// Copyright (c) 2024 ROCm Developer Tools
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

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa_api_trace.h>

#include <functional>
#include <mutex>
#include <set>

namespace rocprofiler
{
namespace thread_trace
{
namespace code_object
{
struct CodeobjCallbackRegistry
{
    using LoadCallback = std::function<void(rocprofiler_agent_id_t, uint64_t, uint64_t, uint64_t)>;
    using UnloadCallback = std::function<void(uint64_t)>;

    CodeobjCallbackRegistry(LoadCallback ld, UnloadCallback unld);
    virtual ~CodeobjCallbackRegistry();

    void        IterateLoaded() const;
    static void Load(rocprofiler_agent_id_t agent, uint64_t id, uint64_t addr, uint64_t size);
    static void Unload(uint64_t id);

private:
    LoadCallback   ld_fn;
    UnloadCallback unld_fn;

    static std::mutex                         mut;
    static std::set<CodeobjCallbackRegistry*> all_registries;
};

void
initialize(HsaApiTable* table);
}  // namespace code_object
}  // namespace thread_trace
}  // namespace rocprofiler
