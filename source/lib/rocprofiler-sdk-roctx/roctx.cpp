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

#include <rocprofiler-sdk-roctx/api_trace.h>
#include <rocprofiler-sdk-roctx/defines.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <rocprofiler-sdk-roctx/types.h>

#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"

#include <glog/logging.h>
#include <rocprofiler-register/rocprofiler-register.h>

#include <array>
#include <atomic>
#include <cassert>

#define ROCP_REG_VERSION                                                                           \
    ROCPROFILER_REGISTER_COMPUTE_VERSION_3(                                                        \
        ROCTX_VERSION_MAJOR, ROCTX_VERSION_MINOR, ROCTX_VERSION_PATCH)

ROCPROFILER_REGISTER_DEFINE_IMPORT(roctx, ROCP_REG_VERSION)

namespace roctx
{
namespace
{
namespace common = ::rocprofiler::common;

constexpr size_t
compute_table_offset(size_t n)
{
    return (sizeof(uint64_t) + (n * sizeof(void*)));
}

constexpr size_t
compute_table_size(size_t nmembers)
{
    return (sizeof(uint64_t) + (nmembers * sizeof(void*)));
}

#define ROCTX_ASSERT_OFFSET(TABLE, MEMBER, IDX)                                                    \
    static_assert(offsetof(TABLE, MEMBER) == compute_table_offset(IDX),                            \
                  "Do not re-arrange the table members")

// core
ROCTX_ASSERT_OFFSET(roctxCoreApiTable_t, roctxMarkA_fn, 0);
ROCTX_ASSERT_OFFSET(roctxCoreApiTable_t, roctxRangePushA_fn, 1);
ROCTX_ASSERT_OFFSET(roctxCoreApiTable_t, roctxRangePop_fn, 2);
ROCTX_ASSERT_OFFSET(roctxCoreApiTable_t, roctxRangeStartA_fn, 3);
ROCTX_ASSERT_OFFSET(roctxCoreApiTable_t, roctxRangeStop_fn, 4);
ROCTX_ASSERT_OFFSET(roctxCoreApiTable_t, roctxGetThreadId_fn, 5);
// control
ROCTX_ASSERT_OFFSET(roctxControlApiTable_t, roctxProfilerPause_fn, 0);
ROCTX_ASSERT_OFFSET(roctxControlApiTable_t, roctxProfilerResume_fn, 1);
// name
ROCTX_ASSERT_OFFSET(roctxNameApiTable_t, roctxNameOsThread_fn, 0);
ROCTX_ASSERT_OFFSET(roctxNameApiTable_t, roctxNameHsaAgent_fn, 1);
ROCTX_ASSERT_OFFSET(roctxNameApiTable_t, roctxNameHipDevice_fn, 2);
ROCTX_ASSERT_OFFSET(roctxNameApiTable_t, roctxNameHipStream_fn, 3);

#undef ROCTX_ASSERT_OFFSET

static_assert(
    sizeof(roctxCoreApiTable_t) == compute_table_size(6),
    "Update table major/step version and add a new offset assertion if this fails to compile");

static_assert(
    sizeof(roctxControlApiTable_t) == compute_table_size(2),
    "Update table major/step version and add a new offset assertion if this fails to compile");

static_assert(
    sizeof(roctxNameApiTable_t) == compute_table_size(4),
    "Update table major/step version and add a new offset assertion if this fails to compile");

auto&
get_nested_range_level()
{
    static thread_local int value = 0;
    return value;
}

auto&
get_start_stop_range_id()
{
    static auto value = std::atomic<roctx_range_id_t>{};
    return value;
}

int
roctxGetThreadId(roctx_thread_id_t* tid)
{
    *tid = rocprofiler::common::get_tid();
    return 0;
}

void
roctxMarkA(const char*)
{}

int
roctxRangePushA(const char*)
{
    return get_nested_range_level()++;
}

int
roctxRangePop()
{
    if(get_nested_range_level() == 0) return -1;
    return --get_nested_range_level();
}

roctx_range_id_t
roctxRangeStartA(const char*)
{
    auto range_id = ++get_start_stop_range_id();
    return range_id;
}

void roctxRangeStop(roctx_range_id_t) {}

int roctxProfilerPause(roctx_thread_id_t /*tid*/) { return 0; }

int roctxProfilerResume(roctx_thread_id_t /*tid*/) { return 0; }

int
roctxNameOsThread(const char*)
{
    return 0;
}

int
roctxNameHsaAgent(const char*, const struct hsa_agent_s*)
{
    return 0;
}

int
roctxNameHipDevice(const char*, int)
{
    return 0;
}

int
roctxNameHipStream(const char*, const struct ihipStream_t*)
{
    return 0;
}

struct roctx_api_table
{
    roctxCoreApiTable_t    core    = common::init_public_api_struct(roctxCoreApiTable_t{});
    roctxControlApiTable_t control = common::init_public_api_struct(roctxControlApiTable_t{});
    roctxNameApiTable_t    name    = common::init_public_api_struct(roctxNameApiTable_t{});
};

auto*&
get_table_impl()
{
    rocprofiler::common::init_logging("ROCTX_LOG_LEVEL");

    auto*& tbl = rocprofiler::common::static_object<roctx_api_table>::construct();

    tbl->core = roctxCoreApiTable_t{sizeof(roctxCoreApiTable_t),
                                    &::roctx::roctxMarkA,
                                    &::roctx::roctxRangePushA,
                                    &::roctx::roctxRangePop,
                                    &::roctx::roctxRangeStartA,
                                    &::roctx::roctxRangeStop,
                                    &::roctx::roctxGetThreadId};

    tbl->control = roctxControlApiTable_t{sizeof(roctxControlApiTable_t),
                                          &::roctx::roctxProfilerPause,
                                          &::roctx::roctxProfilerResume};

    tbl->name = roctxNameApiTable_t{sizeof(roctxNameApiTable_t),
                                    &::roctx::roctxNameOsThread,
                                    &::roctx::roctxNameHsaAgent,
                                    &::roctx::roctxNameHipDevice,
                                    &::roctx::roctxNameHipStream};

    auto table_array = std::array<void*, 3>{&tbl->core, &tbl->control, &tbl->name};
    auto lib_id      = rocprofiler_register_library_indentifier_t{};
    auto rocp_reg_status =
        rocprofiler_register_library_api_table("roctx",
                                               &ROCPROFILER_REGISTER_IMPORT_FUNC(roctx),
                                               ROCP_REG_VERSION,
                                               table_array.data(),
                                               table_array.size(),
                                               &lib_id);

    LOG(INFO) << "[rocprofiler-sdk-roctx][" << getpid() << "] rocprofiler-register returned code "
              << rocp_reg_status << ": " << rocprofiler_register_error_string(rocp_reg_status);

    LOG_IF(WARNING, rocp_reg_status != ROCP_REG_SUCCESS && rocp_reg_status != ROCP_REG_NO_TOOLS)
        << "[rocprofiler-sdk-roctx][" << getpid()
        << "] rocprofiler-register failed with error code " << rocp_reg_status << ": "
        << rocprofiler_register_error_string(rocp_reg_status);

    return tbl;
}

const auto*
get_table()
{
    static auto*& tbl = get_table_impl();
    return tbl;
}
}  // namespace
}  // namespace roctx

ROCTX_EXTERN_C_INIT

void
roctxMarkA(const char* message)
{
    ::roctx::get_table()->core.roctxMarkA_fn(message);
}

int
roctxRangePushA(const char* message)
{
    return ::roctx::get_table()->core.roctxRangePushA_fn(message);
}

int
roctxRangePop()
{
    return ::roctx::get_table()->core.roctxRangePop_fn();
}

roctx_range_id_t
roctxRangeStartA(const char* message)
{
    return ::roctx::get_table()->core.roctxRangeStartA_fn(message);
}

void
roctxRangeStop(roctx_range_id_t id)
{
    return ::roctx::get_table()->core.roctxRangeStop_fn(id);
}

int
roctxGetThreadId(roctx_thread_id_t* tid)
{
    return ::roctx::get_table()->core.roctxGetThreadId_fn(tid);
}

int
roctxProfilerPause(roctx_thread_id_t tid)
{
    return ::roctx::get_table()->control.roctxProfilerPause_fn(tid);
}

int
roctxProfilerResume(roctx_thread_id_t tid)
{
    return ::roctx::get_table()->control.roctxProfilerResume_fn(tid);
}

int
roctxNameOsThread(const char* name)
{
    return ::roctx::get_table()->name.roctxNameOsThread_fn(name);
}

int
roctxNameHsaAgent(const char* name, const struct hsa_agent_s* agent)
{
    return ::roctx::get_table()->name.roctxNameHsaAgent_fn(name, agent);
}

int
roctxNameHipDevice(const char* name, int device_id)
{
    return ::roctx::get_table()->name.roctxNameHipDevice_fn(name, device_id);
}

int
roctxNameHipStream(const char* name, const struct ihipStream_t* stream)
{
    return ::roctx::get_table()->name.roctxNameHipStream_fn(name, stream);
}

ROCTX_EXTERN_C_FINI
