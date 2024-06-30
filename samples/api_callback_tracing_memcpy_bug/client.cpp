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

// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

/**
 * @file samples/api_callback_tracing/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include "client.hpp"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "common/call_stack.hpp"
#include "common/defines.hpp"
#include "common/filesystem.hpp"
#include "common/name_info.hpp"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <ratio>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>
namespace client
{
namespace
{
using common::call_stack_t;
using common::callback_name_info;
using common::source_location;

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx       = {};

void
print_call_stack(const call_stack_t& _call_stack)
{
    common::print_call_stack("api_callback_trace.log", _call_stack);
}

void
tool_tracing_ctrl_callback(rocprofiler_callback_tracing_record_t record,
                           rocprofiler_user_data_t*,
                           void* client_data)
{
    auto* ctx = static_cast<rocprofiler_context_id_t*>(client_data);

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER &&
       record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API &&
       record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause)
    {
        ROCPROFILER_CALL(rocprofiler_stop_context(*ctx), "pausing client context");
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
            record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API &&
            record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume)
    {
        ROCPROFILER_CALL(rocprofiler_start_context(*ctx), "resuming client context");
    }
}


#define ROCPROFILER_CALL2(fn, args, msg) /* return status using GNU ext */     \
({                                                                            \
   rocprofiler_status_t __status = fn args;                                   \
   if (__status != ROCPROFILER_STATUS_SUCCESS) {                              \
     const char *__status_msg = rocprofiler_get_status_string(__status);      \
     fprintf(stderr, "hpcrun: rocprofiler failure '%s' "                      \
             " status = %d, status_msg = '%s'\n", msg, __status,              \
             __status_msg);                                                   \
     exit(-1);                                                                \
  }                                                                           \
  __status;                                                                   \
})


rocprofiler_thread_id_t
rocm_threads_self
(
  void
)
{
  rocprofiler_thread_id_t my_thread_id;

  ROCPROFILER_CALL2
  (
    rocprofiler_get_thread_id,
    (&my_thread_id),
    "get thread id"
  );

  return my_thread_id;
}


uint64_t
rocm_cid_push
(
  rocprofiler_context_id_t context_id
)
{
  static uint64_t correlation_id = 17;

  rocprofiler_user_data_t rud;
  rud.value = correlation_id++;

  ROCPROFILER_CALL2
  (
    rocprofiler_push_external_correlation_id,
    (context_id, rocm_threads_self(), rud),
    "correlation id push"
  );

  return rud.value;
}


uint64_t
rocm_cid_pop
(
  rocprofiler_context_id_t context_id
)
{
  rocprofiler_user_data_t rud;
  ROCPROFILER_CALL2
  (
    rocprofiler_pop_external_correlation_id,
    (context_id, rocm_threads_self(), &rud),
    "correlation id pop"
  );

  return rud.value;
}

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t*              user_data,
                      void*                                 callback_data)
{
  uint64_t ext_correlation_id;

  switch(record.phase) {
    case ROCPROFILER_CALLBACK_PHASE_ENTER:
      ext_correlation_id = rocm_cid_push(client_ctx);

      printf("push correlation id: external 0x%lx internal = 0x%lx\n",
            ext_correlation_id,
            record.correlation_id.internal);
      break;

    case ROCPROFILER_CALLBACK_PHASE_EXIT:
      ext_correlation_id = rocm_cid_pop(client_ctx);

      printf("pop  correlation id: external 0x%lx internal = 0x%lx\n",
            ext_correlation_id,
            record.correlation_id.internal);
      break;

    default:
      break;
  }
}


void
tool_control_init(rocprofiler_context_id_t& primary_ctx)
{
    // Create a specialized (throw-away) context for handling ROCTx profiler pause and resume.
    // A separate context is used because if the context that is associated with roctxProfilerPause
    // disabled that same context, a call to roctxProfilerResume would be ignored because the
    // context that enables the callback for that API call is disabled.
    auto cntrl_ctx = rocprofiler_context_id_t{};
    ROCPROFILER_CALL(rocprofiler_create_context(&cntrl_ctx), "control context creation failed");

    // enable callback marker tracing with only the pause/resume operations
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         cntrl_ctx,
                         ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                         nullptr,
                         0,
                         tool_tracing_ctrl_callback,
                         &primary_ctx),
                     "callback tracing service failed to configure");

    // start the context so that it is always active
    ROCPROFILER_CALL(rocprofiler_start_context(cntrl_ctx), "start of control context");
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    assert(tool_data != nullptr);

    auto* call_stack_v = static_cast<call_stack_t*>(tool_data);

    call_stack_v->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    callback_name_info name_info = common::get_callback_id_names();

    for(const auto& itr : name_info)
    {
        auto name_idx = std::stringstream{};
        name_idx << " [" << std::setw(3) << itr.value << "]";
        call_stack_v->emplace_back(
            source_location{"rocprofiler_callback_tracing_kind_names          " + name_idx.str(),
                            __FILE__,
                            __LINE__,
                            std::string{itr.name}});

        for(auto [didx, ditr] : itr.items())
        {
            auto operation_idx = std::stringstream{};
            operation_idx << " [" << std::setw(3) << didx << "]";
            call_stack_v->emplace_back(source_location{
                "rocprofiler_callback_tracing_kind_operation_names" + operation_idx.str(),
                __FILE__,
                __LINE__,
                std::string{"- "} + std::string{*ditr}});
        }
    }

    client_fini_func = fini_func;

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation failed");

    // enable the control
    tool_control_init(client_ctx);

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       tool_data),
        "callback tracing service failed to configure");

    int valid_ctx = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                     "failure checking context validity");
    if(valid_ctx == 0)
    {
        // notify rocprofiler that initialization failed
        // and all the contexts, buffers, etc. created
        // should be ignored
        return -1;
    }

    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start failed");

    // no errors
    return 0;
}

void
tool_fini(void* tool_data)
{
    assert(tool_data != nullptr);

    auto* _call_stack = static_cast<call_stack_t*>(tool_data);
    _call_stack->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    print_call_stack(*_call_stack);

    delete _call_stack;
}
}  // namespace

void
setup()
{}

void
shutdown()
{
    if(client_id) client_fini_func(*client_id);
}

void
start()
{
    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start failed");
}

void
stop()
{
    int status = 0;
    ROCPROFILER_CALL(rocprofiler_is_initialized(&status), "failed to retrieve init status");
    if(status != 0)
    {
        ROCPROFILER_CALL(rocprofiler_stop_context(client_ctx), "rocprofiler context stop failed");
    }
}
}  // namespace client

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // set the client name
    id->name = "ExampleTool";

    // store client info
    client::client_id = id;

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " (priority=" << priority << ") is using rocprofiler-sdk v" << major << "."
         << minor << "." << patch << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    // demonstration of alternative way to get the version info
    {
        auto version_info = std::array<uint32_t, 3>{};
        ROCPROFILER_CALL(
            rocprofiler_get_version(&version_info.at(0), &version_info.at(1), &version_info.at(2)),
            "failed to get version info");

        if(std::array<uint32_t, 3>{major, minor, patch} != version_info)
        {
            throw std::runtime_error{"version info mismatch"};
        }
    }

    // data passed around all the callbacks
    auto* client_tool_data = new std::vector<client::source_location>{};

    // add first entry
    client_tool_data->emplace_back(
        client::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &client::tool_init,
                                            &client::tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    return &cfg;
}
