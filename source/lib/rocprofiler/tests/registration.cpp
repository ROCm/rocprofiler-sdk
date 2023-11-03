// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include <rocprofiler/registration.h>
#include <rocprofiler/rocprofiler.h>

#include "lib/common/environment.hpp"
#include "lib/common/units.hpp"
#include "lib/common/utility.hpp"

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <pthread.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <string_view>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#define ROCPROFILER_CALL(ARG, MSG)                                                                 \
    {                                                                                              \
        auto _status = (ARG);                                                                      \
        EXPECT_EQ(_status, ROCPROFILER_STATUS_SUCCESS) << MSG << " :: " << #ARG;                   \
    }

namespace
{
struct callback_data
{
    rocprofiler_client_id_t*      client_id             = nullptr;
    rocprofiler_client_finalize_t client_fini_func      = nullptr;
    rocprofiler_context_id_t      client_ctx            = {};
    rocprofiler_buffer_id_t       client_buffer         = {};
    rocprofiler_callback_thread_t client_thread         = {};
    uint64_t                      client_workflow_count = {};
    uint64_t                      client_callback_count = {};
    uint64_t                      client_elapsed        = {};
    int64_t                       current_depth         = 0;
    int64_t                       max_depth             = 0;
};

struct agent_data
{
    uint64_t                       agent_count = 0;
    std::vector<hsa_device_type_t> agents      = {};
};

using callback_kind_names_t = std::map<rocprofiler_service_callback_tracing_kind_t, const char*>;
using callback_kind_operation_names_t =
    std::map<rocprofiler_service_callback_tracing_kind_t, std::map<uint32_t, const char*>>;

struct callback_name_info
{
    callback_kind_names_t           kind_names      = {};
    callback_kind_operation_names_t operation_names = {};
};

auto
get_callback_tracing_names()
{
    auto cb_name_info = callback_name_info{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb =
        [](rocprofiler_service_callback_tracing_kind_t kindv, uint32_t operation, void* data_v) {
            auto* name_info_v = static_cast<callback_name_info*>(data_v);

            if(kindv == ROCPROFILER_CALLBACK_TRACING_HSA_API)
            {
                const char* name = nullptr;
                ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_operation_name(
                                     kindv, operation, &name, nullptr),
                                 "query callback tracing kind operation name");
                if(name) name_info_v->operation_names[kindv][operation] = name;
            }
            return 0;
        };

    //
    //  callback for each callback kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_service_callback_tracing_kind_t kind, void* data) {
        //  store the callback kind name
        auto*       name_info_v = static_cast<callback_name_info*>(data);
        const char* name        = nullptr;
        ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_name(kind, &name, nullptr),
                         "query callback tracing kind operation name");
        if(name) name_info_v->kind_names[kind] = name;

        if(kind == ROCPROFILER_CALLBACK_TRACING_HSA_API)
        {
            ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operations(
                                 kind, tracing_kind_operation_cb, static_cast<void*>(data)),
                             "iterating callback tracing kind operations");
        }
        return 0;
    };

    ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kinds(tracing_kind_cb,
                                                                static_cast<void*>(&cb_name_info)),
                     "iterating callback tracing kinds");

    return cb_name_info;
}

using buffer_kind_names_t = std::map<rocprofiler_service_buffer_tracing_kind_t, const char*>;
using buffer_kind_operation_names_t =
    std::map<rocprofiler_service_buffer_tracing_kind_t, std::map<uint32_t, const char*>>;

struct buffer_name_info
{
    buffer_kind_names_t           kind_names      = {};
    buffer_kind_operation_names_t operation_names = {};
};

buffer_name_info
get_buffer_tracing_names()
{
    auto cb_name_info = buffer_name_info{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb =
        [](rocprofiler_service_buffer_tracing_kind_t kindv, uint32_t operation, void* data_v) {
            auto* name_info_v = static_cast<buffer_name_info*>(data_v);

            if(kindv == ROCPROFILER_BUFFER_TRACING_HSA_API)
            {
                const char* name = nullptr;
                ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_operation_name(
                                     kindv, operation, &name, nullptr),
                                 "query buffer tracing kind operation name");
                if(name) name_info_v->operation_names[kindv][operation] = name;
            }
            return 0;
        };

    //
    //  callback for each buffer kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_service_buffer_tracing_kind_t kind, void* data) {
        //  store the buffer kind name
        auto*       name_info_v = static_cast<buffer_name_info*>(data);
        const char* name        = nullptr;
        ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_name(kind, &name, nullptr),
                         "query buffer tracing kind operation name");
        if(name) name_info_v->kind_names[kind] = name;

        if(kind == ROCPROFILER_BUFFER_TRACING_HSA_API)
        {
            ROCPROFILER_CALL(rocprofiler_iterate_buffer_tracing_kind_operations(
                                 kind, tracing_kind_operation_cb, static_cast<void*>(data)),
                             "iterating buffer tracing kind operations");
        }
        return 0;
    };

    ROCPROFILER_CALL(rocprofiler_iterate_buffer_tracing_kinds(tracing_kind_cb,
                                                              static_cast<void*>(&cb_name_info)),
                     "iterating buffer tracing kinds");

    return cb_name_info;
}

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t*              user_data,
                      void*                                 client_data)
{
    auto* cb_data       = static_cast<callback_data*>(client_data);
    auto  get_timestamp = []() {
        return std::chrono::steady_clock::now().time_since_epoch().count();
    };

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER && cb_data->current_depth == 0)
    {
        user_data->value = get_timestamp();
    }

    static auto name_map = get_callback_tracing_names();

    EXPECT_EQ(name_map.kind_names.size(), ROCPROFILER_CALLBACK_TRACING_LAST);
    EXPECT_EQ(name_map.operation_names.at(ROCPROFILER_CALLBACK_TRACING_HSA_API).size(),
              ROCPROFILER_HSA_API_ID_LAST);

    std::cout << "[" << __FILE__ << ":" << __LINE__ << "] "
              << name_map.operation_names[record.kind][record.operation] << "\n"
              << std::flush;

    cb_data->client_callback_count++;
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        cb_data->current_depth++;
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        cb_data->max_depth = std::max(cb_data->current_depth, cb_data->max_depth);
        cb_data->current_depth--;
    }
    else
    {
        GTEST_FAIL() << "unsupported callback tracing phase " << record.phase;
    }

    struct info_data
    {
        uint64_t          num_args = 0;
        std::stringstream arg_ss   = {};
    } info_data_v;

    auto info_data_cb = [](rocprofiler_service_callback_tracing_kind_t,
                           uint32_t,
                           uint32_t          arg_num,
                           const char*       arg_name,
                           const char*       arg_value_str,
                           const void* const arg_value_addr,
                           void*             data) -> int {
        auto& info = *static_cast<info_data*>(data);
        info.arg_ss << ((arg_num == 0) ? "(" : ", ");
        info.arg_ss << arg_num << ": " << arg_name << "=" << arg_value_str;
        EXPECT_NE(arg_name, nullptr);
        EXPECT_NE(arg_value_str, nullptr);
        EXPECT_NE(arg_value_addr, nullptr);
        EXPECT_EQ(arg_num, info.num_args);
        info.num_args++;
        return 0;
    };

    ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operation_args(
                         record, info_data_cb, static_cast<void*>(&info_data_v)),
                     "Failure iterating trace operation args");
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_HSA_API &&
       !(record.operation == ROCPROFILER_HSA_API_ID_hsa_init ||
         record.operation == ROCPROFILER_HSA_API_ID_hsa_shut_down))
    {
        EXPECT_GT(info_data_v.num_args, 0)
            << name_map.operation_names[record.kind][record.operation] << info_data_v.arg_ss.str();
    }

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT && cb_data->current_depth == 0)
    {
        cb_data->client_elapsed += (get_timestamp() - user_data->value);
    }
}

void
tool_tracing_buffered(rocprofiler_context_id_t      context,
                      rocprofiler_buffer_id_t       buffer_id,
                      rocprofiler_record_header_t** headers,
                      size_t                        num_headers,
                      void*                         buffer_data,
                      uint64_t                      drop_count)
{
    auto* cb_data = static_cast<callback_data*>(buffer_data);

    static auto name_map = get_buffer_tracing_names();

    EXPECT_EQ(name_map.kind_names.size(), ROCPROFILER_BUFFER_TRACING_LAST);
    EXPECT_EQ(name_map.operation_names.at(ROCPROFILER_BUFFER_TRACING_HSA_API).size(),
              ROCPROFILER_HSA_API_ID_LAST);

    auto v_records = std::vector<rocprofiler_buffer_tracing_hsa_api_record_t*>{};
    v_records.reserve(num_headers);

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        ASSERT_TRUE(header != nullptr);
        auto hash = rocprofiler_record_header_compute_hash(header->category, header->kind);
        EXPECT_EQ(header->hash, hash);
        EXPECT_TRUE(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                    header->kind == ROCPROFILER_BUFFER_TRACING_HSA_API);

        v_records.emplace_back(
            static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload));
    }

    std::sort(v_records.begin(), v_records.end(), [](auto lhs, auto rhs) {
        return (lhs->start_timestamp == rhs->start_timestamp)
                   ? (lhs->end_timestamp < rhs->end_timestamp)
                   : (lhs->start_timestamp < rhs->start_timestamp);
    });

    for(auto* record : v_records)
    {
        auto info = std::stringstream{};
        info << "tid=" << record->thread_id << ", context=" << context.handle
             << ", buffer_id=" << buffer_id.handle << ", cid=" << record->correlation_id.internal
             << ", kind=" << name_map.kind_names.at(record->kind) << "(" << record->kind
             << "), operation=" << name_map.operation_names.at(record->kind).at(record->operation)
             << "(" << record->operation << "), drop_count=" << drop_count
             << ", start=" << record->start_timestamp << ", stop=" << record->end_timestamp;

        static int64_t last_corr_id = -1;
        auto           corr_id      = static_cast<int64_t>(record->correlation_id.internal);

        std::cout << info.str() << "\n" << std::flush;
        EXPECT_GE(context.handle, 0) << info.str();
        EXPECT_GT(record->thread_id, 0) << info.str();
        EXPECT_GT(record->kind, 0) << info.str();
        EXPECT_GT(corr_id, last_corr_id) << info.str();
        EXPECT_GT(record->start_timestamp, 0) << info.str();
        EXPECT_GT(record->end_timestamp, 0) << info.str();
        EXPECT_LE(record->start_timestamp, record->end_timestamp) << info.str();

        cb_data->client_callback_count++;
        last_corr_id = corr_id;
    }
}

void
thread_precreate(rocprofiler_runtime_library_t /*lib*/, void* tool_data)
{
    auto* cb_data = static_cast<callback_data*>(tool_data);
    cb_data->client_workflow_count++;
}

void
thread_postcreate(rocprofiler_runtime_library_t /*lib*/, void* tool_data)
{
    auto* cb_data = static_cast<callback_data*>(tool_data);
    cb_data->client_workflow_count++;
}
}  // namespace

TEST(rocprofiler_lib, registration_lambda_no_result)
{
    static rocprofiler_configure_func_t rocp_init =
        [](uint32_t                 version,
           const char*              runtime_version,
           uint32_t                 prio,
           rocprofiler_client_id_t* client_id) -> rocprofiler_tool_configure_result_t* {
        auto expected_version = ROCPROFILER_VERSION;
        EXPECT_EQ(expected_version, version);
        EXPECT_EQ(std::string_view{runtime_version}, std::string_view{ROCPROFILER_VERSION_STRING});
        EXPECT_EQ(prio, 0);
        EXPECT_EQ(client_id->name, nullptr);
        return nullptr;
    };

    auto ctx = rocprofiler_context_id_t{};
    EXPECT_NE(rocprofiler_create_context(&ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_force_configure(rocp_init), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_NE(rocprofiler_create_context(&ctx), ROCPROFILER_STATUS_SUCCESS);
}

TEST(rocprofiler_lib, callback_registration_lambda_with_result)
{
    using init_func_t = int (*)(rocprofiler_client_finalize_t, void*);
    using fini_func_t = void (*)(void*);

    using hsa_iterate_agents_cb_t = hsa_status_t (*)(hsa_agent_t, void*);

    auto cmd_line = rocprofiler::common::read_command_line(getpid());

    ASSERT_FALSE(cmd_line.empty());

    static init_func_t tool_init = [](rocprofiler_client_finalize_t fini_func,
                                      void*                         client_data) -> int {
        auto* cb_data = static_cast<callback_data*>(client_data);

        cb_data->client_workflow_count++;
        cb_data->client_fini_func = fini_func;

        ROCPROFILER_CALL(rocprofiler_create_context(&cb_data->client_ctx),
                         "failed to create context");

        ROCPROFILER_CALL(
            rocprofiler_configure_callback_tracing_service(cb_data->client_ctx,
                                                           ROCPROFILER_CALLBACK_TRACING_HSA_API,
                                                           nullptr,
                                                           0,
                                                           tool_tracing_callback,
                                                           client_data),
            "callback tracing service failed to configure");

        int valid_ctx = 0;
        ROCPROFILER_CALL(rocprofiler_context_is_valid(cb_data->client_ctx, &valid_ctx),
                         "failure checking context validity");

        EXPECT_EQ(valid_ctx, 1);

        ROCPROFILER_CALL(rocprofiler_start_context(cb_data->client_ctx),
                         "rocprofiler context start failed");

        // no errors
        return 0;
    };

    static fini_func_t tool_fini = [](void* client_data) -> void {
        auto* cb_data = static_cast<callback_data*>(client_data);
        ROCPROFILER_CALL(rocprofiler_stop_context(cb_data->client_ctx),
                         "rocprofiler context stop failed");

        static_cast<callback_data*>(client_data)->client_workflow_count++;
    };

    static auto cb_data = callback_data{};

    static auto cfg_result =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            tool_init,
                                            tool_fini,
                                            static_cast<void*>(&cb_data)};

    static rocprofiler_configure_func_t rocp_init =
        [](uint32_t                 version,
           const char*              runtime_version,
           uint32_t                 prio,
           rocprofiler_client_id_t* client_id) -> rocprofiler_tool_configure_result_t* {
        auto expected_version = ROCPROFILER_VERSION;
        EXPECT_EQ(expected_version, version);
        EXPECT_EQ(std::string_view{runtime_version}, std::string_view{ROCPROFILER_VERSION_STRING});
        EXPECT_EQ(prio, 0);
        EXPECT_EQ(client_id->name, nullptr);
        cb_data.client_id       = client_id;
        cb_data.client_id->name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        return &cfg_result;
    };

    auto get_timestamp = []() {
        return std::chrono::steady_clock::now().time_since_epoch().count();
    };

    auto ctx = rocprofiler_context_id_t{};
    EXPECT_NE(rocprofiler_create_context(&ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_force_configure(rocp_init), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_NE(rocprofiler_create_context(&ctx), ROCPROFILER_STATUS_SUCCESS);

    hsa_iterate_agents_cb_t agent_cb = [](hsa_agent_t agent, void* data) {
        static_cast<agent_data*>(data)->agent_count++;

        auto status     = HSA_STATUS_SUCCESS;
        auto agent_type = hsa_device_type_t{};
        if((status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &agent_type)) ==
           HSA_STATUS_SUCCESS)
            static_cast<agent_data*>(data)->agents.emplace_back(agent_type);

        return status;
    };

    hsa_init();
    auto         _agent_data = agent_data{};
    auto         begin_ts    = get_timestamp();
    hsa_status_t itr_status  = hsa_iterate_agents(agent_cb, static_cast<void*>(&_agent_data));
    auto         end_ts      = get_timestamp();
    auto         elapsed     = (end_ts - begin_ts);

    EXPECT_EQ(itr_status, HSA_STATUS_SUCCESS);
    EXPECT_GT(_agent_data.agent_count, 0);
    EXPECT_EQ(_agent_data.agent_count, _agent_data.agents.size());

#if !defined(__OPTIMIZE__) || (defined(CODECOV) && CODECOV > 0)
    EXPECT_GT(cb_data.client_elapsed, 0);
    EXPECT_GT(elapsed, 0);
#else
    decltype(elapsed) elapsed_tolerance = 0.25 * elapsed;
    EXPECT_NEAR(elapsed, cb_data.client_elapsed, elapsed_tolerance)
        << "it is possible this failed due to noise on the machine";
#endif

    ASSERT_NE(cb_data.client_id, nullptr);
    ASSERT_NE(cb_data.client_fini_func, nullptr);

    cb_data.client_fini_func(*cb_data.client_id);

    // expected callback count is two for hsa_iterate_agents and two callbacks for
    // hsa_agent_get_info for each agent.
    uint64_t expected_cb_count = 2 + (2 * _agent_data.agent_count);

    EXPECT_EQ(cb_data.client_workflow_count, 2);
    EXPECT_EQ(cb_data.client_callback_count, expected_cb_count);
    EXPECT_EQ(cb_data.current_depth, 0);
    EXPECT_EQ(cb_data.max_depth, 2);
}

TEST(rocprofiler_lib, buffer_registration_lambda_with_result)
{
    using init_func_t = int (*)(rocprofiler_client_finalize_t, void*);
    using fini_func_t = void (*)(void*);

    using hsa_iterate_agents_cb_t = hsa_status_t (*)(hsa_agent_t, void*);

    auto cmd_line = rocprofiler::common::read_command_line(getpid());
    ASSERT_FALSE(cmd_line.empty());

    static init_func_t tool_init = [](rocprofiler_client_finalize_t fini_func,
                                      void*                         client_data) -> int {
        auto* cb_data = static_cast<callback_data*>(client_data);

        cb_data->client_workflow_count++;
        cb_data->client_fini_func = fini_func;

        ROCPROFILER_CALL(rocprofiler_create_context(&cb_data->client_ctx),
                         "failed to create context");

        ROCPROFILER_CALL(rocprofiler_create_buffer(cb_data->client_ctx,
                                                   4096,
                                                   2048,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   tool_tracing_buffered,
                                                   client_data,
                                                   &cb_data->client_buffer),
                         "buffer creation failed");

        ROCPROFILER_CALL(
            rocprofiler_configure_buffer_tracing_service(cb_data->client_ctx,
                                                         ROCPROFILER_BUFFER_TRACING_HSA_API,
                                                         nullptr,
                                                         0,
                                                         cb_data->client_buffer),
            "buffer tracing service failed to configure");

        ROCPROFILER_CALL(rocprofiler_create_callback_thread(&cb_data->client_thread),
                         "failure creating callback thread");

        ROCPROFILER_CALL(
            rocprofiler_assign_callback_thread(cb_data->client_buffer, cb_data->client_thread),
            "failed to assign thread for buffer");

        int valid_ctx = 0;
        ROCPROFILER_CALL(rocprofiler_context_is_valid(cb_data->client_ctx, &valid_ctx),
                         "failure checking context validity");

        EXPECT_EQ(valid_ctx, 1);

        ROCPROFILER_CALL(rocprofiler_start_context(cb_data->client_ctx),
                         "rocprofiler context start failed");

        // no errors
        return 0;
    };

    static fini_func_t tool_fini = [](void* client_data) -> void {
        auto* cb_data = static_cast<callback_data*>(client_data);
        ROCPROFILER_CALL(rocprofiler_stop_context(cb_data->client_ctx),
                         "rocprofiler context stop failed");

        static_cast<callback_data*>(client_data)->client_workflow_count++;
    };

    static auto cb_data = callback_data{};

    static auto cfg_result =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            tool_init,
                                            tool_fini,
                                            static_cast<void*>(&cb_data)};

    static rocprofiler_configure_func_t rocp_init =
        [](uint32_t                 version,
           const char*              runtime_version,
           uint32_t                 prio,
           rocprofiler_client_id_t* client_id) -> rocprofiler_tool_configure_result_t* {
        auto expected_version = ROCPROFILER_VERSION;
        EXPECT_EQ(expected_version, version);
        EXPECT_EQ(std::string_view{runtime_version}, std::string_view{ROCPROFILER_VERSION_STRING});
        EXPECT_EQ(prio, 0);
        EXPECT_EQ(client_id->name, nullptr);
        cb_data.client_id       = client_id;
        cb_data.client_id->name = ::testing::UnitTest::GetInstance()->current_test_info()->name();

        ROCPROFILER_CALL(rocprofiler_at_internal_thread_create(thread_precreate,
                                                               thread_postcreate,
                                                               ROCPROFILER_LIBRARY,
                                                               static_cast<void*>(&cb_data)),
                         "failed to register for thread creation notifications");

        return &cfg_result;
    };

    auto ctx = rocprofiler_context_id_t{};
    EXPECT_NE(rocprofiler_create_context(&ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_force_configure(rocp_init), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_NE(rocprofiler_create_context(&ctx), ROCPROFILER_STATUS_SUCCESS);

    hsa_iterate_agents_cb_t agent_cb = [](hsa_agent_t agent, void* data) {
        static_cast<agent_data*>(data)->agent_count++;

        auto status     = HSA_STATUS_SUCCESS;
        auto agent_type = hsa_device_type_t{};
        if((status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &agent_type)) ==
           HSA_STATUS_SUCCESS)
            static_cast<agent_data*>(data)->agents.emplace_back(agent_type);

        return status;
    };

    auto _agent_data = agent_data{};
    hsa_init();
    hsa_status_t itr_status = hsa_iterate_agents(agent_cb, static_cast<void*>(&_agent_data));

    EXPECT_EQ(itr_status, HSA_STATUS_SUCCESS);
    EXPECT_GT(_agent_data.agent_count, 0);
    EXPECT_EQ(_agent_data.agent_count, _agent_data.agents.size());

    ASSERT_NE(cb_data.client_id, nullptr);
    ASSERT_NE(cb_data.client_fini_func, nullptr);

    EXPECT_EQ(rocprofiler_flush_buffer(cb_data.client_buffer), ROCPROFILER_STATUS_SUCCESS);

    cb_data.client_fini_func(*cb_data.client_id);

    // expected callback count is two for hsa_iterate_agents and two callbacks for
    // hsa_agent_get_info for each agent.
    uint64_t expected_cb_count = 1 + _agent_data.agent_count;
    // expect the tool init, tool fini, and two calls to thread_precreate and thread_postcreate each
    // (the main thread and the assigned thread for the buffer)
    uint64_t expected_workflow_count = 6;

    EXPECT_EQ(cb_data.client_workflow_count, expected_workflow_count);
    EXPECT_EQ(cb_data.client_callback_count, expected_cb_count);
    EXPECT_GT(cb_data.client_thread.handle, 0);
    EXPECT_EQ(cb_data.current_depth, 0);
    EXPECT_EQ(cb_data.max_depth, 0);
}
