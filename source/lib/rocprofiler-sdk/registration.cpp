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

#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/code_object.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/intercept_table.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/version.h>

#include <fmt/format.h>
#include <glog/logging.h>

#include <dlfcn.h>
#include <link.h>
#include <unistd.h>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

extern "C" {
#pragma weak rocprofiler_configure

extern rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*);
}

namespace rocprofiler
{
namespace registration
{
namespace
{
// invoke all rocprofiler_configure symbols
bool
invoke_client_configures();

// invoke initialize functions returned from rocprofiler_configure
bool
invoke_client_initializers();

// invoke finalize functions returned from rocprofiler_configure
bool
invoke_client_finalizers();

// explicitly invoke the finalize function of a specific client
void invoke_client_finalizer(rocprofiler_client_id_t);

auto*
get_status()
{
    static auto*& _v =
        common::static_object<std::pair<std::atomic<int>, std::atomic<int>>>::construct(0, 0);
    return _v;
}

auto&
get_invoked_configures()
{
    static auto _v = std::unordered_set<rocprofiler_configure_func_t>{};
    return _v;
}

auto&
get_forced_configure()
{
    static rocprofiler_configure_func_t _v = nullptr;
    return _v;
}

std::vector<std::string>
get_link_map()
{
    auto  chain  = std::vector<std::string>{};
    void* handle = nullptr;
    handle       = dlopen(nullptr, RTLD_LAZY | RTLD_NOLOAD);

    if(handle)
    {
        struct link_map* link_map_v = nullptr;
        dlinfo(handle, RTLD_DI_LINKMAP, &link_map_v);
        struct link_map* next_link = link_map_v->l_next;
        while(next_link)
        {
            if(next_link->l_name != nullptr && !std::string_view{next_link->l_name}.empty())
            {
                chain.emplace_back(next_link->l_name);
            }
            next_link = next_link->l_next;
        }
    }

    return chain;
}

struct client_library
{
    client_library() = default;
    ~client_library() { delete configure_result; }

    client_library(const client_library&)     = delete;
    client_library(client_library&&) noexcept = default;

    client_library& operator=(const client_library&) = delete;
    client_library& operator=(client_library&&) noexcept = delete;

    std::string                          name               = {};
    void*                                dlhandle           = nullptr;
    decltype(::rocprofiler_configure)*   configure_func     = nullptr;
    rocprofiler_tool_configure_result_t* configure_result   = nullptr;
    rocprofiler_client_id_t              internal_client_id = {};
    rocprofiler_client_id_t              mutable_client_id  = {};
};

using client_library_vec_t = std::vector<client_library>;

client_library_vec_t
find_clients()
{
    auto data            = client_library_vec_t{};
    auto priority_offset = get_client_offset();

    if(get_forced_configure())
    {
        LOG(ERROR) << "adding forced configure";
        uint32_t _prio = priority_offset + data.size();
        data.emplace_back(client_library{"(forced)",
                                         nullptr,
                                         get_forced_configure(),
                                         nullptr,
                                         rocprofiler_client_id_t{nullptr, _prio},
                                         rocprofiler_client_id_t{nullptr, _prio}});
    }

    auto get_env_libs = []() {
        auto       val       = common::get_env("ROCP_TOOL_LIBRARIES", std::string{});
        auto       val_arr   = std::vector<std::string>{};
        size_t     pos       = 0;
        const auto delimiter = std::string_view{":"};
        auto       token     = std::string{};

        if(val.empty())
        {
            // do nothing
        }
        else if(val.find(delimiter) == std::string::npos)
        {
            val_arr.emplace_back(val);
        }
        else
        {
            while((pos = val.find(delimiter)) != std::string::npos)
            {
                token = val.substr(0, pos);
                if(!token.empty()) val_arr.emplace_back(token);
                val.erase(0, pos + delimiter.length());
            }
        }
        return val_arr;
    };

    auto env = get_env_libs();

    if(!rocprofiler_configure && !get_forced_configure() && env.empty())
    {
        LOG(ERROR) << "no rocprofiler_configure function(s) found";
        return data;
    }

    if(rocprofiler_configure != &rocprofiler_configure)
        throw std::runtime_error("rocprofiler_configure != &rocprofiler_configure");

    if(rocprofiler_configure && rocprofiler_configure != get_forced_configure())
    {
        uint32_t _prio = priority_offset + data.size();
        data.emplace_back(client_library{"unknown",
                                         nullptr,
                                         rocprofiler_configure,
                                         nullptr,
                                         rocprofiler_client_id_t{nullptr, _prio},
                                         rocprofiler_client_id_t{nullptr, _prio}});
    }

    for(const auto& itr : get_link_map())
    {
        LOG(INFO) << "searching " << itr << " for rocprofiler_configure";

        void* handle = dlopen(itr.c_str(), RTLD_LAZY | RTLD_NOLOAD);
        LOG_IF(ERROR, handle == nullptr) << "error dlopening " << itr;

        decltype(::rocprofiler_configure)* _sym = nullptr;
        *(void**) (&_sym)                       = dlsym(handle, "rocprofiler_configure");

        // symbol not found
        if(!_sym)
        {
            LOG(INFO) << "|_" << itr << " did not contain rocprofiler_configure symbol";
            continue;
        }

        // skip the configure function that was forced
        if(_sym == get_forced_configure())
        {
            data.front().name                    = itr;
            data.front().dlhandle                = handle;
            data.front().internal_client_id.name = "(forced)";
            continue;
        }

        if(_sym == &rocprofiler_configure && data.size() == 1)
        {
            data.front().name                    = itr;
            data.front().dlhandle                = handle;
            data.front().internal_client_id.name = "default";
        }
        else
        {
            uint32_t _prio = priority_offset + data.size();
            auto&    entry =
                data.emplace_back(client_library{itr,
                                                 handle,
                                                 _sym,
                                                 nullptr,
                                                 rocprofiler_client_id_t{nullptr, _prio},
                                                 rocprofiler_client_id_t{nullptr, _prio}});
            entry.internal_client_id.name = entry.name.c_str();
        }
    }

    if(!env.empty())
    {
        for(const auto& itr : env)
        {
            void* handle = dlopen(itr.c_str(), RTLD_GLOBAL | RTLD_LAZY);
            LOG_IF(ERROR, handle == nullptr) << "error dlopening " << itr;

            for(const auto& ditr : data)
            {
                if(ditr.dlhandle && ditr.dlhandle == handle)
                {
                    handle = nullptr;
                    break;
                }
            }

            if(handle)
            {
                decltype(::rocprofiler_configure)* _sym = nullptr;
                *(void**) (&_sym)                       = dlsym(handle, "rocprofiler_configure");

                uint32_t _prio = priority_offset + data.size();
                data.emplace_back(client_library{itr,
                                                 handle,
                                                 _sym,
                                                 nullptr,
                                                 rocprofiler_client_id_t{nullptr, _prio},
                                                 rocprofiler_client_id_t{nullptr, _prio}});
            }
        }
    }

    LOG(ERROR) << __FUNCTION__ << " found " << data.size() << " clients";

    return data;
}

client_library_vec_t*
get_clients()
{
    static auto*& _v = common::static_object<client_library_vec_t>::construct(find_clients());
    return _v;
}

using mutex_t       = std::recursive_mutex;
using scoped_lock_t = std::unique_lock<mutex_t>;

mutex_t&
get_registration_mutex()
{
    static auto _v = mutex_t{};
    return _v;
}

bool
invoke_client_configures()
{
    if(get_init_status() > 0) return false;

    auto _lk = scoped_lock_t{get_registration_mutex(), std::defer_lock};
    if(_lk.owns_lock()) return false;
    _lk.lock();

    LOG(ERROR) << __FUNCTION__;

    if(!get_clients()) return false;

    for(auto& itr : *get_clients())
    {
        if(!itr.configure_func)
        {
            LOG(ERROR) << "rocprofiler::registration::invoke_client_configures() attempted to "
                          "invoke configure function from "
                       << itr.name << " that had no configuration function";
            continue;
        }

        if(get_invoked_configures().find(itr.configure_func) != get_invoked_configures().end())
        {
            LOG(ERROR) << "rocprofiler::registration::invoke_client_configures() attempted to "
                          "invoke configure function from "
                       << itr.name << " (addr="
                       << fmt::format("{:#018x}", reinterpret_cast<uint64_t>(itr.configure_func))
                       << ") more than once";
            continue;
        }
        else
        {
            LOG(INFO) << "rocprofiler::registration::invoke_client_configures() invoking configure "
                         "function from "
                      << itr.name << " (addr="
                      << fmt::format("{:#018x}", reinterpret_cast<uint64_t>(itr.configure_func))
                      << ")";
        }

        auto* _result = itr.configure_func(ROCPROFILER_VERSION,
                                           ROCPROFILER_VERSION_STRING,
                                           itr.internal_client_id.handle - get_client_offset(),
                                           &itr.mutable_client_id);

        if(_result)
        {
            itr.configure_result = new rocprofiler_tool_configure_result_t{*_result};
        }
        else
        {
            context::deactivate_client_contexts(itr.internal_client_id);
            context::deregister_client_contexts(itr.internal_client_id);
        }

        get_invoked_configures().emplace(itr.configure_func);
    }

    return true;
}

bool
invoke_client_initializers()
{
    if(get_init_status() > 0) return false;

    auto _lk = scoped_lock_t{get_registration_mutex(), std::defer_lock};
    if(_lk.owns_lock()) return false;
    _lk.lock();

    LOG(ERROR) << __FUNCTION__;

    if(!get_clients()) return false;

    for(auto& itr : *get_clients())
    {
        if(itr.configure_result && itr.configure_result->initialize)
        {
            context::push_client(itr.internal_client_id.handle);
            itr.configure_result->initialize(&invoke_client_finalizer,
                                             itr.configure_result->tool_data);
            context::pop_client(itr.internal_client_id.handle);
            // set to nullptr so initialize only gets called once
            itr.configure_result->initialize = nullptr;
        }
    }

    return true;
}

bool
invoke_client_finalizers()
{
    // NOTE: this function is expected to only be invoked from the finalize function (which sets the
    // fini status)

    if(get_init_status() < 1 || get_fini_status() > 0) return false;

    auto _lk = scoped_lock_t{get_registration_mutex(), std::defer_lock};
    if(_lk.owns_lock()) return false;
    _lk.lock();

    LOG(ERROR) << __FUNCTION__;

    if(!get_clients()) return false;

    for(auto& itr : *get_clients())
    {
        if(itr.configure_result && itr.configure_result->finalize)
        {
            itr.configure_result->finalize(itr.configure_result->tool_data);
            // set to nullptr so finalize only gets called once
            itr.configure_result->finalize = nullptr;
        }
        context::deactivate_client_contexts(itr.internal_client_id);
    }

    return true;
}

void
invoke_client_finalizer(rocprofiler_client_id_t client_id)
{
    auto _lk = scoped_lock_t{get_registration_mutex(), std::defer_lock};
    if(_lk.owns_lock()) return;
    _lk.lock();

    LOG(ERROR) << __FUNCTION__;

    if(!get_clients()) return;

    for(auto& itr : *get_clients())
    {
        if(itr.internal_client_id.handle == client_id.handle &&
           itr.mutable_client_id.handle == client_id.handle)
        {
            if(itr.configure_result && itr.configure_result->finalize)
            {
                auto _fini_status = get_fini_status();
                if(_fini_status == 0) set_fini_status(-1);
                itr.configure_result->finalize(itr.configure_result->tool_data);
                if(_fini_status == 0) set_fini_status(_fini_status);
                // set to nullptr so finalize only gets called once
                itr.configure_result->finalize = nullptr;
            }
            context::deactivate_client_contexts(itr.internal_client_id);
        }
    }
}
}  // namespace

void
init_logging()
{
    static auto _once = std::once_flag{};
    std::call_once(_once, []() {
        auto get_argv0 = []() {
            auto ifs  = std::ifstream{"/proc/self/cmdline"};
            auto sarg = std::string{};
            while(ifs && !ifs.eof())
            {
                ifs >> sarg;
                if(!sarg.empty()) break;
            }
            return sarg;
        };

        static auto argv0 = get_argv0();
        google::InitGoogleLogging(argv0.c_str());
        LOG(INFO) << "logging initialized";
    });
}

uint32_t
get_client_offset()
{
    static uint32_t _v = []() {
        auto gen = std::mt19937{std::random_device{}()};
        auto rng = std::uniform_int_distribution<uint32_t>{
            std::numeric_limits<uint8_t>::max(),
            std::numeric_limits<uint32_t>::max() - std::numeric_limits<uint8_t>::max()};
        return rng(gen);
    }();
    return _v;
}

int
get_init_status()
{
    return (get_status()) ? get_status()->first.load(std::memory_order_acquire) : 1;
}

int
get_fini_status()
{
    return (get_status()) ? get_status()->second.load(std::memory_order_acquire) : 1;
}

void
set_init_status(int v)
{
    if(get_status()) get_status()->first.store(v, std::memory_order_release);
}

void
set_fini_status(int v)
{
    if(get_status()) get_status()->second.store(v, std::memory_order_release);
}

void
initialize()
{
    if(get_init_status() != 0) return;

    static auto _once = std::once_flag{};
    std::call_once(_once, []() {
        // initialization is in process
        set_init_status(-1);
        std::atexit([]() {
            finalize();
            common::destroy_static_objects();
        });
        init_logging();
        invoke_client_configures();
        invoke_client_initializers();
        internal_threading::initialize();
        // initialization is no longer available
        set_init_status(1);
    });
}

void
finalize()
{
    if(get_fini_status() != 0)
    {
        LOG(INFO) << "ignoring finalization request (value=" << get_fini_status() << ")";
        return;
    }

    static auto _sync = std::atomic_flag{};
    if(_sync.test_and_set())
    {
        LOG(INFO) << "ignoring finalization request [already finalized] (value="
                  << get_fini_status() << ")";
        return;
    }
    // above returns true for all invocations after the first one

    LOG(INFO) << "finalizing rocprofiler (value=" << get_fini_status() << ")";

    static auto _once = std::once_flag{};
    std::call_once(_once, []() {
        set_fini_status(-1);
        hsa::code_object_shutdown();
        if(get_init_status() > 0)
        {
            invoke_client_finalizers();
        }
        internal_threading::finalize();
        set_fini_status(1);
    });
}
}  // namespace registration
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_is_initialized(int* status)
{
    *status = rocprofiler::registration::get_init_status();
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_is_finalized(int* status)
{
    *status = rocprofiler::registration::get_fini_status();
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_force_configure(rocprofiler_configure_func_t configure_func)
{
    auto& forced_config = rocprofiler::registration::get_forced_configure();

    // init status may be -1 (currently initializing) or 1 (already initialized).
    // if either case, we want to ignore this function call but if this is
    if(rocprofiler::registration::get_init_status() != 0)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    // if another tool forced configure, the init status should be 1, but
    // let's just make sure that the forced configure function is a nullptr
    if(forced_config) return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    forced_config = configure_func;
    rocprofiler::registration::initialize();

    return ROCPROFILER_STATUS_SUCCESS;
}

int
rocprofiler_set_api_table(const char* name,
                          uint64_t    lib_version,
                          uint64_t    lib_instance,
                          void**      tables,
                          uint64_t    num_tables)
{
    static auto _once = std::once_flag{};
    std::call_once(_once, rocprofiler::registration::initialize);

    // pass to roctx init
    LOG_IF(ERROR, num_tables == 0) << " rocprofiler expected " << name
                                   << " library to pass at least one table, not " << num_tables;
    LOG_IF(ERROR, tables == nullptr) << " rocprofiler expected pointer to array of tables from "
                                     << name << " library, not a nullptr";

    if(std::string_view{name} == "hip")
    {
        // pass to hip init
        LOG_IF(ERROR, num_tables > 1)
            << " rocprofiler expected HIP library to pass 1 API table, not " << num_tables;

        return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
    }
    else if(std::string_view{name} == "hsa")
    {
        // this is a slight hack due to a hsa-runtime bug with rocprofiler-register which
        // causes it to register the API table twice when HSA_TOOL_LIB is set to this
        // rocprofiler library. Fixed in Gerrit review 961592.
        setenv("HSA_TOOLS_ROCPROFILER_V1_TOOLS", "0", 0);

        // pass to hsa init
        LOG_IF(ERROR, num_tables > 1)
            << " rocprofiler expected HSA library to pass 1 API table, not " << num_tables;

        auto* hsa_api_table = static_cast<HsaApiTable*>(*tables);

        // need to construct agent mappings before initializing the queue controller
        rocprofiler::agent::construct_agent_cache(hsa_api_table);
        rocprofiler::hsa::queue_controller_init(hsa_api_table);
        rocprofiler::hsa::code_object_init(hsa_api_table);

        // any internal modifications to the HsaApiTable need to be done before we make the
        // copy or else those modifications will be lost when HSA API tracing is enabled
        // because the HSA API tracing invokes the function pointers from the copy below
        auto& saved_hsa_api_table = rocprofiler::hsa::get_table();
        ::copyTables(hsa_api_table, &saved_hsa_api_table);

        // install rocprofiler API wrappers
        rocprofiler::hsa::update_table(hsa_api_table);

        // allow tools to install API wrappers
        rocprofiler::intercept_table::notify_runtime_api_registration(
            ROCPROFILER_HSA_LIBRARY, lib_version, lib_instance, std::make_tuple(hsa_api_table));
    }
    else if(std::string_view{name} == "roctx")
    {
        // pass to roctx init
        LOG_IF(ERROR, num_tables > 1)
            << " rocprofiler expected ROCTX library to pass 1 API table, not " << num_tables;

        return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
    }
    else
    {
        LOG(ERROR) << "rocprofiler does not accept API tables from " << name;

        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    (void) lib_version;
    (void) lib_instance;
    (void) tables;
    (void) num_tables;

    return 0;
}

bool
OnLoad(HsaApiTable*       table,
       uint64_t           runtime_version,
       uint64_t           failed_tool_count,
       const char* const* failed_tool_names)
{
    rocprofiler::registration::init_logging();

    (void) runtime_version;
    (void) failed_tool_count;
    (void) failed_tool_names;

    fprintf(stderr, "[%s:%i] %s\n", __FILE__, __LINE__, __FUNCTION__);

    void* table_v = static_cast<void*>(table);
    rocprofiler_set_api_table("hsa", runtime_version, 0, &table_v, 1);

    return true;
}

void
OnUnload()
{
    LOG(INFO) << "Unloading hsa-runtime...";
    ::rocprofiler::registration::finalize();
    LOG(INFO) << "Finalization complete.";
}
}
