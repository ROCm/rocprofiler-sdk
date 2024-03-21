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

#include "lib/rocprofiler-sdk/hsa/code_object.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <hsa/hsa.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <glog/logging.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <regex>
#include <string_view>
#include <vector>

#if defined(ROCPROFILER_CI)
#    define ROCP_CI_LOG_IF(NON_CI_LEVEL, ...) LOG_IF(FATAL, __VA_ARGS__)
#    define ROCP_CI_LOG(NON_CI_LEVEL, ...)    LOG(FATAL)
#else
#    define ROCP_CI_LOG_IF(NON_CI_LEVEL, ...) LOG_IF(NON_CI_LEVEL, __VA_ARGS__)
#    define ROCP_CI_LOG(NON_CI_LEVEL, ...)    LOG(NON_CI_LEVEL)
#endif

namespace rocprofiler
{
namespace hsa
{
namespace code_object
{
namespace
{
using context_t              = context::context;
using context_array_t        = common::container::small_vector<const context_t*>;
using external_corr_id_map_t = std::unordered_map<const context_t*, rocprofiler_user_data_t>;

template <size_t OpIdx>
struct code_object_info;

#define SPECIALIZE_CODE_OBJECT_INFO(OPERATION)                                                     \
    template <>                                                                                    \
    struct code_object_info<ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_##OPERATION>                  \
    {                                                                                              \
        static constexpr auto operation_idx =                                                      \
            ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_##OPERATION;                                  \
        static constexpr auto name = #OPERATION;                                                   \
    };

SPECIALIZE_CODE_OBJECT_INFO(NONE)
SPECIALIZE_CODE_OBJECT_INFO(LOAD)
SPECIALIZE_CODE_OBJECT_INFO(DEVICE_KERNEL_SYMBOL_REGISTER)

#undef SPECIALIZE_CODE_OBJECT_INFO

template <size_t Idx, size_t... IdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return code_object_info<Idx>::name;
    if constexpr(sizeof...(IdxTail) > 0)
        return name_by_id(id, std::index_sequence<IdxTail...>{});
    else
        return nullptr;
}

template <size_t Idx, size_t... IdxTail>
uint32_t
id_by_name(const char* name, std::index_sequence<Idx, IdxTail...>)
{
    if(std::string_view{code_object_info<Idx>::name} == std::string_view{name})
        return code_object_info<Idx>::operation_idx;
    if constexpr(sizeof...(IdxTail) > 0)
        return id_by_name(name, std::index_sequence<IdxTail...>{});
    else
        return ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_NONE;
}

template <size_t... Idx>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < static_cast<uint32_t>(ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LAST))
            _vec.emplace_back(_v);
    };

    (_emplace(_id_list, code_object_info<Idx>::operation_idx), ...);
}

template <size_t... Idx>
void
get_names(std::vector<const char*>& _name_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, const char* _v) {
        if(_v != nullptr && strnlen(_v, 1) > 0) _vec.emplace_back(_v);
    };

    (_emplace(_name_list, code_object_info<Idx>::name), ...);
}
}  // namespace

// check out the assembly here... this compiles to a switch statement
const char*
name_by_id(uint32_t id)
{
    return name_by_id(id,
                      std::make_index_sequence<ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LAST>{});
}

uint32_t
id_by_name(const char* name)
{
    return id_by_name(name,
                      std::make_index_sequence<ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LAST>{});
}

std::vector<uint32_t>
get_ids()
{
    auto _data = std::vector<uint32_t>{};
    _data.reserve(ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LAST);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LAST>{});
    return _data;
}

std::vector<const char*>
get_names()
{
    auto _data = std::vector<const char*>{};
    _data.reserve(ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LAST);
    get_names(_data, std::make_index_sequence<ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LAST>{});
    return _data;
}
}  // namespace code_object

namespace
{
using hsa_loader_table_t             = hsa_ven_amd_loader_1_01_pfn_t;
using context_t                      = context::context;
using user_data_t                    = rocprofiler_user_data_t;
using context_array_t                = context::context_array_t;
using context_user_data_map_t        = std::unordered_map<const context_t*, user_data_t>;
using name_array_t                   = std::vector<std::pair<size_t, std::unique_ptr<std::string>>>;
using amd_compute_pgm_rsrc_three32_t = uint32_t;

struct kernel_descriptor_t
{
    uint8_t  reserved0[16];
    int64_t  kernel_code_entry_byte_offset;
    uint8_t  reserved1[20];
    uint32_t compute_pgm_rsrc3;
    uint32_t compute_pgm_rsrc1;
    uint32_t compute_pgm_rsrc2;
    uint16_t kernel_code_properties;
    uint8_t  reserved2[6];
};

// AMD Compute Program Resource Register Three.
enum amd_compute_gfx9_pgm_rsrc_three_t
{
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_ACCUM_OFFSET, 0, 5),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_TG_SPLIT, 16, 1)
};

enum amd_compute_gfx10_gfx11_pgm_rsrc_three_t
{
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_SHARED_VGPR_COUNT, 0, 4),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_INST_PREF_SIZE, 4, 6),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_TRAP_ON_START, 10, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_TRAP_ON_END, 11, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_THREE_IMAGE_OP, 31, 1)
};

// Kernel code properties.
enum amd_kernel_code_property_t
{
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER,
                                     0,
                                     1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR, 1, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR, 2, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR,
                                     3,
                                     1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID, 4, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT, 5, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE,
                                     6,
                                     1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_RESERVED0, 7, 3),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                                     10,
                                     1),  // GFX10+
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK, 11, 1),
    AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTY_RESERVED1, 12, 4),
};

uint32_t
arch_vgpr_count(std::string_view name, kernel_descriptor_t kernel_code)
{
    if(name == "gfx90a" || name.find("gfx94") == 0)
        return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc3,
                                 AMD_COMPUTE_PGM_RSRC_THREE_ACCUM_OFFSET) +
                1) *
               4;

    return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                             AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT) +
            1) *
           (AMD_HSA_BITS_GET(kernel_code.kernel_code_properties,
                             AMD_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32)
                ? 8
                : 4);
}

uint32_t
accum_vgpr_count(std::string_view name, kernel_descriptor_t kernel_code)
{
    if(name == "gfx908")
        return arch_vgpr_count(name, kernel_code);
    else if(name == "gfx90a" || name.find("gfx94") == 0)
        return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                                 AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT) +
                1) *
               (8 - arch_vgpr_count(name, kernel_code));

    bool emplaced = false;
    {
        static auto warned = std::unordered_set<std::string>{};
        static auto mtx    = std::mutex{};
        auto        lk     = std::unique_lock<std::mutex>{mtx};
        emplaced           = warned.emplace(name).second;
    }

    LOG_IF(WARNING, emplaced) << "Missing support for accum_vgpr_count for " << name;
    return 0;
}

uint32_t
sgpr_count(std::string_view name, kernel_descriptor_t kernel_code)
{
    // GFX10 and later always allocate 128 sgprs.
    constexpr uint32_t gfx10_sgprs = 128;

    auto begp = name.find_first_of("0123456789");
    if(!name.empty() && begp != std::string_view::npos)
    {
        auto endp      = name.find_first_not_of("0123456789", begp);
        auto lenp      = (endp - begp) + 1;
        auto gfxip_str = name.substr(begp, lenp);
        auto gfxip_n   = int32_t{0};
        if(!gfxip_str.empty()) gfxip_n = std::stoi(std::string{gfxip_str});

        if(gfxip_n >= 1000)
        {
            return gfx10_sgprs;
        }
        else
        {
            return (AMD_HSA_BITS_GET(kernel_code.compute_pgm_rsrc1,
                                     AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT) /
                        2 +
                    1) *
                   16;
        }
    }

    LOG(WARNING) << "Missing support for sgpr_count for " << name;
    return 0;
}

name_array_t*
get_string_array()
{
    static auto*& _v = common::static_object<name_array_t>::construct();
    return _v;
}

std::string*
get_string_entry(std::string_view name)
{
    auto        _hash_v = std::hash<std::string_view>{}(name);
    static auto _sync   = std::shared_mutex{};
    if(!get_string_array()) return nullptr;

    {
        auto _unlock = common::scope_destructor{[]() { _sync.unlock_shared(); }};
        _sync.lock_shared();
        for(const auto& itr : *get_string_array())
            if(itr.first == _hash_v) return itr.second.get();
    }

    auto _unlock = common::scope_destructor{[]() { _sync.unlock(); }};
    _sync.lock();
    return get_string_array()
        ->emplace_back(std::make_pair(_hash_v, std::make_unique<std::string>(name)))
        .second.get();
}

hsa_loader_table_t&
get_loader_table()
{
    static auto _v = []() {
        auto _val = hsa_loader_table_t{};
        memset(&_val, 0, sizeof(hsa_loader_table_t));
        return _val;
    }();
    return _v;
}

auto*&
get_status_string_function()
{
    static decltype(::hsa_status_string)* _v = nullptr;
    return _v;
}

std::string_view
get_status_string(hsa_status_t _status)
{
    const char* _msg = nullptr;
    if(get_status_string_function() &&
       get_status_string_function()(_status, &_msg) == HSA_STATUS_SUCCESS && _msg)
        return std::string_view{_msg};

    return std::string_view{"(unknown HSA error)"};
}

const kernel_descriptor_t*
get_kernel_descriptor(uint64_t kernel_object)
{
    const kernel_descriptor_t* kernel_code = nullptr;
    if(get_loader_table().hsa_ven_amd_loader_query_host_address == nullptr) return kernel_code;
    hsa_status_t status = get_loader_table().hsa_ven_amd_loader_query_host_address(
        reinterpret_cast<const void*>(kernel_object),  // NOLINT(performance-no-int-to-ptr)
        reinterpret_cast<const void**>(&kernel_code));
    if(status == HSA_STATUS_SUCCESS) return kernel_code;

    LOG(WARNING) << "hsa_ven_amd_loader_query_host_address(kernel_object=" << kernel_object
                 << ") returned " << status << ": " << get_status_string(status);

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<kernel_descriptor_t*>(kernel_object);
}

struct kernel_symbol
{
    using kernel_symbol_data_t =
        rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

    kernel_symbol()  = default;
    ~kernel_symbol() = default;

    kernel_symbol(const kernel_symbol&) = delete;
    kernel_symbol(kernel_symbol&&) noexcept;

    kernel_symbol& operator=(const kernel_symbol&) = delete;
    kernel_symbol& operator                        =(kernel_symbol&&) noexcept;

    bool                    beg_notified   = false;
    bool                    end_notified   = false;
    std::string*            name           = {};
    hsa_executable_t        hsa_executable = {};
    hsa_agent_t             hsa_agent      = {};
    hsa_executable_symbol_t hsa_symbol     = {};
    kernel_symbol_data_t    rocp_data      = common::init_public_api_struct(kernel_symbol_data_t{});
    context_user_data_map_t user_data      = {};
};

kernel_symbol::kernel_symbol(kernel_symbol&& rhs) noexcept { operator=(std::move(rhs)); }

kernel_symbol&
kernel_symbol::operator=(kernel_symbol&& rhs) noexcept
{
    if(this != &rhs)
    {
        beg_notified          = rhs.beg_notified;
        end_notified          = rhs.end_notified;
        name                  = rhs.name;
        hsa_executable        = rhs.hsa_executable;
        hsa_agent             = rhs.hsa_agent;
        hsa_symbol            = rhs.hsa_symbol;
        rocp_data             = rhs.rocp_data;
        user_data             = std::move(rhs.user_data);
        rocp_data.kernel_name = (name) ? name->c_str() : nullptr;
    }

    return *this;
}

bool
operator==(const kernel_symbol& lhs, const kernel_symbol& rhs)
{
    return std::tie(lhs.hsa_executable.handle, lhs.hsa_agent.handle, lhs.hsa_symbol.handle) ==
           std::tie(rhs.hsa_executable.handle, rhs.hsa_agent.handle, rhs.hsa_symbol.handle);
}

struct code_object
{
    using code_object_data_t = rocprofiler_callback_tracing_code_object_load_data_t;
    using symbol_array_t     = std::vector<std::unique_ptr<kernel_symbol>>;

    code_object()  = default;
    ~code_object() = default;

    code_object(const code_object&) = delete;
    code_object(code_object&&) noexcept;

    code_object& operator=(const code_object&) = delete;
    code_object& operator                      =(code_object&&) noexcept;

    bool                     beg_notified    = false;
    bool                     end_notified    = false;
    std::string*             uri             = {};
    hsa_executable_t         hsa_executable  = {};
    hsa_loaded_code_object_t hsa_code_object = {};
    code_object_data_t       rocp_data       = common::init_public_api_struct(code_object_data_t{});
    symbol_array_t           symbols         = {};
    context_array_t          contexts        = {};
    context_user_data_map_t  user_data       = {};
};

code_object::code_object(code_object&& rhs) noexcept { operator=(std::move(rhs)); }

code_object&
code_object::operator=(code_object&& rhs) noexcept
{
    if(this != &rhs)
    {
        beg_notified    = rhs.beg_notified;
        end_notified    = rhs.end_notified;
        uri             = rhs.uri;
        hsa_executable  = rhs.hsa_executable;
        hsa_code_object = rhs.hsa_code_object;
        rocp_data       = rhs.rocp_data;
        user_data       = std::move(rhs.user_data);
        rocp_data.uri   = (uri) ? uri->c_str() : nullptr;
        symbols         = std::move(rhs.symbols);
    }

    return *this;
}

bool
operator==(const code_object& lhs, const code_object& rhs)
{
    return std::tie(lhs.hsa_executable.handle, lhs.hsa_code_object.handle) ==
           std::tie(rhs.hsa_executable.handle, rhs.hsa_code_object.handle);
}

struct code_object_unload
{
    code_object*                object  = nullptr;
    std::vector<kernel_symbol*> symbols = {};
};

auto&
get_code_object_id()
{
    static auto _v = std::atomic<uint64_t>{};
    return _v;
}

auto&
get_kernel_symbol_id()
{
    static auto _v = std::atomic<uint64_t>{};
    return _v;
}

using code_object_array_t        = std::vector<std::unique_ptr<code_object>>;
using kernel_object_map_t        = std::unordered_map<uint64_t, uint64_t>;
using executable_array_t         = std::vector<hsa_executable_t>;
using code_object_unload_array_t = std::vector<code_object_unload>;

std::vector<code_object_unload>
shutdown(hsa_executable_t executable);

bool is_shutdown = false;

auto&
get_executables()
{
    static auto _v = common::Synchronized<executable_array_t>{};
    return _v;
}

auto&
get_code_objects()
{
    static auto _v    = common::Synchronized<code_object_array_t>{};
    static auto _dtor = common::scope_destructor{[]() { code_object_shutdown(); }};
    return _v;
}

auto&
get_kernel_object_map()
{
    static auto _v = common::Synchronized<kernel_object_map_t>{};
    return _v;
}

hsa_status_t
executable_iterate_agent_symbols_load_callback(hsa_executable_t        executable,
                                               hsa_agent_t             agent,
                                               hsa_executable_symbol_t symbol,
                                               void*                   args)
{
#define ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(...)                                                     \
    {                                                                                              \
        auto _status = core_table.hsa_executable_symbol_get_info_fn(symbol, __VA_ARGS__);          \
        LOG_IF(ERROR, _status != HSA_STATUS_SUCCESS)                                               \
            << "core_table.hsa_executable_symbol_get_info_fn(hsa_executable_symbol_t{.handle="     \
            << symbol.handle << "}, " << #__VA_ARGS__ << " failed";                                \
        if(_status != HSA_STATUS_SUCCESS) return _status;                                          \
    }

    auto& core_table = *get_table().core_;
    auto* code_obj_v = static_cast<code_object*>(args);
    auto  symbol_v   = kernel_symbol{};
    auto& data       = symbol_v.rocp_data;

    symbol_v.hsa_executable = executable;
    symbol_v.hsa_agent      = agent;
    symbol_v.hsa_symbol     = symbol;

    auto exists = std::any_of(code_obj_v->symbols.begin(),
                              code_obj_v->symbols.end(),
                              [&symbol_v](auto& itr) { return (itr && symbol_v == *itr); });

    // if there is an existing matching kernel symbol, return success and move onto next symbol
    if(exists) return HSA_STATUS_SUCCESS;

    LOG_IF(FATAL, data.size == 0) << "kernel symbol did not properly initialized the size field "
                                     "upon construction (this is likely a compiler bug)";

    auto type = hsa_symbol_kind_t{};
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &type);

    if(type != HSA_SYMBOL_KIND_KERNEL) return HSA_STATUS_SUCCESS;

    // set the code object id
    data.code_object_id = code_obj_v->rocp_data.code_object_id;

    // compute the kernel name length
    constexpr auto name_length_max = std::numeric_limits<uint32_t>::max();
    uint32_t       _name_length    = 0;
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &_name_length);

    ROCP_CI_LOG_IF(WARNING, _name_length > name_length_max / 2)
        << "kernel symbol name length is extremely large: " << _name_length;

    // set the kernel name
    if(_name_length > 0 && _name_length < name_length_max)
    {
        auto _name = std::string(_name_length + 1, '\0');
        ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_NAME, _name.data());

        symbol_v.name = get_string_entry(_name.substr(0, _name.find_first_of('\0')));
    }
    data.kernel_name = (symbol_v.name) ? symbol_v.name->c_str() : nullptr;

    // these should all be self-explanatory
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                      &data.kernel_object);
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                                      &data.kernarg_segment_size);
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
                                      &data.kernarg_segment_alignment);
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                      &data.group_segment_size);
    ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                      &data.private_segment_size);

    // This works for gfx9 but may not for Navi arch
    const auto* kernel_descript = get_kernel_descriptor(data.kernel_object);
    if(CHECK_NOTNULL(code_obj_v) && CHECK_NOTNULL(kernel_descript))
    {
        const auto* rocp_agent = agent::get_agent(code_obj_v->rocp_data.rocp_agent);
        if(CHECK_NOTNULL(rocp_agent))
        {
            data.arch_vgpr_count  = arch_vgpr_count(rocp_agent->name, *kernel_descript);
            data.accum_vgpr_count = accum_vgpr_count(rocp_agent->name, *kernel_descript);
            data.sgpr_count       = sgpr_count(rocp_agent->name, *kernel_descript);
        }
    }

    // if we have reached this point (i.e. there were no HSA errors returned within macro) then we
    // generate a unique kernel symbol id
    data.kernel_id = ++get_kernel_symbol_id();

    get_kernel_object_map().wlock(
        [](kernel_object_map_t& object_map, uint64_t _kern_obj, uint64_t _kern_id) {
            object_map[_kern_obj] = _kern_id;
        },
        data.kernel_object,
        data.kernel_id);

    code_obj_v->symbols.emplace_back(std::make_unique<kernel_symbol>(std::move(symbol_v)));

    return HSA_STATUS_SUCCESS;

#undef ROCP_HSA_CORE_GET_EXE_SYMBOL_INFO
}

hsa_status_t
executable_iterate_agent_symbols_unload_callback(hsa_executable_t        executable,
                                                 hsa_agent_t             agent,
                                                 hsa_executable_symbol_t symbol,
                                                 void*                   args)
{
    auto symbol_v           = kernel_symbol{};
    symbol_v.hsa_executable = executable;
    symbol_v.hsa_agent      = agent;
    symbol_v.hsa_symbol     = symbol;

    auto* code_obj_v = static_cast<code_object_unload*>(args);
    CHECK_NOTNULL(code_obj_v);
    CHECK_NOTNULL(code_obj_v->object);

    for(const auto& itr : code_obj_v->object->symbols)
    {
        if(itr && *itr == symbol_v) code_obj_v->symbols.emplace_back(itr.get());
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
code_object_load_callback(hsa_executable_t         executable,
                          hsa_loaded_code_object_t loaded_code_object,
                          void*                    cb_data)
{
#define ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(...)                                              \
    {                                                                                              \
        auto _status = loader_table.hsa_ven_amd_loader_loaded_code_object_get_info(                \
            loaded_code_object, __VA_ARGS__);                                                      \
        LOG_IF(ERROR, _status != HSA_STATUS_SUCCESS)                                               \
            << "loader_table.hsa_ven_amd_loader_loaded_code_object_get_info(loaded_code_object, "  \
            << #__VA_ARGS__ << " failed";                                                          \
        if(_status != HSA_STATUS_SUCCESS) return _status;                                          \
    }

    auto& loader_table  = get_loader_table();
    auto  code_obj_v    = code_object{};
    auto& data          = code_obj_v.rocp_data;
    int   _storage_type = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_NONE;

    LOG_IF(FATAL, data.size == 0) << "code object did not properly initialized the size field upon "
                                     "construction (this is likely a compiler bug)";

    code_obj_v.hsa_executable  = executable;
    code_obj_v.hsa_code_object = loaded_code_object;

    auto* code_obj_vec = static_cast<code_object_array_t*>(cb_data);
    auto exists = std::any_of(code_obj_vec->begin(), code_obj_vec->end(), [&code_obj_v](auto& itr) {
        return (itr && code_obj_v == *itr);
    });

    // if there is an existing matching code object, check for any new symbols and then return
    // success and move onto next code object
    if(exists)
    {
        for(auto& itr : *code_obj_vec)
        {
            if(itr && *itr == code_obj_v)
            {
                get_table().core_->hsa_executable_iterate_agent_symbols_fn(
                    executable,
                    data.hsa_agent,
                    executable_iterate_agent_symbols_load_callback,
                    itr.get());
            }
        }

        return HSA_STATUS_SUCCESS;
    }

    ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(
        HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE, &_storage_type);

    LOG_IF(FATAL, _storage_type >= ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_LAST)
        << "HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE returned an "
           "unsupported code object storage type. Expected 0=none, 1=file, or 2=memory but "
           "received a value of "
        << _storage_type;

    data.storage_type = static_cast<rocprofiler_code_object_storage_type_t>(_storage_type);

    if(_storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_FILE)
    {
        ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_FILE,
            &data.storage_file);
    }
    else if(_storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
    {
        ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
            &data.memory_base);
        ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
            &data.memory_size);
    }
    else if(_storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_NONE)
    {
        LOG(WARNING) << "Code object storage type of none was ignored";
        return HSA_STATUS_SUCCESS;
    }

    ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                                             &data.load_base);

    ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
                                             &data.load_size);

    ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA,
                                             &data.load_delta);

    constexpr auto uri_length_max = std::numeric_limits<uint32_t>::max();
    auto           _uri_length    = uint32_t{0};
    ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH,
                                             &_uri_length);

    ROCP_CI_LOG_IF(WARNING, _uri_length > uri_length_max / 2)
        << "code object uri length is extremely large: " << _uri_length;

    if(_uri_length > 0 && _uri_length < uri_length_max)
    {
        auto _uri = std::string(_uri_length + 1, '\0');
        ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI,
                                                 _uri.data());

        code_obj_v.uri = get_string_entry(_uri);
    }
    data.uri = (code_obj_v.uri) ? code_obj_v.uri->data() : nullptr;

    auto _hsa_agent = hsa_agent_t{};
    ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO(HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
                                             &data.hsa_agent);

    const auto* _rocp_agent = agent::get_rocprofiler_agent(data.hsa_agent);
    if(!_rocp_agent)
    {
        ROCP_CI_LOG(ERROR) << "hsa agent (handle=" << _hsa_agent.handle
                           << ") did not map to a rocprofiler agent";
        return HSA_STATUS_ERROR_INVALID_AGENT;
    }
    data.rocp_agent = _rocp_agent->id;

    // if we have reached this point (i.e. there were no HSA errors returned within macro) then we
    // generate a unique code object id
    data.code_object_id = ++get_code_object_id();

    auto _status = get_table().core_->hsa_executable_iterate_agent_symbols_fn(
        executable, data.hsa_agent, executable_iterate_agent_symbols_load_callback, &code_obj_v);

    if(_status == HSA_STATUS_SUCCESS)
    {
        code_obj_vec->emplace_back(std::make_unique<code_object>(std::move(code_obj_v)));
    }
    else
    {
        LOG(ERROR) << "hsa_executable_iterate_agent_symbols failed for " << data.uri;
    }

    return _status;

#undef ROCP_HSA_VEN_LOADER_GET_CODE_OBJECT_INFO
}

hsa_status_t
code_object_unload_callback(hsa_executable_t         executable,
                            hsa_loaded_code_object_t loaded_code_object,
                            void*                    args)
{
    auto code_obj_v            = code_object{};
    code_obj_v.hsa_executable  = executable;
    code_obj_v.hsa_code_object = loaded_code_object;

    auto* code_obj_arr = static_cast<code_object_unload_array_t*>(args);

    CHECK_NOTNULL(code_obj_arr);

    // auto _size = get_code_objects().rlock([](const auto& data) { return data.size(); });
    // LOG(INFO) << "[inp] executable=" << executable.handle
    //            << ", code_object=" << loaded_code_object.handle << " vs. " << _size;

    get_code_objects().rlock([&](const code_object_array_t& arr) {
        for(const auto& itr : arr)
        {
            // LOG(INFO) << "[cmp] executable=" << itr->hsa_executable.handle
            //            << ", code_object=" << itr->hsa_code_object.handle;
            if(itr->hsa_executable.handle == executable.handle &&
               itr->hsa_code_object.handle == loaded_code_object.handle)
            // if(itr && *itr == code_obj_v)
            {
                auto& _last = code_obj_arr->emplace_back(code_object_unload{.object = itr.get()});

                auto agent = itr->rocp_data.hsa_agent;
                get_table().core_->hsa_executable_iterate_agent_symbols_fn(
                    executable, agent, executable_iterate_agent_symbols_unload_callback, &_last);
            }
        }
    });

    return HSA_STATUS_SUCCESS;
}

auto&
get_freeze_function()
{
    static decltype(::hsa_executable_freeze)* _v = nullptr;
    return _v;
}

auto&
get_destroy_function()
{
    static decltype(::hsa_executable_destroy)* _v = nullptr;
    return _v;
}

hsa_status_t
executable_freeze(hsa_executable_t executable, const char* options)
{
    hsa_status_t status = CHECK_NOTNULL(get_freeze_function())(executable, options);
    if(status != HSA_STATUS_SUCCESS) return status;

    LOG(INFO) << "running " << __FUNCTION__ << " (executable=" << executable.handle << ")...";

    get_executables().wlock(
        [executable](executable_array_t& data) { data.emplace_back(executable); });

    auto& code_obj_vec = get_code_objects();
    code_obj_vec.wlock([executable](code_object_array_t& _vec) {
        hsa::get_loader_table().hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
            executable, code_object_load_callback, &_vec);
    });

    constexpr auto CODE_OBJECT_KIND = ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT;
    constexpr auto CODE_OBJECT_LOAD = ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LOAD;
    constexpr auto CODE_OBJECT_KERNEL_SYMBOL =
        ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;

    auto&& context_filter = [](const context_t* ctx) {
        return (ctx->callback_tracer && ctx->callback_tracer->domains(CODE_OBJECT_KIND) &&
                (ctx->callback_tracer->domains(CODE_OBJECT_KIND, CODE_OBJECT_LOAD) ||
                 ctx->callback_tracer->domains(CODE_OBJECT_KIND, CODE_OBJECT_KERNEL_SYMBOL)));
    };

    static thread_local auto ctxs = context_array_t{};
    context::get_active_contexts(ctxs, std::move(context_filter));

    if(!ctxs.empty())
    {
        code_obj_vec.rlock([](const code_object_array_t& data) {
            auto tidx = common::get_tid();
            // set the contexts for each code object
            for(const auto& ditr : data)
                ditr->contexts = ctxs;

            for(const auto& ditr : data)
            {
                for(const auto* citr : ditr->contexts)
                {
                    if(citr->callback_tracer->domains(CODE_OBJECT_KIND, CODE_OBJECT_LOAD))
                    {
                        if(!ditr->beg_notified)
                        {
                            auto co_data = ditr->rocp_data;
                            auto record  = rocprofiler_callback_tracing_record_t{
                                .context_id     = rocprofiler_context_id_t{citr->context_idx},
                                .thread_id      = tidx,
                                .correlation_id = rocprofiler_correlation_id_t{},
                                .kind           = CODE_OBJECT_KIND,
                                .operation      = CODE_OBJECT_LOAD,
                                .phase          = ROCPROFILER_CALLBACK_PHASE_LOAD,
                                .payload        = static_cast<void*>(&co_data)};

                            // invoke callback
                            auto& cb_data =
                                citr->callback_tracer->callback_data.at(CODE_OBJECT_KIND);
                            auto& user_data = ditr->user_data[citr];
                            cb_data.callback(record, &user_data, cb_data.data);
                        }
                    }

                    for(const auto& sitr : ditr->symbols)
                    {
                        if(sitr && citr->callback_tracer->domains(CODE_OBJECT_KIND,
                                                                  CODE_OBJECT_KERNEL_SYMBOL))
                        {
                            if(!sitr->beg_notified)
                            {
                                auto sym_data = sitr->rocp_data;
                                auto record   = rocprofiler_callback_tracing_record_t{
                                    .context_id     = rocprofiler_context_id_t{citr->context_idx},
                                    .thread_id      = tidx,
                                    .correlation_id = rocprofiler_correlation_id_t{},
                                    .kind           = CODE_OBJECT_KIND,
                                    .operation      = CODE_OBJECT_KERNEL_SYMBOL,
                                    .phase          = ROCPROFILER_CALLBACK_PHASE_LOAD,
                                    .payload        = static_cast<void*>(&sym_data)};

                                // invoke callback
                                auto& cb_data =
                                    citr->callback_tracer->callback_data.at(CODE_OBJECT_KIND);
                                auto& user_data = sitr->user_data[citr];
                                cb_data.callback(record, &user_data, cb_data.data);
                            }
                        }
                    }
                }
            }

            for(const auto& ditr : data)
            {
                ditr->beg_notified = true;
                for(auto& sitr : ditr->symbols)
                    sitr->beg_notified = true;
            }
        });
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
executable_destroy(hsa_executable_t executable)
{
    if(is_shutdown) return HSA_STATUS_SUCCESS;

    auto _unloaded = shutdown(executable);

    get_kernel_object_map().wlock([_unloaded](kernel_object_map_t& data) {
        for(const auto& uitr : _unloaded)
        {
            for(const auto& sitr : uitr.symbols)
            {
                data.erase(sitr->rocp_data.kernel_id);
            }
        }
    });

    get_code_objects().wlock([executable](code_object_array_t& data) {
        for(auto& itr : data)
        {
            if(itr->hsa_executable.handle == executable.handle) itr.reset();
        }
        data.erase(
            std::remove_if(data.begin(), data.end(), [](auto& itr) { return (itr == nullptr); }),
            data.end());
    });

    get_executables().wlock([executable](executable_array_t& data) {
        data.erase(std::remove_if(data.begin(),
                                  data.end(),
                                  [executable](hsa_executable_t itr) {
                                      return (itr.handle == executable.handle);
                                  }),
                   data.end());
    });

    return CHECK_NOTNULL(get_destroy_function())(executable);
}

std::vector<code_object_unload>
shutdown(hsa_executable_t executable)
{
    LOG(INFO) << "running " << __FUNCTION__ << " (executable=" << executable.handle << ")...";

    auto _unloaded = std::vector<code_object_unload>{};
    hsa::get_loader_table().hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
        executable, code_object_unload_callback, &_unloaded);

    constexpr auto CODE_OBJECT_KIND = ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT;
    constexpr auto CODE_OBJECT_LOAD = ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LOAD;
    constexpr auto CODE_OBJECT_KERNEL_SYMBOL =
        ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;

    auto tidx = common::get_tid();
    for(auto& itr : _unloaded)
    {
        LOG_IF(FATAL, itr.object == nullptr);
        for(const auto* citr : itr.object->contexts)
        {
            if(citr->callback_tracer->domains(CODE_OBJECT_KIND, CODE_OBJECT_LOAD))
            {
                if(!itr.object->end_notified)
                {
                    auto record = rocprofiler_callback_tracing_record_t{
                        .context_id     = rocprofiler_context_id_t{citr->context_idx},
                        .thread_id      = tidx,
                        .correlation_id = rocprofiler_correlation_id_t{},
                        .kind           = CODE_OBJECT_KIND,
                        .operation      = CODE_OBJECT_LOAD,
                        .phase          = ROCPROFILER_CALLBACK_PHASE_UNLOAD,
                        .payload        = static_cast<void*>(&itr.object->rocp_data)};

                    // invoke callback
                    auto& cb_data   = citr->callback_tracer->callback_data.at(CODE_OBJECT_KIND);
                    auto& user_data = itr.object->user_data.at(citr);
                    cb_data.callback(record, &user_data, cb_data.data);
                }
            }

            // generate callbacks for kernel symbols after the callback for code object
            // unloading so the code object unload can be used to flush the buffer before the
            // symbol information is removed
            if(citr->callback_tracer->domains(CODE_OBJECT_KIND, CODE_OBJECT_KERNEL_SYMBOL))
            {
                for(auto& sitr : itr.symbols)
                {
                    if(!sitr->end_notified)
                    {
                        auto record = rocprofiler_callback_tracing_record_t{
                            .context_id     = rocprofiler_context_id_t{citr->context_idx},
                            .thread_id      = tidx,
                            .correlation_id = rocprofiler_correlation_id_t{},
                            .kind           = CODE_OBJECT_KIND,
                            .operation      = CODE_OBJECT_KERNEL_SYMBOL,
                            .phase          = ROCPROFILER_CALLBACK_PHASE_UNLOAD,
                            .payload        = static_cast<void*>(&sitr->rocp_data)};

                        // invoke callback
                        auto& cb_data   = citr->callback_tracer->callback_data.at(CODE_OBJECT_KIND);
                        auto& user_data = sitr->user_data.at(citr);
                        cb_data.callback(record, &user_data, cb_data.data);
                    }
                }
            }
        }
    }

    for(auto& itr : _unloaded)
    {
        itr.object->end_notified = true;
        for(auto& sitr : itr.symbols)
            sitr->end_notified = true;
    }

    return _unloaded;
}
}  // namespace

void
code_object_init(HsaApiTable* table)
{
    auto& core_table = *table->core_;

    get_status_string_function() = core_table.hsa_status_string_fn;

    auto _status = core_table.hsa_system_get_major_extension_table_fn(
        HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_loader_table_t), &get_loader_table());

    LOG_IF(ERROR, _status != HSA_STATUS_SUCCESS)
        << "hsa_system_get_major_extension_table failed: " << get_status_string(_status);

    if(_status == HSA_STATUS_SUCCESS)
    {
        get_freeze_function()                = CHECK_NOTNULL(core_table.hsa_executable_freeze_fn);
        get_destroy_function()               = CHECK_NOTNULL(core_table.hsa_executable_destroy_fn);
        core_table.hsa_executable_freeze_fn  = executable_freeze;
        core_table.hsa_executable_destroy_fn = executable_destroy;
        LOG_IF(FATAL, get_freeze_function() == core_table.hsa_executable_freeze_fn)
            << "infinite recursion";
        LOG_IF(FATAL, get_destroy_function() == core_table.hsa_executable_destroy_fn)
            << "infinite recursion";
    }
}

uint64_t
get_kernel_id(uint64_t kernel_object)
{
    return get_kernel_object_map().rlock(
        [](const kernel_object_map_t& object_map, uint64_t _kern_obj) -> uint64_t {
            auto itr = object_map.find(_kern_obj);
            return (itr == object_map.end()) ? 0 : itr->second;
        },
        kernel_object);
}

void
code_object_shutdown()
{
    if(is_shutdown) return;

    get_executables().rlock([](const executable_array_t& edata) {
        auto tmp = edata;
        std::reverse(tmp.begin(), tmp.end());
        for(auto itr : tmp)
            shutdown(itr);
    });

    get_code_objects().wlock([](code_object_array_t& data) { data.clear(); });

    is_shutdown = true;
}
}  // namespace hsa
}  // namespace rocprofiler
