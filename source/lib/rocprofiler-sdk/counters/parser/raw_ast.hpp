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

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <glog/logging.h>

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"

namespace rocprofiler
{
namespace counters
{
enum NodeType
{
    NONE = 0,
    ADDITION_NODE,
    DIVIDE_NODE,
    MULTIPLY_NODE,
    NUMBER_NODE,
    RANGE_NODE,
    REDUCE_NODE,
    REFERENCE_NODE,
    SELECT_NODE,
    SUBTRACTION_NODE,
    CONSTANT_NODE,
};

struct LinkedList
{
    std::string name;
    int         data{-1};
    LinkedList* next{nullptr};
    LinkedList(const char* v, LinkedList* next_node)
    : name(std::string{CHECK_NOTNULL(v)})
    , next(next_node)
    {}
    LinkedList(const char* v, int d, LinkedList* next_node)
    : name(std::string{CHECK_NOTNULL(v)})
    , data(d)
    , next(next_node)
    {}
};

struct RawAST
{
    // Node type
    NodeType    type{NONE};  // Operation to perform on the counter set
    std::string reduce_op{};

    // Stores either the name or digit dependening on whether this
    // is a name or number
    std::variant<std::monostate, std::string, int64_t> value{std::monostate{}};

    // Counter set of ASTs needed to compute this counter.
    // Operation is applied to all counters in this set.
    std::vector<RawAST*> counter_set;

    // Dimension set to remove dimensions (such as shader engine)
    // from the result.
    std::unordered_set<rocprofiler_profile_counter_instance_types> reduce_dimension_set;

    // Dimension set to select certain dimensions from the result
    std::unordered_map<rocprofiler_profile_counter_instance_types, int> select_dimension_set;

    // Range restriction on this node
    RawAST* range{nullptr};

    ~RawAST()
    {
        auto deleteVec = [](auto& vec) {
            for(auto val : vec)
            {
                delete val;
            }
        };

        deleteVec(counter_set);
        delete range;
    }

    // Constructors for raw value types
    RawAST(NodeType t, const char* v)
    : type(t)
    , value(std::string{CHECK_NOTNULL(v)})
    {}

    RawAST(NodeType t, int64_t v)
    : type(t)
    , value(v)
    {}

    static const auto& get_dim_map()
    {
        static const auto dim_map = []() {
            std::map<std::string, rocprofiler_profile_counter_instance_types> out;
            const auto& dims = dimension_map();
            for(const auto& [id, name] : dims)
            {
                out.emplace(name, id);
            }
            return out;
        }();
        return dim_map;
    }

    // Reduce operation constructor. Counter is the counter AST
    // to use for the reduce op, op is how to reduce (i.e. SUM,AVG,etc),
    // dimensions contains the set of dimensions which we want to keep
    // in the result. Dimensions not specified are all reduced according to op
    RawAST(NodeType t, RawAST* counter, const char* op, LinkedList* dimensions)
    : type(t)
    , reduce_op(CHECK_NOTNULL(op))
    , counter_set({counter})
    {
        CHECK_EQ(t, REDUCE_NODE);
        if(dimensions)
        {
            while(dimensions)
            {
                const rocprofiler_profile_counter_instance_types* dim =
                    rocprofiler::common::get_val(get_dim_map(), std::string{dimensions->name});
                if(!dim)
                {
                    throw std::runtime_error(
                        fmt::format("Unknown Dimension - {}", dimensions->name));
                }

                reduce_dimension_set.insert(*dim);
                LinkedList* current = dimensions;
                dimensions          = dimensions->next;
                delete current;
            }
        }
    }

    // Select operation constructor. Counter is the counter AST
    // to use for the reduce op, refs is the reference set AST.
    // dimensions contains the mapping for selecting dimensions
    // (XCC=1,SE=2,...)
    RawAST(NodeType t, RawAST* counter, LinkedList* dimensions)
    : type(t)
    , counter_set({counter})
    {
        if(dimensions)
        {
            LinkedList* ptr = dimensions;
            while(ptr)
            {
                const rocprofiler_profile_counter_instance_types* dim =
                    rocprofiler::common::get_val(get_dim_map(), dimensions->name);
                if(!dim)
                {
                    throw std::runtime_error(
                        fmt::format("Unknown Dimension - {}", dimensions->name));
                }

                select_dimension_set.insert({*dim, ptr->data});
                LinkedList* current = ptr;
                ptr                 = ptr->next;
                delete current;
            }
        }
        else
        {
            LOG(ERROR) << "select_dimension_set creation failed.";
        }
    }

    RawAST(NodeType t, std::vector<RawAST*> c)
    : type(t)
    , counter_set(std::move(c))
    {}
};
}  // namespace counters
}  // namespace rocprofiler

namespace fmt
{
// fmt::format support for RawAST
template <>
struct formatter<rocprofiler::counters::RawAST>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(rocprofiler::counters::RawAST const& ast, Ctx& ctx) const
    {
        static const std::map<rocprofiler::counters::NodeType, std::string> NodeTypeToString = {
            {rocprofiler::counters::NONE, "NONE"},
            {rocprofiler::counters::ADDITION_NODE, "ADDITION_NODE"},
            {rocprofiler::counters::DIVIDE_NODE, "DIVIDE_NODE"},
            {rocprofiler::counters::MULTIPLY_NODE, "MULTIPLY_NODE"},
            {rocprofiler::counters::NUMBER_NODE, "NUMBER_NODE"},
            {rocprofiler::counters::RANGE_NODE, "RANGE_NODE"},
            {rocprofiler::counters::REDUCE_NODE, "REDUCE_NODE"},
            {rocprofiler::counters::REFERENCE_NODE, "REFERENCE_NODE"},
            {rocprofiler::counters::SELECT_NODE, "SELECT_NODE"},
            {rocprofiler::counters::SUBTRACTION_NODE, "SUBTRACTION_NODE"},
        };

        auto out = fmt::format_to(ctx.out(),
                                  "{{\"Type\":\"{}\", \"REDUCE_OP\":\"{}\",",
                                  NodeTypeToString.at(ast.type),
                                  ast.reduce_op);

        if(const auto* string_val = std::get_if<std::string>(&ast.value))
        {
            out = fmt::format_to(out, " \"Value\":\"{}\",", *string_val);
        }
        else if(const auto* int_val = std::get_if<int64_t>(&ast.value))
        {
            out = fmt::format_to(out, " \"Value\":{},", *int_val);
        }

        if(ast.range)
        {
            out = fmt::format_to(out, " \"Range\":{},", *ast.range);
        }

        out = fmt::format_to(out, " \"Counter_Set\":[");
        for(const auto& ref : ast.counter_set)
        {
            out = fmt::format_to(
                out, "{}{}", *CHECK_NOTNULL(ref), ref == ast.counter_set.back() ? "" : ",");
        }

        out                   = fmt::format_to(out, "], \"Reduce_Dimension_Set\":[");
        size_t ReduceSetIndex = 0;
        for(const auto& ref : ast.reduce_dimension_set)
        {
            out = fmt::format_to(out,
                                 "\"{}\"{}",
                                 static_cast<int>(ref),
                                 ++ReduceSetIndex == ast.reduce_dimension_set.size() ? "" : ",");
        }

        out                   = fmt::format_to(out, "], \"Select_Dimension_Set\":[");
        size_t SelectSetIndex = 0;
        for(const auto& [type, val] : ast.select_dimension_set)
        {
            out = fmt::format_to(out,
                                 "\"{},{}\"{}",
                                 static_cast<int>(type),
                                 val,
                                 ++SelectSetIndex == ast.select_dimension_set.size() ? "" : ",");
        }

        return fmt::format_to(out, "]}}");
    }
};
}  // namespace fmt
