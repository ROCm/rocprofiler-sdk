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

#include <algorithm>
#include <cstdint>
#include <tuple>

#include <fmt/core.h>
#include <gtest/gtest.h>

#include "evaluate_ast_test.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/counters/parser/reader.hpp"

namespace
{
ReduceOperation
get_reduce_op_type_from_string(const std::string& op)
{
    static const std::unordered_map<std::string, ReduceOperation> reduce_op_string_to_type = {
        {"min", REDUCE_MIN}, {"max", REDUCE_MAX}, {"sum", REDUCE_SUM}, {"avr", REDUCE_AVG}};

    ReduceOperation type           = REDUCE_NONE;
    const auto*     reduce_op_type = rocprofiler::common::get_val(reduce_op_string_to_type, op);
    if(reduce_op_type) type = *reduce_op_type;
    return type;
}

bool
isIdentical(const EvaluateAST& eval_ast, const RawAST& raw_ast)
{
    if(raw_ast.counter_set.size() != eval_ast.children().size() ||
       raw_ast.type != eval_ast.type() ||
       get_reduce_op_type_from_string(raw_ast.reduce_op) != eval_ast.reduce_op())
    {
        return false;
    }

    for(size_t i = 0; i < raw_ast.counter_set.size(); i++)
    {
        if(!isIdentical(eval_ast.children()[i], *raw_ast.counter_set[i]))
        {
            return false;
        }
    }
    return true;
}
}  // namespace

TEST(evaluate_ast, basic_copy)
{
    std::unordered_map<std::string, Metric> metrics = {
        {"SQ_WAVES", Metric("gfx9", "a", "a", "a", "a", "a", "", 0)},
        {"TCC_HIT", Metric("gfx9", "b", "b", "b", "b", "b", "", 1)}};

    RawAST* ast = nullptr;
    auto*   buf = yy_scan_string("SQ_WAVES + TCC_HIT");
    yyparse(&ast);
    ASSERT_TRUE(ast);

    auto eval_ast = EvaluateAST({.handle = 0}, metrics, *ast, "gfx9");

    EXPECT_TRUE(isIdentical(eval_ast, *ast));
    yy_delete_buffer(buf);
    delete ast;
}

TEST(evaluate_ast, counter_expansion)
{
    std::unordered_map<std::string, Metric> metrics = {
        {"SQ_WAVES", Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)},
        {"TCC_HIT", Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1)},
        {"TEST_DERRIVED",
         Metric("gfx9", "TEST_DERRIVED", "C", "C", "C", "SQ_WAVES+TCC_HIT", "", 2)}};

    std::unordered_map<std::string, EvaluateAST> asts;
    for(auto [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        asts.emplace(val, EvaluateAST({.handle = metric.id()}, metrics, *ast, "gfx9"));
        yy_delete_buffer(buf);
        delete ast;
    }

    std::set<Metric> required_counters;
    asts.at("TEST_DERRIVED").get_required_counters(asts, required_counters);
    EXPECT_EQ(required_counters.size(), 2);
    auto expected = std::set<Metric>{{Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1),
                                      Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)}};

    for(auto& counter_found : required_counters)
    {
        EXPECT_NE(expected.find(counter_found), expected.end());
    }
}

TEST(evaluate_ast, counter_expansion_multi_derived)
{
    std::unordered_map<std::string, Metric> metrics = {
        {"SQ_WAVES", Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)},
        {"TCC_HIT", Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1)},
        {"TEST_DERRIVED",
         Metric("gfx9", "TEST_DERRIVED", "C", "C", "C", "SQ_WAVES+TCC_HIT", "", 2)},
        {"TEST_DERRIVED3",
         Metric("gfx9", "TEST_DERRIVED3", "C", "C", "C", "TEST_DERRIVED+SQ_WAVES+TCC_HIT", "", 3)}};

    std::unordered_map<std::string, EvaluateAST> asts;
    for(auto [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        asts.emplace(val, EvaluateAST({.handle = metric.id()}, metrics, *ast, "gfx9"));
        yy_delete_buffer(buf);
        delete ast;
    }

    std::set<Metric> required_counters;
    asts.at("TEST_DERRIVED3").get_required_counters(asts, required_counters);
    EXPECT_EQ(required_counters.size(), 2);
    auto expected = std::set<Metric>{{Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1),
                                      Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)}};

    for(auto& counter_found : required_counters)
    {
        EXPECT_NE(expected.find(counter_found), expected.end());
    }
}

TEST(evaluate_ast, counter_expansion_order)
{
    std::unordered_map<std::string, Metric> metrics = {
        {"SQ_WAVES", Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)},
        {"TCC_HIT", Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1)},
        {"VLL", Metric("gfx9", "VLL", "b", "b", "b", "", "", 4)},
        {"TEST_DERRIVED", Metric("gfx9", "TEST_DERRIVED", "C", "C", "C", "SQ_WAVES+VLL", "", 2)},
        {"TEST_DERRIVED3",
         Metric("gfx9", "TEST_DERRIVED3", "C", "C", "C", "TEST_DERRIVED+SQ_WAVES+TCC_HIT", "", 3)}};

    std::unordered_map<std::string, EvaluateAST> asts;
    for(auto [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast);
        asts.emplace(val, EvaluateAST({.handle = metric.id()}, metrics, *ast, "gfx9"));
        yy_delete_buffer(buf);
        delete ast;
    }

    std::set<Metric> required_counters;
    asts.at("TEST_DERRIVED3").get_required_counters(asts, required_counters);
    EXPECT_EQ(required_counters.size(), 3);
    auto expected = std::set<Metric>{{Metric("gfx9", "VLL", "b", "b", "b", "", "", 4),
                                      Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1),
                                      Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)}};

    for(auto& counter_found : required_counters)
    {
        EXPECT_NE(expected.find(counter_found), expected.end());
    }
}

TEST(evaluate_ast, counter_expansion_function)
{
    std::unordered_map<std::string, Metric> metrics = {
        {"SQ_WAVES", Metric("gfx9", "SQ_WAVES", "a", "a", "a", "", "", 0)},
        {"TCC_HIT", Metric("gfx9", "TCC_HIT", "b", "b", "b", "", "", 1)},
        {"VLL", Metric("gfx9", "VLL", "b", "b", "b", "", "", 4)},
        {"TEST_DERRIVED", Metric("gfx9", "TEST_DERRIVED", "C", "C", "C", "SQ_WAVES+VLL", "", 2)}};

    std::unordered_map<std::string, EvaluateAST> asts;
    for(auto [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast) << metric.expression() << " " << metric.name();
        asts.emplace(val, EvaluateAST({.handle = metric.id()}, metrics, *ast, "gfx9"));
        yy_delete_buffer(buf);
        delete ast;
    }
}

namespace
{
void
add_constants(std::unordered_map<std::string, Metric>& metrics, uint64_t start_id)
{
    // Ensure topology is read
    rocprofiler::agent::get_agents();
    for(const auto& prop : rocprofiler::agent::get_agent_available_properties())
    {
        LOG(ERROR) << prop;
        metrics[prop] = {"constant",
                         prop,
                         "",
                         "",
                         fmt::format("Constant value {} from agent properties", prop),
                         "",
                         "yes",
                         start_id};
        start_id++;
    }
}

}  // namespace

TEST(evaluate_ast, counter_constants)
{
    // Test the construction of counter constants and their evaluation
    std::unordered_map<std::string, Metric> metrics = {
        {"MAX_WAVE_SIZE", Metric("gfx9", "MAX_WAVE_SIZE", "a", "a", "a", "wave_front_size", "", 0)},
        {"SE_NUM",
         Metric("gfx9", "SE_NUM", "b", "b", "b", "array_count/simd_arrays_per_engine", "", 4)},
        {"SIMD_NUM", Metric("gfx9", "SIMD_NUM", "C", "C", "C", "simd_per_cu/CU_NUM", "", 2)},
        {"CU_NUM",
         Metric("gfx9", "CU_NUM", "D", "D", "D", "cu_per_simd_array*array_count", "", 5)}};
    add_constants(metrics, 6);
    std::unordered_map<std::string, std::unordered_map<std::string, EvaluateAST>> asts;

    // Check counter constants can be loaded
    for(const auto& [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast) << metric.expression() << " " << metric.name();
        asts.emplace("gfx9", std::unordered_map<std::string, EvaluateAST>{})
            .first->second.emplace(val,
                                   EvaluateAST({.handle = metric.id()}, metrics, *ast, "gfx9"));
        yy_delete_buffer(buf);
        delete ast;
    }

    // Set some random values for test data
    auto test_data                   = rocprofiler_agent_t{};
    test_data.wave_front_size        = 32;
    test_data.array_count            = 8;
    test_data.simd_arrays_per_engine = 5;
    test_data.simd_per_cu            = 104;
    test_data.cu_per_simd_array      = 156;

    // Check that required counters is calculated correctly
    std::unordered_map<std::string, std::set<std::string>> required_counters = {
        {"MAX_WAVE_SIZE", {"wave_front_size"}},
        {"SE_NUM", {"array_count", "simd_arrays_per_engine"}},
        {"SIMD_NUM", {"simd_per_cu", "cu_per_simd_array", "array_count"}},
        {"CU_NUM", {"cu_per_simd_array", "array_count"}},
    };

    // Check that the values are being read from agent_t correctly
    std::unordered_map<std::string, double> raw_agent_values = {
        {"wave_front_size", 32},
        {"array_count", 8},
        {"simd_arrays_per_engine", 5},
        {"simd_per_cu", 104},
        {"cu_per_simd_array", 156},
    };

    // Check that the evaluation of the special counters is correct
    std::unordered_map<std::string, double> final_computed_values = {
        {"MAX_WAVE_SIZE", 32},
        {"SE_NUM", 8.0 / 5.0},
        {"SIMD_NUM", 104.0 / (156.0 * 8.0)},
        {"CU_NUM", 156 * 8},
    };

    for(const auto& [name, expected] : required_counters)
    {
        auto eval_counters =
            rocprofiler::counters::get_required_hardware_counters(asts, "gfx9", metrics[name]);
        LOG(INFO) << name;
        ASSERT_TRUE(eval_counters);
        EXPECT_EQ(eval_counters->size(), expected.size());
        for(const auto& c : *eval_counters)
        {
            EXPECT_NE(expected.find(c.name()), expected.end());
            EXPECT_TRUE(!c.special().empty());
        }

        // Check that special counters are being decoded properly by the AST
        std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>> decode;
        EvaluateAST::read_special_counters(test_data, *eval_counters, decode);
        for(const auto& c : *eval_counters)
        {
            auto* ptr = rocprofiler::common::get_val(decode, c.id());
            ASSERT_TRUE(ptr) << c.name();
            // Check that the value matches agent_t and that its dim/id is set correctly
            ASSERT_EQ(ptr->size(), 1);
            EXPECT_EQ(ptr->at(0).counter_value, raw_agent_values[c.name()]);
            EXPECT_EQ(rocprofiler::counters::rec_to_counter_id(ptr->at(0).id).handle, c.id());
            EXPECT_EQ(rocprofiler::counters::rec_to_dim_pos(
                          ptr->at(0).id,
                          rocprofiler::counters::rocprofiler_profile_counter_instance_types::
                              ROCPROFILER_DIMENSION_NONE),
                      0);
        }
        asts.at("gfx9").at(name).expand_derived(asts.at("gfx9"));
        std::vector<std::unique_ptr<std::vector<rocprofiler_record_counter_t>>> cache;
        auto ret = asts.at("gfx9").at(name).evaluate(decode, cache);
        EXPECT_EQ(ret->size(), 1);
        EXPECT_FLOAT_EQ(ret->at(0).counter_value, final_computed_values[name]);

        asts.at("gfx9").at(name).set_out_id(*ret);
        EXPECT_EQ(rocprofiler::counters::rec_to_counter_id(ret->at(0).id).handle,
                  metrics[name].id());
        EXPECT_EQ(rocprofiler::counters::rec_to_dim_pos(
                      ret->at(0).id,
                      rocprofiler::counters::rocprofiler_profile_counter_instance_types::
                          ROCPROFILER_DIMENSION_NONE),
                  0);
    }
}

namespace
{
std::vector<rocprofiler_record_counter_t>
construct_test_data_dim(
    rocprofiler_counter_instance_id_t                                              base_id,
    std::vector<rocprofiler::counters::rocprofiler_profile_counter_instance_types> dims,
    size_t                                                                         dim_size)
{
    if(dims.empty()) return {};
    std::vector<rocprofiler_record_counter_t> ret;
    auto                                      my_dim = dims.back();
    dims.pop_back();
    for(size_t i = 0; i < dim_size; i++)
    {
        auto& record = ret.emplace_back();
        record.id    = base_id;
        rocprofiler::counters::set_dim_in_rec(record.id, my_dim, i);
        record.counter_value =
            static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / 50000)) + 1.0;
        auto recursive_dim = construct_test_data_dim(record.id, dims, dim_size);
        ret.insert(ret.end(), recursive_dim.begin(), recursive_dim.end());
    }
    return ret;
}

std::vector<rocprofiler_record_counter_t>
minus_vec(std::vector<rocprofiler_record_counter_t>        a,
          const std::vector<rocprofiler_record_counter_t>& b)
{
    for(size_t i = 0; i < a.size(); i++)
    {
        a[i].counter_value -= b[i].counter_value;
    }
    return a;
};

std::vector<rocprofiler_record_counter_t>
plus_vec(std::vector<rocprofiler_record_counter_t>        a,
         const std::vector<rocprofiler_record_counter_t>& b)
{
    for(size_t i = 0; i < a.size(); i++)
    {
        a[i].counter_value += b[i].counter_value;
    }
    return a;
};

std::vector<rocprofiler_record_counter_t>
times_vec(std::vector<rocprofiler_record_counter_t>        a,
          const std::vector<rocprofiler_record_counter_t>& b)
{
    for(size_t i = 0; i < a.size(); i++)
    {
        a[i].counter_value *= b[i].counter_value;
    }
    return a;
};

std::vector<rocprofiler_record_counter_t>
divide_vec(std::vector<rocprofiler_record_counter_t>        a,
           const std::vector<rocprofiler_record_counter_t>& b)
{
    for(size_t i = 0; i < a.size(); i++)
    {
        a[i].counter_value /= b[i].counter_value;
    }
    return a;
};

}  // namespace

TEST(evaluate_ast, evaluate_simple_counters)
{
    using namespace rocprofiler::counters;

    auto get_base_rec_id = [](uint64_t counter_id) {
        rocprofiler_counter_instance_id_t base_id = 0;
        set_counter_in_rec(base_id, {.handle = counter_id});
        return base_id;
    };

    std::unordered_map<std::string, Metric> metrics = {
        {"VOORHEES", Metric("gfx9", "VOORHEES", "a", "a", "a", "", "", 0)},
        {"KRUEGER", Metric("gfx9", "KRUEGER", "a", "a", "a", "", "", 1)},
        {"MYERS", Metric("gfx9", "MYERS", "a", "a", "a", "", "", 2)},
        {"BATES", Metric("gfx9", "BATES", "a", "a", "a", "VOORHEES+KRUEGER", "", 3)},
        {"KRAMER", Metric("gfx9", "KRAMER", "a", "a", "a", "MYERS*BATES", "", 4)},
        {"TORRANCE", Metric("gfx9", "TORRANCE", "a", "a", "a", "KRAMER/KRUEGER", "", 5)},
        {"GHOSTFACE",
         Metric("gfx9", "GHOSTFACE", "a", "a", "a", "VOORHEES-(KRUEGER+MYERS)", "", 6)}};

    std::unordered_map<std::string, std::vector<rocprofiler_record_counter_t>> base_counter_data = {
        {"VOORHEES", construct_test_data_dim(get_base_rec_id(0), {ROCPROFILER_DIMENSION_NONE}, 8)},
        {"KRUEGER", construct_test_data_dim(get_base_rec_id(1), {ROCPROFILER_DIMENSION_NONE}, 8)},
        {"MYERS", construct_test_data_dim(get_base_rec_id(2), {ROCPROFILER_DIMENSION_NONE}, 8)},
    };

    std::unordered_map<std::string, std::unordered_map<std::string, EvaluateAST>> asts;
    for(const auto& [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast) << metric.expression() << " " << metric.name();
        asts.emplace("gfx9", std::unordered_map<std::string, EvaluateAST>{})
            .first->second.emplace(val,
                                   EvaluateAST({.handle = metric.id()}, metrics, *ast, "gfx9"));
        yy_delete_buffer(buf);
        delete ast;
    }

    // Check base counter evaluation
    for(const auto& [name, expected] : base_counter_data)
    {
        auto eval_counters =
            rocprofiler::counters::get_required_hardware_counters(asts, "gfx9", metrics[name]);
        ASSERT_TRUE(eval_counters);
        ASSERT_EQ(eval_counters->size(), 1);
        EXPECT_EQ(eval_counters->begin()->name(), name);
        EXPECT_TRUE(eval_counters->begin()->special().empty());
        std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>> decode = {
            {metrics[name].id(), expected}};
        std::vector<std::unique_ptr<std::vector<rocprofiler_record_counter_t>>> cache;
        auto ret = asts.at("gfx9").at(name).evaluate(decode, cache);
        EXPECT_EQ(ret->size(), expected.size());
        int pos = 0;
        for(const auto& v : *ret)
        {
            EXPECT_EQ(v.id, expected[pos].id);
            EXPECT_EQ(v.counter_value, expected[pos].counter_value);
            pos++;
        }
    }

    // Check derived counter evaluation
    std::vector<std::tuple<std::string, std::vector<rocprofiler_record_counter_t>, int64_t>>
        derived_counters = {
            {"BATES", plus_vec(base_counter_data["VOORHEES"], base_counter_data["KRUEGER"]), 2},
            {"KRAMER",
             times_vec(base_counter_data["MYERS"],
                       plus_vec(base_counter_data["VOORHEES"], base_counter_data["KRUEGER"])),
             3},
            {"TORRANCE",
             divide_vec(
                 times_vec(base_counter_data["MYERS"],
                           plus_vec(base_counter_data["VOORHEES"], base_counter_data["KRUEGER"])),
                 base_counter_data["KRUEGER"]),
             3},
            {"GHOSTFACE",
             minus_vec(base_counter_data["VOORHEES"],
                       plus_vec(base_counter_data["KRUEGER"], base_counter_data["MYERS"])),
             3},
        };

    std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>> base_counter_decode;
    for(const auto& [name, base_counter_v] : base_counter_data)
    {
        base_counter_decode[metrics[name].id()] = base_counter_v;
    }

    for(auto& [name, expected, eval_count] : derived_counters)
    {
        LOG(INFO) << name;
        auto eval_counters =
            rocprofiler::counters::get_required_hardware_counters(asts, "gfx9", metrics[name]);
        ASSERT_TRUE(eval_counters);
        ASSERT_EQ(eval_counters->size(), eval_count);
        std::vector<std::unique_ptr<std::vector<rocprofiler_record_counter_t>>> cache;
        asts.at("gfx9").at(name).expand_derived(asts.at("gfx9"));
        auto ret = asts.at("gfx9").at(name).evaluate(base_counter_decode, cache);
        EXPECT_EQ(ret->size(), expected.size());
        int pos = 0;
        asts.at("gfx9").at(name).set_out_id(*ret);
        for(const auto& v : *ret)
        {
            set_counter_in_rec(expected[pos].id, {.handle = metrics[name].id()});
            EXPECT_EQ(v.id, expected[pos].id);
            EXPECT_FLOAT_EQ(v.counter_value, expected[pos].counter_value);
            pos++;
        }
    }
}

namespace
{
void
run_reduce_test(
    std::unordered_map<std::string, Metric>&                                    metrics,
    std::unordered_map<std::string, std::vector<rocprofiler_record_counter_t>>& base_counter_data,
    std::vector<std::tuple<std::string, std::vector<rocprofiler_record_counter_t>, int64_t>>&
        derived_counters)
{
    std::unordered_map<std::string, std::unordered_map<std::string, EvaluateAST>> asts;
    for(const auto& [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast) << metric.expression() << " " << metric.name();
        asts.emplace("gfx9", std::unordered_map<std::string, EvaluateAST>{})
            .first->second.emplace(val,
                                   EvaluateAST({.handle = metric.id()}, metrics, *ast, "gfx9"));
        yy_delete_buffer(buf);
        delete ast;
    }

    std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>> base_counter_decode;
    for(const auto& [name, base_counter_v] : base_counter_data)
    {
        base_counter_decode[metrics[name].id()] = base_counter_v;
    }

    for(auto& [name, expected, eval_count] : derived_counters)
    {
        LOG(INFO) << name;
        auto eval_counters =
            rocprofiler::counters::get_required_hardware_counters(asts, "gfx9", metrics[name]);
        ASSERT_TRUE(eval_counters);
        ASSERT_EQ(eval_counters->size(), eval_count);
        std::vector<std::unique_ptr<std::vector<rocprofiler_record_counter_t>>> cache;
        asts.at("gfx9").at(name).expand_derived(asts.at("gfx9"));
        auto ret = asts.at("gfx9").at(name).evaluate(base_counter_decode, cache);
        EXPECT_EQ(ret->size(), expected.size());
        ASSERT_EQ(expected.size(), 1);
        int pos = 0;
        asts.at("gfx9").at(name).set_out_id(*ret);
        for(const auto& v : *ret)
        {
            set_counter_in_rec(expected[pos].id, {.handle = metrics[name].id()});
            EXPECT_EQ(v.id, expected[pos].id);
            EXPECT_FLOAT_EQ(v.counter_value, expected[pos].counter_value);
            pos++;
        }
    }
}
};  // namespace

TEST(evaluate_ast, counter_reduction_sum)
{
    using namespace rocprofiler::counters;

    auto get_base_rec_id = [](uint64_t counter_id) {
        rocprofiler_counter_instance_id_t base_id = 0;
        set_counter_in_rec(base_id, {.handle = counter_id});
        return base_id;
    };

    auto sum_vec = [](auto& a) -> auto&
    {
        for(size_t i = 1; i < a.size(); i++)
        {
            a[0].counter_value += a[i].counter_value;
        }
        a.resize(1);
        CHECK(a.size() == 1);
        return a;
    };

    std::unordered_map<std::string, Metric> metrics = {
        {"VOORHEES", Metric("gfx9", "VOORHEES", "a", "a", "a", "", "", 0)},
        {"KRUEGER", Metric("gfx9", "KRUEGER", "a", "a", "a", "", "", 1)},
        {"MYERS", Metric("gfx9", "MYERS", "a", "a", "a", "", "", 2)},
        {"MYERS_REDUCED",
         Metric("gfx9", "MYERS_REDUCED", "a", "a", "a", "reduce(MYERS,sum)", "", 3)},
        {"BATES",
         Metric(
             "gfx9", "BATES", "a", "a", "a", "reduce(VOORHEES, sum)+reduce(KRUEGER, sum)", "", 4)},
        {"KRAMER",
         Metric("gfx9",
                "KRAMER",
                "a",
                "a",
                "a",
                "5*reduce(VOORHEES, sum)+reduce(KRUEGER, sum)",
                "",
                5)},
        {"GHOSTFACE",
         Metric("gfx9",
                "GHOSTFACE",
                "a",
                "a",
                "a",
                "reduce(VOORHEES, sum)+(reduce(KRUEGER, sum)/5)",
                "",
                6)},
    };

    std::unordered_map<std::string, std::vector<rocprofiler_record_counter_t>> base_counter_data = {
        {"VOORHEES", construct_test_data_dim(get_base_rec_id(0), {ROCPROFILER_DIMENSION_NONE}, 8)},
        {"KRUEGER", construct_test_data_dim(get_base_rec_id(1), {ROCPROFILER_DIMENSION_NONE}, 8)},
        {"MYERS", construct_test_data_dim(get_base_rec_id(2), {ROCPROFILER_DIMENSION_NONE}, 8)},
    };

    // Check derived counter evaluation
    std::vector<std::tuple<std::string, std::vector<rocprofiler_record_counter_t>, int64_t>>
        derived_counters = {
            {"MYERS_REDUCED", sum_vec(base_counter_data["MYERS"]), 1},
            {"BATES",
             plus_vec(sum_vec(base_counter_data["VOORHEES"]),
                      sum_vec(base_counter_data["KRUEGER"])),
             2},
            {"KRAMER",
             plus_vec(times_vec(std::vector<rocprofiler_record_counter_t>{{.id            = 0,
                                                                           .counter_value = 5.0}},
                                sum_vec(base_counter_data["VOORHEES"])),
                      sum_vec(base_counter_data["KRUEGER"])),
             2},
            {"GHOSTFACE",
             plus_vec(sum_vec(base_counter_data["VOORHEES"]),
                      divide_vec(sum_vec(base_counter_data["KRUEGER"]),
                                 std::vector<rocprofiler_record_counter_t>{
                                     {.id = 0, .counter_value = 5.0}})),
             2},
        };

    run_reduce_test(metrics, base_counter_data, derived_counters);
}

TEST(evaluate_ast, counter_reduction_min)
{
    using namespace rocprofiler::counters;

    auto get_base_rec_id = [](uint64_t counter_id) {
        rocprofiler_counter_instance_id_t base_id = 0;
        set_counter_in_rec(base_id, {.handle = counter_id});
        return base_id;
    };

    auto min_vec = [](auto& a) -> auto&
    {
        a[0].counter_value = std::min_element(a.begin(), a.end(), [](const auto& b, const auto& c) {
                                 return b.counter_value < c.counter_value;
                             })->counter_value;
        a.resize(1);
        CHECK(a.size() == 1);
        return a;
    };

    std::unordered_map<std::string, Metric> metrics = {
        {"VOORHEES", Metric("gfx9", "VOORHEES", "a", "a", "a", "", "", 0)},
        {"KRUEGER", Metric("gfx9", "KRUEGER", "a", "a", "a", "", "", 1)},
        {"MYERS", Metric("gfx9", "MYERS", "a", "a", "a", "", "", 2)},
        {"MYERS_REDUCED",
         Metric("gfx9", "MYERS_REDUCED", "a", "a", "a", "reduce(MYERS,min)", "", 3)},
        {"BATES",
         Metric(
             "gfx9", "BATES", "a", "a", "a", "reduce(VOORHEES, min)+reduce(KRUEGER, min)", "", 4)},
        {"KRAMER",
         Metric("gfx9",
                "KRAMER",
                "a",
                "a",
                "a",
                "5*reduce(VOORHEES, min)+reduce(KRUEGER, min)",
                "",
                5)},
        {"GHOSTFACE",
         Metric("gfx9",
                "GHOSTFACE",
                "a",
                "a",
                "a",
                "reduce(VOORHEES, min)+(reduce(KRUEGER, min)/5)",
                "",
                6)},
    };

    std::unordered_map<std::string, std::vector<rocprofiler_record_counter_t>> base_counter_data = {
        {"VOORHEES", construct_test_data_dim(get_base_rec_id(0), {ROCPROFILER_DIMENSION_NONE}, 8)},
        {"KRUEGER", construct_test_data_dim(get_base_rec_id(1), {ROCPROFILER_DIMENSION_NONE}, 8)},
        {"MYERS", construct_test_data_dim(get_base_rec_id(2), {ROCPROFILER_DIMENSION_NONE}, 8)},
    };

    // Check derived counter evaluation
    std::vector<std::tuple<std::string, std::vector<rocprofiler_record_counter_t>, int64_t>>
        derived_counters = {
            {"MYERS_REDUCED", min_vec(base_counter_data["MYERS"]), 1},
            {"BATES",
             plus_vec(min_vec(base_counter_data["VOORHEES"]),
                      min_vec(base_counter_data["KRUEGER"])),
             2},
            {"KRAMER",
             plus_vec(times_vec(std::vector<rocprofiler_record_counter_t>{{.id            = 0,
                                                                           .counter_value = 5.0}},
                                min_vec(base_counter_data["VOORHEES"])),
                      min_vec(base_counter_data["KRUEGER"])),
             2},
            {"GHOSTFACE",
             plus_vec(min_vec(base_counter_data["VOORHEES"]),
                      divide_vec(min_vec(base_counter_data["KRUEGER"]),
                                 std::vector<rocprofiler_record_counter_t>{
                                     {.id = 0, .counter_value = 5.0}})),
             2},
        };

    run_reduce_test(metrics, base_counter_data, derived_counters);
}

TEST(evaluate_ast, counter_reduction_max)
{
    using namespace rocprofiler::counters;

    auto get_base_rec_id = [](uint64_t counter_id) {
        rocprofiler_counter_instance_id_t base_id = 0;
        set_counter_in_rec(base_id, {.handle = counter_id});
        return base_id;
    };

    auto max_vec = [](auto& a) -> auto&
    {
        a[0].counter_value = std::max_element(a.begin(), a.end(), [](const auto& b, const auto& c) {
                                 return b.counter_value < c.counter_value;
                             })->counter_value;
        a.resize(1);
        CHECK(a.size() == 1);
        return a;
    };

    std::unordered_map<std::string, Metric> metrics = {
        {"VOORHEES", Metric("gfx9", "VOORHEES", "a", "a", "a", "", "", 0)},
        {"KRUEGER", Metric("gfx9", "KRUEGER", "a", "a", "a", "", "", 1)},
        {"MYERS", Metric("gfx9", "MYERS", "a", "a", "a", "", "", 2)},
        {"MYERS_REDUCED",
         Metric("gfx9", "MYERS_REDUCED", "a", "a", "a", "reduce(MYERS,max)", "", 3)},
        {"BATES",
         Metric(
             "gfx9", "BATES", "a", "a", "a", "reduce(VOORHEES, max)+reduce(KRUEGER, max)", "", 4)},
        {"KRAMER",
         Metric("gfx9",
                "KRAMER",
                "a",
                "a",
                "a",
                "5*reduce(VOORHEES, max)+reduce(KRUEGER, max)",
                "",
                5)},
        {"GHOSTFACE",
         Metric("gfx9",
                "GHOSTFACE",
                "a",
                "a",
                "a",
                "reduce(VOORHEES, max)+(reduce(KRUEGER, max)/5)",
                "",
                6)},
    };

    std::unordered_map<std::string, std::vector<rocprofiler_record_counter_t>> base_counter_data = {
        {"VOORHEES", construct_test_data_dim(get_base_rec_id(0), {ROCPROFILER_DIMENSION_NONE}, 8)},
        {"KRUEGER", construct_test_data_dim(get_base_rec_id(1), {ROCPROFILER_DIMENSION_NONE}, 8)},
        {"MYERS", construct_test_data_dim(get_base_rec_id(2), {ROCPROFILER_DIMENSION_NONE}, 8)},
    };

    // Check derived counter evaluation
    std::vector<std::tuple<std::string, std::vector<rocprofiler_record_counter_t>, int64_t>>
        derived_counters = {
            {"MYERS_REDUCED", max_vec(base_counter_data["MYERS"]), 1},
            {"BATES",
             plus_vec(max_vec(base_counter_data["VOORHEES"]),
                      max_vec(base_counter_data["KRUEGER"])),
             2},
            {"KRAMER",
             plus_vec(times_vec(std::vector<rocprofiler_record_counter_t>{{.id            = 0,
                                                                           .counter_value = 5.0}},
                                max_vec(base_counter_data["VOORHEES"])),
                      max_vec(base_counter_data["KRUEGER"])),
             2},
            {"GHOSTFACE",
             plus_vec(max_vec(base_counter_data["VOORHEES"]),
                      divide_vec(max_vec(base_counter_data["KRUEGER"]),
                                 std::vector<rocprofiler_record_counter_t>{
                                     {.id = 0, .counter_value = 5.0}})),
             2},
        };

    run_reduce_test(metrics, base_counter_data, derived_counters);
}

TEST(evaluate_ast, counter_reduction_avg)
{
    using namespace rocprofiler::counters;

    auto get_base_rec_id = [](uint64_t counter_id) {
        rocprofiler_counter_instance_id_t base_id = 0;
        set_counter_in_rec(base_id, {.handle = counter_id});
        return base_id;
    };

    auto avg_vec = [](auto& a) -> auto&
    {
        for(size_t i = 1; i < a.size(); i++)
        {
            a[0].counter_value += a[i].counter_value;
        }
        a[0].counter_value /= a.size();
        a.resize(1);
        CHECK(a.size() == 1);
        return a;
    };

    std::unordered_map<std::string, Metric> metrics = {
        {"VOORHEES", Metric("gfx9", "VOORHEES", "a", "a", "a", "", "", 0)},
        {"KRUEGER", Metric("gfx9", "KRUEGER", "a", "a", "a", "", "", 1)},
        {"MYERS", Metric("gfx9", "MYERS", "a", "a", "a", "", "", 2)},
        {"MYERS_REDUCED",
         Metric("gfx9", "MYERS_REDUCED", "a", "a", "a", "reduce(MYERS, avr)", "", 3)},
        {"BATES",
         Metric(
             "gfx9", "BATES", "a", "a", "a", "reduce(VOORHEES, avr)+reduce(KRUEGER, avr)", "", 4)},
        {"KRAMER",
         Metric("gfx9",
                "KRAMER",
                "a",
                "a",
                "a",
                "5*reduce(VOORHEES, avr)+reduce(KRUEGER, avr)",
                "",
                5)},
        {"GHOSTFACE",
         Metric("gfx9",
                "GHOSTFACE",
                "a",
                "a",
                "a",
                "reduce(VOORHEES, avr)+(reduce(KRUEGER, avr)/5)",
                "",
                6)},
    };

    std::unordered_map<std::string, std::vector<rocprofiler_record_counter_t>> base_counter_data = {
        {"VOORHEES", construct_test_data_dim(get_base_rec_id(0), {ROCPROFILER_DIMENSION_NONE}, 8)},
        {"KRUEGER", construct_test_data_dim(get_base_rec_id(1), {ROCPROFILER_DIMENSION_NONE}, 8)},
        {"MYERS", construct_test_data_dim(get_base_rec_id(2), {ROCPROFILER_DIMENSION_NONE}, 8)},
    };

    // Check derived counter evaluation
    std::vector<std::tuple<std::string, std::vector<rocprofiler_record_counter_t>, int64_t>>
        derived_counters = {
            {"MYERS_REDUCED", avg_vec(base_counter_data["MYERS"]), 1},
            {"BATES",
             plus_vec(avg_vec(base_counter_data["VOORHEES"]),
                      avg_vec(base_counter_data["KRUEGER"])),
             2},
            {"KRAMER",
             plus_vec(times_vec(std::vector<rocprofiler_record_counter_t>{{.id            = 0,
                                                                           .counter_value = 5.0}},
                                avg_vec(base_counter_data["VOORHEES"])),
                      avg_vec(base_counter_data["KRUEGER"])),
             2},
            {"GHOSTFACE",
             plus_vec(avg_vec(base_counter_data["VOORHEES"]),
                      divide_vec(avg_vec(base_counter_data["KRUEGER"]),
                                 std::vector<rocprofiler_record_counter_t>{
                                     {.id = 0, .counter_value = 5.0}})),
             2},
        };

    run_reduce_test(metrics, base_counter_data, derived_counters);
}

TEST(evaluate_ast, evaluate_mixed_counters)
{
    using namespace rocprofiler::counters;

    auto get_base_rec_id = [](uint64_t counter_id) {
        rocprofiler_counter_instance_id_t base_id = 0;
        set_counter_in_rec(base_id, {.handle = counter_id});
        return base_id;
    };

    auto sum_vec = [](auto& a) -> auto&
    {
        for(size_t i = 1; i < a.size(); i++)
        {
            a[0].counter_value += a[i].counter_value;
        }
        a.resize(1);
        CHECK(a.size() == 1);
        return a;
    };

    // Set some random values for test data
    auto test_data                   = rocprofiler_agent_t{};
    test_data.wave_front_size        = 32;
    test_data.array_count            = 8;
    test_data.simd_arrays_per_engine = 5;
    test_data.simd_per_cu            = 104;
    test_data.cu_per_simd_array      = 156;

    std::unordered_map<std::string, Metric> metrics = {
        {"MAX_WAVE_SIZE", Metric("gfx9", "MAX_WAVE_SIZE", "a", "a", "a", "wave_front_size", "", 0)},
        {"SE_NUM",
         Metric("gfx9", "SE_NUM", "b", "b", "b", "array_count/simd_arrays_per_engine", "", 1)},
        {"CU_NUM", Metric("gfx9", "CU_NUM", "D", "D", "D", "cu_per_simd_array*array_count", "", 2)},
        {"SIMD_NUM", Metric("gfx9", "SIMD_NUM", "C", "C", "C", "simd_per_cu/CU_NUM", "", 3)},
        {"VOORHEES", Metric("gfx9", "VOORHEES", "a", "a", "a", "", "", 4)},
        {"KRUEGER", Metric("gfx9", "KRUEGER", "a", "a", "a", "", "", 5)},
        {"BATES",
         Metric("gfx9", "BATES", "a", "a", "a", "MAX_WAVE_SIZE*reduce(VOORHEES,sum)", "", 6)},
        {"KRAMER", Metric("gfx9", "KRAMER", "a", "a", "a", "reduce(KRUEGER,sum)*SE_NUM", "", 7)},
        {"TORRANCE",
         Metric("gfx9", "TORRANCE", "a", "a", "a", "reduce(KRUEGER,sum)*SIMD_NUM", "", 8)}};
    add_constants(metrics, 9);

    std::unordered_map<std::string, std::vector<rocprofiler_record_counter_t>> base_counter_data = {
        {"VOORHEES", construct_test_data_dim(get_base_rec_id(0), {ROCPROFILER_DIMENSION_NONE}, 8)},
        {"KRUEGER", construct_test_data_dim(get_base_rec_id(1), {ROCPROFILER_DIMENSION_NONE}, 8)},
    };

    std::vector<std::tuple<std::string, std::vector<rocprofiler_record_counter_t>, int64_t>>
        derived_counters = {
            {"BATES",
             times_vec(std::vector<rocprofiler_record_counter_t>{{.id = 0, .counter_value = 32}},
                       sum_vec(base_counter_data["VOORHEES"])),
             2},
            {"KRAMER",
             times_vec(
                 sum_vec(base_counter_data["KRUEGER"]),
                 std::vector<rocprofiler_record_counter_t>{{.id = 0, .counter_value = 8.0 / 5.0}}),
             3},
            {"TORRANCE",
             times_vec(sum_vec(base_counter_data["KRUEGER"]),
                       std::vector<rocprofiler_record_counter_t>{
                           {.id = 0, .counter_value = 104.0 / (156.0 * 8.0)}}),
             4},
        };

    std::unordered_map<std::string, std::unordered_map<std::string, EvaluateAST>> asts;
    for(const auto& [val, metric] : metrics)
    {
        RawAST* ast = nullptr;
        auto    buf = yy_scan_string(metric.expression().empty() ? metric.name().c_str()
                                                                 : metric.expression().c_str());
        yyparse(&ast);
        ASSERT_TRUE(ast) << metric.expression() << " " << metric.name();
        asts.emplace("gfx9", std::unordered_map<std::string, EvaluateAST>{})
            .first->second.emplace(val,
                                   EvaluateAST({.handle = metric.id()}, metrics, *ast, "gfx9"));
        yy_delete_buffer(buf);
        delete ast;
    }

    std::unordered_map<uint64_t, std::vector<rocprofiler_record_counter_t>> base_counter_decode;
    for(const auto& special_counter : {"MAX_WAVE_SIZE", "SE_NUM", "SIMD_NUM"})
    {
        auto eval_counters = rocprofiler::counters::get_required_hardware_counters(
            asts, "gfx9", metrics[special_counter]);
        EvaluateAST::read_special_counters(test_data, *eval_counters, base_counter_decode);
    }

    for(const auto& [name, base_counter_v] : base_counter_data)
    {
        base_counter_decode[metrics[name].id()] = base_counter_v;
    }

    for(auto& [name, expected, eval_count] : derived_counters)
    {
        LOG(INFO) << name;
        auto eval_counters =
            rocprofiler::counters::get_required_hardware_counters(asts, "gfx9", metrics[name]);
        ASSERT_TRUE(eval_counters);
        ASSERT_EQ(eval_counters->size(), eval_count);
        std::vector<std::unique_ptr<std::vector<rocprofiler_record_counter_t>>> cache;
        asts.at("gfx9").at(name).expand_derived(asts.at("gfx9"));
        auto ret = asts.at("gfx9").at(name).evaluate(base_counter_decode, cache);
        EXPECT_EQ(ret->size(), expected.size());
        ASSERT_EQ(expected.size(), 1);
        int pos = 0;
        asts.at("gfx9").at(name).set_out_id(*ret);
        for(const auto& v : *ret)
        {
            set_counter_in_rec(expected[pos].id, {.handle = metrics[name].id()});
            EXPECT_EQ(v.id, expected[pos].id);
            EXPECT_FLOAT_EQ(v.counter_value, expected[pos].counter_value);
            pos++;
        }
    }
}
