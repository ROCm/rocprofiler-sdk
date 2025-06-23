# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
import re


def test_perfetto_data(
    pftrace_data,
    json_data,
    categories=(
        "hip",
        "hsa",
        "marker",
        "kernel",
        "memory_copy",
        "memory_allocation",
        "rocdecode_api",
        "rocjpeg_api",
        "counter_collection",
        "scratch_memory",
    ),
):

    mapping = {
        "hip": ("hip_api", "hip_api"),
        "hsa": ("hsa_api", "hsa_api"),
        "marker": ("marker_api", "marker_api"),
        "kernel": ("kernel_dispatch", "kernel_dispatch"),
        "memory_copy": ("memory_copy", "memory_copy"),
        "memory_allocation": ("memory_allocation", "memory_allocation"),
        "rocdecode_api": ("rocdecode_api", "rocdecode_api"),
        "rocjpeg_api": ("rocjpeg_api", "rocjpeg_api"),
        "counter_collection": ("counter_collection", "counter_collection"),
        "scratch_memory": ("scratch_memory", "scratch_memory"),
    }

    # make sure they specified valid categories
    for itr in categories:
        assert itr in mapping.keys()

    for pf_category, js_category in [
        itr for key, itr in mapping.items() if key in categories
    ]:
        _pf_data = pftrace_data.loc[pftrace_data["category"] == pf_category]

        _js_data = []
        if js_category != "counter_collection":
            _js_data = json_data["rocprofiler-sdk-tool"]["buffer_records"][js_category]
        else:
            unique_counter_ids = set()

            counter_info = {}
            for itr in json_data["rocprofiler-sdk-tool"]["counters"]:
                counter_info[itr["id"]["handle"]] = itr

            agent_info = {}
            for itr in json_data["rocprofiler-sdk-tool"]["agents"]:
                agent_info[itr["id"]["handle"]] = itr

            for dispatch_entry in json_data["rocprofiler-sdk-tool"]["callback_records"][
                js_category
            ]:
                counter_records = dispatch_entry["records"]
                agent_id = dispatch_entry["dispatch_data"]["dispatch_info"]["agent_id"][
                    "handle"
                ]
                agent_node_id = agent_info[agent_id]["node_id"]

                for record in counter_records:
                    counter_id = record["counter_id"]["handle"]
                    counter_name = counter_info[counter_id]["name"]
                    unique_counter_ids.add(f"Agent [{agent_node_id}] PMC {counter_name}")

            _js_data = [{"counter_id": id} for id in unique_counter_ids]

        assert len(_pf_data) == len(
            _js_data
        ), f"{pf_category} ({len(_pf_data)}):\n\t{_pf_data}\n{js_category} ({len(_js_data)}):\n\t{_js_data}"


def test_otf2_data(
    otf2_data,
    json_data,
    categories=("hip", "hsa", "kernel", "memory_allocation", "memory_copy"),
):
    def get_operation_name(kind_id, op_id):
        return json_data["rocprofiler-sdk-tool"]["strings"]["buffer_records"][kind_id][
            "operations"
        ][op_id]

    def get_kind_name(kind_id):
        return json_data["rocprofiler-sdk-tool"]["strings"]["buffer_records"][kind_id][
            "kind"
        ]

    mapping = {
        "hip": ("hip_api", "hip_api"),
        "hsa": ("hsa_api", "hsa_api"),
        "marker": ("marker_api", "marker_api"),
        "kernel": ("kernel_dispatch", "kernel_dispatch"),
        "memory_copy": ("memory_copy", "memory_copy"),
        "memory_allocation": ("memory_allocation", "memory_allocation"),
        "rocdecode_api": ("rocdecode_api", "rocdecode_api"),
        "rocjpeg_api": ("rocjpeg_api", "rocjpeg_api"),
    }

    # make sure they specified valid categories
    for itr in categories:
        assert itr in mapping.keys()

    for otf2_category, json_category in [
        itr for key, itr in mapping.items() if key in categories
    ]:
        _otf2_data = otf2_data.loc[otf2_data["category"] == otf2_category]
        _json_data = json_data["rocprofiler-sdk-tool"]["buffer_records"][json_category]

        # we do not encode the roctxMark "regions" in OTF2 because
        # they don't map to the OTF2_REGION_ROLE_FUNCTION well
        if json_category == "marker_api":

            def roctx_mark_filter(val):
                return (
                    None
                    if get_kind_name(val.kind)
                    in ["MARKER_CORE_API", "MARKER_CORE_RANGE_API"]
                    and get_operation_name(val.kind, val.operation) == "roctxMarkA"
                    else val
                )

            _json_data = [itr for itr in _json_data if roctx_mark_filter(itr) is not None]

        if json_category == "memory_allocation":
            _json_data_alloc = [itr for itr in _json_data if itr["operation"] == 1]
            _json_data_free = [itr for itr in _json_data if itr["operation"] == 3]
            _otf2_data_alloc = _otf2_data.loc[
                _otf2_data["name"] == "MEMORY_ALLOCATION_ALLOCATE"
            ]
            _otf2_data_free = _otf2_data.loc[
                _otf2_data["name"] == "MEMORY_ALLOCATION_FREE"
            ]
            assert len(_otf2_data_alloc) == len(
                _json_data_alloc
            ), f"memory_allocation allocate otf2 ({len(_otf2_data_alloc)}):\n\t{_otf2_data_alloc} \n json ({len(_json_data_alloc)}):\n\t{_json_data_alloc}"
            assert len(_otf2_data_free) == len(
                _json_data_free
            ), f"memory_allocation free otf2 ({len(_otf2_data_free)}):\n\t{_otf2_data_free} \n json ({len(_json_data_free)}):\n\t{_json_data_free}"

            # problematic timing of free, not seen on visualisation
            # select the first thread where free is reported, order by start time json and otf2 2 data for this thread and compare time stamps one by one
            # testing max two threads
            max_thr = 2
            for ind_thr in range(max_thr):
                if len(_otf2_data_free) > ind_thr:
                    loc_name = _otf2_data_free.iloc[ind_thr]["location"].name
                    thread_id = int(re.match(r"Thread (\d+)", loc_name).group(1))
                    # hope this is ordered by ts
                    _json_data_free_inthread = [
                        itr for itr in _json_data_free if itr["thread_id"] == thread_id
                    ]
                    _otf2_data_free_inthread = []
                    i = 0
                    for i in range(len(_otf2_data_free)):
                        _loc_name = _otf2_data_free.iloc[i]["location"].name
                        _thread_id = int(re.match(r"Thread (\d+)", _loc_name).group(1))
                        if _thread_id == thread_id:
                            _otf2_data_free_inthread.append(_otf2_data_free.iloc[i])
                    assert len(_otf2_data_free_inthread) == len(
                        _json_data_free_inthread
                    ), f"thread {thread_id}, memory_allocation free otf2 ({len(_otf2_data_free_inthread)}):\n\t{_otf2_data_free_inthread} \n json ({len(_json_data_free_inthread)}):\n\t{_json_data_free_inthread}"
                    i = 0
                    for i in range(len(_otf2_data_free_inthread)):
                        _otf2_s_ts = _otf2_data_free_inthread[i]["start_ts"]
                        _otf2_e_ts = _otf2_data_free_inthread[i]["end_ts"]
                        _json_s_ts = _json_data_free_inthread[i]["start_timestamp"]
                        _json_e_ts = _json_data_free_inthread[i]["end_timestamp"]
                        assert (
                            _otf2_s_ts == _json_s_ts
                        ), f"memory_allocation free otf2 record {i} start timestamp ({_otf2_s_ts}):\n\t{_otf2_data_free_inthread[i]} \n json start ts({_json_e_ts}):\n\t{_json_data_free_inthread[i]}"
                        assert (
                            _otf2_e_ts == _json_e_ts
                        ), f"memory_allocation free otf2 record {i} end timestamp ({_otf2_e_ts}):\n\t{_otf2_data_free_inthread[i]} \n json end ts({_json_e_ts}):\n\t{_json_data_free_inthread[i]}"

        assert len(_otf2_data) == len(
            _json_data
        ), f"{otf2_category} ({len(_otf2_data)}):\n\t{_otf2_data}\n{json_category} ({len(_json_data)}):\n\t{_json_data}"


def test_rocpd_data(
    rocpd_data,
    json_data,
    categories=(
        "hip",
        "hsa",
        "marker",
        "kernel",
        "memory_copy",
        "memory_allocation",
        "rocdecode_api",
        "rocjpeg_api",
    ),
):

    mapping = {
        "hip": (
            "hip_api",
            (
                "HIP_COMPILER_API",
                "HIP_COMPILER_API_EXT",
                "HIP_RUNTIME_API",
                "HIP_RUNTIME_API_EXT",
            ),
        ),
        "hsa": (
            "hsa_api",
            (
                "HSA_CORE_API",
                "HSA_AMD_EXT_API",
                "HSA_IMAGE_EXT_API",
                "HSA_FINALIZE_EXT_API",
            ),
        ),
        "marker": (
            "marker_api",
            (
                "MARKER_CORE_API",
                "MARKER_CONTROL_API",
                "MARKER_NAME_API",
                "MARKER_CORE_RANGE_API",
            ),
        ),
        "kernel": ("kernel_dispatch", ("KERNEL_DISPATCH")),
        "memory_copy": ("memory_copy", ("MEMORY_COPY")),
        "memory_allocation": ("memory_allocation", ("MEMORY_ALLOCATION")),
        "rocdecode_api": ("rocdecode_api", ("ROCDECODE_API")),
        "rocjpeg_api": ("rocjpeg_api", ("ROCJPEG_API")),
    }

    view_mapping = {
        "hip_api": "regions",
        "hsa_api": "regions",
        "marker_api": "regions_and_samples",
        "rccl_api": "regions",
        "rocdecode_api": "regions",
        "rocjpeg_api": "regions",
        "kernel_dispatch": "kernels",
        "memory_copy": "memory_copies",
        "memory_allocation": "memory_allocations",
    }

    # make sure they specified valid categories
    for itr in categories:
        assert itr in mapping.keys()

    for js_category, rpd_category in [
        itr for key, itr in mapping.items() if key in categories
    ]:
        _js_data = json_data["rocprofiler-sdk-tool"]["buffer_records"][js_category]
        _rpd_cats = (
            rpd_category if isinstance(rpd_category, (list, tuple)) else [rpd_category]
        )
        _rpd_cond = " OR ".join([f"category = '{itr}'" for itr in _rpd_cats])
        _rpd_query = f"SELECT * FROM {view_mapping[js_category]} WHERE {_rpd_cond}"
        _rpd_data = rocpd_data.execute(_rpd_query).fetchall()

        assert len(_rpd_data) == len(
            _js_data
        ), f"query: {_rpd_query}\n{rpd_category} ({len(_rpd_data)}):\n\t{_rpd_data}\n{js_category} ({len(_js_data)}):\n\t{_js_data}"
