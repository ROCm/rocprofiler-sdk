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

#include <stddef.h>
#include <stdint.h>

/**
 * @defgroup THREAD_TRACE Thread Trace Service
 * @brief ROCprof-trace-decoder defined types. All timestamp values are in shader clock units.
 *
 * @{
 */

/**
 * @brief Describes a timestamp in shader clock units.
 * TS==0 marks the start of the trace on a shader engine.
 * Different shader engines may have different frequencies or start points.
 */
typedef int64_t rocprofiler_shader_timestamp_t;

/**
 * @brief Describes the type of info received.
 */
typedef enum rocprofiler_thread_trace_decoder_info_t
{
    ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE = 0,
    ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST,
    ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE,
    ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST
} rocprofiler_thread_trace_decoder_info_t;

/**
 * @brief Describes a PC address.
 */
typedef struct rocprofiler_thread_trace_decoder_pc_t
{
    size_t addr;       ///< Memory address (marker_id == 0), or ELF vaddr (marker_id != 0).
    size_t marker_id;  ///< Code object load ID. Zero if no code object was found.
} rocprofiler_thread_trace_decoder_pc_t;

/**
 * @brief Describes four performance counter values.
 */
typedef struct rocprofiler_thread_trace_decoder_perfevent_t
{
    rocprofiler_shader_timestamp_t time;

    uint16_t event0;  //< Counter0 (bank==0) or Counter4 (bank==1).
    uint16_t event1;  //< Counter1 (bank==0) or Counter5 (bank==1).
    uint16_t event2;  //< Counter2 (bank==0) or Counter6 (bank==1).
    uint16_t event3;  //< Counter3 (bank==0) or Counter7 (bank==1).
    uint8_t  CU;      ///< Shader compute unit ID these counters were collected from.
    uint8_t  bank;    ///< Selects counter group [0,3] or [4,7]
} rocprofiler_thread_trace_decoder_perfevent_t;

typedef enum rocprofiler_thread_trace_decoder_occupancy_flags_t
{
    ROCPROFILER_THREAD_TRACE_DECODER_OCCUPANCY_FLAGS_WAVE_START = 1 << 0,  // set -> is wave start
} rocprofiler_thread_trace_decoder_occupancy_flags_t;

/**
 * @brief Describes an occupancy event (wave started or wave ended).
 */
typedef struct rocprofiler_thread_trace_decoder_occupancy_t
{
    rocprofiler_thread_trace_decoder_pc_t pc;    ///< Wave start address (kernel entry point)
    rocprofiler_shader_timestamp_t        time;  ///< Timestamp of event

    uint8_t  flags;  ///< See ::rocprofiler_thread_trace_decoder_occupancy_flags_t
    uint8_t  cu;     ///< Compute unit ID (gfx9) or WGP ID (gfx10+)
    uint8_t  simd;   ///< SIMD ID [0,3] within compute unit
    uint8_t  slot;   ///< Wave slot ID within SIMD
    uint32_t reserved;
} rocprofiler_thread_trace_decoder_occupancy_t;

/**
 * @brief Wave state type.
 */
typedef enum rocprofiler_thread_trace_decoder_wstate_type_t
{
    ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY = 0,
    ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE,
    ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC,
    ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT,
    ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL,
    ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST,
} rocprofiler_thread_trace_decoder_wstate_type_t;

/**
 * @brief A wave state change event.
 */
typedef struct rocprofiler_thread_trace_decoder_wave_state_t
{
    int32_t type;      ///< one of rocprofiler_thread_trace_decoder_waveslot_state_type_t
    int32_t duration;  ///< state duration in cycles
} rocprofiler_thread_trace_decoder_wave_state_t;

/**
 * @brief Instruction type.
 */
typedef enum rocprofiler_thread_trace_decoder_inst_category_t
{
    ROCPROFILER_THREAD_TRACE_DECODER_INST_NONE = 0,
    ROCPROFILER_THREAD_TRACE_DECODER_INST_SMEM,     ///< Scalar memory op
    ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU,     ///< Scalar ALU op
    ROCPROFILER_THREAD_TRACE_DECODER_INST_VMEM,     ///< Vector memory op
    ROCPROFILER_THREAD_TRACE_DECODER_INST_FLAT,     ///< Flat addressing vmem or lds
    ROCPROFILER_THREAD_TRACE_DECODER_INST_LDS,      ///< Local Data Share op
    ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU,     ///< Vector ALU op
    ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP,     ///< Branch taken
    ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT,     ///< Branch not taken
    ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED,    ///< Internal operation
    ROCPROFILER_THREAD_TRACE_DECODER_INST_CONTEXT,  ///< Wave context switch
    ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE,  ///< MSG types
    ROCPROFILER_THREAD_TRACE_DECODER_INST_BVH,      ///< Raytrace op
    ROCPROFILER_THREAD_TRACE_DECODER_INST_LAST
} rocprofiler_thread_trace_decoder_inst_category_t;

/**
 * @brief Describes an instruction execution event.
 *
 * Bitfields defined exec_time and category
 *
 * Exec time is defined differently for different architectures:
 * ::exec == issue time (gfx9) or ::exec == completion time (gfx10+)
 *
 * ::time marks when the wave first attempted to execute this instruction
 * ::time + ::duration marks the issue completion (gfx9) or completion (gfx10+) time
 * ::time + stall marks when the wave first attempted to issue the instruction (when ::exec begins)
 * Stalled time can be computed as: ::duration - ::exec
 */
typedef struct rocprofiler_thread_trace_decoder_inst_t
{
    uint8_t  category;
    uint8_t  reserved;
    uint16_t exec;
    int32_t  duration;  ///< Total instruction duration, in clock cycles. Stall + Exec
    rocprofiler_shader_timestamp_t        time;
    rocprofiler_thread_trace_decoder_pc_t pc;
} rocprofiler_thread_trace_decoder_inst_t;

/**
 * @brief Struct describing a wave during it's lifetime.
 * This record is only generated for waves executing in the target_cu and target_simd, selected by
 * ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU and ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT
 *
 * instructions_array contains a time-ordered list of all (traced) instructions by the wave.
 */
typedef struct rocprofiler_thread_trace_decoder_wave_t
{
    uint8_t cu;        ///< CU id (gfx9) or wgp id (gfx10+). This is always the target_cu.
    uint8_t simd;      ///< SIMD ID [0,3].
    uint8_t wave_id;   ///< Wave slot ID within SIMD.
    uint8_t contexts;  ///< Counts how many CWSR events have occured during the wave lifetime.

    uint32_t _rsvd1;
    uint32_t _rsvd2;
    uint32_t _rsvd3;

    rocprofiler_shader_timestamp_t start_time;  ///< Matches occupancy event wave start.
    rocprofiler_shader_timestamp_t end_time;    ///< Matches occupancy event wave end.

    size_t                                         timeline_size;      ///< timeline_array size
    size_t                                         instructions_size;  ///< instructions_array size
    rocprofiler_thread_trace_decoder_wave_state_t* timeline_array;     ///< wave state change events
    rocprofiler_thread_trace_decoder_inst_t*       instructions_array;  ///< Instructions executed
} rocprofiler_thread_trace_decoder_wave_t;

/**
 * @brief Defines the type of payload received by rocprofiler_thread_trace_decoder_callback_t
 */
typedef enum rocprofiler_thread_trace_decoder_record_type_t
{
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP = 0,  ///< Record is gfxip_major, type size_t
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY,  ///< rocprofiler_thread_trace_decoder_occupancy_t*
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT,  ///< rocprofiler_thread_trace_decoder_perfevent_t*
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE,   ///< rocprofiler_thread_trace_decoder_wave_t*
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO,   ///< rocprofiler_thread_trace_decoder_info_t*
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DEBUG,  ///< Debug
    ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST
} rocprofiler_thread_trace_decoder_record_type_t;

/** @} */
