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

#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "lib/rocprofiler-sdk/pc_sampling/parser/translation.hpp"

template <>
struct std::hash<device_handle>
{
    size_t operator()(const device_handle& d) const { return d.handle; }
};
bool inline
operator==(device_handle a, device_handle b)
{
    return a.handle == b.handle;
}

namespace Parser
{
/**
 * @brief Struct immitating the correlation_id returned by the trap handler in raw PC samples.
 */
union trap_correlation_id_t
{
    uint64_t raw;
    struct
    {
        uint64_t dispatch_index : 25;
        uint64_t _reserved0     : 7;
        uint64_t doorbell_id    : 10;
        uint64_t _reserved1     : 22;
    } wrapped;
};

struct DispatchPkt
{
    trap_correlation_id_t correlation_id_in;  //! Correlation ID seen by the trap handler
    device_handle         dev;                //! Which device this is run
};

inline bool
operator==(const trap_correlation_id_t& a, const trap_correlation_id_t& b)
{
    return a.raw == b.raw;
}

inline bool
operator==(const DispatchPkt& a, const DispatchPkt& b)
{
    return a.correlation_id_in == b.correlation_id_in && a.dev == b.dev;
}
}  // namespace Parser

template <>
struct std::hash<Parser::DispatchPkt>
{
    size_t operator()(const Parser::DispatchPkt& d) const
    {
        return (d.correlation_id_in.raw << 8) ^ d.dev.handle;
    }
};

namespace Parser
{
/**
 * Coordinates DispatchMap and DoorBellMap to reconstruct the original correlation_id
 * from the correlation_id seen by the trap handler.
 */
class CorrelationMap
{
public:
    CorrelationMap() = default;

    /**
     * Checks wether a dispatch pkt will generate a collision.
     * @returns true on collision and false when slot is available.
     */
    bool checkDispatch(const dispatch_pkt_id_t& pkt) const
    {
        auto trap = trap_correlation_id(pkt.doorbell_id, pkt.write_index, pkt.queue_size);
        return dispatch_to_correlation.find({trap, pkt.device}) != dispatch_to_correlation.end();
    }

    /**
     * @brief Updates the mapping of dispatch_id to correlation_id
     */
    void newDispatch(const dispatch_pkt_id_t& pkt)
    {
        cache_dev_id = ~0ul;
        auto trap_id = trap_correlation_id(pkt.doorbell_id, pkt.write_index, pkt.queue_size);
        dispatch_to_correlation[{trap_id, pkt.device}] = pkt.correlation_id;
    }

    /**
     * @brief Allows the parser to forget a correlation_id, to save memory.
     */
    void forget(const dispatch_pkt_id_t& pkt)
    {
        cache_dev_id = ~0ul;
        auto trap_id = trap_correlation_id(pkt.doorbell_id, pkt.write_index, pkt.queue_size);
        dispatch_to_correlation.erase({trap_id, pkt.device});
    }

    /**
     * Given a device dev, doorbell and and wrapped dispatch_id,
     * @returns the correlation_id set by dispatch_pkt_id_t
     */
    uint64_t get(device_handle dev, trap_correlation_id_t correlation_in)
    {
#ifndef _PARSER_CORRELATION_DISABLE_CACHE
        if(dev.handle == cache_dev_id && correlation_in == cache_correlation_id_in)
            return cache_correlation_id_out;
#endif
        cache_dev_id             = dev.handle;
        cache_correlation_id_in  = correlation_in;
        cache_correlation_id_out = dispatch_to_correlation.at({correlation_in, dev});
        return cache_correlation_id_out;
    }

    /**
     * Returns the correlation_id as seen by the trap handler, consisting of a
     * - wrapped dispatch_pkt
     * - doorbell_id divibed by 8 Bytes
     * @param[in] doorbell The doorbell handler returned by HSA
     * @param[in] write_idx The dispatch packet write index, [optional] not wrapped
     * @param[in] queue_size The queue size. [optional] If write_index is already wrapped,
     *                       then this value can just be a large integer > queue_size.
     * @returns The correlation_id immitating the ones returned by the trap handler.
     */
    static trap_correlation_id_t trap_correlation_id(uint64_t doorbell,
                                                     uint64_t write_idx,
                                                     uint64_t queue_size)
    {
        trap_correlation_id_t trap{.raw = 0};
        trap.wrapped.dispatch_index = write_idx % queue_size;
        trap.wrapped.doorbell_id    = doorbell >> 3;
        return trap;
    }

private:
    std::unordered_map<DispatchPkt, uint64_t> dispatch_to_correlation{};

    // Making get() const and these cache variables mutable causes performance to be unstable
    trap_correlation_id_t cache_correlation_id_in{.raw = ~0ul};  // Invalid value in cache
    uint64_t              cache_correlation_id_out = ~0ul;
    uint64_t              cache_dev_id             = ~0ul;  // Invalid device Id in cache
};
}  // namespace Parser

template <bool bHostTrap, typename GFXIP>
inline pcsample_status_t
add_upcoming_samples(const device_handle     device,
                     const generic_sample_t* buffer,
                     const size_t            available_samples,
                     Parser::CorrelationMap* corr_map,
                     pcsample_v1_t*          samples)
{
    pcsample_status_t status = PCSAMPLE_STATUS_SUCCESS;
    for(uint64_t p = 0; p < available_samples; p++)
    {
        const auto* snap = reinterpret_cast<const perf_sample_snapshot_v1*>(buffer + p);
        samples[p]       = copySample<bHostTrap, GFXIP>((const void*) (buffer + p));
        try
        {
            Parser::trap_correlation_id_t trap{.raw = snap->correlation_id};
            samples[p].correlation_id = corr_map->get(device, trap);
        } catch(std::exception& e)
        {
            status = PCSAMPLE_STATUS_PARSER_ERROR;
        }
    }
    return status;
}

template <typename GFXIP>
inline pcsample_status_t
_parse_buffer(generic_sample_t*       buffer,
              uint64_t                buffer_size,
              user_callback_t         callback,
              void*                   userdata,
              Parser::CorrelationMap* corr_map)
{
    // Maximum size
    uint64_t index = 0;

    pcsample_status_t status = PCSAMPLE_STATUS_SUCCESS;

    while(index < buffer_size)
    {
        switch(buffer[index].type)
        {
            case AMD_DISPATCH_PKT_ID:
            {
                const auto& pkt = *reinterpret_cast<const dispatch_pkt_id_t*>(buffer + index);
                if(pkt.queue_size >= (1 << 25)) status = PCSAMPLE_STATUS_PARSER_ERROR;
                index += 1;
                corr_map->newDispatch(pkt);
                break;
            }
            case AMD_UPCOMING_SAMPLES:
            {
                const auto& pkt = *reinterpret_cast<const upcoming_samples_t*>(buffer + index);
                index += 1;

                uint64_t pkt_counter = pkt.num_samples;
                if(index + pkt_counter > buffer_size) return PCSAMPLE_STATUS_OUT_OF_BOUNDS_ERROR;

                bool bIsHostTrap = pkt.which_sample_type == AMD_HOST_TRAP_V1;

                while(pkt_counter > 0)
                {
                    pcsample_v1_t* samples           = nullptr;
                    uint64_t       available_samples = callback(&samples, pkt_counter, userdata);

                    if(available_samples == 0 || available_samples > pkt_counter)
                        return PCSAMPLE_STATUS_CALLBACK_ERROR;

                    if(bIsHostTrap)
                    {
                        status |= add_upcoming_samples<true, GFXIP>(
                            pkt.device, buffer + index, available_samples, corr_map, samples);
                    }
                    else
                    {
                        status |= add_upcoming_samples<false, GFXIP>(
                            pkt.device, buffer + index, available_samples, corr_map, samples);
                    }

                    index += available_samples;
                    pkt_counter -= available_samples;
                }
                break;
            }
            default:
                std::cerr << "Index " << index << " - Invalid sample type: " << buffer[index].type
                          << std::endl;
                return PCSAMPLE_STATUS_INVALID_SAMPLE;
        }
    }
    return status;
};

/**
 * @brief Parses a given set of pc samples.
 * @param[in] buffer Pointer to a buffer containing metadata and pcsamples.
 * @param[in] buffer_size The number of elements in the buffer.
 * @param[in] gfxip_major GFXIP major version of the samples.
 * @param[in] callback A callback function that accepts a double pointer to write the samples to,
 * a size requested parameter (number of pc_sample_t) and a void* to userdata.
 * The callback is expected to allocate 64B-aligned memory where the parsed samples are going to
 * be written to, and return the size of memory that was allocated, in multiples of
 * sizeof(generic_sample_t). If the callback returns 0 or a larger size than requested,
 * parse_buffer() will return PCSAMPLE_STATUS_CALLBACK_ERROR. If the callback returns
 * a size smaller than requested, then it may be called again requesting more memory.
 * @param[in] userdata parameter forwarded to the user callback.
 */
pcsample_status_t inline parse_buffer(generic_sample_t* buffer,
                                      uint64_t          buffer_size,
                                      int               gfxip_major,
                                      user_callback_t   callback,
                                      void*             userdata)
{
    static auto corr_map = std::make_unique<Parser::CorrelationMap>();

    auto parseSample_func = _parse_buffer<GFX9>;
    if(gfxip_major == 9)
        parseSample_func = _parse_buffer<GFX9>;
    else if(gfxip_major == 11)
        parseSample_func = _parse_buffer<GFX11>;
    else if(gfxip_major == 12)
        parseSample_func = _parse_buffer<GFX12>;
    else
        return PCSAMPLE_STATUS_INVALID_GFXIP;

    return parseSample_func(buffer, buffer_size, callback, userdata, corr_map.get());
};
