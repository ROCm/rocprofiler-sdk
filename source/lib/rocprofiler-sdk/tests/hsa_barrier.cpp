#include <algorithm>
#include <random>

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include <gtest/gtest.h>

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa_barrier.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

using namespace rocprofiler;
using namespace rocprofiler::hsa;

namespace
{
AmdExtTable&
get_ext_table()
{
    static auto _v = []() {
        auto val                                  = AmdExtTable{};
        val.hsa_amd_memory_pool_get_info_fn       = hsa_amd_memory_pool_get_info;
        val.hsa_amd_agent_iterate_memory_pools_fn = hsa_amd_agent_iterate_memory_pools;
        val.hsa_amd_memory_pool_allocate_fn       = hsa_amd_memory_pool_allocate;
        val.hsa_amd_memory_pool_free_fn           = hsa_amd_memory_pool_free;
        val.hsa_amd_agent_memory_pool_get_info_fn = hsa_amd_agent_memory_pool_get_info;
        val.hsa_amd_agents_allow_access_fn        = hsa_amd_agents_allow_access;
        return val;
    }();
    return _v;
}

CoreApiTable&
get_api_table()
{
    static auto _v = []() {
        auto val                           = CoreApiTable{};
        val.hsa_iterate_agents_fn          = hsa_iterate_agents;
        val.hsa_agent_get_info_fn          = hsa_agent_get_info;
        val.hsa_queue_create_fn            = hsa_queue_create;
        val.hsa_queue_destroy_fn           = hsa_queue_destroy;
        val.hsa_signal_create_fn           = hsa_signal_create;
        val.hsa_signal_destroy_fn          = hsa_signal_destroy;
        val.hsa_signal_store_screlease_fn  = hsa_signal_store_screlease;
        val.hsa_signal_load_scacquire_fn   = hsa_signal_load_scacquire;
        val.hsa_signal_add_relaxed_fn      = hsa_signal_add_relaxed;
        val.hsa_signal_subtract_relaxed_fn = hsa_signal_subtract_relaxed;
        val.hsa_signal_wait_relaxed_fn     = hsa_signal_wait_relaxed;
        return val;
    }();
    return _v;
}

namespace rocprofiler
{
namespace hsa
{
class FakeQueue : public Queue
{
public:
    FakeQueue(const AgentCache& a, rocprofiler_queue_id_t id)
    : Queue(a, get_api_table())
    , _agent(a)
    , _id(id)
    {}
    virtual const AgentCache&      get_agent() const override final { return _agent; };
    virtual rocprofiler_queue_id_t get_id() const override final { return _id; };

    ~FakeQueue() {}

private:
    const AgentCache&      _agent;
    rocprofiler_queue_id_t _id = {};
};

}  // namespace hsa
}  // namespace rocprofiler

QueueController::queue_map_t
create_queue_map(size_t count)
{
    QueueController::queue_map_t ret;

    // ensure test fails if null
    EXPECT_TRUE(hsa::get_queue_controller() != nullptr);

    // prevent segfault
    if(!hsa::get_queue_controller()) return ret;

    auto agents = hsa::get_queue_controller()->get_supported_agents();

    for(size_t i = 0; i < count; i++)
    {
        auto& agent_cache = agents.begin()->second;
        // Create queue
        hsa_queue_t* queue;
        hsa_queue_create(agent_cache.get_hsa_agent(),
                         2048,
                         HSA_QUEUE_TYPE_SINGLE,
                         nullptr,
                         nullptr,
                         0,
                         0,
                         &queue);
        ret[queue] = std::make_unique<rocprofiler::hsa::FakeQueue>(
            agent_cache, rocprofiler_queue_id_t{.handle = i});
    }

    return ret;
}

std::atomic<bool> should_execute_handler{false};
std::atomic<int>  executed_handlers{0};
bool
barrier_signal_handler(hsa_signal_value_t, void* data)
{
    CHECK(data);
    CHECK(should_execute_handler) << "Signal handler called when it should not have been";
    hsa_signal_destroy(*static_cast<hsa_signal_t*>(data));
    delete static_cast<hsa_signal_t*>(data);
    executed_handlers++;
    return false;
}

// Injects a barrier packet into the queue followed by a packet with an async handler
// associated with it. If the barrier is not released, the async handler should not
// be executed (checked with should_execute_handler).
void
inject_barriers(hsa_barrier& barrier, QueueController::queue_map_t& queues)
{
    auto packet_store_release = [](uint32_t* packet, uint16_t header, uint16_t rest) {
        __atomic_store_n(packet, header | (rest << 16), __ATOMIC_RELEASE);
    };

    auto header_pkt = [](hsa_packet_type_t type) {
        uint16_t header = type << HSA_PACKET_HEADER_TYPE;
        header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
        header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
        return header;
    };

    auto enqueue_pkt = [&](auto& queue, auto& packets, auto& pkt) {
        uint64_t packet_id = hsa_queue_add_write_index_screlease(queue, 1);
        while(packet_id - hsa_queue_load_read_index_scacquire(queue) >= queue->size)
            ;
        hsa_barrier_and_packet_t* packet = packets + packet_id % queue->size;
        (*packet)                        = pkt;
        packet_store_release((uint32_t*) packet, header_pkt(HSA_PACKET_TYPE_BARRIER_AND), 0);
        hsa_signal_store_screlease(queue->doorbell_signal, packet_id);
    };

    for(auto& [hsa_queue, fq] : queues)
    {
        auto pkt = barrier.enqueue_packet(fq.get());
        ASSERT_EQ(pkt.has_value(), true);
        hsa_barrier_and_packet_t* packets = (hsa_barrier_and_packet_t*) hsa_queue->base_address;
        enqueue_pkt(hsa_queue, packets, pkt->barrier_and);

        // Construct packet that will trigger async handler after barrier is released
        rocprofiler_packet post_barrier{};
        hsa_signal_t*      completion_signal = new hsa_signal_t;
        hsa_signal_create(1, 0, nullptr, completion_signal);
        post_barrier.barrier_and.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        post_barrier.barrier_and.completion_signal = *completion_signal;
        hsa_amd_signal_async_handler(*completion_signal,
                                     HSA_SIGNAL_CONDITION_EQ,
                                     0,
                                     barrier_signal_handler,
                                     static_cast<void*>(completion_signal));
        enqueue_pkt(hsa_queue, packets, post_barrier.barrier_and);
    }

    // Ensure that the barrier packet is reached on all queues
    usleep(100);
}

void
test_init()
{
    HsaApiTable table;
    table.amd_ext_ = &get_ext_table();
    table.core_    = &get_api_table();
    agent::construct_agent_cache(&table);
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    hsa::get_queue_controller()->init(get_api_table(), get_ext_table());
}
}  // namespace

TEST(hsa_barrier, no_block_single)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    bool complete      = false;
    auto finished_func = [&]() { complete = true; };

    auto queues = create_queue_map(1);

    // Immediate return of barrier due to no active async packets
    hsa::hsa_barrier barrier(finished_func, get_api_table());
    barrier.set_barrier(queues);
    executed_handlers = 0;
    ASSERT_TRUE(barrier.complete());
    should_execute_handler = true;
    inject_barriers(barrier, queues);
    ASSERT_EQ(complete, true);
    while(executed_handlers != 1)
    {
        usleep(10);
    }

    registration::set_init_status(1);
    registration::finalize();
}

TEST(hsa_barrier, no_block_multi)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    bool complete      = false;
    auto finished_func = [&]() { complete = true; };

    auto queues = create_queue_map(10);

    // Immediate return of barrier due to no active async packets
    hsa::hsa_barrier barrier(finished_func, get_api_table());
    barrier.set_barrier(queues);
    ASSERT_TRUE(barrier.complete());
    should_execute_handler = true;
    executed_handlers      = 0;
    inject_barriers(barrier, queues);
    ASSERT_EQ(complete, true);
    while(executed_handlers != 10)
    {
        usleep(10);
    }

    registration::set_init_status(1);
    registration::finalize();
}

TEST(hsa_barrier, block_single)
{
    std::vector<Queue*> pkt_waiting;
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    bool complete      = false;
    auto finished_func = [&]() { complete = true; };

    auto queues = create_queue_map(1);

    hsa::hsa_barrier barrier(finished_func, get_api_table());

    // Simulate waiting on packets already in the queue to complete
    for(auto& [_, queue] : queues)
    {
        pkt_waiting.push_back(queue.get());
        queue->async_started();
    }
    should_execute_handler = false;
    executed_handlers      = 0;

    barrier.set_barrier(queues);
    ASSERT_FALSE(barrier.complete());

    should_execute_handler = false;
    executed_handlers      = 0;
    inject_barriers(barrier, queues);

    ASSERT_EQ(complete, false);
    should_execute_handler = true;

    for(auto& queue : pkt_waiting)
    {
        queue->async_complete();
        barrier.register_completion(queue);
    }

    ASSERT_EQ(complete, true);
    // Wait for the signal handlers to execute. If we deadlock here,
    // we are not triggering the completion of the signal handler.
    while(executed_handlers != 1)
    {
        usleep(100);
    }

    registration::set_init_status(1);
    registration::finalize();
}

TEST(hsa_barrier, block_multi)
{
    std::vector<Queue*> pkt_waiting;
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    bool complete      = false;
    auto finished_func = [&]() { complete = true; };

    auto queues = create_queue_map(10);

    // Immediate return of barrier due to no active async packets
    hsa::hsa_barrier barrier(finished_func, get_api_table());

    // Simulate waiting on packets already in the queue to complete
    for(auto& [_, queue] : queues)
    {
        for(size_t i = 0; i < 30; i++)
        {
            pkt_waiting.push_back(queue.get());
            queue->async_started();
        }
    }
    should_execute_handler = false;
    executed_handlers      = 0;

    barrier.set_barrier(queues);
    ASSERT_FALSE(barrier.complete());

    should_execute_handler = false;
    executed_handlers      = 0;
    inject_barriers(barrier, queues);

    ASSERT_EQ(complete, false);

    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(pkt_waiting), std::end(pkt_waiting), rng);
    for(size_t i = 0; i < pkt_waiting.size(); i++)
    {
        ASSERT_EQ(complete, false);
        ASSERT_FALSE(barrier.complete());
        if(i == pkt_waiting.size() - 1)
        {
            should_execute_handler = true;
        }
        pkt_waiting[i]->async_complete();
        barrier.register_completion(pkt_waiting[i]);
    }

    ASSERT_EQ(complete, true);
    // Wait for the signal handlers to execute. If we deadlock here,
    // we are not triggering the completion of the signal handler.
    while(executed_handlers != 10)
    {
        usleep(100);
    }

    registration::set_init_status(1);
    registration::finalize();
}
