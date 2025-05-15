#include <babeltrace2/babeltrace.h>
#include <babeltrace2/graph/component-class.h>
#include <babeltrace2/graph/component.h>
#include <babeltrace2/graph/port.h>
#include <babeltrace2/trace-ir/field.h>

#include <deque>
#include <cstring>

#include "lib/common/filesystem.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>

#include <utility>

// State for the source component
struct hip_api_source_state {
    std::deque<rocprofiler_buffer_tracing_hip_api_ext_record_t>* hip_api_data;
    bt_stream *stream;
    bt_event_class *event_class;
};

// --- Component methods ---

// Initialization method: create trace/stream/event classes and store state
static
bt_component_class_initialize_method_status hip_api_source_init(
    bt_self_component_source *self_comp_src,
    bt_self_component_source_configuration * /*config*/,
    const bt_value * /*params*/,
    void * init_data)
{
    std::deque<rocprofiler_buffer_tracing_hip_api_ext_record_t>* hip_api_data = (std::deque<rocprofiler_buffer_tracing_hip_api_ext_record_t>*)init_data;

    // Create trace class
    bt_self_component *self_comp = bt_self_component_source_as_self_component(self_comp_src);
    bt_trace_class *trace_class = bt_trace_class_create(self_comp);

    // Create stream class
    bt_stream_class *stream_class = bt_stream_class_create(trace_class);

    // Create event class
    bt_event_class *event_class = bt_event_class_create(stream_class);

    // Create payload field class
    bt_field_class *payload_fc = bt_field_class_structure_create(trace_class);
    bt_field_class_structure_append_member(payload_fc, "kind", bt_field_class_integer_unsigned_create(trace_class));
    bt_field_class_structure_append_member(payload_fc, "operation", bt_field_class_integer_unsigned_create(trace_class));
    bt_field_class_structure_append_member(payload_fc, "correlation_id", bt_field_class_integer_unsigned_create(trace_class));
    bt_field_class_structure_append_member(payload_fc, "thread_id", bt_field_class_integer_unsigned_create(trace_class));
    bt_event_class_set_payload_field_class(event_class, payload_fc);

    // Create trace and stream
    bt_trace *trace = bt_trace_create(trace_class);
    bt_stream *stream = bt_stream_create(stream_class, trace);

    // Store state
    hip_api_source_state* state = static_cast<hip_api_source_state*>(malloc(sizeof(hip_api_source_state)));
    state->hip_api_data = hip_api_data;
    state->stream = stream;
    state->event_class = event_class;
    bt_self_component_set_data(bt_self_component_source_as_self_component(self_comp_src), state);

    return BT_COMPONENT_CLASS_INITIALIZE_METHOD_STATUS_OK;
}

// Next method: emit events from the queue
static
bt_message_iterator_class_next_method_status hip_api_source_next(
    bt_self_message_iterator *self_message_iterator,
		bt_message_array_const msgs, uint64_t capacity,
		uint64_t *count)
{
    // Get the component's user data (state)
    bt_self_component_source *self_comp_src =
    (bt_self_component_source *) bt_self_message_iterator_borrow_component(self_message_iterator);
    hip_api_source_state* state = (hip_api_source_state*)
        bt_self_component_get_data(bt_self_component_source_as_self_component(self_comp_src));

    uint64_t produced = 0;

    while (produced < capacity && !state->hip_api_data->empty()) {
        rocprofiler_buffer_tracing_hip_api_ext_record_t* rec = &state->hip_api_data->back();

        // Create event message
        bt_message *msg = bt_message_event_create(
            self_message_iterator,
            state->event_class,
            state->stream
        );
        if (!msg) break;

        // Set event payload fields
        bt_event *evt = bt_message_event_borrow_event(msg);
        bt_field *payload = bt_event_borrow_payload_field(evt);

        bt_field *kind_field = bt_field_structure_borrow_member_field_by_name(payload, "kind");
        bt_field_integer_unsigned_set_value(kind_field, rec->kind);

        bt_field *op_field = bt_field_structure_borrow_member_field_by_name(payload, "operation");
        bt_field_integer_signed_set_value(op_field, rec->operation);

        bt_field *corr_field = bt_field_structure_borrow_member_field_by_name(payload, "correlation_id");
        bt_field_integer_unsigned_set_value(corr_field, rec->correlation_id.internal);

        bt_field *tid_field = bt_field_structure_borrow_member_field_by_name(payload, "thread_id");
        bt_field_integer_unsigned_set_value(tid_field, rec->thread_id);

        msgs[produced++] = msg;
        state->hip_api_data->pop_back();
    }

    *count = produced;
    return BT_MESSAGE_ITERATOR_CLASS_NEXT_METHOD_STATUS_OK;
}

// Finalize method: cleanup
static
void hip_api_source_finalize(bt_self_component_source *self_comp_src)
{
    hip_api_source_state* state = (hip_api_source_state*)bt_self_component_get_data(
        bt_self_component_source_as_self_component(self_comp_src));
    free(state);
}