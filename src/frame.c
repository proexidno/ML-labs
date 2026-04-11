#include "frame.h"
#include "connection_id.h"
#include "errors.h"
#include "utils.h"
#include "varint.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Reads QUIC frame from given UNENCRYPTED buffer
 *
 * Changes buffer and left in-place
 *
 * Headers should be initialized on stack or in dynamic memory
 * Call free_dynamic_frame_info before freing or exiting the scope
 *
 * Returns negative value if error has occured
 *
 * Returns quic_frame_types_t otherwise
 */
int frame_read_stream(uint8_t **buffer, size_t *left, quic_frames_t *frames) {
  varint_t type;
  if (varint_read_stream(buffer, left, &type)) {
    return FRAME_ENCODING_ERROR;
  }

  switch (type) {
  case PADDING:
    break;
  case PING:
    break;
  case ACK: {
    varint_t ack_range_count = 0;
    ack_frame_t *frame = &frames->ack_frame;
    if (varint_read_stream(buffer, left, &frame->largest_acked)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->ack_delay)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &ack_range_count)) {
      return FRAME_ENCODING_ERROR;
    }
    frame->ack_ranges =
        calloc(1 + ack_range_count * 2, sizeof(*frame->ack_ranges));
    if (frame->ack_ranges == NULL) {
      return MALLOC_FAIL;
    }

    varint_t *pointer = frame->ack_ranges;
    varint_t *end = pointer + 1 + 2 * ack_range_count;
    while (pointer < end) {
      if (varint_read_stream(buffer, left, pointer++)) {
        free_dynamic_frame_info(type, frames);
        return FRAME_ENCODING_ERROR;
      }
    }

    break;
  }
  case ACK_ECN: {
    varint_t ack_range_count = 0;
    ack_ecn_frame_t *frame = &frames->ack_ecn_frame;
    if (varint_read_stream(buffer, left, &frame->ack_frame.largest_acked)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->ack_frame.ack_delay)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &ack_range_count)) {
      return FRAME_ENCODING_ERROR;
    }
    frame->ack_frame.ack_ranges =
        calloc(1 + ack_range_count * 2, sizeof(*frame->ack_frame.ack_ranges));
    if (frame->ack_frame.ack_ranges == NULL) {
      return MALLOC_FAIL;
    }

    varint_t *pointer = frame->ack_frame.ack_ranges;
    varint_t *end = pointer + 1 + 2 * ack_range_count;
    while (pointer < end) {
      if (varint_read_stream(buffer, left, pointer++)) {
        free_dynamic_frame_info(type, frames);
        return FRAME_ENCODING_ERROR;
      }
    }

    if (varint_read_stream(buffer, left, &frame->ecn0_count)) {
      free_dynamic_frame_info(type, frames);
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->ecn1_count)) {
      free_dynamic_frame_info(type, frames);
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->ecn_ce_count)) {
      free_dynamic_frame_info(type, frames);
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case RESET_STREAM: {
    reset_stream_frame_t *frame = &frames->reset_stream_frame;
    if (varint_read_stream(buffer, left, &frame->stream_id)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->application_error_code)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->final_size)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case STOP_SENDING: {
    stop_sending_frame_t *frame = &frames->stop_sending_frame;
    if (varint_read_stream(buffer, left, &frame->stream_id)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->application_error_code)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case CRYPTO: {
    crypto_frame_t *frame = &frames->crypto_frame;
    if (varint_read_stream(buffer, left, &frame->offset)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->length)) {
      return FRAME_ENCODING_ERROR;
    }
    if (frame->offset + frame->length > MAX_VARINT) {
      return FRAME_ENCODING_ERROR;
    }
    frame->data = calloc(frame->length, sizeof(*frame->data));
    if (frame->data == NULL) {
      return MALLOC_FAIL;
    }
    if (stream_read_n_bytes(buffer, left, frame->data, frame->length)) {
      free_dynamic_frame_info(type, frames);
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case NEW_TOKEN: {
    new_token_frame_t *frame = &frames->new_token_frame;
    if (varint_read_stream(buffer, left, &frame->token_length)) {
      return FRAME_ENCODING_ERROR;
    }
    if (frame->token_length == 0) {
      return FRAME_ENCODING_ERROR;
    }
    frame->token = calloc(frame->token_length, sizeof(*frame->token));
    if (frame->token == NULL) {
      return MALLOC_FAIL;
    }
    if (stream_read_n_bytes(buffer, left, frame->token, frame->token_length)) {
      free_dynamic_frame_info(type, frames);
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case STREAM ... STREAM + 7: {
    stream_frame_t *frame = &frames->stream_frame;
    if (type & 0x04) {
      frame->offset_flag = 1;
      if (varint_read_stream(buffer, left, &frame->offset)) {
        return FRAME_ENCODING_ERROR;
      }
    }

    if (type & 0x02) {
      frame->length_flag = 1;
      if (varint_read_stream(buffer, left, &frame->length)) {
        return FRAME_ENCODING_ERROR;
      }
      frame->data = calloc(frame->length, sizeof(*frame->data));
      if (frame->data == NULL) {
        return MALLOC_FAIL;
      }
      if (stream_read_n_bytes(buffer, left, frame->data, frame->length)) {
        free_dynamic_frame_info(type, frames);
        return FRAME_ENCODING_ERROR;
      }
    } else {
      frame->data = calloc(*left, sizeof(*frame->data));
      if (frame->data == NULL) {
        return MALLOC_FAIL;
      }
      if (stream_read_n_bytes(buffer, left, frame->data, *left)) {
        free_dynamic_frame_info(type, frames);
        return FRAME_ENCODING_ERROR;
      }
    }

    frame->fin_flag = type & 0x01;
    type = 0x08;
    break;
  }
  case MAX_DATA: {
    max_data_frame_t *frame = &frames->max_data_frame;
    if (varint_read_stream(buffer, left, &frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case MAX_STREAM_DATA: {
    max_stream_data_frame_t *frame = &frames->max_stream_data_frame;
    if (varint_read_stream(buffer, left, &frame->stream_id)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case MAX_STREAMS_BI:
  case MAX_STREAMS_UNI: {
    max_streams_frame_t *frame = &frames->max_streams_frame;
    if (varint_read_stream(buffer, left, &frame->max_streams)) {
      return FRAME_ENCODING_ERROR;
    }
    if (frame->max_streams > (MAX_VARINT + 1) / 4) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case DATA_BLOCKED: {
    data_blocked_frame_t *frame = &frames->data_blocked_frame;
    if (varint_read_stream(buffer, left, &frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case STREAM_DATA_BLOCKED: {
    stream_data_blocked_frame_t *frame = &frames->stream_data_blocked_frame;
    if (varint_read_stream(buffer, left, &frame->stream_id)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case STREAMS_BLOCKED_BI:
  case STREAMS_BLOCKED_UNI: {
    streams_blocked_frame_t *frame = &frames->streams_blocked_frame;
    if (varint_read_stream(buffer, left, &frame->max_streams)) {
      return FRAME_ENCODING_ERROR;
    }
    if (frame->max_streams > (MAX_VARINT + 1) / 4) {
      return STREAM_LIMIT_ERROR;
    }
    break;
  }
  case NEW_CONNECTION_ID: {
    new_conn_id_frame_t *frame = &frames->new_conn_id_frame;
    if (varint_read_stream(buffer, left, &frame->seq_number)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->retire_prior_to)) {
      return FRAME_ENCODING_ERROR;
    }

    if (frame->retire_prior_to > frame->seq_number) {
      return FRAME_ENCODING_ERROR;
    }

    if (stream_read_n_bytes(buffer, left, &frame->conn_id_length,
                            sizeof(frame->conn_id_length))) {
      return FRAME_ENCODING_ERROR;
    }

    if (frame->conn_id_length < MIN_CONN_ID_LENGTH ||
        frame->conn_id_length > MAX_CONN_ID_LENGTH) {
      return FRAME_ENCODING_ERROR;
    }

    if (stream_read_n_bytes(buffer, left, frame->conn_id,
                            frame->conn_id_length)) {
      return FRAME_ENCODING_ERROR;
    }
    if (stream_read_n_bytes(buffer, left, frame->stateless_reset_token,
                            sizeof(frame->stateless_reset_token) /
                                sizeof(*frame->stateless_reset_token))) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case RETIRE_CONNECTION_ID: {
    retire_conn_id_frame_t *frame = &frames->retire_conn_id_frame;
    if (varint_read_stream(buffer, left, &frame->retire_seq_number)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case PATH_CHALLENGE:
  case PATH_RESPONSE: {
    path_challenge_frame_t *frame = &frames->path_challenge_frame;
    if (stream_read_n_bytes(buffer, left, frame->data,
                            sizeof(frame->data) / sizeof(*frame->data))) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case CONNECTION_CLOSE ... CONNECTION_CLOSE + 1: {
    conn_close_frame_t *frame = &frames->conn_close_frame;
    frame->application_error_flag = type & 0x01;
    if (varint_read_stream(buffer, left, &frame->error_code)) {
      return FRAME_ENCODING_ERROR;
    }
    if (!frame->application_error_flag) {
      if (varint_read_stream(buffer, left, &frame->frame_type)) {
        return FRAME_ENCODING_ERROR;
      }
    }

    if (varint_read_stream(buffer, left, &frame->reason_phrase_length)) {
      return FRAME_ENCODING_ERROR;
    }

    frame->reason_phrase =
        calloc(frame->reason_phrase_length, sizeof(*frame->reason_phrase));
    if (frame->reason_phrase == NULL) {
      return MALLOC_FAIL;
    }

    if (stream_read_n_bytes(buffer, left, frame->reason_phrase,
                            frame->reason_phrase_length)) {
      free_dynamic_frame_info(type, frames);
      return FRAME_ENCODING_ERROR;
    }
    type = CONNECTION_CLOSE;
    break;
  }
  case HANDSHAKE_DONE: {
    break;
  }
  default:
    return FRAME_ENCODING_ERROR;
    break;
  }
  return type;
}

int frame_write_stream(uint8_t **buffer, size_t *left, quic_frame_types_t type,
                       quic_frames_t *frames) {
  uint8_t *pointer = *buffer; // Different from write because on the error we
                              // should make pointer back to original place

  size_t l = *left;

  switch (type) {
  case PADDING:
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  case PING:
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  case ACK: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    ack_frame_t *frame = &frames->ack_frame;
    if (varint_write_stream(&pointer, &l, frame->largest_acked)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->ack_delay)) {
      return FRAME_ENCODING_ERROR;
    }

    if (frame->ack_range_length % 2 != 1) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->ack_range_length / 2)) {
      return FRAME_ENCODING_ERROR;
    }

    // More eddiceint approach, I think
    size_t needed_length = 0;
    if (get_varint_array_length(frame->ack_ranges, frame->ack_range_length,
                                &needed_length)) {
      return FRAME_ENCODING_ERROR;
    }
    if (*left < needed_length) {
      return FRAME_ENCODING_ERROR;
    }

    for (size_t i = 0; i < frame->ack_range_length; i++) {
      varint_write_stream(&pointer, &l, frame->ack_ranges[i]);
    }

    break;
  }
  case ACK_ECN: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    ack_ecn_frame_t *frame = &frames->ack_ecn_frame;
    if (varint_write_stream(&pointer, &l, frame->ack_frame.largest_acked)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->ack_frame.ack_delay)) {
      return FRAME_ENCODING_ERROR;
    }

    if (frame->ack_frame.ack_range_length % 2 != 1) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l,
                            frame->ack_frame.ack_range_length / 2)) {
      return FRAME_ENCODING_ERROR;
    }

    // More eddiceint approach, I think
    size_t needed_length = 0;
    if (get_varint_array_length(frame->ack_frame.ack_ranges,
                                frame->ack_frame.ack_range_length,
                                &needed_length)) {
      return FRAME_ENCODING_ERROR;
    }
    if (*left <
        needed_length + 3) { // + 3 because minimum 3 bytes with ecn counts
      return FRAME_ENCODING_ERROR;
    }

    for (size_t i = 0; i < frame->ack_frame.ack_range_length; i++) {
      varint_write_stream(&pointer, &l, frame->ack_frame.ack_ranges[i]);
    }

    if (varint_write_stream(&pointer, &l, frame->ecn0_count)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->ecn1_count)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->ecn_ce_count)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case RESET_STREAM: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    reset_stream_frame_t *frame = &frames->reset_stream_frame;
    if (varint_write_stream(&pointer, &l, frame->stream_id)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->application_error_code)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->final_size)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case STOP_SENDING: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    stop_sending_frame_t *frame = &frames->stop_sending_frame;
    if (varint_write_stream(&pointer, &l, frame->stream_id)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->application_error_code)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case CRYPTO: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    crypto_frame_t *frame = &frames->crypto_frame;
    if (frame->offset + frame->length > MAX_VARINT) {
      return FRAME_ENCODING_ERROR;
    }

    if (varint_write_stream(&pointer, &l, frame->offset)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->length)) {
      return FRAME_ENCODING_ERROR;
    }
    if (stream_write_n_bytes(&pointer, &l, frame->data, frame->length)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case NEW_TOKEN: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    new_token_frame_t *frame = &frames->new_token_frame;
    if (frame->token_length == 0) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->token_length)) {
      return FRAME_ENCODING_ERROR;
    }
    if (stream_write_n_bytes(&pointer, &l, frame->token, frame->token_length)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case STREAM: {
    stream_frame_t *frame = &frames->stream_frame;
    varint_t masked_type = type;
    masked_type |= (frame->offset_flag != 0) << 2;
    masked_type |= (frame->length_flag != 0) << 1;
    masked_type |= (frame->fin_flag != 0);

    if (varint_write_stream(&pointer, &l, masked_type)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->stream_id)) {
      return FRAME_ENCODING_ERROR;
    }
    if (frame->offset_flag) {
      if (varint_write_stream(&pointer, &l, frame->offset)) {
        return FRAME_ENCODING_ERROR;
      }
    }
    if (frame->length_flag) {
      if (varint_write_stream(&pointer, &l, frame->length)) {
        return FRAME_ENCODING_ERROR;
      }
    }
    if (stream_write_n_bytes(&pointer, &l, frame->data, frame->length)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case MAX_DATA: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    max_data_frame_t *frame = &frames->max_data_frame;
    if (varint_write_stream(&pointer, &l, frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case MAX_STREAM_DATA: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    max_stream_data_frame_t *frame = &frames->max_stream_data_frame;
    if (varint_write_stream(&pointer, &l, frame->stream_id)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case MAX_STREAMS_BI:
  case MAX_STREAMS_UNI: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    max_streams_frame_t *frame = &frames->max_streams_frame;
    if (frame->max_streams > (MAX_VARINT + 1) / 4) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->max_streams)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case DATA_BLOCKED: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    data_blocked_frame_t *frame = &frames->data_blocked_frame;
    if (varint_write_stream(&pointer, &l, frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case STREAM_DATA_BLOCKED: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    stream_data_blocked_frame_t *frame = &frames->stream_data_blocked_frame;
    if (varint_write_stream(&pointer, &l, frame->stream_id)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case STREAMS_BLOCKED_BI:
  case STREAMS_BLOCKED_UNI: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    streams_blocked_frame_t *frame = &frames->streams_blocked_frame;
    if (frame->max_streams > (MAX_VARINT + 1) / 4) {
      return STREAM_LIMIT_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->max_streams)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case NEW_CONNECTION_ID: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    new_conn_id_frame_t *frame = &frames->new_conn_id_frame;
    if (frame->retire_prior_to > frame->seq_number ||
        frame->conn_id_length < MIN_CONN_ID_LENGTH ||
        frame->conn_id_length > MAX_CONN_ID_LENGTH) {
      return FRAME_ENCODING_ERROR;
    }

    if (varint_write_stream(&pointer, &l, frame->seq_number)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_write_stream(&pointer, &l, frame->retire_prior_to)) {
      return FRAME_ENCODING_ERROR;
    }

    if (stream_write_n_bytes(&pointer, &l, &frame->conn_id_length,
                             sizeof(frame->conn_id_length))) {
      return FRAME_ENCODING_ERROR;
    }

    if (stream_write_n_bytes(&pointer, &l, frame->conn_id,
                             frame->conn_id_length)) {
      return FRAME_ENCODING_ERROR;
    }
    if (stream_write_n_bytes(&pointer, &l, frame->stateless_reset_token,
                             sizeof(frame->stateless_reset_token) /
                                 sizeof(*frame->stateless_reset_token))) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case RETIRE_CONNECTION_ID: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    retire_conn_id_frame_t *frame = &frames->retire_conn_id_frame;
    if (varint_write_stream(&pointer, &l, frame->retire_seq_number)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case PATH_CHALLENGE:
  case PATH_RESPONSE: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    path_challenge_frame_t *frame = &frames->path_challenge_frame;
    if (stream_write_n_bytes(&pointer, &l, frame->data,
                             sizeof(frame->data) / sizeof(*frame->data))) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case CONNECTION_CLOSE: {
    conn_close_frame_t *frame = &frames->conn_close_frame;
    varint_t masked_type = type | 0x01;
    if (varint_write_stream(&pointer, &l, masked_type)) {
      return FRAME_ENCODING_ERROR;
    }

    if (varint_write_stream(&pointer, &l, frame->error_code)) {
      return FRAME_ENCODING_ERROR;
    }
    if (!frame->application_error_flag) {
      if (varint_write_stream(&pointer, &l, frame->frame_type)) {
        return FRAME_ENCODING_ERROR;
      }
    }

    if (varint_write_stream(&pointer, &l, frame->reason_phrase_length)) {
      return FRAME_ENCODING_ERROR;
    }

    if (stream_write_n_bytes(&pointer, &l, frame->reason_phrase,
                             frame->reason_phrase_length)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case HANDSHAKE_DONE: {
    if (varint_write_stream(&pointer, &l, type)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  default:
    return FRAME_ENCODING_ERROR;
    break;
  }

  *buffer = pointer;
  *left = l;
  return 0;
}

/*
 * Returns 0 if error has occured
 * Or exactly the length of the given frame
 *
 */
size_t get_frame_length(quic_frame_types_t type, quic_frames_t *frames) {
  size_t length = 0;
  switch (type) {
  case PADDING:
    break;
  case PING:
    get_varint_length(
        type, &length,
        NULL); // in case of variable type (like STREAM) being one length with
               // some flags and another length with others
    break;
  case ACK: {
    get_varint_length(type, &length, NULL);
    ack_frame_t *frame = &frames->ack_frame;
    get_varint_length(frame->largest_acked, &length, NULL);
    get_varint_length(frame->ack_delay, &length, NULL);
    if (frame->ack_range_length % 2 != 1) {
      return 0;
    }
    varint_t ack_range_count = frame->ack_range_length / 2;
    get_varint_length(ack_range_count, &length, NULL);
    get_varint_array_length(frame->ack_ranges, frame->ack_range_length,
                            &length);
    break;
  }
  case ACK_ECN: {
    get_varint_length(type, &length, NULL);
    ack_ecn_frame_t *frame = &frames->ack_ecn_frame;
    get_varint_length(frame->ack_frame.largest_acked, &length, NULL);
    get_varint_length(frame->ack_frame.ack_delay, &length, NULL);
    if (frame->ack_frame.ack_range_length % 2 != 1) {
      return 0;
    }
    varint_t ack_range_count = frame->ack_frame.ack_range_length / 2;
    get_varint_length(ack_range_count, &length, NULL);
    get_varint_array_length(frame->ack_frame.ack_ranges,
                            frame->ack_frame.ack_range_length, &length);

    get_varint_length(frame->ecn0_count, &length, NULL);
    get_varint_length(frame->ecn1_count, &length, NULL);
    get_varint_length(frame->ecn_ce_count, &length, NULL);
    break;
  }
  case RESET_STREAM: {
    get_varint_length(type, &length, NULL);
    reset_stream_frame_t *frame = &frames->reset_stream_frame;
    get_varint_length(frame->stream_id, &length, NULL);
    get_varint_length(frame->application_error_code, &length, NULL);
    get_varint_length(frame->final_size, &length, NULL);
    break;
  }
  case STOP_SENDING: {
    get_varint_length(type, &length, NULL);
    stop_sending_frame_t *frame = &frames->stop_sending_frame;
    get_varint_length(frame->stream_id, &length, NULL);
    get_varint_length(frame->application_error_code, &length, NULL);
    break;
  }
  case CRYPTO: {
    get_varint_length(type, &length, NULL);
    crypto_frame_t *frame = &frames->crypto_frame;
    if (frame->offset + frame->length > MAX_VARINT) {
      return 0;
    }
    get_varint_length(frame->offset, &length, NULL);
    get_varint_length(frame->length, &length, NULL);
    // only true for non-varint data
    length += frame->length * sizeof(*frame->data);
    break;
  }
  case NEW_TOKEN: {
    get_varint_length(type, &length, NULL);
    new_token_frame_t *frame = &frames->new_token_frame;
    if (frame->token_length == 0) {
      return 0;
    }
    get_varint_length(frame->token_length, &length, NULL);
    length += frame->token_length * sizeof(*frame->token);
    break;
  }
  case STREAM: {
    // we will not change type because regardless of flags its always one size
    get_varint_length(type, &length, NULL);
    stream_frame_t *frame = &frames->stream_frame;
    if (frame->offset_flag) {
      get_varint_length(frame->offset, &length, NULL);
    }
    if (frame->length_flag) {
      get_varint_length(frame->length, &length, NULL);
    }
    // length should always be present even if we don't send it
    length += frame->length * sizeof(*frame->data);

    break;
  }
  case MAX_DATA: {
    get_varint_length(type, &length, NULL);
    max_data_frame_t *frame = &frames->max_data_frame;
    get_varint_length(frame->max_data, &length, NULL);
    break;
  }
  case MAX_STREAM_DATA: {
    get_varint_length(type, &length, NULL);
    max_stream_data_frame_t *frame = &frames->max_stream_data_frame;
    get_varint_length(frame->stream_id, &length, NULL);
    get_varint_length(frame->max_data, &length, NULL);
    break;
  }
  case MAX_STREAMS_BI:
  case MAX_STREAMS_UNI: {
    get_varint_length(type, &length, NULL);
    max_streams_frame_t *frame = &frames->max_streams_frame;
    get_varint_length(frame->max_streams, &length, NULL);
    break;
  }
  case DATA_BLOCKED: {
    get_varint_length(type, &length, NULL);
    data_blocked_frame_t *frame = &frames->data_blocked_frame;
    get_varint_length(frame->max_data, &length, NULL);
    break;
  }
  case STREAM_DATA_BLOCKED: {
    get_varint_length(type, &length, NULL);
    stream_data_blocked_frame_t *frame = &frames->stream_data_blocked_frame;
    get_varint_length(frame->stream_id, &length, NULL);
    get_varint_length(frame->max_data, &length, NULL);
    break;
  }
  case STREAMS_BLOCKED_BI:
  case STREAMS_BLOCKED_UNI: {
    get_varint_length(type, &length, NULL);
    streams_blocked_frame_t *frame = &frames->streams_blocked_frame;
    get_varint_length(frame->max_streams, &length, NULL);
    break;
  }
  case NEW_CONNECTION_ID: {
    get_varint_length(type, &length, NULL);
    new_conn_id_frame_t *frame = &frames->new_conn_id_frame;
    if (frame->retire_prior_to > frame->seq_number ||
        frame->conn_id_length < MIN_CONN_ID_LENGTH ||
        frame->conn_id_length > MAX_CONN_ID_LENGTH) {
      return 0;
    }
    get_varint_length(frame->seq_number, &length, NULL);
    get_varint_length(frame->retire_prior_to, &length, NULL);
    get_varint_length(frame->conn_id_length, &length, NULL);
    length += frame->conn_id_length * sizeof(*frame->conn_id);
    length += sizeof(frame->stateless_reset_token);
    break;
  }
  case RETIRE_CONNECTION_ID: {
    get_varint_length(type, &length, NULL);
    retire_conn_id_frame_t *frame = &frames->retire_conn_id_frame;
    get_varint_length(frame->retire_seq_number, &length, NULL);
    break;
  }
  case PATH_CHALLENGE:
  case PATH_RESPONSE: {
    get_varint_length(type, &length, NULL);
    path_challenge_frame_t *frame = &frames->path_challenge_frame;
    length += sizeof(frame->data);
    break;
  }
  case CONNECTION_CLOSE: {
    get_varint_length(type, &length, NULL);
    conn_close_frame_t *frame = &frames->conn_close_frame;
    get_varint_length(frame->error_code, &length, NULL);
    if (!frame->application_error_flag) {
      get_varint_length(frame->frame_type, &length, NULL);
    }
    get_varint_length(frame->reason_phrase_length, &length, NULL);
    length += frame->reason_phrase_length * sizeof(*frame->reason_phrase);
    break;
  }
  case HANDSHAKE_DONE: {
    get_varint_length(type, &length, NULL);
    break;
  }
  default:
    return FRAME_ENCODING_ERROR;
    break;
  }
  return length;
}

void free_dynamic_frame_info(varint_t type, quic_frames_t *frames) {
  if (frames == NULL)
    return;

  switch (type) {
  case ACK: {
    ack_frame_t *frame = &frames->ack_frame;
    if (frame->ack_ranges != NULL) {
      free(frame->ack_ranges);
      frame->ack_ranges = NULL;
    }
    break;
  }
  case ACK_ECN: {
    ack_ecn_frame_t *frame = &frames->ack_ecn_frame;
    if (frame->ack_frame.ack_ranges != NULL) {

      free(frame->ack_frame.ack_ranges);
      frame->ack_frame.ack_ranges = NULL;
    }
    break;
  }
  case CRYPTO: {
    crypto_frame_t *frame = &frames->crypto_frame;
    if (frame->data != NULL) {
      free(frame->data);
      frame->data = NULL;
    }
    break;
  }
  case NEW_TOKEN: {
    new_token_frame_t *frame = &frames->new_token_frame;
    if (frame->token != NULL) {
      free(frame->token);
      frame->token = NULL;
    }
    break;
  }
  case STREAM ... STREAM + 7: {
    stream_frame_t *frame = &frames->stream_frame;
    if (frame->data != NULL) {
      free(frame->data);
      frame->data = NULL;
    }
    break;
  }
  case CONNECTION_CLOSE ... CONNECTION_CLOSE + 1: {
    stream_frame_t *frame = &frames->stream_frame;
    if (frame->data != NULL) {
      free(frame->data);
      frame->data = NULL;
    }
    break;
  }
  }
}
