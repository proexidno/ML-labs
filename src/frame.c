#include "frame.h"
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
 * Returns TYPE of the returned frame
 *
 */
int frame_read_stream(uint8_t **buffer, size_t *left, quic_frames_t *frames) {
  varint_t type;
  if (varint_read_stream(buffer, left, &type)) {
    return -1;
  }

  switch (type) {
  case 0x00: // PADDING
    break;
  case 0x01: // PING
    break;
  case 0x02: { // ACK
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
  case 0x03: { // ACK ECN
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
  case 0x04: { // RESET_STREAM
    reset_stream_frame_t *frame = &frames->reset_stream_frame;
    if (varint_read_stream(buffer, left, &frame->stream_id)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->aplication_error_code)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->final_size)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case 0x05: { // STOP_SENDING
    stop_sending_frame_t *frame = &frames->stop_sending_frame;
    if (varint_read_stream(buffer, left, &frame->stream_id)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->aplication_error_code)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case 0x06: { // CRYPTO
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
    if (stream_read_n_bytes(buffer, left, frame->data, frame->length)) {
      free_dynamic_frame_info(type, frames);
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case 0x07: { // NEW_TOKEN
    new_token_frame_t *frame = &frames->new_token_frame;
    if (varint_read_stream(buffer, left, &frame->token_length)) {
      return FRAME_ENCODING_ERROR;
    }
    if (frame->token_length == 0) {
      return FRAME_ENCODING_ERROR;
    }
    frame->token = calloc(frame->token_length, sizeof(*frame->token));
    if (stream_read_n_bytes(buffer, left, frame->token, frame->token_length)) {
      free_dynamic_frame_info(type, frames);
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case 0x08 ... 0x0f: { // STREAM
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
      if (stream_read_n_bytes(buffer, left, frame->data, frame->length)) {
        free_dynamic_frame_info(type, frames);
        return FRAME_ENCODING_ERROR;
      }
    } else {
      frame->data = calloc(*left, sizeof(*frame->data));
      if (stream_read_n_bytes(buffer, left, frame->data, *left)) {
        free_dynamic_frame_info(type, frames);
        return FRAME_ENCODING_ERROR;
      }
    }

    frame->fin_flag = type & 0x01;
    break;
  }
  case 0x10: { // MAX_DATA
    max_data_frame_t *frame = &frames->max_data_frame;
    if (varint_read_stream(buffer, left, &frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case 0x11: { // MAX_STREAM_DATA
    max_stream_data_frame_t *frame = &frames->max_stream_data_frame;
    if (varint_read_stream(buffer, left, &frame->stream_id)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case 0x12 ... 0x13: { // MAX_STREAMS
    max_streams_frame_t *frame = &frames->max_streams_frame;
    if (varint_read_stream(buffer, left, &frame->max_streams)) {
      return FRAME_ENCODING_ERROR;
    }
    if (frame->max_streams > (MAX_VARINT + 1) / 4) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case 0x14: { // DATA_BLOCKED
    data_blocked_frame_t *frame = &frames->data_blocked_frame;
    if (varint_read_stream(buffer, left, &frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    if (varint_read_stream(buffer, left, &frame->max_data)) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case 0x16 ... 0x17: { // STREAMS_BLOCKED
    streams_blocked_frame_t *frame = &frames->streams_blocked_frame_t;
    if (varint_read_stream(buffer, left, &frame->max_streams)) {
      return FRAME_ENCODING_ERROR;
    }
    if (frame->max_streams > (MAX_VARINT + 1) / 4) {
      return STREAM_LIMIT_ERROR;
    }
    break;
  }
  case 0x18: { // NEW_CONNECTION_ID
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

    if (frame->conn_id_length < 1 || frame->conn_id_length > 20) {
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
  case 0x1a ... 0x1b: { // PATH_CHALLENGE and PATH_RESPONSE
    path_challenge_frame_t *frame = &frames->path_challenge_frame;
    if (stream_read_n_bytes(buffer, left, frame->data,
                            sizeof(frame->data) / sizeof(*frame->data))) {
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case 0x1c ... 0x1d: { // CONNECTION_CLOSE
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

    if (stream_read_n_bytes(buffer, left, frame->reason_phrase,
                            frame->reason_phrase_length)) {
      free_dynamic_frame_info(type, frames);
      return FRAME_ENCODING_ERROR;
    }
    break;
  }
  case 0x1e: { // HANDSHAKE_DONE
    break;
  }
  default:
    return FRAME_ENCODING_ERROR;
    break;
  }
  return type;
}

void free_dynamic_frame_info(varint_t type, quic_frames_t *frames) {
  if (frames == NULL)
    return;

  switch (type) {
  case 0x02: { // ACK
    ack_frame_t *frame = &frames->ack_frame;
    if (frame->ack_ranges != NULL) {
      free(frame->ack_ranges);
      frame->ack_ranges = NULL;
    }
    break;
  }
  case 0x03: { // ACK ECN
    ack_ecn_frame_t *frame = &frames->ack_ecn_frame;
    if (frame->ack_frame.ack_ranges != NULL) {

      free(frame->ack_frame.ack_ranges);
      frame->ack_frame.ack_ranges = NULL;
    }
    break;
  }
  case 0x06: { // CRYPTO
    crypto_frame_t *frame = &frames->crypto_frame;
    if (frame->data != NULL) {
      free(frame->data);
      frame->data = NULL;
    }
    break;
  }
  case 0x07: { // NEW_TOKEN
    new_token_frame_t *frame = &frames->new_token_frame;
    if (frame->token != NULL) {
      free(frame->token);
      frame->token = NULL;
    }
    break;
  }
  case 0x08 ... 0x0f: { // STREAM
    stream_frame_t *frame = &frames->stream_frame;
    if (frame->data != NULL) {
      free(frame->data);
      frame->data = NULL;
    }
    break;
  }
  case 0x1c ... 0x1d: { // CONNECTION_CLOSE
    stream_frame_t *frame = &frames->stream_frame;
    if (frame->data != NULL) {
      free(frame->data);
      frame->data = NULL;
    }
    break;
  }
  }
}
