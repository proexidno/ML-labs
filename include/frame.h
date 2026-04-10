#ifndef FRAME_H
#define FRAME_H

#include "varint.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
  varint_t largest_acked;
  varint_t ack_delay;
  varint_t ack_range_length;
  varint_t *ack_ranges;
} ack_frame_t;

typedef struct {
  ack_frame_t ack_frame;

  varint_t ecn0_count;
  varint_t ecn1_count;
  varint_t ecn_ce_count;
} ack_ecn_frame_t;

typedef struct {
  varint_t stream_id;
  varint_t aplication_error_code;
  varint_t final_size;
} reset_stream_frame_t;

typedef struct {
  varint_t stream_id;
  varint_t aplication_error_code;
} stop_sending_frame_t;

typedef struct {
  varint_t offset;
  varint_t length;
  uint8_t *data;
} crypto_frame_t;

typedef struct {
  varint_t token_length;
  uint8_t *token;
} new_token_frame_t;

typedef struct {
  uint8_t offset_flag;
  uint8_t length_flag;
  uint8_t fin_flag;

  varint_t stream_id;
  varint_t offset;
  varint_t length;
  uint8_t *data;
} stream_frame_t;

typedef struct {
  varint_t max_data;
} max_data_frame_t;

typedef struct {
  varint_t stream_id;
  varint_t max_data;
} max_stream_data_frame_t;

typedef struct {
  varint_t max_streams;
} max_streams_frame_t;

typedef struct {
  varint_t max_data;
} data_blocked_frame_t;

typedef struct {
  varint_t stream_id;
  varint_t max_data;
} stream_data_blocked_frame_t;

typedef struct {
  varint_t max_streams;
} streams_blocked_frame_t;

typedef struct {
  varint_t seq_number;
  varint_t retire_prior_to;
  uint8_t conn_id_length;
  uint8_t conn_id[20];
  uint8_t stateless_reset_token[16];
} new_conn_id_frame_t;

typedef struct {
  varint_t retire_seq_number;
} retire_conn_id_frame_t;

typedef struct {
  uint8_t data[8];
} path_challenge_frame_t;

typedef struct {
  uint8_t application_error_flag;
  varint_t error_code;
  varint_t
      frame_type; // frame that caused error, only when application error != 0
  varint_t reason_phrase_length;
  uint8_t *reason_phrase;
} conn_close_frame_t;

typedef union {
  ack_frame_t ack_frame;
  ack_ecn_frame_t ack_ecn_frame;
  reset_stream_frame_t reset_stream_frame;
  stop_sending_frame_t stop_sending_frame;
  crypto_frame_t crypto_frame;
  new_token_frame_t new_token_frame;
  stream_frame_t stream_frame;
  max_data_frame_t max_data_frame;
  max_stream_data_frame_t max_stream_data_frame;
  max_streams_frame_t max_streams_frame;
  data_blocked_frame_t data_blocked_frame;
  stream_data_blocked_frame_t stream_data_blocked_frame;
  streams_blocked_frame_t streams_blocked_frame_t;
  new_conn_id_frame_t new_conn_id_frame;
  retire_conn_id_frame_t retire_conn_id_frame_t;
  path_challenge_frame_t path_challenge_frame;
  conn_close_frame_t conn_close_frame;

} quic_frames_t;

void free_dynamic_frame_info(varint_t type, quic_frames_t *frames);
int frame_read_stream(uint8_t **buffer, size_t *left, quic_frames_t *frames);

#endif // FRAME_H
