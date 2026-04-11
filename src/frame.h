#ifndef FRAME_H
#define FRAME_H

#include "varint.h"
#include <stddef.h>
#include <stdint.h>

typedef enum {
  PADDING = 0x00,
  PING = 0x01,
  ACK = 0x02,
  ACK_ECN = 0x03,
  RESET_STREAM = 0x04,
  STOP_SENDING = 0x05,
  CRYPTO = 0x06,
  NEW_TOKEN = 0x07,
  STREAM = 0x08,
  MAX_DATA = 0x10,
  MAX_STREAM_DATA = 0x11,
  MAX_STREAMS_BI = 0x12,
  MAX_STREAMS_UNI = 0x13,
  DATA_BLOCKED = 0x14,
  STREAM_DATA_BLOCKED = 0x15,
  STREAMS_BLOCKED_BI = 0x16,
  STREAMS_BLOCKED_UNI = 0x17,
  NEW_CONNECTION_ID = 0x18,
  RETIRE_CONNECTION_ID = 0x19,
  PATH_CHALLENGE = 0x1a,
  PATH_RESPONSE = 0x1b,
  CONNECTION_CLOSE = 0x1c,
  HANDSHAKE_DONE = 0x1e
} quic_frame_types_t;

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
  varint_t application_error_code;
  varint_t final_size;
} reset_stream_frame_t;

typedef struct {
  varint_t stream_id;
  varint_t application_error_code;
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
  varint_t length; // length should always be present
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
  streams_blocked_frame_t streams_blocked_frame;
  new_conn_id_frame_t new_conn_id_frame;
  retire_conn_id_frame_t retire_conn_id_frame;
  path_challenge_frame_t path_challenge_frame;
  conn_close_frame_t conn_close_frame;

} quic_frames_t;

void free_dynamic_frame_info(varint_t type, quic_frames_t *frames);
int frame_read_stream(uint8_t **buffer, size_t *left, quic_frames_t *frames);
int frame_write_stream(uint8_t **buffer, size_t *left, quic_frame_types_t type,
                       quic_frames_t *frames);
size_t get_frame_length(quic_frame_types_t type, quic_frames_t *frames);

#endif // FRAME_H
