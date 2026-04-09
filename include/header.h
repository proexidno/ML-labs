#ifndef HEADER_H
#define HEADER_H
#include "frames.h"
#include "varint.h"
#include <stddef.h>
#include <stdint.h>

#define QUIC_VERSION 0x00000001

typedef enum {
  QUIC_PACKET_INITIAL = 0x00,
  QUIC_PACKET_0RTT = 0x01,
  QUIC_PACKET_HANDSHAKE = 0x02,
  QUIC_PACKET_RETRY = 0x03
} quic_long_header_type_t;

typedef struct {
  uint32_t packet_number; // 8-32 bits length specified in header's form two
                          // least significant bits
  frame_t *frames;        // list of frames
} packets_info_t;

/*
 * Version Negotiation Packet {
 *   Header Form (1) = 1,
 *   Unused (7),
 *   Version (32) = 0,
 *   Destination Connection ID Length (8),
 *   Destination Connection ID (0..2040),
 *   Source Connection ID Length (8),
 *   Source Connection ID (0..2040),
 *   Supported Version (32) ...,
 * }
 */
typedef struct {
  uint8_t dst_conn_id_length; // in bytes
  uint8_t dst_conn_id[255];
  uint8_t src_conn_id_length; // in bytes
  uint8_t src_conn_id[255];
  size_t supported_versions_length;
  uint32_t *supported_versions;
} version_negotiation_header_t;

/*
 * Initial Packet {
 *   Header Form (1) = 1,
 *   Fixed Bit (1) = 1,
 *   Long Packet Type (2) = QUIC_PACKET_INTERVAL,
 *   Reserved Bits (2),
 *   Packet Number Length (2),
 *   Version (32),
 *   Destination Connection ID Length (8),
 *   Destination Connection ID (0..160),
 *   Source Connection ID Length (8),
 *   Source Connection ID (0..160),
 *   Token Length (i),
 *   Token (..),
 *   Length (i),
 *   Packet Number (8..32),
 *   Packet Payload (8..),
 * }
 */
typedef struct {
  uint8_t packet_number_length;

  varint_t token_length;
  uint8_t *token;
  varint_t length;
  packets_info_t frames;
} init_v1_packet_info_t;

/* 0-RTT Packet {
 *   Header Form (1) = 1,
 *   Fixed Bit (1) = 1,
 *   Long Packet Type (2) = QUIC_PACKET_0RTT,
 *   Reserved Bits (2),
 *   Packet Number Length (2),
 *   Version (32),
 *   Destination Connection ID Length (8),
 *   Destination Connection ID (0..160),
 *   Source Connection ID Length (8),
 *   Source Connection ID (0..160),
 *   Length (i),
 *   Packet Number (8..32),
 *   Packet Payload (8..),
 * }
 */
typedef struct {
  uint8_t packet_number_length;

  varint_t token_length;
  uint8_t *token;
  varint_t length;
  packets_info_t frames;
} zero_rtt_v1_packet_info_t;

/*
 * Handshake Packet {
 *   Header Form (1) = 1,
 *   Fixed Bit (1) = 1,
 *   Long Packet Type (2) = QUIC_PACKET_HANDSHAKE,
 *   Reserved Bits (2),
 *   Packet Number Length (2),
 *   Version (32),
 *   Destination Connection ID Length (8),
 *   Destination Connection ID (0..160),
 *   Source Connection ID Length (8),
 *   Source Connection ID (0..160),
 *   Length (i),
 *   Packet Number (8..32),
 *   Packet Payload (8..),
 * }
 */
typedef struct {
  uint8_t packet_number_length;

  varint_t token_length;
  uint8_t *token;
  varint_t length;

  packets_info_t frames;
} handshake_v1_packet_info_t;

/*
 * Retry Packet {
 *   Header Form (1) = 1,
 *   Fixed Bit (1) = 1,
 *   Long Packet Type (2) = QUIC_PACKET_RETRY,
 *   Unused (4),
 *   Version (32),
 *   Destination Connection ID Length (8),
 *   Destination Connection ID (0..160),
 *   Source Connection ID Length (8),
 *   Source Connection ID (0..160),
 *   Retry Token (..),
 *   Retry Integrity Tag (128),
 * }
 */
typedef struct {
  uint64_t retry_token_length;
  uint8_t *retry_token;
  uint8_t retry_integrity_tag[16];
} retry_v1_packet_info_t;

/*
 * 1-RTT Packet {
 *   Header Form (1) = 0,
 *   Fixed Bit (1) = 1,
 *   Spin Bit (1),
 *   Reserved Bits (2),
 *   Key Phase (1),
 *   Packet Number Length (2),
 *   Destination Connection ID (0..160),
 *   Packet Number (8..32),
 *   Packet Payload (8..),
 * }
 */
typedef struct {
  uint8_t spin_bit;
  uint8_t key_phase;
  uint8_t packet_number_length;

  uint8_t dst_conn_id[20];
  packets_info_t frames;
} short_header_v1_t;

/*
 * Long Header Packet {
 *   Header Form (1) = 1,
 *   Fixed Bit (1) = 1,
 *   Long Packet Type (2),
 *   Type-Specific Bits (4),
 *   Version (32),
 *   Destination Connection ID Length (8),
 *   Destination Connection ID (0..160),
 *   Source Connection ID Length (8),
 *   Source Connection ID (0..160),
 *   Type-Specific Payload (..),
 * }
 */
typedef struct {
  quic_long_header_type_t long_header_type;
  uint32_t version;
  uint8_t dst_conn_id_length; // in bytes
  uint8_t dst_conn_id[20];
  uint8_t src_conn_id_length; // in bytes
  uint8_t src_conn_id[20];
  union {
    init_v1_packet_info_t *init_v1_packet;
    zero_rtt_v1_packet_info_t *zero_rtt_v1_packet;
    handshake_v1_packet_info_t *handshake_v1_packet;
    retry_v1_packet_info_t *retry_v1_packet;
  };
} long_header_v1_t;

/*
 * Helper union for packet_read_stream function
 */
typedef union {
  version_negotiation_header_t *version_negotiation_header;
  short_header_v1_t *short_header_v1;
  long_header_v1_t *long_header_v1;
} quic_headers_t;

#define DISCARD_PACKET -1
#define UNSUPPORTED_VERSION -2
#define MALLOC_FAIL -3

int packet_read_stream(uint8_t *buffer, size_t left, quic_headers_t headers);

int read_short_header_v1(uint8_t *buffer, size_t left,
                         short_header_v1_t *header, uint8_t fb);
int read_long_header(uint8_t *buffer, size_t left, quic_headers_t headers,
                     uint8_t fb);

int read_version_header(uint8_t *buffer, size_t left,
                        version_negotiation_header_t *header, uint8_t fb);
int read_long_header_v1(uint8_t *buffer, size_t left, long_header_v1_t *header,
                        uint8_t fb);

int read_init_header_v1(uint8_t *buffer, size_t left,
                        init_v1_packet_info_t *header_info, uint8_t fb);
int read_zero_rtt_header_v1(uint8_t *buffer, size_t left,
                            zero_rtt_v1_packet_info_t *header_info, uint8_t fb);
int read_handshake_header_v1(uint8_t *buffer, size_t left,
                             handshake_v1_packet_info_t *header_info,
                             uint8_t fb);
int read_retry_header_v1(uint8_t *buffer, size_t left,
                         retry_v1_packet_info_t *header_info);

#endif // HEADER_H
