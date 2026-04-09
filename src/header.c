#include "header.h"
#include "utils.h"
#include "varint.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/*
 * Reads QUIC packet from given buffer
 *
 * Returns negative value if error occured
 *
 * Returns 0 if version negotiation packet is read
 * Returns 1 if short header v1 is read
 * Returns 2 if long  header v1 is read
 */
int packet_read_stream(uint8_t *buffer, size_t left, quic_headers_t headers) {
  if (headers.version_negotiation_header != NULL) {
    return DISCARD_PACKET; // Discard
  }

  int exit_code;
  uint8_t fb = 0, fixed_bit;
  if (stream_read_n_bytes(&buffer, &left, &fb, sizeof(fb)))
    return DISCARD_PACKET;
  ;
  fixed_bit = fb & 0x40;
  if (fixed_bit != 0x40) {
    return DISCARD_PACKET; // not a quic packet
  }
  left--;
  uint8_t header_type = fb & 0x80;
  switch (header_type) {
  case 0: // short header
    headers.short_header_v1 = calloc(1, sizeof(short_header_v1_t));
    exit_code =
        parse_short_header_v1(buffer, left, headers.short_header_v1, fb);
    if (exit_code < 0) {
      free(headers.short_header_v1);
      headers.short_header_v1 = NULL;
      return exit_code;
    }
    return 1;
    break;
  case 0x80: // long header
    exit_code = parse_long_header(buffer, left, headers, fb);
    return exit_code;
    break;
  default:
    return DISCARD_PACKET;
  };
}

int parse_short_header_v1(uint8_t *buffer, size_t left,
                          short_header_v1_t *header, uint8_t fb) {
  uint8_t reserved = (fb & 0x18); // they are protected
  if (reserved != 0) {            // reserved bits
    return DISCARD_PACKET;
  }

  header->spin_bit = fb & 0x20 >> 5;
  header->key_phase = fb & 0x4 >> 2;              // protected 0x7
  header->packet_number_length = (fb & 0x03) + 1; // protected 0x7

  size_t dst_conn_id_len = 20; // TODO: size of dst_conn_id instead of 20

  if (stream_read_n_bytes(&buffer, &left, header->dst_conn_id,
                          dst_conn_id_len)) {
    return DISCARD_PACKET;
  }

  header->frames.packet_number = 0;
  if (stream_read_n_bytes(&buffer, &left,
                          (uint8_t *)&header->frames.packet_number,
                          header->packet_number_length)) {
    return DISCARD_PACKET;
  }
  // TODO: read frames

  return 1;
}

int parse_long_header(uint8_t *buffer, size_t left, quic_headers_t headers,
                      uint8_t fb) {
  uint32_t version = 0;
  if (stream_read_n_bytes(&buffer, &left, (uint8_t *)&version, sizeof(version)))
    return DISCARD_PACKET;

  int exit_code = 0;
  switch (version) {
  case 0:
    headers.version_negotiation_header =
        calloc(1, sizeof(version_negotiation_header_t));
    exit_code = parse_version_header(buffer, left,
                                     headers.version_negotiation_header, fb);
    if (exit_code < 0) {
      free(headers.version_negotiation_header);
      headers.long_header_v1 = NULL;
      return exit_code;
    }
    return 0;
    break;
  case 1:
    headers.long_header_v1 = calloc(1, sizeof(long_header_v1_t));
    headers.long_header_v1->version = 1;
    exit_code = parse_long_header_v1(buffer, left, headers.long_header_v1, fb);
    if (exit_code < 0) {
      free(headers.long_header_v1);
      headers.long_header_v1 = NULL;
      return exit_code;
    }
    return 1;
    break;
  default:
    return UNSUPPORTED_VERSION; // Unsupported version
  }
}

int parse_long_header_v1(uint8_t *buffer, size_t left, long_header_v1_t *header,
                         uint8_t fb) {

  if (stream_read_n_bytes(&buffer, &left, &header->dst_conn_id_length,
                          sizeof(header->dst_conn_id_length)))
    return DISCARD_PACKET;
  if (header->dst_conn_id_length >= 20) {
    return DISCARD_PACKET;
  }

  if (stream_read_n_bytes(&buffer, &left, (uint8_t *)&header->dst_conn_id,
                          header->dst_conn_id_length))
    return DISCARD_PACKET;

  if (stream_read_n_bytes(&buffer, &left, &header->src_conn_id_length,
                          sizeof(header->src_conn_id_length)))
    return DISCARD_PACKET;
  if (header->src_conn_id_length >= 20) {
    return DISCARD_PACKET;
  }

  if (stream_read_n_bytes(&buffer, &left, (uint8_t *)&header->src_conn_id,
                          header->src_conn_id_length))
    return DISCARD_PACKET;

  int exit_code;

  header->long_header_type = (fb & 0x30) >> 4;
  switch (header->long_header_type) {
  case QUIC_PACKET_INITIAL:
    header->init_v1_packet = calloc(1, sizeof(init_v1_packet_info_t));
    header->init_v1_packet->packet_number_length = (fb & 0x03) + 1;
    exit_code = parse_init_header_v1(buffer, left, header->init_v1_packet);
    if (exit_code < 0) {
      free(header->init_v1_packet);
      header->handshake_v1_packet = NULL;
      return exit_code;
    }
    break;
  case QUIC_PACKET_0RTT:
    header->zero_rtt_v1_packet = calloc(1, sizeof(zero_rtt_v1_packet_info_t));
    header->zero_rtt_v1_packet->packet_number_length = (fb & 0x03) + 1;
    exit_code =
        parse_zero_rtt_header_v1(buffer, left, header->zero_rtt_v1_packet);
    if (exit_code < 0) {
      free(header->zero_rtt_v1_packet);
      header->handshake_v1_packet = NULL;
      return exit_code;
    }
    break;
  case QUIC_PACKET_HANDSHAKE:
    header->handshake_v1_packet = calloc(1, sizeof(handshake_v1_packet_info_t));
    header->handshake_v1_packet->packet_number_length = (fb & 0x03) + 1;
    exit_code =
        parse_handshake_header_v1(buffer, left, header->handshake_v1_packet);
    if (exit_code < 0) {
      free(header->handshake_v1_packet);
      header->handshake_v1_packet = NULL;
      return exit_code;
    }
    break;
  case QUIC_PACKET_RETRY:
    header->retry_v1_packet = calloc(1, sizeof(retry_v1_packet_info_t));
    exit_code = parse_retry_header_v1(buffer, left, header->retry_v1_packet);
    if (exit_code < 0) {
      free(header->retry_v1_packet);
      header->retry_v1_packet = NULL;
      return exit_code;
    }
    break;
  }
  return 1;
}

int parse_version_header(uint8_t *buffer, size_t left,
                         version_negotiation_header_t *header, uint8_t fb) {
  if (fb != 0x80) {
    return DISCARD_PACKET;
  }

  if (stream_read_n_bytes(&buffer, &left, &header->dst_conn_id_length,
                          sizeof(header->dst_conn_id_length)))
    return DISCARD_PACKET;

  if (stream_read_n_bytes(&buffer, &left, (uint8_t *)&header->dst_conn_id,
                          header->dst_conn_id_length))
    return DISCARD_PACKET;
  if (stream_read_n_bytes(&buffer, &left, &header->src_conn_id_length,
                          sizeof(header->src_conn_id_length)))
    return DISCARD_PACKET;
  if (stream_read_n_bytes(&buffer, &left, (uint8_t *)&header->src_conn_id,
                          header->src_conn_id_length))
    return DISCARD_PACKET;

  if (left == 0 || left % sizeof(*header->supported_versions) != 0)
    return DISCARD_PACKET;

  if (left / sizeof(*header->supported_versions) < 64) {
    header->supported_versions_length =
        left / sizeof(*header->supported_versions);
  } else {
    header->supported_versions_length = 0;
  }
  header->supported_versions = calloc(header->supported_versions_length,
                                      sizeof(*header->supported_versions));

  size_t curr_len = 0;
  while (1) {
    curr_len++;
    if (stream_read_n_bytes(&buffer, &left,
                            (uint8_t *)&header->supported_versions[curr_len],
                            sizeof(*header->supported_versions))) {
      break;
    }

    if (curr_len >= header->supported_versions_length) {
      header->supported_versions_length *= 2;
      header->supported_versions = realloc(
          header->supported_versions, header->supported_versions_length *
                                          sizeof(*header->supported_versions));
      if (header->supported_versions == NULL) {
        return MALLOC_FAIL; // Fatal out of memory
      }
    }
  }

  header->supported_versions_length = curr_len;
  header->supported_versions = realloc(header->supported_versions,
                                       header->supported_versions_length *
                                           sizeof(*header->supported_versions));
  if (header->supported_versions == NULL) {
    return MALLOC_FAIL;
  }

  return 0;
}

int parse_init_header_v1(uint8_t *buffer, size_t left,
                         init_v1_packet_info_t *header_info, uint8_t fb) {
  uint8_t reserved = (fb & 0x18); // they are protected
  if (reserved != 0) {            // reserved bits
    return DISCARD_PACKET;
  }

  varint_read_stream(&buffer, &left, &header_info->token_length);
  header_info->token =
      calloc(header_info->token_length, sizeof(*header_info->token));

  if (stream_read_n_bytes(&buffer, &left, header_info->token,
                          header_info->token_length)) {
    free(header_info->token);
    header_info->token = NULL;
    return DISCARD_PACKET;
  }

  if (varint_read_stream(&buffer, &left, &header_info->length)) {
    return DISCARD_PACKET;
  }

  header_info->frames.packet_number = 0;
  if (stream_read_n_bytes(&buffer, &left,
                          (uint8_t *)&header_info->frames.packet_number,
                          header_info->packet_number_length)) {
    free(header_info->token);
    header_info->token = NULL;
    return DISCARD_PACKET;
  }
  // TODO: read frames
  return 0;
}

int parse_zero_rtt_header_v1(uint8_t *buffer, size_t left,
                             zero_rtt_v1_packet_info_t *header_info,
                             uint8_t fb) {
  uint8_t reserved = (fb & 0x18); // they are protected
  if (reserved != 0) {            // reserved bits
    return DISCARD_PACKET;
  }
  if (varint_read_stream(&buffer, &left, &header_info->length)) {
    return DISCARD_PACKET;
  }

  header_info->frames.packet_number = 0;
  if (stream_read_n_bytes(&buffer, &left,
                          (uint8_t *)&header_info->frames.packet_number,
                          header_info->packet_number_length)) {
    return DISCARD_PACKET;
  }
  // TODO: read frames
  return 0;
}
int parse_handshake_header_v1(uint8_t *buffer, size_t left,
                              handshake_v1_packet_info_t *header_info,
                              uint8_t fb) {
  uint8_t reserved = (fb & 0x18); // they are protected
  if (reserved != 0) {            // reserved bits
    return DISCARD_PACKET;
  }
  if (varint_read_stream(&buffer, &left, &header_info->length)) {
    return DISCARD_PACKET;
  }

  header_info->frames.packet_number = 0;
  if (stream_read_n_bytes(&buffer, &left,
                          (uint8_t *)&header_info->frames.packet_number,
                          header_info->packet_number_length)) {
    return DISCARD_PACKET;
  }
  // TODO: read frames
  return 0;
}
int parse_retry_header_v1(uint8_t *buffer, size_t left,
                          retry_v1_packet_info_t *header_info) {
  if (left <= 16) {
    return DISCARD_PACKET;
  }
  size_t integrity_tag_length = sizeof(header_info->retry_integrity_tag) /
                                sizeof(*header_info->retry_integrity_tag);
  header_info->retry_token_length = left - integrity_tag_length;
  header_info->retry_token =
      calloc(header_info->retry_token_length, sizeof(uint8_t));
  if (stream_read_n_bytes(&buffer, &left, header_info->retry_token,
                          integrity_tag_length)) {
    free(header_info->retry_token);
    header_info->retry_token = NULL;
    return DISCARD_PACKET;
  }

  if (stream_read_n_bytes(&buffer, &left,
                          (uint8_t *)header_info->retry_integrity_tag,
                          integrity_tag_length)) {
    free(header_info->retry_token);
    header_info->retry_token = NULL;
    return DISCARD_PACKET;
  }
  return 0;
}
