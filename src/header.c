#include "header.h"
#include "utils.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

int packet_read_stream(uint8_t *buffer, size_t left, quic_headers_t headers) {
  if (headers.version_negotiation_header != NULL) {
    return -1; // Discard
  }

  int exit_code;
  uint8_t fb = 0, fixed_bit;
  if (stream_read_n_bytes(&buffer, &left, &fb, sizeof(fb)))
    return -1;
  ;
  fixed_bit = fb & 0x40;
  if (fixed_bit != 0x40) {
    return -1; // not a quic packet
  }
  left--;
  uint8_t header_type = fb & 0x80;
  switch (header_type) {
  case 0: // short header
    headers.short_header_v1 = calloc(1, sizeof(short_header_v1_t));
    exit_code =
        format_short_header_v1(buffer, left, headers.short_header_v1, fb);
    if (exit_code < 0) {
      free(headers.short_header_v1);
      headers.short_header_v1 = NULL;
    }
    return exit_code;
    break;
  case 0x80: // long header
    exit_code = format_long_header(buffer, left, headers, fb);
    return exit_code;
    break;
  default:
    return -1;
  };
}

int format_short_header_v1(uint8_t *buffer, size_t left,
                           short_header_v1_t *header, uint8_t fb) {
  if ((fb & 0x18) != 0) { // reserved bits
    return -1;
  }

  header->spin_bit = fb & 0x20 >> 5;
  header->key_phase = fb & 0x4 >> 2;              // protected 0x7
  header->packet_number_length = (fb & 0x03) + 1; // protected 0x7

  size_t dst_conn_id_len = 20; // TODO: size of dst_conn_id instead of 20

  if (stream_read_n_bytes(&buffer, &left, header->dst_conn_id,
                          dst_conn_id_len)) {
    return -1;
  }

  header->frames.packet_number = 0;
  if (stream_read_n_bytes(&buffer, &left,
                          (uint8_t *)&header->frames.packet_number,
                          header->packet_number_length)) {
    return -1;
  }

  return 1;
}

int format_long_header(uint8_t *buffer, size_t left, quic_headers_t headers,
                       uint8_t fb) {
  uint32_t version = 0;
  if (stream_read_n_bytes(&buffer, &left, (uint8_t *)&version, sizeof(version)))
    return -1;

  int exit_code = 0;
  switch (version) {
  case 0:
    headers.version_negotiation_header =
        calloc(1, sizeof(version_negotiation_header_t));
    exit_code = format_version_header(buffer, left,
                                      headers.version_negotiation_header, fb);
    if (exit_code < 0) {
      free(headers.version_negotiation_header);
      headers.long_header_v1 = NULL;
    }
    return exit_code;
    break;
  case 1:
    headers.long_header_v1 = calloc(1, sizeof(long_header_v1_t));
    headers.long_header_v1->version = 1;
    exit_code = format_long_header_v1(buffer, left, headers.long_header_v1, fb);
    if (exit_code < 0) {
      free(headers.long_header_v1);
      headers.long_header_v1 = NULL;
    }
    return exit_code;
    break;
  default:
    return -2; // Unsupported version
  }
}

int format_long_header_v1(uint8_t *buffer, size_t left,
                          long_header_v1_t *header, uint8_t fb) {
  header->long_header_type = (fb & 0x30) >> 4;
  switch (header->long_header_type) {
  case QUIC_PACKET_INITIAL:
    break;
  case QUIC_PACKET_0RTT:
    break;
  case QUIC_PACKET_HANDSHAKE:
    break;
  case QUIC_PACKET_RETRY:
    break;
  }
  return 0;
}

int format_version_header(uint8_t *buffer, size_t left,
                          version_negotiation_header_t *header, uint8_t fb) {
  if (fb != 0x80) {
    return -1;
  }

  if (stream_read_n_bytes(&buffer, &left, &header->dst_conn_id_len,
                          sizeof(header->dst_conn_id_len)))
    return -1; // Discard

  if (stream_read_n_bytes(&buffer, &left, (uint8_t *)&header->dst_conn_id,
                          header->dst_conn_id_len))
    return -1; // Discard
  if (stream_read_n_bytes(&buffer, &left, &header->src_conn_id_len,
                          sizeof(header->src_conn_id_len)))
    return -1; // Discard
  if (stream_read_n_bytes(&buffer, &left, (uint8_t *)&header->src_conn_id,
                          header->src_conn_id_len))
    return -1; // Discard

  if (left == 0 || left % sizeof(*header->supported_versions) != 0)
    return -1;

  if (left / sizeof(*header->supported_versions) < 64) {
    header->supported_versions_len = left / sizeof(*header->supported_versions);
  } else {
    header->supported_versions_len = 0;
  }
  header->supported_versions = calloc(header->supported_versions_len,
                                      sizeof(*header->supported_versions));

  size_t curr_len = 0;
  while (1) {
    curr_len++;
    if (stream_read_n_bytes(&buffer, &left,
                            (uint8_t *)&header->supported_versions[curr_len],
                            sizeof(*header->supported_versions))) {
      break;
    }

    if (curr_len >= header->supported_versions_len) {
      header->supported_versions_len *= 2;
      header->supported_versions = realloc(
          header->supported_versions,
          header->supported_versions_len * sizeof(*header->supported_versions));
      if (header->supported_versions == NULL) {
        return -3; // Fatal out of memory
      }
    }
  }

  header->supported_versions_len = curr_len;
  header->supported_versions = realloc(header->supported_versions,
                                       header->supported_versions_len *
                                           sizeof(*header->supported_versions));
  if (header->supported_versions == NULL) {
    return -3; // Fatal out of memory
  }

  return 0;
}
