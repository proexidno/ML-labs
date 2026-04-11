#include "header.h"
#include "errors.h"
#include "utils.h"
#include "varint.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/*
 * Reads QUIC packet from given buffer
 * Changes buffer and left in-place
 *
 * Headers should be initialized as NULL
 * To free header, call free_header with return value and headers
 * You should not free headers yourself
 *
 * Returns negative value if error has occured
 *
 * Returns 0 if version negotiation packet is read
 * Returns 1 if short header v1 is read
 * Returns 2 if long  header v1 is read
 */
int header_read_stream(uint8_t **buffer, size_t *left,
                       quic_headers_t *headers) {
  if (headers->version_negotiation_header != NULL) {
    return DISCARD_PACKET;
  }

  int exit_code;
  uint8_t fb = 0, fixed_bit;
  if (stream_read_n_bytes(buffer, left, &fb, sizeof(fb)))
    return DISCARD_PACKET;

  fixed_bit = fb & 0x40;
  if (fixed_bit != 0x40) {
    return DISCARD_PACKET; // not a quic packet
  }

  uint8_t header_type = fb & 0x80;
  switch (header_type) {
  case 0: // short header
    headers->short_header_v1 = calloc(1, sizeof(short_header_v1_t));
    exit_code =
        read_short_header_v1(buffer, left, headers->short_header_v1, fb);
    if (exit_code < 0) {
      free_header(1, headers);
      return exit_code;
    }
    return 1;
    break;
  case 0x80: // long header
    exit_code = read_long_header(buffer, left, headers, fb);
    return exit_code;
    break;
  default:
    return DISCARD_PACKET;
  };
}

int read_short_header_v1(uint8_t **buffer, size_t *left,
                         short_header_v1_t *header, uint8_t fb) {
  header->spin_bit = fb & 0x20 >> 5;
  size_t dst_conn_id_len = 20; // TODO: size of dst_conn_id instead of 20

  if (stream_read_n_bytes(buffer, left, header->dst_conn_id, dst_conn_id_len)) {
    return DISCARD_PACKET;
  }

  uint8_t *pointer = *buffer + 4;
  size_t l = *left;
  const size_t decrypt_sample_length = 16;
  uint8_t decrypt_sample[decrypt_sample_length];

  if (stream_read_n_bytes(&pointer, &l, decrypt_sample,
                          decrypt_sample_length)) {
    return DISCARD_PACKET;
  }

  // TODO: Generate mask using decrypt sample

  uint8_t mask[5] = {0}; // protection mask
  fb ^= mask[0] & 0x1f;

  uint8_t reserved = (fb & 0x18);
  if (reserved != 0) { // reserved bits
    return DISCARD_PACKET;
  }

  header->key_phase = fb & 0x4 >> 2;
  header->packet_number_length = (fb & 0x03) + 1;

  l = *left; // saving for mask so that we never make l < packet number length
  if (stream_read_n_bytes(buffer, left, (uint8_t *)&header->packet_number,
                          header->packet_number_length)) {
    return DISCARD_PACKET;
  }

  // packet number decryption
  uint32_t packet_number_mask = 0;
  pointer = &mask[1];
  stream_read_n_bytes(&pointer, &l, (uint8_t *)&packet_number_mask,
                      header->packet_number_length);
  header->packet_number ^= packet_number_mask;

  return 1;
}

int read_long_header(uint8_t **buffer, size_t *left, quic_headers_t *headers,
                     uint8_t fb) {
  uint32_t version = 0;
  if (stream_read_n_bytes(buffer, left, (uint8_t *)&version, sizeof(version)))
    return DISCARD_PACKET;

  int exit_code = 0;
  switch (version) {
  case 0:
    headers->version_negotiation_header =
        calloc(1, sizeof(version_negotiation_header_t));
    exit_code = read_version_header(buffer, left,
                                    headers->version_negotiation_header, fb);
    if (exit_code < 0) {
      free_header(0, headers);
      return exit_code;
    }
    return 0;
    break;
  case 1:
    headers->long_header_v1 = calloc(1, sizeof(long_header_v1_t));
    exit_code = read_long_header_v1(buffer, left, headers->long_header_v1, fb);
    if (exit_code < 0) {
      free_header(1, headers);
      return exit_code;
    }
    return 2;
    break;
  default:
    return UNSUPPORTED_VERSION;
  }
}

int read_long_header_v1(uint8_t **buffer, size_t *left,
                        long_header_v1_t *header, uint8_t fb) {
  if (stream_read_n_bytes(buffer, left, &header->dst_conn_id_length,
                          sizeof(header->dst_conn_id_length)))
    return DISCARD_PACKET;
  if (header->dst_conn_id_length >= 20) {
    return DISCARD_PACKET;
  }

  if (stream_read_n_bytes(buffer, left, header->dst_conn_id,
                          header->dst_conn_id_length))
    return DISCARD_PACKET;

  if (stream_read_n_bytes(buffer, left, &header->src_conn_id_length,
                          sizeof(header->src_conn_id_length)))
    return DISCARD_PACKET;
  if (header->src_conn_id_length >= 20) {
    return DISCARD_PACKET;
  }

  if (stream_read_n_bytes(buffer, left, header->src_conn_id,
                          header->src_conn_id_length))
    return DISCARD_PACKET;

  int exit_code;

  header->long_header_type = (fb & 0x30) >> 4;
  switch (header->long_header_type) {
  case QUIC_PACKET_INITIAL:
    header->init_v1_packet = calloc(1, sizeof(init_v1_packet_info_t));
    exit_code = read_init_header_v1(buffer, left, header->init_v1_packet, fb);
    if (exit_code < 0) {
      free(header->init_v1_packet);
      header->handshake_v1_packet = NULL;
      return exit_code;
    }
    break;
  case QUIC_PACKET_0RTT:
    header->zero_rtt_v1_packet = calloc(1, sizeof(zero_rtt_v1_packet_info_t));
    exit_code =
        read_zero_rtt_header_v1(buffer, left, header->zero_rtt_v1_packet, fb);
    if (exit_code < 0) {
      free(header->zero_rtt_v1_packet);
      header->handshake_v1_packet = NULL;
      return exit_code;
    }
    break;
  case QUIC_PACKET_HANDSHAKE:
    header->handshake_v1_packet = calloc(1, sizeof(handshake_v1_packet_info_t));
    exit_code =
        read_handshake_header_v1(buffer, left, header->handshake_v1_packet, fb);
    if (exit_code < 0) {
      free(header->handshake_v1_packet);
      header->handshake_v1_packet = NULL;
      return exit_code;
    }
    break;
  case QUIC_PACKET_RETRY:
    header->retry_v1_packet = calloc(1, sizeof(retry_v1_packet_info_t));
    exit_code = read_retry_header_v1(buffer, left, header->retry_v1_packet);
    if (exit_code < 0) {
      free(header->retry_v1_packet);
      header->retry_v1_packet = NULL;
      return exit_code;
    }
    break;
  }
  return 1;
}

int read_version_header(uint8_t **buffer, size_t *left,
                        version_negotiation_header_t *header, uint8_t fb) {

  if (stream_read_n_bytes(buffer, left, &header->dst_conn_id_length,
                          sizeof(header->dst_conn_id_length)))
    return DISCARD_PACKET;

  if (stream_read_n_bytes(buffer, left, header->dst_conn_id,
                          header->dst_conn_id_length))
    return DISCARD_PACKET;
  if (stream_read_n_bytes(buffer, left, &header->src_conn_id_length,
                          sizeof(header->src_conn_id_length)))
    return DISCARD_PACKET;
  if (stream_read_n_bytes(buffer, left, header->src_conn_id,
                          header->src_conn_id_length))
    return DISCARD_PACKET;

  if (*left == 0 || *left % sizeof(*header->supported_versions) != 0)
    return DISCARD_PACKET;

  if (*left / sizeof(*header->supported_versions) < 64) {
    header->supported_versions_length =
        *left / sizeof(*header->supported_versions);
  } else {
    header->supported_versions_length = 64;
  }
  header->supported_versions = calloc(header->supported_versions_length,
                                      sizeof(*header->supported_versions));

  size_t curr_len = 0;
  while (1) {
    curr_len++;
    if (stream_read_n_bytes(buffer, left,
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
        return MALLOC_FAIL;
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

int read_init_header_v1(uint8_t **buffer, size_t *left,
                        init_v1_packet_info_t *header_info, uint8_t fb) {

  varint_read_stream(buffer, left, &header_info->token_length);
  header_info->token =
      calloc(header_info->token_length, sizeof(*header_info->token));

  if (stream_read_n_bytes(buffer, left, header_info->token,
                          header_info->token_length)) {
    free(header_info->token);
    header_info->token = NULL;
    return DISCARD_PACKET;
  }

  if (varint_read_stream(buffer, left, &header_info->length)) {
    free(header_info->token);
    header_info->token = NULL;
    return DISCARD_PACKET;
  }

  uint8_t *pointer = *buffer + 4;
  size_t l = *left;
  const size_t decrypt_sample_length = 16;
  uint8_t decrypt_sample[decrypt_sample_length];

  if (stream_read_n_bytes(&pointer, &l, decrypt_sample,
                          decrypt_sample_length)) {
    free(header_info->token);
    header_info->token = NULL;
    return DISCARD_PACKET;
  }

  // TODO: Generate mask using decrypt sample

  uint8_t mask[5] = {0}; // protection mask
  fb ^= mask[0] & 0x0f;

  uint8_t reserved = (fb & 0x0c);
  if (reserved != 0) { // reserved bits
    free(header_info->token);
    header_info->token = NULL;
    return DISCARD_PACKET;
  }

  header_info->packet_number_length = (fb & 0x03) + 1;

  l = *left; // saving for mask so that we never make l < packet number length
  if (stream_read_n_bytes(buffer, left, (uint8_t *)&header_info->packet_number,
                          header_info->packet_number_length)) {
    free(header_info->token);
    header_info->token = NULL;
    return DISCARD_PACKET;
  }

  // packet number decryption
  uint32_t packet_number_mask = 0;
  pointer = &mask[1];
  stream_read_n_bytes(&pointer, &l, (uint8_t *)&packet_number_mask,
                      header_info->packet_number_length);
  header_info->packet_number ^= packet_number_mask;

  return 0;
}

int read_zero_rtt_header_v1(uint8_t **buffer, size_t *left,
                            zero_rtt_v1_packet_info_t *header_info,
                            uint8_t fb) {
  if (varint_read_stream(buffer, left, &header_info->length)) {
    return DISCARD_PACKET;
  }

  uint8_t *pointer = *buffer + 4;
  size_t l = *left;
  const size_t decrypt_sample_length = 16;
  uint8_t decrypt_sample[decrypt_sample_length];

  if (stream_read_n_bytes(&pointer, &l, decrypt_sample,
                          decrypt_sample_length)) {
    return DISCARD_PACKET;
  }

  // TODO: Generate mask using decrypt sample

  uint8_t mask[5] = {0}; // protection mask
  fb ^= mask[0] & 0x0f;

  uint8_t reserved = (fb & 0x0c);
  if (reserved != 0) { // reserved bits
    return DISCARD_PACKET;
  }

  header_info->packet_number_length = (fb & 0x03) + 1;

  l = *left; // saving for mask so that we never make l < packet number length
  if (stream_read_n_bytes(buffer, left, (uint8_t *)&header_info->packet_number,
                          header_info->packet_number_length)) {
    return DISCARD_PACKET;
  }

  // packet number decryption
  uint32_t packet_number_mask = 0;
  pointer = &mask[1];
  stream_read_n_bytes(&pointer, &l, (uint8_t *)&packet_number_mask,
                      header_info->packet_number_length);
  header_info->packet_number ^= packet_number_mask;

  return 0;
}
int read_handshake_header_v1(uint8_t **buffer, size_t *left,
                             handshake_v1_packet_info_t *header_info,
                             uint8_t fb) {
  if (varint_read_stream(buffer, left, &header_info->length)) {
    return DISCARD_PACKET;
  }

  uint8_t *pointer = *buffer + 4;
  size_t l = *left;
  const size_t decrypt_sample_length = 16;
  uint8_t decrypt_sample[decrypt_sample_length];

  if (stream_read_n_bytes(&pointer, &l, decrypt_sample,
                          decrypt_sample_length)) {
    return DISCARD_PACKET;
  }

  // TODO: Generate mask using decrypt sample

  uint8_t mask[5] = {0}; // protection mask
  fb ^= mask[0] & 0x0f;

  uint8_t reserved = (fb & 0x0c);
  if (reserved != 0) { // reserved bits
    return DISCARD_PACKET;
  }

  header_info->packet_number_length = (fb & 0x03) + 1;

  l = *left; // saving for mask so that we never make l < packet number length
  if (stream_read_n_bytes(buffer, left, (uint8_t *)&header_info->packet_number,
                          header_info->packet_number_length)) {
    return DISCARD_PACKET;
  }

  // packet number decryption
  uint32_t packet_number_mask = 0;
  pointer = &mask[1];
  stream_read_n_bytes(&pointer, &l, (uint8_t *)&packet_number_mask,
                      header_info->packet_number_length);
  header_info->packet_number ^= packet_number_mask;

  return 0;
}
int read_retry_header_v1(uint8_t **buffer, size_t *left,
                         retry_v1_packet_info_t *header_info) {
  size_t integrity_tag_length = sizeof(header_info->retry_integrity_tag) /
                                sizeof(*header_info->retry_integrity_tag);
  if (*left <= integrity_tag_length) {
    return DISCARD_PACKET;
  }
  header_info->retry_token_length = *left - integrity_tag_length;
  header_info->retry_token = calloc(header_info->retry_token_length,
                                    sizeof(*header_info->retry_token));
  if (stream_read_n_bytes(buffer, left, header_info->retry_token,
                          integrity_tag_length)) {
    free(header_info->retry_token);
    header_info->retry_token = NULL;
    return DISCARD_PACKET;
  }

  if (stream_read_n_bytes(buffer, left, header_info->retry_integrity_tag,
                          integrity_tag_length)) {
    free(header_info->retry_token);
    header_info->retry_token = NULL;
    return DISCARD_PACKET;
  }
  return 0;
}

void free_header(int type, quic_headers_t *headers) {
  if (type < 0 || headers == NULL || headers->freable == NULL)
    return;

  switch (type) {
  case 0: { // VERSION_NEGOTIATION
    version_negotiation_header_t *header = headers->version_negotiation_header;
    if (header->supported_versions != NULL) {
      free(header->supported_versions);
    }
    break;
  }
  case 1: // SHORT_HEADER_V1
    break;
  case 2: { // LONG_HEADER_V1
    long_header_v1_t *header = headers->long_header_v1;
    switch (header->long_header_type) {
    case QUIC_PACKET_INITIAL: {
      init_v1_packet_info_t *header_info = header->init_v1_packet;
      if (header_info->token != NULL) {
        free(header_info->token);
        header_info->token = NULL;
      }
      break;
    }
    case QUIC_PACKET_0RTT: {
      zero_rtt_v1_packet_info_t *header_info = header->zero_rtt_v1_packet;
      if (header_info->token != NULL) {
        free(header_info->token);
        header_info->token = NULL;
      }
      break;
    }
    case QUIC_PACKET_HANDSHAKE: {
      handshake_v1_packet_info_t *header_info = header->handshake_v1_packet;
      if (header_info->token != NULL) {
        free(header_info->token);
        header_info->token = NULL;
      }
      break;
    }
    case QUIC_PACKET_RETRY: {
      retry_v1_packet_info_t *header_info = header->retry_v1_packet;
      if (header_info->retry_token != NULL) {
        free(header_info->retry_token);
        header_info->retry_token = NULL;
      }
      break;
    }
    }
    break;
  }
  }

  free(headers->freable);
  headers->freable = NULL;
}
