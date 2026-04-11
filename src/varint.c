#include "varint.h"
#include "utils.h"
#include <stddef.h>
#include <stdint.h>
#include <strings.h>

/*
 * buffer changed in-place, left changed in-place, dst is the result
 */
int varint_read_stream(uint8_t **buffer, size_t *left, varint_t *dst) {

  if (*left < 1) {
    return -1;
  }
  size_t length = 1 << (**buffer >> 6); // in bytes

  *dst = 0;
  if (stream_read_n_bytes(buffer, left, ((uint8_t *)dst), length)) {
    return -1;
  }
  *dst = *dst & ~(0b11ULL << ((length * 8) - 2)); // Skipping length bits

  return 0;
}

/*
 * buffer changed in-place, left changed in-place
 */
int varint_write_stream(uint8_t **buffer, size_t *left, varint_t src) {
  size_t length = 0;
  varint_t val = 0;
  if (get_varint_length(src, &length, &val)) {
    return -1;
  }

  if (stream_write_n_bytes(buffer, left, ((uint8_t *)&val) + 8 - length,
                           length)) {
    return -1;
  }

  return length;
}

/*
 * Get varint length and optionally the varint representation
 *
 * Unencoded varint in src
 * Adds varint length to the variable storred in *length
 * Optionally you can pass address where the encoded varint value will be
 * storred (NOTE: dst should be initialized as 0) or pass NULL if not needed
 *
 * Returens 0 if succesful
 * Anything bellow zero should be considered an error
 */
int get_varint_length(varint_t src, size_t *length, varint_t *dst) {
  if (src > MAX_VARINT) {
    return -1;
  }

  uint32_t length_step = (src >= 0x40) + (src >= 0x4000) +
                         (src >= 0x40000000); // avoiding branching
  *length += 1 << length_step;

  if (dst != NULL) {
    *dst = src | VARINT_PREFIXES[length_step];
  }

  return 0;
}

int get_varint_array_length(varint_t *src, size_t length, size_t *sum) {
  varint_t *end = src + length;
  while (src != end) {
    if (*src > MAX_VARINT) {
      return -1;
    }

    uint32_t length_step = (*src >= 0x40) + (*src >= 0x4000) +
                           (*src >= 0x40000000); // avoiding branching
    *sum += 1 << length_step;
    src++;
  }

  return 0;
}
