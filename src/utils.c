#include "utils.h"
#include <stdint.h>

/*
 * Needs zeroed dst pointer if count is not full size of the integer
 */
int stream_read_n_bytes(uint8_t **buffer, size_t *left, uint8_t *dst,
                        size_t count) {
  if (*left < count) {
    return 1;
  }

  uint8_t *end = dst + count;
  while (dst != end) {
    *(dst++) = *((*buffer)++);
  }
  (*left) -= count;
  return 0;
}

int stream_write_n_bytes(uint8_t **buffer, size_t *left, uint8_t *src,
                         size_t count) {
  if (*left < count) {
    return 1;
  }

  uint8_t *end = src + count;
  while (src != end) {
    *((*buffer)++) = *(src++);
  }
  (*left) -= count;
  return 0;
}
