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
    *dst = *((*buffer)++);
    dst++;
  }
  (*left) -= count;
  return 0;
}
