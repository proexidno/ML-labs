#include "varint.h"
#include "utils.h"

/*
 * buffer changed in-place, left changed in-place, dst is the result
 */
int varint_read_stream(uint8_t **buffer, size_t *left, varint_t *dst) {

  if (*left < 1) {
    return 1;
  }
  size_t length = 1 << (**buffer >> 6); // in bytes

  *dst = 0;
  if (stream_read_n_bytes(buffer, left, ((uint8_t *)dst), length)) {
    return 1;
  }
  *dst = *dst & ~(0b11ULL << ((length * 8) - 2)); // Skipping length bits

  return 0;
}

/*
 * buffer changed in-place, left changed in-place
 */
int varint_write_stream(uint8_t **buffer, size_t *left, varint_t src) {
  if (*left < 1) {
    return -1;
  }
  size_t length;
  varint_t val = src;
  if (src < 0x40) {
    length = 1;
  } else if (src < 0x4000) {
    val |= 0b01 << 14;
    length = 2;
  } else if (src < 0x40000000) {
    val |= 0b10 << 30;
    length = 4;
  } else if (src <= MAX_VARINT) {
    val |= 0b11ULL << 62;
    length = 8;
  } else {
    // Larger than 2^62 - 1
    return -1;
  }
  if (length > *left) {
    return -1;
  }

  // network order (in case 8 all cases run; case 4 skips 8; case 2 skips 4, 8;)
  switch (length) {
  case 8:
    (*buffer)[7] = (uint8_t)val;
    (*buffer)[6] = (uint8_t)(val >> 8);
    (*buffer)[5] = (uint8_t)(val >> 16);
    (*buffer)[4] = (uint8_t)(val >> 24);
    val >>= 32;
  case 4:
    (*buffer)[3] = (uint8_t)val;
    (*buffer)[2] = (uint8_t)(val >> 8);
    val >>= 16;
  case 2:
    (*buffer)[1] = (uint8_t)val;
    (*buffer)[0] = (uint8_t)(val >> 8);
    val >>= 8;
  case 1:
    (*buffer)[0] = (uint8_t)val;
  }
  *buffer += length;
  *left -= length;
  return length;
}
