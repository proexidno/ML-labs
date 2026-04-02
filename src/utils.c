#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <utils.h>

/*
 * buffer changed in-place, left changed in-place, dst is the result
 */
void varint_read_stream(uint8_t **buffer, size_t *left, uint64_t *dst) {

  assert(*left > 0);
  size_t length = 1 << (**buffer >> 6);

  if (length > *left) {
    return;
  }

  *dst = **buffer & 0x3f;
  uint8_t *end = *buffer + length - 1;
  while (*buffer != end) {
    *dst = *dst << 8 | *++(*buffer);
  }
  *left -= length;
}

/*
 * buffer changed in-place, left changed in-place
 */
size_t varint_write_stream(uint8_t **buffer, size_t *left, uint64_t src) {
  assert(*left > 0);
  size_t length;
  uint64_t val = src;
  if (src < 0x40) {
    length = 1;
  } else if (src < 0x4000) {
    val |= 0b01 << 14;
    length = 2;
  } else if (src < 0x40000000) {
    val |= 0b10 << 30;
    length = 4;
  } else if (src < 0x4000000000000000) {
    val |= 0b11ULL << 62;
    length = 8;
  } else {
    // Larger than 2^62 - 1
    return 0;
  }
  if (length > *left) {
    return 0;
  }

  switch (length) {
  case 8:
    (*buffer)[7] = (uint8_t)val;
    (*buffer)[6] = (uint8_t)(val >> 8);
    (*buffer)[5] = (uint8_t)(val >> 16);
    (*buffer)[4] = (uint8_t)(val >> 24);
    (*buffer)[3] = (uint8_t)(val >> 32);
    (*buffer)[2] = (uint8_t)(val >> 40);
    (*buffer)[1] = (uint8_t)(val >> 48);
    (*buffer)[0] = (uint8_t)(val >> 56);
    break;
  case 4:
    (*buffer)[3] = (uint8_t)val;
    (*buffer)[2] = (uint8_t)(val >> 8);
    (*buffer)[1] = (uint8_t)(val >> 16);
    (*buffer)[0] = (uint8_t)(val >> 24);
    break;
  case 2:
    (*buffer)[1] = (uint8_t)val;
    (*buffer)[0] = (uint8_t)(val >> 8);
    break;
  case 1:
    (*buffer)[0] = (uint8_t)val;
    break;
  }
  *buffer += length;
  *left -= length;
  return length;
}
