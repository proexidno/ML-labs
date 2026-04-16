#ifndef VARINT_H
#define VARINT_H

#include <stddef.h>
#include <stdint.h>

typedef uint64_t varint_t;

#define MAX_VARINT 0x3fffffffffffffff // 2^61 - 1

static const uint64_t VARINT_PREFIXES[] = {
    0x00 << 6,     // length of 1
    0b01 << 14,    // length of 2
    0b10ULL << 30, // length of 4
    0b11ULL << 62  // length of 8
};

int varint_read_stream(uint8_t **buffer, size_t *left, varint_t *dst);
int varint_write_stream(uint8_t **buffer, size_t *left, varint_t src);
int get_varint_length(varint_t src, size_t *length, varint_t *dst);
int get_varint_array_length(varint_t *src, size_t length, size_t *sum);

#endif // VARINT_H
