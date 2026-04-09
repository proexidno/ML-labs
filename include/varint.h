#ifndef VARINT_H
#define VARINT_H

#include <stddef.h>
#include <stdint.h>

typedef uint64_t varint_t;

#define MAX_VARINT 0x3fffffffffffffff

int varint_read_stream(uint8_t **buffer, size_t *left, uint64_t *dst);
int varint_write_stream(uint8_t **buffer, size_t *left, uint64_t src);

#endif // VARINT_H
