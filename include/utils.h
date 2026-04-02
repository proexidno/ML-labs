#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>

void varint_read_stream(uint8_t **buffer, size_t *left, uint64_t *dst);
size_t varint_write_stream(uint8_t **buffer, size_t *left, uint64_t src);

#endif // UTILS_H
