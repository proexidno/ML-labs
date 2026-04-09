#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>

int stream_read_n_bytes(uint8_t **buffer, size_t *left, uint8_t *dst,
                        size_t count);

#endif // UTILS_H
