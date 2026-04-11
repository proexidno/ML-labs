#ifndef ERRORS_H
#define ERRORS_H

typedef enum {
  DISCARD_PACKET = -1,
  UNSUPPORTED_VERSION = -2,
  MALLOC_FAIL = -3,
  FRAME_ENCODING_ERROR = -4,
  STREAM_LIMIT_ERROR = -5,
  PROTOCOL_VIOLATION = -6,
} internal_errors_t;

#endif // ERRORS_H
