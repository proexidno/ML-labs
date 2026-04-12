#ifndef LLIST_H
#define LLIST_H

#include <assert.h>
#include <stdatomic.h>
#include <stddef.h>

#define __same_type(a, b)                                                      \
  __builtin_types_compatible_p(__typeof__(a), __typeof__(b))

#define container_of(ptr, type, member_name)                                   \
  ({                                                                           \
    void *__mptr = (void *)(ptr);                                              \
    static_assert(__same_type(*(ptr), ((type *)0)->member_name) ||             \
                      __same_type(*(ptr), void),                               \
                  "pointer type mismatch in container_of()");                  \
    ((type *)(__mptr - offsetof(type, member_name)));                          \
  })

struct linked_list_node {
  struct linked_list_node *next;
  struct linked_list_node *prev;
};

typedef struct {
  struct linked_list_node *head;
  struct linked_list_node *tail;
} linked_list_t;

void ll_append(linked_list_t *dst, struct linked_list_node *src);
void ll_prepend(linked_list_t *dst, struct linked_list_node *src);
struct linked_list_node *ll_delete(linked_list_t *dst,
                                   struct linked_list_node **src);

#endif // LLIST_H
