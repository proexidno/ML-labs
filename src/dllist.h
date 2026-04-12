#ifndef DLLIST_H
#define DLLIST_H
#include <stddef.h>

#define container_of(ptr, type, member)                                        \
  ({                                                                           \
    void *__mptr = (void *)(ptr);                                              \
    ((type *)(__mptr - offsetof(type, member)));                               \
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
void ll_delete(struct linked_list_node *src);

#endif // DLLIST_H
