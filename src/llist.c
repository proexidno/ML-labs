#include "llist.h"
#include <stdatomic.h>

/*
 * Append at the tail of the ll
 */
void ll_append(linked_list_t *dst, struct linked_list_node *src) {
  src->next = NULL;
  if (dst->tail == NULL) {
    dst->tail = src;
    src->prev = NULL;
  } else {
    dst->tail->next = src;
    src->prev = dst->tail;
  }
}

/*
 * Prepend to the head of the ll
 */
void ll_prepend(linked_list_t *dst, struct linked_list_node *src) {
  src->prev = NULL;
  src->next = dst->head;
  dst->head = src;
  if (dst->tail == NULL) {
    dst->tail = src;
  }
}

/*
 * Does not free actual container of the ll
 */
struct linked_list_node *ll_delete(linked_list_t *dst,
                                   struct linked_list_node **src) {
  struct linked_list_node *curr_node = *src;
  if (curr_node->next != NULL) {
    curr_node->next->prev = curr_node->prev;
  }
  if (curr_node->prev != NULL) {
    curr_node->prev->next = curr_node->next;
  }
  if (src == &dst->head) {
    dst->head = dst->head->next;
  } else if (src == &dst->tail) {
    dst->tail = dst->tail->prev;
  }
  return curr_node;
}
