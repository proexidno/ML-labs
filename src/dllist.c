#include "dllist.h"

/*
 * Append at the tail of the ll
 */
void ll_append(linked_list_t *dst, struct linked_list_node *src) {
  src->prev = dst->tail;
  src->next = dst->tail->next;

  dst->tail->next = src;
}

/*
 * Prepend to the head of the ll
 */
void ll_prepend(linked_list_t *dst, struct linked_list_node *src) {
  src->next = dst->head;
  src->prev = dst->head->prev;

  dst->head->prev = src;
}

/*
 * Does not free actual container of the ll
 *
 */
void ll_delete(struct linked_list_node *src) {
  src->next->prev = src->prev;
  src->prev->next = src->next;
}
