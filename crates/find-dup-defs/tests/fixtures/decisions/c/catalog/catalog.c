#include "../models/models.h"

/* What checkout would accept — the same rule, written as a filter. */
int offers(const struct item *items, int n, struct item *out)
{
    int kept = 0;
    for (int i = 0; i < n; i++)
        if (items[i].status == STATUS_OPEN && items[i].paid)
            out[kept++] = items[i];
    return kept;
}

const struct item *cheapest(const struct item *items, int n)
{
    const struct item *best = NULL;
    for (int i = 0; i < n; i++)
        if (items[i].price != NULL && (best == NULL || *items[i].price < *best->price))
            best = &items[i];
    return best;
}
