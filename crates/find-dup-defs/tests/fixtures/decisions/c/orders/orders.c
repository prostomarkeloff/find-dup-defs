#include "../models/models.h"

#define ERR_CLOSED (-2)
#define ERR_UNPAID (-3)

/* How checkout refuses an order that is not open and paid. */
int refuse(const struct order *order)
{
    if (order->status != STATUS_OPEN)
        return ERR_CLOSED;
    if (!order->paid)
        return ERR_UNPAID;
    return 0;
}

int total(const struct order *order)
{
    int sum = 0;
    for (int i = 0; i < order->line_count; i++)
        if (order->lines[i].quantity > 0)
            sum += order->lines[i].price;
    return sum;
}

const char *label(const struct order *order)
{
    if (order->number != NULL)
        return order->number;
    return "draft";
}
