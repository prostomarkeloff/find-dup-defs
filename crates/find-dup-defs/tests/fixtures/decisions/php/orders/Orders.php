<?php

namespace Shop\Orders;

use Shop\Models\Order;
use Shop\Models\Status;

/** How checkout refuses an order that is not open and paid. */
function refuse(Order $order): void
{
    if ($order->status !== Status::OPEN) {
        throw new OrderClosed('order closed');
    }
    if (!$order->paid) {
        throw new OrderUnpaid('order unpaid');
    }
}

function total(Order $order): int
{
    $sum = 0;
    foreach ($order->lines as $line) {
        if ($line->quantity > 0) {
            $sum += $line->price;
        }
    }
    return $sum;
}

function label(Order $order): string
{
    if ($order->number !== null) {
        return $order->number;
    }
    return 'draft';
}
