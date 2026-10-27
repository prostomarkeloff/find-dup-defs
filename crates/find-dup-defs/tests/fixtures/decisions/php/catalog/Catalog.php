<?php

namespace Shop\Catalog;

use Shop\Models\Status;

/** What checkout would accept — the same rule, written as a filter. */
function offers(array $items): array
{
    $out = [];
    foreach ($items as $item) {
        if (Status::OPEN === $item->status && $item->paid) {
            $out[] = $item;
        }
    }
    return $out;
}

function cheapest(array $items): ?object
{
    $best = null;
    foreach ($items as $item) {
        if ($item->price !== null && ($best === null || $item->price < $best->price)) {
            $best = $item;
        }
    }
    return $best;
}
