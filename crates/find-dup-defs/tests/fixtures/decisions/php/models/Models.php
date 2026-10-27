<?php

namespace Shop\Models;

final class Status
{
    public const OPEN = 'open';
    public const HELD = 'held';
    public const CLOSED = 'closed';
}

final class Carrier
{
    public const POST = 'post';
    public const COURIER = 'courier';
}

final class Line
{
    public function __construct(public int $price, public int $quantity)
    {
    }
}

final class Order
{
    /** @param Line[] $lines */
    public function __construct(
        public int $shopId,
        public string $status,
        public bool $paid,
        public int $placedAt,
        public int $total,
        public ?string $number,
        public array $lines,
    ) {
    }
}

final class Item
{
    public function __construct(public string $status, public bool $paid, public ?int $price)
    {
    }
}
