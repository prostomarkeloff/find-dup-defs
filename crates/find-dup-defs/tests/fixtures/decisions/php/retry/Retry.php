<?php

namespace Shop\Retry;

final class Code
{
    public const TIMEOUT = 110;
    public const BUSY = 16;
}

/** Whether a failed call is worth another try. */
function retryable(int $code): bool
{
    return in_array($code, [Code::TIMEOUT, Code::BUSY], true);
}

/** Waits longer on the same two failures — the rule again, written as a switch. */
function backoff(int $code, int $attempt): int
{
    switch ($code) {
        case Code::TIMEOUT:
        case Code::BUSY:
            return $attempt * 1000;
    }
    return 0;
}
