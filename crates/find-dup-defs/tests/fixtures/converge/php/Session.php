<?php

// The session API both batch operations reach.

namespace Chat\Session;

abstract class SessionApi
{
    abstract public static function open(string $channel): object;

    abstract public static function resolve(object $session, string $channel): object;

    abstract public static function identifier(object $chat): int;
}
