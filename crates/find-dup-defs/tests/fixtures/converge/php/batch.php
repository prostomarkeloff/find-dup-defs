<?php

// Two batch operations around one session API, written in two vocabularies.
//
// They share the API they reach and the shape of what they do with it; almost nothing else.

namespace Chat\Batch;

use Chat\Session\SessionApi;

function revoke_admins(string $channel, array $admins): array
{
    $handle = SessionApi::open($channel);
    $target = SessionApi::resolve($handle, $channel);
    $key = SessionApi::identifier($target);
    $done = array_fill(0, count($admins), null);
    foreach ($admins as $i => $admin) {
        $done[$i] = demote_one($handle, $key, $admin);
    }
    return assembled($channel, $done);
}

function expel_members(string $chat, array $people): array
{
    $sess = SessionApi::open($chat);
    $room = SessionApi::resolve($sess, $chat);
    $id = SessionApi::identifier($room);
    $results = array_fill(0, count($people), null);
    foreach ($people as $k => $person) {
        $results[$k] = kick_one($sess, $id, $person);
    }
    return assembled($chat, $results);
}
