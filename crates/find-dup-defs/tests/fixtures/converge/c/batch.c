/* Two batch operations around one session API, written in two vocabularies.
 *
 * They share the header they reach and the shape of what they do with it; almost nothing else. */
#include <stdlib.h>
#include "session.h"

struct outcome revoke_admins(const char *channel, const member_id *admins, int n)
{
    struct session *handle = session_open(channel);
    struct chat *target = session_resolve(handle, channel);
    long key = chat_identifier(target);
    struct result *done = calloc(n, sizeof(*done));
    for (int i = 0; i < n; i++)
        done[i] = demote_one(handle, key, admins[i]);
    return assembled(channel, done, n);
}

struct outcome expel_members(const char *chat, const member_id *people, int count)
{
    struct session *sess = session_open(chat);
    struct chat *room = session_resolve(sess, chat);
    long id = chat_identifier(room);
    struct result *results = calloc(count, sizeof(*results));
    for (int k = 0; k < count; k++)
        results[k] = kick_one(sess, id, people[k]);
    return assembled(chat, results, count);
}
