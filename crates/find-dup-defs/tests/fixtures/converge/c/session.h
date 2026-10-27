/* The session API both batch operations reach. */
#ifndef SESSION_H
#define SESSION_H

struct session;
struct chat;

struct session *session_open(const char *channel);
struct chat *session_resolve(struct session *s, const char *channel);
long chat_identifier(const struct chat *c);

#endif
