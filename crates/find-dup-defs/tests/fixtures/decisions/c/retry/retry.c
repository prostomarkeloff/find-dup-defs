#include <errno.h>

/* Whether a failed call is worth another try. */
int retryable(int err)
{
    return err == ETIMEDOUT || err == EBUSY;
}

/* Waits longer on the same two failures — the rule again, written as a switch. */
long backoff(int err, int attempt)
{
    switch (err) {
    case ETIMEDOUT:
    case EBUSY:
        return attempt * 1000L;
    }
    return 0;
}
