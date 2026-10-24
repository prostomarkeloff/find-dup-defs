package retry

import (
	"errors"
	"time"
)

var (
	ErrTimeout = errors.New("timeout")
	ErrBusy    = errors.New("busy")
)

// Retryable says whether a failed call is worth another try.
func Retryable(err error) bool {
	return errors.Is(err, ErrTimeout) || errors.Is(err, ErrBusy)
}

// Backoff waits longer on the same two failures — the rule again, written as a switch.
func Backoff(err error, attempt int) time.Duration {
	switch err {
	case ErrTimeout, ErrBusy:
		return time.Duration(attempt) * time.Second
	}
	return 0
}
