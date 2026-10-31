package shop.retry;

public final class Code {
    public static final int TIMEOUT = 110;
    public static final int BUSY = 16;
}

/** Whether a failed call is worth another try. */
public class Retry {
    public static boolean retryable(int code) {
        return code == Code.TIMEOUT || code == Code.BUSY;
    }

    /** Waits longer on the same two failures — the rule again, written as a switch. */
    public static int backoff(int code, int attempt) {
        switch (code) {
            case Code.TIMEOUT:
            case Code.BUSY:
                return attempt * 1000;
            default:
                return 0;
        }
    }

    public static boolean exhausted(int attempt, int budget) {
        return attempt >= budget;
    }

    public static int capped(int delay, int max) {
        if (delay > max) {
            return max;
        }
        return delay;
    }

    public static boolean atLeast(int attempt, int floor) {
        return attempt < floor;
    }

    public static int remaining(int budget, int spent) {
        return budget - spent;
    }

    public static boolean outOf(int budget, int spent) {
        return spent >= budget;
    }

    public static int firstOf(int[] set) {
        return set[0];
    }

    public static boolean blank(int[] set) {
        return set.length == 0;
    }
}
