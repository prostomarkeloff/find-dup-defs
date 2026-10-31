// The session API both batch operations reach.

package chat.session;

public abstract class SessionApi {
    public static Object open(String channel) {
        return new Object();
    }

    public static Object resolve(Object session, String channel) {
        return session;
    }

    public static int identifier(Object chat) {
        return chat.hashCode();
    }
}
