// Two batch operations around one session API, written in two vocabularies.
//
// They share the API they reach and the shape of what they do with it; almost nothing else.

package chat.batch;

import chat.session.SessionApi;

public class Batch {
    public static String[] revokeAdmins(String channel, String[] admins) {
        Object handle = SessionApi.open(channel);
        Object target = SessionApi.resolve(handle, channel);
        int key = SessionApi.identifier(target);
        String[] done = new String[admins.length];
        for (int i = 0; i < admins.length; i++) {
            done[i] = demoteOne(handle, key, admins[i]);
        }
        return assembled(channel, done);
    }

    public static String[] expelMembers(String chat, String[] people) {
        Object sess = SessionApi.open(chat);
        Object room = SessionApi.resolve(sess, chat);
        int id = SessionApi.identifier(room);
        String[] results = new String[people.length];
        for (int k = 0; k < people.length; k++) {
            results[k] = kickOne(sess, id, people[k]);
        }
        return assembled(chat, results);
    }

    private static String demoteOne(Object handle, int key, String admin) {
        return admin;
    }

    private static String kickOne(Object sess, int id, String person) {
        return person;
    }

    private static String[] assembled(String channel, String[] done) {
        return done;
    }
}
