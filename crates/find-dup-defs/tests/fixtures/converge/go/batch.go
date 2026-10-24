// Two batch operations around one session package, written in two vocabularies.
//
// They share the package they reach and the shape of what they do with it; almost nothing else.
package batch

import "example.com/chat/session"

func RevokeAdmins(channel string, admins []MemberID) Outcome {
	handle := session.Open(channel)
	target := session.Resolve(handle, channel)
	key := target.Identifier()
	done := make([]Result, 0, len(admins))
	for _, admin := range admins {
		done = append(done, demoteOne(handle, key, admin))
	}
	return Assembled(channel, done)
}

func ExpelMembers(chat string, people []MemberID) Outcome {
	sess := session.Open(chat)
	room := session.Resolve(sess, chat)
	id := room.Identifier()
	results := make([]Result, 0, len(people))
	for _, person := range people {
		results = append(results, kickOne(sess, id, person))
	}
	return Assembled(chat, results)
}
