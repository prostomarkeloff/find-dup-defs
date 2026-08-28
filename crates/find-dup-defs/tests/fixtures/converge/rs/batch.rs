//! Two batch operations around one session module, written in two vocabularies.
//!
//! They share the module they reach and the shape of what they do with it; almost nothing else.
use crate::session::{open_session, resolve_chat};
use crate::types::{BatchOutcome, MemberId};

pub fn revoke_admins(channel: &str, admins: &[MemberId]) -> BatchOutcome {
    let handle = open_session(channel);
    let target = resolve_chat(&handle, channel);
    let key = target.identifier();
    let done = admins.iter().map(|admin| demote_one(&handle, key, *admin)).collect();
    BatchOutcome::assembled(channel, done)
}

pub fn expel_members(chat: &str, people: &[MemberId]) -> BatchOutcome {
    let session = open_session(chat);
    let room = resolve_chat(&session, chat);
    let id = room.identifier();
    let results = people.iter().map(|person| kick_one(&session, id, *person)).collect();
    BatchOutcome::assembled(chat, results)
}
