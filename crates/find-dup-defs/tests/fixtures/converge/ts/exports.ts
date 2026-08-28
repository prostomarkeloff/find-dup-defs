// Two exports around one module, written in two vocabularies.
import { buildStreamingResponse } from "@/shared/export";
import { personRow, eventRow } from "@/shared/rows";

export function exportContacts(items: unknown[], title: string, locale: string) {
  const shaped = items.map((item) => personRow(item));
  const stamped = shaped.map((entry) => ({ ...entry, at: Date.now() }));
  return buildStreamingResponse(stamped, title, locale, "contacts");
}

export function exportTimeline(records: unknown[], caption: string, lang: string) {
  const converted = records.map((record) => eventRow(record));
  const marked = converted.map((row) => ({ ...row, at: Date.now() }));
  return buildStreamingResponse(marked, caption, lang, "events");
}
