"""Two budgets written independently around one module — the subject anchor's case."""

from infra.quota.repo import attempts_prune, attempt_record, source_key
from infra.quota.types import HOUR_SECONDS


def enforce_submit_budget(source, settings):
    started = now(UTC) - timedelta(seconds=HOUR_SECONDS)
    with write_session() as session:
        attempts_prune(session, scope=SUBMIT, older_than=started)
        used = attempt_record(session, key=source_key(SUBMIT, source), started_at=started)
    if used > settings.submit_limit_per_hour:
        raise SubmitRateLimited
    return used


def apply_source_limit(source, plan):
    at = utcnow()
    window = at - timedelta(hours=1)
    with write_session() as session:
        attempts_prune(session, scope=APPLY, older_than=window)
        seen = attempt_record(session, key=source_key(APPLY, source), started_at=window)
    if seen > plan.source_limit_per_hour:
        raise ApplyRateLimited
    return seen
