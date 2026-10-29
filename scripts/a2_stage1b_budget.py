"""Shared admission ledger for the two registered FWE parts and evaluations."""
from datetime import datetime, timezone
import math


def budget(spent, preparation, requested, deadline, now=None):
    if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0
               for v in (spent, preparation, requested)) or requested <= 0:
        raise ValueError('Invalid Stage1b budget')
    end = datetime.fromisoformat(deadline)
    if end.tzinfo is None:
        raise ValueError('Explicit deadline timezone required')
    now = now or datetime.now(timezone.utc)
    total = spent + preparation + requested + 660
    if total > 24 * 3600 or (end - now).total_seconds() < requested + 660:
        raise ValueError('Stage1b 24h cap or continuation deadline exceeded')
    return dict(prior_seconds=spent, preparation_seconds=preparation,
                timeout_seconds=requested, cleanup_reserve_seconds=660,
                stage_hard_seconds=24*3600, report_due=total > 20*3600,
                deadline=end.isoformat())
