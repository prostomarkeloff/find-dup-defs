"""Three guards around one module, one shape, three vocabularies — the family's case.

Every name differs; what they share is the module they reach and the shape of the decision they
make. Renamed, they are copies one callee apart — Type-3 says so; the family rubric names the shape.
"""

from domain.access.plans import plan_for, seat_ceiling, row_ceiling, byte_ceiling


def channel_admits(code, members):
    tier = plan_for(code)
    if tier is None:
        return True
    if members is None:
        return True
    return members <= seat_ceiling(tier)


def import_permits(subtype, lines):
    quota = plan_for(subtype)
    if quota is None:
        return True
    if lines is None:
        return True
    return lines <= row_ceiling(quota)


def upload_allows(product, weight):
    allowance = plan_for(product)
    if allowance is None:
        return True
    if weight is None:
        return True
    return weight <= byte_ceiling(allowance)
