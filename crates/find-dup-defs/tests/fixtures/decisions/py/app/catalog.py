from app.models import Status


def offers(items):
    """The storefront lists what checkout would accept — the same rule, written as a filter."""
    return [item for item in items if item.status == Status.OPEN and item.paid]


def cheapest(items):
    return min((i for i in items if i.price is not None), key=lambda i: i.price, default=None)
