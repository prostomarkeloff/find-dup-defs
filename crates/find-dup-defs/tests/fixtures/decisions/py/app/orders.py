from app.models import Status


class Closed(Exception):
    pass


class Unpaid(Exception):
    pass


def refuse(order):
    """Checkout refuses an order that is not open and paid."""
    if order.status != Status.OPEN:
        raise Closed
    if not order.paid:
        raise Unpaid
    return order


def total(order):
    return sum(line.price for line in order.lines if line.quantity > 0)


def label(order):
    return order.number if order.number is not None else "draft"
