from sqlalchemy import func, select

from app.models import Order, Status


def open_orders(session, shop_id):
    """Every open, non-empty order of a shop, newest first."""
    statement = (
        select(Order)
        .where(Order.shop_id == shop_id, Order.status == Status.OPEN, Order.total > 0)
        .order_by(Order.placed_at.desc())
    )
    return session.execute(statement).scalars().all()


def count_open(session, shop_id):
    """How many open, non-empty orders a shop has — the same filter, counted."""
    statement = (
        select(func.count())
        .select_from(Order)
        .where(Order.shop_id == shop_id, Order.status == Status.OPEN, Order.total > 0)
    )
    return session.execute(statement).scalar_one()
