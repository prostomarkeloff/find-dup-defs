from enum import Enum


class Status(Enum):
    OPEN = "open"
    HELD = "held"
    CLOSED = "closed"


class Carrier(Enum):
    POST = "post"
    COURIER = "courier"
