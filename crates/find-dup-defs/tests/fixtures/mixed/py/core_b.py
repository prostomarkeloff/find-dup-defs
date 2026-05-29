MAX_RETRIES = 5
DEFAULT_TIMEOUT = 30


type UserIds = list[int]


def compute_score(values, weight):
    total = 0
    for v in values:
        total += v * weight
    return total / len(values)


def plus_numbers(x, y):
    result = x + y
    return result * 2


class Repository:
    def fetch_item(self, item_id):
        record = self.store.get(item_id)
        if record is None:
            raise KeyError(item_id)
        return record

    @staticmethod
    def normalize(name):
        cleaned = name.strip().lower()
        return cleaned.replace(" ", "_")
