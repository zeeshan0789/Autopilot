import pytest
from redis.exceptions import RedisError

from app.core.quota import COUNTER_TTL_SECONDS, DailyQuota

DAY = 86400
T0 = 1_700_000_000  # a moment in a UTC day
TOMORROW = (T0 // DAY + 1) * DAY + 5


class FakePipe:
    def __init__(self, redis):
        self.redis = redis
        self.ops = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def incr(self, key):
        self.ops.append(("incr", key))

    def expire(self, key, seconds):
        self.ops.append(("expire", key, seconds))

    async def execute(self):
        if self.redis.down:
            raise RedisError("down")
        results = []
        for op in self.ops:
            if op[0] == "incr":
                self.redis.counts[op[1]] = self.redis.counts.get(op[1], 0) + 1
                results.append(self.redis.counts[op[1]])
            else:
                self.redis.ttls[op[1]] = op[2]
                results.append(True)
        return results


class FakeRedis:
    def __init__(self):
        self.counts, self.ttls, self.down = {}, {}, False

    def pipeline(self, transaction=True):
        return FakePipe(self)


def make(now=T0):
    clock = {"now": now}
    quota = DailyQuota("redis://unused", clock=lambda: clock["now"])
    quota.client = FakeRedis()
    return quota, clock


def run(coro):
    import asyncio

    return asyncio.run(coro)


def test_requests_are_counted_until_the_limit():
    quota, _ = make()
    first = run(quota.consume(7, 2))
    second = run(quota.consume(7, 2))
    third = run(quota.consume(7, 2))
    assert (first.allowed, first.remaining) == (True, 1)
    assert (second.allowed, second.remaining) == (True, 0)
    assert (third.allowed, third.remaining) == (False, 0)
    assert third.limit == 2


def test_every_user_has_a_counter_of_their_own():
    quota, _ = make()
    run(quota.consume(1, 1))
    assert run(quota.consume(1, 1)).allowed is False
    assert run(quota.consume(2, 1)).allowed is True


def test_the_counter_restarts_the_next_utc_day():
    quota, clock = make()
    run(quota.consume(1, 1))
    assert run(quota.consume(1, 1)).allowed is False
    clock["now"] = TOMORROW
    assert run(quota.consume(1, 1)).allowed is True


def test_the_retry_time_is_the_time_until_utc_midnight():
    quota, _ = make()
    result = run(quota.consume(1, 5))
    assert result.retry_after == DAY - (T0 % DAY)


def test_the_counter_expires_by_itself():
    quota, _ = make()
    run(quota.consume(1, 5))
    assert list(quota.client.ttls.values()) == [COUNTER_TTL_SECONDS]


def test_a_redis_failure_lets_the_request_through():
    quota, _ = make()
    quota.client.down = True
    result = run(quota.consume(1, 5))
    assert result.allowed is True and result.limit == 5

def test_the_platform_counter_is_shared_by_everyone_and_restarts_each_day():
    quota, clock = make()
    assert run(quota.consume_platform(2)).allowed is True
    assert run(quota.consume_platform(2)).allowed is True
    assert run(quota.consume_platform(2)).allowed is False
    assert run(quota.consume(1, 10)).allowed is True  # a user's own counter is another counter
    clock["now"] = TOMORROW
    assert run(quota.consume_platform(2)).allowed is True    