from analyst_crew.tools.retry import RetryOutcome, retry_with_backoff


class Transient(Exception):
    pass


class Permanent(Exception):
    pass


def test_retry_with_backoff_succeeds_on_first_attempt_without_sleeping():
    sleeps = []
    calls = []

    def call():
        calls.append(1)
        return "ok"

    result, outcome = retry_with_backoff(
        call, is_transient=lambda exc: True, max_attempts=3, sleep=sleeps.append,
    )
    assert result == "ok"
    assert outcome == RetryOutcome(attempts=1, retries=0)
    assert not sleeps
    assert len(calls) == 1


def test_retry_with_backoff_retries_transient_failures_then_succeeds():
    sleeps = []
    calls = []

    def call():
        calls.append(1)
        if len(calls) < 3:
            raise Transient("temporary")
        return "ok"

    result, outcome = retry_with_backoff(
        call, is_transient=lambda exc: isinstance(exc, Transient),
        max_attempts=3, backoff_seconds=1.0, sleep=sleeps.append,
    )
    assert result == "ok"
    assert outcome == RetryOutcome(attempts=3, retries=2)
    assert sleeps == [1.0, 2.0]  # exponential backoff between attempt 1->2 and 2->3


def test_retry_with_backoff_exhausts_and_reraises_last_transient_exception():
    calls = []

    def call():
        calls.append(1)
        raise Transient("always fails")

    try:
        retry_with_backoff(
            call, is_transient=lambda exc: isinstance(exc, Transient),
            max_attempts=3, backoff_seconds=0, sleep=lambda _: None,
        )
        assert False, "expected Transient to propagate"
    except Transient:
        pass
    assert len(calls) == 3


def test_retry_with_backoff_never_retries_a_non_transient_failure():
    calls = []

    def call():
        calls.append(1)
        raise Permanent("auth failed")

    try:
        retry_with_backoff(
            call, is_transient=lambda exc: isinstance(exc, Transient),
            max_attempts=5, sleep=lambda _: None,
        )
        assert False, "expected Permanent to propagate immediately"
    except Permanent:
        pass
    assert len(calls) == 1
