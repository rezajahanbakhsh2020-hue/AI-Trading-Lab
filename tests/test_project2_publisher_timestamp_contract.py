import datetime

from src.integration.project2_publisher import Project2Publisher


def test_publisher_does_not_treat_300_seconds_as_signal_validity_ttl():
    publisher = Project2Publisher(
        enabled=False,
        max_age_seconds=300,
    )

    ts = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(hours=4)
    ).isoformat()

    valid, status, reason = publisher.validate_timestamp(ts)

    assert valid is True
    assert status is None
    assert reason is None


def test_publisher_still_rejects_future_event_timestamp():
    publisher = Project2Publisher(enabled=False)

    ts = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(seconds=10)
    ).isoformat()

    valid, status, reason = publisher.validate_timestamp(ts)

    assert valid is False
    assert status == "INVALID_RESPONSE"
    assert "Future event timestamp" in reason
