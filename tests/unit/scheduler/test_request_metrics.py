from unittest.mock import patch

import pytest

from forgeserve.scheduler.request import RequestState


def test_time_to_first_token():
    request = RequestState(
        request_id="test-1",
        prompt="Hello",
        max_new_tokens=10,
        created_at=10.0,
    )

    assert request.time_to_first_token_ms is None

    with patch(
        "forgeserve.scheduler.request.time.perf_counter",
        side_effect=[10.1, 10.25],
    ):
        request.mark_running()
        request.mark_first_token()

    assert request.time_to_first_token_ms == pytest.approx(250.0)