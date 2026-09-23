# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json

import pytest

from nemo_gym import http_stats


@pytest.fixture(autouse=True)
def _clean_state():
    """Each test gets an empty stats store, and cleans up after itself."""
    http_stats._durations.clear()
    http_stats._statuses.clear()
    http_stats._inflight.clear()
    http_stats._completed_since_dump = 0
    yield
    http_stats._durations.clear()
    http_stats._statuses.clear()
    http_stats._inflight.clear()
    http_stats._completed_since_dump = 0


def test_step_key_strips_query_and_rollout_segment():
    assert http_stats.step_key("post", "http://localhost:1234/v1/responses?x=1") == "POST localhost:1234/v1/responses"
    assert (
        http_stats.step_key("POST", "http://localhost:1234/ng-rollout/abc-123/v1/responses")
        == "POST localhost:1234/v1/responses"
    )


def test_finish_records_duration_and_status():
    handle = http_stats.start("POST host/path")
    elapsed = http_stats.finish(handle, status=200)
    assert elapsed >= 0

    rows = http_stats.snapshot()
    row = rows["POST host/path"]
    assert row["n"] == 1
    assert row["statuses"] == {"200": 1}
    assert row["inflight"] == 0
    assert row["max_s"] >= 0
    assert row["p50_s"] >= 0


def test_snapshot_tracks_in_flight_separately_from_completed():
    key = "POST host/slow"
    in_flight_handle = http_stats.start(key)  # never finished

    done_handle = http_stats.start(key)
    http_stats.finish(done_handle, status=504)

    row = http_stats.snapshot()[key]
    assert row["n"] == 1
    assert row["statuses"] == {"504": 1}
    assert row["inflight"] == 1
    assert row["oldest_inflight_s"] >= 0

    # clean up the still-open handle so it doesn't leak into other tests
    http_stats.finish(in_flight_handle, status=None)


def test_snapshot_orders_steps_by_total_time_spent():
    fast_key = "POST host/fast"
    slow_key = "POST host/slow"

    http_stats.finish(http_stats.start(fast_key), status=200)

    slow_handle = http_stats.start(slow_key)
    import time

    time.sleep(0.02)
    http_stats.finish(slow_handle, status=200)

    ordered_keys = list(http_stats.snapshot().keys())
    assert ordered_keys.index(slow_key) < ordered_keys.index(fast_key)


def test_dump_writes_valid_json(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_HTTP_STATS_DIR", str(tmp_path))
    http_stats.finish(http_stats.start("POST host/path"), status=429)

    http_stats.dump()

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["POST host/path"]["statuses"] == {"429": 1}
