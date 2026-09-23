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
"""Per-step HTTP latency/error stats, readable mid-run.

`nemo_gym.server_utils.request` is the single chokepoint every async HTTP call goes
through (model server, judge, search providers, internal server-to-server), so it's
instrumented once here instead of in every caller. Steps are keyed by "METHOD host/path"
(query dropped, rollout-id path segment collapsed) so retries of the same logical call
land in the same bucket.

Timing wraps success *and* failure (a request that errors or hangs is still time spent,
and is exactly the thing we're trying to find), and a wedged in-flight request is tracked
separately so it's visible even though it never completes.

This module never raises out of the hot path -- a stats failure must not take down a
rollout.
"""

import atexit
import os
import time
from collections import Counter
from itertools import count
from pathlib import Path
from re import sub as re_sub
from threading import Lock
from typing import Dict, Optional
from urllib.parse import urlsplit

import orjson

from nemo_gym.config_types import ROLLOUT_PATH_PREFIX


# Statuses worth calling out individually in the summary/log -- the ones this eval cares
# about (429 rate limit, 504 gateway timeout, other retryable 5xx). Kept as a local tuple
# rather than importing openai_utils.RETRY_ERROR_CODES to avoid inverting the existing
# server_utils -> openai_utils import direction.
NOTABLE_STATUSES = (429, 500, 502, 503, 504, 520)

DUMP_EVERY_N = 50

_ROLLOUT_SEGMENT_RE = rf"/{ROLLOUT_PATH_PREFIX}/[^/]+"

_lock = Lock()
_durations: Dict[str, list] = {}
_statuses: Dict[str, Counter] = {}
_inflight: Dict[str, Dict[int, float]] = {}
_completed_since_dump = 0
_token_counter = count()


def step_key(method: str, url: str) -> str:
    """ "METHOD host/path" for a request URL: query dropped, rollout-id segment collapsed
    so `/ng-rollout/<id>/v1/responses` and `/v1/responses` share one bucket."""
    parts = urlsplit(url)
    path = re_sub(_ROLLOUT_SEGMENT_RE, "", parts.path) or "/"
    return f"{method.upper()} {parts.netloc}{path}"


def start(key: str) -> tuple:
    """Record a request as in-flight. Returns an opaque token for `finish`."""
    token = next(_token_counter)
    with _lock:
        _inflight.setdefault(key, {})[token] = time.monotonic()
    return key, token, time.monotonic()


def finish(handle: tuple, status: Optional[int]) -> float:
    """Record completion (success or failure) of a request started with `start`.

    Returns the elapsed seconds, so the caller can log it inline.
    """
    key, token, started_at = handle
    elapsed = time.monotonic() - started_at
    global _completed_since_dump
    with _lock:
        _inflight.get(key, {}).pop(token, None)
        _durations.setdefault(key, []).append(elapsed)
        _statuses.setdefault(key, Counter())[status] += 1
        _completed_since_dump += 1
        should_dump = _completed_since_dump >= DUMP_EVERY_N
        if should_dump:
            _completed_since_dump = 0

    if should_dump:
        dump()

    return elapsed


def _percentile(sorted_values: list, pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * pct))
    return sorted_values[idx]


def snapshot() -> Dict[str, dict]:
    """Per-step stats dict, sorted by total time spent (the bottleneck ordering)."""
    now = time.monotonic()
    with _lock:
        keys = set(_durations) | set(_inflight)
        rows = {}
        for key in keys:
            durations = sorted(_durations.get(key, []))
            inflight_starts = list(_inflight.get(key, {}).values())
            rows[key] = {
                "n": len(durations),
                "p50_s": round(_percentile(durations, 0.5), 3),
                "p95_s": round(_percentile(durations, 0.95), 3),
                "max_s": round(durations[-1], 3) if durations else 0.0,
                "total_s": round(sum(durations), 3),
                # orjson (unlike stdlib json) requires string dict keys.
                "statuses": {str(status): count for status, count in _statuses.get(key, {}).items()},
                "inflight": len(inflight_starts),
                "oldest_inflight_s": round(now - min(inflight_starts), 3) if inflight_starts else 0.0,
            }

    return dict(sorted(rows.items(), key=lambda item: item[1]["total_s"], reverse=True))


def _loaded_global_config_dict() -> Optional[dict]:
    """The global config dict, only if it's already been resolved by the run command (or a
    parent process, via env var) -- never triggers parsing (which can read argv / exit the
    process) from this diagnostic side channel."""
    try:
        from nemo_gym import global_config

        if global_config._GLOBAL_CONFIG_DICT is not None:
            return global_config._GLOBAL_CONFIG_DICT
        if os.environ.get(global_config.NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME):
            return global_config.get_global_config_dict()
    except BaseException:
        pass

    return None


def _stats_dir() -> Path:
    env_dir = os.environ.get("NEMO_GYM_HTTP_STATS_DIR")
    if env_dir:
        return Path(env_dir)

    config_dict = _loaded_global_config_dict()
    if config_dict is not None:
        try:
            from nemo_gym.global_config import NEMO_GYM_LOG_DIR_KEY_NAME

            log_dir = config_dict.get(NEMO_GYM_LOG_DIR_KEY_NAME)
            if log_dir:
                return Path(log_dir) / "http_stats"
        except BaseException:
            pass

    return Path(".nemo_gym/http_stats")


def _component_name() -> str:
    config_dict = _loaded_global_config_dict()
    if config_dict is not None:
        try:
            from nemo_gym.global_config import COMPONENT_NAME_KEY_NAME

            name = config_dict.get(COMPONENT_NAME_KEY_NAME)
            if name:
                return str(name)
        except BaseException:
            pass

    return str(os.getpid())


def dump() -> None:
    """Atomically write the current snapshot to `<stats_dir>/<component>-<pid>.json`."""
    try:
        stats_dir = _stats_dir()
        stats_dir.mkdir(parents=True, exist_ok=True)
        out_path = stats_dir / f"{_component_name()}-{os.getpid()}.json"
        tmp_path = out_path.with_suffix(f"{out_path.suffix}.tmp")
        tmp_path.write_bytes(orjson.dumps(snapshot(), option=orjson.OPT_INDENT_2))
        os.replace(tmp_path, out_path)
    except Exception:
        # Stats are a diagnostic side channel; never let them break a live request path.
        pass


def print_summary() -> None:
    rows = snapshot()
    if not rows:
        return

    print("[http_summary] step n p50_s p95_s max_s total_s inflight statuses", flush=True)
    for key, row in rows.items():
        print(
            f"[http_summary] {key} n={row['n']} p50_s={row['p50_s']} p95_s={row['p95_s']} "
            f"max_s={row['max_s']} total_s={row['total_s']} inflight={row['inflight']} "
            f"statuses={row['statuses']}",
            flush=True,
        )


def _on_exit() -> None:
    dump()
    print_summary()


atexit.register(_on_exit)
