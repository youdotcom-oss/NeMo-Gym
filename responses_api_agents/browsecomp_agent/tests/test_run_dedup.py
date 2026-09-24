# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""A duplicate /run for a task that's already in flight (e.g. the orchestrator's HTTP
client retrying after a dropped connection, unaware the original request is still being
served) must attach to the ORIGINAL rollout instead of starting a second one from scratch
-- that was silently doubling compute and orphaning one of the two results. But attaching
must not risk waiting forever on a genuinely hung original: past `dedup_wait_timeout_seconds`
it falls back to an independent rollout, same as before this change.
"""

import asyncio
import json as jsonlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.browsecomp_agent.app import (
    BrowsecompAgent,
    BrowsecompAgentConfig,
    BrowsecompAgentRunRequest,
)


def _make_config(**kwargs) -> BrowsecompAgentConfig:
    defaults = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="test_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="test_resources"),
        model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        nudge_steps=False,
    )
    return BrowsecompAgentConfig(**(defaults | kwargs))


def _model_response() -> dict:
    return NeMoGymResponse(
        id="resp_001",
        created_at=0.0,
        model="test_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg_001",
                content=[NeMoGymResponseOutputText(annotations=[], text="Exact Answer: X", type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        usage=NeMoGymResponseUsage(
            input_tokens=0,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=0,
        ),
    ).model_dump()


class _GatedServerClient:
    """Drives /run like the real one, but every /seed_session call blocks on `gate` until
    the test releases it -- so the test can deterministically hold a rollout "in flight"
    for as long as it needs before letting it proceed."""

    def __init__(self, gate: asyncio.Event, reward: float = 1.0):
        self.gate = gate
        self.reward = reward
        self.url_paths_called: list[str] = []
        self._seed_session_calls = 0

    async def post(self, server_name=None, url_path=None, json=None, cookies=None):
        self.url_paths_called.append(url_path)
        http = MagicMock()
        http.cookies = {}
        http.status = 200
        http.ok = True
        http.raise_for_status = MagicMock()
        if url_path == "/seed_session":
            self._seed_session_calls += 1
            if self._seed_session_calls == 1:
                # Only the very first rollout (the one the test means to hang) waits on the
                # gate -- a later, independent fallback rollout must proceed normally.
                await self.gate.wait()
            http.read = AsyncMock(return_value=b"{}")
            http.content.read = AsyncMock(return_value=b"{}")
        elif url_path == "/v1/responses":
            payload = _model_response()
            http.read = AsyncMock(return_value=jsonlib.dumps(payload).encode())
            http.content.read = AsyncMock(return_value=jsonlib.dumps(payload).encode())
        elif url_path == "/verify":
            payload = dict(json or {}) | {"reward": self.reward}
            blob = jsonlib.dumps(payload, default=str).encode()
            http.read = AsyncMock(return_value=blob)
            http.content.read = AsyncMock(return_value=blob)
        else:
            http.read = AsyncMock(return_value=b"{}")
            http.content.read = AsyncMock(return_value=b"{}")
        return http


def _make_agent(client: _GatedServerClient, **config_kwargs) -> BrowsecompAgent:
    server_client = MagicMock(spec=ServerClient)
    server_client.post = client.post
    return BrowsecompAgent(config=_make_config(**config_kwargs), server_client=server_client)


def _run_body(**extra) -> BrowsecompAgentRunRequest:
    return BrowsecompAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Q?"}]},
            "question": "Q?",
            **extra,
        }
    )


def _run(agent: BrowsecompAgent, **extra):
    request_mock = MagicMock()
    request_mock.cookies = {}
    return agent.run(request_mock, _run_body(**extra))


async def test_duplicate_run_attaches_to_in_flight_rollout_instead_of_restarting() -> None:
    gate = asyncio.Event()
    client = _GatedServerClient(gate, reward=1.0)
    agent = _make_agent(client, dedup_wait_timeout_seconds=5.0)

    original = asyncio.create_task(_run(agent))
    await asyncio.sleep(0)  # let it register itself in _inflight_rollouts and block on the gate

    assert agent._inflight_rollouts, "the original rollout must be tracked as in-flight"
    duplicate = asyncio.create_task(_run(agent))
    await asyncio.sleep(0)  # let the duplicate reach the dedup branch and start waiting

    gate.set()  # release the original; the duplicate should now piggyback on its result
    original_result, duplicate_result = await asyncio.gather(original, duplicate)

    assert original_result.reward == 1.0
    assert duplicate_result.reward == 1.0
    # Only ONE full rollout's worth of calls -- the duplicate must not have dispatched its
    # own seed_session/responses/verify.
    assert client.url_paths_called.count("/seed_session") == 1
    assert client.url_paths_called.count("/v1/responses") == 1
    assert client.url_paths_called.count("/verify") == 1
    assert not agent._inflight_rollouts, "the entry must be cleaned up once the rollout finishes"


async def test_duplicate_run_falls_back_to_independent_rollout_if_original_is_hung() -> None:
    gate = asyncio.Event()  # deliberately never set -- the original hangs forever in seed_session
    client = _GatedServerClient(gate, reward=1.0)
    agent = _make_agent(client, dedup_wait_timeout_seconds=0.02)

    original = asyncio.create_task(_run(agent))
    await asyncio.sleep(0)

    # The duplicate must give up on the hung original after dedup_wait_timeout_seconds and run
    # its own independent rollout -- not hang alongside it.
    duplicate_result = await asyncio.wait_for(_run(agent), timeout=2.0)

    assert duplicate_result.reward == 1.0
    assert client.url_paths_called.count("/seed_session") == 2  # original (hung) + independent fallback

    # The abandoned original must be CANCELLED, not left running to completion for nothing --
    # it was burning compute/API calls for a result nobody would ever read.
    with pytest.raises(asyncio.CancelledError):
        await original
    assert original.cancelled()


async def test_concurrent_timeouts_on_same_hung_original_start_only_one_independent_rollout() -> None:
    """Two duplicates both time out on the same hung original at effectively the same time --
    only one of them may cancel+replace it; the other must safely re-attach to whatever that
    one starts (or start its own), never crash on an unhandled CancelledError, and the hung
    original must end up cancelled exactly once."""
    gate = asyncio.Event()  # deliberately never set -- the original hangs forever in seed_session
    client = _GatedServerClient(gate, reward=1.0)
    agent = _make_agent(client, dedup_wait_timeout_seconds=0.02)

    original = asyncio.create_task(_run(agent))
    await asyncio.sleep(0)

    duplicate_a = asyncio.create_task(_run(agent))
    duplicate_b = asyncio.create_task(_run(agent))

    result_a, result_b = await asyncio.wait_for(asyncio.gather(duplicate_a, duplicate_b), timeout=2.0)

    assert result_a.reward == 1.0
    assert result_b.reward == 1.0
    # Exactly one independent rollout replaced the hung original -- not two.
    assert client.url_paths_called.count("/seed_session") == 2
    assert original.cancelled()
    assert not agent._inflight_rollouts, "the entry must be cleaned up once the rollout finishes"


async def test_distinct_rollout_index_does_not_dedup() -> None:
    """Same question, different trial (num_repeats > 1) -- these must NOT be treated as
    duplicates of each other, or a multi-trial eval silently collapses to one rollout."""
    gate = asyncio.Event()
    client = _GatedServerClient(gate, reward=1.0)
    agent = _make_agent(client, dedup_wait_timeout_seconds=5.0)

    trial_0 = asyncio.create_task(_run(agent, _ng_task_index=3, _ng_rollout_index=0))
    await asyncio.sleep(0)  # let it register itself in _inflight_rollouts and block on the gate

    assert agent._inflight_rollouts, "the first trial must be tracked as in-flight"
    trial_1 = asyncio.create_task(_run(agent, _ng_task_index=3, _ng_rollout_index=1))
    await asyncio.sleep(0)

    gate.set()
    trial_0_result, trial_1_result = await asyncio.gather(trial_0, trial_1)

    assert trial_0_result.reward == 1.0
    assert trial_1_result.reward == 1.0
    # Each trial must run its own full rollout -- neither may piggyback on the other's.
    assert client.url_paths_called.count("/seed_session") == 2
    assert client.url_paths_called.count("/v1/responses") == 2
    assert client.url_paths_called.count("/verify") == 2
    assert not agent._inflight_rollouts, "both entries must be cleaned up once the rollouts finish"
