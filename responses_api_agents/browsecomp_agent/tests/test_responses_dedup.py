# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""_run_rollout calls this agent's OWN /v1/responses endpoint to run an entire multi-step
rollout in one HTTP request/response, which can legitimately take hours. If that connection
drops mid-flight, ServerClient silently retries with the identical body, and without a guard
the retry starts a second, fully-independent execution while the first keeps running unseen
-- confirmed via manual log tracing on 2026-09-24. `_responses_impl`'s dedup wrapper, keyed on
(qid, _dedup_id), must attach a retry to the in-flight original instead."""

import asyncio
from unittest.mock import MagicMock

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_agents.browsecomp_agent.app import BrowsecompAgent, BrowsecompAgentConfig


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


def _model_response() -> NeMoGymResponse:
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
    )


def _body(dedup_id: str | None) -> NeMoGymResponseCreateParamsNonStreaming:
    metadata = {"_dedup_id": dedup_id} if dedup_id is not None else None
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {"input": [{"role": "user", "content": "Q?"}], "metadata": metadata}
    )


def _request() -> MagicMock:
    request = MagicMock()
    request.cookies = {}
    return request


def _gated_inner(agent: BrowsecompAgent, gate: asyncio.Event, calls: list):
    async def _fake_inner(request, body):
        calls.append(body)
        if len(calls) == 1:
            await gate.wait()
        return _model_response(), {"session": f"call-{len(calls)}"}

    agent._responses_impl_inner = _fake_inner


async def test_duplicate_self_call_attaches_to_in_flight_instead_of_restarting() -> None:
    agent = BrowsecompAgent(config=_make_config(dedup_wait_timeout_seconds=5.0), server_client=MagicMock(spec=ServerClient))
    gate = asyncio.Event()
    calls: list = []
    _gated_inner(agent, gate, calls)

    response_1, response_2 = MagicMock(), MagicMock()
    original = asyncio.create_task(agent._responses_impl(_request(), response_1, _body("dedup-a")))
    await asyncio.sleep(0)  # let the outer task start and spawn the inner _responses_impl_inner task
    await asyncio.sleep(0)  # let the inner task itself start and block on the gate

    assert agent._inflight_responses, "the original self-call must be tracked as in-flight"
    duplicate = asyncio.create_task(agent._responses_impl(_request(), response_2, _body("dedup-a")))
    await asyncio.sleep(0)  # let the duplicate reach the dedup branch and start waiting

    gate.set()  # release the original; the duplicate should now piggyback on its result
    await asyncio.gather(original, duplicate)

    assert len(calls) == 1, "the duplicate must not have started a second execution"
    # Both callers' own Response objects must still get the cookies from whichever call ran.
    response_1.set_cookie.assert_called_once_with("session", "call-1")
    response_2.set_cookie.assert_called_once_with("session", "call-1")
    assert not agent._inflight_responses, "the entry must be cleaned up once the call finishes"


async def test_duplicate_self_call_falls_back_after_timeout_on_hung_original() -> None:
    agent = BrowsecompAgent(config=_make_config(dedup_wait_timeout_seconds=0.02), server_client=MagicMock(spec=ServerClient))
    gate = asyncio.Event()  # deliberately never set -- the original hangs forever
    calls: list = []
    _gated_inner(agent, gate, calls)

    original = asyncio.create_task(agent._responses_impl(_request(), MagicMock(), _body("dedup-a")))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    duplicate_result = await asyncio.wait_for(
        agent._responses_impl(_request(), MagicMock(), _body("dedup-a")), timeout=2.0
    )

    assert duplicate_result.id == "resp_001"
    assert len(calls) == 2, "the fallback must have started its own independent execution"
    assert original.cancelled(), "the abandoned original must be cancelled, not left running for nothing"


async def test_distinct_dedup_id_does_not_dedup() -> None:
    """A legitimate concurrent repeat of the same question (a fresh _run_rollout attempt, or a
    different num_repeats trial) gets its own _dedup_id and must never be merged with another
    in-flight attempt just because the question text matches."""
    agent = BrowsecompAgent(config=_make_config(dedup_wait_timeout_seconds=5.0), server_client=MagicMock(spec=ServerClient))
    gate = asyncio.Event()
    calls: list = []
    _gated_inner(agent, gate, calls)

    trial_0 = asyncio.create_task(agent._responses_impl(_request(), MagicMock(), _body("dedup-a")))
    await asyncio.sleep(0)

    trial_1 = asyncio.create_task(agent._responses_impl(_request(), MagicMock(), _body("dedup-b")))
    await asyncio.sleep(0)

    gate.set()
    await asyncio.gather(trial_0, trial_1)

    assert len(calls) == 2, "each distinct dedup_id must run its own independent execution"


async def test_no_dedup_id_skips_dedup_entirely() -> None:
    """A direct caller that never sets _dedup_id (anything outside _run_rollout) is unaffected
    -- unchanged behavior from before this dedup layer existed."""
    agent = BrowsecompAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))
    calls: list = []
    always_open_gate = asyncio.Event()
    always_open_gate.set()
    _gated_inner(agent, always_open_gate, calls)

    response = MagicMock()
    result_1 = await agent._responses_impl(_request(), response, _body(None))
    result_2 = await agent._responses_impl(_request(), response, _body(None))

    assert len(calls) == 2, "no dedup_id means every call runs independently"
    assert result_1.id == "resp_001" and result_2.id == "resp_001"
    assert not agent._inflight_responses
