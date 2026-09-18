# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""include_domains is injected into the `search` tool schema at request time, since the
dataset rows (pre-generated BrowseComp jsonls) don't carry it. See
resources_servers/browsecomp_advanced_harness/app.py's YouSearchResourcesServer for the
resources-server side that actually honors the field.
"""

from unittest.mock import MagicMock

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

from .test_progress_tracking import SYSTEM, USER, _make_agent, _make_msg, _model_response


def _search_tool(name: str = "search") -> dict:
    return {
        "type": "function",
        "name": name,
        "description": "d",
        "parameters": {"type": "object", "properties": {"queries": {"type": "array"}}, "required": ["queries"]},
        "strict": False,
    }


async def test_include_domains_added_to_search_tool() -> None:
    agent, fake = _make_agent([_model_response([_make_msg("Exact Answer: X")])])
    request_mock = MagicMock()
    request_mock.cookies = {}
    response_mock = MagicMock()
    response_mock.set_cookie = MagicMock()
    body = NeMoGymResponseCreateParamsNonStreaming(input=[SYSTEM, USER], tools=[_search_tool()])

    await agent.responses(request_mock, response_mock, body)

    sent_tools = fake.model_bodies[0].tools
    search_tool = next(t for t in sent_tools if (t["name"] if isinstance(t, dict) else t.name) == "search")
    parameters = search_tool["parameters"] if isinstance(search_tool, dict) else search_tool.parameters
    assert "include_domains" in parameters["properties"]
    # queries stays required; include_domains is optional
    assert parameters["required"] == ["queries"]


async def test_no_search_tool_is_left_untouched() -> None:
    agent, fake = _make_agent([_model_response([_make_msg("Exact Answer: X")])])
    request_mock = MagicMock()
    request_mock.cookies = {}
    response_mock = MagicMock()
    response_mock.set_cookie = MagicMock()
    body = NeMoGymResponseCreateParamsNonStreaming(input=[SYSTEM, USER], tools=[_search_tool(name="browse")])

    await agent.responses(request_mock, response_mock, body)

    sent_tools = fake.model_bodies[0].tools
    browse_tool = next(t for t in sent_tools if (t["name"] if isinstance(t, dict) else t.name) == "browse")
    parameters = browse_tool["parameters"] if isinstance(browse_tool, dict) else browse_tool.parameters
    assert "include_domains" not in parameters["properties"]
