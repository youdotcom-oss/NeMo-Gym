# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""include_domains is injected into the `search` tool's `queries` item schema at request time,
since the dataset rows (pre-generated BrowseComp jsonls) don't carry it. Scoping is per-query
(each `queries` item is either a plain string or {query, include_domains}), never a call-level
field -- a call-level field would leak the restriction onto sibling queries run in the same
search() call. See resources_servers/browsecomp_advanced_harness/app.py's
YouSearchResourcesServer.SearchQuery for the resources-server side that actually honors it.
"""

from unittest.mock import MagicMock

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

from .test_progress_tracking import SYSTEM, USER, _make_agent, _make_msg, _model_response


def _search_tool(name: str = "search") -> dict:
    return {
        "type": "function",
        "name": name,
        "description": "d",
        "parameters": {
            "type": "object",
            "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
            "required": ["queries"],
        },
        "strict": False,
    }


async def test_query_scoping_added_to_search_tool_queries_items() -> None:
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
    items = parameters["properties"]["queries"]["items"]
    variants = items["anyOf"]
    assert {"type": "string"} in variants
    object_variant = next(v for v in variants if v.get("type") == "object")
    assert "include_domains" in object_variant["properties"]
    assert object_variant["required"] == ["query"]
    # queries itself stays required -- only the per-item include_domains is optional
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
    assert parameters["properties"]["queries"]["items"] == {"type": "string"}
