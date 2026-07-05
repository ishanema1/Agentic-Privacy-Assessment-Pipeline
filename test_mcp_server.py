"""
Tests for mcp_server.py. Fully offline — mocked Confluence, empty
assessment store, and a stub attacker-model function.

Run with: pytest test_mcp_server.py -v
"""

import io
import json

import pytest

from agent import AgentContext
from confluence_client import ConfluenceClient
from fakes import SAMPLE_STORAGE_HTML, FakeSession, make_confluence_page_flow
from mcp_server import MCPServer, build_pipeline_tools, run_stdio_server
from vector_store import HashingEmbedder, PriorAssessmentStore


def make_server(run_attacker_model=None) -> MCPServer:
    confluence_client = ConfluenceClient(
        base_url="https://example.atlassian.net",
        email="pipeline-bot@example.com",
        api_token="fake-token-not-real",
        session=FakeSession(make_confluence_page_flow(
            "Architecture Overview", SAMPLE_STORAGE_HTML, "/spaces/CUST014/pages/page-1"
        )),
    )
    context = AgentContext(
        confluence_client=confluence_client,
        assessment_store=PriorAssessmentStore(embedder=HashingEmbedder(dim=128)),
        run_attacker_model=run_attacker_model or (lambda use_case: 0.3),
        space_key="CUST014",
    )
    return MCPServer(tools=build_pipeline_tools(context))


# --------------------------------------------------------------------------
# Protocol methods
# --------------------------------------------------------------------------

def test_initialize_returns_protocol_info():
    server = make_server()
    response = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

    assert response["id"] == 1
    assert "protocolVersion" in response["result"]
    assert response["result"]["serverInfo"]["name"] == "privacy-pipeline-mcp-server"


def test_notifications_initialized_returns_none():
    """Per the MCP/JSON-RPC spec, notifications must not receive a response."""
    server = make_server()
    response = server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert response is None


def test_tools_list_returns_exactly_the_exposed_tools():
    server = make_server()
    response = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

    names = {t["name"] for t in response["result"]["tools"]}
    assert names == {"search_confluence_docs", "search_prior_assessments", "run_attacker_model"}
    # submit_privacy_assessment is agent.py-internal and must NOT be exposed here
    assert "submit_privacy_assessment" not in names


def test_tools_list_entries_use_camelcase_input_schema_key():
    """MCP's wire format uses 'inputSchema' (camelCase), not Anthropic tool-use's
    'input_schema' — a detail worth getting right since it's a real interop bug
    if missed."""
    server = make_server()
    response = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    for tool in response["result"]["tools"]:
        assert "inputSchema" in tool
        assert "input_schema" not in tool


# --------------------------------------------------------------------------
# tools/call
# --------------------------------------------------------------------------

def test_tools_call_executes_real_tool_and_returns_content():
    server = make_server()
    response = server.handle_request({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "search_confluence_docs", "arguments": {"space_key": "CUST014"}},
    })

    assert response["result"]["isError"] is False
    payload = json.loads(response["result"]["content"][0]["text"])
    assert "anonymized vehicle trajectory data" in payload["docs"][0]["content"]


def test_tools_call_unknown_tool_returns_json_rpc_error():
    server = make_server()
    response = server.handle_request({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "not_a_real_tool", "arguments": {}},
    })

    assert "error" in response
    assert response["error"]["code"] == -32602


def test_tools_call_with_failing_tool_returns_is_error_true_not_a_protocol_error():
    """A tool that raises during execution is a successful RPC call with a
    failed result (isError: true) — NOT a JSON-RPC protocol-level error.
    Clients rely on this distinction to tell 'bad request' from 'tool failed'."""

    def failing_attacker_model(use_case):
        raise RuntimeError("synthetic population generation failed")

    server = make_server(run_attacker_model=failing_attacker_model)
    response = server.handle_request({
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "run_attacker_model", "arguments": {"use_case_summary": "test"}},
    })

    assert "error" not in response  # not a protocol error
    assert response["result"]["isError"] is True
    assert "synthetic population generation failed" in response["result"]["content"][0]["text"]


# --------------------------------------------------------------------------
# Unknown methods / malformed input
# --------------------------------------------------------------------------

def test_unknown_method_returns_method_not_found_error():
    server = make_server()
    response = server.handle_request({"jsonrpc": "2.0", "id": 6, "method": "not/a/real/method"})

    assert response["error"]["code"] == -32601


def test_response_id_always_matches_request_id():
    server = make_server()
    for request_id in [1, "string-id", 999]:
        response = server.handle_request({"jsonrpc": "2.0", "id": request_id, "method": "ping"})
        assert response["id"] == request_id


# --------------------------------------------------------------------------
# stdio transport
# --------------------------------------------------------------------------

def test_stdio_server_skips_notifications_and_responds_to_requests():
    server = make_server()
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},  # no id -> notification
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    stdout = io.StringIO()

    run_stdio_server(server, stdin=stdin, stdout=stdout)

    lines = [line for line in stdout.getvalue().strip().split("\n") if line]
    assert len(lines) == 2  # exactly the 2 real requests, notification produced nothing
    responses = [json.loads(line) for line in lines]
    assert [r["id"] for r in responses] == [1, 2]


def test_stdio_server_ignores_malformed_json_lines():
    server = make_server()
    stdin = io.StringIO('not valid json\n{"jsonrpc": "2.0", "id": 1, "method": "ping"}\n')
    stdout = io.StringIO()

    run_stdio_server(server, stdin=stdin, stdout=stdout)

    lines = [line for line in stdout.getvalue().strip().split("\n") if line]
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == 1


def test_stdio_server_ignores_blank_lines():
    server = make_server()
    stdin = io.StringIO('\n\n{"jsonrpc": "2.0", "id": 1, "method": "ping"}\n\n')
    stdout = io.StringIO()

    run_stdio_server(server, stdin=stdin, stdout=stdout)

    lines = [line for line in stdout.getvalue().strip().split("\n") if line]
    assert len(lines) == 1
