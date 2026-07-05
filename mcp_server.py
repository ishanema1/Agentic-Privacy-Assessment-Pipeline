"""
mcp_server.py

A Model Context Protocol (MCP) server, implemented directly against the
wire protocol — JSON-RPC 2.0 messages over stdio — rather than through
the official SDK. Same reasoning as agent.py's from-scratch tool-calling
loop: implementing the protocol by hand is a stronger demonstration of
understanding what MCP actually is (newline-delimited JSON-RPC over
stdin/stdout) than importing a decorator that hides it.

Exposes the same read-only tools agent.py's tool-calling agent uses —
search_confluence_docs, search_prior_assessments, run_attacker_model —
by reusing agent.py's TOOL_SCHEMAS and dispatch_tool directly, rather
than redefining the tool contracts a second time. Any MCP-compatible
client (Claude Desktop, another agent, a different LLM application) can
call these tools over this server without importing any Python from this
repo — that's the whole point of the protocol.

Protocol reference: https://modelcontextprotocol.io

Everything in the demo/tests here uses the same synthetic/mocked context
as the rest of this repo. No real customer data.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Callable, Optional, TextIO

from agent import TOOL_SCHEMAS, AgentContext, dispatch_tool

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "privacy-pipeline-mcp-server"
SERVER_VERSION = "0.1.0"

# Tools exposed over MCP — deliberately excludes submit_privacy_assessment,
# which is agent.py's internal "final answer" mechanism, not a general-
# purpose tool an external MCP client should be calling.
EXPOSED_TOOL_NAMES = {"search_confluence_docs", "search_prior_assessments", "run_attacker_model"}


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], dict]


class MCPProtocolError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------
# Core protocol logic (transport-independent — this is what tests exercise)
# --------------------------------------------------------------------------

class MCPServer:
    """
    Implements MCP's JSON-RPC methods (initialize, tools/list, tools/call)
    independent of the stdio transport, so tests can call handle_request()
    directly without spinning up a subprocess or piping stdin/stdout.
    """

    def __init__(self, tools: list[ToolDefinition]):
        self._tools = {t.name: t for t in tools}

    def handle_request(self, request: dict) -> Optional[dict]:
        """Returns a JSON-RPC response dict, or None for notifications
        (which the spec says must not receive a response)."""
        method = request.get("method")
        request_id = request.get("id")
        is_notification = "id" not in request

        try:
            if method == "initialize":
                result = self._handle_initialize()
            elif method == "notifications/initialized":
                return None
            elif method == "tools/list":
                result = self._handle_tools_list()
            elif method == "tools/call":
                result = self._handle_tools_call(request.get("params", {}))
            elif method == "ping":
                result = {}
            else:
                raise MCPProtocolError(-32601, f"Method not found: {method}")
        except MCPProtocolError as e:
            return None if is_notification else self._error_response(request_id, e.code, e.message)
        except Exception as e:
            return None if is_notification else self._error_response(request_id, -32000, str(e))

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _handle_initialize(self) -> dict:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    def _handle_tools_list(self) -> dict:
        return {
            "tools": [
                {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
                for t in self._tools.values()
            ]
        }

    def _handle_tools_call(self, params: dict) -> dict:
        name = params.get("name")
        arguments = params.get("arguments", {})

        tool = self._tools.get(name)
        if tool is None:
            raise MCPProtocolError(-32602, f"Unknown tool: {name}")

        try:
            result = tool.handler(arguments)
            return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}
        except Exception as e:
            # Tool execution failures are reported IN the result (isError: true),
            # not as a JSON-RPC protocol error — the call itself succeeded, the
            # tool's own logic failed. MCP clients rely on this distinction.
            return {"content": [{"type": "text", "text": f"Tool execution failed: {e}"}], "isError": True}

    @staticmethod
    def _error_response(request_id, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


# --------------------------------------------------------------------------
# Tool wiring — reuses agent.py's tool contracts, doesn't redefine them
# --------------------------------------------------------------------------

def build_pipeline_tools(context: AgentContext) -> list[ToolDefinition]:
    """
    Wraps the subset of agent.py's tools that make sense as standalone MCP
    tools. Deliberately reuses TOOL_SCHEMAS and dispatch_tool rather than
    redefining the same three tool contracts a second time — one source of
    truth for what each tool does, whether it's called by the in-process
    agent loop or by an external MCP client.
    """
    tools = []
    for schema in TOOL_SCHEMAS:
        if schema["name"] not in EXPOSED_TOOL_NAMES:
            continue
        tools.append(ToolDefinition(
            name=schema["name"],
            description=schema["description"],
            input_schema=schema["input_schema"],
            handler=lambda args, _name=schema["name"]: dispatch_tool(_name, args, context),
        ))
    return tools


# --------------------------------------------------------------------------
# stdio transport — what an MCP client actually launches as a subprocess
# --------------------------------------------------------------------------

def run_stdio_server(server: MCPServer, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    """
    Reads newline-delimited JSON-RPC requests from stdin, writes responses
    to stdout — the actual transport a real MCP client (Claude Desktop,
    etc.) speaks when it launches this file as a subprocess.
    """
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue  # malformed input with no parseable id — nothing valid to respond with

        response = server.handle_request(request)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


def _demo() -> None:
    from confluence_client import ConfluenceClient
    from vector_store import HashingEmbedder, PriorAssessmentStore
    from attacker_model import evaluate_reidentification_risk
    from fakes import SAMPLE_STORAGE_HTML, FakeSession, make_confluence_page_flow

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
        run_attacker_model=lambda use_case: evaluate_reidentification_risk(cell_size=30, n_individuals=20, seed=0),
        space_key="CUST014",
    )
    server = MCPServer(tools=build_pipeline_tools(context))

    print("-- initialize --")
    print(json.dumps(server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"}), indent=2))

    print("\n-- tools/list --")
    tools_list = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    print(json.dumps(tools_list, indent=2))

    print("\n-- tools/call: search_confluence_docs --")
    call_result = server.handle_request({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "search_confluence_docs", "arguments": {"space_key": "CUST014"}},
    })
    print(json.dumps(call_result, indent=2))


if __name__ == "__main__":
    # `python mcp_server.py` with no piped input just runs the demo. A real
    # deployment would wire in a real ConfluenceClient (real credentials) and
    # a populated PriorAssessmentStore, then call run_stdio_server(...) so an
    # MCP client (e.g. Claude Desktop) can launch this file as a subprocess
    # and speak JSON-RPC over its stdin/stdout.
    _demo()
