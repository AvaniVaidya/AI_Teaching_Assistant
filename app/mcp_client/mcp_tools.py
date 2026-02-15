from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# AITeachingAssistant/
#   mcp_server/server.py
#   app/...
# If this file is: AITeachingAssistant/app/something/mcp_client.py
# then parents[2] points to AITeachingAssistant/
_SERVER_PATH = (Path(__file__).resolve().parents[2] / "mcp_server" / "server.py").resolve()


"""
mcp_client.py

Client wrapper that calls MCP tools exposed by mcp_server/server.py.

Usage:

  async with PracticeResultsClient() as c:
      await c.save_quiz_result({...})
      stats = await c.get_chapter_stats("student_001")
"""


class PracticeResultsClient:
    """
    MCP client wrapper for quiz result storage and stats.

    Key idea:
    - Keep the stdio connection + ClientSession open while you call tools.
    - Use this class as: async with PracticeResultsClient() as client: ...
    """

    def __init__(self, server_path: Path | str = _SERVER_PATH):
        # Server script to launch
        self.server_path = str(server_path)

        # Session will be created on connect
        self.session: Optional[ClientSession] = None

        # Tool metadata (optional, for debugging/UI)
        self.available_tools: List[dict] = []

        # We store these so we can close them cleanly in __aexit__
        self._server_params: Optional[StdioServerParameters] = None
        self._stdio_cm = None
        self._session_cm = None
        self._read = None
        self._write = None

    async def __aenter__(self) -> "PracticeResultsClient":
        """
        Async context manager entry:
        - starts server subprocess
        - opens stdio transport
        - initializes MCP session
        - optionally lists tools
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """
        Async context manager exit:
        - closes session
        - closes stdio transport (stops subprocess)
        """
        await self.close()

    async def connect(self) -> None:
        """
        Establish the MCP connection and keep it open.
        """
        # Create server parameters for stdio connection
        self._server_params = StdioServerParameters(
            command="python",            # Executable to run
            args=[self.server_path],     # Script path
            env=dict(os.environ),        # Environment variables
        )

        # Open stdio transport (starts the server subprocess)
        self._stdio_cm = stdio_client(self._server_params)
        self._read, self._write = await self._stdio_cm.__aenter__()

        # Create MCP session over stdio
        self._session_cm = ClientSession(self._read, self._write)
        self.session = await self._session_cm.__aenter__()

        # Initialize the MCP session (handshake)
        await self.session.initialize()

        # List available tools (optional but useful)
        response = await self.session.list_tools()
        tools = response.tools
        #print("\nConnected to server with tools:", [tool.name for tool in tools])

        self.available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in tools
        ]

    async def close(self) -> None:
        """
        Close the MCP connection and stop the server subprocess.
        """
        # Close MCP session first
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)

        # Then close stdio transport
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(None, None, None)

        # Reset fields
        self.session = None
        self._session_cm = None
        self._stdio_cm = None
        self._read = None
        self._write = None
        self._server_params = None

    async def save_quiz_result(self, payload: Dict[str, Any]) -> Any:
        """
        Call MCP tool: save_quiz_result

        payload should include:
          student_id, quiz_id, chapter_id, score, total, created_at
        """
        if not self.session:
            raise RuntimeError("Client not connected. Use: async with PracticeResultsClient()")

        result = await self.session.call_tool("save_quiz_result", payload)
        return result  # return raw result; parse based on your MCP SDK output

    async def get_chapter_stats(self, student_id: str) -> Any:
        """
        Call MCP tool: get_chapter_stats
        """
        if not self.session:
            raise RuntimeError("Client not connected. Use: async with PracticeResultsClient()")

        result = await self.session.call_tool("get_chapter_stats", {"student_id": student_id})
        sc = getattr(result, "structuredContent", None) or {}
        payload = sc.get("result", sc)  # unwrap if wrapped as {"result": ...}
        return payload  # return raw result; parse based on your MCP SDK output
