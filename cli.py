"""CLI launcher for AI Video Clipper.

Supports both direct CLI command execution and FastMCP stdio server mode.
"""

import sys

if __name__ == "__main__":
    # If invoked without arguments or with --mcp, run FastMCP server for assistant compatibility
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "--mcp"):
        from mcp_server.server import main as mcp_main
        mcp_main()
    else:
        # Otherwise route through standard CLI entrypoint
        from main import cli_entrypoint
        cli_entrypoint()
