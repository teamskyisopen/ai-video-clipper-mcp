"""Tests for the standalone CLI entry point (main.py)."""

from unittest.mock import patch
import pytest

from core.config import Settings
from domain.models import AnalysisResult, ClipResult, ViralHighlight


def test_argument_parser_defaults() -> None:
    """Verify default arguments for CLI."""
    from main import cli_entrypoint
    import sys

    test_args = ["main.py", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    with patch.object(sys, "argv", test_args):
        with patch("asyncio.run") as mock_run:
            mock_run.return_value = 0
            with patch("sys.exit", side_effect=SystemExit) as mock_exit:
                with pytest.raises(SystemExit):
                    cli_entrypoint()
                mock_exit.assert_called_with(0)


def test_argument_parser_custom_flags() -> None:
    """Verify custom flags parsing."""
    from main import cli_entrypoint
    import sys

    test_args = [
        "main.py",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "-c", "3",
        "-p", "neon_cyan",
        "--no-ai",
        "--analyze-only",
    ]
    with patch.object(sys, "argv", test_args):
        with patch("asyncio.run") as mock_run:
            mock_run.return_value = 0
            with patch("sys.exit", side_effect=SystemExit) as mock_exit:
                with pytest.raises(SystemExit):
                    cli_entrypoint()
                mock_exit.assert_called_with(0)


def test_mcp_flag_redirection() -> None:
    """Verify --mcp flag delegates to FastMCP server."""
    from main import cli_entrypoint
    import sys

    test_args = ["main.py", "--mcp"]
    with patch.object(sys, "argv", test_args):
        with patch("mcp_server.server.main") as mock_mcp_main:
            with patch("sys.exit", side_effect=SystemExit) as mock_exit:
                with pytest.raises(SystemExit):
                    cli_entrypoint()
                mock_mcp_main.assert_called_once()
                mock_exit.assert_called_with(0)
