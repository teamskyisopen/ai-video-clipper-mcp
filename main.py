"""AI Video Clipper - Standalone CLI & MCP Launcher.

Transforms landscape videos into high-retention 9:16 vertical shorts
with dynamic speaker tracking and animated karaoke subtitles.
"""

import argparse
import asyncio
import os
from pathlib import Path
import sys
import time
from typing import List, Optional

from loguru import logger

from core.config import Settings, get_settings
from domain.models import AnalysisResult, ClipResult, ViralHighlight
from services.coordinator import PipelineCoordinator

VERSION = "1.0.0"

# ANSI Color codes for clean terminal presentation
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner() -> None:
    """Print clean ASCII header."""
    print(f"{CYAN}{BOLD}============================================================{RESET}")
    print(f"{CYAN}{BOLD}           AI Video Clipper & 9:16 Shorts Generator         {RESET}")
    print(f"{DIM}          High-Retention Viral Moments | v{VERSION}{RESET}")
    print(f"{CYAN}{BOLD}============================================================{RESET}\n")


def print_step(step_num: int, total_steps: int, title: str) -> None:
    """Print an active progress step indicator."""
    print(f"{CYAN}{BOLD}[{step_num}/{total_steps}]{RESET} {BOLD}{title}{RESET}")


def print_highlight_table(highlights: List[ViralHighlight]) -> None:
    """Render a clean terminal table displaying detected viral highlights."""
    print(f"\n{YELLOW}{BOLD}{'Rank':<5} {'Time Range':<16} {'Score':<8} {'Hook Title':<32}{RESET}")
    print(f"{DIM}{'-'*65}{RESET}")
    for idx, h in enumerate(highlights, 1):
        m_start, s_start = divmod(int(h.start_time), 60)
        m_end, s_end = divmod(int(h.end_time), 60)
        time_str = f"{m_start:02d}:{s_start:02d} - {m_end:02d}:{s_end:02d}"
        score_color = GREEN if h.virality_score >= 80 else YELLOW
        print(
            f"#{idx:<4} {time_str:<16} {score_color}{h.virality_score:>3}/100{RESET}  {BOLD}{h.hook_title[:30]:<32}{RESET}"
        )
        if h.clean_quote:
            quote_snippet = h.clean_quote.strip().replace("\n", " ")
            if len(quote_snippet) > 80:
                quote_snippet = quote_snippet[:77] + "..."
            print(f"      {DIM}\"{quote_snippet}\"{RESET}")
    print(f"{DIM}{'-'*65}{RESET}\n")


def print_clip_summary(results: List[ClipResult]) -> None:
    """Print summary card for rendered clips."""
    print(f"\n{GREEN}{BOLD}============================================================{RESET}")
    print(f"{GREEN}{BOLD}        [SUCCESS] Short Rendering Completed Successfully!  {RESET}")
    print(f"{GREEN}{BOLD}============================================================{RESET}")
    for idx, r in enumerate(results, 1):
        size_mb = r.file_size_bytes / (1024 * 1024)
        print(f"  {BOLD}Clip #{idx}:{RESET}")
        print(f"  * {BOLD}Path:{RESET}     {CYAN}{r.output_path}{RESET}")
        print(f"  * {BOLD}Duration:{RESET} {r.duration_seconds:.1f} seconds")
        print(f"  * {BOLD}Size:{RESET}     {size_mb:.2f} MB")
        print(f"{DIM}{'-'*60}{RESET}")


async def run_pipeline(args: argparse.Namespace) -> int:
    """Execute the end-to-end video clipping pipeline."""
    settings = get_settings()

    # Apply CLI runtime overrides
    if args.no_ai:
        settings.offline_mode = True
        settings.llm_provider = "none"
    if args.output_dir:
        settings.output_dir = Path(args.output_dir).resolve()

    # Configure logging verbosity
    logger.remove()
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="WARNING")

    print_banner()

    mode_label = (
        f"{YELLOW}Offline Speech Heuristic (Zero-AI){RESET}"
        if (settings.offline_mode or settings.llm_provider == "none")
        else f"{GREEN}AI Scoring ({settings.llm_provider.capitalize()}: {settings.gemini_model if settings.llm_provider == 'gemini' else settings.openai_model}){RESET}"
    )

    print(f"[*] {BOLD}Target URL:{RESET}     {args.url}")
    print(f"[*] {BOLD}Engine Mode:{RESET}    {mode_label}")
    print(f"[*] {BOLD}Subtitle Style:{RESET} {args.preset}")
    print(f"[*] {BOLD}Clips Target:{RESET}   {args.clips}\n")

    coordinator = PipelineCoordinator(settings)
    total_steps = 2 if args.analyze_only else 4
    start_wall = time.time()

    # Step 1: Download & Transcribe
    print_step(1, total_steps, "Downloading fast audio & transcribing speech (Whisper)...")
    try:
        # Step 2: Highlight Analysis
        print_step(2, total_steps, "Evaluating retention hooks & virality rubrics...")
        analysis = await coordinator.analyze_video_highlights(
            url=args.url,
            max_clips=args.clips,
        )

        if not analysis.highlights:
            print(f"\n{RED}[X] No highlight candidates detected in source video.{RESET}")
            return 1

        print(f"\n{GREEN}[OK] Detected {len(analysis.highlights)} viral highlight moment(s):{RESET}")
        print_highlight_table(analysis.highlights)

        if args.analyze_only:
            print(f"{GREEN}[OK] Analysis completed in {time.time() - start_wall:.1f}s. Exiting (--analyze-only).{RESET}")
            return 0

        # Step 3 & 4: Render Clips
        rendered_clips: List[ClipResult] = []
        for idx, h in enumerate(analysis.highlights[: args.clips], 1):
            print_step(
                3,
                total_steps,
                f"Cropping 9:16 portrait view & tracking speaker (Clip {idx}/{args.clips})...",
            )
            print_step(
                4,
                total_steps,
                f"Burning animated subtitles & rendering MP4 via FFmpeg...",
            )

            clip_id = f"short_{idx:02d}_{int(h.start_time)}_{int(h.end_time)}"
            clip_res = await coordinator.render_highlight_clip(
                url=args.url,
                start_time=h.start_time,
                end_time=h.end_time,
                subtitle_preset=args.preset,
                clip_id=clip_id,
            )
            rendered_clips.append(clip_res)

        print_clip_summary(rendered_clips)
        print(f"\n{DIM}Total processing time: {time.time() - start_wall:.1f}s{RESET}\n")
        return 0

    except Exception as exc:
        print(f"\n{RED}{BOLD}[X] Pipeline Error:{RESET} {exc}")
        if args.verbose:
            logger.exception("Detailed error traceback:")
        else:
            print(f"{DIM}Tip: Run with --verbose to view the full diagnostic stack trace.{RESET}\n")
        return 1


def cli_entrypoint() -> None:
    """Parse command line arguments and launch CLI or MCP server."""
    parser = argparse.ArgumentParser(
        prog="ai-video-clipper",
        description="Transform landscape videos into viral 9:16 vertical shorts with smart framing and animated subtitles.",
    )

    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="Source video URL (YouTube, Vimeo, or direct media link).",
    )
    parser.add_argument(
        "-c",
        "--clips",
        type=int,
        default=1,
        help="Number of viral highlight clips to detect and render (default: 1).",
    )
    parser.add_argument(
        "-p",
        "--preset",
        type=str,
        default="karaoke_yellow",
        choices=["karaoke_yellow", "neon_cyan", "clean_white", "bold_magenta"],
        help="Visual style preset for burned subtitles (default: 'karaoke_yellow').",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Force offline speech heuristic detection (bypasses Gemini/OpenAI API keys).",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Analyze video and display viral candidate timestamps/scores without rendering.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store rendered short clips (default: ./storage/output).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging output.",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Launch as FastMCP stdio server for AI assistants (Claude Desktop, Antigravity).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    args = parser.parse_args()

    # Route to MCP server if requested
    if args.mcp:
        from mcp_server.server import main as mcp_main
        mcp_main()
        sys.exit(0)

    # Validate URL requirement for standard CLI run
    if not args.url:
        print_banner()
        parser.print_help()
        print(f"\n{YELLOW}[!] Please provide a video URL or use --mcp to start the MCP server.{RESET}")
        sys.exit(1)

    # Run async pipeline
    exit_code = asyncio.run(run_pipeline(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    cli_entrypoint()
