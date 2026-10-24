# AI Video Clipper & 9:16 Shorts Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-libass%20enabled-green.svg)](https://ffmpeg.org/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Protocol%20Ready-purple.svg)](https://modelcontextprotocol.io/)

An autonomous video clipping engine that turns long-form landscape videos (YouTube, Vimeo, podcasts, interviews) into high-retention **9:16 vertical shorts** with intelligent speaker tracking and animated karaoke subtitles.

Works in **dual-modes**:
1. **Simple Standalone CLI**: Run directly from your terminal (`python main.py <URL>`).
2. **Model Context Protocol (MCP) Server**: Connect directly to **Claude Desktop**, **Antigravity**, **Cursor**, or any MCP-compatible AI assistant.

Supports running **with AI** (Google Gemini, OpenAI, or local Ollama) or **100% offline without any AI API keys** using built-in speech density heuristics.

---

## Key Features

* **High-Retention Highlight Detection**:
  * **AI Mode**: Evaluates transcript against virality rubrics (first 3-second hook, emotional peaks, contrarian statements, punchline payoffs) using **Google Gemini** or **OpenAI**.
  * **Zero-AI / Offline Mode**: Uses local Whisper cadence and sentence boundary heuristics. No API keys or remote calls required.
* **Smart Vertical 9:16 Framing**:
  * Detects human faces with OpenCV Haar Cascades and dynamically repositions the 9:16 portrait viewport.
  * **Exponential Moving Average (EMA)** smoothing eliminates camera jitter while resetting instantly across scene cuts.
  * **Dual-Speaker Split-Screen**: Automatically stacks two speakers vertically into a split layout when both are active.
* **Animated Karaoke Subtitles**:
  * Emits word-by-word highlighted subtitles (`.ass`) rendered by `libass`.
  * Customizable style presets (`karaoke_yellow`, `neon_cyan`, `clean_white`, `bold_magenta`).
  * Enforces safe-zone margins avoiding TikTok, Shorts, and Reels UI buttons.
* **Instant Section Downloads**:
  * Uses `yt-dlp` HTTP range requests to download only the exact 30–60 second video slice, avoiding multi-gigabyte downloads.
* **Production-Grade Hexagonal Architecture**:
  * Zero hardcoded constants, fully isolated components, automatic temporary file cleanup.

---

## Operating Modes: AI vs. Zero-AI (Offline)

| Feature | AI Mode (Gemini / OpenAI) | Offline Mode (`--no-ai` / No Keys) |
| :--- | :--- | :--- |
| **Speech-to-Text** | Local `faster-whisper` (CPU/GPU) | Local `faster-whisper` (CPU/GPU) |
| **Highlight Detection** | Gemini 3.8 Flash / GPT-4o Virality Scoring | Punctuation & Speech Cadence Heuristic |
| **Speaker Tracking** | Local OpenCV Computer Vision | Local OpenCV Computer Vision |
| **Subtitle Animation** | Local `libass` Word Karaoke | Local `libass` Word Karaoke |
| **Video Compositing** | Local FFmpeg x264/AAC | Local FFmpeg x264/AAC |
| **API Costs & Network** | Minimal API tokens for text analysis | **$0.00 / 100% Free & Offline** |

---

## Quickstart & Installation

### 1. Prerequisites
* **Python**: Version 3.10 or higher.
* **FFmpeg**: Must be compiled with `--enable-libass` for subtitle rendering.

### 2. Clone & Set Up Python Environment
```bash
git clone https://github.com/your-username/ai-video-clipper.git
cd ai-video-clipper

# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Automated FFmpeg Setup
We provide automated setup scripts that download or configure FFmpeg with `libass` support into `./bin/`:

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_ffmpeg.ps1
```

**Linux / macOS (Bash):**
```bash
chmod +x ./scripts/setup_ffmpeg.sh
./scripts/setup_ffmpeg.sh
```

*(Alternatively, ensure your system `ffmpeg` has `libass` support enabled via `ffmpeg -version | grep libass`).*

### 4. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

If you plan to use AI highlight scoring, provide your Gemini or OpenAI API key:
```env
# For Google Gemini (Recommended, fast & cost-efficient)
LLM_PROVIDER="gemini"
GEMINI_API_KEY="AIzaSy..."
GEMINI_MODEL="gemini-3.8-flash"

# Or for OpenAI / OpenAI-Compatible (Ollama, Groq, vLLM)
# LLM_PROVIDER="openai"
# OPENAI_API_KEY="sk-..."
# OPENAI_MODEL="gpt-4o"
```

*If you do not provide any API key, the engine automatically runs in offline heuristic mode.*

---

## Command-Line Usage (CLI)

### Simplest Command (Generate Top 1 Short)
```bash
python main.py "https://www.youtube.com/watch?v=Gy_kI0S3UgA"
```

### Generate Multiple Shorts with Custom Subtitles
```bash
python main.py "https://www.youtube.com/watch?v=Gy_kI0S3UgA" -c 3 -p neon_cyan
```

### 100% Offline Mode (Zero AI / Free)
```bash
python main.py "https://www.youtube.com/watch?v=Gy_kI0S3UgA" --no-ai
```

### Analyze Highlights Only (No Video Rendering)
Displays detected viral moments, timestamps, and virality scores in a terminal table without downloading full video slices:
```bash
python main.py "https://www.youtube.com/watch?v=Gy_kI0S3UgA" --analyze-only
```

### CLI Options Reference
```text
usage: ai-video-clipper [-h] [-c CLIPS]
                        [-p {karaoke_yellow,neon_cyan,clean_white,bold_magenta}]
                        [--no-ai] [--analyze-only] [-o OUTPUT_DIR] [-v]
                        [--mcp] [--version]
                        [url]

positional arguments:
  url                   Source video URL (YouTube, Vimeo, or direct media link).

options:
  -h, --help            Show this help message and exit.
  -c, --clips CLIPS     Number of viral highlight clips to render (default: 1).
  -p, --preset PRESET   Subtitle style preset (default: 'karaoke_yellow').
  --no-ai               Force offline speech heuristic detection (no API calls).
  --analyze-only        Display candidate timestamps/scores without rendering.
  -o, --output-dir DIR  Target output directory (default: ./storage/output).
  -v, --verbose         Enable verbose debug logging.
  --mcp                 Launch as FastMCP stdio server for AI assistants.
  --version             Show version information.
```

---

## Using as an MCP Server (Model Context Protocol)

You can connect this tool directly to AI assistants like **Claude Desktop**, **Antigravity**, or **Cursor**.

### 1. Claude Desktop Configuration
Add the server entry to `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "ai-video-clipper": {
      "command": "python",
      "args": ["/absolute/path/to/ai-video-clipper/cli.py"]
    }
  }
}
```

### 2. Antigravity Configuration
Add to `~/.gemini/config/mcp_config.json`:

```json
{
  "mcpServers": {
    "ai-video-clipper": {
      "command": "C:\\path\\to\\venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\ai-video-clipper\\cli.py"]
    }
  }
}
```

### Exposed MCP Tools

AI assistants gain access to 3 tools:
1. `analyze_video_highlights(url: str, max_clips: int = 3)`: Returns JSON metadata of top viral moments.
2. `render_highlight_clip(url: str, start_time: float, end_time: float, subtitle_preset: str = 'karaoke_yellow')`: Renders a specific time range into a vertical short.
3. `auto_clip_video(url: str, clip_count: int = 3)`: Macro tool executing the complete pipeline end-to-end.

---

## Configuration Reference (`.env`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `gemini` | LLM service provider (`gemini`, `openai`, `none`) |
| `GEMINI_API_KEY` | `""` | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model name |
| `OPENAI_API_KEY` | `""` | OpenAI or OpenAI-compatible API key |
| `OPENAI_MODEL` | `gpt-4o` | Model name for OpenAI provider |
| `OFFLINE_MODE` | `false` | When true, disables all external LLM calls |
| `WHISPER_MODEL` | `base` | Model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `WHISPER_DEVICE` | `auto` | Execution device (`auto`, `cuda`, `cpu`) |
| `TARGET_WIDTH` | `1080` | Rendered video width |
| `TARGET_HEIGHT` | `1920` | Rendered video height |
| `TARGET_FPS` | `30` | Rendered frame rate |
| `SUBTITLE_HIGHLIGHT_COLOR` | `&H0000FFFF&` | BGR hex color for karaoke highlight (yellow) |
| `SUBTITLE_MARGIN_V` | `250` | Bottom safe margin in pixels |

---

## Technical Architecture & Deep Dive

For mathematical derivations, computer vision bounding box calculations, EMA smoothing algorithms, and library specifications, please read:
**[docs/TECHNICAL_DOCUMENTATION.md](file:///c:/MyData/4_temp/ai-video-clipper-mcp/docs/TECHNICAL_DOCUMENTATION.md)**

---

## Running Automated Tests

Run the full test suite with PyTest:
```bash
pytest -v
```

All 51 automated unit, property, and integration tests cover:
- Architecture boundary rules and Pydantic validation.
- Audio ingestion and Whisper word-timestamp alignment.
- Virality scoring rubrics and sentence boundary snapping.
- Face bounding box tracking and EMA scene cut math.
- FFmpeg compositing, subtitle styling, and FastMCP server tools.

---

## Contributing

Contributions are warmly welcomed! Please open an issue or submit a pull request for improvements, bug fixes, or new subtitle animation styles.

---

## License

This project is open-source software licensed under the **[MIT License](file:///c:/MyData/4_temp/ai-video-clipper-mcp/LICENSE)**.
