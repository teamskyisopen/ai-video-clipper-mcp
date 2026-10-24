# Workspace Architecture & Engineering Rules: AI Video Clipper MCP

## 1. Architectural Style: Hexagonal (Ports & Adapters)

The project adheres to a strict **Hexagonal Architecture** (Ports and Adapters) pattern. The core domain and application services represent the hexagon interior, while MCP endpoints, CLI tools, external media tools (yt-dlp, ffmpeg), and remote AI APIs (Whisper, LLMs) represent adapters outside the hexagon.

```
       +-------------------------------------------------------+
       |                  Driving Adapters                     |
       |     [ MCP Server: mcp_server/ ]    [ CLI / Scripts ]   |
       +---------------------------+---------------------------+
                                   |
                                   v (Calls services via clean APIs)
       +-------------------------------------------------------+
       |                    Hexagon Core                       |
       |  +-------------------------------------------------+  |
       |  |  services/  (Standalone Business Services)      |  |
       |  |  domain/    (Pydantic v2 Models & Entities)     |  |
       |  |  core/      (Settings, Exceptions, Contexts)    |  |
       |  +-------------------------------------------------+  |
       +---------------------------+---------------------------+
                                   |
                                   v (Invokes driven adapters)
       +-------------------------------------------------------+
       |                  Driven Adapters                      |
       | [ yt-dlp ]   [ ffmpeg/ffprobe ]   [ LLM/Whisper APIs] |
       | [ Local Filesystem / storage/temp Context Managers ]  |
       +-------------------------------------------------------+
```

### Dependency Direction Rules
1. **Inner layers never import outer layers**:
   - `domain/` and `core/` MUST NEVER import from `services/` or `mcp_server/`.
   - `services/` MUST NEVER import from `mcp_server/` or CLI presentation layers.
   - `mcp_server/` may import from `services/`, `domain/`, and `core/`.
2. **Ports and Dependency Injection**:
   - All services must accept their dependencies and configuration via `__init__`.
   - No service should instantiate global singletons or reach out to external CLI state directly.

---

## 2. Core Project Invariants

### Invariant 1: ZERO Hardcoding
- **No magic strings, numbers, or hardcoded paths.**
- Every path, model identifier, timeout, video dimension, and API key MUST be declared in `core/config.py` using Pydantic's `BaseSettings` (`pydantic-settings`).
- All configurations must be loadable from environment variables or a local `.env` file with validated defaults.

### Invariant 2: Service Isolation & Modularity
- Every module under `services/` (`downloader.py`, `transcriber.py`, `analyzer.py`, `framer.py`, `compositor.py`) must define a standalone class/interface.
- **Zero circular imports**: Services must communicate solely via shared domain models (`domain/models.py`) or constructor parameters.
- No direct coupling to MCP tools, request contexts, or CLI runners.

### Invariant 3: Strict PEP 484 Typing & Pydantic v2 Validation
- All functions, methods, and class signatures must be fully typed (PEP 484 / PEP 585 / PEP 604).
- Runtime inputs, inter-service DTOs, and outputs must be validated using **Pydantic v2** models (`BaseModel` with strict typing, field descriptions, and custom validators where applicable).

### Invariant 4: Structured Logging
- **Strictly NO `print()` statements** anywhere in production code.
- Use structured logging via `loguru` (or standard `logging` configured with structured formatting).
- Log messages must include contextual metadata (e.g., job IDs, video URLs, processing durations) rather than string interpolation alone.

### Invariant 5: Safe Media Lifecycle & Temporary Directories
- External media tools (`yt-dlp`, `ffmpeg`, Whisper audio dumps, frame extractions) MUST write intermediate artifacts to temporary, auto-cleaning directories.
- Always use `tempfile.TemporaryDirectory`, context managers, or managed workspaces in `storage/temp/` guaranteed to clean up intermediate chunks, even on failure or unhandled exceptions.
- Do not leave untracked video/audio chunks in the project root or source directories.

---

## 3. Directory Layout & Responsibilities

| Path | Purpose | Layer |
| :--- | :--- | :--- |
| `core/config.py` | Central `Settings` class inheriting `BaseSettings`. Loads `.env`. | Core |
| `core/exceptions.py` | Custom domain exception hierarchy. | Core |
| `domain/models.py` | Domain entities, schemas, DTOs, validation logic. | Domain |
| `services/downloader.py` | YouTube & media fetching via yt-dlp into managed temp storage. | Application Service |
| `services/transcriber.py` | Audio extraction and timestamped transcription engine. | Application Service |
| `services/analyzer.py` | Transcript analysis, hook identification, clip scoring via LLM. | Application Service |
| `services/framer.py` | Face tracking, active speaker detection, dynamic 9:16 crop calculation. | Application Service |
| `services/compositor.py` | FFmpeg video cutting, aspect ratio conversion, subtitle burn-in. | Application Service |
| `mcp_server/server.py` | MCP Server tool exposure and invocation orchestration. | Driving Adapter |
| `storage/temp/` | Ephemeral media cache (ignored by git, auto-cleaned). | Infrastructure |
| `vendor/` | External third-party scripts, custom plugins, or vendor binaries. | Infrastructure |
| `tests/` | Unit, integration, and contract tests. | Testing |

---

## 4. Error Handling Protocol
1. Services must catch low-level subprocess, network, or parsing errors and translate them into typed exceptions from `core/exceptions.py`.
2. MCP server endpoints catch domain exceptions and convert them into structured error payloads for clients without leaking stack traces or raw paths unless in debug mode.
