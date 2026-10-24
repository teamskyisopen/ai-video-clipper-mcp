# Technical Architecture & Developer Deep Dive

This document provides a comprehensive technical specification of **AI Video Clipper**, detailing every machine learning and media library used, the software architecture, and the mathematical formulations driving its computer vision framing, speech alignment, and video compositing algorithms.

---

## 1. System Architecture Overview

AI Video Clipper operates on a clean, modular **hexagonal pipeline architecture** orchestrated by `PipelineCoordinator`. Each stage communicates through strongly-typed Pydantic domain models with strict temporal invariants.

```
                    ┌────────────────────────────┐
                    │    Video Input Source      │
                    │  (YouTube URL or Media)    │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │   MediaDownloader (yt-dlp) │  ──> Fast Audio Stream (128k MP3/AAC)
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ FasterWhisperTranscriber   │  ──> Word-level Timestamps (CTranslate2)
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │   ViralHighlightDetector   │  ──> Ranked Moments & Sentence Snapping
                    │   (Gemini / OpenAI / Fallback) │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │   MediaDownloader (yt-dlp) │  ──> Download Exact Video Slice
                    └─────────────┬──────────────┘
                                  │
          ┌───────────────────────┴───────────────────────┐
          │                                               │
┌─────────▼──────────────┐                     ┌──────────▼─────────────┐
│   SmartVideoFramer     │                     │   SubtitleGenerator    │
│  - Scene Cut Tracking  │                     │  - Word Delta Timing   │
│  - Face Tracking (CV2) │                     │  - Karaoke ASS Tags    │
│  - EMA Smoothing       │                     │  - Platform UI Margins │
└─────────┬──────────────┘                     └──────────┬─────────────┘
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │      FFmpegCompositor      │  ──> 1080x1920 9:16 Short (x264/AAC)
                    └────────────────────────────┘
```

---

## 2. Machine Learning & Media Libraries

### 2.1 Audio Transcription & Speech-to-Text: `faster-whisper` & `ctranslate2`
* **Library**: `faster-whisper` (powered by `ctranslate2`)
* **Role**: Transcribes spoken audio into text segments and extracts word-level start and end timestamps.
* **Why faster-whisper over standard whisper**:
  - Up to **4x faster execution** and uses **half the memory** compared to OpenAI's vanilla PyTorch implementation.
  - Implements **CTranslate2**, a custom C++ inference engine using tensor quantization (`int8` on CPU, `float16` on CUDA).
* **Word Alignment Mechanism**:
  Standard Whisper models emit sequence tokens for text segments. `faster-whisper` computes the **cross-attention weights** between audio encoder states and text decoder tokens to deduce exact millisecond start and end boundaries for every individual spoken word:
  $$w_i = (\text{word}, t_{\text{start}}, t_{\text{end}}, p_{\text{confidence}})$$
  These word timestamps enable synchronized karaoke animations in the burned subtitles.
* **Voice Activity Detection (VAD)**:
  Uses the `Silero VAD` neural network pre-filter to prune non-speech silence intervals, preventing hallucinated subtitles during long pauses.

---

### 2.2 Virality Intelligence: `google-genai` & `httpx` (OpenAI-Compatible)
* **Libraries**: `google-genai` (official Google GenAI SDK) and `httpx` (HTTP/2 async client for OpenAI/Ollama/Groq)
* **Role**: Evaluates full-length transcripts against a structured virality rubric to discover high-retention short clips.
* **Rubric Constraints**:
  1. **Hook Strength (First 3 Seconds)**: The opening statement must create an information gap or pose a provocative question.
  2. **Emotional Peaks & Contrarian Statements**: Prioritizes humor, debate, counter-intuitive claims, or emotional vulnerability.
  3. **Self-Contained Arc**: Highlighting must never begin or end mid-thought.
  4. **Duration Window**: Strictly bounded between 15 seconds ($T_{\min}$) and 60 seconds ($T_{\max}$).
* **Structured JSON Output**:
  Guarantees type safety by passing Pydantic schemas (`AnalysisResult`) directly into the LLM engine (`response_mime_type="application/json"`), completely eliminating regex parsing failures.

---

### 2.3 Computer Vision & Face Tracking: `OpenCV` (`cv2`)
* **Library**: `opencv-python` (`cv2`)
* **Role**: Detects human subjects and computes optimal portrait framing bounding boxes.
* **Technique**:
  - Uses **Haar Feature-Based Cascade Classifiers** (`haarcascade_frontalface_default.xml`).
  - Samples video frames at uniform intervals across scene shots to identify facial bounding boxes $[x, y, w, h]$.
  - Calculates the spatial centroid $(c_x, c_y)$ of the primary speaker to guide horizontal pan-and-scan camera repositioning.

---

### 2.4 Subtitle Rendering & Styling: `libass` & Advanced SubStation Alpha (`.ass`)
* **Library**: `libass` compiled directly into `FFmpeg`
* **Role**: Burns word-by-word animated karaoke subtitles into the output video.
* **Why ASS over SRT**:
  - SRT files only support static full-sentence rendering without millisecond animations.
  - ASS scripts allow vector-level positioning, outline shadows, custom margin offsets, and the `{\k<duration>}` karaoke tag to highlight active spoken words dynamically.

---

### 2.5 Media Ingestion & Video Compositing: `yt-dlp` & `FFmpeg`
* **Libraries**: `yt-dlp`, `ffmpeg`, `ffprobe`
* **Role**:
  - `yt-dlp`: Fetches stream manifests and performs non-destructive section downloading via HTTP range requests (`download_ranges`), downloading only the selected 30–60 second slice rather than the entire multi-gigabyte video.
  - `FFmpeg`: Performs GPU/CPU video decoding, crop filters, scaling to $1080 \times 1920$, subtitle burn-in, and H.264/AAC encoding.

---

## 3. Mathematical & Algorithmic Formulations

### 3.1 Geometric Aspect Ratio Transformation (16:9 to 9:16)
Given a source video with dimensions $W_{\text{src}} \times H_{\text{src}}$ (e.g. $1920 \times 1080$ or $640 \times 360$), the target aspect ratio is:
$$R_{\text{target}} = \frac{W_{\text{target}}}{H_{\text{target}}} = \frac{1080}{1920} = \frac{9}{16} = 0.5625$$

To maximize resolution without distortion, we crop a vertical slice of height $H_{\text{crop}}$ and width $W_{\text{crop}}$:
$$H_{\text{crop}} = H_{\text{src}}$$
$$W_{\text{crop}} = \min\left(W_{\text{src}},\; \text{round}(H_{\text{crop}} \cdot R_{\text{target}})\right)$$

To comply with H.264 macroblock constraints (4:2:0 chroma subsampling), all crop dimensions must be strictly even:
$$W_{\text{crop}} \leftarrow W_{\text{crop}} - (W_{\text{crop}} \bmod 2)$$
$$H_{\text{crop}} \leftarrow H_{\text{crop}} - (H_{\text{crop}} \bmod 2)$$

---

### 3.2 Face Centroid & Bounding Box Placement
Let detected faces in frame $k$ have bounding boxes $B_i = [x_i, y_i, w_i, h_i]$. The primary subject centroid is calculated as:
$$c_x = x_i + \frac{w_i}{2}, \quad c_y = y_i + \frac{h_i}{2}$$

The ideal horizontal crop origin $X_{\text{raw}}$ places the speaker's face in the horizontal center of the 9:16 frame:
$$X_{\text{raw}} = c_x - \frac{W_{\text{crop}}}{2}$$

To prevent the crop window from exceeding video boundaries, it is clamped:
$$X_{\text{clamped}} = \max\left(0, \; \min\left(X_{\text{raw}}, \; W_{\text{src}} - W_{\text{crop}}\right)\right)$$

---

### 3.3 Temporal Smoothing & Scene Cut Discontinuity
If the camera pans instantaneously between frame detections, the resulting video suffers from jarring jitter. To achieve smooth, cinematic pan-and-scan camera motion, we apply an **Exponential Moving Average (EMA)** filter to the horizontal crop coordinate:

$$X_t = \alpha \cdot X_{\text{observed}, t} + (1 - \alpha) \cdot X_{t-1}$$
where $\alpha \in (0, 1]$ is the smoothing factor (default $\alpha = 0.25$).

#### Scene Cut Reset Condition:
When a camera shot transitions (e.g., cutting from host to guest), smoothing across the cut causes an unnatural fast pan.
Let the scene difference metric between consecutive frames be $\Delta_{\text{scene}}$. A scene transition is triggered when:
$$\Delta_{\text{scene}} \ge \Theta_{\text{threshold}} \quad (\text{default } \Theta = 27.0)$$

At scene cuts, the EMA state resets instantaneously to avoid panning across discontinuous shots:
$$X_t = X_{\text{observed}, t} \quad \text{if } \text{is\_scene\_cut}(t)$$

```
Frame Index:    t-2        t-1        t (Scene Cut)       t+1        t+2
Raw Detection:  [ 420px ]  [ 425px ]  [ 1200px ]        [ 1210px ]  [ 1205px ]
Filtered X:     [ 420px ]  [ 421px ]  [ 1200px ]*RESET  [ 1202px ]  [ 1203px ]
```

---

### 3.4 Dual-Speaker Stacked Split-Screen Geometry
When two participants are simultaneously engaged (e.g. podcasts or debate interviews) and their horizontal separation satisfies:
$$\frac{|c_{x, 1} - c_{x, 2}|}{W_{\text{src}}} \ge S_{\min} \quad (\text{default } S_{\min} = 0.20)$$

The system activates **Split-Screen Mode**, dividing the $1080 \times 1920$ canvas into two stacked $1080 \times 960$ viewports:
* **Top Viewport** ($1080 \times 960$): Centered around Speaker 1 ($c_{x, 1}$).
* **Bottom Viewport** ($1080 \times 960$): Centered around Speaker 2 ($c_{x, 2}$).

```
┌─────────────────────────────────┐
│                                 │
│       Speaker 1 Viewport        │  1080 x 960 (Crop around c_x1)
│                                 │
├─────────────────────────────────┤
│                                 │
│       Speaker 2 Viewport        │  1080 x 960 (Crop around c_x2)
│                                 │
└─────────────────────────────────┘
 Total Canvas: 1080 x 1920 (9:16)
```

---

### 3.5 Subtitle Timing & Karaoke ASS Tag Math
Advanced SubStation Alpha specifies karaoke duration in **centiseconds** ($1\text{ cs} = 10\text{ ms} = 0.01\text{ s}$).
For each word $w_j$ in a sentence segment:
$$\Delta t_j = t_{\text{end}, j} - t_{\text{start}, j}$$
$$\text{Duration}_{\text{cs}} = \max\left(1, \; \text{round}(\Delta t_j \times 100)\right)$$

The animated ASS dialogue event is formulated as:
$$\text{Dialogue: } 0, \; T_{\text{start}}, \; T_{\text{end}}, \; \text{KaraokeStyle}, \; \dots, \; \sum_{j} \left( \{\backslash\text{k}\text{Duration}_{\text{cs}, j}\} \cdot w_j \right)$$

#### UI Safe Margins:
Mobile platforms (TikTok, Instagram Reels, YouTube Shorts) render UI buttons (like, comment, share, description) in the lower third and top header. To prevent subtitles from being obscured:
$$\text{MarginV} = 250\text{ px}$$
Alignment is set to **Bottom-Center** (`Alignment: 2` in ASS specification).

---

### 3.6 Sentence Boundary Snapping Algorithm
Raw LLM timestamp recommendations frequently truncate sentences prematurely. To ensure complete, professional sentence delivery, timestamps are snapped strictly to transcript sentence boundaries:

$$\hat{t}_{\text{start}} = \arg\min_{s \in S} |s.\text{start} - t_{\text{raw, start}}|$$
$$\hat{t}_{\text{end}} = \arg\min_{s \in S, \; s.\text{end} > \hat{t}_{\text{start}}} |s.\text{end} - t_{\text{raw, end}}|$$
$$\text{subject to: } \hat{t}_{\text{end}} - \hat{t}_{\text{start}} \ge T_{\text{min\_duration}}$$

---

### 3.7 Offline Heuristic Virality Scoring (Zero-AI Mode)
When running without an LLM (`--no-ai` or unconfigured API keys), the system detects highlights using speech density and punctuation heuristics:
1. **Pacing Score**: Measured by word velocity:
   $$V = \frac{\text{Word Count}}{\text{Segment Duration}}$$
   Optimal pacing for viral shorts is $2.8 \text{ to } 3.8\text{ words/sec}$.
2. **Hook Detection**: Scans transcript boundaries for interrogative markers (`?`), exclamatory punchlines (`!`), or opening thesis propositions.
3. **Partition Sampling**: Partitions duration into $K$ target clip windows, snapping each to high-confidence speech boundaries.

---

## 4. Error Handling & Invariant Enforcement

| Component | Error Class | Invariant Enforced | Fallback Strategy |
| :--- | :--- | :--- | :--- |
| **Downloader** | `MediaDownloadError` | Non-empty audio/video stream; valid duration | Retries with alternate format selector |
| **Transcriber** | `TranscriptionError` | Accurate word-level start $\le$ end timestamps | Falls back to CPU INT8 if CUDA unconfigured |
| **Analyzer** | `HighlightAnalysisError` | $15\text{s} \le \text{duration} \le 60\text{s}$; complete sentence | Automatic offline heuristic fallback |
| **Framer** | `FramingError` | $0 \le X_{\text{crop}} \le W_{\text{src}} - W_{\text{crop}}$ | Center crop fallback ($X = \frac{W_s - W_c}{2}$) |
| **Compositor** | `CompositionError` | Non-zero byte output file; even pixel dimensions | Detailed FFmpeg stderr diagnostics |
