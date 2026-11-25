# Video Processing Module — Multimodal Video Search Engine (Video-RAG)

Stage 1 of the project: **video ingestion + keyframe extraction**.
This module takes a raw video and produces a folder of representative
keyframes plus a `metadata.json` file with exact timestamps — ready to be
fed into a later CLIP/SigLIP embedding stage. No embedding, indexing,
agent, or API code is included here by design.

## Pipeline

```
Video → Read metadata → Sample frames (time interval + scene change)
      → Remove near-duplicates → Save frames + timestamps + metadata.json
```

## Project structure

```
video_rag_processor/
├── video_processor/              # the reusable package
│   ├── __init__.py                # public API: VideoProcessor, ProcessingConfig
│   ├── config.py                  # all tunable settings (ProcessingConfig)
│   ├── utils.py                   # logging, timestamp formatting, filename helpers
│   ├── metadata_extractor.py      # video validation + duration/fps/resolution
│   ├── scene_detector.py          # pluggable scene/shot-change detection
│   ├── dedup.py                   # perceptual-hash near-duplicate filtering
│   └── processor.py               # VideoProcessor: orchestrates the full pipeline
├── examples/
│   └── run_example.py             # CLI script to process a real video file
├── tests/
│   └── test_video_processor.py    # unittest suite (generates its own test video)
├── requirements.txt
└── README.md

processed_videos/                  # created at runtime
└── <video_name>/
    ├── frames/
    │   ├── frame_0000000_t00-00-00.jpg
    │   ├── frame_0000015_t00-00-01.jpg
    │   └── ...
    └── metadata.json
```

## Install

```bash
pip install -r requirements.txt
```

## Run on a sample video

```bash
python examples/run_example.py --video path/to/your_video.mp4 --interval 1.0
```

Useful flags:

```bash
python examples/run_example.py \
  --video path/to/your_video.mp4 \
  --interval 1.5 \                 # sample at least every 1.5s
  --output processed_videos \      # output root folder
  --max-width 720 \                # resize saved frames (keeps aspect ratio)
  --scene-threshold 0.55           # lower = more sensitive scene-change detection
```

## Use as a library

```python
from video_processor import VideoProcessor, ProcessingConfig

config = ProcessingConfig(
    sampling_interval_sec=1.0,
    output_root="processed_videos",
    max_frame_width=720,
)

processor = VideoProcessor(config)
result = processor.process("path/to/video.mp4")

for frame in result["frames"]:
    print(frame["timestamp"], "->", frame["frame_path"])
```

## Run tests

```bash
python -m unittest tests.test_video_processor -v
```

The test suite generates its own short synthetic video (solid-color segments
with abrupt cuts), so no external sample file is required to verify the
pipeline works.

## metadata.json shape

```json
{
  "video_metadata": {
    "filename": "sample_video.mp4",
    "filepath": "/abs/path/sample_video.mp4",
    "duration_sec": 10.0,
    "fps": 15.0,
    "width": 480,
    "height": 360,
    "total_frames": 150
  },
  "processing_config": { "...": "..." },
  "num_frames_extracted": 10,
  "processing_time_sec": 0.315,
  "frames": [
    {
      "frame_filename": "frame_0000000_t00-00-00.jpg",
      "frame_path": "processed_videos/sample_video/frames/frame_0000000_t00-00-00.jpg",
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "timestamp": "00:00:00",
      "timestamp_precise": "00:00:00.000",
      "trigger": "interval",
      "width": 480,
      "height": 360,
      "phash": "a6ae0c92d2bbb05d"
    }
  ]
}
```

`timestamp` (`HH:MM:SS`) is exactly what a later "relevant moment found at
01:23:42" search result needs. `timestamp_sec` and `timestamp_precise`
(millisecond precision) are kept too in case the search UI wants finer
granularity. `trigger` tells you whether a frame was captured because of
uniform time sampling or a detected scene change — useful later for
weighting/debugging retrieval quality.

## Design notes / how each requirement is satisfied

- **Video input & validation**: `metadata_extractor.py` checks the file
  exists, has a plausible extension (mp4/avi/mov/mkv/webm), and can actually
  be opened by OpenCV; it raises `InvalidVideoError` otherwise. Duration,
  fps, width/height and total frame count are all read directly from the
  container via `cv2.VideoCapture`.
- **Configurable time-based sampling**: `ProcessingConfig.sampling_interval_sec`
  controls how often a frame is guaranteed to be saved (e.g. every 1–2s).
  This is converted to a frame-count interval using the video's actual FPS,
  so timestamps stay accurate regardless of frame rate.
- **Scene-change detection, modular**: `scene_detector.py` defines
  `BaseSceneChangeDetector` as an abstract interface; `HistogramSceneChangeDetector`
  is the current lightweight implementation (grayscale histogram
  correlation between consecutive frames). Swapping in a smarter method
  later (SSIM, deep embeddings, PySceneDetect) only means adding a new class
  and registering it in `get_scene_detector`.
- **Duplicate/near-duplicate filtering**: `dedup.py` computes a perceptual
  hash (pHash) for every *candidate* frame and compares it (Hamming
  distance) against the last few *saved* frames, so visually static periods
  don't flood the output.
- **Corrupted frame handling**: the read loop in `processor.py` catches
  `cap.read()` failures and processing exceptions per-frame, logs a
  warning, and continues; it aborts early only if too many consecutive
  reads fail (configurable), rather than crashing the whole run.
- **Output structure**: `processed_videos/<video_name>/frames/*.jpg` +
  `processed_videos/<video_name>/metadata.json`, exactly as requested.
  Frame filenames embed both the zero-padded frame index and the `HH:MM:SS`
  timestamp so they're identifiable at a glance.
- **Resolution for downstream embedding**: frames are resized (default max
  width 720px, configurable, aspect-ratio preserved) before saving — a
  reasonable balance between visual detail for CLIP/SigLIP and disk usage.

## Known limitations / future improvements (student-project scope)

- The scanning loop currently decodes every frame sequentially (`cap.read()`)
  rather than seeking, which is the simplest and most timestamp-accurate
  approach but not the fastest for very long videos. A future optimization
  could use `cap.grab()` to cheaply skip frames between sampling points and
  only `cap.retrieve()`/decode when a frame is actually a candidate.
- Scene-change detection is intentionally simple (global histogram
  correlation) so it's fast and has no extra ML dependencies. It's isolated
  behind `BaseSceneChangeDetector` specifically so it can be upgraded later.
- Not covered here (by design, per project scope): CLIP/SigLIP embeddings,
  Qdrant/FAISS indexing, the search agent, FastAPI service, and any
  frontend — this module only produces the frames + timestamps + metadata
  that those later stages will consume.
