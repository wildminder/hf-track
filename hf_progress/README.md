# hf-progress

Plug-and-play progress tracking for HuggingFace Hub uploads and downloads.

## Features

- **Unified API** — Single `HfProgressTracker` class handles all transfer types
- **Auto-detection** — Automatically selects the best method (Xet direct, tqdm_class, or tqdm monkey-patching)
- **Thread-safe** — Run transfers in background threads, consume events from any thread
- **Typed events** — All progress data is structured as `ProgressEvent` dataclasses
- **SSE-ready** — Built-in FastAPI/SSE integration for real-time frontend updates
- **Zero-config** — Works out of the box with just a HuggingFace token

## Installation

```bash
# Basic (HTTP download progress + LFS upload progress)
pip install hf-progress

# With Xet support (detailed upload/download progress with dedup data)
pip install "hf-progress[xet]"

# With SSE integration (FastAPI endpoints)
pip install "hf-progress[sse]"

# Everything
pip install "hf-progress[xet,sse]"
```

## Quick Start

### Upload with Progress

```python
from hf_progress import HfProgressTracker, EventType

tracker = HfProgressTracker(token="hf_...")

# Upload a file (auto-selects Xet or LFS)
result = tracker.upload_file(
    file_path="/path/to/model.safetensors",
    repo_id="username/my-model",
)

# Upload bytes (auto-writes to temp file if needed for LFS progress)
result = tracker.upload_bytes(
    file_content=open("model.bin", "rb").read(),
    filename="model.bin",
    repo_id="username/my-model",
)
```

### Download with Progress

```python
# Download a single file
path = tracker.download_file(
    repo_id="bert-base-uncased",
    filename="config.json",
)

# Download a snapshot (file-count progress, not per-file bytes)
path = tracker.download_snapshot(
    repo_id="bert-base-uncased",
    allow_patterns=["*.json", "*.safetensors"],
)
```

### Consume Progress Events

```python
import threading

# Run transfer in background
def do_upload():
    tracker.upload_file("model.bin", "user/repo", transfer_id="ul-1")

thread = threading.Thread(target=do_upload, daemon=True)
thread.start()

# Consume events from main thread
for event in tracker.events(timeout=1.0, stop_on=EventType.COMPLETE):
    if event.event_type == EventType.PROGRESS:
        print(f"  {event.filename}: {event.percentage:.1f}% "
              f"({event.bytes_completed}/{event.total_bytes}) "
              f"@ {event.speed:.0f} B/s")
    elif event.event_type == EventType.COMPLETE:
        print(f"  ✅ {event.filename} complete!")
    elif event.event_type == EventType.ERROR:
        print(f"  ❌ {event.filename} error: {event.error}")
```

### Wait for Completion

```python
# Blocking wait with timeout
result = tracker.wait_for_complete("ul-1", timeout=300)
if result and result.event_type == EventType.COMPLETE:
    print("Upload finished!")
```

## Architecture

The library uses a **two-tier approach** based on what's available:

| Scenario | Method | Data Quality | How It Works |
|----------|--------|-------------|--------------|
| **Xet upload** | Direct `hf_xet.upload_files()` | ⭐⭐⭐ Excellent | Rust callback with dedup, transfer speed, per-file progress |
| **Xet download** | Direct `hf_xet.download_files()` | ⭐⭐ Good | Per-file `(int)` byte increment callbacks |
| **HTTP download** | `tqdm_class` override | ⭐⭐ Good | Custom tqdm subclass receives `update(n)` per chunk |
| **LFS upload** | tqdm monkey-patching | ⭐⭐ Good | Global tqdm patch filters file-level bars |
| **BytesIO upload** | Write to temp file first | ⭐⭐ Good | Enables tqdm progress for bytes content |

### Event Schema

All events are `ProgressEvent` dataclasses with consistent fields:

```python
@dataclass
class ProgressEvent:
    event_type: EventType        # start, progress, complete, error
    transfer_id: str             # Unique transfer identifier
    direction: TransferDirection # upload or download
    filename: str                # File being transferred
    phase: ProgressPhase         # hashing, uploading, downloading, verifying, complete, error
    bytes_completed: int         # Bytes processed so far
    total_bytes: int             # Total bytes to process
    percentage: float            # 0.0 - 100.0
    speed: float                 # Bytes/second
    # Xet-specific (only when hf_xet is available):
    transfer_bytes_completed: int  # Actual network bytes (may differ due to dedup)
    transfer_bytes_total: int      # Total scheduled network bytes
    transfer_speed: float          # Network transfer speed
    dedup_saved_bytes: int         # Bytes saved by deduplication
```

### Xet Upload Callback Critical Detail

When using `XetUploadProgressCallback` directly with `hf_xet`, the callback
parameter names **MUST** be `total_update` and `item_updates` for the Rust
runtime's `WrappedProgressUpdaterImpl` to detect the detailed callback signature:

```python
# ✅ CORRECT — Rust detects detailed mode
def progress_callback(total_update, item_updates):
    ...

# ❌ WRONG — Rust falls back to simple (int) mode
def progress_callback(a, b):
    ...
```

## SSE Integration

### FastAPI Endpoints

```python
from fastapi import FastAPI
from hf_progress import HfProgressTracker
from hf_progress.integrations.sse import create_progress_router

app = FastAPI()
tracker = HfProgressTracker(token="hf_...")
router = create_progress_router(tracker)
app.include_router(router)
```

This adds:
- `POST /hf-progress/upload` — Start an upload
- `POST /hf-progress/download` — Start a download
- `GET /hf-progress/events/{transfer_id}` — SSE stream
- `GET /hf-progress/status` — Active transfers

### Frontend Consumer

```javascript
const eventSource = new EventSource(`/hf-progress/events/${transferId}`);

eventSource.onmessage = (event) => {
    const progress = JSON.parse(event.data);

    switch (progress.event_type) {
        case "start":
            showProgressBar(progress.filename, progress.total_bytes);
            break;
        case "progress":
            updateProgressBar(progress.filename, progress.percentage, progress.speed);
            if (progress.dedup_saved_bytes > 0) {
                showDedupSavings(progress.dedup_saved_bytes);
            }
            break;
        case "complete":
            markComplete(progress.filename);
            eventSource.close();
            break;
        case "error":
            showError(progress.error);
            eventSource.close();
            break;
    }
};
```

## Low-Level API

For advanced use cases, you can use the individual callback classes directly:

### Xet Upload Callback

```python
from hf_progress import XetUploadProgressCallback
import queue

q = queue.Queue()
callback = XetUploadProgressCallback(
    filename="model.bin",
    total_bytes=1000000,
    event_queue=q,
    transfer_id="my-upload",
    report_interval=0.1,  # 100ms throttle
)

# Pass to hf_xet directly
import hf_xet
results = hf_xet.upload_files(
    file_paths=["model.bin"],
    endpoint=endpoint,
    token_info=(token, expiry),
    token_refresher=None,  # Required! Can be None
    progress_updater=callback,  # Our callback
    _repo_type="model",
)
```

### Download Progress Tqdm

```python
from hf_progress import DownloadProgressTqdm
from huggingface_hub import hf_hub_download

# Create a bound class
tqdm_class = DownloadProgressTqdm.bind(event_queue, "dl-001", "config.json")

# Use with hf_hub_download
path = hf_hub_download(
    repo_id="bert-base-uncased",
    filename="config.json",
    tqdm_class=tqdm_class,
)
```

### Upload tqdm Patcher

```python
from hf_progress import tqdm_upload_patcher
from huggingface_hub import HfApi

api = HfApi(token="hf_...")

with tqdm_upload_patcher(event_queue, transfer_id="ul-001", filename="model.bin"):
    api.upload_file(
        path_or_fileobj="model.bin",
        path_in_repo="model.bin",
        repo_id="username/repo",
    )
```

## Limitations

| Scenario | Limitation | Workaround |
|----------|-----------|------------|
| BytesIO uploads (no Xet) | No progress tracking for bytes | Write to temp file first |
| Small file uploads (<10MB) | Uploaded as git blobs, no progress | No workaround available |
| Files already upstream | Upload skipped, no progress events | Synthetic complete event emitted |
| Multipart LFS uploads | Multiple tqdm bars per file | Aggregate by transfer_id |
| tqdm monkey-patching | Affects ALL tqdm bars globally | Only use in single-threaded contexts |
| `snapshot_download()` | File-count progress only, not per-file bytes | Use `download_file()` per file |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=hf_progress --cov-report=html
```

## License

MIT
