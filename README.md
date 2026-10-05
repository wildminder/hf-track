# hf-track

Progress tracking for HuggingFace Hub transfers — including the transfers `huggingface_hub` reports nothing about.

[![Python][python-shield]][python-url]
[![License][license-shield]][license-url]
[![PyPI][pypi-shield]][pypi-url]
[![Tests][tests-shield]][tests-url]

```bash
pip install hf-track              # HTTP downloads, LFS uploads
pip install "hf-track[xet]"        # + Xet storage (faster, dedup-aware)
pip install "hf-track[sse]"        # + FastAPI / Server-Sent Events
```

---

<details>
<summary>Table of Contents</summary>

- [Why this exists](#why-this-exists)
- [How it resolves it](#how-it-resolves-it)
- [Usage](#usage)
- [Cancellation](#cancellation)
- [Async](#async)
- [Events](#events)
- [SSE](#sse)
- [Limitations](#limitations)
- [Development](#development)
- [License](#license)

</details>

---

## Why this exists

`huggingface_hub` reports progress unevenly, and the gaps are not cosmetic:

| Transfer | What `huggingface_hub` gives you |
| :--- | :--- |
| HTTP download | a `tqdm` bar |
| LFS upload | a `tqdm` bar |
| **Xet download** | **nothing** — Xet stores files content-addressed, so `hf_hub_download` has no byte count to report |
| **Xet upload** | **nothing** — progress arrives inside the Rust extension, not in Python |

Two further problems compound this:

**Xet buffers.** The Rust runtime holds chunks in memory and flushes at the
end, so a 2 GB file shows 0% for most of its life, then jumps to 100%.

**Xet cannot be cancelled in-process.** `hf_xet` compiles to a `.pyd` that
starts background threads. Once imported, a stalled transfer cannot be
interrupted — not with `KeyboardInterrupt`, not with a cancel flag, not by
letting the GIL go. The only way out is to kill the process.

So you are left choosing between a library that is blind on your largest
files and one that can hang your program permanently.

## How it resolves it

**Subprocess isolation.** Every Xet transfer runs in a child process. The
Rust threads live only in the child, so cancelling means signalling a
cooperative stop, then escalating `SIGTERM` → `SIGKILL`. The parent holds a
lock only long enough to snapshot the child handle — never across a join —
so liveness checks stay responsive during termination.

**Progress from where the bytes are.** With the runtime in its own process,
it can be asked directly. Xet transfers report real byte counts, per-file
progress, network speed, and dedup savings.

**Credential drift is contained.** The library talks to `hf_xet` through
`XetSession`, and `huggingface_hub`'s private credential helpers change
between releases. That coupling lives in one module behind a CI gate that
fails on undefined names, instead of being discovered mid-download.

| Path | Used for | Progress quality |
| :--- | :--- | :---: |
| `hf_xet` download group | Xet-stored files | byte-level, + dedup |
| `tqdm_class` override | plain HTTP downloads | byte-level |
| `tqdm` monkey-patch | LFS uploads via `HfApi` | byte-level |
| subprocess wrapper | all of the above | cancellable |

## Usage

```python
from hf_track import HfTracker, EventType

tracker = HfTracker(token="hf_...")   # token optional for public repos

path = tracker.download_file("bert-base-uncased", "config.json")
path = tracker.download_snapshot("user/repo", allow_patterns=["*.safetensors"])
path = tracker.download_snapshot_streaming("user/repo")   # byte-level, per file
tracker.upload_file("model.safetensors", "user/repo")
tracker.upload_bytes(b"...".decode(), "notes.txt", "user/repo")
tracker.upload_folder("./checkpoint", "user/repo")

for event in tracker.events(timeout=1.0, stop_on=EventType.COMPLETE):
    if event.event_type is EventType.PROGRESS:
        print(f"{event.filename}: {event.percentage:.1f}% "
              f"({event.bytes_completed}/{event.total_bytes}) @ {event.speed:.0f} B/s")
```

Transfer methods accept `transfer_id=` if you want to correlate concurrent
transfers, and `use_xet=False` to force the reliable HTTP path.

## Cancellation

```python
tracker.cancel("ul-1")
event = tracker.wait_for_complete("ul-1", timeout=300)
```

On a subprocess-backed transfer, `cancel()` signals the child cooperatively.
If it does not exit, the runner escalates to `SIGTERM` then `SIGKILL` — the
transfer always ends, and a cancelled transfer raises `TransferCancelledError`
rather than hanging.

## Async

```python
path = await tracker.download_file_async("user/repo", "config.json")
```

Wrappers exist for `download_file`, `download_snapshot`, `upload_file`,
`upload_bytes` and `upload_folder`. Cancelling the awaiting task cancels the
underlying transfer, so abandoning an `await` does not leave a thread running.

## Events

Every operation emits typed `ProgressEvent` dataclasses onto the tracker's
queue — `START`, `PROGRESS`, `COMPLETE`, `ERROR`, `CANCELLED`.

```python
event = tracker.events(timeout=1.0, stop_on=EventType.COMPLETE)
recent = tracker.get_events()          # already-collected events
```

| Field | Meaning |
| :--- | :--- |
| `bytes_completed` / `total_bytes` | logical bytes for this file |
| `transfer_bytes_*`, `transfer_speed` | network bytes — differs from the above when Xet dedups |
| `dedup_saved_bytes` | bytes not sent because the content already existed |
| `file_index` / `total_files` | position within a multi-file transfer |

Serialise with `event.to_dict()` — the output is stable JSON.

## SSE

```python
from hf_track.integrations.sse import EventSourceResponse

@app.get("/events/{transfer_id}")
async def stream(transfer_id: str, request: Request):
    async def gen():
        while not await request.is_disconnected():
            for e in tracker.get_events():
                if e.transfer_id == transfer_id:
                    yield {"data": e.to_dict()}
            await asyncio.sleep(0.1)
    return EventSourceResponse(gen())
```

A complete FastAPI application is in
[`examples/web_app/app.py`](examples/web_app/app.py).

## Limitations

| Scenario | Behaviour | Option |
| :--- | :--- | :--- |
| Files under ~10 MB | uploaded as git blobs, no progress | — |
| File already upstream | skipped, emits `COMPLETE` | — |
| `tqdm` monkey-patching | affects every `tqdm` bar in the process | avoid in threaded code |
| `snapshot_download()` | file-count progress, not bytes | `download_file()` per file |

## Development

```bash
pip install -e ".[dev,xet]"

pytest                      # unit suite
pytest -m integration       # requires network access to huggingface.co
pytest --cov                # with coverage
ruff check src              # the blocking lint gate
```

The project uses a src layout: `src/hf_track/`, tests in `tests/`. CI runs the
suite on Linux **and Windows** — process spawn behaves differently on
Windows, and Linux-only testing hides that.

Further reading lives in [`docs/`](docs/README.md), including the
[technical review](docs/reviews/2026-10-04-technical-review.md) and the
[issues tracker](docs/reviews/issues-improvements.md).

## License

MIT

<!-- HEADER BADGES -->
[python-shield]: https://img.shields.io/badge/python-3.9%2B-blue
[python-url]: https://www.python.org
[license-shield]: https://img.shields.io/badge/license-MIT-green
[license-url]: LICENSE
[pypi-shield]: https://img.shields.io/badge/pypi-hf--track-orange
[pypi-url]: https://pypi.org/project/hf-track/
[tests-shield]: https://img.shields.io/badge/tests-753%20passing-brightgreen
[tests-url]: .github/workflows/lint.yml
