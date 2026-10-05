<a id="readme-top"></a>

<p align="center">▁▂▃▄▅▆▇█▇▆▅▄▃▂▁</p>

<div align="center">

# █ HF-Track █

<img  alt="image" src="https://github.com/user-attachments/assets/b38684c4-d22f-4683-80ec-9c554475d713" />
</div>


<p align="center">

  <b>Byte-level progress for Hugging Face transfers, including the Xet paths
  <code>huggingface_hub</code> reports nothing about.</b>
</p>

<div align="center">

[![Python][python-shield]][python-url]
[![License][license-shield]][license-url]

</div>

```bash
pip install hf-track# HTTP downloads, LFS uploads
pip install "hf-track[xet]"   # + Xet storage (faster, dedup-aware)
pip install "hf-track[sse]"   # + FastAPI / Server-Sent Events
```

<details>
<summary><b>◈ Table of Contents</b></summary>

- [The gap](#the-gap)
- [How it resolves](#how-it-resolves)
- [Choosing a method](#choosing-a-method)
- [Downloads](#downloads)
- [Uploads](#uploads)
- [Events](#events)
- [Cancellation](#cancellation)
- [Async](#async)
- [Server-sent events](#sse)
- [Retries](#retries)
- [Limitations](#limitations)
- [Development](#development)

</details>

<p id="the-gap" align="center">▰ ▰ ▰ ▰ ▰ ▰ ▰</p>

## ◈ The gap

`huggingface_hub` reports progress unevenly, and the gaps are load-bearing.

| Transfer | What `huggingface_hub` gives you |
| :--- | :--- |
| HTTP download | a `tqdm` bar |
| LFS upload | a `tqdm` bar |
| **Xet download** | **nothing.** Xet stores files content-addressed, so `hf_hub_download` has no byte count to report |
| **Xet upload** | **nothing.** progress arrives inside the Rust extension, not in Python |

<p align="center">· · · · ·</p>

Two further problems compound it, and both belong to the Xet runtime instead of
your code.

The runtime holds chunks in memory and flushes at the end, so a 2 GB file shows
0% for most of its life and then jumps to 100%. It also cannot be cancelled
in-process: `hf_xet` compiles to a `.pyd` that starts background threads, and
once imported, a stalled transfer cannot be interrupted. Not with
`KeyboardInterrupt`, not with a cancel flag, not by letting the GIL go. Killing
the process is the only way out.

So you are choosing between a library that is blind on your largest files and
one that can hang your program permanently.

<p id="how-it-resolves" align="center">▰ ▰ ▰ ▰ ▰ ▰ ▰</p>

## ◈ How it resolves

| Mechanism | What it buys |
| :--- | :--- |
| Subprocess isolation | Every Xet transfer runs in a child process, so the Rust threads die with it. Cancelling signals a cooperative stop, then escalates `SIGTERM` → `SIGKILL` |
| Progress from where the bytes are | With the runtime in its own process it can be asked directly: real byte counts, per-file progress, network speed, dedup savings |
| Credential drift in one module | `huggingface_hub`'s private credential helpers live in `token/manager.py`, behind an upper bound on the dependency, so 1.x dropping the two this package used cannot arrive silently through a solver |

The parent holds its lock only long enough to snapshot the child handle, never
across a join, so liveness checks stay responsive while a child is terminating.


| Path | Used for | Progress quality |
| :--- | :--- | :---: |
| `download_file` on a Xet repo | Xet-stored files | ![dedup][chip-dedup] |
| `download_snapshot_streaming` | whole repos, byte-level per file | ![dedup][chip-dedup] |
| `download_snapshot` | whole repos, per-file counting | ![filecount][chip-filecount] |
| `tqdm_class` override | plain HTTP downloads | ![bytes][chip-bytes] |
| `tqdm` monkey-patch | LFS uploads via `HfApi` | ![bytes][chip-bytes] |
| subprocess wrapper | all of the above | ![cancel][chip-cancel] |

The `transfer_id` you pass correlates concurrent transfers and is what
`cancel()` takes. Omit it and one is generated per call.

<p align="center"><img  width="60%" src="https://github.com/user-attachments/assets/413653f2-8032-48bc-b2e8-5d234911f087" /></p>


<p id="choosing-a-method" align="center">▰ ▰ ▰ ▰ ▰ ▰ ▰</p>

## ◈ Choosing a method

| I want to… | Use | Progress you get |
| :--- | :--- | :--- |
| Fetch one file | `download_file()` | byte-level |
| Fetch a repo and watch every byte | `download_snapshot_streaming()` | byte-level, per file, cancellable |
| Fetch a repo, cheapest option | `download_snapshot()` | file count |
| Push a checkpoint | `upload_folder()` | byte-level over Xet |
| Push one file | `upload_file()` | byte-level over Xet |
| Avoid Xet entirely | any method with `use_xet=False` | byte-level over HTTP |

`download_snapshot_streaming()` runs a three-tier strategy per file: Xet first,
then a plain HTTP fallback if the Xet handle has not completed within
`tier_timeout_s` (60 s by default). Set `enable_http_fallback=False` to raise
instead of falling back.

<p id="downloads" align="center">▰ ▰ ▰ ▰ ▰ ▰ ▰</p>

## ◈ Downloads

<a id="quick-start"></a>

### Quick start

```python
from hf_track import HfTracker, EventType

tracker = HfTracker(token="hf_...")   # token optional for public repos

path = tracker.download_file("bert-base-uncased", "config.json")
path = tracker.download_snapshot("user/repo", allow_patterns=["*.safetensors"])
paths = tracker.download_snapshot_streaming("user/repo")

for event in tracker.events(timeout=1.0, stop_on=EventType.COMPLETE):
    if event.event_type is EventType.PROGRESS:
        print(f"{event.filename}: {event.percentage:.1f}% "
              f"({event.bytes_completed}/{event.total_bytes}) @ {event.speed:.0f} B/s")
```

<a id="download-options"></a>

### Per-call options

Every transfer accepts `transfer_id=` to correlate concurrent work, and
`use_xet=False` to force the reliable HTTP path. `download_file()` also takes
`repo_type`, `revision`, and `local_dir`, which pass straight through to
`huggingface_hub`.

<p id="uploads" align="center">▰ ▰ ▰ ▰ ▰ ▰ ▰</p>

## ◈ Uploads

```python
tracker.upload_file("model.safetensors", "user/repo")
tracker.upload_bytes(b"...", "notes.txt", "user/repo")   # bytes, not str
tracker.upload_folder("./checkpoint", "user/repo")
```

`upload_bytes()` takes `bytes`. Payloads over 10 MB are staged through a
temporary file, because the Xet upload path wants a file handle.

<p id="events" align="center">▰ ▰ ▰ ▰ ▰ ▰ ▰</p>

## ◈ Events

Every operation emits typed `ProgressEvent` dataclasses onto the tracker's
queue: `START`, `PROGRESS`, `COMPLETE`, `ERROR`, `CANCELLED`. Each event also
carries a `phase` (`hashing`, `uploading`, `downloading`, `verifying`,
`complete`, `error`).

```python
event = tracker.events(timeout=1.0, stop_on=EventType.COMPLETE)
recent = tracker.get_events()          # already-collected events
tracker.wait_for_complete("ul-1", timeout=300)
```

| Field | Meaning |
| :--- | :--- |
| `bytes_completed` / `total_bytes` | logical bytes for this file |
| `transfer_bytes_*`, `transfer_speed` | network bytes, which differ from the above when Xet dedups |
| `dedup_saved_bytes` | bytes never sent because the content already existed upstream |
| `file_index` / `total_files` | position within a multi-file transfer |
| `extra["transport"]` | `"xet"` or `"http"`, so you can tell which path carried it |

`event.to_dict()` produces stable JSON. `ProgressEvent.from_dict()` reads it
back, which is what makes the SSE and WebSocket paths possible.

<p id="cancellation" align="center">▰ ▰ ▰ ▰ ▰ ▰ ▰</p>

## ◈ Cancellation

```python
tracker.cancel("ul-1")
```

On a subprocess-backed transfer this signals the child cooperatively, gives it
`grace` seconds to exit on its own, then escalates `SIGTERM` → `SIGKILL`. The
transfer always ends, one way or another: a cancelled transfer raises
`TransferCancelledError` and the call returns.

<p id="async" align="center">▰ ▰ ▰ ▰ ▰ ▰ ▰</p>

## ◈ Async

```python
path = await tracker.download_file_async("user/repo", "config.json")
```

Wrappers exist for `download_file`, `download_snapshot`, `upload_file`, `upload_bytes`, and `upload_folder`. Cancelling the awaiting task cancels the underlying transfer, so abandoning an `await` does not leave a thread running.

<p id="sse" align="center">▰ ▰ ▰ ▰ ▰ ▰ ▰</p>

## ◈ Server-sent events

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

The module re-exports `sse_starlette.EventSourceResponse` and nothing else. The old `create_progress_router()` factory is gone, having broken Starlette 1.0's response serialization. A complete FastAPI application, with endpoints defined at module level, is in [`examples/web_app/app.py`](examples/web_app/app.py).

<p id="retries" align="center">▰ ▰ ▰ ▰ ▰ ▰ ▰</p>

## ◈ Retries

The Xet download paths retry transient failures (`ConnectionError`,
`TimeoutError`, `OSError`) with exponential backoff: 3 retries by default, a
0.5 s base delay, and an 8 s ceiling per wait. Backoff polls in slices, so a
`cancel()` issued mid-wait is still observed.

<p id="limitations" align="center">▰ ▰ ▰ ▰ ▰ ▰ ▰</p>

## ◈ Limitations

| Scenario | Behaviour | Option |
| :--- | :--- | :--- |
| Payloads under 10 MB | uploaded as git blobs, no byte progress | — |
| File already upstream | skipped, emits `COMPLETE` | — |
| `tqdm` monkey-patching | affects every `tqdm` bar in the process | avoid in threaded code |
| Event queue full (10 000) | the overflowing `PROGRESS` event is dropped and logged | consume events promptly |

<p id="development" align="center">▰ ▰ ▰ ▰ ▰ ▰ ▰</p>

## ◈ Development

```bash
pip install -e ".[dev,xet]"

pytest                      # unit suite
pytest -m integration       # requires network access to huggingface.co
pytest --cov                # with coverage
ruff check src              # the blocking lint gate: F821, F811
```

Src layout, with tests in `tests/`. CI runs the unit suite on Linux **and** Windows, because process spawn behaves differently there and a Linux-only matrix hid a class of ordering bug. The integration job is `continue-on-error`: it hits `huggingface.co` and must never gate a commit.

The ruff gate covers `F821` and `F811` only. Those are the two rules that hide runtime breakage: a name that does not exist at the point of use, and a redefinition that silently shadows the original.

> [!NOTE]
> The suite is **748 unit tests plus 5 integration tests**, of which one unit
> test (`test_xet_streaming_realworld.py`) needs `huggingface.co` to be
> reachable and fails without it. Expect `1 failed, 747 passed, 5 deselected`
> offline.


<!-- HEADER BADGES -->
[value-shield]: https://img.shields.io/badge/what%20you%20get-byte--level%20%2B%20dedup%20%2B%20cancellable-8957e5?style=for-the-badge&logo=huggingface&logoColor=white
[the-gap]: #the-gap
[python-shield]: https://img.shields.io/badge/python-3.9%2B-0077cc?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org
[license-shield]: https://img.shields.io/badge/License-MIT-8957e5?style=for-the-badge&logo=opensourceinitiative&logoColor=white
[license-url]: LICENSE
[pypi-shield]: https://img.shields.io/pypi/v/hf-track?style=for-the-badge&logo=pypi&logoColor=white
[pypi-url]: https://pypi.org/project/hf-track/
[tests-shield]: https://img.shields.io/badge/tests-753%20collected-0077cc?style=for-the-badge&logo=pytest&logoColor=white
[tests-url]: #development
[ci-shield]: https://img.shields.io/badge/ci-unit%20on%20linux%20%2B%20windows-28a745?style=for-the-badge&logo=githubactions&logoColor=white
[ci-url]: .github/workflows/lint.yml

<!-- BODY CHIPS -->
[chip-bytes]: https://img.shields.io/badge/byte--level-0077cc?style=flat-square
[chip-dedup]: https://img.shields.io/badge/byte--level%20%2B%20dedup-28a745?style=flat-square
[chip-filecount]: https://img.shields.io/badge/file%20count-fe7d37?style=flat-square
[chip-cancel]: https://img.shields.io/badge/cancellable-6f42c1?style=flat-square