"""Generate ``docs/README.md`` from what is actually on disk (IMP-023).

The index used to be maintained by hand, which meant it drifted the moment
a document was added, renamed or deleted — and a drifted index is worse
than no index, because its links look authoritative. This script makes the
index a *derived* artefact:

    python tools/generate_docs_index.py           # rewrite docs/README.md
    python tools/generate_docs_index.py --check   # exit 1 if it is stale

``--check`` is the part that matters. It is a pure comparison, so it can
run in CI without modifying anything: the test
``test_module_size.py::TestDocsIndex::test_generated_index_is_up_to_date``
shells out to it and fails the build when the committed index no longer
matches the tree.

Everything the index needs is derived from the filesystem:

* the directory structure, from ``docs/`` itself;
* the file tables, from a short per-directory blurb plus the file list;
* the plan list, newest first;
* the link check, by resolving every ``](...)`` target in the result.

The one piece of prose that cannot be derived is the header block (the
"state of the tree" note). It lives between two marker comments so that
regeneration preserves it verbatim — hand-written analysis stays hand
written, and only the mechanical part is regenerated.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Dict, List, Tuple

# ``<repo>/tools/generate_docs_index.py`` -> ``<repo>``. The project root
# and the repository root are the same directory now that pyproject.toml
# sits at the top level; there is no package directory to step through.
REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
INDEX_PATH = DOCS_ROOT / "README.md"

#: Markers delimiting the hand-written header. Regeneration copies
#: whatever is between them through untouched.
HEADER_START = "<!-- BEGIN HAND-WRITTEN HEADER -->"
HEADER_END = "<!-- END HAND-WRITTEN HEADER -->"

#: One-line description per directory. The generator owns the *list* of
#: files; it does not own what a directory is for, so the blurb stays
#: editable here rather than being inferred from file contents.
SECTION_BLENDS: Dict[str, Tuple[str, str]] = {
    "architecture": ("Architecture", "How the package is put together."),
    "features": ("Features", "What the library does, one document per capability."),
    "guides": ("Guides", "Task-oriented walkthroughs."),
    "plans": ("Plans", "Dated implementation plans. Historical record, kept in full."),
    "reviews": ("Reviews & Analysis", "Reviews, issue trackers and post-mortems."),
}

#: Link targets that are deliberately not files under ``docs/``.
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#")

#: A trailing ``:38`` / ``:38-52`` / ``:38:4`` on a link target marks a
#: source citation rather than a document reference.
CITATION_SUFFIX = re.compile(r":\d+(?:[-:]\d+)?$")


def _iter_doc_files() -> List[Path]:
    """Every markdown file under ``docs/``, excluding the index itself."""
    return sorted(
        path
        for path in DOCS_ROOT.rglob("*.md")
        if path.resolve() != INDEX_PATH.resolve()
    )


def _relative_link(path: Path) -> str:
    """POSIX path of ``path`` relative to ``docs/`` — the link form used here."""
    return path.relative_to(DOCS_ROOT).as_posix()


def _existing_header() -> str:
    """The hand-written header block, verbatim, markers included.

    On first run (no markers yet) an empty block is emitted so the file
    still has somewhere to put the note.
    """
    if not INDEX_PATH.is_file():
        return (
            f"{HEADER_START}\n"
            "<!-- Anything between these two markers is preserved verbatim by\n"
            "     `tools/generate_docs_index.py`. Put analysis here; the\n"
            "     structure below is regenerated. -->\n"
            f"{HEADER_END}\n"
        )
    text = INDEX_PATH.read_text(encoding="utf-8")
    start = text.find(HEADER_START)
    end = text.find(HEADER_END)
    if start == -1 or end == -1:
        return text.split("\n## Structure", 1)[0].rstrip() + "\n"
    return text[start:end + len(HEADER_END)] + "\n"


def _first_heading(path: Path) -> str:
    """The document's own H1, or its filename when it has none."""
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except OSError:
        pass
    return path.stem


def _describe(path: Path) -> str:
    """A one-line description: the first bolded sentence under the H1.

    Falls back to the first non-empty paragraph. A file with neither is
    listed with an em dash rather than being omitted — an incomplete table
    is visible, a silently-dropped row is not.

    Markdown links in the source paragraph are flattened to their text.
    A table cell is a summary, not a document: carrying the link through
    would import that document's link debt into the index, and a stale
    citation three documents deep is not something regenerating an index
    can fix.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "—"
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped or stripped.startswith(("#", "|", ">", "-", "*")):
            continue
        if stripped.startswith("**") and stripped.endswith("**"):
            return _flatten_links(stripped.strip("*"))
        return _flatten_links(stripped)
    return "—"


def _flatten_links(text: str) -> str:
    """``[label](target)`` becomes ``label``; ``:line`` suffixes are dropped."""
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return CITATION_SUFFIX.sub("", text)


def _dead_links(markdown: str) -> List[str]:
    """Every relative ``](...)`` target in ``markdown`` that resolves to nothing.

    Percent-decoded and resolved against ``docs/``, because a link written
    as ``reviews/Some%20Doc.md`` is a file, not a dead reference.
    """
    dead: List[str] = []
    for target in re.findall(r"\]\(([^)]+)\)", markdown):
        target = target.strip()
        if target.startswith(EXTERNAL_PREFIXES):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        # ``src/hf_track/tracker.py:356`` is a source citation,
        # not a document link — the ``:line`` suffix is part of the
        # convention this repo writes citations in, and resolving it as a
        # filename would report every citation in the index as dead.
        path_part = CITATION_SUFFIX.sub("", path_part)
        decoded = urllib.parse.unquote(path_part)
        if (DOCS_ROOT / decoded).exists():
            continue
        # A target that resolves against the *repository* root is a
        # source citation (``src/hf_track/tracker.py``), not a
        # broken document link. The index cites source constantly and
        # those references go stale for entirely different reasons.
        if (REPO_ROOT / decoded).exists():
            continue
        dead.append(target)
    return sorted(set(dead))


def render() -> str:
    """Build the complete index text from the tree."""
    header = _existing_header()
    docs = _iter_doc_files()

    by_dir: Dict[str, List[Path]] = {}
    for path in docs:
        by_dir.setdefault(path.parent.relative_to(DOCS_ROOT).as_posix(), []).append(path)

    out: List[str] = [header.rstrip(), "", "---", "", "## Structure", ""]

    for name in sorted(by_dir):
        title, blurb = SECTION_BLENDS.get(name, (name, ""))
        files = sorted(by_dir[name])
        out.append(f"### 📁 `/{name}` — {title}")
        out.append("")
        if blurb:
            out.append(f"{blurb}")
            out.append("")
        out.append("| File | Description |")
        out.append("|------|-------------|")
        for path in files:
            link = _relative_link(path)
            out.append(f"| [`{link}`]({link}) | {_describe(path)} |")
        out.append("")

    plans = sorted(
        (p for p in docs if p.parent.name == "plans"),
        key=lambda p: p.stem,
        reverse=True,
    )
    out.append("## Plans")
    out.append("")
    if plans:
        for path in plans:
            link = _relative_link(path)
            out.append(f"- [`{path.name}`]({link}) — {_first_heading(path)}")
    else:
        out.append("_none_")
    out.append("")

    out.append("## Verification")
    out.append("")
    out.append("```console")
    out.append(
        "python tools/generate_docs_index.py --check   # index matches the tree?"
    )
    out.append("```")
    out.append("")

    return "\n".join(out).rstrip() + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if docs/README.md differs from the generated text",
    )
    parser.add_argument(
        "--allow-dead-links",
        action="store_true",
        help="warn about unresolvable links instead of failing",
    )
    args = parser.parse_args(argv)

    generated = render()

    dead = _dead_links(generated)
    if dead and not args.allow_dead_links:
        print("dead links in the generated index:", file=sys.stderr)
        for target in dead:
            print(f"  {target}", file=sys.stderr)
        return 2

    if args.check:
        current = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.is_file() else ""
        if current != generated:
            print(
                "docs/README.md is out of date. Run:\n"
                "  python tools/generate_docs_index.py",
                file=sys.stderr,
            )
            return 1
        print("docs/README.md is up to date.")
        return 0

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(generated, encoding="utf-8")
    print(f"wrote {INDEX_PATH} ({len(_iter_doc_files())} documents indexed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Kept out of __all__ but used by the tests, which pin the date format so a
# regeneration does not produce a spurious diff on a different day.
INDEX_DATE_FORMAT = "%Y-%m-%d"


def today() -> str:
    """Today's date in the index's format."""
    return _dt.date.today().strftime(INDEX_DATE_FORMAT)