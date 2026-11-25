"""Read the official WDC Products 50%-corner-case pair-wise benchmark.

Accept official 50pair.zip or a directory of extracted *.json.gz files.
ZIP members are streamed; no archive paths are extracted to disk.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

WDC_50_PAIR_URL = (
    "https://data.dws.informatik.uni-mannheim.de/"
    "largescaleproductcorpus/data/wdc-products/50pair.zip"
)
MAX_JSONL_ROWS = 100_000
MAX_MEMBER_SIZE = 150_000_000


class SplitSelectionError(ValueError):
    """Raised if source members cannot be selected uniquely and safely."""


def file_sha256(path: Path) -> str:
    """Compute a stable digest of the downloaded artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def available_members(source: Path) -> list[str]:
    """Return sorted relative JSONL-gzip members, never extracting ZIP content."""
    if source.is_dir():
        return sorted(p.relative_to(source).as_posix() for p in source.rglob("*.json.gz"))
    if source.is_file() and source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            return sorted(
                member.filename for member in archive.infolist()
                if not member.is_dir() and member.filename.lower().endswith(".json.gz")
            )
    raise FileNotFoundError(f"expected a .zip archive or extracted directory: {source}")


def _role_candidates(members: list[str], role: str) -> list[str]:
    results = []
    for member in members:
        name = Path(member).name.lower()
        norm = re.sub(r"[^a-z0-9]+", "_", name)
        if "multi" in name or not name.endswith(".json.gz"):
            continue
        if role == "train":
            valid = "train" in norm and "medium" in norm
        elif role == "validation":
            valid = ("valid" in norm or "val_" in norm) and "medium" in norm
        elif role == "test_seen":
            valid = ("test" in norm or "gs_" in norm) and "000un" in norm
        elif role == "test_unseen":
            valid = ("test" in norm or "gs_" in norm) and "100un" in norm
        else:
            raise SplitSelectionError(f"unknown role: {role}")
        if valid:
            results.append(member)
    return results


def select_split_members(
    members: list[str], overrides: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Resolve official train, validation, seen-test and unseen-test members."""
    overrides = dict(overrides or {})
    roles = ("train", "validation", "test_seen", "test_unseen")
    if set(overrides) - set(roles):
        raise SplitSelectionError(f"unknown overrides: {sorted(set(overrides) - set(roles))}")

    selected: dict[str, str] = {}
    for role in roles:
        if role in overrides:
            member = overrides[role]
            if member not in members:
                raise SplitSelectionError(
                    f"{role}: member {member!r} not in archive: {members}"
                )
            selected[role] = member
            continue
        candidates = _role_candidates(members, role)
        if role in {"train", "validation"} and len(candidates) > 1:
            seen = [m for m in candidates if "000un" in m.lower()]
            if len(seen) == 1:
                candidates = seen
        if len(candidates) != 1:
            raise SplitSelectionError(
                f"cannot uniquely select {role}: candidates={candidates}. "
                "Use --list-members followed by explicit --*-member arguments."
            )
        selected[role] = candidates[0]
    if len(set(selected.values())) != len(selected):
        raise SplitSelectionError("a JSONL member was assigned to multiple split roles")
    return selected


@contextmanager
def _member_stream(source: Path, member: str) -> Iterator[io.TextIOBase]:
    if source.is_dir():
        root = source.resolve()
        path = (source / member).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise SplitSelectionError(f"invalid member path: {member}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            yield stream
    else:
        with ZipFile(source) as archive:
            info = archive.getinfo(member)
            if info.file_size > MAX_MEMBER_SIZE:
                raise SplitSelectionError(f"compressed member too large: {member}")
            with archive.open(info) as raw:
                with gzip.GzipFile(fileobj=raw, mode="rb") as decoded:
                    with io.TextIOWrapper(decoded, encoding="utf-8") as stream:
                        yield stream


def read_jsonl_member(source: Path, member: str) -> list[dict[str, object]]:
    """Load gzip JSON Lines with explicit error locations and a row limit."""
    rows: list[dict[str, object]] = []
    with _member_stream(source, member) as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number > MAX_JSONL_ROWS:
                raise SplitSelectionError(f"{member}: exceeded safe row limit")
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SplitSelectionError(f"{member}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise SplitSelectionError(f"{member}:{line_number}: expected JSON object")
            rows.append(value)
    return rows
