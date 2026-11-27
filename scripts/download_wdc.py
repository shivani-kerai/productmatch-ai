"""Download the official WDC 50% corner-case pairwise archive.

Network access is only required to run this script; CI uses local synthetic fixtures.
The source URL is fixed to the official dataset host, not a user-supplied endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from productmatch.data.ingestion import WDC_50_PAIR_URL

MAX_DOWNLOAD_BYTES = 90 * 1024 * 1024


def download_official(destination: Path) -> tuple[int, str]:
    """Download atomically, fail closed on oversize/truncated/non-ZIP payloads."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists; do not overwrite an unverified dataset"
        )

    size = 0
    sha = hashlib.sha256()
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".wdc-", suffix=".part", dir=destination.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            request = urllib.request.Request(
                WDC_50_PAIR_URL,
                headers={"User-Agent": "productmatch-ai/0.2 (research benchmark download)"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError("download exceeds 90 MiB safety limit")
                    temporary.write(chunk)
                    sha.update(chunk)
        with ZipFile(temporary_name) as archive:
            if not archive.namelist() or archive.testzip() is not None:
                raise ValueError("dataset ZIP is empty or corrupt")
        Path(temporary_name).replace(destination)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
    return size, sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/raw/wdc/50pair.zip"))
    args = parser.parse_args()
    size, digest = download_official(args.dest)
    print(f"Saved {args.dest} ({size:,} bytes) SHA256={digest}")


if __name__ == "__main__":
    main()
