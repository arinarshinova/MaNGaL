from __future__ import annotations

from pathlib import Path
import os
import re
import zipfile
from functools import lru_cache
from typing import Set


def normalize_star_id(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "", str(name))
    return s.upper()


def _collect_from_directory(base: Path) -> Set[str]:
    ids: set[str] = set()
    if not base.is_dir():
        return ids
    for p in base.glob("m*.dat"):
        stem = p.stem
        if not stem:
            continue
        ids.add(normalize_star_id(stem[1:]))
    return ids


def _collect_from_zip(zip_path: Path) -> Set[str]:
    ids: set[str] = set()
    if not zip_path.exists():
        return ids
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                base = Path(name).name
                if not base.lower().startswith("m") or not base.lower().endswith(".dat"):
                    continue
                stem = Path(base).stem
                if stem:
                    ids.add(normalize_star_id(stem[1:]))
    except Exception:
        return set()
    return ids


@lru_cache()
def load_mangal_standards(okestan_dir: str | Path | None = None) -> Set[str]:
    candidates_dirs = []
    candidates_zips = []

    if okestan_dir is not None:
        p = Path(okestan_dir).expanduser()
        if p.suffix.lower() == ".zip":
            candidates_zips.append(p)
        else:
            candidates_dirs.append(p)
    else:
        env = os.environ.get("MANGAL_OKESTAN_DIR")
        if env:
            p = Path(env).expanduser()
            if p.suffix.lower() == ".zip":
                candidates_zips.append(p)
            else:
                candidates_dirs.append(p)

        cwd = Path.cwd()
        candidates_dirs.append(cwd / "okestan")
        candidates_zips.append(cwd / "okestan.zip")
        # standard-star SEDs shipped with the pipeline
        candidates_dirs.append(Path(__file__).resolve().parent / "data" / "okestan")

    ids: set[str] = set()
    for p in candidates_dirs:
        ids |= _collect_from_directory(p)
    for p in candidates_zips:
        ids |= _collect_from_zip(p)
    return ids


def load_mangal_standards_for_night(night_dir: str | Path | None) -> Set[str]:
    ids: set[str] = set()
    if night_dir is not None:
        n = Path(night_dir).expanduser()
        probe_dirs = [n / "okestan", n.parent / "okestan"]
        probe_zips = [n / "okestan.zip", n.parent / "okestan.zip"]
        for p in probe_dirs:
            ids |= _collect_from_directory(p)
        for p in probe_zips:
            ids |= _collect_from_zip(p)
        if ids:
            return ids
    return load_mangal_standards()


