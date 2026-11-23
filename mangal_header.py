"""
FITS header bookkeeping for the products of the pipeline (pure Python + pandas, no astropy import).

A product that is a SUM of many frames must not carry the keywords of the first frame. Like IDL
addexp.pro (EXPTIME = total(texp), ZMEAN = airmass-weighted zenith distance, FILE removed, the input
names in HISTORY) this module sets, from the per-frame table of the sidecar `<cube>.inputs.csv`:

    EXPTIME   total exposure [s] of the frames that went into the product
    NFRAMES   number of those frames
    ZMEAN     mean zenith distance [deg]: airmass 1/cos(z) averaged with the exposures as weights
    ZSPACE    kept only if all frames have the same value (otherwise removed)
    DATE-OBS / TIME-OBS / UT   start of the observation: first frame of the cube
    FILENAME  name of THIS file (not of the first raw frame)
    DATE      when THIS file was created, UT
    HISTORY   FRAMES: t261010_0801-0836 (36 frames) ...
ZDIST (zenith distance of one frame) is removed: it is replaced by ZMEAN.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# keywords of every raw frame that are stored per slice in the sidecar (besides FILE, ZSPACE, EXPTIME)
RAW_KEYS = ("ZDIST", "DATE-OBS", "TIME-OBS", "UT")
_START_KEYS = ("DATE-OBS", "TIME-OBS", "UT")


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def stamp(hdr, path) -> None:
    """FILENAME / DATE of the file that is being written."""
    hdr["FILENAME"] = (Path(path).name[:68], "Name of this file")
    hdr["DATE"] = (utc_now_str(), "file creation date (YYYY-MM-DDThh:mm:ss UT)")


def _del(hdr, key) -> None:
    try:
        if key in hdr:
            del hdr[key]
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# per-frame metadata (written by step 3 into the sidecar)
# --------------------------------------------------------------------------------------
def frame_meta_from_header(name: str, hdr) -> dict:
    """One sidecar row from the header of a raw frame."""
    try:
        z = int(hdr.get("ZSPACE"))
    except Exception:
        z = None
    try:
        t = float(hdr.get("EXPTIME", hdr.get("EXPOSURE", np.nan)))
    except Exception:
        t = float("nan")
    row = {"FILE": name, "ZSPACE": z, "EXPTIME": t}
    for k in RAW_KEYS:
        try:
            v = hdr.get(k)
        except Exception:
            v = None
        row[k] = "" if v is None else v
    return row


# --------------------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------------------
def mean_zenith(zdist, texp) -> float | None:
    """
    IDL addexp.pro: air = 1/cos(zdist); mair = sum(air*texp)/sum(texp); zmean = acos(1/mair).
    Only frames with zdist > 0 count. None if there are none.
    """
    z = pd.to_numeric(pd.Series(list(zdist)), errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(pd.Series(list(texp)), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(z) & (z > 0) & (z < 90)
    if not ok.any():
        return None
    t = np.where(np.isfinite(t) & (t > 0), t, 1.0)[ok]
    air = 1.0 / np.cos(np.radians(z[ok]))
    mair = float(np.sum(air * t) / np.sum(t))
    return float(np.degrees(np.arccos(1.0 / mair)))


def combine_zmean(zmeans, texps) -> float | None:
    """ZMEAN of a combination of products that have their own ZMEAN and exposure (exact for the IDL definition)."""
    return mean_zenith(zmeans, texps)


def frame_ranges(names) -> str:
    """['t261010_0801', ..., 't261010_0836'] -> 't261010_0801-0836' (consecutive numbers are merged)."""
    out: list[str] = []
    run: list[tuple[str, str, int]] = []  # (prefix, digits, number)

    def flush():
        if not run:
            return
        a, b = run[0], run[-1]
        out.append(f"{a[0]}{a[1]}" if len(run) == 1 else f"{a[0]}{a[1]}-{b[1]}")
        run.clear()

    for nm in names:
        m = re.match(r"^(.*?)(\d+)$", str(nm))
        if not m:
            flush()
            out.append(str(nm))
            continue
        item = (m.group(1), m.group(2), int(m.group(2)))
        if run and run[-1][0] == item[0] and len(run[-1][1]) == len(item[1]) and item[2] == run[-1][2] + 1:
            run.append(item)
        else:
            flush()
            run.append(item)
    flush()
    return ", ".join(out)


def _drop_history(hdr, prefix: str) -> None:
    """Remove HISTORY cards that start with prefix (so that repeated stages do not pile up FRAMES lines)."""
    try:
        old = [str(x) for x in hdr["HISTORY"]]
        keep = [x for x in old if not x.startswith(prefix)]
        if len(keep) == len(old):
            return
        del hdr["HISTORY"]
        for x in keep:
            hdr["HISTORY"] = x
    except Exception:
        pass


def _hist_chunks(text: str, prefix: str, width: int = 66):
    words = text.split(", ")
    line = prefix
    for w in words:
        add = (w if line == prefix else ", " + w)
        if len(line) + len(add) > width and line != prefix:
            yield line
            line = prefix + w
        else:
            line += add
    if line != prefix:
        yield line


# --------------------------------------------------------------------------------------
# header of a product made of several frames
# --------------------------------------------------------------------------------------
def frame_summary(frames: pd.DataFrame) -> dict:
    """{"texp", "n", "zmean"} of a per-frame table (columns EXPTIME, optional ZDIST)."""
    t = pd.to_numeric(frames["EXPTIME"], errors="coerce")
    texp = float(t[t > 0].sum()) if len(frames) else 0.0
    zm = mean_zenith(frames["ZDIST"], t) if "ZDIST" in frames.columns else None
    return {"texp": texp, "n": int(len(frames)), "zmean": zm}


def apply_frame_keys(hdr, frames: pd.DataFrame, out_path=None, *, history: bool = True) -> dict:
    """
    Write EXPTIME / NFRAMES / ZMEAN / ZSPACE / DATE-OBS ... for the frames in `frames` into hdr (see module
    docstring), stamp FILENAME and DATE when out_path is given. Returns frame_summary(frames).
    """
    frames = frames.reset_index(drop=True)
    info = frame_summary(frames)
    bad = pd.to_numeric(frames["EXPTIME"], errors="coerce").isna() | (pd.to_numeric(frames["EXPTIME"], errors="coerce") <= 0)

    hdr["EXPTIME"] = (info["texp"], "Total exposure time [s] of all frames")
    hdr["NFRAMES"] = (info["n"], "Number of frames summed")
    if info["zmean"] is not None:
        hdr["ZMEAN"] = (round(info["zmean"], 3), "Mean zenith distance [deg], airmass weighted")
    else:
        _del(hdr, "ZMEAN")
    _del(hdr, "ZDIST")

    zs = pd.to_numeric(frames["ZSPACE"], errors="coerce").dropna().astype(int).unique() if "ZSPACE" in frames.columns else []
    if len(zs) == 1:
        hdr["ZSPACE"] = int(zs[0])
    elif len(zs) > 1:
        _del(hdr, "ZSPACE")

    if len(frames):
        for k in _START_KEYS:  # start of the observation = first frame of the product
            if k in frames.columns:
                v = frames[k].iloc[0]
                if isinstance(v, str) and v.strip() == "":
                    continue
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    hdr[k] = v

    if history and "FILE" in frames.columns and len(frames):
        _drop_history(hdr, "FRAMES:")
        txt = frame_ranges(frames["FILE"].tolist())
        for line in _hist_chunks(txt, "FRAMES: "):
            hdr["HISTORY"] = line
        hdr["HISTORY"] = f"FRAMES: {info['n']} frames, total exposure {info['texp']:.1f} s"
    if bad.any():
        hdr["HISTORY"] = f"WARNING: {int(bad.sum())} frame(s) without valid EXPTIME not counted"
    if out_path is not None:
        stamp(hdr, out_path)
    return info
