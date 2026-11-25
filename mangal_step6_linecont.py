from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

try:
    from astropy.io import fits
except Exception as e:
    raise ImportError("astropy is required for step6 (line/cont cube building)") from e

from mangal_names import line_tag, write_layer_table, side_path, find_side
from mangal_header import apply_frame_keys, frame_summary


_Z_RE = re.compile(r"(?i)\bz\s*space\s*[:=]?\s*(-?\d+)\b(.*)$")
_ZSIMPLE_RE = re.compile(r"(?i)\bzspace\s*[:=]?\s*(-?\d+)\b(.*)$")


def parse_zspace_log(p: str | Path) -> dict[int, str]:
    """
    Parse mapping from ZSPACE integer to a label string.
    Accepts lines like:
      Zspace=-1807 cont
      Zspace=-1486 [OIII]
      ZSPACE = -1486 / ...
    Returns {zspace_int: label_str} for every ZSPACE that has a real label somewhere in the log
    (empty / "NA" repeats of the same ZSPACE are ignored).
    """
    p = Path(p)
    txt = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: dict[int, str] = {}
    seen = False
    for line in txt:
        s = line.strip()
        if not s:
            continue
        m = _Z_RE.search(s) or _ZSIMPLE_RE.search(s)
        if not m:
            continue
        seen = True
        z = int(m.group(1))
        tail = (m.group(2) or "").strip()
        # sanitize tail: remove leading punctuation
        tail = re.sub(r"^[\s:=/-]+", "", tail)
        # The same ZSPACE is logged again for every frame, often without a label (or as "NA").
        # Such repeats must not erase a label that was given before: the last REAL label wins.
        if tail and tail.upper() != "NA":
            out[z] = tail
    if not seen:
        raise ValueError(f"No Zspace/ZSPACE lines found in log: {p}")
    return out


def classify_label(label: str) -> str:
    """Return 'cont' or 'line'."""
    if label is None:
        return "unknown"
    s = str(label).lower()
    return "cont" if "cont" in s else "line"


def _read_cube_and_hdr(cube_path: str | Path):
    p = Path(cube_path)
    with fits.open(p) as hdul:
        data = hdul[0].data.astype(np.float32)
        hdr = hdul[0].header.copy()
    if data.ndim != 3:
        raise ValueError(f"Expected 3D cube (nslice, ny, nx), got {data.shape} in {p}")
    return data, hdr


def _slice_meta_from_source_cube(source_cube_path: str | Path) -> pd.DataFrame:
    """
    Return per-slice metadata (at least ZSPACE, ideally EXPTIME) from the sidecar CSV:
      <cube>.inputs.csv created by step3 and propagated through intermediate products.

    Expected columns (case-insensitive):
      - ZSPACE
      - EXPTIME (or EXPOSURE)
      - FILE (optional)

    Raises FileNotFoundError if sidecar is missing.
    """
    source_cube_path = Path(source_cube_path)
    sidecar = find_side(source_cube_path, ".inputs.csv")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"Cannot find per-slice metadata sidecar: {sidecar}. "
            f"Step6 needs per-slice ZSPACE and EXPTIME; make sure step3 wrote it and it was propagated."
        )
    df = pd.read_csv(sidecar)
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    if "zspace" not in cols:
        raise ValueError(f"Sidecar {sidecar} has no ZSPACE column. Columns={list(df.columns)}")
    zcol = cols["zspace"]
    # EXPTIME optional but strongly preferred
    ecol = cols.get("exptime") or cols.get("exposure")
    if ecol is None:
        df["EXPTIME"] = np.nan
        ecol = "EXPTIME"
    # Ensure numeric
    df["ZSPACE"] = pd.to_numeric(df[zcol], errors="coerce").astype("Int64")
    df["EXPTIME"] = pd.to_numeric(df[ecol], errors="coerce")
    # ZSPACE / EXPTIME plus whatever else the sidecar has (FILE, ZDIST, DATE-OBS, ...)
    return df.drop(columns=[c for c in (zcol, ecol) if c not in ("ZSPACE", "EXPTIME")]).copy()


def frame_id(file_name) -> str | None:
    """'t260417_1005' / 't260417_1005.fts' -> '1005' (cube + frame number of the observation log)."""
    m = re.search(r"_(\d{4})(?:\.f\w*)?$", str(file_name).strip())
    return m.group(1) if m else None


def parse_frame_list(spec) -> tuple[set[str], list[str]]:
    """
    Frames given by hand: '1005, 1210-1212, t260417_0815' -> ({'1005', '1210', '1211', '1212', '0815'}, []).
    Numbers are CCNN of the observation log (cube CC, frame NN). Returns (ids, tokens that were not understood).
    """
    ids, bad = set(), []
    if spec is None:
        return ids, bad
    toks = spec if isinstance(spec, (list, tuple, set)) else re.split(r"[,\s;]+", str(spec))
    for t in toks:
        t = str(t).strip()
        if not t:
            continue
        m = re.fullmatch(r"(\d{3,4})-(\d{3,4})", t)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b < a or b - a > 500:
                bad.append(t)
                continue
            ids.update(f"{n:04d}" for n in range(a, b + 1))
            continue
        if re.fullmatch(r"\d{3,4}", t):
            ids.add(f"{int(t):04d}")
            continue
        fid = frame_id(t)
        if fid:
            ids.add(fid)
        else:
            bad.append(t)
    return ids, bad


def _write_linecont_product(out_path: Path, hdr, layer_imgs: list, layers: list[dict],
                            *, tline: float, tcont: float, source_name: str,
                            labels, kinds, zspaces, good_mask, used, slice_groups=None, frames=None,
                            excluded=None, excluded_files=None):
    """Write the multi-layer cube (one layer per entry of `layers`) and the per-slice report."""
    out_cube = np.stack(layer_imgs, axis=0).astype(np.float32)
    out_hdr = hdr.copy()
    if excluded_files:
        out_hdr["HISTORY"] = "STEP6: frames excluded by hand: " + ", ".join(excluded_files)
    out_hdr["HISTORY"] = f"STEP6: line/cont layers from {source_name}"
    out_hdr["HISTORY"] = "STEP6: normalized by total EXPTIME per layer (counts/sec)"
    out_hdr["HISTORY"] = "STEP6: layers: " + ", ".join(str(L["name"]) for L in layers)
    out_hdr["UNITS"] = ("counts/sec", "After EXPTIME normalization")
    out_hdr["TLINE"] = (float(tline), "[s] Sum of EXPTIME for used line slices")
    out_hdr["TCONT"] = (float(tcont), "[s] Sum of EXPTIME for used cont slices")
    if frames is not None:  # EXPTIME / NFRAMES / ZMEAN of ALL frames used in the cube; FILENAME, DATE of this file
        apply_frame_keys(out_hdr, frames[np.asarray(used, dtype=bool)], out_path)
    else:
        from mangal_header import stamp
        stamp(out_hdr, out_path)
    write_layer_table(out_hdr, layers)  # NLAYERS, LAYERn, LKINDn, LROLEn, LTAGn, LZSPn, LTEXPn
    fits.writeto(out_path, out_cube, out_hdr, overwrite=True)

    rep = {
        "slice": np.arange(len(zspaces), dtype=int),
        "FILE": (frames["FILE"].astype(str).tolist() if frames is not None and "FILE" in frames.columns
                 else [""] * len(zspaces)),
        "ZSPACE": zspaces,
        "label": labels,
        "kind": kinds,
        "good_phot": np.asarray(good_mask, dtype=int),
        "used": np.asarray(used, dtype=int),
        "excluded_by_hand": np.asarray(excluded if excluded is not None else np.zeros(len(zspaces)), dtype=int),
    }
    if slice_groups is not None:
        rep["group"] = slice_groups
    rep_path = side_path(out_path, ".csv")
    pd.DataFrame(rep).to_csv(rep_path, index=False)


def _group_slices(labels, kinds, zspaces, used, zmap, group_by: str):
    """
    Return (line_groups, cont_groups): lists of dicts {kind, zspace, zspaces, label, tag, idx}.
      * LINES are grouped by emission line (Ha, NII, 5007, ...): ALL slices of one line are summed
        together, whatever their ZSPACE (no splitting by dz inside a line).
      * CONTINUA are grouped by one ZSPACE (group_by="zspace") or one log label (group_by="label").
    Groups are ordered as they appear in the ZSPACE log.
    """
    if group_by not in ("zspace", "label"):
        raise ValueError(f"group_by must be 'zspace' or 'label', got {group_by!r}")
    log_order = {int(z): n for n, z in enumerate(zmap.keys())}
    groups: dict = {}
    for i, (lab, kind, z) in enumerate(zip(labels, kinds, zspaces)):
        if not used[i] or kind not in ("line", "cont"):
            continue
        if kind == "line":
            tag = line_tag(lab) or "line"
            key = (kind, tag.lower())
        else:
            tag = ""
            key = (kind, int(z)) if group_by == "zspace" else (kind, str(lab).strip().lower())
        g = groups.setdefault(key, {"kind": kind, "zspace": int(z), "zspaces": [], "label": lab,
                                    "tag": tag, "idx": []})
        if int(z) not in g["zspaces"]:
            g["zspaces"].append(int(z))
        g["idx"].append(i)

    def order(g):
        return min(log_order.get(int(zspaces[i]), 10**9) for i in g["idx"]), g["idx"][0]

    lines = sorted((g for g in groups.values() if g["kind"] == "line"), key=order)
    conts = sorted((g for g in groups.values() if g["kind"] == "cont"), key=order)
    return lines, conts


def build_linecont_cube(
    cube_path: str | Path,
    source_cube_path: str | Path,
    zspace_log_path: str | Path,
    out_path: str | Path,
    *,
    phot_table_path: str | Path | None = None,
    split_groups: bool = True,
    group_by: str = "zspace",
    exclude_frames=None,
) -> Path:
    """
    Build the multi-layer line/continuum cube (counts/sec):
      one layer per emission line (Ha, NII, 5007, ...)  - all slices of a line together, any ZSPACE
      one layer per continuum (cont, or cont1, cont2, ...) - one ZSPACE (group_by="zspace") or one
                                                             log label (group_by="label") each
    With split_groups=False there are just two layers, "line" (all lines) and "cont" (all continuum).
    There are NO separate line_sum / cont_sum layers: the combined continuum that is subtracted is the
    exposure-weighted mean of the continuum layers (computed in step 7).
    The layer table is stored in the header (NLAYERS, LAYERn, LKINDn, LROLEn, LTAGn, LZSPn, LTEXPn) and
    is what step7 uses to pick the layers. Uses ZSPACE per slice from the sidecar of source_cube_path
    and the mapping from zspace_log_path; if phot_table_path is given, only slices with good==1.
    exclude_frames: frames left out by hand (e.g. satellite tracks the photometry did not catch), as CCNN numbers of
    the observation log: "1005, 1210-1212" or file names.
    """
    cube, hdr = _read_cube_and_hdr(cube_path)
    meta_full = _slice_meta_from_source_cube(source_cube_path)
    zspaces_full = meta_full['ZSPACE'].astype(int).tolist()
    exptimes_full = meta_full['EXPTIME'].to_numpy(dtype=float)

    # Determine which slices are "good" from step5.
    good_mask_full = None
    if phot_table_path is not None and Path(phot_table_path).exists():
        dfp = pd.read_csv(phot_table_path)
        col = "good" if "good" in dfp.columns else ("GOOD" if "GOOD" in dfp.columns else None)
        if col:
            good_mask_full = dfp[col].astype(int).values.astype(bool)

    # If cube_path is already filtered (phot cube), but the ZSPACE sidecar corresponds to the
    # pre-filtered cube, then lengths won't match. In that case, filter ZSPACE list using
    # the phot good-mask to match the filtered cube.
    zspaces = zspaces_full
    exptimes = exptimes_full.copy()
    frames = meta_full.reset_index(drop=True)  # per-slice table aligned to the cube slices
    good_mask = np.ones(cube.shape[0], dtype=bool)
    if len(zspaces_full) != cube.shape[0]:
        if good_mask_full is None:
            raise ValueError(
                f"ZSPACE list length {len(zspaces_full)} != nslice {cube.shape[0]} and no phot table provided."
            )
        if len(good_mask_full) != len(zspaces_full):
            raise ValueError(
                f"Phot good-mask length {len(good_mask_full)} != ZSPACE list length {len(zspaces_full)}"
            )
        zspaces = [int(z) for z, g in zip(zspaces_full, good_mask_full) if bool(g)]
        exptimes = np.array([t for t, g in zip(exptimes_full, good_mask_full) if bool(g)], dtype=float)
        if len(zspaces) != cube.shape[0]:
            raise ValueError(
                f"After filtering by phot mask, ZSPACE list length {len(zspaces)} != nslice {cube.shape[0]}"
            )
        frames = meta_full[np.asarray(good_mask_full, dtype=bool)].reset_index(drop=True)
        # cube is already filtered, so per-slice good flag is 1 for all its slices
        good_mask = np.ones(cube.shape[0], dtype=bool)
    else:
        # cube not filtered (or sidecar already matched), so good_mask can be used directly
        if good_mask_full is not None:
            good_mask = good_mask_full

    zmap = parse_zspace_log(zspace_log_path)

    # frames excluded by hand
    excl_ids, excl_bad = parse_frame_list(exclude_frames)
    slice_ids = [frame_id(f) for f in frames["FILE"]] if "FILE" in frames.columns else [None] * len(zspaces)
    excluded = [sid is not None and sid in excl_ids for sid in slice_ids]
    excluded_files = [str(frames["FILE"].iloc[i]) for i in range(len(excluded)) if excluded[i]]
    if excl_ids or excl_bad:
        all_ids = {frame_id(f) for f in meta_full["FILE"]} if "FILE" in meta_full.columns else set()
        in_cube = {s for s in slice_ids if s}
        rejected = sorted(i for i in excl_ids if i in all_ids and i not in in_cube)
        unknown = sorted(i for i in excl_ids if i not in all_ids)
        print(f"[linecont] excluded by hand: {len(excluded_files)} frame(s)"
              + (f" ({', '.join(excluded_files)})" if excluded_files else ""), flush=True)
        if rejected:
            print(f"[linecont] already rejected by the photometry: {', '.join(rejected)}", flush=True)
        if unknown or excl_bad:
            print(f"[linecont] WARNING: not frames of this cube: {', '.join(unknown + excl_bad)} "
                  f"(numbers are CCNN of the observation log, e.g. 1005 or 1210-1212)", flush=True)

    labels = []
    kinds = []
    used = []
    for i, z in enumerate(zspaces):
        label = zmap.get(int(z), "")
        kind = classify_label(label) if label != "" else "unknown"
        labels.append(label)
        kinds.append(kind)
        used.append(bool(good_mask[i]) and kind in ("line", "cont") and not excluded[i])
    nolabel = sorted({int(z) for z, lab in zip(zspaces, labels) if lab == ""})
    if nolabel:
        n = sum(1 for lab in labels if lab == "")
        print(f"[linecont] WARNING: {n} frame(s) with ZSPACE {', '.join(map(str, nolabel))} have no label in the ZSPACE log "
              f"{Path(zspace_log_path).name} and are NOT used (merged cubes: are the ZSPACE values of every cube in the log?)",
              flush=True)

    def _mean_img(idx):
        """EXPTIME-weighted mean (counts/sec) of the slices `idx`, and their total exposure."""
        acc = np.zeros(cube.shape[1:], dtype=np.float64)
        tt = 0.0
        for i in idx:
            t = exptimes[i]
            if not np.isfinite(t) or t <= 0:
                t = 1.0
            acc += cube[i].astype(np.float64)
            tt += float(t)
        img = (acc / tt).astype(np.float32) if tt > 0 else np.zeros(cube.shape[1:], dtype=np.float32)
        return img, tt

    def _lay(name, kind, role, tag, idx, texp):
        """Layer description with exposure, number of frames and mean zenith distance of its frames."""
        fs = frame_summary(frames.iloc[list(idx)]) if len(idx) else {"n": 0, "zmean": None}
        return {"name": name, "kind": kind, "role": role, "tag": tag, "zspace": _zs(idx), "texp": texp,
                "nframes": fs["n"], "zmean": fs["zmean"]}

    def _zs(idx):
        return ";".join(str(z) for z in dict.fromkeys(int(zspaces[i]) for i in idx))

    line_idx = [i for i in range(cube.shape[0]) if used[i] and kinds[i] == "line"]
    cont_idx = [i for i in range(cube.shape[0]) if used[i] and kinds[i] == "cont"]
    line_img, tline = _mean_img(line_idx)
    cont_img, tcont = _mean_img(cont_idx)

    line_groups, cont_groups = _group_slices(labels, kinds, zspaces, used, zmap, group_by)

    slice_groups = [""] * cube.shape[0]
    for g in line_groups:
        for i in g["idx"]:
            slice_groups[i] = g["tag"]
    for c, g in enumerate(cont_groups, start=1):
        for i in g["idx"]:
            slice_groups[i] = f"cont{c}"

    layer_imgs: list = []
    layers: list[dict] = []
    if split_groups:
        # one layer per emission line (all its ZSPACE values together)
        for g in line_groups:
            img, tt = _mean_img(g["idx"])
            layer_imgs.append(img)
            layers.append(_lay(g["tag"], "line", "group", g["tag"], g["idx"], tt))
        # one layer per continuum; a lone continuum is simply "cont"
        for c, g in enumerate(cont_groups, start=1):
            img, tt = _mean_img(g["idx"])
            layer_imgs.append(img)
            layers.append(_lay("cont" if len(cont_groups) == 1 else f"cont{c}", "cont", "group", "", g["idx"], tt))
    else:
        # --no-lc-split: everything of a kind merged into one layer
        layer_imgs = [line_img, cont_img]
        layers = [
            _lay("line", "line", "total", "", line_idx, tline),
            _lay("cont", "cont", "total", "", cont_idx, tcont),
        ]
    if not layers:
        raise ValueError("No usable line or continuum slices (check the ZSPACE log labels and the phot mask)")

    out_path = Path(out_path)
    _write_linecont_product(
        out_path, hdr, layer_imgs, layers,
        tline=tline, tcont=tcont, source_name=Path(cube_path).name,
        labels=labels, kinds=kinds, zspaces=zspaces, good_mask=good_mask, used=used,
        slice_groups=slice_groups, frames=frames, excluded=excluded, excluded_files=excluded_files,
    )
    return out_path
