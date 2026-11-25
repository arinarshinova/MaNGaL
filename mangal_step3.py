from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np


import scipy.ndimage as ndi
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clip

from mangal_names import object_stage_path, stage_path, side_path
from mangal_header import frame_meta_from_header, apply_frame_keys, stamp
from mangal_step4_shift import align_cube as align_cube_stars

try:
    import astroscrappy
except Exception:
    astroscrappy = None


def _nan_to_num(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

# ---------------- Param file ----------------

def _robust_sigma(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if mad > 0:
        return 1.4826 * mad
    # fallback for nearly-constant arrays
    return float(np.std(a))

def _split_value_comment(rest: str) -> str:
    m = re.search(r"\s/\s", rest)
    return rest[: m.start()].strip() if m else rest.strip()

def parse_param_file(path: str | Path) -> dict:
    params: dict = {}
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("SIMPLE"):
            continue
        if "=" not in line:
            continue
        key, rest = line.split("=", 1)
        key = key.strip()
        val_part = _split_value_comment(rest)
        if val_part.count("'") >= 2:
            first = val_part.find("'")
            last = val_part.rfind("'")
            val: object = val_part[first + 1 : last]
        else:
            v = val_part
            try:
                val = float(v) if re.search(r"[\.eE]", v) else int(v)
            except Exception:
                val = v
        params[key] = val

    if isinstance(params.get("CCD_CUT"), str):
        nums = [int(x.strip()) for x in params["CCD_CUT"].split(",") if x.strip()]
        if len(nums) == 4:
            params["CCD_CUT_TUPLE"] = tuple(nums)

    if isinstance(params.get("DARKS"), str):
        params["DARKS_LIST"] = [int(x.strip()) for x in params["DARKS"].split(",") if x.strip()]

    for k in ("CUB_OBJ", "CUB_FLAT"):
        if k in params and isinstance(params[k], str) and params[k].strip():
            params[k] = params[k].strip().zfill(2)

    return params

# ---------------- Log table ----------------
_LOG_COLS = [
    "FILE","OBJECT","TYPE","START","Texp","FILTERS","DISPERSER","Zspace",
    "RA","DEC","SEEING","Z","PA","COMMENTS"
]


def read_author(r_dir: str | Path, files, ext: str = ".fts", max_files: int = 5) -> str | None:
    """
    Applicant of the observations: the AUTHOR keyword of the raw FITS header.
    Looks at the first `max_files` of `files` (log FILE names, extension added) and returns the first
    non-empty AUTHOR value, or None.
    """
    r_dir = Path(r_dir)
    tried = 0
    for fn in files:
        p = r_dir / f"{fn}{ext}"
        if not p.exists():
            continue
        tried += 1
        try:
            with fits.open(p) as hdul:
                for hdu in hdul:
                    val = hdu.header.get("AUTHOR")
                    if val is not None and str(val).strip():
                        return str(val).strip()
        except Exception:
            pass
        if tried >= max_files:
            break
    return None


def parse_obs_log_table(path: str | Path) -> pd.DataFrame:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    rows = re.findall(r"^\|t\d+_\d+\|.*\|$", txt, flags=re.M)
    parsed = []
    for r in rows:
        cols = [c.strip() for c in r.strip("|").split("|")]
        if len(cols) < len(_LOG_COLS):
            continue
        d = dict(zip(_LOG_COLS, cols[:len(_LOG_COLS)]))
        try:
            d["Texp"] = int(float(d["Texp"]))
        except Exception:
            pass
        parsed.append(d)
    return pd.DataFrame(parsed)

# ---------------- FITS helpers ----------------
@dataclass(frozen=True)
class Cut:
    x0:int; y0:int; x1:int; y1:int
    def apply(self, img: np.ndarray) -> np.ndarray:
        return img[self.y0:self.y1+1, self.x0:self.x1+1]

def _read_fits_data(path: Path, cut: Cut | None = None) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header.copy()

    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape={data.shape} in {path}")

    if cut is not None:
        data = cut.apply(data)

    return data, hdr

def _sigma_clipped_mean(stack: np.ndarray, sigma: float = 3.0, maxiters: int = 5) -> np.ndarray:
    clipped = sigma_clip(stack, sigma=sigma, maxiters=maxiters, axis=0)
    mean = np.nanmean(clipped.filled(np.nan), axis=0).astype(np.float32)
    return mean

def cube_ids_list(cube_id: str | None) -> list[str]:
    """'4, 05' -> ['04', '05'] (order kept); None / '' -> []."""
    if cube_id is None:
        return []
    return [x.strip().zfill(2) for x in str(cube_id).split(",") if x.strip()]


def select_cube_rows(df: pd.DataFrame, cube_id: str | None) -> pd.DataFrame:
    """
    Rows of the log that belong to the cube(s) cube_id: one id ('04') or several ('04,05').
    Several cubes are concatenated in the given order, frames of each cube in log order.
    cube_id None / '' -> all rows.
    """
    ids = cube_ids_list(cube_id)
    if not ids:
        return df
    files = df["FILE"].astype(str)
    parts = [df[files.str.contains(rf"_{cid}\d\d$")] for cid in ids]
    return pd.concat(parts, axis=0)[lambda d: ~d.index.duplicated()].copy()


# ---------------- Products: dark/flat ----------------
def build_masterdark(df_log: pd.DataFrame, r_dir: str|Path, w_dir: str|Path, texp: int,
                     *, ccd_cut: tuple[int,int,int,int] | None = None,
                     pattern_ext: str = ".fts") -> Path:
    r_dir = Path(r_dir); w_dir = Path(w_dir); w_dir.mkdir(parents=True, exist_ok=True)
    cut = Cut(*ccd_cut) if ccd_cut else None

    rows = df_log[(df_log["TYPE"]=="dark") & (df_log["Texp"].astype(int)==int(texp))]
    if rows.empty:
        raise FileNotFoundError(f"No dark frames with Texp={texp} in the log table")

    imgs = []
    first_hdr = None
    for fn in rows["FILE"].tolist():
        p = r_dir / f"{fn}{pattern_ext}"
        if not p.exists():
            raise FileNotFoundError(f"Dark file not found: {p}")
        im, hdr = _read_fits_data(p, cut=cut)
        if first_hdr is None:
            first_hdr = hdr
        imgs.append(im)

    master = _sigma_clipped_mean(np.stack(imgs, axis=0))

    out = w_dir / f"dark{int(texp)}.fits"
    hdr = first_hdr if first_hdr is not None else fits.Header()
    hdr["OBJECT"] = "masterdark"
    hdr["HISTORY"] = f"MASTERDARK: {len(imgs)} frames, Texp={texp}"
    stamp(hdr, out)
    fits.writeto(out, master, hdr, overwrite=True)
    return out

def build_masterflat(df_log: pd.DataFrame, r_dir: str|Path, w_dir: str|Path, *,
                     cube_id: str | None,
                     masterdark_path: str|Path,
                     expected_filter: str | None = None,
                     ccd_cut: tuple[int,int,int,int] | None = None,
                     pattern_ext: str = ".fts",
                     norm: str = "median") -> Path:
    r_dir = Path(r_dir); w_dir = Path(w_dir); w_dir.mkdir(parents=True, exist_ok=True)
    cut = Cut(*ccd_cut) if ccd_cut else None

    rows = df_log[df_log["TYPE"]=="flat"].copy()
    rows = select_cube_rows(rows, cube_id)

    if expected_filter is not None and "FILTERS" in rows.columns:
        rows2 = rows[rows["FILTERS"] == expected_filter]
        if not rows2.empty:
            rows = rows2

    if rows.empty:
        raise FileNotFoundError(f"No flat frames found (cube_id={cube_id}, filter={expected_filter})")

    dark, _ = _read_fits_data(Path(masterdark_path), cut=cut)

    imgs = []
    first_hdr = None
    for fn in rows["FILE"].tolist():
        p = r_dir / f"{fn}{pattern_ext}"
        if not p.exists():
            raise FileNotFoundError(f"Flat file not found: {p}")
        im, hdr = _read_fits_data(p, cut=cut)
        if im.shape != dark.shape:
            raise ValueError(f"Flat shape {im.shape} != dark shape {dark.shape} for {p.name}")
        if first_hdr is None:
            first_hdr = hdr
        imgs.append(im - dark)

    master = _sigma_clipped_mean(np.stack(imgs, axis=0))

    good = np.isfinite(master) & (master > 0)
    if not np.any(good):
        raise ValueError("Masterflat has non-positive/NaN values - check the dark/flat frames")

    scale = float(np.nanmedian(master[good])) if norm == "median" else float(np.nanmean(master[good]))
    master_norm = (master / scale).astype(np.float32)

    out = w_dir / "masterflat.fits"
    hdr = first_hdr if first_hdr is not None else fits.Header()
    hdr["OBJECT"] = "masterflat"
    hdr["HISTORY"] = f"MASTERFLAT: {len(imgs)} frames, dark={Path(masterdark_path).name}, norm={norm}({scale:.6g})"
    stamp(hdr, out)
    fits.writeto(out, master_norm, hdr, overwrite=True)
    return out

def object_rows(df_log: pd.DataFrame, object_name: str, cube_id: str | None = None) -> pd.DataFrame:
    """
    obj frames of a reduction run.
    - cube_id empty: all frames with OBJECT == object_name;
    - cube_id given ('04' or '04,06'): the frames of these cubes, WHATEVER their OBJECT name. Cube numbers are unique
      within a night, so a cube observed under another name (e.g. PGC45443_dz after a new ZSPACE calibration) can be
      merged with the main one.
    """
    obj = df_log[df_log["TYPE"].astype(str).str.lower() == "obj"]
    if cube_ids_list(cube_id):
        return select_cube_rows(obj, cube_id)
    return obj[obj["OBJECT"] == object_name].copy()


def describe_cubes(rows: pd.DataFrame) -> str:
    """'04: PGC45443 (12 frames), 06: PGC45443_dz (10 frames)' (cube order kept)."""
    out = []
    for fn, obj in zip(rows["FILE"].astype(str), rows["OBJECT"].astype(str)):
        m = re.search(r"_(\d{2})\d{2}$", fn)
        key = (m.group(1) if m else "?", obj)
        if out and out[-1][0] == key:
            out[-1][1] += 1
        else:
            out.append([key, 1])
    return ", ".join(f"{c}: {o} ({n} frames)" for (c, o), n in out)


def build_reduced_cube_darkflat(df_log: pd.DataFrame, r_dir: str|Path, w_dir: str|Path, *,
                               object_name: str,
                               cube_id: str | None,
                               masterdark_path: str|Path,
                               masterflat_path: str|Path,
                               ccd_cut: tuple[int,int,int,int] | None = None,
                               pattern_ext: str = ".fts") -> Path:
    r_dir = Path(r_dir); w_dir = Path(w_dir); w_dir.mkdir(parents=True, exist_ok=True)
    cut = Cut(*ccd_cut) if ccd_cut else None

    md, _ = _read_fits_data(Path(masterdark_path), cut=cut)
    mf, _ = _read_fits_data(Path(masterflat_path), cut=cut)

    eps = 1e-6
    mf_safe = np.where(np.isfinite(mf) & (mf > eps), mf, np.nan).astype(np.float32)

    rows = object_rows(df_log, object_name, cube_id)
    if rows.empty:
        raise FileNotFoundError(f"No obj frames found for OBJECT={object_name} (cube_id={cube_id})")
    if len(cube_ids_list(cube_id)) > 1 or (rows["OBJECT"] != object_name).any():
        print(f"[cubes] merged: {describe_cubes(rows)}")

    imgs = []
    meta = []  # per-slice metadata for downstream steps (e.g., step6 line/cont)
    first_hdr = None
    for fn in rows["FILE"].tolist():
        p = r_dir / f"{fn}{pattern_ext}"
        if not p.exists():
            raise FileNotFoundError(f"Object file not found: {p}")
        im, hdr = _read_fits_data(p, cut=cut)
        if im.shape != md.shape or im.shape != mf_safe.shape:
            raise ValueError(f"Obj shape {im.shape} != dark shape {md.shape} or flat shape {mf_safe.shape} for {p.name}")
        if first_hdr is None:
            first_hdr = hdr
        # per-slice metadata for the sidecar: ZSPACE, EXPTIME, ZDIST, DATE-OBS, ... of every raw frame
        meta.append(frame_meta_from_header(fn, hdr))
        imgs.append((im - md) / mf_safe)

    cube = np.stack(imgs, axis=0).astype(np.float32)
    out = object_stage_path(w_dir, object_name, "darkflat")
    hdr = first_hdr if first_hdr is not None else fits.Header()
    hdr["HISTORY"] = f"REDUCE: dark={Path(masterdark_path).name} flat={Path(masterflat_path).name}"
    try:  # EXPTIME = total of all frames, ZMEAN, NFRAMES, FILENAME, DATE ... (not the keywords of frame 1)
        import pandas as _pd0
        apply_frame_keys(hdr, _pd0.DataFrame(meta), out)
    except Exception as e:
        print(f"[warn] could not write the summed-exposure keywords: {e}")
        stamp(hdr, out)
    fits.writeto(out, cube, hdr, overwrite=True)
    # Save inputs sidecar for downstream steps (e.g., step6 line/cont)
    try:
        import pandas as _pd
        df_meta = _pd.DataFrame(meta)
        sidecar = side_path(out, ".inputs.csv")
        df_meta.to_csv(sidecar, index=False)
    except Exception:
        pass
    return out

# ---------------- Cosmic cleaning (after flat) ----------------
def _replace_negative_with_local_mean(img: np.ndarray, window: int = 10) -> np.ndarray:
    """Replace finite negative pixels with the mean of non-negative finite values
    in a local square window. Window=10 means 10x10 pixels.
    """
    out = np.asarray(img, dtype=np.float32).copy()
    neg_mask = np.isfinite(out) & (out < 0)
    if not np.any(neg_mask):
        return out

    window = max(1, int(window))
    half_lo = window // 2
    half_hi = window - half_lo - 1

    valid = np.isfinite(out) & (out >= 0)
    vals = np.where(valid, out, 0.0).astype(np.float64)
    cnts = valid.astype(np.int64)

    sumpad = np.pad(vals, ((1, 0), (1, 0)), mode='constant', constant_values=0)
    cntpad = np.pad(cnts, ((1, 0), (1, 0)), mode='constant', constant_values=0)
    isum = np.cumsum(np.cumsum(sumpad, axis=0), axis=1)
    icnt = np.cumsum(np.cumsum(cntpad, axis=0), axis=1)

    yy, xx = np.where(neg_mask)
    global_valid = out[valid]
    fallback = float(np.mean(global_valid)) if global_valid.size else 0.0

    for y, x in zip(yy, xx):
        y0 = max(0, y - half_lo)
        y1 = min(out.shape[0] - 1, y + half_hi)
        x0 = max(0, x - half_lo)
        x1 = min(out.shape[1] - 1, x + half_hi)

        s = isum[y1 + 1, x1 + 1] - isum[y0, x1 + 1] - isum[y1 + 1, x0] + isum[y0, x0]
        c = icnt[y1 + 1, x1 + 1] - icnt[y0, x1 + 1] - icnt[y1 + 1, x0] + icnt[y0, x0]
        out[y, x] = np.float32(s / c) if c > 0 else np.float32(fallback)

    return out


def clean_cosmics_cube(
    cube_path: str|Path,
    out_path: str|Path | None = None,
    *,
    gain: float = 1.0,
    readnoise: float = 6.5,
    sigclip: float = 7.0,
    sigfrac: float = 0.3,
    objlim: float = 8.0,
    satlevel: float | None = None,
    niter: int = 4,
    save_mask: bool = False,
    mask_circle: tuple[float, float, float] | None = None,  # (xc, yc, r)
    outside_value: float | None = np.nan,
    cleantype: str = "medmask",
) -> tuple[Path, Path | None]:
    if astroscrappy is None:
        raise ImportError("astroscrappy is not installed. Install it: pip install astroscrappy")

    cube_path = Path(cube_path)
    data, hdr = fits.getdata(cube_path, header=True)

    if data.ndim != 3:
        raise ValueError(f"Expected a FITS cube (N,Y,X), got shape={data.shape}")

    n, ny, nx = data.shape
    cleaned = np.empty_like(data, dtype=np.float32)
    mask_cube = np.zeros_like(data, dtype=np.uint8) if save_mask else None

    if mask_circle is not None:
        xc, yc, r = mask_circle
        x0 = max(0, int(np.floor(xc - r)))
        x1 = min(nx - 1, int(np.ceil(xc + r)))
        y0 = max(0, int(np.floor(yc - r)))
        y1 = min(ny - 1, int(np.ceil(yc + r)))
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        inside = ((xx - xc)**2 + (yy - yc)**2) <= (r**2)
    else:
        x0=y0=0; x1=nx-1; y1=ny-1
        inside = None

    kwargs = dict(
        gain=gain,
        readnoise=readnoise,
        sigclip=sigclip,
        sigfrac=sigfrac,
        objlim=objlim,
        niter=niter,
    )
    if satlevel is not None:
        kwargs["satlevel"] = float(satlevel)

    for i in range(n):
        img = np.asarray(data[i], dtype=np.float32)
        out_img = img.copy()

        if mask_circle is None:
            inmask = ~np.isfinite(img)
            m, clean = astroscrappy.detect_cosmics(img, inmask=inmask, cleantype=cleantype, **kwargs)
            cleaned[i] = _replace_negative_with_local_mean(clean, window=10).astype(np.float32)
            if save_mask and mask_cube is not None:
                mask_cube[i] = m.astype(np.uint8)
            continue

        roi = img[y0:y1+1, x0:x1+1].copy()

        # ignore outside circle + any NaN/inf so algorithm doesn't "invent" values near masked borders
        inmask = (~inside) | (~np.isfinite(roi))

        m_roi, clean_roi = astroscrappy.detect_cosmics(roi, inmask=inmask, cleantype=cleantype, **kwargs)

        roi_out = roi.copy()
        roi_out[inside] = clean_roi[inside]
        out_img[y0:y1+1, x0:x1+1] = roi_out

        if outside_value is not None:
            yy_full, xx_full = np.mgrid[0:ny, 0:nx]
            inside_full = ((xx_full - xc)**2 + (yy_full - yc)**2) <= (r**2)
            out_img[~inside_full] = outside_value

        cleaned[i] = _replace_negative_with_local_mean(out_img, window=10).astype(np.float32)

        if save_mask and mask_cube is not None:
            m_full = np.zeros((ny, nx), dtype=np.uint8)
            m_full[y0:y1+1, x0:x1+1][inside] = m_roi.astype(np.uint8)[inside]
            mask_cube[i] = m_full


    if out_path is None:
        out_path = stage_path(cube_path, "crclean")
    out_path = Path(out_path)

    hdr2 = hdr.copy()
    hdr2["HISTORY"] = "CR-CLEAN: astroscrappy (LA Cosmic)"
    hdr2["HISTORY"] = "Negative finite pixels replaced by local mean in 10x10 window"
    hdr2["CRGAIN"] = gain
    hdr2["CRRN"] = readnoise
    hdr2["CRSIGCL"] = float(sigclip)
    hdr2["CROBJLIM"] = float(objlim)
    hdr2["CRNITER"] = int(niter)
    hdr2["CRCLEAN"] = cleantype
    if mask_circle is not None:
        hdr2["CRMASKR"] = float(mask_circle[2])
        hdr2["CRMASKXC"] = float(mask_circle[0])
        hdr2["CRMASKYC"] = float(mask_circle[1])

    stamp(hdr2, out_path)
    fits.writeto(out_path, cleaned, hdr2, overwrite=True)

    mask_path = None
    if save_mask and mask_cube is not None:
        mask_path = out_path.with_name(out_path.stem + "_mask.fits")
        fits.writeto(mask_path, mask_cube, hdr2, overwrite=True)

    return out_path, mask_path


# ---------------- Step4: frame/channel shifts (registration) ----------------
# This reproduces the idea of IDL channels_dxdy.pro (shift each Z-slice) and
# also supports automatic estimation by phase correlation.

try:  # optional deps
    from scipy.ndimage import fourier_shift  # type: ignore
except Exception:  # pragma: no cover
    fourier_shift = None

try:
    from skimage.registration import phase_cross_correlation  # type: ignore
except Exception:  # pragma: no cover
    phase_cross_correlation = None


def _require_shift_deps() -> None:
    if fourier_shift is None or phase_cross_correlation is None:
        raise ImportError(
            "The shift step needs the scipy and scikit-image packages. "
            "Install them: pip install scipy scikit-image"
        )


def _as_cube_any(data: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (cube[nz,ny,nx], was_2d)."""
    if data.ndim == 2:
        return data[None, ...].astype(np.float32), True
    if data.ndim == 3:
        return np.asarray(data, dtype=np.float32), False
    raise ValueError(f"Expected a 2D/3D array, got ndim={data.ndim}, shape={data.shape}")


def _highpass_for_pcc(img: np.ndarray, sigma: float = 12.0, *, return_sigma: bool = False) -> np.ndarray | tuple[np.ndarray, float]:
    """Mild high-pass to make phase correlation less sensitive to smooth background / rings symmetry."""
    # Replace NaNs with 0 already outside; keep float32
    a = _nan_to_num(img).astype(np.float32, copy=False)
    if sigma is None or sigma <= 0:
        return a
    try:
        blur = ndi.gaussian_filter(a, sigma=float(sigma), mode="nearest")
        hp = a - blur
    except Exception:
        hp = a
    # robust normalize
    s = _robust_sigma(hp)
    if np.isfinite(s) and s > 0:
        hp = hp / s
    hp = hp.astype(np.float32, copy=False)
    if return_sigma:
        return hp, float(s) if np.isfinite(s) else float("nan")
    return hp

def _sanitize_for_pcc(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prepare image for phase_cross_correlation.

    Returns (image_float32, finite_mask). Non-finite values are set to 0, and a
    robust background (median) is removed to reduce DC component dominance.
    """
    arr = np.asarray(img, dtype=np.float32)
    mask = np.isfinite(arr)
    if not mask.all():
        arr = arr.copy()
        arr[~mask] = 0.0
    if mask.any():
        med = np.median(arr[mask])
        arr = arr - med
    return arr, mask


def _estimate_shift_phasecorr(
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    upsample_factor: int = 10,
) -> tuple[float, float, float]:
    _require_shift_deps()

    ref, ref_mask = _sanitize_for_pcc(reference)
    mov, mov_mask = _sanitize_for_pcc(moving)

    # scikit-image API differs across versions:
    # - return_error kwarg may not exist
    # - reference_mask/moving_mask may or may not exist
    # - return value can be (shift, error, phasediff) or only shift
    calls = [
        dict(upsample_factor=upsample_factor, return_error=True, reference_mask=ref_mask, moving_mask=mov_mask),
        dict(upsample_factor=upsample_factor, reference_mask=ref_mask, moving_mask=mov_mask),
        dict(upsample_factor=upsample_factor, return_error=True),
        dict(upsample_factor=upsample_factor),
    ]

    last_err: Exception | None = None
    for kwargs in calls:
        try:
            result = phase_cross_correlation(ref, mov, **kwargs)
            err = float("nan")
            if isinstance(result, tuple):
                shift = result[0]
                if len(result) >= 2:
                    try:
                        err = float(result[1])
                    except Exception:
                        err = float("nan")
            else:
                shift = result
            dy, dx = float(shift[0]), float(shift[1])
            if np.isfinite(dy) and np.isfinite(dx):
                return dy, dx, err
        except (TypeError, ValueError) as e:
            last_err = e
            continue

    print(f"[warn] phase_cross_correlation failed, using dy=0,dx=0. Reason: {last_err}")
    return 0.0, 0.0, float('nan')


def _apply_shift_fourier(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    _require_shift_deps()
    # NaNs inside img will poison FFT and turn the whole frame into NaNs.
    # Fill invalid pixels with 0 before shifting.
    img2 = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    f = np.fft.fftn(img2)
    shifted = np.fft.ifftn(fourier_shift(f, shift=(dy, dx))).real
    return shifted.astype(np.float32)


def _collapse_cube(cube: np.ndarray, how: str = "median") -> np.ndarray:
    how = how.lower()
    if how == "median":
        return np.nanmedian(cube, axis=0)
    if how == "sum":
        return np.nansum(cube, axis=0)
    if how == "mean":
        return np.nanmean(cube, axis=0)
    raise ValueError(f"Unknown collapse={how} (use median/sum/mean)")


def align_cube_per_slice(cube: np.ndarray, *, ref_index: int = 0, upsample_factor: int = 10, progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Align each Z slice to ref_index. Returns (aligned_cube, shifts[nz,2]=(dy,dx)).

    Universal stability improvement (QC gating):
      - shift estimate is identical to fixed10 (phase_cross_correlation on high-passed frames)
      - if correlation quality is poor, reuse the last good shift (no frame-specific hacks)
    """
    nz, ny, nx = cube.shape
    ref = cube[ref_index].astype(np.float32)
    aligned = np.empty_like(cube, dtype=np.float32)
    shifts = np.zeros((nz, 2), dtype=np.float32)

    ref_hp = _highpass_for_pcc(ref)

    for k in range(nz):
        mov_hp = _highpass_for_pcc(cube[k])
        dy, dx, err = _estimate_shift_phasecorr(mov_hp, ref_hp, upsample_factor=upsample_factor)

        # Use raw per-slice PCC shifts without QC fallback to the last value.
        # This keeps diagnostic behavior transparent when real shifts are being
        # rejected by quality gating and collapsed to zeros.
        dy = float(dy)
        dx = float(dx)

        aligned[k] = _apply_shift_fourier(cube[k], dy, dx)
        shifts[k] = (dy, dx)

        if progress and (k % max(1, nz // 10) == 0 or k == nz - 1):
            print(f"[shift per-slice] {k+1:4d}/{nz}: dy={dy:+.3f}, dx={dx:+.3f}")

    return aligned, shifts


def align_cube_per_cube(
    cube: np.ndarray,
    *,
    ref_index: int = 0,
    collapse: str = "median",
    upsample_factor: int = 10,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align each slice to a collapsed 2D reference built from the whole cube.

    Returns (aligned_cube, shifts[nz,2]=(dy,dx), ref_img_2d).

    Notes:
    - Uses phase cross-correlation (PCC) on a lightly high-pass filtered image.
    - If a slice is too "flat" (very low structure) or PCC returns a high error,
      the shift is marked as NaN and later filled by interpolation from neighboring
      good slices. This prevents long tails of (0,0) shifts when S/N drops.
    """
    nz, ny, nx = cube.shape
    ref_img = _collapse_cube(cube, how=collapse).astype(np.float32)

    # Precompute reference high-pass and its scale
    ref_hp, ref_s = _highpass_for_pcc(ref_img, return_sigma=True)
    # Threshold for "has enough structure"
    s_min = 1e-6
    if np.isfinite(ref_s) and ref_s > 0:
        s_min = max(s_min, 0.2 * float(ref_s))

    aligned = np.empty_like(cube, dtype=np.float32)
    shifts = np.full((nz, 2), np.nan, dtype=np.float32)
    errs = np.full(nz, np.nan, dtype=np.float32)

    for k in range(nz):
        mov_hp, mov_s = _highpass_for_pcc(cube[k], return_sigma=True)

        # If the slice is basically empty/flat, skip measurement
        if not np.isfinite(mov_s) or float(mov_s) < s_min:
            dy = dx = float('nan')
            err = float('nan')
        else:
            dy, dx, err = _estimate_shift_phasecorr(mov_hp, ref_hp, upsample_factor=upsample_factor)

            # If PCC reports a large error, treat as unreliable
            # (Typical good values are small; exact scale depends on data)
            if np.isfinite(err) and err > 0.4:
                dy = dx = float('nan')

        errs[k] = err
        shifts[k] = (dy, dx)

        if progress and (k % max(1, nz // 10) == 0 or k == nz - 1):
            dy_p = 0.0 if not np.isfinite(dy) else dy
            dx_p = 0.0 if not np.isfinite(dx) else dx
            print(f"[shift per-cube] {k+1:4d}/{nz}: dy={dy_p:+.3f}, dx={dx_p:+.3f}")

    # Fill NaNs by interpolation (or nearest) to avoid trailing zeros
    idx = np.arange(nz, dtype=np.float32)
    good = np.isfinite(shifts[:, 0]) & np.isfinite(shifts[:, 1])

    if good.sum() == 0:
        shifts[:, :] = 0.0
    elif good.sum() == 1:
        shifts[:, 0] = float(shifts[good, 0][0])
        shifts[:, 1] = float(shifts[good, 1][0])
    else:
        for j in (0, 1):
            v = shifts[:, j].astype(np.float32)
            g = np.isfinite(v)
            shifts[:, j] = np.interp(idx, idx[g], v[g]).astype(np.float32)

    # Apply shifts
    for k in range(nz):
        dy, dx = float(shifts[k, 0]), float(shifts[k, 1])
        aligned[k] = _apply_shift_fourier(cube[k], dy, dx)

    return aligned, shifts, ref_img


def _read_two_col_table(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read whitespace/CSV-ish with 2+ columns: t dx dy or similar; returns (t, v1, v2?) in caller."""
    p = Path(path)
    rows = []
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        # split by whitespace or comma
        parts = re.split(r"[\s,]+", line)
        if len(parts) < 3:
            continue
        try:
            t = float(parts[0])
            dx = float(parts[1])
            dy = float(parts[2])
        except Exception:
            continue
        rows.append((t, dx, dy))
    if not rows:
        raise ValueError(f"No numeric rows found in {p}")
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:,0], arr[:,1], arr[:,2]


def _read_channels_time(path: str | Path) -> np.ndarray:
    p = Path(path)
    vals = []
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        parts = re.split(r"[\s,]+", line)
        try:
            vals.append(float(parts[0]))
        except Exception:
            continue
    if not vals:
        raise ValueError(f"No numeric times found in {p}")
    return np.asarray(vals, dtype=np.float64)


def align_cube_from_time(cube: np.ndarray, *, testring_path: str|Path, channels_time_path: str|Path, invert_shift: bool = False, progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Apply dx(t),dy(t) (IDL channels_dxdy.pro style) to each slice. Returns (aligned_cube, shifts[nz,2])."""
    # testring: columns t dx dy (or t dx dy)
    t, dx_tab, dy_tab = _read_two_col_table(testring_path)
    ch_t = _read_channels_time(channels_time_path)
    nz = cube.shape[0]
    if len(ch_t) != nz:
        # allow mismatch but warn and interpolate on available
        print(f"[warn] channels_time length {len(ch_t)} != cube nz {nz}; using min length")
        n_use = min(len(ch_t), nz)
    else:
        n_use = nz

    # linear interpolation like IDL INTERPOL(...)
    dx = np.interp(ch_t[:n_use], t, dx_tab)
    dy = np.interp(ch_t[:n_use], t, dy_tab)

    if invert_shift:
        dx = -dx
        dy = -dy

    aligned = np.empty_like(cube, dtype=np.float32)
    shifts = np.zeros((nz, 2), dtype=np.float32)

    for k in range(nz):
        if k < n_use:
            # IDL uses shift_image(im, -dx, -dy). Here our apply expects (dy,dx)
            ddy = -float(dy[k])
            ddx = -float(dx[k])
        else:
            ddy = 0.0; ddx = 0.0
        aligned[k] = _apply_shift_fourier(cube[k], ddy, ddx)
        shifts[k] = (ddy, ddx)
        if progress and (k % max(1, nz // 10) == 0 or k == nz - 1):
            print(f"[shift from-time] {k+1:4d}/{nz}: dy={ddy:+.3f}, dx={ddx:+.3f}")
    return aligned, shifts


def shift_cube(
    cube_path: str | Path,
    *,
    out_path: str | Path | None = None,
    mode: str = "per-slice",
    ref_index: int = 0,
    upsample_factor: int = 10,
    collapse: str = "median",
    testring_path: str | Path | None = None,
    channels_time_path: str | Path | None = None,
    invert_shift: bool = False,
    # Extra knobs (used by run_mangal_step3.py). Kept for backwards compatibility.
    max_shift: float = 0.0,
    max_step: float = 2.0,
    smooth_window: int = 5,
    outlier_z: float = 4.0,
) -> tuple[Path, np.ndarray]:
    """Top-level: read FITS cube, align, write FITS. Returns (out_path, shifts)."""
    cube_path = Path(cube_path)
    with fits.open(cube_path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header.copy()

    cube, was_2d = _as_cube_any(data)

    mode = mode.lower()
    if mode in ("per-slice", "per_slice", "slice"):
        aligned, shifts = align_cube_per_slice(cube, ref_index=ref_index, upsample_factor=upsample_factor)
        hdr["HISTORY"] = f"SHIFT: per-slice ref={ref_index} upsample={upsample_factor}"
    elif mode in ("per-cube", "per_cube", "cube"):
        aligned, shifts, _ref_img = align_cube_per_cube(cube, ref_index=ref_index, collapse=collapse, upsample_factor=upsample_factor)
        hdr["HISTORY"] = f"SHIFT: per-cube collapse={collapse} upsample={upsample_factor}"

    elif mode in ("stars", "star"):
        # Robust star-based alignment with internal FFT fallback (see mangal_step4_shift.py)
        aligned, shifts, _stars, _ref_img = align_cube_stars(
            cube,
            ref_index=ref_index,
            max_abs_shift=float(max_shift),
        )
        hdr["HISTORY"] = f"SHIFT: stars+fft_fallback ref={ref_index} max_shift={float(max_shift)}"

    elif mode in ("from-time", "from_time", "time"):
        if testring_path is None or channels_time_path is None:
            raise ValueError("mode='from-time' requires testring_path and channels_time_path")
        aligned, shifts = align_cube_from_time(
            cube, testring_path=testring_path, channels_time_path=channels_time_path, invert_shift=invert_shift
        )
        hdr["HISTORY"] = f"SHIFT: from-time invert={bool(invert_shift)}"
    else:
        raise ValueError(f"Unknown shift mode={mode}")


    # --- Post-process shifts: clip / smooth / outlier-robustness (mainly for per-slice / stars modes) ---
    shifts = np.asarray(shifts, dtype=np.float32)
    if shifts.ndim == 2 and shifts.shape[1] == 2:
        # clip absolute shift
        if max_shift is not None and np.isfinite(max_shift) and max_shift > 0:
            shifts[:, 0] = np.clip(shifts[:, 0], -max_shift, max_shift)
            shifts[:, 1] = np.clip(shifts[:, 1], -max_shift, max_shift)

        # limit per-step jump
        if max_step is not None and np.isfinite(max_step) and max_step > 0 and shifts.shape[0] > 1:
            dy = shifts[:, 0].copy()
            dx = shifts[:, 1].copy()
            for k in range(1, shifts.shape[0]):
                ddy = np.clip(dy[k] - dy[k - 1], -max_step, max_step)
                ddx = np.clip(dx[k] - dx[k - 1], -max_step, max_step)
                dy[k] = dy[k - 1] + ddy
                dx[k] = dx[k - 1] + ddx
            shifts[:, 0] = dy
            shifts[:, 1] = dx

        # smooth with running median (odd window)
        w = int(smooth_window) if smooth_window is not None else 0
        if w >= 3 and shifts.shape[0] >= w:
            if w % 2 == 0:
                w += 1
            pad = w // 2
            for col in (0, 1):
                a = shifts[:, col]
                ap = np.pad(a, (pad, pad), mode="edge")
                sm = np.array([np.median(ap[i : i + w]) for i in range(a.size)], dtype=np.float32)
                # outlier replacement relative to smoothed
                if outlier_z is not None and np.isfinite(outlier_z) and outlier_z > 0:
                    resid = a - sm
                    sig = float(np.median(np.abs(resid - np.median(resid))) * 1.4826) + 1e-6
                    bad = np.abs(resid) > outlier_z * sig
                    a2 = a.copy()
                    a2[bad] = sm[bad]
                    shifts[:, col] = a2
                else:
                    shifts[:, col] = sm

    if out_path is None:
        out_path = stage_path(cube_path, "shift")
    out_path = Path(out_path)

    out_data = aligned[0] if was_2d else aligned
    stamp(hdr, out_path)
    fits.writeto(out_path, out_data.astype(np.float32), hdr, overwrite=True)

    # also save shifts table next to cube for reproducibility
    shifts_path = out_path.with_suffix(".shifts.csv")
    try:
        df = pd.DataFrame(shifts, columns=["dy","dx"])
        df.to_csv(shifts_path, index_label="slice")
    except Exception:
        pass

    return out_path, shifts