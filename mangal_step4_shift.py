#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from mangal_header import stamp
from mangal_names import side_path, find_side, stage_path
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, maximum_filter, shift as ndi_shift, binary_closing, binary_dilation, binary_erosion, binary_fill_holes
from scipy.optimize import curve_fit
import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import sys

# backends that cannot open a window (inline = Jupyter / VS Code / Spyder, set via MPLBACKEND)
_NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def _is_interactive_backend(name: str) -> bool:
    n = str(name).strip().lower()
    return not (n in _NON_INTERACTIVE_BACKENDS or n.startswith("module://"))


def _ensure_interactive_backend() -> bool:
    """
    The star-selection window needs a window-capable matplotlib backend. If the current one cannot open
    windows (e.g. MPLBACKEND=Agg or the inline backend inherited from an IDE), switch to macosx / QtAgg / TkAgg.
    Returns False if no such backend works (the window is then not shown at all).
    """
    cur = matplotlib.get_backend()
    if _is_interactive_backend(cur):
        return True
    cands = (["macosx"] if sys.platform == "darwin" else []) + ["QtAgg", "TkAgg"]
    for b in cands:
        try:
            plt.switch_backend(b)
        except Exception:
            continue
        print(f"[shift stars] matplotlib backend {cur!r} cannot open windows -> switched to {b}", flush=True)
        return True
    print(f"[shift stars] ERROR: matplotlib backend {cur!r} cannot open windows and none of "
          f"{', '.join(cands)} works, so the star-selection window cannot be shown. "
          f"Unset MPLBACKEND (or set MPLBACKEND=QtAgg) and run again.", flush=True)
    return False


class NoReferenceStarsError(RuntimeError):
    """No reference star is left for the shift stage: the stage must not run."""


@dataclass
class RefStar:
    x_ref: float
    y_ref: float
    peak: float


@dataclass
class StarMeasure:
    ok: bool
    x_prev: float
    y_prev: float
    x_peak: float
    y_peak: float
    x_fit: float
    y_fit: float
    peak: float
    bg: float
    score: float


@dataclass
class ShiftDebug:
    method: str
    ngood: int
    ntotal: int
    dy_raw: float
    dx_raw: float
    dy_apply: float
    dx_apply: float


def _nan(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _robust_sigma(a: np.ndarray) -> float:
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return float(1.4826 * mad)


def _fon_corners_like_idl(image: np.ndarray, win: int = 30, mode: str = "plane", addvalue: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    im = np.asarray(image, dtype=np.float32)
    if im.ndim != 2:
        raise ValueError("image must be 2D")
    ny, nx = im.shape
    if nx < 2 * win + 3 or ny < 2 * win + 3:
        bg = np.zeros_like(im, dtype=np.float32)
        return im.copy(), bg, np.full(4, np.nan, dtype=np.float32)

    xp = np.array([win, nx - win - 1, win, nx - win - 1], dtype=int)
    yp = np.array([win, win, ny - win - 1, ny - win - 1], dtype=int)

    vals = []
    for xc, yc in zip(xp, yp):
        box = im[yc - win: yc + win + 1, xc - win: xc + win + 1]
        vals.append(float(np.nanmedian(box)))
    a = np.asarray(vals, dtype=np.float32)

    if mode == "mean":
        bg = np.full_like(im, float(np.nanmean(a)), dtype=np.float32)
    elif mode == "min":
        bg = np.full_like(im, float(np.nanmin(a)), dtype=np.float32)
    else:
        x = np.arange(nx, dtype=np.float32) / float(nx)
        y = np.arange(ny, dtype=np.float32) / float(ny)
        xx, yy = np.meshgrid(x, y)
        xc = xp.astype(np.float32) / float(nx)
        yc = yp.astype(np.float32) / float(ny)
        A = np.array([
            [1.0, xc[0], yc[0], xc[0] * yc[0]],
            [1.0, xc[1], yc[1], xc[1] * yc[1]],
            [1.0, xc[2], yc[2], xc[2] * yc[2]],
            [1.0, xc[3], yc[3], xc[3] * yc[3]],
        ], dtype=np.float32)
        try:
            c0, c1, c2, c3 = np.linalg.solve(A, a.astype(np.float32))
            bg = (c0 + c1 * xx + c2 * yy + c3 * xx * yy).astype(np.float32)
        except np.linalg.LinAlgError:
            bg = np.full_like(im, float(np.nanmean(a)), dtype=np.float32)
    out = (im - bg + float(addvalue)).astype(np.float32)
    return out, bg, a


def _field_mask_from_image(image: np.ndarray, win: int = 30) -> np.ndarray:
    im = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(im)
    pos = im[finite & (im > 0)]
    if pos.size == 0:
        return finite
    sigma = max(2.0, float(win) / 3.0)
    sm = gaussian_filter(np.where(finite, np.maximum(im, 0.0), 0.0), sigma=sigma)
    ref = pos
    p05 = float(np.percentile(ref, 5))
    p50 = float(np.percentile(ref, 50))
    thr = max(1e-6, min(p50 * 0.35, p05 * 1.5 if p05 > 0 else p50 * 0.1))
    mask = finite & (sm > thr)
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    return np.asarray(mask, dtype=bool)


def _source_mask_from_image(image: np.ndarray, field_mask: np.ndarray, win: int = 30) -> np.ndarray:
    im = np.asarray(image, dtype=np.float32)
    vals = im[field_mask & np.isfinite(im)]
    if vals.size == 0:
        return np.zeros_like(field_mask, dtype=bool)
    med = float(np.median(vals))
    sig = max(_robust_sigma(vals), 1e-6)
    sm = gaussian_filter(np.where(np.isfinite(im), im, med), sigma=1.2)
    peak = maximum_filter(sm, size=max(3, int(win // 2) | 1))
    bright = field_mask & (sm == peak) & (sm > med + 5.0 * sig)
    src = binary_dilation(bright, iterations=max(3, int(win // 6)))
    return np.asarray(src, dtype=bool)


def _background_mask(image: np.ndarray, win: int = 30, exclude_center: tuple[float, float, float] | None = None,
                     field: np.ndarray | None = None) -> np.ndarray:
    if field is None:
        field = _field_mask_from_image(image, win=win)
    safe = binary_erosion(field, iterations=max(4, int(win // 3)))
    src = _source_mask_from_image(image, safe, win=win)
    bgmask = safe & (~src)
    if exclude_center is not None:
        cx, cy, cr = exclude_center
        yy, xx = np.indices(bgmask.shape, dtype=np.float32)
        bgmask &= ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 > float(cr) ** 2)
    return np.asarray(bgmask, dtype=bool)


def _masked_smooth_background(image: np.ndarray, bgmask: np.ndarray, win: int = 30,
                              field: np.ndarray | None = None) -> np.ndarray:
    im = np.asarray(image, dtype=np.float32)
    valid = bgmask & np.isfinite(im)
    if not np.any(valid):
        return np.full_like(im, float(np.nanmedian(im)), dtype=np.float32)
    med = float(np.nanmedian(im[valid]))
    work = np.where(valid, im, 0.0).astype(np.float32)
    w = valid.astype(np.float32)
    sigma = max(3.0, float(win))
    num = gaussian_filter(work, sigma=sigma, mode='nearest')
    den = gaussian_filter(w, sigma=sigma, mode='nearest')
    bg = np.divide(num, np.maximum(den, 1e-6), out=np.full_like(im, med, dtype=np.float32), where=den > 1e-6)
    if field is None:
        field = _field_mask_from_image(im, win=win)
    if np.any(field):
        vals = bg[field]
        if vals.size:
            fill = float(np.nanmedian(vals))
            bg = np.where(field, bg, fill).astype(np.float32)
    return bg.astype(np.float32)


def _backgr_sub_masked(image: np.ndarray, win: int = 30, exclude_center: tuple[float, float, float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    im = np.asarray(image, dtype=np.float32)
    field = _field_mask_from_image(im, win=win)  # computed once, used by both steps
    bgmask = _background_mask(im, win=win, exclude_center=exclude_center, field=field)
    bg = _masked_smooth_background(im, bgmask, win=win, field=field)
    out = (im - bg).astype(np.float32)
    return out, bg, bgmask.astype(np.uint8)


def _apply_background_mode(image: np.ndarray, mode: str = "none", win: int = 30, exclude_center: tuple[float, float, float] | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    mode = (mode or "none").strip().lower()
    if mode in ("none", "0", "off"):
        return np.asarray(image, dtype=np.float32), None, None
    if mode in ("fon_corners", "corners_plane", "plane"):
        out, bg, _ = _fon_corners_like_idl(image, win=win, mode="plane")
        return out, bg, None
    if mode in ("corners_mean", "mean"):
        out, bg, _ = _fon_corners_like_idl(image, win=win, mode="mean")
        return out, bg, None
    if mode in ("corners_min", "min"):
        out, bg, _ = _fon_corners_like_idl(image, win=win, mode="min")
        return out, bg, None
    if mode in ("backgr_sub", "backgr_sub_masked", "image_background", "background"):
        return _backgr_sub_masked(image, win=win, exclude_center=exclude_center)
    raise ValueError(f"Unsupported background mode: {mode}")

def subtract_background_fits(
    cube_fits: str | Path,
    mode: str = "backgr_sub",
    window: int = 30,
    exclude_center: tuple[float, float, float] | None = None,
) -> bool:
    """
    Step 1: subtract the background from every frame of cube_fits IN PLACE (the cube of step 1, darkflat or
    crclean, is written anew each time step 1 runs). The header gets BGMODE / BGWIN, so later stages know it was
    done. Returns True if the background was subtracted now.
    """
    mode = str(mode or "none").strip().lower()
    if mode in ("none", "0", "off"):
        return False
    cube_fits = Path(cube_fits)
    with fits.open(cube_fits) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header.copy()
    if str(hdr.get("BGMODE", "none")).strip().lower() not in ("none", "0", "off", ""):
        print(f"[bg] {cube_fits.name}: background already subtracted (BGMODE={hdr['BGMODE']}), skipped", flush=True)
        return False
    cube = data[None, :, :] if data.ndim == 2 else data
    nz = cube.shape[0]
    out = np.empty_like(cube)
    print(f"[bg] background subtraction ({mode}, window {int(window)}) in {nz} frames of {cube_fits.name} ...", flush=True)
    for k in range(nz):
        out[k], _, _ = _apply_background_mode(cube[k], mode=mode, win=int(window), exclude_center=exclude_center)
        if k + 1 == nz or (k + 1) % 5 == 0:
            print(f"[bg] {k + 1}/{nz}", flush=True)
    if data.ndim == 2:
        out = out[0]
    hdr["BGMODE"] = (mode, "background subtracted at step 1")
    hdr["BGWIN"] = (int(window), "background window [px]")
    hdr["HISTORY"] = f"STEP1: background subtracted ({mode}, window {int(window)})"
    stamp(hdr, cube_fits)
    fits.PrimaryHDU(out.astype(np.float32), header=hdr).writeto(cube_fits, overwrite=True)
    return True


def background_already_subtracted(cube_fits: str | Path) -> str | None:
    """BGMODE of the cube if step 1 subtracted the background, else None."""
    try:
        v = str(fits.getheader(Path(cube_fits)).get("BGMODE", "none")).strip()
    except Exception:
        return None
    return None if v.lower() in ("none", "0", "off", "") else v


def _extract_stamp(img: np.ndarray, x: float, y: float, half: int) -> tuple[Optional[np.ndarray], int, int]:
    ny, nx = img.shape
    xi = int(round(x))
    yi = int(round(y))
    x0 = xi - half
    x1 = xi + half + 1
    y0 = yi - half
    y1 = yi + half + 1
    if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
        return None, x0, y0
    return _nan(img[y0:y1, x0:x1]).copy(), x0, y0


def _gauss2d(coords, amp, x0, y0, sx, sy, bg):
    xx, yy = coords
    return (amp * np.exp(-0.5 * (((xx - x0) / sx) ** 2 + ((yy - y0) / sy) ** 2)) + bg).ravel()


def _fit_local_star(stamp: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    a = _nan(stamp).astype(np.float32, copy=False)
    if a.shape[0] < 5 or a.shape[1] < 5:
        return None
    sm = gaussian_filter(a, 0.7)
    iy, ix = np.unravel_index(int(np.argmax(sm)), sm.shape)
    flat = np.sort(a.ravel())
    nbg = max(5, int(0.2 * flat.size))
    bg0 = float(np.median(flat[:nbg]))
    amp0 = float(a[iy, ix] - bg0)
    if not np.isfinite(amp0) or amp0 <= 0:
        return None
    yy, xx = np.indices(a.shape, dtype=np.float32)
    try:
        popt, _ = curve_fit(
            _gauss2d,
            (xx, yy),
            a.ravel(),
            p0=[amp0, float(ix), float(iy), 1.8, 1.8, bg0],
            bounds=(
                [0.0, max(0.0, ix - 2.5), max(0.0, iy - 2.5), 0.5, 0.5, float(np.min(a))],
                [float(np.max(a) * 3.0 + 1.0), min(a.shape[1] - 1.0, ix + 2.5), min(a.shape[0] - 1.0, iy + 2.5), 8.0, 8.0, float(np.max(a))],
            ),
            maxfev=5000,
        )
        _, x0, y0, sx, sy, bg = [float(v) for v in popt]
        if not (0 <= x0 <= a.shape[1] - 1 and 0 <= y0 <= a.shape[0] - 1):
            return None
        if not (0.5 <= sx <= 8.0 and 0.5 <= sy <= 8.0):
            return None
        peak = float(_gauss2d((np.array([[x0]]), np.array([[y0]])), *popt)[0] - bg)
        return x0, y0, peak, bg
    except Exception:
        w = a - bg0
        w[w < 0] = 0.0
        s = float(w.sum())
        if s <= 0:
            return None
        yy, xx = np.indices(w.shape, dtype=np.float32)
        x0 = float((xx * w).sum() / s)
        y0 = float((yy * w).sum() / s)
        peak = float(np.max(a) - bg0)
        return x0, y0, peak, bg0


def _refine_second_pass(img: np.ndarray, x_fit: float, y_fit: float, fit_half: int) -> tuple[float, float, float, float]:
    sub2, sx2, sy2 = _extract_stamp(img, x_fit, y_fit, max(3, fit_half - 1))
    if sub2 is None:
        return x_fit, y_fit, np.nan, np.nan
    fit2 = _fit_local_star(sub2)
    if fit2 is None:
        return x_fit, y_fit, np.nan, np.nan
    xf2, yf2, peak2, bg2 = fit2
    return sx2 + xf2, sy2 + yf2, peak2, bg2


def find_reference_stars(
    img: np.ndarray,
    detect_snr: float = 20.0,
    search_half: int = 30,
    max_stars: int = 80,
    min_dist: Optional[int] = None,
    exclude_center: Optional[Tuple[float, float, float]] = None,
) -> tuple[List[RefStar], float]:
    """IDL-like: detect stars only on the first frame."""
    a = _nan(img).astype(np.float32, copy=False)
    sm_small = gaussian_filter(a, 1.2)
    sm_large = gaussian_filter(a, 4.0)
    det = sm_small - sm_large
    sig = _robust_sigma(det)
    med = float(np.median(det))
    thr = med + float(detect_snr) * sig
    if min_dist is None:
        min_dist = max(8, int(0.5 * search_half))

    mx = maximum_filter(det, size=int(min_dist), mode="nearest")
    ys, xs = np.where((det == mx) & (det > thr))
    cand = []
    ny, nx = a.shape
    edge = max(2 * int(search_half), 24)

    for y, x in zip(ys.tolist(), xs.tolist()):
        if x < edge or x >= nx - edge or y < edge or y >= ny - edge:
            continue
        if exclude_center is not None:
            xc, yc, rc = exclude_center
            if (x - xc) ** 2 + (y - yc) ** 2 <= rc ** 2:
                continue
        cand.append((float(det[y, x]), float(x), float(y)))

    cand.sort(reverse=True)
    stars: List[RefStar] = []
    for _, x, y in cand:
        too_close = False
        for s in stars:
            if (x - s.x_ref) ** 2 + (y - s.y_ref) ** 2 < float(min_dist * min_dist):
                too_close = True
                break
        if too_close:
            continue
        stamp, x0, y0 = _extract_stamp(a, x, y, max(4, search_half // 4))
        if stamp is None:
            continue
        fit = _fit_local_star(stamp)
        if fit is None:
            continue
        xf, yf, peak, _ = fit
        xabs = x0 + xf
        yabs = y0 + yf
        if np.hypot(xabs - x, yabs - y) > max(3.0, 0.20 * search_half):
            continue
        if peak <= 0:
            continue
        stars.append(RefStar(x_ref=xabs, y_ref=yabs, peak=float(peak)))
        if len(stars) >= int(max_stars):
            break
    return stars, thr


def measure_star_in_window(img: np.ndarray, x_prev: float, y_prev: float, window: int) -> StarMeasure:
    stamp, x0, y0 = _extract_stamp(img, x_prev, y_prev, int(window))
    if stamp is None:
        return StarMeasure(False, x_prev, y_prev, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    a = _nan(stamp).astype(np.float32, copy=False)
    sm = gaussian_filter(a, 0.7)
    iy, ix = np.unravel_index(int(np.argmax(sm)), sm.shape)
    fit_half = max(4, min(8, int(window) // 3))
    x_peak_abs = x0 + ix
    y_peak_abs = y0 + iy
    sub, sx0, sy0 = _extract_stamp(img, x_peak_abs, y_peak_abs, fit_half)
    if sub is None:
        return StarMeasure(False, x_prev, y_prev, x_peak_abs, y_peak_abs, np.nan, np.nan, np.nan, np.nan, np.nan)
    fit = _fit_local_star(sub)
    if fit is None:
        return StarMeasure(False, x_prev, y_prev, x_peak_abs, y_peak_abs, np.nan, np.nan, np.nan, np.nan, np.nan)
    xf, yf, peak, bg = fit
    x_fit = sx0 + xf
    y_fit = sy0 + yf
    x_fit2, y_fit2, peak2, bg2 = _refine_second_pass(img, x_fit, y_fit, fit_half)
    if np.isfinite(x_fit2) and np.isfinite(y_fit2) and np.hypot(x_fit2 - x_fit, y_fit2 - y_fit) <= 1.5:
        x_fit, y_fit = x_fit2, y_fit2
        if np.isfinite(peak2):
            peak = peak2
        if np.isfinite(bg2):
            bg = bg2
    score = float(np.hypot(x_fit - x_prev, y_fit - y_prev))
    local_sig = _robust_sigma(sub)
    if score > max(3.0, 0.55 * float(window)):
        return StarMeasure(False, x_prev, y_prev, x_peak_abs, y_peak_abs, x_fit, y_fit, peak, bg, score)
    if np.isfinite(local_sig) and local_sig > 0 and peak < 3.0 * local_sig:
        return StarMeasure(False, x_prev, y_prev, x_peak_abs, y_peak_abs, x_fit, y_fit, peak, bg, score)
    return StarMeasure(True, x_prev, y_prev, x_peak_abs, y_peak_abs, x_fit, y_fit, peak, bg, score)


def _clean_raw_shifts(dy: np.ndarray, dx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.isfinite(dy) & np.isfinite(dx)
    if keep.sum() < 3:
        return dy, dx, keep
    med_dy = np.median(dy[keep])
    med_dx = np.median(dx[keep])
    sig_dy = _robust_sigma(dy[keep])
    sig_dx = _robust_sigma(dx[keep])
    keep2 = keep.copy()
    if sig_dy > 0:
        keep2 &= np.abs(dy - med_dy) <= 3.0 * sig_dy
    if sig_dx > 0:
        keep2 &= np.abs(dx - med_dx) <= 3.0 * sig_dx
    return dy, dx, keep2


def save_stars_png(path: Path, img: np.ndarray, stars: List[RefStar], threshold: float, search_half: int, exclude_center=None, title: str = "") -> None:
    with plt.rc_context({'text.usetex': False}):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
        im = _nan(img)
        vmin, vmax = np.percentile(im, [5, 99.5])
        ax.imshow(im, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        for i, s in enumerate(stars, start=1):
            ax.add_patch(plt.Rectangle((s.x_ref - search_half, s.y_ref - search_half), 2 * search_half + 1, 2 * search_half + 1, fill=False, ec="deepskyblue", lw=0.9, alpha=0.7))
            ax.add_patch(plt.Circle((s.x_ref, s.y_ref), radius=max(6, int(0.35 * search_half)), fill=False, ec="yellowgreen", lw=1.1))
            ax.plot(s.x_ref, s.y_ref, marker="+", ms=7, mew=1.2, color="crimson")
            ax.text(s.x_ref + 4, s.y_ref + 4, f"S{i}", color="deepskyblue", fontsize=6.5)
        if exclude_center is not None:
            xc, yc, rc = exclude_center
            ax.add_patch(plt.Circle((xc, yc), rc, fill=False, color="crimson", lw=2))
        ax.set_title(f"{title}\nthreshold={threshold:.3f}, window={search_half}, nstars={len(stars)}")
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def refine_manual_star(img: np.ndarray, x: float, y: float, search_half: int, min_snr: float = 0.0) -> Optional[RefStar]:
    """
    Snap a hand-picked position (x, y) to the nearest star: take the brightest smoothed pixel within
    ~search_half/4 px, then refine it with the same Gaussian fit as for automatically found stars.
    If the fit fails, the local maximum itself is used. Returns None only if the point is too close
    to the frame edge, there is no signal at all, or (min_snr > 0) the peak is lower than
    min_snr * local noise (used for mouse clicks so that a click on empty sky adds nothing).
    """
    a = _nan(img).astype(np.float32, copy=False)
    r = max(4, int(search_half) // 4)
    stamp, x0, y0 = _extract_stamp(a, x, y, r)
    if stamp is None:
        return None
    sm = gaussian_filter(stamp, 1.0)
    iy, ix = np.unravel_index(int(np.argmax(sm)), sm.shape)
    xs, ys = float(x0 + ix), float(y0 + iy)
    if min_snr > 0:
        wide, _wx, _wy = _extract_stamp(a, xs, ys, max(r, 15))
        sig = _robust_sigma(wide if wide is not None else stamp)
        if not (np.isfinite(sig) and sig > 0 and float(sm[iy, ix] - np.nanmedian(stamp)) >= float(min_snr) * sig):
            return None
    sub, sx0, sy0 = _extract_stamp(a, xs, ys, r)
    sig = _robust_sigma(stamp)
    fit = _fit_local_star(sub) if sub is not None else None
    if fit is not None:
        xf, yf, peak, _ = fit
        xa, ya = sx0 + xf, sy0 + yf
        if np.hypot(xa - xs, ya - ys) <= max(3.0, 0.5 * r) and peak > 0:
            if min_snr > 0 and not (np.isfinite(sig) and sig > 0 and peak >= min_snr * sig):
                return None
            return RefStar(x_ref=float(xa), y_ref=float(ya), peak=float(peak))
    peak = float(a[int(ys), int(xs)] - np.nanmedian(stamp))
    if not np.isfinite(peak):
        return None
    if min_snr > 0 and not (np.isfinite(sig) and sig > 0 and peak >= min_snr * sig):
        return None
    return RefStar(x_ref=xs, y_ref=ys, peak=peak)


def _raise_window(fig) -> None:
    """Bring a matplotlib window to the front (it may open behind the GUI window)."""
    try:
        mgr = fig.canvas.manager
        try:
            mgr.set_window_title("shift stars")
        except Exception:
            pass
        mgr.show()
        w = getattr(mgr, "window", None)
        if w is not None:
            for name in ("raise_", "activateWindow", "lift"):
                if hasattr(w, name):
                    getattr(w, name)()
            if hasattr(w, "attributes"):  # Tk: topmost for a moment
                w.attributes("-topmost", True)
                w.after(500, lambda: w.attributes("-topmost", False))
    except Exception:
        pass


def interactive_prune_stars(img: np.ndarray, stars: List[RefStar], search_half: int, exclude_center=None) -> List[RefStar]:
    """
    Interactive choice of the shift reference stars, directly on the picture:
      * click on a star (green / orange circle) ... exclude it (turns red) / include it again
      * click on an empty place ................... ADD a star by hand (snapped to the nearest peak;
                                                     nothing is added if there is no star there)
      * right click on a hand-added star (M) ...... delete it;   U ... delete the star added last
      * Enter / Esc ............................... finish
    Zoom / pan of the matplotlib toolbar do not add or remove anything while active.
    Works also when no star was found automatically. Returns the selected stars (possibly an empty
    list - the caller then refuses to run the shift stage).
    """
    orig = list(stars)
    stars = list(stars)
    selected: List[bool] = [True] * len(stars)
    manual: List[int] = []  # indices (into `stars`) of hand-added stars, in order of adding
    state = {"msg": ""}
    if not _ensure_interactive_backend():
        print(f"[shift stars] using the {len(orig)} automatically found stars without the window", flush=True)
        return orig
    try:
        plt.ion()
        with plt.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'}):
            fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=120)
            im = _nan(img)
            vmin, vmax = np.percentile(im, [5, 99.5])
            min_d2 = float(max(8, int(0.5 * search_half)) ** 2)

            def redraw():
                xlim, ylim = (ax.get_xlim(), ax.get_ylim()) if ax.images else (None, None)  # keep the zoom
                ax.clear()
                ax.imshow(im, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
                if xlim is not None:
                    ax.set_xlim(xlim); ax.set_ylim(ylim)
                for i, s in enumerate(stars, start=1):
                    active = bool(selected[i - 1])
                    is_man = (i - 1) in manual
                    circle_color = ("orange" if is_man else "yellowgreen") if active else "crimson"
                    label_color = ("orange" if is_man else "deepskyblue") if active else "crimson"
                    ax.add_patch(plt.Circle((s.x_ref, s.y_ref), radius=max(6, int(0.35 * search_half)), fill=False, ec=circle_color, lw=1.2))
                    ax.add_patch(plt.Rectangle((s.x_ref - search_half, s.y_ref - search_half), 2 * search_half + 1, 2 * search_half + 1, fill=False, ec="deepskyblue", lw=0.9, alpha=0.65))
                    ax.text(s.x_ref + 3, s.y_ref + 3, f"{'M' if is_man else 'S'}{i}", color=label_color, fontsize=5.8)
                if exclude_center is not None:
                    xc, yc, rc = exclude_center
                    ax.add_patch(plt.Circle((xc, yc), rc, fill=False, color="crimson", lw=1.6))
                ax.set_title(
                    "click on a star: exclude/include | click on empty place: ADD star | right click on M or U: delete added | Enter/Esc: continue\n"
                    f"window={search_half}, kept={int(sum(selected))}/{len(stars)}, added by hand={len(manual)}"
                    + (f"\n{state['msg']}" if state["msg"] else ""),
                    fontsize=7.5,
                )
                ax.set_xlabel("x [pix]", fontsize=8)
                ax.set_ylabel("y [pix]", fontsize=8)
                ax.tick_params(labelsize=7)
                fig.canvas.draw_idle()

            def nearest(x, y, r2):
                best_i, best_d2 = None, None
                for i, s in enumerate(stars):
                    d2 = (x - s.x_ref) ** 2 + (y - s.y_ref) ** 2
                    if d2 <= r2 and (best_d2 is None or d2 < best_d2):
                        best_d2, best_i = d2, i
                return best_i

            def remove_star(idx):
                stars.pop(idx)
                selected.pop(idx)
                manual[:] = [m - 1 if m > idx else m for m in manual if m != idx]

            def add_star(x, y):
                s = refine_manual_star(im, x, y, search_half, min_snr=5.0)
                if s is None:
                    state["msg"] = "no star found at this place (or too close to the frame edge)"
                    return
                if nearest(s.x_ref, s.y_ref, min_d2) is not None:
                    state["msg"] = "there is already a star here"
                    return
                stars.append(s)
                selected.append(True)
                manual.append(len(stars) - 1)
                state["msg"] = f"added M{len(stars)} at ({s.x_ref:.1f}, {s.y_ref:.1f})"
                print(f"[shift stars] manual star added: x={s.x_ref:.2f} y={s.y_ref:.2f} peak={s.peak:.1f}", flush=True)

            def toolbar_busy():
                tb = getattr(fig.canvas, "toolbar", None)
                mode = getattr(tb, "mode", "") if tb is not None else ""
                return bool(str(getattr(mode, "value", mode)))

            def on_click(event):
                if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                    return
                if toolbar_busy():  # zoom / pan active
                    return
                x = float(event.xdata)
                y = float(event.ydata)
                state["msg"] = ""
                best_i = nearest(x, y, float(max(8, 0.6 * search_half) ** 2))
                if best_i is not None:
                    if event.button == 3 and best_i in manual:
                        remove_star(best_i)
                        state["msg"] = "added star deleted"
                    else:
                        selected[best_i] = not selected[best_i]
                else:
                    add_star(x, y)
                redraw()

            done = {'value': False}
            def on_key(event):
                key = (event.key or "").lower()
                if key in ('enter', 'return', 'escape'):
                    done['value'] = True
                    plt.close(fig)
                elif key == 'u' and manual:
                    remove_star(manual[-1])
                    state["msg"] = "last added star deleted"
                    redraw()

            fig.canvas.mpl_connect('button_press_event', on_click)
            fig.canvas.mpl_connect('key_press_event', on_key)
            redraw()
            _raise_window(fig)
            print("[shift stars] WAITING: select the stars in the window 'shift stars' "
                  f"(click = exclude / add, Enter or Esc = continue; backend {matplotlib.get_backend()})", flush=True)
            while plt.fignum_exists(fig.number) and not done['value']:
                plt.pause(0.1)
    except Exception as e:
        print(f"[shift stars] ERROR in the star-selection window: {type(e).__name__}: {e}; "
              f"using the {len(orig)} automatically found stars", flush=True)
        return orig
    return [s for i, s in enumerate(stars) if selected[i]]


def _render_frame_preview(fig, cube_frame: np.ndarray, stars: List[RefStar], frame_meas: List[StarMeasure], dy: float, dx: float, window: int, k: int) -> None:
    nshow = min(12, len(stars))
    ncols = min(4, max(1, nshow))
    nrows = int(np.ceil(max(1, nshow) / ncols))
    fig.clf()
    axes = fig.subplots(nrows, ncols)
    axes = np.atleast_1d(axes).ravel()
    for i in range(nshow):
        ax = axes[i]
        m = frame_meas[i]
        st, x0, y0 = _extract_stamp(cube_frame, m.x_prev, m.y_prev, int(window))
        if st is None:
            ax.set_axis_off()
            continue
        vmin, vmax = np.percentile(st, [10, 99.5])
        ax.imshow(st, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        ax.axvline(window, color='deepskyblue', lw=0.9, ls='--')
        ax.axhline(window, color='deepskyblue', lw=0.9, ls='--')
        ax.add_patch(plt.Rectangle((0, 0), st.shape[1] - 1, st.shape[0] - 1, fill=False, ec='deepskyblue', lw=0.8, alpha=0.8))
        ax.plot(m.x_prev - x0, m.y_prev - y0, marker='+', color='deepskyblue', ms=9, mew=1.3)
        if np.isfinite(m.x_peak):
            ax.plot(m.x_peak - x0, m.y_peak - y0, marker='o', mfc='none', mec='yellowgreen', ms=8, mew=1.3)
        if m.ok:
            ax.add_patch(plt.Circle((m.x_fit - x0, m.y_fit - y0), radius=max(2.5, 0.12 * window), fill=False, ec='yellowgreen', lw=1.2))
            ax.plot(m.x_fit - x0, m.y_fit - y0, marker='x', color='crimson', ms=8, mew=1.5)
        ax.set_title(f'S{i+1} prev=({m.x_prev:.1f},{m.y_prev:.1f})', fontsize=7)
        info = f'ok={int(m.ok)} peak={m.peak:.1f}' if np.isfinite(m.peak) else 'ok=0'
        ax.text(0.02, 0.02, info, transform=ax.transAxes, color='white', fontsize=8, va='bottom', ha='left')
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(nshow, len(axes)):
        axes[j].set_axis_off()
    fig.suptitle(f'Frame {k+1}: apply dy={dy:+.3f}, dx={dx:+.3f}, window={window}', fontsize=10)


def save_window_diagnostics(out_dir: Path, cube: np.ndarray, stars: List[RefStar], measurements: List[List[StarMeasure]], shifts_apply: np.ndarray, window: int, preview_delay: float = 0.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(stars) == 0:
        return
    preview_fig = None
    if preview_delay > 0:
        plt.ion()
        preview_fig = plt.figure(figsize=(7.2, 5.4), dpi=110)
    for k in range(cube.shape[0]):
        with plt.rc_context({'text.usetex': False}):
            fig = plt.figure(figsize=(7.2, 5.4), dpi=140)
            _render_frame_preview(fig, cube[k], stars, measurements[k], float(shifts_apply[k, 0]), float(shifts_apply[k, 1]), window, k)
            fig.savefig(out_dir / f'frame_{k:03d}.png', bbox_inches='tight')
            if preview_fig is not None:
                _render_frame_preview(preview_fig, cube[k], stars, measurements[k], float(shifts_apply[k, 0]), float(shifts_apply[k, 1]), window, k)
                preview_fig.canvas.draw_idle()
                plt.pause(preview_delay)
            plt.close(fig)
    if preview_fig is not None:
        plt.pause(max(0.2, float(preview_delay)))
        plt.ioff()
        plt.close(preview_fig)


def frame_cube_ids(in_fits: str | Path, nz: int) -> Optional[List[str]]:
    """Cube number of every frame (from the FILE column of the .inputs.csv table), or None."""
    import re as _re
    import pandas as _pd
    in_fits = Path(in_fits)
    cands = [find_side(in_fits, ".inputs.csv")]
    try:
        cands.append(find_side(stage_path(in_fits, "darkflat"), ".inputs.csv"))
    except Exception:
        pass
    for c in cands:
        if not c.exists():
            continue
        try:
            files = _pd.read_csv(c)["FILE"].astype(str).tolist()
        except Exception:
            continue
        if len(files) != nz:
            continue
        ids = []
        for f in files:
            m = _re.search(r"_(\d{2})\d{2}$", Path(f).stem if f.endswith((".fts", ".fits")) else f)
            ids.append(m.group(1) if m else "?")
        return ids
    return None


def register_cube_to(target_img: np.ndarray, cube_work: np.ndarray, idx: Sequence[int], cid: str, target_label: str,
                     manual: bool = False) -> dict:
    """
    Register one observing cube (its frames idx) to target_img, which is already on the reference grid, and resample
    its frames IN PLACE onto that grid. Stars matched with any rotation (as for the combination of nights) or, if that
    fails / manual=True, pairs of stars clicked in a window, refined with all stars.
    """
    from mangal_combine import (detect_stars, fit_affine, manual_pairs, match_stars, refine_transform,
                                warp_to_reference)
    idx = list(idx)
    ts = detect_stars(target_img)
    img = cube_work[idx[0]].copy()  # its first frame (a median of several frames would erase drifting stars)
    ms = detect_stars(img)
    m = None if manual else match_stars(ts, ms)
    if m is not None:
        A, b, rms, npairs, method = m["A"], m["b"], m["rms"], len(m["ref_idx"]), "auto"
    else:
        if not manual:
            print(f"[shift cubes] cube {cid}: automatic star match with {target_label} failed "
                  f"({len(ms)} stars): manual pairs", flush=True)
        rxy, mxy = manual_pairs(target_img, img, f"cube {cid} (frames {idx[0] + 1}-{idx[-1] + 1}) -> {target_label}")
        if len(rxy) < 3:
            raise NoReferenceStarsError(f"cube {cid}: at least 3 star pairs are needed to register it, got {len(rxy)}")
        A, b, _, rms = fit_affine(mxy, rxy)
        npairs, method = len(rxy), "manual"
        r2 = refine_transform(ts, ms, A, b)
        if r2 is not None:
            A, b, rms, npairs, method = r2["A"], r2["b"], r2["rms"], len(r2["ref_idx"]), "manual + refined"
    angle = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    print(f"[shift cubes] cube {cid} (frames {idx[0] + 1}-{idx[-1] + 1}) -> {target_label}: {method}, {npairs} stars, "
          f"rms {rms:.2f} px, rotation {angle:+.2f} deg, shift ({b[0]:+.1f}, {b[1]:+.1f}) px", flush=True)
    shape = cube_work.shape[1:]
    for k in idx:
        cube_work[k] = np.nan_to_num(warp_to_reference(cube_work[k], A, b, shape), nan=0.0)
    return {"cube": cid, "frames": (idx[0] + 1, idx[-1] + 1), "to": target_label, "method": method,
            "npairs": npairs, "rms": rms, "angle": angle, "A": A, "b": b}


def align_cube(
    cube: np.ndarray,
    ref_index: int = 0,
    exclude_center: Optional[Tuple[float, float, float]] = None,
    detect_snr: float = 20.0,
    search_half: int = 30,
    shift_order: int = 3,
    preview_delay: float = 1.0,
    background_mode: str = "none",
    background_window: int = 30,
    extra_stars: Optional[Sequence[Tuple[float, float]]] = None,
    interactive: bool = True,
    skip_frames: Optional[Sequence[int]] = None,
    frame_cubes: Optional[Sequence[str]] = None,
    cube_manual: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[RefStar], np.ndarray, List[ShiftDebug], List[List[StarMeasure]], float, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    cube0 = _nan(cube).astype(np.float32, copy=False)
    if cube0.ndim == 2:
        cube0 = cube0[None, :, :]
    nz = cube0.shape[0]

    cube_work = np.empty_like(cube0)
    bg_cube = np.zeros_like(cube0)
    bg_mask_cube = np.zeros_like(cube0, dtype=np.uint8)
    bg_on = str(background_mode or "none").strip().lower() not in ("none", "0", "off")
    if bg_on:
        print(f"[shift] background subtraction ({background_mode}, window {int(background_window)}) "
              f"in {nz} frames ...", flush=True)
    for k in range(nz):
        cube_work[k], bgk, bgmk = _apply_background_mode(cube0[k], mode=background_mode, win=int(background_window), exclude_center=exclude_center)
        if bg_on and (k + 1 == nz or (k + 1) % 5 == 0):
            print(f"[shift] background: {k + 1}/{nz}", flush=True)
        if bgk is not None:
            bg_cube[k] = np.asarray(bgk, dtype=np.float32)
        if bgmk is not None:
            bg_mask_cube[k] = np.asarray(bgmk, dtype=np.uint8)

    # several observing cubes: a cube may be shifted far beyond the search window, or rotated, with respect to the
    # previous one. Cubes BEFORE the reference cube are registered to the reference frame now; every cube AFTER it
    # is registered, when the frame loop reaches it, to the last shifted frame of the previous cube (the stars drift
    # within a cube, so the previous cube's last frame is what the tracker expects next: no jump at the boundary).
    cube_transforms: List[dict] = []
    multi = frame_cubes is not None and len(set(frame_cubes)) > 1
    fc = list(frame_cubes) if multi else []
    if multi:
        ref_cid = fc[int(ref_index)]
        first_ref = fc.index(ref_cid)
        for cid in dict.fromkeys(fc[:first_ref]):
            idx = [k for k, c in enumerate(fc) if c == cid]
            cube_transforms.append(register_cube_to(cube_work[int(ref_index)], cube_work, idx, cid,
                                                    f"reference frame {int(ref_index) + 1}", manual=cube_manual))

    ref_img = cube_work[int(ref_index)].copy()

    stars, threshold = find_reference_stars(ref_img, detect_snr=detect_snr, search_half=search_half, exclude_center=exclude_center)
    print(f"[shift stars] found automatically: {len(stars)}", flush=True)

    # stars given by hand as coordinates (x, y) on the reference frame (ignore the center exclusion)
    min_d2 = float(max(8, int(0.5 * search_half)) ** 2)
    for xm, ym in (extra_stars or []):
        st = refine_manual_star(ref_img, float(xm), float(ym), search_half)
        if st is None:
            print(f"[shift stars] extra star ({xm:.1f}, {ym:.1f}) skipped: edge of the frame / no signal", flush=True)
        elif any((st.x_ref - t.x_ref) ** 2 + (st.y_ref - t.y_ref) ** 2 < min_d2 for t in stars):
            print(f"[shift stars] extra star ({xm:.1f}, {ym:.1f}) skipped: already in the list", flush=True)
        else:
            stars.append(st)
            print(f"[shift stars] extra star added at ({st.x_ref:.2f}, {st.y_ref:.2f})", flush=True)

    if interactive:  # also when nothing was found: stars can be added by hand in the window
        stars = interactive_prune_stars(ref_img, stars, search_half=search_half, exclude_center=exclude_center)
        print(f"[shift stars] reference stars kept after manual selection: {len(stars)}", flush=True)
    if len(stars) == 0:
        ec = (f"center exclusion x={exclude_center[0]:.0f} y={exclude_center[1]:.0f} r={exclude_center[2]:.0f}"
              if exclude_center is not None else "no center exclusion")
        raise NoReferenceStarsError(
            f"no reference stars selected on frame {ref_index} (threshold={detect_snr:g} sigma, {ec}). "
            f"Lower the threshold / center exclusion r, or keep at least one star when pruning. "
            f"Shift stage was NOT run, no shifted cube was written.")

    prev_positions = np.array([[s.x_ref, s.y_ref] for s in stars], dtype=np.float32)
    ref_positions = prev_positions.copy()
    shifts_apply = np.zeros((nz, 2), dtype=np.float32)
    debug_rows: List[ShiftDebug] = []
    all_meas: List[List[StarMeasure]] = []
    preview_fig = None
    if preview_delay > 0:
        plt.ion()
        preview_fig = plt.figure(figsize=(7.2, 5.4), dpi=110)

    # frames (0-based) that must not take part in the shift measurement
    skip = set()
    for f in (skip_frames or []):
        f = int(f)
        if not (0 <= f < nz):
            print(f"[shift stars] skip frame {f + 1} ignored: the cube has {nz} frames", flush=True)
        elif f == int(ref_index):
            print(f"[shift stars] frame {f + 1} is the reference frame (stars were defined on it): it cannot be skipped", flush=True)
        else:
            skip.add(f)
    if skip:
        print("[shift stars] frames not used for the shifts (1-based): " + ", ".join(str(f + 1) for f in sorted(skip)), flush=True)

    for k in range(nz):
        # start of a new observing cube after the reference cube: register it to the last shifted frame of the
        # previous cube and continue the tracking from the reference positions (the cube is now on the reference grid)
        if multi and k > int(ref_index) and fc[k] != fc[k - 1] and fc.index(fc[k]) == k:
            prev = [j for j in range(k) if fc[j] == fc[k - 1]]
            prev_meas = [j for j in prev if j not in skip] or prev
            j_last = prev_meas[-1]
            target = ndi_shift(cube_work[j_last], shift=(float(shifts_apply[j_last, 0]), float(shifts_apply[j_last, 1])),
                               order=1, mode="constant", cval=0.0)
            idx = [i for i, c in enumerate(fc) if c == fc[k]]
            cube_transforms.append(register_cube_to(target, cube_work, idx, fc[k],
                                                    f"shifted frame {j_last + 1} (cube {fc[j_last]})", manual=cube_manual))
            prev_positions = ref_positions.copy()
        frame_meas: List[StarMeasure] = []
        dy_raws = []
        dx_raws = []
        good_idx = []
        if k in skip:
            # no measurement: keep the tracker where it was and hold the previous shift for now;
            # the real value is interpolated from the neighbouring measured frames after the loop
            dy_apply = float(shifts_apply[k - 1, 0]) if k > 0 else 0.0
            dx_apply = float(shifts_apply[k - 1, 1]) if k > 0 else 0.0
            frame_meas = [StarMeasure(False, float(prev_positions[i, 0]), float(prev_positions[i, 1]),
                                      np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan) for i in range(len(stars))]
            shifts_apply[k, 0] = dy_apply
            shifts_apply[k, 1] = dx_apply
            debug_rows.append(ShiftDebug("skipped", 0, len(stars), -dy_apply, -dx_apply, dy_apply, dx_apply))
            all_meas.append(frame_meas)
            print(f"[shift stars] {k+1}/{nz}: skipped (not used for the shifts)", flush=True)
            continue
        for i, s in enumerate(stars):
            x_prev, y_prev = prev_positions[i]
            m = measure_star_in_window(cube_work[k], float(x_prev), float(y_prev), search_half)
            frame_meas.append(m)
            if not m.ok:
                dy_raws.append(np.nan)
                dx_raws.append(np.nan)
                continue
            dy_raw = float(m.y_fit - s.y_ref)
            dx_raw = float(m.x_fit - s.x_ref)
            dy_raws.append(dy_raw)
            dx_raws.append(dx_raw)
            good_idx.append(i)

        dy_arr = np.array(dy_raws, dtype=np.float32)
        dx_arr = np.array(dx_raws, dtype=np.float32)
        _, _, keep = _clean_raw_shifts(dy_arr, dx_arr)
        ngood = int(np.sum(keep))

        if ngood >= 2:
            w = np.array([max(float(m.peak), 1.0) if (i < len(frame_meas) and frame_meas[i].ok and keep[i] and np.isfinite(frame_meas[i].peak)) else 0.0 for i in range(len(frame_meas))], dtype=np.float32)
            if np.sum(w[keep]) > 0:
                dy_raw = float(np.average(dy_arr[keep], weights=w[keep]))
                dx_raw = float(np.average(dx_arr[keep], weights=w[keep]))
                method = "stars+w"
            else:
                dy_raw = float(np.median(dy_arr[keep]))
                dx_raw = float(np.median(dx_arr[keep]))
                method = "stars"
            dy_apply = -dy_raw
            dx_apply = -dx_raw
            for i, m in enumerate(frame_meas):
                if m.ok and keep[i]:
                    prev_positions[i, 0] = m.x_fit
                    prev_positions[i, 1] = m.y_fit
                else:
                    prev_positions[i, 0] = ref_positions[i, 0] + dx_raw
                    prev_positions[i, 1] = ref_positions[i, 1] + dy_raw
        elif k > 0:
            dy_apply = float(shifts_apply[k - 1, 0])
            dx_apply = float(shifts_apply[k - 1, 1])
            dy_raw = -dy_apply
            dx_raw = -dx_apply
            method = "prev"
            print(f"[shift stars] WARNING: frame {k + 1}: fewer than 2 stars found within the window ({search_half} px): "
                  f"the shift of the previous frame is used. Larger window, 'skip frames', or manual star pairs "
                  f"(frames of another cube)", flush=True)
            prev_positions[:, 0] = ref_positions[:, 0] + dx_raw
            prev_positions[:, 1] = ref_positions[:, 1] + dy_raw
        else:
            dy_apply = dx_apply = dy_raw = dx_raw = 0.0
            method = "zero"

        shifts_apply[k, 0] = dy_apply
        shifts_apply[k, 1] = dx_apply
        debug_rows.append(ShiftDebug(method, ngood, len(stars), dy_raw, dx_raw, dy_apply, dx_apply))
        all_meas.append(frame_meas)
        print(f"[shift stars] {k+1}/{nz}: dy={dy_apply:+.3f}, dx={dx_apply:+.3f}, ngood={ngood}/{len(stars)}, method={method}", flush=True)
        if preview_fig is not None:
            with plt.rc_context({'text.usetex': False}):
                _render_frame_preview(preview_fig, cube_work[k], stars, frame_meas, dy_apply, dx_apply, search_half, k)
                preview_fig.canvas.draw_idle()
                plt.pause(preview_delay)

    if preview_fig is not None:
        plt.pause(max(0.2, float(preview_delay)))
        plt.ioff()
        plt.close(preview_fig)

    if skip:  # interpolate the skipped frames from the measured ones (hold the nearest at the ends)
        measured = [k for k in range(nz) if k not in skip]
        for k in sorted(skip):
            dy_i = float(np.interp(k, measured, shifts_apply[measured, 0]))
            dx_i = float(np.interp(k, measured, shifts_apply[measured, 1]))
            shifts_apply[k, 0], shifts_apply[k, 1] = dy_i, dx_i
            debug_rows[k] = ShiftDebug("skipped(interp)", 0, len(stars), -dy_i, -dx_i, dy_i, dx_i)
            print(f"[shift stars] {k+1}/{nz}: skipped -> interpolated dy={dy_i:+.3f}, dx={dx_i:+.3f}", flush=True)

    aligned = np.empty_like(cube0)
    for k in range(nz):
        aligned[k] = ndi_shift(cube_work[k], shift=(float(shifts_apply[k, 0]), float(shifts_apply[k, 1])), order=shift_order, mode="constant", cval=0.0, prefilter=True)

    return aligned, shifts_apply, stars, ref_img, debug_rows, all_meas, threshold, cube_work, bg_cube, bg_mask_cube, cube_transforms


def write_shifts_csv(path: Path, shifts_apply: np.ndarray, debug_rows: List[ShiftDebug]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("k,dy,dx,method,ngood,ntotal,dy_raw,dx_raw,dy_apply,dx_apply\n")
        for k, (sh, dbg) in enumerate(zip(shifts_apply, debug_rows)):
            f.write(f"{k},{float(sh[0]):.6f},{float(sh[1]):.6f},{dbg.method},{dbg.ngood},{dbg.ntotal},{dbg.dy_raw:.6f},{dbg.dx_raw:.6f},{dbg.dy_apply:.6f},{dbg.dx_apply:.6f}\n")


def write_stars_xy(path: Path, stars: List[RefStar]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for s in stars:
            f.write(f"{s.x_ref:.3f} {s.y_ref:.3f}\n")


def align_fits(
    in_fits: str | Path,
    out_fits: str | Path,
    ref_index: int = 0,
    exclude_center: Optional[Tuple[float, float, float]] = None,
    detect_snr: float = 20.0,
    search_half: int = 30,
    dump_frames: bool = True,
    preview_delay: float = 1.0,
    background_mode: str = "none",
    background_window: int = 30,
    extra_stars: Optional[Sequence[Tuple[float, float]]] = None,
    interactive: bool = True,
    skip_frames: Optional[Sequence[int]] = None,
    cube_manual: bool = False,
) -> Tuple[Path, Path, Path, Path, Optional[Path]]:
    in_fits = Path(in_fits)
    out_fits = Path(out_fits)
    with fits.open(in_fits) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header.copy()
    frame_cubes = frame_cube_ids(in_fits, data.shape[0] if data.ndim == 3 else 1)
    aligned, shifts, stars, ref_img, debug_rows, measurements, threshold, *_rest, cube_tr = align_cube(
        data,
        ref_index=ref_index,
        exclude_center=exclude_center,
        detect_snr=detect_snr,
        search_half=search_half,
        preview_delay=preview_delay,
        background_mode=background_mode,
        background_window=background_window,
        extra_stars=extra_stars,
        interactive=interactive,
        skip_frames=skip_frames,
        frame_cubes=frame_cubes,
        cube_manual=cube_manual,
    )
    for t in cube_tr:
        A, b = t["A"], t["b"]
        hdr["HISTORY"] = (f"SHIFT: cube {t['cube']} registered ({t['method']}): A=[[{A[0,0]:.6f},{A[0,1]:.6f}],"
                          f"[{A[1,0]:.6f},{A[1,1]:.6f}]] b=[{b[0]:.3f},{b[1]:.3f}] rms={t['rms']:.3f}px")
    if str(background_mode).strip().lower() not in ("none", "0", "off"):  # else keep BGMODE of step 1
        hdr['BGMODE'] = str(background_mode)
        hdr['BGWIN'] = int(background_window)
    stamp(hdr, out_fits)
    fits.PrimaryHDU(aligned.astype(np.float32), header=hdr).writeto(out_fits, overwrite=True)
    shifts_csv = side_path(out_fits, ".shifts.csv")
    stars_xy = side_path(out_fits, ".stars.xy")
    stars_png = side_path(out_fits, ".stars.png")
    write_shifts_csv(shifts_csv, shifts, debug_rows)
    write_stars_xy(stars_xy, stars)
    save_stars_png(stars_png, ref_img, stars, threshold=threshold, search_half=search_half, exclude_center=exclude_center, title=f"Step4 reference frame {ref_index}")
    frames_dir = None
    if dump_frames:
        frames_dir = side_path(out_fits, "_frames")
        save_window_diagnostics(frames_dir, _nan(aligned), stars, measurements, shifts, search_half, preview_delay=0.0)
    return out_fits, shifts_csv, stars_xy, stars_png, frames_dir


def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("in_fits")
    ap.add_argument("out_fits")
    ap.add_argument("--ref-index", type=int, default=0)
    ap.add_argument("--detect-snr", type=float, default=20.0)
    ap.add_argument("--search-half", type=int, default=30)
    ap.add_argument("--dump-frames", action="store_true")
    ap.add_argument("--preview-delay", type=float, default=1.0)
    ap.add_argument("--exclude-center-x", type=float, default=None)
    ap.add_argument("--exclude-center-y", type=float, default=None)
    ap.add_argument("--exclude-center-r", type=float, default=None)
    ap.add_argument("--background-mode", default="backgr_sub")
    ap.add_argument("--background-window", type=int, default=30)
    args = ap.parse_args()
    exclude = None
    if args.exclude_center_x is not None and args.exclude_center_y is not None and args.exclude_center_r is not None:
        exclude = (args.exclude_center_x, args.exclude_center_y, args.exclude_center_r)
    align_fits(args.in_fits, args.out_fits, ref_index=args.ref_index, exclude_center=exclude, detect_snr=args.detect_snr, search_half=args.search_half, dump_frames=args.dump_frames, preview_delay=args.preview_delay, background_mode=args.background_mode, background_window=args.background_window)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
