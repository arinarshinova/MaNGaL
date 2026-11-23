#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaNGaL step5: transparency diagnostics via stellar photometry + frame rejection.

IDL reference: DIWID.PRO step (5) CH-photometry + step (6) CH-remove
- For each slice, fit a 2D elliptical MOFFAT profile to pre-selected stars
- Compute per-slice metrics:
    RelFlux  = mean over stars of (Flux_i / median(Flux_* for that star))
    FWHMpx   = mean over stars of sqrt(FWHMx^2 + FWHMy^2)
    Ratio    = mean over stars of min(FWHMy/FWHMx, FWHMx/FWHMy)  (<=1)
- Mark slice as "bad" if ANY metric fails limits:
    RelFlux < lim_flux  OR  FWHMpx > lim_fwhm  OR  Ratio < lim_ratio
- Save:
    <obj>_phot.csv        (table with good/bad flags), <obj>_phot.png (plot)
    <obj>_phot.fits       (filtered cube, only good slices)
    <obj>_phot.stars.csv  (per-star fits, optional, can be large)

Inputs expected from step4:
- <object>_shift.fits
- <object>_shift.stars.xy  (text: x y [peak], one star per line)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Optional, List

from mangal_header import apply_frame_keys, stamp
from mangal_names import side_path, find_side
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

try:  # optional: nicer plots with SciencePlots (without LaTeX); plain matplotlib otherwise
    import scienceplots  # noqa: F401  (registers the 'science' style)
    _PLOT_STYLE = ["science", "no-latex"]
except ImportError:
    _PLOT_STYLE = ["default"]
_PLOT_RC = {"font.family": "DejaVu Sans", "font.size": 16, "pdf.fonttype": 42}


@dataclass(frozen=True)
class StarXY:
    x: float
    y: float


def _nan_to_num(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _robust_sigma(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return float(1.4826 * mad)


def collapse_cube_median(cube: np.ndarray) -> np.ndarray:
    """Collapse cube to 2D reference image (IDL-style median over slices)."""
    cube = _nan_to_num(cube)
    return np.median(cube, axis=0)


def write_stars_xy(path: Path, stars: List[StarXY]) -> None:
    lines = ["# x  y  (auto-picked)"]
    for s in stars:
        lines.append(f"{s.x:.3f} {s.y:.3f}")
    path.write_text("\n".join(lines) + "\n")


def _read_stars_xy(path: str | Path) -> List[StarXY]:
    p = Path(path)
    rows: List[StarXY] = []
    for raw in p.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0]); y = float(parts[1])
        except ValueError:
            continue
        rows.append(StarXY(x=x, y=y))
    return rows


def _extract_stamp(img: np.ndarray, x: float, y: float, win: int) -> Tuple[np.ndarray, int, int]:
    """Return stamp and integer center offsets (x0i,y0i) inside stamp coordinates."""
    ny, nx = img.shape
    xi = int(round(x))
    yi = int(round(y))
    if xi - win < 0 or yi - win < 0 or xi + win >= nx or yi + win >= ny:
        raise ValueError("star too close to edge")
    stamp = img[yi - win : yi + win + 1, xi - win : xi + win + 1].astype(np.float32, copy=False)
    return stamp, xi, yi


def _moffat_model(params: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """
    Elliptical rotated Moffat:
        I = sky + amp * (1 + r2)^(-beta),
        r2 = (x'/ax)^2 + (y'/ay)^2,
        x' =  (x-x0)*cos(t) + (y-y0)*sin(t)
        y' = -(x-x0)*sin(t) + (y-y0)*cos(t)
    params = [sky, amp, ax, ay, x0, y0, theta, beta]
    """
    sky, amp, ax, ay, x0, y0, th, beta = params
    ct = np.cos(th); st = np.sin(th)
    dx = xx - x0
    dy = yy - y0
    xp = dx * ct + dy * st
    yp = -dx * st + dy * ct
    r2 = (xp / ax) ** 2 + (yp / ay) ** 2
    return sky + amp * (1.0 + r2) ** (-beta)


def _fit_star_moffat(stamp: np.ndarray, *, max_iter: int = 200) -> Tuple[bool, dict]:
    """
    Fit a MOFFAT profile to a stamp.
    Returns (ok, metrics) with metrics containing:
        flux, fwhm_x, fwhm_y, ratio (<=1), sky, beta, x0, y0
    """
    # Coordinates in stamp space
    h, w = stamp.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Robust background from border pixels (like IDL uses local background)
    border = np.hstack([
        stamp[0, :], stamp[-1, :], stamp[:, 0], stamp[:, -1]
    ])
    sky0 = float(np.median(border))
    data = stamp - sky0

    # initial peak location and amplitude
    mp = np.nanargmax(data)
    y0p, x0p = np.unravel_index(int(mp), data.shape)
    amp0 = float(np.clip(data[y0p, x0p], 1.0, None))

    # width guess from second moments (clipped)
    pos = np.clip(data, 0, None)
    s = float(pos.sum())
    if s <= 0:
        return False, {}
    xbar = float((xx * pos).sum() / s)
    ybar = float((yy * pos).sum() / s)
    vx = float(((xx - xbar) ** 2 * pos).sum() / s)
    vy = float(((yy - ybar) ** 2 * pos).sum() / s)
    ax0 = float(np.clip(np.sqrt(vx), 0.8, 8.0))
    ay0 = float(np.clip(np.sqrt(vy), 0.8, 8.0))

    # beta initial
    beta0 = 2.5
    th0 = 0.0

    # params = [sky, amp, ax, ay, x0, y0, theta, beta]
    p0 = np.array([sky0, amp0, ax0, ay0, float(x0p), float(y0p), th0, beta0], dtype=np.float64)

    # bounds: keep beta>1.05 (flux finite), widths positive, theta any
    lb = np.array([sky0 - 5*np.std(border), 0.0, 0.3, 0.3, 0.0, 0.0, -np.pi, 1.05], dtype=np.float64)
    ub = np.array([sky0 + 5*np.std(border) + np.max(stamp), np.max(stamp) * 5, 30.0, 30.0,
                   float(w - 1), float(h - 1), np.pi, 10.0], dtype=np.float64)

    def resid(p: np.ndarray) -> np.ndarray:
        mod = _moffat_model(p, xx, yy)
        r = (mod - stamp).ravel()
        # down-weight outliers (cosmics) via soft clipping
        s = np.median(np.abs(r)) + 1e-6
        r = np.clip(r, -8*s, 8*s)
        return r

    try:
        res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=max_iter, method="trf")
    except Exception:
        return False, {}

    if not res.success:
        return False, {}

    sky, amp, ax, ay, x0, y0, th, beta = res.x

    # reject fits wandering outside plausible region (IDL: 0<a(4,5)<2*win)
    if not (0.0 <= x0 <= w - 1 and 0.0 <= y0 <= h - 1):
        return False, {}

    # FWHM along axes (IDL mulw=2*sqrt(2^(1/beta)-1), xfw=ax*mulw, yfw=ay*mulw)
    mulw = 2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0)
    fwhm_x = float(ax * mulw)
    fwhm_y = float(ay * mulw)

    # total flux (IDL: amp*ax*ay*pi/(beta-1))
    flux = float(amp * ax * ay * np.pi / (beta - 1.0))

    # ratio like IDL: yfw/xfw but forced <= 1
    if fwhm_x <= 0 or fwhm_y <= 0:
        return False, {}
    ratio = float(fwhm_y / fwhm_x)
    if ratio > 1:
        ratio = 1.0 / ratio

    return True, {
        "flux": flux,
        "fwhm_x": fwhm_x,
        "fwhm_y": fwhm_y,
        "fwhm": float(np.hypot(fwhm_x, fwhm_y)),
        "ratio": ratio,
        "sky": float(sky),
        "beta": float(beta),
        "x0": float(x0),
        "y0": float(y0),
    }


def photometry_cube(
    cube: np.ndarray,
    stars: Iterable[StarXY],
    *,
    win: int = 30,
    lim_flux: float = 0.0,
    lim_fwhm: float = 1e9,
    lim_ratio: float = 0.0,
    save_per_star: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[pd.DataFrame]]:
    """
    Run IDL-like CH-photometry on a (nslice, ny, nx) cube.
    Returns:
      - table per slice
      - good_mask (nslice,)
      - per-star table (optional)
    """
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError("cube must be (nslice, ny, nx)")

    star_list = list(stars)
    if len(star_list) == 0:
        raise ValueError("no stars provided")

    ns = cube.shape[0]
    nst = len(star_list)

    # fits[k, i, s] like IDL:
    # k=0 flux/exptime, k=1 fwhm, k=2 ratio
    fits_arr = np.full((3, ns, nst), np.nan, dtype=np.float32)

    per_star_rows = [] if save_per_star else None

    for i in range(ns):
        img = _nan_to_num(cube[i])
        for sidx, st in enumerate(star_list):
            try:
                stamp, xi, yi = _extract_stamp(img, st.x, st.y, win=win)
            except Exception:
                continue
            ok, met = _fit_star_moffat(stamp)
            if not ok:
                continue
            fits_arr[0, i, sidx] = met["flux"]
            fits_arr[1, i, sidx] = met["fwhm"]
            fits_arr[2, i, sidx] = met["ratio"]
            if save_per_star:
                per_star_rows.append({
                    "slice": i,
                    "star": sidx,
                    "x": st.x,
                    "y": st.y,
                    **met
                })

    # IDL "good stars" logic:
    # keep only stars that have valid flux in ALL frames (no <0 or NaN)
    flux = fits_arr[0]
    good_star = np.all(np.isfinite(flux) & (flux > 0), axis=0)
    if not np.any(good_star):
        raise RuntimeError("no good stars (valid in all slices) after fitting")
    fits_arr = fits_arr[:, :, good_star]
    nst2 = fits_arr.shape[2]

    # Normalize flux per star by its median across frames
    for sidx in range(nst2):
        med = float(np.nanmedian(fits_arr[0, :, sidx]))
        if not np.isfinite(med) or med <= 0:
            continue
        fits_arr[0, :, sidx] /= med

    # Per-slice mean over stars (IDL uses MOMENT(...)->mean)
    rel_flux = np.nanmean(fits_arr[0], axis=1)
    mean_fwhm = np.nanmean(fits_arr[1], axis=1)
    mean_ratio = np.nanmean(fits_arr[2], axis=1)

    # Apply limits exactly like IDL:
    # div = res - lims; div(1) = -div(1); if any div < 0 => bad
    # => good if: rel_flux>=lim_flux AND mean_fwhm<=lim_fwhm AND mean_ratio>=lim_ratio
    good = (rel_flux >= lim_flux) & (mean_fwhm <= lim_fwhm) & (mean_ratio >= lim_ratio)

    tab = pd.DataFrame({
        "slice": np.arange(ns, dtype=int),
        "good": good.astype(bool),
        "RelFlux": rel_flux.astype(np.float64),
        "FWHMpx": mean_fwhm.astype(np.float64),
        "Ratio": mean_ratio.astype(np.float64),
        "nstars": int(nst2),
    })

    per_star_df = pd.DataFrame(per_star_rows) if save_per_star else None
    return tab, good, per_star_df


def plot_photometry_idl_style(
    out_png: Path,
    tab: pd.DataFrame,
    *,
    lim_flux: float,
    lim_fwhm: float,
    lim_ratio: float,
) -> None:
    """Create IDL-like 3-panel diagnostic plot (RelFlux, FWHM, Ratio vs slice)."""
    x = tab["slice"].to_numpy() + 1  # IDL uses 1-based indices
    y1 = tab["RelFlux"].to_numpy()
    y2 = tab["FWHMpx"].to_numpy()
    y3 = tab["Ratio"].to_numpy()

    with plt.style.context(_PLOT_STYLE), plt.rc_context(_PLOT_RC):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 9), dpi=300, sharex=True)

        panels = [
            (axes[0], y1, "Relative Flux", lim_flux),
            (axes[1], y2, "FWHM, px", lim_fwhm),
            (axes[2], y3, "FWHM min/max", lim_ratio),  # keep IDL label, even though it's ratio
        ]

        for ax, y, ylab, lim in panels:
            ax.plot(x, y, marker="o", linestyle="-", color='black')
            ax.axhline(lim, linewidth=2, color='crimson')
            ax.set_ylabel(ylab)

            # mark bad frames
            bad = ~tab["good"].to_numpy(dtype=bool)
            if bad.any():
                ax.plot(x[bad], y[bad], marker="x", linestyle="None", color='crimson')

            # IDL used yr=[0,max(res)] - keep non-negative for readability
            ymax = float(np.nanmax(y)) if np.isfinite(np.nanmax(y)) else 1.0
            ax.set_ylim(0, max(ymax * 1.05, lim * 1.05))

        axes[-1].set_xlabel("Slice")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

def run_step5(
    cube_in: str | Path,
    cube_out: str | Path,
    stars_xy: str | Path | None,
    *,
    win: int = 30,
    lim_flux: float = 0.85,
    lim_fwhm: float = 20.0,
    lim_ratio: float = 0.5,
    save_per_star: bool = False,
) -> Tuple[Path, Path]:
    cube_in = Path(cube_in)
    cube_out = Path(cube_out)

    data, hdr = fits.getdata(cube_in, header=True)
    cube = np.asarray(data, dtype=np.float32)
    if cube.ndim == 2:
        cube = cube[None, :, :]

    # Stars: use the list from the previous shift step (IDL-like workflow).
    if stars_xy is None:
        stars_xy = find_side(cube_in, ".stars.xy")
    stars_xy = Path(stars_xy)
    if not stars_xy.exists():
        raise FileNotFoundError(f"stars-xy not found: {stars_xy}")

    stars = _read_stars_xy(stars_xy)
    if len(stars) == 0:
        raise RuntimeError(f"stars list is empty: {stars_xy}")

    selected_stars_xy = side_path(cube_out, ".selected.stars.xy")
    write_stars_xy(selected_stars_xy, stars)

    tab, good, per_star = photometry_cube(
        cube,
        stars,
        win=win,
        lim_flux=lim_flux,
        lim_fwhm=lim_fwhm,
        lim_ratio=lim_ratio,
        save_per_star=save_per_star,
    )

    # Write photometry table
    phot_csv = side_path(cube_out, ".csv")
    tab.to_csv(phot_csv, index=False)

    # IDL-like diagnostic plot

    plot_photometry_idl_style(
        side_path(cube_out, ".png"),
        tab,
        lim_flux=lim_flux,
        lim_fwhm=lim_fwhm,
        lim_ratio=lim_ratio
    )


    # Optionally write per-star fits table
    if per_star is not None:
        per_star_csv = side_path(cube_out, ".stars.csv")
        per_star.to_csv(per_star_csv, index=False)

    # Filtered cube
    cube_f = cube[good]
    hdr2 = hdr.copy()
    hdr2["HISTORY"] = f"STEP5: transparency filter: kept {cube_f.shape[0]}/{cube.shape[0]} slices"
    hdr2["HISTORY"] = f"STEP5: limits: lim_flux={lim_flux} lim_fwhm={lim_fwhm} lim_ratio={lim_ratio} win={win}"
    hdr2["PHOTSRC"] = (cube_in.name[:60], "step5 input cube (slice order of the ZSPACE sidecar)")
    # exposure of what is really left: sum over the kept frames (sidecar of the input cube), ZMEAN, FILENAME, DATE
    try:
        side = find_side(cube_in, ".inputs.csv")
        fr = pd.read_csv(side) if side.exists() else None
        if fr is not None and len(fr) == len(good):
            apply_frame_keys(hdr2, fr[np.asarray(good, dtype=bool)], cube_out)
        else:
            stamp(hdr2, cube_out)
    except Exception as e:
        print(f"[warn] could not update EXPTIME/ZMEAN after the transparency filter: {e}")
        stamp(hdr2, cube_out)
    fits.writeto(cube_out, cube_f.astype(np.float32), hdr2, overwrite=True)

    return cube_out, phot_csv


def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="MaNGaL step5: photometry stability + reject bad slices")
    ap.add_argument("cube_in", help="Input cube (typically <object>_shift.fits)")
    ap.add_argument("cube_out", help="Output filtered cube")
    ap.add_argument("--win", type=int, default=30, help="Half-window (px) for star fitting (IDL win_stars)")
    ap.add_argument("--lim-flux", type=float, default=0.85, help="Min acceptable relative flux")
    ap.add_argument("--lim-fwhm", type=float, default=20.0, help="Max acceptable FWHM (sqrt(x^2+y^2)) in pixels")
    ap.add_argument("--lim-ratio", type=float, default=0.5, help="Min acceptable axis-ratio (<=1)")
    ap.add_argument("--save-per-star", action="store_true", help="Write per-star fits table (can be large)")

    a = ap.parse_args()
    run_step5(
        a.cube_in, a.cube_out, a.stars_xy,
        win=a.win, lim_flux=a.lim_flux, lim_fwhm=a.lim_fwhm, lim_ratio=a.lim_ratio,
        save_per_star=a.save_per_star,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
