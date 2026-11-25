"""
Combination of the line / continuum cubes of one object observed on different nights.

Every night is reduced on its own up to the line+cont stage (own darks, flat, shift, frame selection, ZSPACE log).
Then the <obj>_linecont.fits cubes are combined here into one cube, as if it were one night:

1. Registration: the first night is the reference. Stars are found in the summed image of every night and matched
   with the reference by a rotation + shift search (pairs of stars with the same separation vote for the rotation
   angle and the shift), then an affine transformation is fitted. When the automatic match fails (or on request)
   the pairs are clicked in a window.
2. Flux scale to the reference night, one factor per night:
     stars      median ratio of the fluxes of the matched field stars (sensitivity + transparency);
     standards  ratio of the QE calibrations of the nights, coefficient(night) / coefficient(reference);
     both       stars, and the standard-star ratio is reported for comparison (a difference means that the
                transparency was not the same for the standard and the object); the standard ratio is used
                when there are not enough stars.
3. Layers with the same name (Ha, NII, cont1, ...) are averaged with the exposure as weight, in counts/sec of the
   reference night, like the slices of one night. Outside the footprint of a night only the other nights count.
4. ZSPACE values of different nights are not comparable (the ZSPACE calibration changes), so every layer gets
   the mean wavelength LLAMn: ZSPACE -> wavelength with the .calib file of its own night, averaged with the
   exposure as weight. EXPTIME, NFRAMES, ZMEAN, TLINE, TCONT are summed / combined over the nights.

The result <obj>_linecont.fits (in its own folder) is processed by net / astrometry / apply ergs as usual; the erg
calibration of the reference night applies to it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import map_coordinates

from mangal_header import combine_zmean, stamp
from mangal_names import base_name, find_side, read_layer_table, side_path, write_layer_table


# ----------------------------------------------------------------------------------------------------------
# night description
# ----------------------------------------------------------------------------------------------------------
def _param_file_of(w_dir: Path, object_name: str | None) -> Path | None:
    """Parameter file written by the GUI into W_DIR (<obj>_auto.txt), or any *.txt with R_DIR = ..."""
    cands = []
    if object_name:
        cands.append(w_dir / f"{object_name}_auto.txt")
    cands += sorted(w_dir.glob("*_auto.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        try:
            if p.exists() and "R_DIR" in p.read_text(encoding="utf-8", errors="ignore"):
                return p
        except Exception:
            continue
    return None


def find_night_qe_json(r_dir: Path | None, filter_name: str | None) -> Path | None:
    """
    QE_*.json of the night of r_dir: the json records the .calib file it used, which lies in R_DIR of its night.
    Only a json of this very night is returned (None otherwise): nights lying side by side must not mix.
    """
    if r_dir is None:
        return None
    r_dir = Path(r_dir).resolve()
    for p in sorted(r_dir.parent.rglob("QE_*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if filter_name and str(d.get("filter", "")).strip() != str(filter_name).strip():
            continue
        cf = d.get("calib_file")
        if cf and Path(cf).resolve().parent == r_dir:
            return p
    return None


def night_info(w_dir, object_name: str | None = None, linecont: Path | None = None) -> dict:
    """linecont cube, R_DIR, filter, .calib and QE json of the night reduced in w_dir."""
    from mangal_step3 import parse_param_file
    from mangal_step8_ergcal import _find_calib_file

    w_dir = Path(w_dir)
    if linecont is None:
        cands = []
        if object_name:
            cands.append(w_dir / f"{object_name}_linecont.fits")
        cands += sorted(w_dir.glob("*_linecont.fits"), key=lambda p: p.stat().st_mtime, reverse=True)
        linecont = next((c for c in cands if c.exists()), None)
    if linecont is None:
        raise FileNotFoundError(f"No *_linecont.fits in {w_dir}: run the line+cont stage for this night first")
    hdr = fits.getheader(linecont)
    info = {"w_dir": w_dir, "linecont": Path(linecont), "r_dir": None, "filter": None, "calib": None,
            "qe_json": None, "qe_coef": None, "date": str(hdr.get("DATE-OBS", "") or "")}
    filt = str(hdr.get("FILTERS") or hdr.get("FILTER") or "").strip() or None
    pf = _param_file_of(w_dir, object_name)
    if pf is not None:
        try:
            pp = parse_param_file(pf)
            info["r_dir"] = Path(pp["R_DIR"])
            filt = filt or (str(pp.get("FILTERS", "")).strip() or None)
        except Exception:
            pass
    info["filter"] = filt
    if info["r_dir"] is not None:
        try:
            info["calib"] = _find_calib_file(info["r_dir"])
        except Exception:
            info["calib"] = None
        info["qe_json"] = find_night_qe_json(info["r_dir"], filt)
        if info["qe_json"] is not None:
            d = json.loads(Path(info["qe_json"]).read_text(encoding="utf-8"))
            info["qe_coef"] = float(d.get("line_mul_adu_to_erg", d.get("median_mul_adu_to_erg")))
    return info


def _layer_zspaces(layer: dict) -> list[int]:
    out = []
    for t in str(layer.get("zspace", "")).replace("/", ";").split(";"):
        t = t.strip()
        if t:
            try:
                out.append(int(float(t)))
            except ValueError:
                pass
    return out


def layer_lambda(layer: dict, calib: Path | None, filter_name: str | None, zcount: dict | None = None) -> float | None:
    """Mean wavelength of a layer: its ZSPACE values -> wavelength with the night's .calib, weighted by frames."""
    if layer.get("lam") not in (None, ""):
        return float(layer["lam"])
    zs = _layer_zspaces(layer)
    if not zs or calib is None or not filter_name:
        return None
    from mangal_step8_ergcal import mangal_z_to_lambda
    lams, ws = [], []
    for z in zs:
        try:
            lams.append(mangal_z_to_lambda(calib, filter_name, z)[0])
            ws.append(float((zcount or {}).get(z, 1)))
        except Exception:
            return None
    return float(np.average(lams, weights=ws)) if sum(ws) > 0 else float(np.mean(lams))


def _zspace_counts(linecont: Path) -> dict:
    """{ZSPACE: number of used slices} from the slice report <obj>_linecont.csv (equal weights without it)."""
    rep = find_side(linecont, ".csv", legacy_ext=".linecont.csv")
    if not rep.exists():
        return {}
    try:
        df = pd.read_csv(rep)
        df = df[df["used"].astype(int) == 1]
        return {int(z): int(n) for z, n in df["ZSPACE"].value_counts().items()}
    except Exception:
        return {}


# ----------------------------------------------------------------------------------------------------------
# stars and registration
# ----------------------------------------------------------------------------------------------------------
def sum_image(cube: np.ndarray, layers: list[dict]) -> np.ndarray:
    """Exposure-weighted mean of all layers: the deepest image of the night, used to find and match stars."""
    w = np.array([float(L.get("texp") or 1.0) for L in layers], dtype=float)
    acc = np.zeros(cube.shape[1:], dtype=float)
    ww = np.zeros(cube.shape[1:], dtype=float)
    for k in range(cube.shape[0]):
        ok = np.isfinite(cube[k])
        acc[ok] += w[k] * cube[k][ok]
        ww[ok] += w[k]
    out = np.full(cube.shape[1:], np.nan)
    good = ww > 0
    out[good] = acc[good] / ww[good]
    return out


def detect_stars(img: np.ndarray, max_stars: int = 60) -> np.ndarray:
    """Point sources (x, y, flux), brightest first."""
    from mangal_step7_net_astro import find_point_sources
    src = find_point_sources(img, img, nsigma=8.0, max_sources=400)
    src = [s for s in src if not s.get("cont_only")] or src
    src = sorted(src, key=lambda s: -s["flux_cont"])[:max_stars]
    return np.array([[s["xc"], s["yc"], s["flux_cont"]] for s in src], dtype=float).reshape(-1, 3)


def fit_affine(mov_xy: np.ndarray, ref_xy: np.ndarray):
    """ref = A @ mov + b (least squares). Returns A, b, residuals, rms."""
    n = len(mov_xy)
    X = np.hstack([mov_xy, np.ones((n, 1))])
    M, *_ = np.linalg.lstsq(X, ref_xy, rcond=None)
    A, b = M[:2, :].T, M[2, :]
    res = ref_xy - (mov_xy @ A.T + b)
    return A, b, res, float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))) if n else np.inf


def _nearest_pairs(ref_xy, mov_t, radius):
    """Mutual nearest neighbours closer than radius: list of (i_ref, j_mov)."""
    if len(ref_xy) == 0 or len(mov_t) == 0:
        return []
    d = np.hypot(ref_xy[:, None, 0] - mov_t[None, :, 0], ref_xy[:, None, 1] - mov_t[None, :, 1])
    i_best = np.argmin(d, axis=1)
    j_best = np.argmin(d, axis=0)
    return [(i, j) for i, j in enumerate(i_best) if j_best[j] == i and d[i, j] < radius]


def match_stars(ref: np.ndarray, mov: np.ndarray, *, top: int = 30, min_sep: float = 25.0,
                dist_tol: float = 0.015, min_pairs: int = 4, max_rms: float = 0.8) -> dict | None:
    """
    Match two star lists (x, y, flux) related by any rotation + shift (scale ~1).
    Pairs of stars with the same separation in both lists vote for the rotation angle, then for the shift; the best
    transformation gives mutual-nearest pairs, an affine fit is iterated on them with outlier rejection.
    Returns {"A", "b", "ref_idx", "mov_idx", "rms", "angle"} or None.
    """
    if len(ref) < min_pairs or len(mov) < min_pairs:
        return None
    R, M = ref[:top, :2], mov[:top, :2]
    ir, jr = np.triu_indices(len(R), 1)
    im, jm = np.triu_indices(len(M), 1)
    vr, vm = R[jr] - R[ir], M[jm] - M[im]
    dr, dm = np.hypot(*vr.T), np.hypot(*vm.T)
    ar, am = np.arctan2(vr[:, 1], vr[:, 0]), np.arctan2(vm[:, 1], vm[:, 0])
    okr, okm = dr > min_sep, dm > min_sep
    ir, jr, dr, ar = ir[okr], jr[okr], dr[okr], ar[okr]
    im, jm, dm, am = im[okm], jm[okm], dm[okm], am[okm]
    if len(dr) == 0 or len(dm) == 0:
        return None
    rel = np.abs(dr[:, None] / dm[None, :] - 1.0)
    pr, pm = np.nonzero(rel < dist_tol)
    if len(pr) == 0:
        return None
    # the pair can be oriented both ways: theta and theta + pi. Both are voted for, so the angle histogram has twin
    # peaks; every candidate angle (a few highest peaks, each also + pi) is judged by how many pairs agree on the shift.
    th = np.concatenate([ar[pr] - am[pm], ar[pr] - am[pm] + np.pi])
    th = np.mod(th + np.pi, 2 * np.pi) - np.pi
    flip_all = np.concatenate([np.zeros(len(pr), bool), np.ones(len(pr), bool)])
    pr2, pm2 = np.concatenate([pr, pr]), np.concatenate([pm, pm])
    nb = 720
    h, edges = np.histogram(th, bins=nb, range=(-np.pi, np.pi))
    hs = h + np.roll(h, 1) + np.roll(h, -1)
    cands = []
    for k in np.argsort(-hs)[:6]:
        c0 = 0.5 * (edges[k] + edges[k + 1])
        cands += [c0, np.mod(c0 + 2 * np.pi, 2 * np.pi) - np.pi]
    best = None
    for theta0 in cands:
        sel = np.abs(np.mod(th - theta0 + np.pi, 2 * np.pi) - np.pi) < 1.5 * (2 * np.pi / nb)
        if sel.sum() < 3:
            continue
        theta = float(np.median(th[sel]))
        c, s_ = np.cos(theta), np.sin(theta)
        Rm = np.array([[c, -s_], [s_, c]])
        # shift votes: start of the reference pair <- start (or end, if flipped) of the moving pair
        m_start = np.where(flip_all[sel], jm[pm2[sel]], im[pm2[sel]])
        t = R[ir[pr2[sel]]] - (M[m_start] @ Rm.T)
        t_round = np.round(t / 2.0).astype(int)
        keys, counts = np.unique(t_round, axis=0, return_counts=True)
        kb = keys[np.argmax(counts)]
        near = np.all(np.abs(t_round - kb) <= 1, axis=1)
        if best is None or near.sum() > best[0]:
            best = (int(near.sum()), Rm, np.median(t[near], axis=0))
    if best is None or best[0] < 3:
        return None
    _, A, b = best
    return refine_transform(ref, mov, A, b, min_pairs=min_pairs, max_rms=max_rms)


def refine_transform(ref: np.ndarray, mov: np.ndarray, A: np.ndarray, b: np.ndarray, *, min_pairs: int = 4,
                     max_rms: float = 0.8, radii=(6.0, 3.0, 2.0)) -> dict | None:
    """
    Improve an approximate transformation (from the rotation search or from clicked pairs) with all detected stars:
    mutual-nearest pairs within shrinking radii, affine fit with outlier rejection.
    """
    ri = mi = np.array([], dtype=int)
    rms = np.inf
    for radius in radii:
        mt = mov[:, :2] @ A.T + b
        pairs = _nearest_pairs(ref[:, :2], mt, radius)
        if len(pairs) < min_pairs:
            return None
        ri = np.array([p[0] for p in pairs])
        mi = np.array([p[1] for p in pairs])
        A, b, res, rms = fit_affine(mov[mi, :2], ref[ri, :2])
        r = np.hypot(*res.T)
        keep = r < max(3.0 * np.median(r) * 1.4826, 0.5)
        if keep.sum() >= min_pairs and keep.sum() < len(keep):
            A, b, res, rms = fit_affine(mov[mi[keep], :2], ref[ri[keep], :2])
            ri, mi = ri[keep], mi[keep]
    # a true match explains a good part of the stars; a few chance coincidences do not
    need = max(min_pairs, int(0.25 * min(len(ref), len(mov))))
    if rms > max_rms or len(ri) < need:
        return None
    return {"A": A, "b": b, "ref_idx": ri, "mov_idx": mi, "rms": rms, "angle": float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))}


def centroid_refine(img: np.ndarray, x: float, y: float, radius: int = 8) -> tuple[float, float]:
    h, w = img.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    cut = np.array(img[y0:y1, x0:x1], dtype=float)
    if cut.size == 0 or not np.isfinite(cut).any():
        return float(x), float(y)
    cut = np.where(np.isfinite(cut), cut - np.nanmedian(cut), 0.0)
    cut[cut < 0] = 0.0
    s = cut.sum()
    if s <= 0:
        return float(x), float(y)
    yy, xx = np.indices(cut.shape, dtype=float)
    return float((cut * xx).sum() / s + x0), float((cut * yy).sum() / s + y0)


def manual_pairs(ref_img: np.ndarray, mov_img: np.ndarray, title: str, radius: int = 8):
    """
    Click pairs of the same stars: LEFT (reference night) then RIGHT (this night), again and again.
    Each click is refined to the light centroid. Right click: undo the last click. Enter / Esc / closing the
    window: done (at least 3 pairs are needed). Returns (ref_xy, mov_xy).
    """
    import matplotlib.pyplot as plt
    from mangal_step4_shift import _ensure_interactive_backend, _raise_window
    if not _ensure_interactive_backend():
        raise RuntimeError("the window for manual star pairs cannot be shown (no interactive matplotlib backend)")

    def lim(img):
        v = img[np.isfinite(img)]
        if v.size == 0:
            return 0.0, 1.0
        lo, hi = np.percentile(v, 5), np.percentile(v, 99.7)
        return lo, (hi if hi > lo else lo + 1.0)

    L, Rr = [], []
    st = {"done": False}
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(14, 7))
    for ax, img in ((axl, ref_img), (axr, mov_img)):
        lo, hi = lim(img)
        ax.imshow(np.clip(img, lo, hi), origin="lower", cmap="gray")
    axl.set_title("reference night")
    axr.set_title(title)
    fig.suptitle("Pairs of the same star: LEFT, then RIGHT; right click = undo; Enter = done (>= 3 pairs)")

    def redraw():
        for a in list(axl.lines) + list(axr.lines) + list(axl.texts) + list(axr.texts):
            a.remove()
        for ax, pts, col in ((axl, L, "cyan"), (axr, Rr, "yellow")):
            for n, (x, y) in enumerate(pts, start=1):
                ax.plot(x, y, "o", ms=10, mfc="none", mec=col, mew=1.5)
                ax.text(x + 6, y + 6, str(n), color=col, fontsize=10, weight="bold")
        nxt = "LEFT" if len(L) == len(Rr) else "RIGHT"
        axl.set_xlabel(f"pairs: {len(Rr)}; next click: {nxt}")
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.button == 3:
            if len(L) > len(Rr):
                L.pop()
            elif Rr:
                Rr.pop()
            redraw()
            return
        if ev.xdata is None:
            return
        if len(L) == len(Rr) and ev.inaxes is axl:
            L.append(centroid_refine(ref_img, ev.xdata, ev.ydata, radius))
        elif len(L) > len(Rr) and ev.inaxes is axr:
            Rr.append(centroid_refine(mov_img, ev.xdata, ev.ydata, radius))
        redraw()

    def on_key(ev):
        if ev.key in ("enter", "return", "escape"):
            st["done"] = True

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    _raise_window(fig)
    print(f"[combine] WAITING: click star pairs in the window ({title}); Enter = done", flush=True)
    plt.show(block=False)
    while plt.fignum_exists(fig.number) and not st["done"]:
        plt.pause(0.1)
    plt.close(fig)
    n = min(len(L), len(Rr))
    return np.array(L[:n], dtype=float).reshape(-1, 2), np.array(Rr[:n], dtype=float).reshape(-1, 2)


def warp_to_reference(img: np.ndarray, A: np.ndarray, b: np.ndarray, shape) -> np.ndarray:
    """
    Resample an image of another night onto the reference grid (ref = A @ mov + b), bilinear, surface brightness
    conserved (/|det A|). Pixels outside its footprint or touched by NaN become NaN, so that only the other nights
    count there (no NaN or fill wedges in the combination).
    """
    h, w = shape
    yy, xx = np.indices((h, w), dtype=float)
    Ai = np.linalg.inv(A)
    p = np.stack([xx.ravel() - b[0], yy.ravel() - b[1]])
    sx, sy = (Ai @ p)
    sx, sy = sx.reshape(h, w), sy.reshape(h, w)
    valid = np.isfinite(img)
    fill = float(np.nanmedian(img[valid])) if valid.any() else 0.0
    out = map_coordinates(np.where(valid, img, fill), [sy, sx], order=1, mode="constant", cval=np.nan)
    vm = map_coordinates(valid.astype(float), [sy, sx], order=1, mode="constant", cval=0.0)
    out[(vm < 0.999) | ~np.isfinite(out)] = np.nan
    return (out / abs(np.linalg.det(A))).astype(np.float32)


def star_flux_scale(ref_img: np.ndarray, img: np.ndarray, xy: np.ndarray, win: int = 10) -> dict:
    """
    Median ratio ref / night of the Gaussian-fit fluxes of the stars xy (reference grid), on registered images.
    Returns {"k", "n", "scatter"} (k None with fewer than 3 stars).
    """
    from mangal_step7_net_astro import _ss_fit_star
    ratios = []
    for x, y in xy:
        a = _ss_fit_star(ref_img, x, y, win)
        b = _ss_fit_star(img, x, y, win)
        if a is None or b is None or a["flux"] <= 0 or b["flux"] <= 0:
            continue
        ratios.append(a["flux"] / b["flux"])
    r = np.array(ratios)
    if r.size < 3:
        return {"k": None, "n": int(r.size), "scatter": None}
    k = float(np.median(r))
    mad = float(1.4826 * np.median(np.abs(r - k)))
    good = np.abs(r - k) <= max(3 * mad, 1e-6 * k)
    k = float(np.median(r[good]))
    return {"k": k, "n": int(good.sum()), "scatter": float(np.std(r[good]) / k) if good.sum() > 1 else 0.0}


# ----------------------------------------------------------------------------------------------------------
# the combination
# ----------------------------------------------------------------------------------------------------------
def choose_scale(mode: str, k_star: float | None, k_std: float | None) -> tuple[float, str]:
    """Scale factor of a night and where it comes from."""
    mode = (mode or "both").lower()
    if mode == "standards":
        if k_std is not None:
            return k_std, "standards"
        if k_star is not None:
            return k_star, "stars (no QE calibration of this night)"
    else:
        if k_star is not None:
            return k_star, "stars"
        if k_std is not None:
            return k_std, "standards (not enough stars)"
    return 1.0, "none (no stars, no QE calibration): 1.0"


def combined_qe_calibration(per_night) -> dict | None:
    """
    Calibration of the combined cube from the standards of ALL nights. The cube is in counts/sec of night 0; the data
    of night n were multiplied by its scale k_n, so its calibration in these units is coefficient_n / k_n. These are
    averaged with the line exposure of the nights as weights (nights without a QE calibration are left out).
    """
    items = []
    for nt, layers, _reg, rep in per_night:
        if not nt.get("qe_json"):
            continue
        try:
            d = json.loads(Path(nt["qe_json"]).read_text(encoding="utf-8"))
        except Exception:
            continue
        w = sum(float(L.get("texp") or 0) for L in layers if L.get("kind") == "line") or float(rep["hdr"].get("EXPTIME") or 1.0)
        items.append((rep, d, w))
    if not items:
        print("[combine] no QE calibration of any night: QE_combined_*.json is not written")
        return None
    out = {"filter": items[0][1].get("filter"), "nights": [],
           "note": "average of coefficient(night)/scale(night), exposure-weighted; units: counts/sec of night 0"}
    for key in ("line_mul_adu_to_erg", "cont_mul_adu_to_erg", "median_mul_adu_to_erg"):
        vals = [(float(d[key]) / float(rep["k"]), w) for rep, d, w in items if d.get(key) is not None]
        if vals:
            out[key] = float(sum(v * w for v, w in vals) / sum(w for _, w in vals))
    if "line_mul_adu_to_erg" not in out and "median_mul_adu_to_erg" in out:
        out["line_mul_adu_to_erg"] = out["median_mul_adu_to_erg"]
    for rep, d, w in items:
        c = d.get("line_mul_adu_to_erg", d.get("median_mul_adu_to_erg"))
        out["nights"].append({"night": rep["night"], "label": rep["label"], "qe_json": rep["qe_json"],
                              "coefficient": c, "scale": rep["k"], "coefficient_night0_units": float(c) / float(rep["k"]),
                              "weight_s": w})
    msg = "[combine] averaged QE calibration %.4e" % out["line_mul_adu_to_erg"]
    for e in out["nights"]:
        msg += f"; night {e['night']} ({e['label']}): {e['coefficient_night0_units']:.4e}"
    if len(out["nights"]) > 1:
        v = [e["coefficient_night0_units"] for e in out["nights"]]
        msg += f"; spread {100 * (max(v) - min(v)) / np.mean(v):.1f}%"
    print(msg)
    return out


def combine_layer_stacks(stacks: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Pixel-wise weighted mean of the nights' images (NaN pixels of a night are left out)."""
    acc = np.zeros(stacks[0].shape, dtype=float)
    ww = np.zeros(stacks[0].shape, dtype=float)
    for img, w in zip(stacks, weights):
        ok = np.isfinite(img)
        acc[ok] += w * img[ok]
        ww[ok] += w
    out = np.full(stacks[0].shape, np.nan, dtype=np.float32)
    good = ww > 0
    out[good] = (acc[good] / ww[good]).astype(np.float32)
    return out


def combine_nights(ref_w_dir, other_w_dirs, out_dir, *, object_name: str | None = None, scale: str = "both",
                   manual: bool = False, ref_linecont: Path | None = None) -> Path:
    """
    Combine the <obj>_linecont.fits cubes of several nights (reference first) into out_dir/<obj>_linecont.fits.
    Returns the path of the combined cube.
    """
    ref = night_info(ref_w_dir, object_name, ref_linecont)
    nights = [ref] + [night_info(d, object_name) for d in other_w_dirs]
    with fits.open(ref["linecont"]) as h:
        ref_cube = np.asarray(h[0].data, dtype=np.float32)
        ref_hdr = h[0].header.copy()
    ref_layers = read_layer_table(ref_hdr, ref_cube.shape[0])
    shape = ref_cube.shape[1:]
    ref_sum = sum_image(ref_cube, ref_layers)
    ref_stars = detect_stars(ref_sum)
    print(f"[combine] reference night: {ref['linecont']} ({len(ref_stars)} stars; "
          f"QE: {ref['qe_json'].name if ref['qe_json'] else 'none'})")

    per_night = []  # (info, layers, registered cube, scale, report)
    for n, nt in enumerate(nights):
        with fits.open(nt["linecont"]) as h:
            cube = np.asarray(h[0].data, dtype=np.float32)
            hdr = h[0].header.copy()
        layers = read_layer_table(hdr, cube.shape[0])
        label = " ".join(x for x in ((nt["r_dir"].name if nt["r_dir"] is not None else nt["w_dir"].name),
                                     (nt["date"] or "")[:10]) if x)
        rep = {"night": n, "label": label, "w_dir": str(nt["w_dir"]), "linecont": nt["linecont"].name, "date": nt["date"],
               "qe_json": nt["qe_json"].name if nt["qe_json"] else "", "hdr": hdr}
        if n == 0:
            reg = cube
            rep.update({"method": "reference", "rms": 0.0, "npairs": 0, "angle": 0.0, "k": 1.0, "k_star": 1.0,
                        "k_std": 1.0, "k_source": "reference", "A": np.eye(2), "b": np.zeros(2)})
        else:
            img = sum_image(cube, layers)
            stars = detect_stars(img)
            m = None if manual else match_stars(ref_stars, stars)
            if m is not None:
                A, b, rms, npairs, method = m["A"], m["b"], m["rms"], len(m["ref_idx"]), "auto"
            else:
                if not manual:
                    print(f"[combine] night {n}: automatic star match failed ({len(stars)} stars found): manual pairs")
                rxy, mxy = manual_pairs(ref_sum, img, f"night {n}: {nt['linecont'].name}")
                if len(rxy) < 3:
                    raise RuntimeError(f"night {n}: at least 3 star pairs are needed, got {len(rxy)}")
                A, b, _, rms = fit_affine(mxy, rxy)
                npairs, method = len(rxy), "manual"
                ref_m = refine_transform(ref_stars, stars, A, b)  # clicks = first guess, all stars = precision
                if ref_m is not None:
                    A, b, rms, npairs, method = ref_m["A"], ref_m["b"], ref_m["rms"], len(ref_m["ref_idx"]), "manual + refined"
            angle = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
            print(f"[combine] night {n}: {method} registration, {npairs} stars, rms {rms:.2f} px, rotation {angle:+.2f} deg, "
                  f"shift ({b[0]:+.1f}, {b[1]:+.1f}) px")
            reg = np.stack([warp_to_reference(cube[k], A, b, shape) for k in range(cube.shape[0])])
            ss = star_flux_scale(ref_sum, sum_image(reg, layers), ref_stars[:, :2])
            k_std = (nt["qe_coef"] / ref["qe_coef"]) if (nt["qe_coef"] and ref["qe_coef"]) else None
            k, src = choose_scale(scale, ss["k"], k_std)
            msg = f"[combine] night {n}: scale {k:.4f} from {src}"
            if ss["k"] is not None:
                msg += f"; stars: {ss['k']:.4f} ({ss['n']} stars, scatter {100 * ss['scatter']:.1f}%)"
            if k_std is not None:
                msg += f"; standards: {k_std:.4f}"
            if ss["k"] is not None and k_std is not None:
                msg += f"; stars / standards = {ss['k'] / k_std:.3f} (transparency difference {100 * (ss['k'] / k_std - 1):+.1f}%)"
            print(msg)
            reg = reg * np.float32(k)
            rep.update({"method": method, "rms": rms, "npairs": npairs, "angle": angle, "k": k, "k_star": ss["k"],
                        "k_std": k_std, "k_source": src, "A": A, "b": b})
        zc = _zspace_counts(nt["linecont"])
        for L in layers:
            L["lam_night"] = layer_lambda(L, nt["calib"], nt["filter"], zc)
        per_night.append((nt, layers, reg, rep))

    # layers by name, in the order of the reference night, then new names
    names = []
    for _, layers, _, _ in per_night:
        for L in layers:
            if L["name"] not in names:
                names.append(L["name"])
    out_imgs, out_layers = [], []
    for name in names:
        imgs, ws, texps, zms, nfr, zsp, lams, lw, kind, role, tag = [], [], [], [], 0, [], [], [], None, None, ""
        for nt, layers, reg, rep in per_night:
            for k, L in enumerate(layers):
                if L["name"] != name:
                    continue
                t = float(L.get("texp") or 0.0)
                imgs.append(reg[k])
                ws.append(t if t > 0 else 1.0)
                texps.append(t)
                zms.append(L.get("zmean"))
                nfr += int(L.get("nframes") or 0)
                zsp.append(str(L.get("zspace", "")))
                if L.get("lam_night") is not None:
                    lams.append(L["lam_night"])
                    lw.append(t if t > 0 else 1.0)
                kind, role, tag = L["kind"], L.get("role", "group"), L.get("tag", "") or tag
        out_imgs.append(combine_layer_stacks(imgs, ws))
        zm_ok = [(z, t) for z, t in zip(zms, texps) if z is not None and t > 0]
        out_layers.append({
            "name": name, "kind": kind, "role": role, "tag": tag,
            "zspace": "/".join(zsp), "texp": float(sum(texps)), "nframes": nfr,
            "zmean": combine_zmean([z for z, _ in zm_ok], [t for _, t in zm_ok]) if zm_ok else None,
            "lam": float(np.average(lams, weights=lw)) if lams else None,
        })
        print(f"[combine] layer {name}: {len(imgs)} night(s), exposure {sum(texps):.0f} s"
              + (f", mean wavelength {out_layers[-1]['lam']:.2f} A" if out_layers[-1]["lam"] else ""))

    qe_comb = combined_qe_calibration(per_night)

    # header: reference night + totals over the nights
    hdr = ref_hdr.copy()
    hdrs = [rep["hdr"] for _, _, _, rep in per_night]
    def _sum(key):
        v = [float(h.get(key)) for h in hdrs if h.get(key) is not None]
        return float(sum(v)) if v else None
    for key, com in (("EXPTIME", "Total exposure time [s] of all frames, all nights"),
                     ("TLINE", "[s] Sum of EXPTIME for used line slices, all nights"),
                     ("TCONT", "[s] Sum of EXPTIME for used cont slices, all nights")):
        v = _sum(key)
        if v is not None:
            hdr[key] = (v, com)
    nf = _sum("NFRAMES")
    if nf is not None:
        hdr["NFRAMES"] = (int(nf), "Number of frames summed, all nights")
    zz = [(float(h["ZMEAN"]), float(h.get("EXPTIME") or 0)) for h in hdrs if h.get("ZMEAN") is not None]
    if zz:
        zm = combine_zmean([z for z, _ in zz], [t for _, t in zz])
        if zm is not None:
            hdr["ZMEAN"] = (round(zm, 3), "Mean zenith distance [deg], all nights")
    if "ZSPACE" in hdr:
        del hdr["ZSPACE"]
    hdr["NNIGHTS"] = (len(per_night), "number of nights combined")
    for nt, _, _, rep in per_night:
        n = rep["night"]
        hdr[f"NIGHT{n}"] = (rep["label"][:60], "night " + ("(reference)" if n == 0 else "combined with night 0"))
        hdr[f"NSCALE{n}"] = (round(float(rep["k"]), 6), f"flux scale to night 0 ({rep['k_source'][:30]})")
        hdr["HISTORY"] = f"COMBINE night {n}: {nt['linecont'].name} in {nt['w_dir']}"
        if n:
            A, b = rep["A"], rep["b"]
            hdr["HISTORY"] = (f"COMBINE night {n}: {rep['method']} affine A=[[{A[0,0]:.6f},{A[0,1]:.6f}],"
                              f"[{A[1,0]:.6f},{A[1,1]:.6f}]] b=[{b[0]:.3f},{b[1]:.3f}] rms={rep['rms']:.3f}px")
            hdr["HISTORY"] = (f"COMBINE night {n}: scale {rep['k']:.5f} ({rep['k_source']}); stars "
                              f"{rep['k_star'] if rep['k_star'] is None else round(rep['k_star'], 5)}, standards "
                              f"{rep['k_std'] if rep['k_std'] is None else round(rep['k_std'], 5)}")
    hdr["HISTORY"] = "COMBINE: layers of the same name averaged with exposure weights (counts/sec of night 0)"
    if qe_comb is not None:
        hdr["QECOMB"] = (float(f"{qe_comb['line_mul_adu_to_erg']:.6e}"), "averaged QE calibration [erg/cm2/ADU], night 0 units")
        hdr["HISTORY"] = f"COMBINE: averaged QE calibration of {len(qe_comb['nights'])} night(s) -> QE_combined_*.json"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ref["linecont"].name
    stamp(hdr, out_path)
    write_layer_table(hdr, out_layers)
    fits.writeto(out_path, np.stack(out_imgs).astype(np.float32), hdr, overwrite=True)

    # star list of the reference night (same grid) for the continuum scaling of the net stage
    bn = base_name(ref["linecont"])
    for cand in (ref["w_dir"] / f"{bn}_shift.stars.xy", ref["w_dir"] / f"{bn}_shift.fits.stars.xy"):
        if cand.exists():
            shutil.copy2(cand, out_dir / f"{bn}_shift.stars.xy")
            break
    if qe_comb is not None:
        filt = qe_comb.get("filter") or "filter"
        qp = out_dir / f"QE_combined_{filt}.json"
        qp.write_text(json.dumps(qe_comb, indent=2), encoding="utf-8")
        print(f"[OK] averaged QE calibration: {qp}")
    # report
    rows = [{k: v for k, v in rep.items() if k not in ("hdr", "A", "b")} for _, _, _, rep in per_night]
    pd.DataFrame(rows).to_csv(side_path(out_path, ".nights.csv"), index=False)
    print(f"[OK] combined cube of {len(per_night)} nights: {out_path}")
    return out_path
