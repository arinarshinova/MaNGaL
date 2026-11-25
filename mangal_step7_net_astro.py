from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import tempfile
import time
import math
import io
import csv
import re
from mangal_header import stamp, combine_zmean
from mangal_names import (base_name, product_path, strip_kind, split_product_stem, legacy_line_path, line_tag_from_stem,
                          read_layer_table, plan_layers)
def _add_history(hdr_like, msg: str) -> None:
    """Add HISTORY entry to fits.Header or dict (stored as list under 'HISTORY')."""
    try:
        if hasattr(hdr_like, "add_history"):
            hdr_like.add_history(msg)
            return
    except Exception:
        pass
    try:
        # dict case
        if isinstance(hdr_like, dict):
            hdr_like.setdefault("HISTORY", [])
            if isinstance(hdr_like["HISTORY"], list):
                hdr_like["HISTORY"].append(msg)
            else:
                hdr_like["HISTORY"] = [str(hdr_like["HISTORY"]), msg]
    except Exception:
        pass

def _is_fits_bytes(b: bytes) -> bool:
    head = (b or b"")[:80].lstrip()
    return head.startswith(b"SIMPLE") or head.startswith(b"XTENSION")

def _download_new_fits(base_url: str, job_id: int, out_path: Path,
                       timeout: int = 300, max_retries: int = 5) -> Path:
    """Download nova new_fits_file/JOBID and validate it is FITS; otherwise save HTML and raise."""
    import requests
    url = f"{base_url}/new_fits_file/{job_id}"
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            content = r.content or b""
            ctype = (r.headers.get("Content-Type") or "").lower()
            if (not _is_fits_bytes(content)) and ("fits" not in ctype):
                bad = out_path.with_suffix(".html")
                bad.write_bytes(content)
                raise RuntimeError(
                    f"new_fits_file/{job_id} is not FITS (Content-Type={ctype!r}). Saved response to {bad}."
                )
            out_path.write_bytes(content)
            return out_path
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Failed to download new_fits_file/{job_id}: {last_err}")

def _pa_idl_deg_from_header(h: fits.Header) -> float:
    """Estimate PA of +Y axis (image column axis) before rotation, like IDL n_pa(h).

    Returns PA in degrees measured from North through East.
    The angle is computed from the +Y-axis WCS terms directly, without an extra cos(dec)
    correction, so that cub_rot(..., -PA) fully aligns North upward.
    """
    # Prefer CD matrix
    cd12 = h.get("CD1_2")
    cd22 = h.get("CD2_2")
    if cd12 is None or cd22 is None:
        # Try PC*CDELT
        if h.get("PC1_2") is not None and h.get("PC2_2") is not None and h.get("CDELT1") is not None and h.get("CDELT2") is not None:
            cd12 = float(h.get("PC1_2", 0.0)) * float(h.get("CDELT1"))
            cd22 = float(h.get("PC2_2", 1.0)) * float(h.get("CDELT2"))
        else:
            return float("nan")

    cd12 = float(cd12)
    cd22 = float(cd22)

    # For a celestial FITS WCS with a CD/PC matrix, the position angle of the +Y axis
    # should be derived directly from the matrix terms of that axis.
    #
    # Using an extra cos(dec) factor here systematically under-rotates frames at high |dec|,
    # because the CD matrix already encodes the local linear transformation in the sky tangent plane.
    # This was the reason NE was not fully aligned for northern targets like J0811+53.
    dra = cd12
    ddec = cd22

    pa = math.degrees(math.atan2(dra, ddec))
    return pa

def _rotate_wcs_cd_crpix_like_idl(h: fits.Header, angle_deg: float, shape: tuple[int,int]) -> fits.Header:
    """Rotate WCS (CD/PC and CRPIX) for an image rotated by angle_deg around image center (reshape=False)."""
    ny, nx = shape
    if not np.isfinite(angle_deg):
        angle_deg = 0.0
    th = math.radians(angle_deg)
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th),  math.cos(th)]], dtype=float)
    # FITS CRPIX is 1-based, convert to 0-based for rotation math
    crpix1 = float(h.get("CRPIX1", (nx+1)/2.0))
    crpix2 = float(h.get("CRPIX2", (ny+1)/2.0))
    old = np.array([crpix1-1.0, crpix2-1.0], dtype=float)
    center = np.array([(nx-1)/2.0, (ny-1)/2.0], dtype=float)
    new = center + (R @ (old - center))
    h2 = fits.Header(h)

    h2["CRPIX1"] = new[0] + 1.0
    h2["CRPIX2"] = new[1] + 1.0

    # Rotate CD or PC; for pixel rotation, cd_new = cd @ R.T (because x_old = R.T x_new)
    if h.get("CD1_1") is not None:
        cd = np.array([[float(h["CD1_1"]), float(h.get("CD1_2", 0.0))],
                       [float(h.get("CD2_1", 0.0)), float(h["CD2_2"])]], dtype=float)
        cd2 = cd @ R.T
        h2["CD1_1"] = cd2[0,0]; h2["CD1_2"] = cd2[0,1]
        # Keep IDL-like tiny CD1_2; if numerical noise is extremely small, keep as-is
        # (do not force to zero unless you explicitly want a perfectly diagonal CD matrix)
        h2["CD2_1"] = cd2[1,0]; h2["CD2_2"] = cd2[1,1]
        # IDL CUB_ROT effectively removes tiny residual shear; clamp near-zero CD2_1
        if abs(float(h2.get("CD2_1", 0.0))) < 5e-7:
            h2["CD2_1"] = 0.0
    elif h.get("PC1_1") is not None and h.get("CDELT1") is not None:
        pc = np.array([[float(h.get("PC1_1", 1.0)), float(h.get("PC1_2", 0.0))],
                       [float(h.get("PC2_1", 0.0)), float(h.get("PC2_2", 1.0))]], dtype=float)
        pc2 = pc @ R.T
        h2["PC1_1"] = pc2[0,0]; h2["PC1_2"] = pc2[0,1]
        h2["PC2_1"] = pc2[1,0]; h2["PC2_2"] = pc2[1,1]
    return h2

def _append_wcs_sip_like_idl(dst_hdr: fits.Header, ref_hdr: fits.Header) -> fits.Header:
    """IDL-like: keep existing header, overwrite/append only WCS + SIP keywords from ref_hdr."""
    dst = fits.Header(dst_hdr)
    keep_prefixes = ("CTYPE","CRVAL","CRPIX","CD","PC","CDELT","CUNIT","WCS","LONPOLE","LATPOLE","EQUINOX","RADESYS")
    sip_prefixes = ("A_","B_","AP_","BP_")
    sip_keys = {"A_ORDER","B_ORDER","AP_ORDER","BP_ORDER"}
    for k in ref_hdr.keys():
        if k in ("SIMPLE","BITPIX","NAXIS","EXTEND"):
            continue
        if k in sip_keys or any(k.startswith(p) for p in sip_prefixes) or any(k.startswith(p) for p in keep_prefixes):
            try:
                dst[k] = ref_hdr[k]
            except Exception:
                pass
    return dst

def _append_sip_only_like_idl(dst_hdr: fits.Header, ref_hdr: fits.Header) -> fits.Header:
    """Append only SIP distortion keywords (A_/B_/AP_/BP_ and *_ORDER) from ref_hdr into dst_hdr.
    IMPORTANT: does NOT overwrite CD/CRPIX/CRVAL etc, so rotated WCS stays intact (IDL cub_rot behavior).
    """
    dst = fits.Header(dst_hdr)
    sip_prefixes = ("A_","B_","AP_","BP_")
    sip_keys = {"A_ORDER","B_ORDER","AP_ORDER","BP_ORDER"}
    for k in ref_hdr.keys():
        if k in ("SIMPLE","BITPIX","NAXIS","EXTEND","HISTORY","COMMENT","CONTINUE"):
            continue
        if k in sip_keys or any(k.startswith(p) for p in sip_prefixes):
            try:
                dst[k] = ref_hdr[k]
            except Exception:
                pass
    return dst


import numpy as np

try:
    from astropy.io import fits
except Exception as e:
    raise ImportError("astropy is required for step7 (net + astrometry)") from e


try:
    from scipy.ndimage import rotate as _nd_rotate, gaussian_filter as _nd_gaussian_filter, median_filter as _nd_median_filter
    from scipy.special import erf as _sp_erf
except Exception as e:
    raise ImportError("scipy is required for NE rotation in step7") from e

@dataclass
class AstroHints:
    ra_deg: float | None = None
    dec_deg: float | None = None
    radius_deg: float | None = None
    scale_arcsec_per_pix: float | None = None
    scale_lower: float | None = None
    scale_upper: float | None = None


def _sanitize_header_for_2d_write(hdr: fits.Header, data_shape: tuple[int, int] | None = None) -> fits.Header:
    """Return a FITS header safe for writing a 2D image.

    Removes stale higher-dimensional axis cards (e.g. NAXIS3, CTYPE3, CRPIX3,
    CD3_*, PC3_*, SIP/WCS bookkeeping for extra axes) that may survive after
    squeezing a cube-like input down to a 2D image.
    """
    out = fits.Header()
    remove_exact = {
        'NAXIS3','NAXIS4','NAXIS5','NAXIS6','NAXIS7','NAXIS8','NAXIS9',
        'CTYPE3','CTYPE4','CTYPE5','CTYPE6','CTYPE7','CTYPE8','CTYPE9',
        'CUNIT3','CUNIT4','CUNIT5','CUNIT6','CUNIT7','CUNIT8','CUNIT9',
        'CRVAL3','CRVAL4','CRVAL5','CRVAL6','CRVAL7','CRVAL8','CRVAL9',
        'CRPIX3','CRPIX4','CRPIX5','CRPIX6','CRPIX7','CRPIX8','CRPIX9',
        'CDELT3','CDELT4','CDELT5','CDELT6','CDELT7','CDELT8','CDELT9',
        'CROTA3','CROTA4','CROTA5','CROTA6','CROTA7','CROTA8','CROTA9',
    }

    def _drop_key(k: str) -> bool:
        if k in remove_exact:
            return True
        if k.startswith(('CD3_','CD4_','CD5_','CD6_','CD7_','CD8_','CD9_')):
            return True
        if k.startswith(('CD1_3','CD1_4','CD1_5','CD1_6','CD1_7','CD1_8','CD1_9',
                         'CD2_3','CD2_4','CD2_5','CD2_6','CD2_7','CD2_8','CD2_9',
                         'CD3_1','CD4_1','CD5_1','CD6_1','CD7_1','CD8_1','CD9_1',
                         'CD3_2','CD4_2','CD5_2','CD6_2','CD7_2','CD8_2','CD9_2')):
            return True
        if k.startswith(('PC3_','PC4_','PC5_','PC6_','PC7_','PC8_','PC9_')):
            return True
        if k.startswith(('PC1_3','PC1_4','PC1_5','PC1_6','PC1_7','PC1_8','PC1_9',
                         'PC2_3','PC2_4','PC2_5','PC2_6','PC2_7','PC2_8','PC2_9',
                         'PC3_1','PC4_1','PC5_1','PC6_1','PC7_1','PC8_1','PC9_1',
                         'PC3_2','PC4_2','PC5_2','PC6_2','PC7_2','PC8_2','PC9_2')):
            return True
        return False

    for card in hdr.cards:
        k = card.keyword
        if k in ('NAXIS', 'NAXIS1', 'NAXIS2'):
            continue
        if _drop_key(k):
            continue
        try:
            out.append(card, end=True)
        except Exception:
            pass

    out['NAXIS'] = 2
    if data_shape is not None and len(data_shape) == 2:
        ny, nx = map(int, data_shape)
        out['NAXIS1'] = nx
        out['NAXIS2'] = ny
    else:
        if 'NAXIS1' in hdr:
            out['NAXIS1'] = int(hdr['NAXIS1'])
        if 'NAXIS2' in hdr:
            out['NAXIS2'] = int(hdr['NAXIS2'])
    return out

def _precess_to_j2000(ra_deg: float, dec_deg: float, epoch: float) -> tuple[float, float]:
    """Mean coordinates of the equinox `epoch` (Julian year) -> J2000 (IAU 1976 precession)."""
    t = (float(epoch) - 2000.0) / 100.0
    if abs(t) < 1e-6:
        return float(ra_deg), float(dec_deg)
    asec = np.pi / 180.0 / 3600.0
    zeta = (2306.2181 * t + 0.30188 * t * t + 0.017998 * t ** 3) * asec
    z = (2306.2181 * t + 1.09468 * t * t + 0.018203 * t ** 3) * asec
    th = (2004.3109 * t - 0.42665 * t * t - 0.041833 * t ** 3) * asec

    def rz(a):
        c, s_ = np.cos(a), np.sin(a)
        return np.array([[c, s_, 0], [-s_, c, 0], [0, 0, 1]])

    def ry(a):
        c, s_ = np.cos(a), np.sin(a)
        return np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]])

    P = rz(-z) @ ry(th) @ rz(-zeta)              # J2000 -> epoch
    ra, de = np.radians(ra_deg), np.radians(dec_deg)
    v = np.array([np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)])
    w = P.T @ v                                   # epoch -> J2000
    return float(np.degrees(np.arctan2(w[1], w[0])) % 360.0), float(np.degrees(np.arcsin(np.clip(w[2], -1, 1))))


def _sexa_to_deg(text, hours: bool) -> float | None:
    try:
        t = str(text).strip().replace(" ", ":")
        if ":" not in t:
            v = float(t)
            return v      # plain number: already degrees
        sgn = -1.0 if t.startswith("-") else 1.0
        parts = [abs(float(x)) for x in t.lstrip("+-").split(":")]
        v = parts[0] + (parts[1] if len(parts) > 1 else 0) / 60.0 + (parts[2] if len(parts) > 2 else 0) / 3600.0
        return sgn * v * (15.0 if hours else 1.0)
    except Exception:
        return None


def _make_upload_image(cont_fits: Path) -> Path:
    """
    Copy of the image for astrometry.net: NaN (outside the circular mask, uncovered borders) -> sky level, cropped
    to the bounding box of the valid data, no WCS / instrument keywords.  The source extractor of the service is
    confused by NaN and by the sharp edge of the mask.  Returns the path; its pixel (0,0) = (x0,y0) of the original
    (written as keywords UPX0/UPY0, so the WCS can be shifted back).
    """
    data = np.asarray(fits.getdata(cont_fits), dtype=np.float32)
    data = np.squeeze(data)
    ok = np.isfinite(data) & (data != 0)
    if not ok.any():
        raise RuntimeError(f"{cont_fits.name}: no valid pixels")
    ys, xs = np.nonzero(ok)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = data[y0:y1, x0:x1].copy()
    okk = ok[y0:y1, x0:x1]
    # erode the valid area a little: the mask edge itself carries ringing / vignetting
    from scipy.ndimage import binary_erosion
    okk = binary_erosion(okk, iterations=4)
    sky = float(np.median(sub[okk]))
    sub[~okk] = sky
    h = fits.Header()
    h["UPX0"] = (int(x0), "x offset of this cut in the original image")
    h["UPY0"] = (int(y0), "y offset of this cut in the original image")
    # temporary file (deleted after the solve): nothing extra is left in W_DIR
    outp = Path(tempfile.mkdtemp(prefix="mangal_astro_")) / (cont_fits.stem + "_astro_upload.fits")
    fits.PrimaryHDU(_to_int16_for_upload(sub), header=h).writeto(outp, overwrite=True)
    return outp


def _to_int16_for_upload(img: np.ndarray) -> np.ndarray:
    """16-bit copy for the upload (half the size of float32; plenty for finding stars): sky .. bright stars -> 0 .. 32000."""
    lo, hi = np.percentile(img, [0.5, 99.995])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return img.astype(np.float32)
    return (np.clip((img - lo) / (hi - lo), 0.0, 1.0) * 32000.0).astype(np.int16)


def _hints_from_header(hdr) -> AstroHints:
    """
    RA/DEC + pixel scale hints from the header.
    RA/DEC in the header are the telescope coordinates of the date (EPOCH = 2025.9 ...), not J2000: 26 years of
    precession move them by ~0.35 deg, far outside the search radius -> astrometry.net fails.  They are precessed
    to J2000 here, and the search radius is generous (pointing errors, offsets from the object).
    """
    hints = AstroHints()
    ra = _sexa_to_deg(hdr.get("RA"), hours=True) if hdr.get("RA") is not None else None
    dec = _sexa_to_deg(hdr.get("DEC"), hours=False) if hdr.get("DEC") is not None else None
    if ra is not None and dec is not None:
        ep = None
        for key in ("EPOCH", "EQUINOX"):
            try:
                v = float(hdr.get(key))
                if 1900.0 < v < 2100.0:
                    ep = v
                    break
            except (TypeError, ValueError):
                pass
        if ep is not None and abs(ep - 2000.0) > 0.01:
            ra, dec = _precess_to_j2000(ra, dec, ep)
        hints.ra_deg, hints.dec_deg = ra, dec
    try:
        ps = hdr.get("PIXSCALE", None)
        if ps is not None:
            hints.scale_arcsec_per_pix = float(ps)
    except Exception:
        pass
    if hints.scale_arcsec_per_pix:
        hints.scale_lower = 0.85 * hints.scale_arcsec_per_pix
        hints.scale_upper = 1.15 * hints.scale_arcsec_per_pix
    hints.radius_deg = 0.5
    return hints



def _read_stars_xy(stars_xy_path: Path) -> np.ndarray:
    """Read whitespace-separated star positions file with columns x y (like *.stars.xy)."""
    stars_xy_path = Path(stars_xy_path)
    arr = []
    for line in stars_xy_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0]); y = float(parts[1])
        except Exception:
            continue
        arr.append((x, y))
    if not arr:
        raise ValueError(f"No stars read from {stars_xy_path}")
    return np.array(arr, dtype=float)


def _extract_stamp(img: np.ndarray, x: float, y: float, win: int) -> tuple[np.ndarray, float, float]:
    """Extract (2*win+1)^2 stamp around (x,y), return stamp and local coordinates of center."""
    ny, nx = img.shape
    xi = int(round(x)); yi = int(round(y))
    x0 = max(0, xi - win); x1 = min(nx, xi + win + 1)
    y0 = max(0, yi - win); y1 = min(ny, yi + win + 1)
    stamp = img[y0:y1, x0:x1].astype(np.float32, copy=False)
    return stamp, float(x - x0), float(y - y0)


def _moffat_model(params: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    sky, amp, ax, ay, x0, y0, th, beta = params
    ct = np.cos(th); st = np.sin(th)
    dx = xx - x0
    dy = yy - y0
    xp = dx * ct + dy * st
    yp = -dx * st + dy * ct
    r2 = (xp / ax) ** 2 + (yp / ay) ** 2
    return sky + amp * (1.0 + r2) ** (-beta)


def _fit_star_moffat(stamp: np.ndarray, max_iter: int = 200) -> tuple[bool, dict]:
    """Lightweight copy of step5 Moffat fit, for 2D images."""
    from scipy.optimize import least_squares

    h, w = stamp.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    border = np.hstack([stamp[0, :], stamp[-1, :], stamp[:, 0], stamp[:, -1]])
    sky0 = float(np.median(border))
    data = stamp - sky0
    if not np.isfinite(data).any():
        return False, {}

    mp = np.nanargmax(data)
    y0p, x0p = np.unravel_index(int(mp), data.shape)
    amp0 = float(np.clip(data[y0p, x0p], 1.0, None))

    pos = np.clip(data, 0, None)
    s = float(np.nansum(pos))
    if s <= 0:
        return False, {}
    xbar = float(np.nansum(xx * pos) / s)
    ybar = float(np.nansum(yy * pos) / s)
    vx = float(np.nansum(((xx - xbar) ** 2) * pos) / s)
    vy = float(np.nansum(((yy - ybar) ** 2) * pos) / s)
    ax0 = float(np.clip(np.sqrt(vx), 0.8, 8.0))
    ay0 = float(np.clip(np.sqrt(vy), 0.8, 8.0))

    beta0 = 2.5
    th0 = 0.0
    p0 = np.array([sky0, amp0, ax0, ay0, float(x0p), float(y0p), th0, beta0], dtype=np.float64)

    sigb = float(np.std(border)) if np.isfinite(border).any() else 1.0
    lb = np.array([sky0 - 5*sigb, 0.0, 0.3, 0.3, 0.0, 0.0, -np.pi, 1.05], dtype=np.float64)
    ub = np.array([sky0 + 5*sigb + float(np.nanmax(stamp)), float(np.nanmax(stamp)) * 5,
                   30.0, 30.0, float(w - 1), float(h - 1), np.pi, 10.0], dtype=np.float64)

    def resid(p: np.ndarray) -> np.ndarray:
        mod = _moffat_model(p, xx, yy)
        r = (mod - stamp).ravel()
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
    if not (0.0 <= x0 <= w - 1 and 0.0 <= y0 <= h - 1):
        return False, {}

    mulw = 2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0)
    fwhm_x = float(ax * mulw)
    fwhm_y = float(ay * mulw)
    if fwhm_x <= 0 or fwhm_y <= 0 or beta <= 1.05:
        return False, {}
    flux = float(amp * ax * ay * np.pi / (beta - 1.0))
    return True, {"flux": flux, "fwhm": float(np.hypot(fwhm_x, fwhm_y))}


def _robomean(x: np.ndarray, nsig: float = 3.0, niter: int = 1) -> float:
    """IDL robomean-like: sigma-clip around median and take mean."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    for _ in range(int(max(niter, 0))):
        med = float(np.nanmedian(x))
        sig = float(np.nanstd(x))
        if not np.isfinite(sig) or sig <= 0:
            break
        m = np.abs(x - med) <= nsig * sig
        if m.sum() == 0:
            break
        x = x[m]
    return float(np.nanmean(x)) if x.size else float("nan")


def _integrated_gaussian_1d(coords: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-3)
    a = (coords - 0.5 - x0) / (np.sqrt(2.0) * sigma)
    b = (coords + 0.5 - x0) / (np.sqrt(2.0) * sigma)
    return np.sqrt(np.pi / 2.0) * sigma * (_sp_erf(b) - _sp_erf(a))


def _integrated_gaussian_2d(xx: np.ndarray, yy: np.ndarray, amp: float, x0: float, y0: float, sx: float, sy: float) -> np.ndarray:
    gx = _integrated_gaussian_1d(xx[0, :].astype(float), x0, sx)[None, :]
    gy = _integrated_gaussian_1d(yy[:, 0].astype(float), y0, sy)[:, None]
    return amp * gy * gx


def _fit_star_gauss2d_like_idl(stamp: np.ndarray, win: int) -> tuple[bool, dict]:
    """Closer replica of IDL star_2d/gauss_2d.

    - local sky from median of darkest 4*win pixels
    - start center from max(median(image,3))
    - fit a pixel-integrated Gaussian
    - fit only a compact window (3x3, 5x5, 7x7) depending on sigma
    """
    from scipy.optimize import least_squares

    if stamp.ndim != 2:
        return False, {}
    h, w = stamp.shape
    if h < 5 or w < 5:
        return False, {}

    arr = stamp.astype(np.float64, copy=False)
    flat = arr.ravel()
    k = int(max(1, 4 * int(win)))
    k = min(k, flat.size)
    m1 = float(np.nanmedian(np.sort(flat)[:k]))
    img = arr - m1
    if not np.isfinite(img).any():
        return False, {}

    sm = _nd_median_filter(img, size=3, mode="nearest")
    if not np.isfinite(sm).any():
        return False, {}
    mp = int(np.nanargmax(sm))
    y0p, x0p = np.unravel_index(mp, sm.shape)
    amp0 = float(img[y0p, x0p])
    if not np.isfinite(amp0) or amp0 <= 0:
        return False, {}

    # Initial sigma from central second moments
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    pos = np.clip(img, 0.0, None)
    s = float(np.nansum(pos))
    if s > 0:
        xbar = float(np.nansum(xx * pos) / s)
        ybar = float(np.nansum(yy * pos) / s)
        sx = float(np.clip(np.sqrt(max(np.nansum(((xx - xbar) ** 2) * pos) / s, 0.5)), 0.6, 4.0))
        sy = float(np.clip(np.sqrt(max(np.nansum(((yy - ybar) ** 2) * pos) / s, 0.5)), 0.6, 4.0))
    else:
        sx = sy = 1.5
    x0 = float(x0p)
    y0 = float(y0p)

    def choose_halfbox(sig: float) -> int:
        if sig <= 1.0:
            return 1  # 3x3
        if sig <= 3.0:
            return 2  # 5x5
        return 3      # 7x7

    # Two-stage local fit: compact window from sigma, then refit once more
    p = np.array([amp0, x0, y0, sx, sy], dtype=np.float64)
    for _ in range(2):
        hb = max(1, choose_halfbox(float(max(p[3], p[4]))))
        xc = int(round(p[1])); yc = int(round(p[2]))
        xlo = max(0, xc - hb); xhi = min(w, xc + hb + 1)
        ylo = max(0, yc - hb); yhi = min(h, yc + hb + 1)
        if (xhi - xlo) < 3 or (yhi - ylo) < 3:
            xlo = max(0, int(round(x0p)) - 1); xhi = min(w, int(round(x0p)) + 2)
            ylo = max(0, int(round(y0p)) - 1); yhi = min(h, int(round(y0p)) + 2)

        yyf, xxf = np.mgrid[ylo:yhi, xlo:xhi].astype(np.float64)
        z = img[ylo:yhi, xlo:xhi]
        good = np.isfinite(z)
        if good.sum() < 6:
            return False, {}

        def resid(par: np.ndarray) -> np.ndarray:
            amp, xcen, ycen, sx1, sy1 = par
            mod = _integrated_gaussian_2d(xxf, yyf, amp, xcen, ycen, max(sx1, 1e-3), max(sy1, 1e-3))
            r = (mod - z)[good]
            return r

        lb = np.array([0.0, xlo - 0.5, ylo - 0.5, 0.3, 0.3], dtype=np.float64)
        ub = np.array([np.inf, xhi - 0.5, yhi - 0.5, 8.0, 8.0], dtype=np.float64)
        try:
            res = least_squares(resid, p, bounds=(lb, ub), method="trf", max_nfev=200)
        except Exception:
            return False, {}
        if not res.success:
            return False, {}
        p = res.x.astype(np.float64)

    amp, xcen, ycen, sx, sy = map(float, p)
    if amp <= 0 or sx <= 0 or sy <= 0:
        return False, {}
    if not (0.0 <= xcen <= w - 1 and 0.0 <= ycen <= h - 1):
        return False, {}

    xfw = 2.354820045 * sx
    yfw = 2.354820045 * sy
    flux = float(amp * 2.0 * np.pi * sx * sy)
    return True, {"flux": flux, "xfw": float(xfw), "yfw": float(yfw), "sky": float(m1), "x0": xcen, "y0": ycen}


def _fit_stars_star2d_like_idl(img: np.ndarray, stars_xy: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (fluxes, xfw) using IDL-like local integrated-Gaussian fits."""
    n = stars_xy.shape[0]
    flux = np.full(n, np.nan, dtype=float)
    xfw = np.full(n, np.nan, dtype=float)
    for i, (x, y) in enumerate(stars_xy):
        try:
            stamp, _, _ = _extract_stamp(img, float(x), float(y), win=win)
        except Exception:
            continue
        ok, met = _fit_star_gauss2d_like_idl(stamp, win=int(win))
        if not ok:
            continue
        flux[i] = float(met["flux"])
        xfw[i] = float(met["xfw"])
    return flux, xfw


def _design_matrix_scale(x: np.ndarray, y: np.ndarray, mode: str = "global") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if mode == "global":
        return np.column_stack([np.ones_like(x)])
    if mode == "plane":
        return np.column_stack([np.ones_like(x), x, y])
    if mode == "quadratic":
        return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    if mode == "radial":
        xc = np.nanmedian(x)
        yc = np.nanmedian(y)
        r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        return np.column_stack([np.ones_like(x), r, r * r])
    if mode == "y":
        return np.column_stack([np.ones_like(x), y, y * y])
    raise ValueError(f"Unknown scale mode: {mode}")


def _eval_scale_model(xx: np.ndarray, yy: np.ndarray, coeffs: np.ndarray, mode: str = "global") -> np.ndarray:
    xx = np.asarray(xx, dtype=np.float64)
    yy = np.asarray(yy, dtype=np.float64)
    if mode == "global":
        return coeffs[0] + 0.0 * xx
    if mode == "plane":
        return coeffs[0] + coeffs[1] * xx + coeffs[2] * yy
    if mode == "quadratic":
        return coeffs[0] + coeffs[1] * xx + coeffs[2] * yy + coeffs[3] * xx * xx + coeffs[4] * xx * yy + coeffs[5] * yy * yy
    if mode == "radial":
        xc = np.nanmedian(xx)
        yc = np.nanmedian(yy)
        rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
        return coeffs[0] + coeffs[1] * rr + coeffs[2] * rr * rr
    if mode == "y":
        return coeffs[0] + coeffs[1] * yy + coeffs[2] * yy * yy
    raise ValueError(f"Unknown scale mode: {mode}")


def fit_spatial_scale_model(
    x: np.ndarray,
    y: np.ndarray,
    ratio: np.ndarray,
    mode: str = "global",
    sigma_clip: float = 2.5,
    max_iter: int = 5,
    min_stars: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ratio = np.asarray(ratio, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(ratio) & (ratio > 0)
    x = x[finite]
    y = y[finite]
    ratio = ratio[finite]
    if len(ratio) == 0:
        coeffs = np.array([1.0], dtype=np.float64)
        used = np.zeros(0, dtype=bool)
        model = np.array([], dtype=np.float64)
        resid = np.array([], dtype=np.float64)
        return coeffs, used, model, resid, {"mode": "fallback_global", "n_total": 0, "n_used": 0, "sigma_clip": float(sigma_clip)}
    if len(ratio) < min_stars:
        coeffs = np.array([float(np.nanmedian(ratio))], dtype=np.float64)
        used = np.ones_like(ratio, dtype=bool)
        model = np.full_like(ratio, coeffs[0], dtype=np.float64)
        resid = ratio - model
        return coeffs, used, model, resid, {"mode": "fallback_global", "n_total": int(len(ratio)), "n_used": int(np.sum(used)), "sigma_clip": float(sigma_clip)}
    A = _design_matrix_scale(x, y, mode=mode)
    npar = A.shape[1]
    if len(ratio) < max(npar + 2, min_stars):
        mode = "plane" if len(ratio) >= 5 else "global"
        A = _design_matrix_scale(x, y, mode=mode)
    used = np.ones(len(ratio), dtype=bool)
    for _ in range(max_iter):
        Au = A[used]
        ru = ratio[used]
        coeffs, *_ = np.linalg.lstsq(Au, ru, rcond=None)
        model = A @ coeffs
        resid = ratio - model
        mad = np.nanmedian(np.abs(resid[used] - np.nanmedian(resid[used])))
        sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(resid[used])
        if not np.isfinite(sigma) or sigma <= 0:
            break
        new_used = np.abs(resid) <= float(sigma_clip) * sigma
        if np.sum(new_used) < max(A.shape[1] + 1, 4):
            break
        if np.array_equal(new_used, used):
            used = new_used
            break
        used = new_used
    Au = A[used]
    ru = ratio[used]
    coeffs, *_ = np.linalg.lstsq(Au, ru, rcond=None)
    model = A @ coeffs
    resid = ratio - model
    stats = {
        "mode": mode,
        "n_total": int(len(ratio)),
        "n_used": int(np.sum(used)),
        "sigma_clip": float(sigma_clip),
        "ratio_median": float(np.nanmedian(ratio)),
        "ratio_std": float(np.nanstd(ratio)),
        "resid_std_used": float(np.nanstd(resid[used])) if np.any(used) else np.nan,
    }
    return coeffs, used, model, resid, stats


def _kmap_from_stars(
    ratios: np.ndarray,
    stars: np.ndarray,
    shape: tuple[int, int],
    mode: str = "radial",
    sigma_clip: float = 2.5,
    nbins: int = 12,
    smooth_bins: int = 1,
    center: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict]:
    """Build a spatially varying k-map from star flux ratios.
    mode:
      - 'radial': k = k(r) in radial bins around center
      - 'y':      k = k(y) in horizontal stripes
    Returns (k_map, info)
    """
    h, w = shape
    info: dict = {"local_mode": mode, "local_nbins": int(nbins), "local_smooth_bins": int(smooth_bins), "local_sigma_clip": float(sigma_clip)}
    nb = max(3, int(nbins))
    sm = max(1, int(smooth_bins))
    ok = np.isfinite(ratios) & (ratios > 0)
    if ok.sum() < 5:
        raise RuntimeError(f"Too few valid star ratios for local scaling: {int(ok.sum())}")
    stars_ok = stars[ok]
    ratios_ok = ratios[ok].astype(np.float64)

    if mode == "radial":
        if center is None:
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        else:
            cx, cy = float(center[0]), float(center[1])
        dx = stars_ok[:, 0] - cx
        dy = stars_ok[:, 1] - cy
        r = np.sqrt(dx * dx + dy * dy)
        rmax = float(np.sqrt((max(cx, w-1-cx))**2 + (max(cy, h-1-cy))**2))
        edges = np.linspace(0.0, rmax, nb + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        kbin = np.full(nb, np.nan, dtype=np.float64)
        nbin = np.zeros(nb, dtype=int)
        for i in range(nb):
            sel = (r >= edges[i]) & (r < edges[i+1])
            nbin[i] = int(sel.sum())
            if nbin[i] >= 3:
                vals = ratios_ok[sel]
                med = np.nanmedian(vals)
                mad = np.nanmedian(np.abs(vals - med))
                if np.isfinite(mad) and mad > 0:
                    sig = 1.4826 * mad
                    keep = np.abs(vals - med) <= 3.0 * sig
                    if keep.sum() >= 3:
                        vals = vals[keep]
                kbin[i] = float(np.nanmedian(vals))
        # fill missing bins by interpolation of available
        good = np.isfinite(kbin)
        if good.sum() < 3:
            raise RuntimeError(f"Too few radial bins with k estimates: {int(good.sum())}/{nb}")
        kbin_filled = np.interp(centers, centers[good], kbin[good])
        # smooth in bin space (moving average)
        if sm > 1:
            pad = sm // 2
            kk = np.pad(kbin_filled, (pad, pad), mode="edge")
            ksm = np.convolve(kk, np.ones(sm)/sm, mode="valid")
        else:
            ksm = kbin_filled
        # build map
        yy, xx = np.mgrid[0:h, 0:w]
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        k_map = np.interp(rr, centers, ksm).astype(np.float32)
        info.update({"local_center_x": float(cx), "local_center_y": float(cy), "local_rmax": float(rmax)})
        info["local_bin_counts"] = nbin.tolist()
        info["local_kbin_median"] = [float(x) for x in kbin_filled.tolist()]
        info["local_kbin_smoothed"] = [float(x) for x in ksm.tolist()]
        return k_map, info

    elif mode == "y":
        y = stars_ok[:, 1]
        edges = np.linspace(0.0, float(h), nb + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        kbin = np.full(nb, np.nan, dtype=np.float64)
        nbin = np.zeros(nb, dtype=int)
        for i in range(nb):
            sel = (y >= edges[i]) & (y < edges[i+1])
            nbin[i] = int(sel.sum())
            if nbin[i] >= 3:
                vals = ratios_ok[sel]
                med = np.nanmedian(vals)
                mad = np.nanmedian(np.abs(vals - med))
                if np.isfinite(mad) and mad > 0:
                    sig = 1.4826 * mad
                    keep = np.abs(vals - med) <= 3.0 * sig
                    if keep.sum() >= 3:
                        vals = vals[keep]
                kbin[i] = float(np.nanmedian(vals))
        good = np.isfinite(kbin)
        if good.sum() < 3:
            raise RuntimeError(f"Too few y-bins with k estimates: {int(good.sum())}/{nb}")
        kbin_filled = np.interp(centers, centers[good], kbin[good])
        if sm > 1:
            pad = sm // 2
            kk = np.pad(kbin_filled, (pad, pad), mode="edge")
            ksm = np.convolve(kk, np.ones(sm)/sm, mode="valid")
        else:
            ksm = kbin_filled
        yy = np.arange(h, dtype=np.float64)
        k_row = np.interp(yy, centers, ksm).astype(np.float32)
        k_map = np.repeat(k_row[:, None], w, axis=1)
        info["local_bin_counts"] = nbin.tolist()
        info["local_kbin_median"] = [float(x) for x in kbin_filled.tolist()]
        info["local_kbin_smoothed"] = [float(x) for x in ksm.tolist()]
        return k_map, info

    elif mode in {"plane", "quadratic"}:
        xn = stars_ok[:, 0].astype(np.float64) / max(float(w - 1), 1.0)
        yn = stars_ok[:, 1].astype(np.float64) / max(float(h - 1), 1.0)
        coeff, keep, model, resid, fit_stats = fit_spatial_scale_model(
            xn, yn, ratios_ok, mode=mode, sigma_clip=float(sigma_clip), max_iter=5, min_stars=5
        )
        need = 3 if mode == "plane" else 6
        if int(keep.sum()) < min(need, len(ratios_ok)):
            raise RuntimeError(f"Too few stars for {mode} k fit: {int(keep.sum())}")
        xx = np.arange(w, dtype=np.float64) / max(float(w - 1), 1.0)
        yy = np.arange(h, dtype=np.float64) / max(float(h - 1), 1.0)
        xg, yg = np.meshgrid(xx, yy)
        k_map = _eval_scale_model(xg, yg, coeff, mode=fit_stats["mode"]).astype(np.float32)
        info["local_fit_mode_used"] = str(fit_stats["mode"])
        info["local_fit_n_kept"] = int(keep.sum())
        info["local_fit_resid_std_used"] = float(fit_stats.get("resid_std_used", np.nan))
        if fit_stats["mode"] == "plane":
            info["local_plane_coeff"] = [float(x) for x in coeff.tolist()]
        elif fit_stats["mode"] == "quadratic":
            info["local_quadratic_coeff"] = [float(x) for x in coeff.tolist()]
        else:
            info["local_fallback_coeff"] = [float(x) for x in coeff.tolist()]
        info["local_keep_mask"] = [bool(x) for x in keep.tolist()]
        return k_map, info

    else:
        raise ValueError(f"Unknown local k mode: {mode}")
def _star_flux_aperture(img: np.ndarray, stars: np.ndarray, win: int = 30) -> np.ndarray:
    """Robust star flux estimate without nonlinear fitting.
    Uses a square box of size (2*win+1), background = median of darkest 5*win pixels (IDL-like),
    flux = sum(max(img-bg,0)) inside a circular aperture of radius win/2.
    """
    h, w = img.shape
    r = max(3.0, win / 2.0)
    yy, xx = np.mgrid[-win:win+1, -win:win+1]
    rr = np.sqrt(xx*xx + yy*yy)
    ap_mask = rr <= r
    flux = np.full(len(stars), np.nan, dtype=np.float64)
    for i, (x, y) in enumerate(stars):
        xi, yi = int(round(x)), int(round(y))
        if xi - win < 0 or yi - win < 0 or xi + win >= w or yi + win >= h:
            continue
        box = img[yi-win:yi+win+1, xi-win:xi+win+1].astype(np.float64, copy=False)
        # background like IDL star_2d: sort pixels and take median of first 0:4*win
        flat = np.sort(box.ravel())
        nbg = min(len(flat), max(10, 4*win + 1))
        bg = np.median(flat[:nbg])
        resid = box - bg
        resid[~np.isfinite(resid)] = 0.0
        # clip negative to reduce sensitivity to background errors / absorption features
        resid = np.where(resid > 0, resid, 0.0)
        flux[i] = float(resid[ap_mask].sum())
    return flux
def continuum_scale_like_idl(
    emis: np.ndarray,
    cont: np.ndarray,
    stars_xy_path: Path,
    *,
    win: int = 15,
    seeing_lim: float = 0.25,
    mul: float = 1.0,
    local_mode: str = "global",
    local_nbins: int = 12,
    local_smooth_bins: int = 1,
    local_center: tuple[float, float] | None = None,
    scale_sigma_clip: float = 2.5,
) -> tuple[object, np.ndarray, np.ndarray, dict]:
    """IDL cre_emis_map-like scaling: PSF match + k = mul*median(flux_emis/flux_cont)."""
    stars = _read_stars_xy(stars_xy_path)
    # initial fits
    f_em, w_em = _fit_stars_star2d_like_idl(emis, stars, win=win)
    f_co, w_co = _fit_stars_star2d_like_idl(cont, stars, win=win)
    ok = np.isfinite(f_em) & np.isfinite(f_co) & (f_em > 0) & (f_co > 0) & np.isfinite(w_em) & np.isfinite(w_co)
    info = {"n_star_total": int(stars.shape[0]), "n_star_used": int(ok.sum())}

    if ok.sum() < 3:
        # Fallback: robust aperture fluxes without nonlinear fitting
        f_em_ap = _star_flux_aperture(emis, stars, win=win)
        f_co_ap = _star_flux_aperture(cont, stars, win=win)
        ok_ap = np.isfinite(f_em_ap) & np.isfinite(f_co_ap) & (f_em_ap > 0) & (f_co_ap > 0)
        info["fallback_aperture_used"] = True
        info["n_stars_ok_fit"] = int(ok.sum())
        info["n_stars_ok_aperture"] = int(ok_ap.sum())
        if ok_ap.sum() >= 3:
            f_em = f_em_ap
            f_co = f_co_ap
            ok = ok_ap
        else:
            # Final fallback: do not crash; use mul only (k=mul)
            info["k_median_flux_ratio"] = np.nan
            info["k_final"] = float(mul)
            info["psf_matched"] = "none"
            info["fallback_mul_only"] = True
            return float(mul), emis.astype(np.float32, copy=False), cont.astype(np.float32, copy=False), info

    av_em = _robomean(w_em[ok], nsig=3.0, niter=1)
    av_co = _robomean(w_co[ok], nsig=3.0, niter=1)
    info["fwhm_emis_px"] = av_em
    info["fwhm_cont_px"] = av_co
    info["fwhm_ratio_cont_over_emis"] = float(av_co / av_em) if av_em > 0 else np.nan

    # PSF match (Gaussian approximation), like IDL: convolve narrower image if |ratio-1| > seeing_lim
    ratio = (av_co / av_em) if (av_em > 0 and av_co > 0) else 1.0
    emis2 = emis.astype(np.float32, copy=True)
    cont2 = cont.astype(np.float32, copy=True)
    if np.isfinite(ratio) and abs(ratio - 1.0) > float(seeing_lim):
        if ratio < 1.0:
            # continuum narrower -> convolve cont to match emis
            f_psf = float(np.sqrt(max(av_em**2 - av_co**2, 0.0)))
            sigma = min(float(f_psf / 2.354820045), 12.0)
            cont2 = _nd_gaussian_filter(cont2, sigma=float(sigma), mode="nearest")
            info["psf_matched"] = "cont_convolved"
            info["psf_fwhm_px"] = f_psf
            info["psf_sigma_px"] = sigma
        else:
            f_psf = float(np.sqrt(max(av_co**2 - av_em**2, 0.0)))
            sigma = min(float(f_psf / 2.354820045), 12.0)
            emis2 = _nd_gaussian_filter(emis2, sigma=float(sigma), mode="nearest")
            info["psf_matched"] = "emis_convolved"
            info["psf_fwhm_px"] = f_psf
            info["psf_sigma_px"] = sigma
    else:
        info["psf_matched"] = "none"

    # refit fluxes and widths after PSF match (matches IDL where it fits after convolution)
    f_em2, w_em2 = _fit_stars_star2d_like_idl(emis2, stars, win=win)
    f_co2, w_co2 = _fit_stars_star2d_like_idl(cont2, stars, win=win)
    okw2 = np.isfinite(w_em2) & np.isfinite(w_co2)
    if okw2.sum() >= 3:
        info["fwhm_emis_post_px"] = _robomean(w_em2[okw2], nsig=3.0, niter=1)
        info["fwhm_cont_post_px"] = _robomean(w_co2[okw2], nsig=3.0, niter=1)
        if np.isfinite(info["fwhm_emis_post_px"]) and info["fwhm_emis_post_px"] > 0:
            info["fwhm_ratio_post"] = float(info["fwhm_cont_post_px"] / info["fwhm_emis_post_px"])
    ok2 = np.isfinite(f_em2) & np.isfinite(f_co2) & (f_em2 > 0) & (f_co2 > 0)
    if ok2.sum() < 3:
        # fall back to pre-PSF fluxes; if those came from fit and are sparse, try aperture
        ok2 = ok
        f_em2, f_co2 = f_em, f_co
        if ok2.sum() < 3:
            f_em_ap2 = _star_flux_aperture(emis2, stars, win=win)
            f_co_ap2 = _star_flux_aperture(cont2, stars, win=win)
            ok_ap2 = np.isfinite(f_em_ap2) & np.isfinite(f_co_ap2) & (f_em_ap2 > 0) & (f_co_ap2 > 0)
            if ok_ap2.sum() >= 3:
                f_em2, f_co2 = f_em_ap2, f_co_ap2
                ok2 = ok_ap2
                info["fallback_aperture_after_psf"] = True

    ratios = (f_em2[ok2] / f_co2[ok2]).astype(np.float64)
    info["n_star_used_post_psf"] = int(ok2.sum())

    # IDL cre_emis_map.pro behaviour:
    # k = mul * median(gau1.flux/gau2.flux) without additional sigma-clipping.
    if ratios.size == 0 or not np.any(np.isfinite(ratios)):
        # Fallback: no usable stars after PSF match -> use mul only (k = mul)
        info["k_median_flux_ratio"] = float("nan")
        info["k_final"] = float(mul)
        info["k_clip_used"] = False
        info["fallback_mul_only"] = True
        return float(mul), emis2.astype(np.float32, copy=False), cont2.astype(np.float32, copy=False), info

    k0 = float(np.nanmedian(ratios))
    k = float(mul) * k0

    info["k_median_flux_ratio"] = k0
    info["k_clip_used"] = False
    info["k_clip_n_kept"] = int(ratios.size)
    info["k_clip_n_all"] = int(ratios.size)
    try:
        info["k_ratio_min"] = float(np.nanmin(ratios))
        info["k_ratio_max"] = float(np.nanmax(ratios))
    except Exception:
        pass
    info["k_final"] = float(k)

    # Per-star diagnostics after PSF matching.
    star_rows = []
    ratio_full = np.full(stars.shape[0], np.nan, dtype=np.float64)
    ratio_full[ok2] = f_em2[ok2] / f_co2[ok2]
    for i, (xy, fe, fc, rr) in enumerate(zip(stars, f_em2, f_co2, ratio_full), start=1):
        star_rows.append({
            "star_id": int(i),
            "x": float(xy[0]),
            "y": float(xy[1]),
            "flux_emis": float(fe) if np.isfinite(fe) else np.nan,
            "flux_cont": float(fc) if np.isfinite(fc) else np.nan,
            "ratio": float(rr) if np.isfinite(rr) else np.nan,
            "used": bool(np.isfinite(rr) and rr > 0),
            "used_in_fit": bool(np.isfinite(rr) and rr > 0),
        })
    info["star_ratio_rows"] = star_rows

    mode = str(local_mode).lower()
    if mode == "global":
        return float(k), emis2, cont2, info

    n_ok = int(np.sum(np.isfinite(ratio_full) & (ratio_full > 0)))
    min_stars = {"plane": 6, "quadratic": 10}.get(mode, 5)
    try:
        if n_ok < min_stars:
            raise RuntimeError(f"only {n_ok} stars with a valid flux ratio (need >= {min_stars} for '{mode}')")
        k_map0, info_local = _kmap_from_stars(
            ratio_full,
            stars,
            emis.shape,
            mode=mode,
            nbins=int(local_nbins),
            smooth_bins=int(local_smooth_bins),
            center=local_center,
            sigma_clip=float(scale_sigma_clip),
        )
    except RuntimeError as e:
        print(f"[warn] local k ('{mode}') not possible: {e}; using the single global k", flush=True)
        info["local_mode"] = "global (fallback)"
        return float(k), emis2, cont2, info
    k_map = (float(mul) * k_map0).astype(np.float32)
    info.update(info_local)
    info["k_map_median"] = float(np.nanmedian(k_map))
    info["k_final"] = float(np.nanmedian(k_map))
    keep_mask = info_local.get("local_keep_mask")
    keep_iter = iter(keep_mask) if isinstance(keep_mask, list) else None
    for row in star_rows:
        rr = row.get("ratio")
        if np.isfinite(rr):
            row["used_in_fit"] = bool(next(keep_iter)) if keep_iter is not None else True
            x = float(row["x"])
            y = float(row["y"])
            xi = int(np.clip(round(x), 0, k_map.shape[1] - 1))
            yi = int(np.clip(round(y), 0, k_map.shape[0] - 1))
            row["k_local"] = float(k_map[yi, xi])
            row["ratio_minus_kdivmul"] = float(rr - (k_map[yi, xi] / max(float(mul), 1e-12)))
        else:
            row["used_in_fit"] = False
            row["k_local"] = np.nan
            row["ratio_minus_kdivmul"] = np.nan
    return k_map, emis2, cont2, info


def make_subtraction_figure(line: np.ndarray, cont: np.ndarray, net: np.ndarray, out_png: Path, *,
                            k: float, info: dict | None = None, title: str = "") -> Path:
    """
    Picture of the DIFFERENCE only: net = line - k*cont, symmetric stretch around 0
    (red = positive residual / emission, blue = over-subtracted). No pyplot -> does not touch
    the caller's matplotlib backend.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    net = np.asarray(net, dtype=np.float32)
    valid = np.isfinite(net) & (net != 0)  # frame outside the mask is exactly 0
    if valid.sum() < 10:
        valid = np.isfinite(net)
    nv = net[valid]
    lim = float(np.nanpercentile(np.abs(nv), 99.0)) if nv.size else 1.0
    lim = lim if lim > 0 else 1.0
    med = float(np.nanmedian(nv)) if nv.size else 0.0
    sig = float(1.4826 * np.nanmedian(np.abs(nv - med))) if nv.size else 0.0

    ny, nx = net.shape
    w = 8.0
    fig = Figure(figsize=(w + 1.2, w * ny / max(nx, 1) + 1.0), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(np.where(np.isfinite(net), net, np.nan), origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title(f"{title}\nnet = line - {k:.4g} * cont    (median {med:.3g}, sigma {sig:.3g})", fontsize=10)
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png)
    return out_png


_LAYER_KEY_RE = re.compile(r"^(NLAYERS|(LAYER|LKIND|LROLE|LTAG|LZSP|LTEXP|LNFR|LZMEAN|LLAM)\d+)$")


def _layer_zspaces(L) -> list[str]:
    """ZSPACE values of a layer; combined nights separate their values with '/' (not comparable between nights)."""
    return [z for z in re.split(r"[;/]", str(L.get("zspace") or "")) if z.strip()]


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _hdr_for_layer(hdr, layers: list[dict], line_layer: int, cont_layer: int | None = None):
    """
    Header for 2D products cut from a multi-layer cube: layer-table keywords removed and the keywords
    of the layer itself written: EXPTIME = exposure of THIS layer (not of the first frame), NFRAMES,
    ZMEAN, ZSPACE (if the layer has a single value). TLINE / TCONT note the exposures of the line and
    of the continuum layer.
    """
    h = hdr.copy()
    for key in [k for k in list(h.keys()) if _LAYER_KEY_RE.match(str(k))]:
        try:
            del h[key]
        except Exception:
            pass

    def _lay(i):
        return layers[i] if (i is not None and 0 <= i < len(layers)) else None

    L = _lay(line_layer)
    h["SRCLAYER"] = ((L["name"] if L else f"layer{line_layer}")[:60], "layer of the step6 cube")
    if L:
        t = _num(L.get("texp"))
        if t:
            h["EXPTIME"] = (t, "Exposure time [s] of this image")
        n = _num(L.get("nframes"))
        if n:
            h["NFRAMES"] = (int(n), "Number of frames summed")
        zm = _num(L.get("zmean"))
        if zm:
            h["ZMEAN"] = (zm, "Mean zenith distance [deg], airmass weighted")
        else:
            for k in ("ZMEAN",):
                if k in h:
                    del h[k]
        zs = _layer_zspaces(L)
        lam = _num(L.get("lam"))
        if lam:
            h["LAMBDA"] = (round(lam, 2), "[A] mean wavelength of the layer")
        if len(zs) == 1:
            h["ZSPACE"] = int(float(zs[0]))
        elif len(zs) > 1 and "ZSPACE" in h:
            del h["ZSPACE"]
        if "ZDIST" in h:
            del h["ZDIST"]
    for key in ("TLINE", "TCONT"):  # totals of the whole cube from step6: not valid for one layer
        if key in h:
            del h[key]
    for i in (line_layer, cont_layer):
        Li = _lay(i)
        t = _num(Li.get("texp")) if Li else None
        if t:
            h["TLINE" if Li["kind"] == "line" else "TCONT"] = (t, "[s] exposure of the layer")
    return h


def _combined_cont_stats(layers: list[dict], idx: list[int]) -> tuple[float, int, float | None]:
    """Total exposure, number of frames and ZMEAN of the combination of the continuum layers idx."""
    ts = [_num(layers[i].get("texp")) or 0.0 for i in idx]
    ns = [int(_num(layers[i].get("nframes")) or 0) for i in idx]
    zs = [_num(layers[i].get("zmean")) for i in idx]
    pairs = [(z, t) for z, t in zip(zs, ts) if z]
    zm = combine_zmean([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None
    return float(sum(ts)), int(sum(ns)), zm


def _stamped_writeto(path, data, hdr, **kw):
    """fits.writeto that first writes FILENAME / DATE of the file being created into the header."""
    stamp(hdr, path)
    fits.writeto(path, data, hdr, **kw)


# ====================================================================================================
# Stars in the continuum subtraction (registration of the continuum, per-star match)
#
# Star handling for the continuum subtraction net = line - k * cont (direct images in two different filters).
#
# Two things a single number k cannot do:
#
# 1. Geometry.  The line and the continuum filters are different pieces of glass: the image scale differs by a few
#    1e-4, i.e. some tenths of a pixel at the edge of the field.  Stars then leave +/- "dipoles" growing towards the
#    edge.  register_from_matches() fits a linear (affine) transformation by the stars and resamples the continuum
#    onto the grid of the line image (the IDL counterpart is im_align_stars / im_bilin).
#
# 2. Colour.  The line filter and the continuum filter are hundreds of Angstrom apart, so flux_line / flux_cont is an
#    individual number of every star (spectral type, absorption features in the narrow filter).  With one k some
#    stars are under-subtracted and others over-subtracted whatever k is.  per_star_correction() makes the subtraction equal
#    to the global k everywhere and to the star's own ratio in a patch around every point source.  It is a cosmetic
#    operation: an unresolved source with real line emission is removed as well, so it is optional and every
#    touched source is listed in a table.
# ====================================================================================================
from scipy import ndimage as _ss_ndi

_SS_FWHM2SIG = 2.354820045


def _ss_fit_star(img: np.ndarray, x: float, y: float, win: int):
    ny, nx = img.shape
    xi, yi = int(round(x)), int(round(y))
    if not (win <= xi < nx - win and win <= yi < ny - win):
        return None
    stamp, _, _ = _extract_stamp(img, float(xi), float(yi), win=win)
    if stamp.shape != (2 * win + 1, 2 * win + 1) or not np.isfinite(stamp).all():
        return None
    try:
        ok, m = _fit_star_gauss2d_like_idl(stamp, win=int(win))
    except Exception:
        return None
    if not ok:
        return None
    return {"flux": float(m["flux"]), "xfw": float(m["xfw"]), "yfw": float(m["yfw"]),
            "x": xi - win + float(m["x0"]), "y": yi - win + float(m["y0"]), "peak": float(np.nanmax(stamp) - m["sky"])}


def _ss_robust_sigma(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def find_point_sources(line: np.ndarray, cont: np.ndarray, *, win: int = 12, nsigma: float = 8.0,
                       fwhm_tol: float = 0.25, max_sources: int = 400) -> list[dict]:
    """
    Point sources seen in the continuum image, measured in both images (same Gaussian fit as for k).
    Kept: finite fits in both images, positive fluxes, FWHM within fwhm_tol of the median stellar FWHM (both axes)
    -> galaxies, blends and defects are dropped.
    Returns a list of dicts: x, y (line image), xc, yc (continuum image), flux_line, flux_cont, ratio, fwhm, peak_cont.
    """
    c0 = np.nan_to_num(cont, nan=0.0)
    sm = _ss_ndi.gaussian_filter(c0, 1.5)
    sig = _ss_robust_sigma(sm[np.isfinite(cont)])
    if sig <= 0:
        return []
    peaks = (sm == _ss_ndi.maximum_filter(sm, size=2 * win + 1)) & (sm > nsigma * sig) & np.isfinite(cont)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-sm[ys, xs])[:max_sources]
    rows = []
    for x, y in zip(xs[order], ys[order]):
        a = _ss_fit_star(line, x, y, win)
        b = _ss_fit_star(cont, x, y, win)
        if b is None or b["flux"] <= 0:
            continue
        if a is None or a["flux"] <= 0 or np.hypot(a["x"] - b["x"], a["y"] - b["y"]) > 1.5:
            # seen in the continuum only: a moving object (asteroid) or a defect, not a star
            if not np.isfinite(line[int(round(b["y"])), int(round(b["x"]))]):
                continue
            rows.append({"x": b["x"], "y": b["y"], "xc": b["x"], "yc": b["y"], "flux_line": 0.0, "flux_cont": b["flux"],
                         "ratio": 0.0, "xfw": b["xfw"], "yfw": b["yfw"], "fwhm": 0.5 * (b["xfw"] + b["yfw"]),
                         "peak_cont": b["peak"], "cont_only": True})
            continue
        rows.append({"x": a["x"], "y": a["y"], "xc": b["x"], "yc": b["y"], "flux_line": a["flux"], "flux_cont": b["flux"],
                     "ratio": a["flux"] / b["flux"], "xfw": b["xfw"], "yfw": b["yfw"],
                     "fwhm": 0.5 * (b["xfw"] + b["yfw"]), "peak_cont": b["peak"], "cont_only": False})
    if len(rows) < 3:
        return rows
    # the brightest sources define the stellar FWHM
    stars = [r for r in rows if not r["cont_only"]]
    fw = np.median([min(r["xfw"], r["yfw"]) for r in (stars or rows)[: max(5, len(rows) // 2)]])
    lo, hi = fw * (1.0 - fwhm_tol), fw * (1.0 + fwhm_tol)
    out = []
    for r in rows:
        top = 5.0 * fw if r["cont_only"] else hi          # a moving object is elongated
        if lo <= min(r["xfw"], r["yfw"]) and max(r["xfw"], r["yfw"]) <= top:
            out.append(r)
    return out


def _ss_taper(shape_slices, x, y, R):
    y0, y1, x0, x1 = shape_slices
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    r0 = 0.6 * R
    return np.where(rr <= r0, 1.0, np.where(rr >= R, 0.0, 0.5 * (1 + np.cos(np.pi * (rr - r0) / max(R - r0, 1e-6)))))


def local_match(line: np.ndarray, cont: np.ndarray, sources: list[dict], k: float, *, noise: float,
                full_snr: float = 30.0) -> list[dict]:
    """
    Least-squares match of the continuum to the line image in a patch around every point source:

        line = a * cont + b * dcont/dx + c * dcont/dy + e * laplacian(cont) + d

    a - flux ratio of THIS star (its colour), b/a and c/a - residual shift of the star between the two images (px),
    e - difference of the PSF widths, d - local pedestal (diffuse emission under the star; it is fitted, never removed).
    Faint stars (k * peak / noise < full_snr) are fitted with a and d only.
    Adds to every source: a, dx, dy, e, snr, radius, fit ('full' / 'scale' / '').
    """
    ny, nx = line.shape
    c0 = np.nan_to_num(cont, nan=0.0).astype(np.float64)
    gy, gx = np.gradient(c0)
    gyy, gyx = np.gradient(gy); gxy, gxx = np.gradient(gx)
    out = []
    for s in sources:
        s = dict(s)
        snr = float(k) * s["peak_cont"] / max(noise, 1e-12)
        R = float(s["fwhm"] * (2.0 + 1.2 * np.log10(max(snr, 1.0))))
        h = int(np.ceil(R)) + 1
        xi, yi = int(round(s["x"])), int(round(s["y"]))
        y0, y1, x0, x1 = max(0, yi - h), min(ny, yi + h + 1), max(0, xi - h), min(nx, xi + h + 1)
        sl = (slice(y0, y1), slice(x0, x1))
        L = line[sl].astype(np.float64); C = cont[sl].astype(np.float64)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m = np.isfinite(L) & np.isfinite(C) & (np.hypot(xx - s["x"], yy - s["y"]) <= R)
        s.update({"snr": snr, "radius": R, "box": (y0, y1, x0, x1), "fit": "", "a": np.nan, "dx": 0.0, "dy": 0.0, "e": (0.0, 0.0, 0.0)})
        if m.sum() >= 30:
            full = snr >= full_snr
            if not full:
                # faint star: a regression on the noisy continuum pixels is biased low -> matched filter with a
                # noiseless Gaussian template of the star; local pedestals from the outer ring of the patch
                rr = np.hypot(xx - s["x"], yy - s["y"])
                ring = m & (rr > 0.75 * R)
                sgx, sgy = s["xfw"] / _SS_FWHM2SIG, s["yfw"] / _SS_FWHM2SIG
                T = np.exp(-0.5 * (((xx - s["x"]) / sgx) ** 2 + ((yy - s["y"]) / sgy) ** 2))
                if ring.sum() >= 10:
                    dL, dC = np.median(L[ring]), np.median(C[ring])
                    den = float(np.sum(T[m] * (C[m] - dC)))
                    if den > 0:
                        s["a"] = float(np.sum(T[m] * (L[m] - dL)) / den); s["d"] = float(dL - s["a"] * dC)
                        s["fit"] = "scale"
                out.append(s)
                continue
            cols = [C[m], np.ones(m.sum())] + ([gx[sl][m], gy[sl][m], gxx[sl][m], gyy[sl][m], gxy[sl][m]] if full else [])
            A = np.stack(cols, axis=1)
            try:
                p, *_ = np.linalg.lstsq(A, L[m], rcond=None)
                s["a"] = float(p[0]); s["d"] = float(p[1]); s["fit"] = "full" if full else "scale"
                if full and p[0] != 0:
                    s["b"], s["c"], s["e"] = float(p[2]), float(p[3]), (float(p[4]), float(p[5]), float(p[6]))
                    s["dx"], s["dy"] = float(p[2] / p[0]), float(p[3] / p[0])
            except Exception:
                pass
        out.append(s)
    return out


def register_from_matches(cont: np.ndarray, matches: list[dict], *, min_stars: int = 6, min_effect_px: float = 0.03,
                          clip: float = 3.0) -> tuple[np.ndarray, dict]:
    """
    Affine transformation of the continuum fitted to the per-star shifts (dx, dy) of local_match() (bright stars only).
    line(x, y) ~ cont(x + dx, y + dy), so the continuum is resampled at (x + dx(x, y), y + dy(x, y)).
    """
    st = [m for m in matches if m["fit"] == "full" and not m.get("cont_only") and np.hypot(m["dx"], m["dy"]) < 1.0]
    info: dict = {"register": "none", "register_nstars": len(st)}
    if len(st) < min_stars:
        info["register_reason"] = f"only {len(st)} bright stars (need >= {min_stars})"
        return cont, info
    ny, nx = cont.shape
    x = np.array([m["x"] for m in st]); y = np.array([m["y"] for m in st])
    dx = np.array([m["dx"] for m in st]); dy = np.array([m["dy"] for m in st])
    wgt = np.sqrt(np.clip(np.array([m["snr"] for m in st]), 1, 1000.0))
    A = np.c_[np.ones_like(x), x - nx / 2.0, y - ny / 2.0]
    keep = np.ones(len(x), bool)
    for _ in range(4):
        W = wgt[keep, None]
        cx, *_ = np.linalg.lstsq(A[keep] * W, dx[keep] * W[:, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(A[keep] * W, dy[keep] * W[:, 0], rcond=None)
        res = np.hypot(dx - A @ cx, dy - A @ cy)
        sg = max(1.4826 * np.median(res[keep]), 0.005)
        new = res < clip * sg + 0.01
        if new.sum() < min_stars or np.array_equal(new, keep):
            break
        keep = new
    cor = np.array([[1, -nx / 2, -ny / 2], [1, nx / 2, -ny / 2], [1, -nx / 2, ny / 2], [1, nx / 2, ny / 2]], float)
    effect = float(np.max(np.hypot(cor @ cx, cor @ cy)))
    info.update({"register_nused": int(keep.sum()),
                 "register_rms_before_px": float(np.sqrt(np.mean(dx[keep] ** 2 + dy[keep] ** 2))),
                 "register_rms_after_px": float(np.sqrt(np.mean(res[keep] ** 2))),
                 "register_max_shift_px": effect,
                 "register_dx": [float(v) for v in cx], "register_dy": [float(v) for v in cy]})
    if effect < min_effect_px:
        info["register_reason"] = f"transformation below {min_effect_px} px everywhere"
        return cont, info
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    u, v = xx - nx / 2.0, yy - ny / 2.0
    xs = xx + cx[0] + cx[1] * u + cx[2] * v
    ys = yy + cy[0] + cy[1] * u + cy[2] * v
    bad = ~np.isfinite(cont)
    out = _ss_ndi.map_coordinates(np.where(bad, 0.0, cont).astype(np.float64), [ys, xs], order=3, mode="constant", cval=0.0)
    cov = _ss_ndi.map_coordinates((~_ss_ndi.binary_dilation(bad, iterations=2)).astype(np.float64), [ys, xs], order=1,
                              mode="constant", cval=0.0)
    out[cov < 0.999] = np.nan
    info["register"] = "affine"
    return out.astype(np.float32), info


def per_star_correction(cont: np.ndarray, matches: list[dict], k: float | np.ndarray, *,
                        ratio_range: tuple[float, float] = (-0.3, 1.8), min_snr: float = 5.0,
                        protect: tuple[float, float, float] | None = None) -> tuple[np.ndarray, list[dict]]:
    """
    Correction image for  net = line - k * cont - corr :  inside a tapered patch around every point source the
    global k * cont is replaced by the local model of local_match() (own flux ratio, shift, PSF width of the star).
    Skipped: ratio a / k outside ratio_range (not a normal star: emission-line object, defect), too faint sources,
    sources inside the protected circle (x, y, r).  Returns (corr, table rows).
    """
    ny, nx = cont.shape
    c0 = np.nan_to_num(cont, nan=0.0).astype(np.float64)
    gy, gx = np.gradient(c0)
    gyy, gyx = np.gradient(gy); gxy, gxx = np.gradient(gx)
    corr = np.zeros(cont.shape, np.float64)
    wtot = np.zeros(cont.shape, np.float64)
    rows = []
    for m in sorted(matches, key=lambda q: -q["snr"]):
        xi = int(np.clip(round(m["x"]), 0, nx - 1)); yi = int(np.clip(round(m["y"]), 0, ny - 1))
        k0 = float(k) if np.isscalar(k) else float(np.asarray(k)[yi, xi])
        row = {"x": m["x"], "y": m["y"], "flux_cont": m["flux_cont"], "ratio_gauss": m["ratio"], "a": m["a"],
               "k_global": k0, "dx": m["dx"], "dy": m["dy"], "fwhm": m["fwhm"], "snr": m["snr"], "radius": m["radius"],
               "fit": m["fit"], "used": False, "why": ""}
        rel = m["a"] / k0 if (k0 > 0 and np.isfinite(m["a"])) else np.nan
        if not m["fit"]:
            row["why"] = "no fit"
        elif not np.isfinite(rel) or not (ratio_range[0] <= rel <= ratio_range[1]):
            row["why"] = "ratio/k outside the stellar range"
        elif m["snr"] < min_snr:
            row["why"] = "too faint"
        elif protect is not None and np.hypot(m["x"] - protect[0], m["y"] - protect[1]) < protect[2]:
            row["why"] = "inside the protected circle"
        else:
            y0, y1, x0, x1 = m["box"]
            sl = (slice(y0, y1), slice(x0, x1))
            w = _ss_taper(m["box"], m["x"], m["y"], m["radius"])
            w = w * (1.0 - wtot[sl])          # a pixel already owned by a brighter star is not modelled twice
            model = (m["a"] - k0) * c0[sl]
            if m["fit"] == "full":
                model = (model + m["b"] * gx[sl] + m["c"] * gy[sl]
                         + m["e"][0] * gxx[sl] + m["e"][1] * gyy[sl] + m["e"][2] * gxy[sl])
            corr[sl] += w * model
            wtot[sl] += w
            row["used"] = True
        rows.append(row)
    return corr.astype(np.float32), rows


def subtract_continuum(
    linecont_path: Path,
    out_net_path: Path,
    *,
    mul_cont: float = 1.0,
    auto_scale_from_stars: bool = False,
    stars_xy_path: Path | None = None,
    win: int = 15,
    seeing_lim: float = 0.25,
    write_line_cont: bool = True,
    out_line_path: Path | None = None,
    local_mu_mode: str = "global",
    local_mu_nbins: int = 12,
    local_mu_smooth_bins: int = 1,
    write_kmap: bool = True,
    write_star_table: bool = True,
    scale_sigma_clip: float = 2.5,
    write_cont: bool = True,
    write_plot: bool = True,
    line_layer: int = 0,
    cont_layer: int | None = 1,
    cont_image: np.ndarray | None = None,
    cont_texp: float | None = None,
    register_cont: bool = True,
    star_match: bool = False,
    star_protect: tuple[float, float, float] | None = None,
) -> tuple[Path, Path, Path, dict]:
    """
    IDL reference: cre_emis_map.pro (continuum scaling via stars + optional PSF match)

    Input:  step6 multi-layer cube (layer table in the header; line_layer / cont_layer pick the pair, or cont_image is used as the continuum) (counts/sec)
    Output: net = line_sum - k * cont_sum (2D), where:
            - if auto_scale_from_stars: k = mul_cont * median(Flux_line / Flux_cont) from star fits,
              after optional PSF matching if |FWHM_cont/FWHM_line - 1| > seeing_lim
            - else: k = mul_cont

    Returns: (line2d_path, cont2d_path, net2d_path, info_dict)
    """
    linecont_path = Path(linecont_path)
    out_net_path = Path(out_net_path)

    cube, hdr = fits.getdata(linecont_path, header=True)
    if cube.ndim != 3 or cube.shape[0] < 2:
        raise ValueError(f"Expected a multi-layer cube from step6, got shape={cube.shape}")
    layers = read_layer_table(hdr, cube.shape[0])
    hdr = _hdr_for_layer(hdr, layers, line_layer, cont_layer)

    line = cube[line_layer].astype(np.float32)
    if cont_image is not None:  # combined continuum computed from several continuum layers
        cont = np.asarray(cont_image, dtype=np.float32)
        if cont_texp:
            hdr["TCONT"] = (float(cont_texp), "[s] exposure of the combined continuum")
        hdr["HISTORY"] = "STEP7: continuum = exposure-weighted mean of all continuum layers"
    else:
        cont = cube[cont_layer].astype(np.float32)

    info: dict = {"mul_cont_cli": float(mul_cont), "auto_scale_from_stars": bool(auto_scale_from_stars)}
    k = float(mul_cont)

    if auto_scale_from_stars:
        if stars_xy_path is None:
            # auto-detect star list if not provided
            if stars_xy_path is None:
                cand: list[Path] = []
                # 1) <object>_shift.stars.xy (written by the shift stage; older versions: _shift.fits.stars.xy)
                bn = base_name(linecont_path)
                cand.append(linecont_path.with_name(f"{bn}_shift.stars.xy"))
                cand.append(linecont_path.with_name(f"{bn}_shift.fits.stars.xy"))
                # 2) other spellings (legacy chained names, same stem as the input)
                cand.append(linecont_path.with_suffix(".stars.xy"))
                cand.append(linecont_path.with_name(f"cube_{bn}_darkflat_crclean_shift.fits.stars.xy"))
                cand.append(linecont_path.with_name(f"{bn}.fits.stars.xy"))
                # 3) any *.stars.xy in the same directory (pick newest)
                cand.extend(sorted(linecont_path.parent.glob("*.stars.xy"), key=lambda p: p.stat().st_mtime, reverse=True))
                for p in cand:
                    try:
                        if p is not None and Path(p).exists():
                            stars_xy_path = Path(p)
                            break
                    except Exception:
                        pass
            if stars_xy_path is None:
                raise ValueError(
                    "auto_scale_from_stars=True requires stars_xy_path. "
                    "Run the shift stage so that the star list <object>_shift.stars.xy exists, or pass --stars-xy explicitly."
                )
        k_obj, line2, cont2, info2 = continuum_scale_like_idl(
            emis=line, cont=cont, stars_xy_path=Path(stars_xy_path),
            win=int(win), seeing_lim=float(seeing_lim), mul=float(mul_cont),
            local_mode=str(local_mu_mode), local_nbins=int(local_mu_nbins),
            local_smooth_bins=int(local_mu_smooth_bins),
            scale_sigma_clip=float(scale_sigma_clip),
        )
        line = line2
        cont = cont2
        info.update(info2)
        # k_obj is either a scalar k or a 2D k-map
        k_map = None
        if isinstance(k_obj, np.ndarray):
            k_map = k_obj.astype(np.float32, copy=False)
            k = float(np.nanmedian(k_map))
        else:
            k = float(k_obj)

    k_map = locals().get("k_map")
    k_any = k_map if k_map is not None else k

    # stars: geometry (line and continuum images may differ by a fraction of a pixel) and individual flux ratios (colours)
    star_rows = []
    corr = None
    if bool(register_cont) or bool(star_match):
        try:
            sources = find_point_sources(line, cont)
            noise = _ss_robust_sigma(line - k_any * cont)
            info["n_point_sources"] = len(sources)
            matches = local_match(line, cont, sources, k, noise=noise)
            if bool(register_cont):
                cont_reg, reg = register_from_matches(cont, matches)
                info.update(reg)
                if reg.get("register") == "affine":
                    cont = cont_reg
                    hdr["HISTORY"] = (f"STEP7: continuum registered to the line image by {reg['register_nused']} stars "
                                      f"(affine, max shift {reg['register_max_shift_px']:.2f} px)")
                    matches = local_match(line, cont, sources, k, noise=noise)
            if bool(star_match):
                corr, star_rows = per_star_correction(cont, matches, k_any, protect=star_protect)
                info["star_match_used"] = int(sum(r["used"] for r in star_rows))
                info["star_match_total"] = len(star_rows)
        except Exception as e:  # the plain subtraction must not die because of the star cosmetics
            print(f"[warn] star registration / per-star match skipped: {e}", flush=True)
            corr = None

    net = (line - k_any * cont).astype(np.float32)
    if corr is not None:
        net = (net - corr).astype(np.float32)

    # write outputs
    base = Path(out_net_path)
    if base.suffix.lower() != ".fits":
        base = base.with_suffix(".fits")

    # net
    hdr_net = hdr.copy()
    hdr_net["HISTORY"] = f"STEP7: net = line - {k:.6g} * cont"
    hdr_net["MULCONT"] = float(k)
    if auto_scale_from_stars:
        hdr_net["HISTORY"] = f"STEP7: k from stars (cre_emis_map-like), win={win}, seeing_lim={seeing_lim}"
    if ("k_map" in locals() and k_map is not None):
        hdr_net["HISTORY"] = f"STEP7: spatially varying k-map (mode={local_mu_mode}, nbins={local_mu_nbins}, smooth={local_mu_smooth_bins})"
    if corr is not None:
        hdr_net["HISTORY"] = (f"STEP7: per-star match: {info['star_match_used']} point sources subtracted with their own "
                              f"flux ratio / shift / PSF width (cosmetic)")
        hdr_net["STARMTCH"] = (int(info["star_match_used"]), "point sources subtracted individually")
    _stamped_writeto(base, net, hdr_net, overwrite=True)
    if star_rows:
        sm_path = base.with_name(strip_kind(base.stem, "net") + "_star_match.csv")
        cols = ["x", "y", "flux_cont", "ratio_gauss", "a", "k_global", "dx", "dy", "fwhm", "snr", "radius", "fit", "used", "why"]
        with open(sm_path, "w", newline="") as fcsv:
            wr = csv.DictWriter(fcsv, fieldnames=cols)
            wr.writeheader()
            for row in star_rows:
                wr.writerow({c: (f"{row[c]:.5g}" if isinstance(row[c], float) else row[c]) for c in cols})
        info["star_match_table_path"] = str(sm_path)

    # optional k-map product
    if ("k_map" in locals() and k_map is not None) and bool(write_kmap):
        kmap_path = base.with_name(strip_kind(base.stem, "net") + "_kmap.fits")
        hdr_k = hdr.copy()
        hdr_k["HISTORY"] = f"STEP7: k-map used for cont scaling (mode={local_mu_mode})"
        _stamped_writeto(kmap_path, k_map.astype(np.float32), hdr_k, overwrite=True)
        info["kmap_path"] = str(kmap_path)

    # optional per-star diagnostics table
    if bool(write_star_table) and info.get("star_ratio_rows"):
        table_path = base.with_name(strip_kind(base.stem, "net") + "_star_ratios.csv")
        fieldnames = ["star_id", "x", "y", "flux_emis", "flux_cont", "ratio", "used", "used_in_fit", "k_local", "ratio_minus_kdivmul"]
        with open(table_path, "w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            for row in info["star_ratio_rows"]:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        info["star_ratio_table_path"] = str(table_path)

    # cont/line products (for astrometry and QA)
    line_path = Path(out_line_path) if out_line_path else base.with_name(strip_kind(base.stem, "net") + "_line.fits")
    cont_path = base.with_name(strip_kind(base.stem, "net") + "_cont.fits")

    hdr_c = hdr.copy()
    hdr_c["HISTORY"] = "STEP7: extracted cont_sum layer from step6 (after optional PSF match)"
    hdr_c["MULCONT"] = float(k)
    if write_cont:
        _stamped_writeto(cont_path, cont, hdr_c, overwrite=True)

    if write_line_cont:
        hdr_l = hdr.copy()
        hdr_l["HISTORY"] = "STEP7: extracted line_sum layer from step6 (after optional PSF match)"
        hdr_l["MULCONT"] = float(k)
        _stamped_writeto(line_path, line, hdr_l, overwrite=True)

    info["k_used"] = float(k)

    if write_plot:
        try:
            png = base.with_name(base.stem + "_subtract.png")
            make_subtraction_figure(line, cont, net, png, k=float(k), info=info, title=base.stem)
            info["subtract_png"] = str(png)
        except Exception as e:  # a QA picture must never break the reduction
            print(f"[warn] subtraction plot failed: {e}")
    return line_path, cont_path, base, info


def _rotate_data_and_wcs_to_NE(data: np.ndarray, base_hdr, wcs_hdr: dict, outside_value: float = 0.0) -> tuple[np.ndarray, dict]:
    """Rotate the image by angle=-PA to N up / E left, update CD/CRPIX consistently, keep the SIP keywords."""
    # Merge base + WCS into a Header to compute the position angle of +Y
    merged = fits.Header(base_hdr)
    for card in fits.Header(wcs_hdr).cards:
        if card.keyword in ("HISTORY","COMMENT","CONTINUE"):
            continue
        try:
            merged[card.keyword] = card.value
        except Exception:
            pass

    pa = _pa_idl_deg_from_header(merged)
    if not np.isfinite(pa):
        pa = 0.0
    angle = -pa

    # NaN (outside the circular mask / uncovered borders) must not enter the cubic spline: its prefilter spreads a
    # single NaN over the whole image -> the rotated file was entirely NaN.  Rotate NaN-free data and the validity
    # mask separately; pixels whose interpolation touched invalid data become NaN.
    arr = np.asarray(data, dtype=np.float32)
    valid = np.isfinite(arr)
    fill = float(np.nanmedian(arr)) if valid.any() else 0.0
    rot = _nd_rotate(np.where(valid, arr, fill).astype(np.float32), angle=angle, reshape=False, order=3,
                     mode="constant", cval=fill)
    vrot = _nd_rotate(valid.astype(np.float32), angle=angle, reshape=False, order=1, mode="constant", cval=0.0)
    if valid.all():
        rot[vrot < 0.999] = float(outside_value)
    else:
        rot[vrot < 0.999] = np.nan

    # Update WCS in merged header for the rotated pixel grid
    merged2 = _rotate_wcs_cd_crpix_like_idl(merged, angle_deg=(-angle), shape=data.shape)

    out_hdr = dict(base_hdr)
    # Append WCS+SIP from merged2 (IDL-style: overwrite keys)
    out_hdr = dict(_append_wcs_sip_like_idl(_to_fits_header(out_hdr), merged2))
    out_hdr["PA_Y"] = (float(pa), "[deg] position angle of +Y before rotation (N through E)")
    out_hdr["ROT_ANG"] = (float(angle), "[deg] rotation applied to N up, E left (= -PA_Y)")
    _add_history(out_hdr, f"ROTATION: rotated by {angle:.3f} deg to N up, E left")
    return rot.astype(np.float32), out_hdr



def _to_fits_header(hdr_like) -> fits.Header:
    """Convert dict/fits.Header to fits.Header safely (supports (value, comment) tuples)."""
    if isinstance(hdr_like, fits.Header):
        return fits.Header(hdr_like)
    h = fits.Header()
    if hdr_like is None:
        return h
    if isinstance(hdr_like, dict):
        for k, v in hdr_like.items():
            if k in ('HISTORY','COMMENT'):
                # skip here; history will be handled elsewhere if needed
                continue
            try:
                if isinstance(v, tuple) and len(v) == 2:
                    h[k] = (v[0], v[1])
                else:
                    h[k] = v
            except Exception:
                pass
    return h

def solve_astrometry_net(cont_fits: Path,
                         api_key_env: str = "ASTROMETRY_NET_API_KEY",
                         ra_deg: float | None = None, dec_deg: float | None = None, radius_deg: float | None = None,
                         scale_lower: float | None = None, scale_upper: float | None = None,
                         scale_units: str = "arcsecperpix",
                         downsample_factor: int | None = None,
                         tweak_order: int = 2,
                         parity: int | None = None,
                         crpix_center: bool = False,
                         solve_timeout: int = 1200,
                         max_retries: int = 5,
                         retry_wait_s: float = 15.0,
                         save_astro_fits: bool = True,
                         astro_suffix: str = "_astro") -> tuple[fits.Header, int, Path | None]:
    """
    Solve astrometry via nova.astrometry.net API directly and (optionally) download the solved image (new_fits).

    Returns:
        (wcs_header, job_id, astro_fits_path_or_None)

    Notes:
      - We intentionally use the service-produced "new_fits_file/JOBID" as the authoritative WCS,
        because it contains a complete celestial WCS (CTYPE/CRVAL/CD/...).
      - API docs: https://astrometry.net/doc/net/api.html
    """
    import json
    import requests

    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Astrometry.net API key not found in environment variable {api_key_env}. "
            f"Export it, e.g.: export {api_key_env}='YOUR_KEY'"
        )

    base_url = "https://nova.astrometry.net"
    api_url = base_url + "/api"

    import time as _time

    def _post_json(endpoint: str, payload: dict, file_path: Path | None = None, timeout=120):
        """
        POST with retries: nova.astrometry.net sometimes drops the connection ('Connection reset by peer') or answers
        5xx; the request is repeated up to max_retries times with a growing pause.
        """
        # the API expects form field: request-json=<json-string>
        data = {"request-json": json.dumps(payload)}
        wait = float(retry_wait_s)
        for attempt in range(1, max_retries + 1):
            try:
                if file_path is not None:
                    with open(file_path, "rb") as f:  # reopened at every attempt
                        r = requests.post(api_url + endpoint, data=data, timeout=timeout,
                                          files={"file": (Path(file_path).name, f, "application/fits")})
                else:
                    r = requests.post(api_url + endpoint, data=data, timeout=timeout)
                if r.status_code >= 500:
                    raise requests.HTTPError(f"server error {r.status_code}", response=r)
                return r
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                if attempt == max_retries:
                    raise RuntimeError(f"astrometry.net {endpoint}: no answer after {max_retries} attempts "
                                       f"(last error: {type(e).__name__}: {e}). Check the internet connection "
                                       f"or try again later.") from e
                print(f"[astro] {endpoint}: {type(e).__name__} ({e}); attempt {attempt}/{max_retries} failed, "
                      f"retrying in {wait:.0f} s", flush=True)
                _time.sleep(wait)
                wait = min(wait * 2, 120.0)

    def _get(endpoint: str, timeout=120):
        return requests.get(api_url + endpoint, timeout=timeout)

    # --- login ---
    r = _post_json("/login", {"apikey": api_key})
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "success":
        raise RuntimeError(f"Astrometry.net login failed: {j}")
    session = j["session"]

    # --- upload ---
    settings: dict = {
        "session": session,
        "tweak_order": tweak_order,
        "crpix_center": bool(crpix_center),
        "publicly_visible": "y",
        "allow_commercial_use": "d",
        "allow_modifications": "d",
    }
    if scale_lower is not None and scale_upper is not None:
        settings.update({"scale_units": scale_units, "scale_lower": float(scale_lower), "scale_upper": float(scale_upper)})
    if ra_deg is not None and dec_deg is not None and radius_deg is not None:
        settings.update({"center_ra": float(ra_deg), "center_dec": float(dec_deg), "radius": float(radius_deg)})
    if downsample_factor is not None:
        settings["downsample_factor"] = int(downsample_factor)
    if parity is not None:
        settings["parity"] = int(parity)

    r = _post_json("/upload", settings, file_path=cont_fits, timeout=300)
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "success":
        raise RuntimeError(f"Astrometry.net upload failed: {j}")
    subid = int(j["subid"])

    # --- poll for job id + status ---
    start = _time.time()
    job_id: int | None = None
    status = None

    # transient network errors happen; retry GETs with backoff
    def _get_with_retry(endpoint: str):
        wait = float(retry_wait_s)
        for attempt in range(max_retries):
            try:
                rr = _get(endpoint, timeout=120)
                rr.raise_for_status()
                return rr
            except Exception:
                if attempt == max_retries - 1:
                    raise
                _time.sleep(wait)
                wait = min(wait * 2, 120.0)

    while True:
        if _time.time() - start > solve_timeout:
            raise TimeoutError(f"Astrometry solve timed out after {solve_timeout}s (submission {subid})")

        rr = _get_with_retry(f"/submissions/{subid}")
        jobs = rr.json().get("jobs") or []
        if jobs and jobs[0]:
            job_id = int(jobs[0])

        if job_id is not None:
            rr = _get_with_retry(f"/jobs/{job_id}")
            status = rr.json().get("status")

        if status in ("success", "failure"):
            break

        _time.sleep(2.0)

    if status != "success" or job_id is None:
        raise RuntimeError(f"Astrometry solve failed: status={status}, submission={subid}, job={job_id}")

    # --- download solved WCS ---
    # NOTE: nova.astrometry.net may protect new_fits_file with a human-check (HTML). Therefore:
    #   1) Try new_fits_file (best: includes TAN-SIP in FITS header)
    #   2) Always fall back to wcs_file/JOBID (text header; typically includes TAN-SIP)
    # We do NOT use /api/jobs/JOBID/calibration because it is TAN-only and will not match IDL TAN-SIP results.

    astro_path: Path | None = None
    wcs_hdr = fits.Header()

    # Try new_fits_file
    if save_astro_fits:
        astro_path = cont_fits.with_name(cont_fits.stem + astro_suffix + ".fits")
        try:
            astro_path = _download_new_fits(base_url, job_id, astro_path, timeout=300, max_retries=max_retries)
            wcs_hdr = fits.getheader(astro_path)
            if wcs_hdr.get("CTYPE1") is None:
                # not a valid celestial WCS; ignore and fall back
                wcs_hdr = fits.Header()
        except Exception:
            # new_fits_file is often blocked by human-check (HTML); ignore and fall back to wcs_file
            astro_path = None
            wcs_hdr = fits.Header()

# Mandatory fallback: wcs_file (may be returned as FITS bytes or as text cards)
    if wcs_hdr.get("CTYPE1") is None:
        wcs_url = f"{base_url}/wcs_file/{job_id}"
        rr = requests.get(wcs_url, timeout=120, allow_redirects=True)
        rr.raise_for_status()
        content = rr.content or b""

        # Case A: server returns a FITS file (starts with SIMPLE/XTENSION)
        if _is_fits_bytes(content):
            try:
                with fits.open(io.BytesIO(content)) as hdul:
                    wcs_hdr = fits.Header(hdul[0].header)
            except Exception:
                wcs_hdr = fits.Header()

        # Case B: server returns plain-text header cards
        if wcs_hdr.get("CTYPE1") is None:
            text = rr.text or content.decode("utf-8", errors="ignore")
            # Normalize newlines
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            # Some responses include leading HTML; guard quickly
            if "<html" in text.lower():
                dbg = cont_fits.with_name(cont_fits.stem + "_wcs_file_response.html")
                dbg.write_text(text, encoding="utf-8", errors="ignore")
                raise RuntimeError(f"wcs_file returned HTML (saved to {dbg})")
            # Parse
            wcs_hdr = fits.Header().fromstring(text, sep="\n")

        if wcs_hdr.get("CTYPE1") is None:
            dbg = cont_fits.with_name(cont_fits.stem + "_wcs_file_response.txt")
            try:
                dbg.write_bytes(content)
            except Exception:
                pass
            raise RuntimeError(f"Could not parse WCS from wcs_file; saved raw response to {dbg}")


    return wcs_hdr, job_id, astro_path


def _solve_wcs_for_cont(cont_fits: Path, api_key_env: str, solve_timeout: int,
                        max_retries: int, retry_wait_s: float) -> fits.Header:
    """
    WCS of a 2D continuum image from astrometry.net.  Attempts: (1) position (J2000) + scale hints, (2) scale only,
    (3) blind.  The uploaded image is a cleaned crop (_make_upload_image); the WCS is shifted back to the pixel
    grid of the original image.
    """
    cont_fits = Path(cont_fits)
    cont_hdr = fits.getheader(cont_fits)
    hints = _hints_from_header(cont_hdr)
    up = _make_upload_image(cont_fits)
    try:
        return _solve_upload(cont_fits, cont_hdr, hints, up, api_key_env, solve_timeout, max_retries, retry_wait_s)
    finally:
        shutil.rmtree(up.parent, ignore_errors=True)


def _solve_upload(cont_fits: Path, cont_hdr, hints, up: Path, api_key_env: str, solve_timeout: int,
                  max_retries: int, retry_wait_s: float) -> fits.Header:
    uh = fits.getheader(up)
    dx, dy = int(uh.get("UPX0", 0)), int(uh.get("UPY0", 0))
    if hints.ra_deg is not None:
        print(f"[astro] hint: RA={hints.ra_deg:.4f} Dec={hints.dec_deg:.4f} (J2000), radius {hints.radius_deg} deg; "
              f"scale {hints.scale_lower}-{hints.scale_upper} arcsec/px; upload {up.name} (offset {dx},{dy})", flush=True)
    attempts = []
    if hints.ra_deg is not None and hints.scale_lower is not None:
        attempts.append(("position + scale", dict(ra_deg=hints.ra_deg, dec_deg=hints.dec_deg, radius_deg=hints.radius_deg,
                                                   scale_lower=hints.scale_lower, scale_upper=hints.scale_upper)))
    if hints.scale_lower is not None:
        attempts.append(("scale only", dict(scale_lower=hints.scale_lower, scale_upper=hints.scale_upper)))
    attempts.append(("blind", dict()))
    last = None
    for what, kw in attempts:
        try:
            wcs_hdr, job_id, _ = solve_astrometry_net(cont_fits=up, api_key_env=api_key_env, solve_timeout=solve_timeout,
                                                      max_retries=max_retries, retry_wait_s=retry_wait_s,
                                                      save_astro_fits=False, **kw)
            print(f"[astro] solved ({what}), job {job_id}", flush=True)
            break
        except (RuntimeError, TimeoutError) as e:
            print(f"[astro] attempt '{what}' failed: {e}", flush=True)
            last = e
    else:
        raise RuntimeError(f"astrometry.net could not solve {cont_fits.name} (last error: {last}). "
                           f"Enough stars in {cont_fits.name}? Hint RA/Dec from the header: {cont_hdr.get('RA')} {cont_hdr.get('DEC')} "
                           f"EPOCH {cont_hdr.get('EPOCH')}")
    wcs_hdr = fits.Header(wcs_hdr)
    for key in ("CRPIX1", "CRPIX2"):
        if key in wcs_hdr:
            wcs_hdr[key] = float(wcs_hdr[key]) + (dx if key == "CRPIX1" else dy)
    for key in ("IMAGEW", "IMAGEH"):
        if key in wcs_hdr:
            del wcs_hdr[key]
    return wcs_hdr



def _write_NE_product(in_f: Path, wcs_hdr, out_suffix: str = "_NE", wmed: int = 1) -> Path:
    """Rotate a 2D image to N-up/E-left with the given WCS and write <in_f.stem><out_suffix>.fits."""
    in_f = Path(in_f)
    data, hdr = fits.getdata(in_f, header=True)
    arr = np.squeeze(np.asarray(data))
    if arr.ndim != 2:
        raise RuntimeError(f"Astrometry expects a 2D image, got shape {tuple(np.shape(data))} -> {tuple(arr.shape)} from {in_f}")
    arr = arr.astype(np.float32, copy=False)
    rot, hdr_out = _rotate_data_and_wcs_to_NE(arr, hdr, wcs_hdr, outside_value=0.0)
    if wmed and int(wmed) > 2:
        rot = _nd_median_filter(rot, size=int(wmed), mode="nearest")
    hdr_final = _append_sip_only_like_idl(_to_fits_header(hdr_out), _to_fits_header(wcs_hdr))
    hdr_final = _sanitize_header_for_2d_write(hdr_final, tuple(rot.shape))
    outp = in_f.with_name(in_f.stem + out_suffix + ".fits")
    _stamped_writeto(outp, rot, hdr_final, overwrite=True, output_verify='silentfix')
    return outp


# --------------------------------------------------------------------------------------
# Multi-layer line / continuum cube (step6)
# --------------------------------------------------------------------------------------

def extract_layer_products(linecont_path: Path) -> dict:
    """
    Read the layer table of <obj>_linecont.fits and plan the 2D products:
      {"base", "cont", "cont_layer", "conts": [(name, layer)], "lines": [{"tag","layer","line","net"}]}
    Output names: <obj>_cont, <obj>_cont1.., <obj>_<tag> (line; <obj>_line without tag), <obj>[_<tag>]_net (.fits).
    """
    linecont_path = Path(linecont_path)
    hdr = fits.getheader(linecont_path)
    nl = int(hdr.get("NAXIS3", 0) or 0)
    layers = read_layer_table(hdr, nl or None)
    plan = plan_layers(layers)
    base = base_name(linecont_path)
    d = linecont_path.parent
    prods = {
        "base": base,
        "cont": product_path(d, base, "cont"),
        "cont_layer": plan["cont_total"],
        "conts": [(n, i) for n, i in plan["conts"]],
        "lines": [],
        "layers": layers,
    }
    for tag, idx in plan["lines"]:
        prods["lines"].append({"tag": tag, "layer": idx,
                               "line": product_path(d, base, "line", tag),
                               "net": product_path(d, base, "net", tag)})
    return prods


def combined_continuum(linecont_path: Path, prods: dict) -> tuple[np.ndarray, float]:
    """
    The combined continuum image (counts/s) and its total exposure [s]: the layer itself if the cube has
    a ready one, otherwise the exposure-weighted mean of all continuum layers
    (= sum of all continuum frames / total exposure).
    """
    cube = fits.getdata(Path(linecont_path))
    layers = prods["layers"]
    if prods["cont_layer"] is not None:
        i = prods["cont_layer"]
        t = layers[i].get("texp") if i < len(layers) else None
        return cube[i].astype(np.float32), float(t or 0.0)
    idx = [i for _n, i in prods["conts"]]
    if not idx:
        raise ValueError(f"{linecont_path}: no continuum layer in the cube")
    ts = []
    for i in idx:
        t = layers[i].get("texp")
        try:
            t = float(t)
        except (TypeError, ValueError):
            t = float("nan")
        ts.append(t if np.isfinite(t) and t > 0 else 1.0)  # no exposure info: plain mean
    w = np.asarray(ts, dtype=np.float64)
    acc = np.zeros(cube.shape[1:], dtype=np.float64)
    for wi, i in zip(w, idx):
        acc += wi * cube[i].astype(np.float64)
    return (acc / w.sum()).astype(np.float32), float(w.sum())


def write_cont_products(linecont_path: Path, prods: dict, *, write_total: bool = True,
                        overwrite: bool = False) -> list[Path]:
    """
    Write the continuum images of the cube: <obj>_cont.fits (combined; only if write_total) and the
    individual <obj>_cont1.fits, <obj>_cont2.fits ... Existing files are kept unless overwrite=True.
    Returns the individual continuum paths (empty when the cube has a single continuum).
    """
    linecont_path = Path(linecont_path)
    cube, hdr = fits.getdata(linecont_path, header=True)
    outs: list[Path] = []
    if write_total and (overwrite or not Path(prods["cont"]).exists()):
        img, texp = combined_continuum(linecont_path, prods)
        if prods["cont_layer"] is not None:
            h = _hdr_for_layer(hdr, prods["layers"], prods["cont_layer"])
        else:  # combined from several continuum layers: exposure / frames / ZMEAN of the whole set
            idx = [i for _n, i in prods["conts"]]
            h = _hdr_for_layer(hdr, prods["layers"], idx[0])
            h["SRCLAYER"] = ("+".join(prods["layers"][i]["name"] for i in idx)[:60], "layers of the step6 cube")
            t_all, n_all, zm_all = _combined_cont_stats(prods["layers"], idx)
            h["EXPTIME"] = (t_all, "Exposure time [s] of this image")
            h["NFRAMES"] = (n_all, "Number of frames summed")
            if zm_all:
                h["ZMEAN"] = (round(zm_all, 3), "Mean zenith distance [deg], airmass weighted")
            elif "ZMEAN" in h:
                del h["ZMEAN"]
            zall = [z for i in idx for z in _layer_zspaces(prods["layers"][i])]
            if "ZSPACE" in h and len(set(zall)) > 1:
                del h["ZSPACE"]
        if texp:
            h["TCONT"] = (float(texp), "[s] exposure of the combined continuum")
        h["HISTORY"] = "STEP7: combined continuum from the step6 cube"
        _stamped_writeto(prods["cont"], img, h, overwrite=True)
    for n, (name, idx) in enumerate(prods["conts"], 1):
        outp = product_path(linecont_path.parent, prods["base"], f"cont{n}")
        if overwrite or not outp.exists():
            h2 = _hdr_for_layer(hdr, prods["layers"], idx)
            h2["HISTORY"] = f"STEP7: individual continuum {name}"
            _stamped_writeto(outp, cube[idx].astype(np.float32), h2, overwrite=True)
        outs.append(outp)
    return outs


def subtract_continuum_all(
    linecont_path: Path,
    prods: dict,
    *,
    mul_list: list[float],
    **kwargs,
) -> dict:
    """
    For every emission line of the cube run subtract_continuum() against the combined continuum.
    The j-th line (layer order) uses mul_list[j] (the last value is reused if the list is shorter).
    Writes <obj>_cont.fits once (raw combined continuum), every <obj>_<tag>.fits (line) / <obj>_<tag>_net.fits, and the individual
    continua <obj>_cont1.fits, <obj>_cont2.fits ... when the cube has them.

    Returns {"lines": [{"tag","line","net","info","mul"}...], "cont": Path, "conts": [Path...]}
    """
    kwargs.pop("write_line_cont", None)
    kwargs.pop("write_cont", None)
    kwargs.pop("line_layer", None)
    kwargs.pop("cont_layer", None)
    kwargs.pop("cont_image", None)
    linecont_path = Path(linecont_path)
    res: dict = {"lines": [], "cont": None, "conts": []}
    cont_img = cont_t = None
    if prods["cont_layer"] is None:  # several continua and no ready combined one
        cont_img, cont_t = combined_continuum(linecont_path, prods)
    for j, L in enumerate(prods["lines"]):
        mul = float(mul_list[min(j, len(mul_list) - 1)]) if mul_list else 1.0
        _l, cont2d, net2d, info = subtract_continuum(
            linecont_path, out_net_path=L["net"], out_line_path=L["line"], mul_cont=mul,
            write_line_cont=True, write_cont=False,
            line_layer=L["layer"], cont_layer=prods["cont_layer"],
            cont_image=cont_img, cont_texp=cont_t, **kwargs)
        res["lines"].append({"tag": L["tag"], "line": L["line"], "net": net2d, "info": info, "mul": mul})
    # the combined continuum as it is in the cube (reference image for astrometry)
    res["conts"] = write_cont_products(linecont_path, prods, write_total=True, overwrite=True)
    res["cont"] = prods["cont"]
    return res


def apply_wcs_to_group_products(
    cont_ref: Path,
    *,
    cont_groups: list[Path] = (),
    net_fits: list[Path] = (),
    line_fits: list[Path] = (),
    out_suffix: str = "_NE",
    api_key_env: str = "ASTROMETRY_NET_API_KEY",
    solve_timeout: int = 1200,
    max_retries: int = 5,
    retry_wait_s: float = 15.0,
    wmed: int = 1,
) -> dict[str, Path]:
    """
    Astrometry for a set of products.
      * cont_ref (combined continuum, best S/N) is solved first; its WCS is applied to every
        line/net image, so all line and net maps share ONE pixel grid (needed for line ratios).
      * every individual continuum in cont_groups is solved on its own and written with its own WCS;
        if its solve fails the reference WCS is used instead (with a warning).
    Output keys: cont_NE, net_NE[_<tag>], line_NE[_<tag>], cont1_NE, cont2_NE ...
    """
    cont_ref = Path(cont_ref)
    ref_wcs = _solve_wcs_for_cont(cont_ref, api_key_env, solve_timeout, max_retries, retry_wait_s)
    out: dict[str, Path] = {"cont_NE": _write_NE_product(cont_ref, ref_wcs, out_suffix, wmed)}
    base = strip_kind(cont_ref.stem, "cont")

    def _key(prefix: str, tag: str) -> str:
        return f"{prefix}_{tag}" if tag else prefix

    for p in net_fits:
        if Path(p).exists():
            tag, _kind = split_product_stem(Path(p).stem, base)
            out[_key("net_NE", tag)] = _write_NE_product(Path(p), ref_wcs, out_suffix, wmed)
    for p in line_fits:
        if Path(p).exists():
            out[_key("line_NE", line_tag_from_stem(Path(p).stem, base))] = _write_NE_product(Path(p), ref_wcs, out_suffix, wmed)

    for p in cont_groups:
        p = Path(p)
        try:
            w = _solve_wcs_for_cont(p, api_key_env, solve_timeout, max_retries, retry_wait_s)
        except Exception as e:  # keep going: one failed continuum must not lose the other products
            print(f"[warn] astrometry failed for {p.name}: {e}; using the reference WCS from {cont_ref.name}")
            w = ref_wcs
        out[f"{p.stem.rsplit('_', 1)[-1]}_NE"] = _write_NE_product(p, w, out_suffix, wmed)
    return out


def find_astro_inputs(w_dir: Path, object_name: str | None = None, base: str | None = None) -> dict:
    """
    Locate astrometry inputs in w_dir WITHOUT any earlier pipeline stage: only the 2D products
    <base>_cont.fits (combined continuum), <base>_cont1/2.fits (individual continua),
    <base>[_<line>]_net.fits and the line images <base>_<line>.fits (older runs: <base>[_<line>]_line.fits).
    `base` (object name) may be given; otherwise the newest <obj>_cont.fits (matching object_name,
    if given) defines it.  Returns {"base", "cont_ref", "cont_groups", "net", "line"}.
    """
    import glob as _glob
    w_dir = Path(w_dir)
    if base:
        pat = _glob.escape(base)
    elif object_name:
        pat = f"*{_glob.escape(object_name)}*"
    else:
        pat = "*"

    def _ok(p: Path) -> bool:
        st = p.stem
        return not (st.endswith("_NE") or "_NE_" in st or "_erg" in st or "_astro" in st
                    or st.endswith("_kmap"))

    conts = [p for p in w_dir.glob(pat + "_cont.fits") if _ok(p)]
    if not conts:  # no combined continuum: fall back to an individual one (cont1, ...)
        conts = [p for p in w_dir.glob(pat + "_cont[0-9].fits") if _ok(p)]
    if not conts:
        raise FileNotFoundError(
            f"No continuum image (*_cont.fits) found in {w_dir}"
            + (f" for object {base or object_name}" if (base or object_name) else "")
            + ". Astrometry needs only the continuum (and optionally line / net) FITS files.")
    ref = max(conts, key=lambda p: p.stat().st_mtime)
    if ref.stem.endswith("_cont"):
        base = ref.stem[: -len("_cont")]
    else:  # individual continuum used as reference
        base = ref.stem.rsplit("_cont", 1)[0]

    def _find(suffix: str) -> list[Path]:
        found = [p for p in w_dir.glob(f"{_glob.escape(base)}_*{suffix}.fits") if _ok(p)]
        plain = w_dir / f"{base}{suffix}.fits"
        if plain.exists():
            found.append(plain)
        return sorted(set(found))

    cont_groups = sorted(p for p in w_dir.glob(f"{_glob.escape(base)}_cont[0-9]*.fits") if _ok(p))
    nets = _find("_net")
    # line images belong to the net images: <base>_<tag>.fits, or <base>[_<tag>]_line.fits of older runs
    lines = []
    for n in nets:
        tag, _kind = split_product_stem(n.stem, base)
        cands = [p for p in (product_path(w_dir, base, "line", tag), legacy_line_path(w_dir, base, tag)) if p.exists()]
        if cands:
            lines.append(max(cands, key=lambda p: p.stat().st_mtime))
    return {"base": base, "cont_ref": ref, "cont_groups": cont_groups, "net": nets, "line": sorted(set(lines))}


def run_astrometry_standalone(w_dir: Path, object_name: str | None = None, base: str | None = None, **kw) -> dict[str, Path]:
    """Astrometry for products already present in w_dir (see find_astro_inputs)."""
    inp = find_astro_inputs(w_dir, object_name, base)
    ref = inp["cont_ref"]
    groups = [p for p in inp["cont_groups"] if p != ref]
    print(f"[astro] continuum (reference): {ref.name}")
    for p in groups:
        print(f"[astro] individual continuum: {p.name}")
    for p in inp["net"]:
        print(f"[astro] net:  {p.name}")
    for p in inp["line"]:
        print(f"[astro] line: {p.name}")
    return apply_wcs_to_group_products(ref, cont_groups=groups, net_fits=inp["net"], line_fits=inp["line"], **kw)


def final_cr_clean_products(prod: dict[str, Path],
                            sigclip: float = 5.0,
                            sigfrac: float = 0.3,
                            objlim: float = 6.0,
                            niter: int = 4,
                            cleantype: str = "medmask",
                            gain: float = 1.0,
                            readnoise: float = 6.5) -> None:
    """IDL-like: run a final L.A.Cosmic pass on the last products (net NE images, and line NE images if present),
    overwriting the same files (no new outputs)."""
    try:
        import astroscrappy
    except Exception as e:
        raise RuntimeError("astroscrappy is required for --cr-final") from e

    def _clean_one(path: Path) -> None:
        data, hdr = fits.getdata(path, header=True)
        if data is None:
            return
        import numpy as np
        if data.ndim != 2:
            out = np.empty_like(data)
            for i in range(data.shape[0]):
                _, clean = astroscrappy.detect_cosmics(
                    data[i].astype("float32"),
                    sigclip=float(sigclip), sigfrac=float(sigfrac), objlim=float(objlim),
                    niter=int(niter), cleantype=str(cleantype),
                    gain=float(gain), readnoise=float(readnoise),
                )
                out[i] = clean
            _stamped_writeto(path, out, hdr, overwrite=True)
            try:
                hdr.add_history(f"STEP7: final L.A.Cosmic on cube (sigclip={sigclip}, niter={niter}) overwrite")
            except Exception:
                pass
            return

        _, clean = astroscrappy.detect_cosmics(
            data.astype("float32"),
            sigclip=float(sigclip), sigfrac=float(sigfrac), objlim=float(objlim),
            niter=int(niter), cleantype=str(cleantype),
            gain=float(gain), readnoise=float(readnoise),
        )
        _stamped_writeto(path, clean, hdr, overwrite=True)
        try:
            hdr.add_history(f"STEP7: final L.A.Cosmic (sigclip={sigclip}, niter={niter}) overwrite")
        except Exception:
            pass

    for key in [k for k in prod if k.startswith(("net_NE", "line_NE"))]:
        p = prod.get(key)
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            _clean_one(p)
            print(f"[OK] final CR-clean overwrite: {p}")
