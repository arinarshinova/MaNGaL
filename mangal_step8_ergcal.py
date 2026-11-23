
from __future__ import annotations

from pathlib import Path
import io
import os
import json
import zipfile
import re

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

from mangal_header import stamp
from mangal_names import find_side

# filter curves and standard-star SEDs shipped with the pipeline (used when none are found near the data)
DATA_DIR = Path(__file__).resolve().parent / "data"
from mangal_step5_photometry import collapse_cube_median, _extract_stamp, _fit_star_moffat


def lorentz_profile(lam: np.ndarray, params: list[float]) -> np.ndarray:
    amp, lam0, fw = [float(x) for x in params]
    gamma = fw / 2.0
    return amp / (1.0 + ((lam - lam0) / gamma) ** 2)


def parse_calib_file(calib_file: str | Path) -> list[dict]:
    rows = []
    for raw in Path(calib_file).read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        rows.append({
            "filter": parts[0],
            "A": float(parts[1]),
            "B": float(parts[2]),
            "n": int(parts[3]),
            "cwl": float(parts[4]),
        })
    if not rows:
        raise ValueError(f"No calibration rows found in {calib_file}")
    return rows


def idl_filter_cwl(filter_name: str) -> float:
    f = str(filter_name).strip().upper()
    if not f:
        return 0.0
    if f.startswith("S") and len(f) >= 6:
        try:
            return float(f[3:6]) * 10.0
        except Exception:
            return 0.0
    if f.startswith("M") and len(f) >= 5:
        try:
            return float(f[2:5]) * 10.0
        except Exception:
            return 0.0
    return 0.0


def mangal_z_to_lambda(calib_file: str | Path, filter_name: str, z: int | float):
    cal = parse_calib_file(calib_file)
    cwl = idl_filter_cwl(filter_name)
    diffs = [abs(cwl - r["cwl"]) for r in cal]
    k = int(np.argmin(diffs))
    row = cal[k]
    lam = (row["A"] + row["B"] * float(z)) / row["n"]
    return float(lam), k, cal, float(cwl), row


def _find_calib_file(r_dir: Path, override: str | Path | None = None) -> Path:
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = r_dir / p
        if p.exists():
            return p
        raise FileNotFoundError(f"calib file not found: {p}")
    cands = sorted(r_dir.glob("*.calib"))
    if not cands:
        raise FileNotFoundError(f"No *.calib found in {r_dir}")
    return cands[0]


def _find_filters_dir(r_dir: Path, override: str | Path | None = None) -> Path:
    probes = []
    if override:
        probes.append(Path(override).expanduser())
    env = os.environ.get("MANGAL_FILTERS_DIR")
    if env:
        probes.append(Path(env).expanduser())
    probes += [
        r_dir / "FIltersBTA",
        r_dir / "FiltersBTA",
        r_dir.parent / "FIltersBTA",
        r_dir.parent / "FiltersBTA",
        DATA_DIR / "FiltersBTA",
    ]
    for p in probes:
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Filter profiles directory was not found (also not in {DATA_DIR / 'FiltersBTA'}). "
                            "Set --filters-dir or MANGAL_FILTERS_DIR.")


def _read_filter_profile(filters_dir: Path, filter_name: str) -> tuple[np.ndarray, np.ndarray]:
    p = filters_dir / f"{filter_name}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Filter profile not found: {p}")
    arr = np.loadtxt(p, unpack=True)
    return np.asarray(arr[0], dtype=float), np.asarray(arr[1], dtype=float)


def _find_okestan_sed_path(night_dir: Path, star_name: str, override: str | Path | None = None):
    norm = re.sub(r"[^A-Za-z0-9]+", "", str(star_name)).upper()
    probes = []
    if override:
        probes.append(Path(override).expanduser())
    env = os.environ.get("MANGAL_OKESTAN_DIR")
    if env:
        probes.append(Path(env).expanduser())
    probes += [night_dir / "okestan", night_dir.parent / "okestan", night_dir / "okestan.zip", night_dir.parent / "okestan.zip",
               DATA_DIR / "okestan"]

    for p in probes:
        if p.is_dir():
            for f in p.glob("m*.dat"):
                stem = re.sub(r"[^A-Za-z0-9]+", "", f.stem[1:]).upper()
                if stem == norm:
                    return f
        elif p.suffix.lower() == ".zip" and p.exists():
            with zipfile.ZipFile(p, "r") as z:
                for name in z.namelist():
                    base = Path(name).name
                    if not base.lower().startswith("m") or not base.lower().endswith(".dat"):
                        continue
                    stem = re.sub(r"[^A-Za-z0-9]+", "", Path(base).stem[1:]).upper()
                    if stem == norm:
                        return (p, name)
    raise FileNotFoundError(f"Could not find okestan SED for {star_name}")


def _read_okestan_sed(night_dir: Path, star_name: str, override: str | Path | None = None):
    p = _find_okestan_sed_path(night_dir, star_name, override=override)
    if isinstance(p, tuple):
        zpath, inner = p
        with zipfile.ZipFile(zpath, "r") as z:
            data = z.read(inner).decode("utf-8", errors="ignore")
        arr = np.loadtxt(io.StringIO(data), unpack=True)
    else:
        arr = np.loadtxt(p, unpack=True)
    lam = np.asarray(arr[0], dtype=float)
    mag = np.asarray(arr[1], dtype=float)
    return lam, mag


def _select_center(cube: np.ndarray, raut: float = 0.1) -> tuple[float, float]:
    img = collapse_cube_median(cube)
    ny, nx = img.shape
    lx = int(round(min(nx, ny) * float(raut)))
    x1 = nx // 2
    y1 = ny // 2
    xs0 = max(0, x1 - lx); xs1 = min(nx, x1 + lx + 1)
    ys0 = max(0, y1 - lx); ys1 = min(ny, y1 + lx + 1)
    sub = img[ys0:ys1, xs0:xs1]
    mp = int(np.nanargmax(sub))
    yy, xx = np.unravel_index(mp, sub.shape)
    xc = xs0 + xx
    yc = ys0 + yy
    return float(xc), float(yc)


def _read_sidecar(source_cube_path: str | Path) -> pd.DataFrame:
    p = find_side(source_cube_path, ".inputs.csv")
    if not p.exists():
        raise FileNotFoundError(f"inputs sidecar not found: {p}")
    return pd.read_csv(p)


def _filter_meta_for_phot(sidecar: pd.DataFrame, phot_table_path: str | Path | None, nslice: int) -> pd.DataFrame:
    df = sidecar.copy()
    if phot_table_path is not None and Path(phot_table_path).exists():
        dphot = pd.read_csv(phot_table_path)
        col = "good" if "good" in dphot.columns else ("GOOD" if "GOOD" in dphot.columns else None)
        if col and len(dphot) == len(df):
            mask = dphot[col].astype(bool).to_numpy()
            df = df.loc[mask].reset_index(drop=True)
    if len(df) != nslice:
        raise ValueError(f"Phot cube nslice={nslice} but filtered sidecar has {len(df)} rows")
    return df


def _measure_slice_flux(img: np.ndarray, xc: float, yc: float, win: int = 20, nosky: bool = True):
    stamp, xi, yi = _extract_stamp(img, xc, yc, win=win)
    ok, met = _fit_star_moffat(stamp)
    if not ok:
        raise RuntimeError("Bad star fitting")
    xfw = float(met["fwhm_x"])
    yc_stamp = float(met["y0"])
    xc_stamp = float(met["x0"])
    h, w = stamp.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dis = np.hypot(xx - xc_stamp, yy - yc_stamp)
    r_ap = np.array([3, 3.5, 4, 4.5, 5, 5.5], dtype=float) * xfw
    sky = 0.0
    if not nosky:
        rec = (dis < r_ap[-1]) & (dis > r_ap[-2])
        if np.any(rec):
            sky = float(np.nanmedian(stamp[rec]))
    stamp2 = stamp - sky
    rec = dis < r_ap[4]
    fl_ap = float(np.nansum(stamp2[rec]))
    return {
        "flux_ap": fl_ap,
        "xfw": xfw,
        "sky": sky,
        "fit": met,
    }



def _aggregate_by_lambda(df, cwl):
    dfa = df.copy()
    dfa["lambda_key"] = dfa["lambda0"].round().astype(int)
    g = dfa.groupby("lambda_key", as_index=False).agg(
        lambda0=("lambda0","mean"),
        dqe=("dqe","mean"),
        mul_e_to_erg=("mul_e_to_erg","mean"),
        mul_adu_to_erg=("mul_adu_to_erg","mean"),
        n=("lambda0","size")
    ).sort_values("lambda0").reset_index(drop=True)

    # choose line λ closest to CWL
    dist = abs(g["lambda0"]-cwl)
    i_line = dist.idxmin()

    if len(g)>1:
        i_cont = dist.sort_values().index[1]
    else:
        i_cont = i_line

    line = g.loc[i_line]
    cont = g.loc[i_cont]

    return g,{
        "line_mul_adu_to_erg":float(line["mul_adu_to_erg"]),
        "cont_mul_adu_to_erg":float(cont["mul_adu_to_erg"]),
        "line_qe":float(line["dqe"]),
        "cont_qe":float(cont["dqe"]),
        "line_lambda":float(line["lambda0"]),
        "cont_lambda":float(cont["lambda0"])
    }


def compute_qe_from_standard(
    *,
    phot_cube_path: str | Path,
    source_cube_path: str | Path,
    r_dir: str | Path,
    standard_name: str,
    filter_name: str,
    phot_table_path: str | Path | None = None,
    calib_file: str | Path | None = None,
    filters_dir: str | Path | None = None,
    okestan_dir: str | Path | None = None,
    fw: float = 13.8,
    win: int = 20,
    auto_center: bool = True,
    raut: float = 0.1,
    nosky: bool = True,
) -> dict:
    r_dir = Path(r_dir)
    phot_cube_path = Path(phot_cube_path)
    source_cube_path = Path(source_cube_path)

    calib_file = _find_calib_file(r_dir, calib_file)
    filters_dir = _find_filters_dir(r_dir, filters_dir)

    data, hdr = fits.getdata(phot_cube_path, header=True)
    cube = np.asarray(data, dtype=np.float32)
    if cube.ndim == 2:
        cube = cube[None, :, :]

    meta_full = _read_sidecar(source_cube_path)
    meta = _filter_meta_for_phot(meta_full, phot_table_path, cube.shape[0])

    lam_sed, mag_ab = _read_okestan_sed(r_dir, standard_name, override=okestan_dir)
    lam_filt, trans_filt = _read_filter_profile(filters_dir, filter_name)

    xc, yc = _select_center(cube, raut=raut) if auto_center else ((cube.shape[2]-1)/2.0, (cube.shape[1]-1)/2.0)

    rows = []
    fig, ax = plt.subplots(figsize=(6, 7), dpi=200)
    first_done = False

    for i in range(cube.shape[0]):
        row = meta.iloc[i]
        zspace = int(row["ZSPACE"])
        raw_file = str(row["FILE"])
        raw_path = r_dir / f"{raw_file}.fts"
        if not raw_path.exists():
            # fallback to header from cube if original raw is not present
            hraw = hdr
        else:
            hraw = fits.getheader(raw_path)

        lam0, kk, calib_rows, cwl, calib_row = mangal_z_to_lambda(calib_file, filter_name, zspace)

        dl = 1.0
        dw = 300.0
        lam = np.arange(cwl - dw, cwl + dw, dl, dtype=float)

        prf = np.interp(lam, lam_filt, trans_filt, left=0.0, right=0.0)
        if np.nanmax(prf) > 0:
            prf = prf / np.nanmax(prf)

        ifp = lorentz_profile(lam, [1.0, lam0, fw])

        fnu = 10.0 ** (-0.4 * (48.59 + mag_ab) + 23.0)
        flam = fnu / (3.34e4 * lam_sed**2)
        fint_star = np.interp(lam, lam_sed, flam)
        fstar = np.sum(prf * fint_star * ifp) * dl

        meas = _measure_slice_flux(cube[i], xc, yc, win=win, nosky=nosky)
        texp = float(hraw.get("EXPTIME", hraw.get("EXPOSURE", row.get("EXPTIME", 1.0))))
        gain = float(hraw.get("GAIN", 1.0))
        creator = str(hraw.get("CREATOR", ""))
        if creator == "qhy4040serv":
            gain = 1.0
        zdist = float(hraw.get("ZDIST", 0.0))

        a = 0.012
        c = 0.12
        extin = (a * 1.0 / ((lam0 / 10000.0) ** 4.0) + c) * 2.5 * np.log10(np.e)
        extin = extin / np.cos(np.deg2rad(zdist))

        star_obs_e_per_s = meas["flux_ap"] * gain * (10.0 ** (0.4 * extin)) / max(texp, 1e-9)

        r1 = 125.0
        r2 = np.sqrt(0.17) * r1
        S = np.pi * (r1**2 - r2**2)
        N_tab = 948.0 * S * (10.0 ** (-0.4 * mag_ab))
        N_tab = N_tab * (5500.0 / lam_sed) ** 2
        fint_e = np.interp(lam, lam_sed, N_tab)
        f_tab = np.sum(prf * fint_e * ifp) * dl

        mul_e = fstar / star_obs_e_per_s
        dqe = 100.0 * star_obs_e_per_s / f_tab
        mul_adu = mul_e * gain

        rows.append({
            "slice": i,
            "raw_file": raw_file,
            "zspace": zspace,
            "lambda0": lam0,
            "cwl": cwl,
            "filter": filter_name,
            "texp": texp,
            "gain": gain,
            "zdist": zdist,
            "xfw": meas["xfw"],
            "flux_ap_adu": meas["flux_ap"],
            "star_obs_e_per_s": star_obs_e_per_s,
            "fstar_erg_s_cm2": fstar,
            "f_tab_e_per_s": f_tab,
            "mul_e_to_erg": mul_e,
            "mul_adu_to_erg": mul_adu,
            "dqe": dqe,
            "plot_lam": " ".join(map(str, lam.tolist())),
            "plot_prf": " ".join(map(str, prf.tolist())),
            "plot_ifp": " ".join(map(str, ifp.tolist())),
            "plot_fint_norm": " ".join(map(str, (fint_e / np.nanmax(fint_e)).tolist())) if np.nanmax(fint_e) > 0 else "",
        })

        if not first_done:
            ax.plot(lam, prf * 100.0, label="filter *100")
            first_done = True
        ax.plot(lam, ifp * 100.0, alpha=0.35)
        if np.nanmax(fint_e) > 0:
            ax.plot(lam, 100.0 * fint_e / np.nanmax(fint_e), alpha=0.25)
        # plotting moved to averaged points

    ax.set_xlabel("Wavelength, Å")
    ax.set_ylabel("QE / profiles, %")
    ax.set_title(f"{standard_name}  {filter_name}")
    ax.grid(True, alpha=0.3)

    df = pd.DataFrame(rows).sort_values("lambda0").reset_index(drop=True)
    out_txt = phot_cube_path.with_name(f"QE_{standard_name}_{filter_name}.txt")
    out_png = phot_cube_path.with_name(f"QE_{standard_name}_{filter_name}.png")
    out_json = phot_cube_path.with_name(f"QE_{standard_name}_{filter_name}.json")
    out_csv = phot_cube_path.with_name(f"QE_{standard_name}_{filter_name}.csv")

    agg, aggmap = _aggregate_by_lambda(df, float(cwl))

    df.to_csv(out_csv, index=False)

    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"# {phot_cube_path.parent.name}   FW[A]={fw}\n")
        f.write("# averaged by wavelength\n")
        f.write("#lda       QE         N    mul: [e/sec]->[erg/s/cm2]\n")
        for _, r in agg.iterrows():
            f.write(f"{int(round(r['lambda0'])):5d}{r['dqe']:10.2f}{int(r['n']):10d}{r['mul_e_to_erg']:15.5E}\n")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    plot_qe_avg(df, agg, out_png, standard_name, filter_name)

    out = {
        "standard_name": standard_name,
        "filter": filter_name,
        "calib_file": str(calib_file),
        "filters_dir": str(filters_dir),
        "phot_cube": str(phot_cube_path),
        "source_cube": str(source_cube_path),
        "qe_txt": str(out_txt),
        "qe_png": str(out_png),
        "median_mul_e_to_erg": float(df["mul_e_to_erg"].median()),
        "median_mul_adu_to_erg": float(df["mul_adu_to_erg"].median()),
        "median_qe": float(df["dqe"].median()),
        "qe_detailed_csv": str(out_csv),
        "n_slices": int(len(df)),
        "n_lambda_groups": int(len(agg)),
        **aggmap,
    }
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["qe_json"] = str(out_json)
    return out


def _find_latest_json(root: Path, filter_name: str | None = None) -> Path | None:
    cands = sorted(root.rglob("QE_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if filter_name:
        for p in cands:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if str(d.get("filter", "")).strip() == str(filter_name).strip():
                    return p
            except Exception:
                continue
    return cands[0] if cands else None


def apply_ergs_to_fits(
    in_fits: str | Path,
    *,
    ergcal_json: str | Path,
    out_fits: str | Path | None = None,
) -> Path:
    in_fits = Path(in_fits)
    d = json.loads(Path(ergcal_json).read_text(encoding="utf-8"))

    name = in_fits.name
    if name.endswith("_net_NE.fits"):
        coeff = float(d.get("line_mul_adu_to_erg", d.get("median_mul_adu_to_erg")))
        coeff_kind = "line"
    elif re.search(r"_cont\d*_NE\.fits$", name):
        coeff = float(d.get("cont_mul_adu_to_erg", d.get("median_mul_adu_to_erg")))
        coeff_kind = "cont"
    elif name.endswith("_NE.fits"):  # line image: <obj>_<tag>_NE.fits (older runs: ..._line_NE.fits)
        coeff = float(d.get("line_mul_adu_to_erg", d.get("median_mul_adu_to_erg")))
        coeff_kind = "line"
    else:
        coeff = float(d.get("median_mul_adu_to_erg"))
        coeff_kind = "median"

    data, hdr = fits.getdata(in_fits, header=True)
    out = np.asarray(data, dtype=np.float32) * coeff
    hdr["HISTORY"] = f"ERG calibration from {Path(ergcal_json).name}"
    hdr["ERGCF"] = coeff
    hdr["ERGKIND"] = coeff_kind
    hdr["BUNIT"] = "erg/s/cm2"

    if out_fits is None:
        out_fits = in_fits.with_name(in_fits.stem + "_erg.fits")
    out_fits = Path(out_fits)
    stamp(hdr, out_fits)
    fits.writeto(out_fits, out.astype(np.float32), hdr, overwrite=True)
    return out_fits


# averaged QE plot
def plot_qe_avg(df, agg, out_png, standard, filter_name):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 7), dpi=200)

    # Restore the old-style QE plot:
    # - filter profile * 100
    # - representative IFP profiles * 100
    # - normalized tabulated spectrum * 100
    # - QE points only for wavelength-averaged values
    if "plot_lam" in df.columns and "plot_prf" in df.columns:
        first = df.iloc[0]
        lam = np.fromstring(str(first["plot_lam"]), sep=" ")
        prf = np.fromstring(str(first["plot_prf"]), sep=" ")
        if lam.size and prf.size and lam.size == prf.size:
            ax.plot(lam, prf * 100.0, color="mediumpurple", lw=1.5, zorder=3)

    # Plot one IFP curve per averaged wavelength (closest raw row)
    if "plot_lam" in df.columns and "plot_ifp" in df.columns:
        used = set()
        for _, ar in agg.iterrows():
            idx = (df["lambda0"] - float(ar["lambda0"])).abs().idxmin()
            if idx in used:
                continue
            used.add(idx)
            row = df.loc[idx]
            lam = np.fromstring(str(row["plot_lam"]), sep=" ")
            ifp = np.fromstring(str(row["plot_ifp"]), sep=" ")
            if lam.size and ifp.size and lam.size == ifp.size:
                ax.plot(lam, ifp * 100.0, color="deepskyblue", lw=1.0, alpha=0.8, zorder=2)

    # Plot normalized reference spectrum once
    if "plot_lam" in df.columns and "plot_fint_norm" in df.columns:
        first = df.iloc[0]
        lam = np.fromstring(str(first["plot_lam"]), sep=" ")
        fintn = np.fromstring(str(first["plot_fint_norm"]), sep=" ")
        if lam.size and fintn.size and lam.size == fintn.size:
            ax.plot(lam, fintn * 100.0, color="lightgray", lw=1.2, zorder=1)

    # Averaged QE points only
    ax.scatter(agg["lambda0"], agg["dqe"], s=140, marker="*", color="crimson", zorder=5)

    ax.set_xlabel("Wavelength, Å")
    ax.set_ylabel("QE / profiles, %")
    ax.set_title(f"{standard} {filter_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
