#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import glob
from pathlib import Path
import sys
import numpy as np
from astropy.io import fits
import re



def _propagate_inputs_sidecar(cube_in: Path, cube_out: Path) -> None:
    """Copy <cube>.inputs.csv if it exists and slice count is unchanged."""
    try:
        src = find_side(cube_in, ".inputs.csv")
        if not src.exists():
            return
        dst = side_path(cube_out, ".inputs.csv")
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass


def _extract_cube_id_from_filename(fn: str) -> str:
    """Return 2-digit cube ID from filename: t241007_0801 -> '08'."""
    m = re.search(r"_([0-9]{2})[0-9]{2}$", fn)
    if not m:
        raise ValueError(f"Cannot infer cube id from filename {fn}")
    return m.group(1)



def _check_same_setup(obj_rows, cub_obj) -> None:
    """
    Several cubes (CUB_OBJ = '04,05') are merged into one cube with ONE master dark and ONE master flat,
    so all of them must have the same filter and the same exposure time.
    """
    if len(cube_ids_list(cub_obj)) < 2:
        return
    cols = [c for c in ("FILTERS", "Texp") if c in obj_rows.columns]
    setups = {}
    for cid in cube_ids_list(cub_obj):
        sel = select_cube_rows(obj_rows, cid)
        if not sel.empty:
            setups[cid] = tuple(sorted({tuple(str(v).strip() for v in r) for r in sel[cols].itertuples(index=False)}))
    if len({v for v in setups.values()}) > 1 or any(len(v) > 1 for v in setups.values()):
        lines = "\n".join(f"  cube {cid}: " + "; ".join(", ".join(f"{c}={v}" for c, v in zip(cols, s)) for s in st)
                          for cid, st in setups.items())
        raise SystemExit(
            f"Cubes {cub_obj} cannot be merged: they differ in {' / '.join(cols)} "
            f"(one master dark and one flat are used for the merged cube):\n{lines}\n"
            f"Select only cubes with the same filter and exposure, or process them one by one."
        )


def _resolve_zspace_log(params: dict, r_dir: Path, w_dir: Path, cli_override: str | None) -> Path:
    """Resolve ZSPACE log file path from params/CLI or auto-detect under r_dir."""
    if cli_override:
        p = Path(cli_override).expanduser()
        if not p.is_absolute():
            p = r_dir / p
        if not p.exists():
            raise FileNotFoundError(f"ZSPACE log not found: {p}")
        return p

    z = params.get("ZSPACE_LOG")
    if z:
        p = Path(str(z).strip()).expanduser()
        if not p.is_absolute():
            cand = r_dir / p
            if cand.exists():
                return cand
            cand2 = w_dir / p
            if cand2.exists():
                return cand2
        if p.exists():
            return p

    # auto-detect: prefer *log*.txt then *.txt; verify by parsing
    candidates: list[Path] = []
    candidates += sorted(r_dir.glob("*log*.txt"))
    candidates += sorted([p for p in r_dir.glob("*.txt") if p not in candidates])
    for p in candidates:
        try:
            parse_zspace_log(p)
            return p
        except Exception:
            continue
    raise FileNotFoundError(
        f"Cannot auto-detect ZSPACE log in R_DIR={r_dir}. "
        f"Set ZSPACE_LOG in param file or pass --zspace-log."
    )
from mangal_step4_shift import align_fits, NoReferenceStarsError, subtract_background_fits, background_already_subtracted
from mangal_step5_photometry import run_step5
from mangal_step6_linecont import build_linecont_cube, parse_zspace_log
from mangal_step7_net_astro import (final_cr_clean_products, extract_layer_products, subtract_continuum_all,
                                    write_cont_products, apply_wcs_to_group_products,
                                    run_astrometry_standalone)
from mangal_names import object_stage_path, stage_path, legacy_line_path, side_path, find_side

from mangal_step3 import (
    parse_param_file, parse_obs_log_table,
    build_masterdark, build_masterflat,
    build_reduced_cube_darkflat, clean_cosmics_cube, shift_cube, select_cube_rows, cube_ids_list, object_rows
)

from mangal_step8_ergcal import compute_qe_from_standard, apply_ergs_to_fits, _find_latest_json


def find_log_auto(r_dir: Path) -> Path:
    logs = sorted(r_dir.glob("t*.txt"))
    if not logs:
        raise FileNotFoundError(f"No t*.txt found in R_DIR={r_dir}")
    if len(logs) > 1:
        print("[warn] multiple log files found, using:", logs[0])
    return logs[0]


def _latest_existing(paths):
    existing = [Path(p) for p in paths if p is not None and Path(p).exists()]
    if not existing:
        return None
    existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return existing[0]


def _find_existing_products(w_dir: Path, object_name: str) -> dict:
    """
    Intermediate cubes of the object: <obj>_darkflat / _crclean / _shift / _phot / _linecont .fits.
    Cubes of older runs (cube_<obj>_darkflat_crclean_shift_phot.fits ...) are still found.
    """
    def one(stage: str, legacy_glob: str):
        new = object_stage_path(w_dir, object_name, stage)
        if new.exists():
            return new
        return _latest_existing(w_dir.glob(legacy_glob))

    o = glob.escape(object_name)
    return {
        "darkflat": one("darkflat", f"cube_{o}_darkflat.fits"),
        "crclean": one("crclean", f"cube_{o}_darkflat*_crclean.fits"),
        "shift": one("shift", f"cube_{o}_darkflat*_shift.fits"),
        "phot": one("phot", f"cube_{o}_darkflat*_phot.fits"),
        "linecont": one("linecont", f"cube_{o}_darkflat*_linecont.fits"),
    }


def _phot_predecessor_from_cube(cube_path: Path) -> Path | None:
    """The cube a photometry cube was made from (its slices match the ZSPACE sidecar)."""
    cube_path = Path(cube_path)
    try:  # new files remember their source in the header
        src = str(fits.getheader(cube_path).get("PHOTSRC") or "").strip()
        if src and (cube_path.parent / src).exists():
            return cube_path.parent / src
    except Exception:
        pass
    stem = cube_path.stem
    if stem.endswith("_phot"):  # legacy chained names: cube_..._shift_phot.fits -> cube_..._shift.fits
        cand = cube_path.with_name(stem[:-5] + cube_path.suffix)
        if cand.exists():
            return cand
    return _latest_existing([stage_path(cube_path, st) for st in ("shift", "crclean", "darkflat")])


def _resolve_existing_input(products: dict, *, for_phot: bool) -> Path | None:
    if for_phot:
        return _latest_existing([products.get("shift"), products.get("crclean"), products.get("darkflat")])
    return _latest_existing([products.get("phot"), products.get("shift"), products.get("crclean"), products.get("darkflat")])


def _apply_ergs_stage(args, w_dir: Path, r_dir: Path, exp_filter, object_name: str | None = None) -> None:
    """
    Erg calibration of existing *_net_NE.fits in W_DIR. Needs no cubes / log / earlier stages:
    only the QE_*.json (auto-found under R_DIR/.. or --ergcal-json) and the net_NE images.
    If object_name is given, only that object's *_net_NE.fits are used.
    """
    pat = f"*{glob.escape(object_name)}*_net_NE.fits" if object_name else "*_net_NE.fits"
    net_ne_files = sorted(w_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    if not net_ne_files:
        raise FileNotFoundError(f"No {pat} files found in {w_dir} for erg calibration (run astrometry first)")

    if not exp_filter:  # no log in standalone mode: try the FITS header of the newest net image
        try:
            h = fits.getheader(net_ne_files[0])
            exp_filter = str(h.get("FILTERS") or h.get("FILTER") or "").strip() or None
        except Exception:
            exp_filter = None

    combined = sorted(w_dir.glob(f"QE_combined_{exp_filter}.json" if exp_filter else "QE_combined_*.json"))
    if combined and not args.ergcal_json:  # combined nights: calibration averaged over the standards of all nights
        erg_json = combined[0]
    elif args.ergcal_json:
        erg_json = Path(args.ergcal_json)
    else:  # the calibration of THIS night (nights lying side by side must not mix), else the newest one
        from mangal_combine import find_night_qe_json
        erg_json = find_night_qe_json(r_dir, exp_filter) or _find_latest_json(r_dir.parent, filter_name=exp_filter)
    if erg_json is None or not Path(erg_json).exists():
        raise FileNotFoundError("No QE_*.json calibration file found. Run a standard star with --qe-calib first or pass --ergcal-json.")

    try:
        erg_info = json.loads(Path(erg_json).read_text(encoding="utf-8"))
        print(f"[OK] used calibration: {erg_json}" + (f" (filter {exp_filter})" if exp_filter else ""))
        print(f"[info] line_mul_adu_to_erg = {float(erg_info.get('line_mul_adu_to_erg', erg_info.get('median_mul_adu_to_erg'))):.6e}")
        if "line_lambda" in erg_info:
            print(f"[info] line_lambda = {float(erg_info['line_lambda']):.2f} A")
        print("[info] calib_ergs.pro logic: apply erg calibration only to *_net_NE.fits")
    except Exception as e:
        print(f"[OK] used calibration: {erg_json}")
        print(f"[warn] could not read details from erg calibration json: {e}")

    for f in net_ne_files:
        out_erg = apply_ergs_to_fits(f, ergcal_json=erg_json)
        print(f"[OK] erg-calibrated net product: {out_erg}")
        print(f"[info] applied line coefficient to {f.name}")


def _parse_frame_list(text: str) -> list[int]:
    """'3, 7-9' (1-based) -> [2, 6, 7, 8] (0-based)"""
    out: set[int] = set()
    for part in re.split(r"[,;\s]+", str(text or "").strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:\s*[-:]\s*(\d+))?", part)
        if not m:
            print(f"[warn] cannot parse frame spec {part!r} (expected N or N-M)", file=sys.stderr)
            continue
        lo = int(m.group(1)); hi = int(m.group(2) or lo)
        if lo < 1 or hi < lo:
            print(f"[warn] bad frame range {part!r}", file=sys.stderr)
            continue
        out.update(range(lo - 1, hi))
    return sorted(out)


def _shift_exclude_center(params: dict, args, label: str = "shift"):
    """Center exclusion (shift reference stars, background mask) from --shift-center-radius (0 = disabled)."""
    rr = float(args.shift_center_radius or 0.0)
    if rr <= 0:
        return None
    try:
        xc = float(params.get("MASK_XC"))
        yc = float(params.get("MASK_YC"))
    except (TypeError, ValueError):
        print("[warn] --shift-center-radius > 0 but MASK_XC/MASK_YC are not set in the param file: "
              "center exclusion is NOT applied", file=sys.stderr)
        return None
    print(f"[{label}] exclude center: xc={xc:.1f}, yc={yc:.1f}, r={rr:.1f}")
    return (xc, yc, rr)


def _run_shift_stars(final_cube, out_shift, args, params):
    """
    align_fits with the GUI/CLI shift parameters; stops the whole run if no star is available.
    The background is NOT subtracted here: that is done once at step 1 (--background-mode).
    """
    bg = background_already_subtracted(final_cube)
    if bg:
        print(f"[shift] background was subtracted at step 1 (BGMODE={bg})")
    else:
        print("[shift] note: the input cube has no background subtraction "
              "(run step 1 with --background-mode backgr_sub to subtract it)")
    try:
        return align_fits(
            final_cube,
            out_shift,
            ref_index=args.ref_index,
            exclude_center=_shift_exclude_center(params, args),
            detect_snr=args.shift_detect_snr,
            search_half=args.shift_search_half,
            dump_frames=True,
            background_mode="none",
            skip_frames=_parse_frame_list(getattr(args, "shift_skip_frames", "")),
            cube_manual=bool(getattr(args, "shift_manual_pairs", False)),
        )
    except NoReferenceStarsError as e:
        raise SystemExit(f"[shift] ERROR: {e}")


def _print_layers(linecont_path: Path) -> None:
    """Print the layer table of a line/continuum cube."""
    try:
        from mangal_names import read_layer_table
        hdr = fits.getheader(linecont_path)
        print(f"[OK] line/cont cube:   {linecont_path}")
        for L in read_layer_table(hdr, int(hdr.get("NAXIS3", 0) or 0) or None):
            texp = L.get("texp")
            print(f"     layer {L['index']}: {L['name']:<10s} {L['kind']:<4s} {L['role']:<5s}"
                  f"{('  tag=' + L['tag']) if L['tag'] else ''}"
                  f"{('  ZSPACE=' + L['zspace']) if L['zspace'] else ''}"
                  f"{('  texp=%.0fs' % float(texp)) if texp is not None else ''}")
    except Exception as e:
        print(f"[warn] cannot print the layer table: {e}")


def _locate_linecont(w_dir: Path, object_name: str, hints) -> Path | None:
    """The <obj>_linecont.fits cube: explicit hints first, then W_DIR (legacy cube_..._linecont.fits too)."""
    cands = [Path(h) for h in hints if h is not None]
    cands.append(object_stage_path(w_dir, object_name, "linecont"))
    for c in cands:
        if c.exists():
            return c
    return _latest_existing(w_dir.glob(f"cube_{glob.escape(object_name)}_darkflat*_linecont.fits"))


def _parse_protect(text):
    try:
        v = [float(t) for t in str(text or "").replace(";", ",").split(",") if t.strip()]
    except ValueError:
        v = []
    return (v[0], v[1], v[2]) if len(v) == 3 else None


def _combine_stage(args, w_dir: Path, hints) -> Path:
    """Combine this night's linecont cube with those of --combine-with; returns the folder of the result."""
    from mangal_combine import combine_nights
    ref_lc = _locate_linecont(w_dir, args.object_name, hints)
    if ref_lc is None:
        raise FileNotFoundError(f"Combination of nights needs <object>_linecont.fits in {w_dir}: run line+cont first")
    out_dir = Path(args.combine_out) if args.combine_out else w_dir.parent / f"{args.object_name}_combined"
    others = [Path(d) for d in args.combine_with]
    if any(d.resolve() == w_dir.resolve() for d in others):
        raise SystemExit("--combine-with must list the W_DIRs of the OTHER nights, not this one")
    out = combine_nights(w_dir, others, out_dir, object_name=args.object_name, scale=args.combine_scale,
                         manual=bool(args.combine_manual), ref_linecont=ref_lc)
    return out.parent


def _run_step7(args, params: dict, w_dir: Path, hints) -> None:
    """
    --net: continuum subtraction for every emission line of the linecont cube.
    --astro: astrometry of the continuum / line / net images (WCS of the combined continuum for all).
    Works from the linecont cube if there is one; --astro alone also works from the 2D images only.
    """
    lc = _locate_linecont(w_dir, args.object_name, hints)
    if lc is None:
        if args.net:
            raise FileNotFoundError(
                f"Net subtraction needs <object>_linecont.fits in {w_dir}. Run the --linecont stage first.")
        prod = run_astrometry_standalone(w_dir, args.object_name, wmed=args.wmed,
                                         api_key_env=args.astro_env, solve_timeout=args.astro_timeout)
    else:
        prods = extract_layer_products(lc)
        print(f"[info] step7 from {lc.name}: {len(prods['lines'])} line(s), "
              f"{len(prods['conts'])} individual continuum image(s) (+ combined continuum)")
        if args.net:
            mul_cont = args.mul_cont if args.mul_cont is not None else float(params.get("MUL_CONT", 1.0))
            mul_list = [mul_cont]
            if args.mul_cont_list:
                mul_list = [float(v) for v in str(args.mul_cont_list).split(",") if v.strip()] or mul_list
            res7 = subtract_continuum_all(
                lc, prods, mul_list=mul_list,
                auto_scale_from_stars=bool(args.auto_mul_cont),
                stars_xy_path=(Path(args.stars_xy) if args.stars_xy else None),
                win=int(args.win),
                seeing_lim=float(args.seeing_lim_net),
                local_mu_mode=str(args.local_mu),
                local_mu_nbins=int(args.local_mu_nbins),
                local_mu_smooth_bins=int(args.local_mu_smooth_bins),
                write_kmap=bool(args.write_kmap),
                write_star_table=True,
                scale_sigma_clip=float(args.scale_sigma_clip),
                register_cont=bool(args.register_cont),
                star_match=bool(args.star_match),
                star_protect=_parse_protect(args.star_protect),
            )
            for item in res7["lines"]:
                i7 = item["info"]
                if i7.get("auto_scale_from_stars"):
                    print("[info] star-scaled continuum:", {
                        k: i7[k] for k in ["k_used", "k_median_flux_ratio", "psf_matched", "fwhm_emis_px",
                                           "fwhm_cont_px", "fwhm_ratio_post", "k_map_median", "star_ratio_table_path"]
                        if k in i7})
                print(f"[OK] net[{item['tag'] or '-'}] (line - {i7.get('k_used', item['mul']):g}*cont): {item['net']}")
                if i7.get("subtract_png"):
                    print(f"[OK] subtraction plot: {i7['subtract_png']}")
            conts = list(res7["conts"])
        else:  # astrometry only: reuse the images that already exist, write missing continua from the cube
            conts = write_cont_products(lc, prods)
            for L in prods["lines"]:
                old = legacy_line_path(Path(L["line"]).parent, prods["base"], L["tag"])
                if not Path(L["line"]).exists() and old.exists():  # line image of an older run
                    L["line"] = old
                for kind in ("net", "line"):
                    if Path(L[kind]).exists():
                        print(f"[reuse] existing {kind} image for astrometry: {L[kind]}")
        if not args.astro:
            return
        prod = apply_wcs_to_group_products(
            prods["cont"],
            cont_groups=conts,
            net_fits=[L["net"] for L in prods["lines"] if Path(L["net"]).exists()],
            line_fits=[L["line"] for L in prods["lines"] if Path(L["line"]).exists()],
            wmed=args.wmed, api_key_env=args.astro_env, solve_timeout=args.astro_timeout,
        )
    if getattr(args, "cr_final", False):
        final_cr_clean_products(
            prod,
            sigclip=float(getattr(args, "cr_final_sigclip", 5.0)),
            sigfrac=float(getattr(args, "cr_final_sigfrac", 0.3)),
            objlim=float(getattr(args, "cr_final_objlim", 6.0)),
            niter=int(getattr(args, "cr_final_niter", 4)),
            cleantype=str(getattr(args, "cr_final_cleantype", "medmask")),
            gain=float(getattr(args, "gain", 1.0)),
            readnoise=float(getattr(args, "readnoise", 6.5)),
        )
    for k, v in prod.items():
        print(f"[OK] {k}: {v}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="MaNGaL pipeline: dark + flat + (optional) cosmicrays + (optional) shifts + (optional) transparency photometry"
    )
    ap.add_argument("--param", required=True)
    ap.add_argument("--object", required=True, dest="object_name")
    ap.add_argument("--log", default=None)
    ap.add_argument("--ext", default=".fts")
    ap.add_argument("--flat-norm", default="median", choices=["median", "mean"])

    # Step3: cosmic rays
    ap.add_argument("--prepare-calib", action="store_true", help="Build masterdarks/masterflat/reduced cube before downstream steps")
    ap.add_argument("--cr", action="store_true", help="Enable cosmic-ray cleaning (after flat)")
    ap.add_argument("--cr-final", action="store_true", help="Run an additional L.A.Cosmic pass at the very end on the *_net_NE images (and the line *_NE images if present), overwriting the same files (IDL-like)")
    ap.add_argument("--cr-final-sigclip", type=float, default=5.0, help="Final L.A.Cosmic sigclip (lower = more aggressive)")
    ap.add_argument("--cr-final-sigfrac", type=float, default=0.3, help="Final L.A.Cosmic sigfrac")
    ap.add_argument("--cr-final-objlim", type=float, default=6.0, help="Final L.A.Cosmic objlim (higher = protects real sources)")
    ap.add_argument("--cr-final-niter", type=int, default=4, help="Final L.A.Cosmic niter")
    ap.add_argument("--cr-final-cleantype", type=str, default="medmask", choices=["medmask","meanmask","idw","median"], help="Final L.A.Cosmic cleantype")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--readnoise", type=float, default=6.5)
    ap.add_argument("--sigclip", type=float, default=7.0)
    ap.add_argument("--objlim", type=float, default=8.0)
    ap.add_argument("--niter", type=int, default=4)
    ap.add_argument("--cleantype", default="medmask", choices=["meanmask", "medmask", "median", "idw"])
    ap.add_argument("--save-cr-mask", action="store_true")
    ap.add_argument("--outside-value", default="nan", help="Outside MASK_R circle: nan (default), none (keep), or number")

    # Step4: shifts / registration
    ap.add_argument("--shift", action="store_true", help="Enable frame/channel registration (shift cube slices)")
    ap.add_argument("--shift-mode", default="stars", choices=["stars"])
    ap.add_argument("--ref-index", type=int, default=0, help="Reference slice index for per-slice/per-cube")
    ap.add_argument("--upsample", type=int, default=10, help="Subpixel upsample factor for phase correlation")
    ap.add_argument("--shift-detect-snr", type=float, default=20.0, help="Detection threshold in robust sigma units for shift reference stars")
    ap.add_argument("--shift-search-half", type=int, default=30, help="Half-size of search window around previous-frame star position")
    ap.add_argument("--lc-exclude-frames", default="",
                    help="line+cont: frames left out of the sum by hand (e.g. satellite tracks), as numbers CCNN of the "
                         "observation log: \"1005, 1210-1212\" (or file names)")
    ap.add_argument("--shift-manual-pairs", action="store_true",
                    help="Merged cubes: click star pairs to register every other cube to the reference frame "
                         "(otherwise automatic, with the window only when that fails)")
    ap.add_argument("--shift-skip-frames", default="", help="Frames NOT used when measuring shifts, 1-based as in the '[shift stars] N/M' log lines, e.g. '3,7-9'. They stay in the cube; their shift is interpolated from the neighbouring frames")
    ap.add_argument("--shift-center-radius", type=float, default=0.0, help="Exclude shift reference stars inside radius r around mask center (pixels); 0 disables exclusion")
    ap.add_argument("--background-mode", default="backgr_sub", choices=["none","fon_corners","backgr_sub","plane","mean","min","corners_plane","corners_mean","corners_min"], help="Background subtraction at step 1 (with --prepare-calib / --cr), after CR cleaning; 'none' to switch off")
    ap.add_argument("--background-window", type=int, default=30, help="Window [px] of the background subtraction at step 1")

    # Step5: transparency photometry / filtering
    ap.add_argument("--phot", action="store_true", help="Run step5: photometry of stars and remove bad transparency slices")
    ap.add_argument("--win", type=int, default=15, help="Half-size of cutout window (pixels) around each star")
    ap.add_argument("--lim-flux", type=float, default=None, help="Lower limit on relative flux (mean over stars)")
    ap.add_argument("--lim-fwhm", type=float, default=None, help="Upper limit on FWHMpx (mean over stars)")
    ap.add_argument("--lim-ratio", type=float, default=None, help="Lower limit on Ratio (mean over stars, <=1)")
    ap.add_argument("--stars-xy", default=None, help="Star positions file (x y). Default for photometry: stars from the previous shift step (<obj>_shift.stars.xy); for --auto-mul-cont it is auto-detected if omitted")

    ap.add_argument("--std-win", type=int, default=None, help="Half-size of cutout window (pixels) for standard star photometry")
    ap.add_argument("--std-lim-flux", type=float, default=None, help="Lower limit on relative flux for standard star (mean over stars)")
    ap.add_argument("--std-lim-fwhm", type=float, default=None, help="Upper limit on FWHMpx for standard star (mean over stars)")
    ap.add_argument("--std-lim-ratio", type=float, default=None, help="Lower limit on Ratio for standard star (mean over stars, <=1)")
    
    # step6: line/continuum cube from ZSPACE mapping
    ap.add_argument("--linecont", action="store_true", help="Run step6: build <obj>_linecont.fits, a multi-layer cube: one layer per emission line and one per continuum (layer table in the header)")
    ap.add_argument("--zspace-log", default=None, help="Optional path/name of ZSPACE log; overrides ZSPACE_LOG in param file")
    ap.add_argument("--no-lc-split", action="store_true", help="Only two layers: 'line' (all emission lines merged) and 'cont' (all continuum merged)")
    ap.add_argument("--lc-group-by", default="zspace", choices=["zspace", "label"], help="What defines one individual continuum layer: one ZSPACE value (default) or one label from the ZSPACE log (lines are always grouped by emission line)")

    # step7: continuum subtraction + astrometry (astrometry.net via env var)
    ap.add_argument("--net", action="store_true", help="Run step7: subtract continuum from emission: net = line - k*cont")
    ap.add_argument("--mul-cont", type=float, default=None, help="Continuum multiplier (mul). If --auto-mul-cont is set, k = mul * median(Flux_line/Flux_cont) from stars; otherwise k=mul")
    ap.add_argument("--mul-cont-list", default=None, help="Comma-separated continuum multipliers, one per emission line in layer order, e.g. 1.0,1.2 (the last value is reused if the list is shorter)")
    ap.add_argument("--auto-mul-cont", action="store_true", help="Scale continuum like IDL cre_emis_map: use stars to estimate k and optionally PSF-match line/cont before subtraction")
    ap.add_argument("--local-mu", default="global", choices=["global","plane","quadratic","radial","y"], help="Continuum scaling mode for auto-mul-cont: global, plane, quadratic, radial, or y")
    ap.add_argument("--local-mu-nbins", type=int, default=12, help="Number of bins for local mu map (radial or y)")
    ap.add_argument("--local-mu-smooth-bins", type=int, default=3, help="Smoothing width in bins for local mu map (moving average)")
    ap.add_argument("--no-write-kmap", dest="write_kmap", action="store_false", help="Do not write the *_kmap.fits diagnostic product when using local-mu")
    ap.add_argument("--scale-sigma-clip", type=float, default=2.5, help="Sigma clipping threshold for stellar ratio spatial-fit")
    ap.add_argument("--no-register-cont", dest="register_cont", action="store_false",
                    help="Do not register the continuum to the line image by stars (affine; applied only if the shift is >= 0.03 px)")
    ap.add_argument("--star-match", action="store_true",
                    help="Cosmetic: subtract every point source with its own flux ratio / shift / PSF width "
                         "(star colours make a single k leave +/- residuals); table *_star_match.csv")
    ap.add_argument("--star-protect", default="",
                    help="'X,Y,R' (px): point sources inside this circle are never touched by --star-match")

    ap.add_argument("--seeing-lim-net", type=float, default=0.25, help="Seeing relative difference threshold for PSF matching (IDL seeing_lim)")
    ap.add_argument("--wmed", type=int, default=1, help="Median filter window (odd int). Applied to final NE images if >2 (IDL Wmed).")
    ap.add_argument("--astro", action="store_true", help="Solve astrometry on continuum via nova.astrometry.net and write *_NE products")
    ap.add_argument("--astro-env", default="ASTROMETRY_NET_API_KEY", help="Environment variable with astrometry.net API key")
    ap.add_argument("--astro-timeout", type=int, default=1200, help="Solve timeout (seconds)")

    ap.add_argument("--standard", action="store_true", help="Treat this run as standard star: skip continuum subtraction and astrometry")

    # combination of nights (between line+cont and net)
    ap.add_argument("--combine-with", nargs="+", default=None, metavar="W_DIR",
                    help="W_DIR(s) of the same object reduced on other nights (up to line+cont): their linecont cubes are "
                         "registered to this night, scaled and averaged; net / astro / apply-ergs then work on the result")
    ap.add_argument("--combine-out", default=None,
                    help="Folder for the combined cube (default: <W_DIR>/../<object>_combined)")
    ap.add_argument("--combine-scale", choices=["both", "stars", "standards"], default="both",
                    help="Flux scale between nights: field stars, QE calibrations of the nights, or both "
                         "(stars, with the standard-star ratio reported; standards if too few stars)")
    ap.add_argument("--combine-manual", action="store_true",
                    help="Click the star pairs for every night instead of the automatic match")

    # step5: diagnostics outputs
    ap.add_argument("--save-per-star", action="store_true", help="Also save per-star table")
    ap.add_argument("--qe-calib", action="store_true", help="Compute QE and erg calibration from a processed standard star")
    ap.add_argument("--apply-ergs", action="store_true", help="Apply previously computed erg calibration to the latest object product")
    ap.add_argument("--ergcal-json", default=None, help="Path to QE_*.json calibration file for --apply-ergs")
    ap.add_argument("--calib-file", default=None, help="Path to *.calib file (default: auto-detect in R_DIR)")
    ap.add_argument("--filters-dir", default=None, help="Directory with filter transmission profiles like MA500.txt")
    ap.add_argument("--okestan-dir", default=None, help="Path to okestan directory or zip")
    ap.add_argument("--qe-fw", type=float, default=13.8, help="IFP FWHM in Angstrom for QE calibration")
    ap.add_argument("--qe-win", type=int, default=20, help="Half window for standard-star fitting during QE calibration")

    args = ap.parse_args(argv)

    params = parse_param_file(args.param)
    r_dir = Path(params["R_DIR"])
    w_dir = Path(params["W_DIR"])
    ccd_cut = params.get("CCD_CUT_TUPLE")
    cub_obj = params.get("CUB_OBJ")
    cub_flat = params.get("CUB_FLAT")
    exp_filter = params.get("FILTERS")

    # Astrometry and erg calibration are independent final stages: alone (no --cr/--shift/--phot/
    # --linecont/--qe-calib) they need only the files of the previous final stages in W_DIR
    # (the linecont cube for --net; continuum / line / net images for --astro; QE_*.json for ergs):
    # no log, no raw data, no other earlier steps.
    _earlier = (args.prepare_calib or args.cr or args.shift or args.phot or args.linecont
                or args.qe_calib)
    if (args.net or args.astro or args.apply_ergs or args.combine_with) and not _earlier and not getattr(args, "standard", False):
        if args.combine_with:
            w_dir = _combine_stage(args, w_dir, [])
            args.net = True  # line, continuum and net images of the combined cube
        if args.net or args.astro:
            _run_step7(args, params, w_dir, [])
        if args.apply_ergs:
            _apply_ergs_stage(args, w_dir, r_dir, exp_filter, object_name=args.object_name)
        return 0

    if args.log is None:
        log_path = find_log_auto(r_dir)
        print(f"[auto] using log file: {log_path}")
    else:
        log_path = Path(args.log)
        if not log_path.is_absolute():
            log_path = r_dir / log_path
        if not log_path.exists():
            raise FileNotFoundError(f"log file not found: {log_path}")

    df = parse_obs_log_table(log_path)

    # object exposure (prefer series)
    # with CUB_OBJ the cubes are taken by number, whatever their OBJECT name (e.g. PGC45443 + PGC45443_dz)
    obj_rows = object_rows(df, args.object_name, cub_obj)
    if obj_rows.empty:
        raise SystemExit(f"No obj frames for OBJECT={args.object_name}"
                         + (f" in cube(s) CUB_OBJ={cub_obj}" if cub_obj else ""))
    _check_same_setup(obj_rows, cub_obj)
    texp_obj = int(obj_rows.iloc[0]["Texp"])

    # auto detect FILTERS from OBJECT rows if not set
    if "FILTERS" in obj_rows.columns:
        obj_filter = str(obj_rows.iloc[0]["FILTERS"]).strip()
        if not exp_filter:
            exp_filter = obj_filter
            params["FILTERS"] = obj_filter
            print(f"[auto] FILTERS not set in param file, using {obj_filter} from OBJECT rows")
        elif exp_filter != obj_filter:
            print(
                f"[warn] FILTERS in param file ({exp_filter}) != FILTERS in OBJECT rows ({obj_filter}), using param value",
                file=sys.stderr,
            )

    # flat exposure (prefer series)
    flat_rows = df[df["TYPE"] == "flat"].copy()
    if cub_flat:
        flat_rows2 = select_cube_rows(flat_rows, cub_flat)
        if not flat_rows2.empty:
            flat_rows = flat_rows2
    if exp_filter and "FILTERS" in flat_rows.columns:
        flat_rows2 = flat_rows[flat_rows["FILTERS"] == exp_filter]
        if not flat_rows2.empty:
            flat_rows = flat_rows2
    if flat_rows.empty:
        raise SystemExit("No flat frames found (check data; auto-detect CUB_FLAT/FILTERS failed)")

    # auto detect CUB_FLAT if not set
    if not cub_flat:
        try:
            auto_cub_flat = _extract_cube_id_from_filename(flat_rows.iloc[0]["FILE"])
            cub_flat = auto_cub_flat
            params["CUB_FLAT"] = auto_cub_flat
            print(f"[auto] CUB_FLAT not set in param file, using {auto_cub_flat} from FLAT rows")
        except Exception as e:
            print(f"[warn] could not auto-infer CUB_FLAT: {e}", file=sys.stderr)

    texp_flat = int(flat_rows.iloc[0]["Texp"])

    products = _find_existing_products(w_dir, args.object_name)

    phot_table = None
    zspace_source_cube = None

    # Build calibration products only when explicitly requested, or when CR-clean needs a fresh reduced cube.
    if args.prepare_calib or args.cr:
        for texp in sorted({texp_obj, texp_flat, *params.get("DARKS_LIST", [])}):
            try:
                out = build_masterdark(df, r_dir, w_dir, int(texp), ccd_cut=ccd_cut, pattern_ext=args.ext)
                print(f"[OK] masterdark: {out}")
            except FileNotFoundError as e:
                print(f"[err] {e}", file=sys.stderr)

        md_obj = w_dir / f"dark{texp_obj}.fits"
        md_flat = w_dir / f"dark{texp_flat}.fits"
        if not md_obj.exists():
            raise SystemExit(f"Missing masterdark for object exposure: {md_obj}")
        if not md_flat.exists():
            raise SystemExit(f"Missing masterdark for flat exposure: {md_flat}")

        mf = build_masterflat(
            df, r_dir, w_dir,
            cube_id=cub_flat,
            masterdark_path=md_flat,
            expected_filter=exp_filter,
            ccd_cut=ccd_cut,
            pattern_ext=args.ext,
            norm=args.flat_norm
        )
        print(f"[OK] masterflat: {mf}")

        cube = build_reduced_cube_darkflat(
            df, r_dir, w_dir,
            object_name=args.object_name,
            cube_id=cub_obj,
            masterdark_path=md_obj,
            masterflat_path=mf,
            ccd_cut=ccd_cut,
            pattern_ext=args.ext
        )
        print(f"[OK] reduced cube (dark+flat): {cube}")

        final_cube = cube

        if args.cr:
            mask_circle = None
            if all(k in params for k in ("MASK_XC", "MASK_YC", "MASK_R")):
                try:
                    mask_circle = (float(params["MASK_XC"]), float(params["MASK_YC"]), float(params["MASK_R"]))
                    print(f"[info] cosmic cleaning only inside circle: xc={mask_circle[0]}, yc={mask_circle[1]}, r={mask_circle[2]}")
                except Exception:
                    mask_circle = None

            ov = args.outside_value.strip().lower()
            if ov in ("none", "keep"):
                outside_value = None
            elif ov in ("nan", "NaN"):
                outside_value = np.nan
            else:
                outside_value = float(ov)

            out_cube, mask_path = clean_cosmics_cube(
                cube,
                gain=args.gain,
                readnoise=args.readnoise,
                sigclip=args.sigclip,
                objlim=args.objlim,
                niter=args.niter,
                save_mask=args.save_cr_mask,
                mask_circle=mask_circle,
                outside_value=outside_value,
                cleantype=args.cleantype,
            )
            print(f"[OK] CR-clean cube: {out_cube}")
            _propagate_inputs_sidecar(Path(cube), Path(out_cube))
            final_cube = out_cube
            if mask_path:
                print(f"[OK] CR mask cube: {mask_path}")

        # background: once here (after CR cleaning, which needs the sky level), not at every shift run
        bg_done = subtract_background_fits(
            final_cube,
            mode=args.background_mode,
            window=args.background_window,
            exclude_center=_shift_exclude_center(params, args, label="bg"),
        )
        if bg_done:
            print(f"[OK] background subtracted in {Path(final_cube).name}")

        if args.shift:
            if args.shift_mode == "stars":
                out_shift = stage_path(final_cube, "shift")

                _run_shift_stars(final_cube, out_shift, args, params)
                print(f"[OK] shifted cube: {out_shift}")
                _propagate_inputs_sidecar(Path(final_cube), Path(out_shift))
                print(f"[OK] shifts table: {side_path(out_shift, '.shifts.csv')}")
                print(f"[OK] stars diagnostic: {side_path(out_shift, '.stars.png')}")
                final_cube = out_shift
            else:
                tes = args.testring
                cht = args.channels_time
                if tes is not None:
                    tes_p = Path(tes)
                    if not tes_p.is_absolute():
                        tes_p = r_dir / tes_p
                    tes = str(tes_p)
                if cht is not None:
                    cht_p = Path(cht)
                    if not cht_p.is_absolute():
                        cht_p = r_dir / cht_p
                    cht = str(cht_p)

                out_shift, shifts = shift_cube(
                    final_cube,
                    mode=args.shift_mode,
                    ref_index=args.ref_index,
                    upsample_factor=args.upsample,
                    max_shift=args.shift_max_shift,
                    max_step=args.shift_max_step,
                    smooth_window=args.shift_smooth_window,
                    outlier_z=args.shift_outlier_z,
                    collapse=args.collapse,
                    testring_path=tes,
                    channels_time_path=cht,
                    invert_shift=args.invert_shift,
                )
                print(f"[OK] shifted cube: {out_shift}")
                print(f"[OK] shifts table: {out_shift.with_suffix('.shifts.csv')}")
                _propagate_inputs_sidecar(Path(final_cube), Path(out_shift))
                final_cube = out_shift

        zspace_source_cube = Path(final_cube)

    elif args.shift:
        existing = _latest_existing([products.get("crclean"), products.get("darkflat")])
        if existing is None:
            raise FileNotFoundError(
                "Shift was requested, but no existing reduced cube was found in W_DIR. "
                "Run with 'Mastercube + masterflat + CR-clean' first."
            )
        final_cube = Path(existing)
        print(f"[reuse] using existing input for shift: {final_cube}")
        if args.shift_mode == "stars":
            out_shift = stage_path(final_cube, "shift")

            _run_shift_stars(final_cube, out_shift, args, params)
            print(f"[OK] shifted cube: {out_shift}")
            _propagate_inputs_sidecar(Path(final_cube), Path(out_shift))
            print(f"[OK] shifts table: {side_path(out_shift, '.shifts.csv')}")
            print(f"[OK] stars diagnostic: {side_path(out_shift, '.stars.png')}")
            final_cube = out_shift
        zspace_source_cube = Path(final_cube)

    else:
        # Downstream-only mode: reuse products from previous steps.
        if args.phot:
            existing = _resolve_existing_input(products, for_phot=True)
            if existing is None:
                raise FileNotFoundError(
                    "Photometry requested, but no existing darkflat/CR/shift cube was found in W_DIR. "
                    "Run with CR-clean + Shift first."
                )
            final_cube = Path(existing)
            print(f"[reuse] using existing input for photometry: {final_cube}")
            zspace_source_cube = Path(final_cube)
        else:
            existing = _resolve_existing_input(products, for_phot=False)
            if existing is None:
                raise FileNotFoundError(
                    "No existing intermediate products found in W_DIR. "
                    "Run the earlier steps first (typically CR-clean + Shift, then Photometry)."
                )
            final_cube = Path(existing)
            print(f"[reuse] using existing downstream input: {final_cube}")

            if final_cube.stem.endswith("_phot"):
                phot_table = find_side(final_cube, ".csv", legacy_ext=".phot.csv")
                if not phot_table.exists():
                    phot_table = None
                pred = _phot_predecessor_from_cube(final_cube)
                zspace_source_cube = pred if pred is not None else final_cube
            else:
                zspace_source_cube = Path(final_cube)

    # Remember the cube whose slice order still matches the per-slice ZSPACE sidecar.
    # If step5 filters slices, step6 will use phot_table to map this to the filtered cube.
    if zspace_source_cube is None:
        zspace_source_cube = Path(final_cube)

    if args.phot:
        # Defaults can come from param file
        win = args.win
        lim_flux = args.lim_flux if args.lim_flux is not None else float(params.get("LIM_FLUX", 0.85))
        lim_fwhm = args.lim_fwhm if args.lim_fwhm is not None else float(params.get("LIM_FWHM", 20.0))
        lim_ratio = args.lim_ratio if args.lim_ratio is not None else float(params.get("LIM_RATIO", 0.5))

        # For standard star we may want a different photometry setup
        if getattr(args, "standard", False):
            if args.std_win is not None:
                win = args.std_win
            if args.std_lim_flux is not None:
                lim_flux = args.std_lim_flux
            if args.std_lim_fwhm is not None:
                lim_fwhm = args.std_lim_fwhm
            if args.std_lim_ratio is not None:
                lim_ratio = args.std_lim_ratio

        cube_in = Path(final_cube)
        cube_out = stage_path(cube_in, "phot")

        if args.stars_xy:
            stars_xy = Path(args.stars_xy)
        else:
            # Use exactly the same reference stars as in the previous shift step
            # (step4 writes <obj>_shift.stars.xy next to the shifted cube; older versions: <obj>_shift.fits.stars.xy).
            candidates = [
                find_side(cube_in, ".stars.xy"),
                find_side(stage_path(cube_in, "shift"), ".stars.xy"),
            ]
            stars_xy = next((p for p in candidates if p.exists()), candidates[0])

        if not stars_xy.exists():
            raise FileNotFoundError(
                f"Star list from the shift step was not found: {stars_xy}\n"
                f"Run the shift stage (shift by stars) first, or pass --stars-xy explicitly."
            )

        out_cube, out_tab = run_step5(
            cube_in=cube_in,
            cube_out=cube_out,
            stars_xy=stars_xy,
            win=win,
            lim_flux=lim_flux,
            lim_fwhm=lim_fwhm,
            lim_ratio=lim_ratio,
            save_per_star=args.save_per_star,
        )
        print(f"[OK] photometry table: {out_tab}")
        print(f"[OK] filtered cube:    {out_cube}")
        final_cube = out_cube
        phot_table = out_tab

    if args.linecont:
        zlog = _resolve_zspace_log(params, r_dir, w_dir, args.zspace_log)
        print(f"[info] ZSPACE log: {zlog}")
        cube_in = Path(final_cube)
        out_lc = stage_path(cube_in, "linecont")
        # Use pre-phot cube for per-slice ZSPACE sidecar, and phot table to map to filtered cube.
        phot_path = Path(phot_table) if phot_table is not None else None
        outp = build_linecont_cube(
            cube_path=cube_in,
            source_cube_path=zspace_source_cube,
            zspace_log_path=zlog,
            out_path=out_lc,
            phot_table_path=phot_path,
            split_groups=not args.no_lc_split,
            group_by=args.lc_group_by,
            exclude_frames=args.lc_exclude_frames,
        )
        _print_layers(outp)

    if args.qe_calib:
        if not getattr(args, "standard", False):
            raise SystemExit("--qe-calib is intended for a standard-star run (--standard)")
        cube_for_qe = Path(final_cube)
        if not cube_for_qe.exists():
            raise FileNotFoundError(f"Phot cube for QE calibration not found: {cube_for_qe}")
        phot_path = Path(phot_table) if phot_table is not None else find_side(cube_for_qe, ".csv", legacy_ext=".phot.csv")
        if not phot_path.exists():
            phot_path = None
        qe = compute_qe_from_standard(
            phot_cube_path=cube_for_qe,
            source_cube_path=zspace_source_cube,
            r_dir=r_dir,
            standard_name=args.object_name,
            filter_name=exp_filter,
            phot_table_path=phot_path,
            calib_file=args.calib_file,
            filters_dir=args.filters_dir,
            okestan_dir=args.okestan_dir,
            fw=float(args.qe_fw),
            win=int(args.qe_win),
        )
        print(f"[OK] QE table: {qe['qe_txt']}")
        print(f"[OK] QE plot:  {qe['qe_png']}")
        print(f"[OK] erg cal:  {qe['qe_json']}")
        print(f"[OK] median mul [ADU/s -> erg/s/cm2]: {qe['median_mul_adu_to_erg']:.6e}")

    # Combination of nights: the linecont cubes of the other nights are added to this one; the later stages
    # (net, astrometry, ergs) work in the folder of the combined cube
    _hint = [Path(outp)] if ("outp" in locals() and outp is not None) else []
    _hint.append(stage_path(final_cube, "linecont"))
    if args.combine_with and not getattr(args, "standard", False):
        w_dir = _combine_stage(args, w_dir, _hint)
        _hint = []
        args.net = True  # line, continuum and net images of the combined cube

    # Step7: continuum subtraction and astrometry
    if (args.net or args.astro) and not getattr(args, "standard", False):
        _run_step7(args, params, w_dir, _hint)

    if args.apply_ergs and (not getattr(args, "standard", False)):
        _apply_ergs_stage(args, w_dir, r_dir, exp_filter)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
