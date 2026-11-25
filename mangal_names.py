"""
File naming for the MaNGaL pipeline - the single place that knows how product files are called.
Pure Python (no astropy), so every step can import it.

Intermediate cubes carry ONE stage label that is *replaced* at every stage, never appended:

    <obj>_darkflat.fits -> <obj>_crclean.fits -> <obj>_shift.fits -> <obj>_phot.fits -> <obj>_linecont.fits

The final 2D products carry only the labels of the last stages: the line (if known), the product kind
and the labels of the stages that produced them - net, NE (astrometry), erg (flux calibration):

    <obj>_cont.fits   <obj>_cont1.fits  <obj>_cont2.fits          continuum (combined / individual)
    <obj>[_<line>]_line.fits   <obj>[_<line>]_net.fits            emission line, continuum-subtracted line
    <obj>_cont_NE.fits   <obj>[_<line>]_net_NE.fits   ...         + astrometry
    <obj>[_<line>]_net_NE_erg.fits                                + erg calibration

<line> is Ha, NII, 5007 (for [OIII]) or the first word of the label in the ZSPACE log.
"""
from __future__ import annotations

import re
from pathlib import Path

# stage labels of the intermediate cubes, in processing order
CUBE_STAGES = ("darkflat", "crclean", "shift", "phot", "linecont")

# trailing tokens removed when the "base" (= object) name is recovered from a cube file name;
# "cr" is a legacy spelling that old runs chained into the names
_STAGE_TOKENS = set(CUBE_STAGES) | {"cr"}


# --------------------------------------------------------------------------------------
# cube names
# --------------------------------------------------------------------------------------
def _stem(x) -> str:
    name = Path(str(x)).name
    return name[:-5] if name.lower().endswith(".fits") else name


def base_name(x) -> str:
    """
    Object part of a pipeline file name: drops '.fits', a leading 'cube_' and any trailing stage labels.
        NGC7803_shift.fits                                    -> NGC7803
        cube_NGC7803_darkflat_crclean_shift_phot_linecont.fits -> NGC7803   (legacy chained name)
    """
    s = _stem(x)
    if s.startswith("cube_") and len(s) > 5:
        s = s[5:]
    while True:
        head, sep, tail = s.rpartition("_")
        if sep and head and tail in _STAGE_TOKENS:
            s = head
        else:
            return s


def stage_path(like, stage: str, directory=None) -> Path:
    """Name of the `stage` cube that belongs to the same object as the file `like`."""
    like = Path(like)
    return Path(directory if directory is not None else like.parent) / f"{base_name(like)}_{stage}.fits"


def object_stage_path(w_dir, object_name: str, stage: str) -> Path:
    return Path(w_dir) / f"{object_name}_{stage}.fits"


# --------------------------------------------------------------------------------------
# side files of a cube: <cube stem><ext>, e.g. NGC7803_shift.stars.xy (older versions: NGC7803_shift.fits.stars.xy)
# --------------------------------------------------------------------------------------
def side_path(cube, ext: str) -> Path:
    """'NGC7803_shift.fits', '.stars.xy' -> 'NGC7803_shift.stars.xy' (no '.fits' inside the name)."""
    cube = Path(cube)
    stem = cube.name[:-5] if cube.name.endswith(".fits") else cube.stem
    return cube.with_name(stem + ext)


def find_side(cube, ext: str, legacy_ext: str | None = None) -> Path:
    """Existing side file of a cube: the new name, else the name of older versions (<cube>.fits<ext>).
    Returns the new name when neither exists."""
    new = side_path(cube, ext)
    if new.exists():
        return new
    old = Path(cube).with_name(Path(cube).name + (legacy_ext or ext))
    return old if old.exists() else new


# --------------------------------------------------------------------------------------
# emission-line tags
# --------------------------------------------------------------------------------------
_RE_NII = re.compile(r"nii|n\s*ii\b|\[n2\]", re.I)
_RE_OIII = re.compile(r"o\s*iii|\[o3\]|5007", re.I)
_RE_HA = re.compile(r"h[\s\-_]?alpha|\bh\s*a\b|hα|\[ha\]", re.I)


def sanitize_tag(label: str) -> str:
    """First word of a log label reduced to [A-Za-z0-9] ('[SII] 6717' -> 'SII')."""
    tok = (str(label or "").split() or [""])[0]
    return re.sub(r"[^0-9A-Za-z]+", "", tok)[:20]


def known_line_tag(label: str) -> str:
    """Tag of a recognised emission line ('NII', '5007' for [OIII], 'Ha'), else ''."""
    s = str(label or "")
    if _RE_NII.search(s):
        return "NII"
    if _RE_OIII.search(s):
        return "5007"
    if _RE_HA.search(s):
        return "Ha"
    return ""


def line_tag(label: str) -> str:
    """Tag used in file names of an emission line: known tag, else the first word of the label."""
    return known_line_tag(label) or sanitize_tag(label)


# --------------------------------------------------------------------------------------
# 2D products
# --------------------------------------------------------------------------------------
def product_path(directory, base: str, kind: str, tag: str = "") -> Path:
    """
    <base>[_<tag>]_<kind>.fits, kind = net | cont | cont1 | cont2 ...
    Emission-line images carry no kind label when the line is known: <base>_<tag>.fits (NGC7803_NII.fits);
    without a tag they stay <base>_line.fits.
    """
    if kind == "line":
        stem = f"{base}_{tag}" if tag else f"{base}_line"
    else:
        stem = f"{base}_{tag}_{kind}" if tag else f"{base}_{kind}"
    return Path(directory) / f"{stem}.fits"


def legacy_line_path(directory, base: str, tag: str = "") -> Path:
    """Line image name of older versions: <base>[_<tag>]_line.fits."""
    return Path(directory) / (f"{base}_{tag}_line.fits" if tag else f"{base}_line.fits")


def line_tag_from_stem(stem: str, base: str) -> str:
    """Tag of a line image: 'NGC7803_NII' -> 'NII', 'NGC7803_NII_line' -> 'NII', 'NGC7803_line' -> ''."""
    rest = stem[len(base):].lstrip("_") if stem.startswith(base) else stem
    rest = strip_kind(rest, "line")
    return "" if rest in ("", "line") else rest


def strip_kind(stem: str, kind: str) -> str:
    """'NGC7803_NII_net' -> 'NGC7803_NII' (removes a trailing '_<kind>' only)."""
    suf = "_" + kind
    return stem[: -len(suf)] if stem.endswith(suf) else stem


def split_product_stem(stem: str, base: str) -> tuple[str, str]:
    """('NII', 'net') from 'NGC7803_NII_net' and base 'NGC7803'; tag is '' when absent."""
    rest = stem[len(base):].lstrip("_") if stem.startswith(base) else stem
    tag, _, kind = rest.rpartition("_")
    return (tag, kind) if tag or kind else ("", "")


# --------------------------------------------------------------------------------------
# layer table of the multi-layer line/continuum cube (header keywords)
# --------------------------------------------------------------------------------------
def write_layer_table(hdr, layers: list[dict]) -> None:
    """
    Store the layer description in the FITS header (hdr behaves like astropy Header):
        NLAYERS, and for every layer n: LAYERn name, LKINDn line|cont, LROLEn total|group,
        LTAGn file-name tag (lines), LZSPn ZSPACE values, LTEXPn exposure [s], LNFRn frames, LZMEANn mean zenith distance,
        LLAMn mean wavelength [A] (combined nights).
    """
    hdr["NLAYERS"] = (len(layers), "number of layers in the cube")
    for n, L in enumerate(layers):
        hdr[f"LAYER{n}"] = (str(L["name"]), "layer name")
        hdr[f"LKIND{n}"] = (str(L["kind"]), "line or cont")
        hdr[f"LROLE{n}"] = (str(L.get("role", "group")), "total (sum of all) or group")
        if L.get("tag"):
            hdr[f"LTAG{n}"] = (str(L["tag"]), "tag used in output file names")
        if L.get("zspace") not in (None, ""):
            hdr[f"LZSP{n}"] = (str(L["zspace"])[:60], "ZSPACE values")
        if L.get("texp") is not None:
            hdr[f"LTEXP{n}"] = (float(L["texp"]), "[s] exposure of the layer")
        if L.get("nframes") is not None:
            hdr[f"LNFR{n}"] = (int(L["nframes"]), "number of frames in the layer")
        if L.get("zmean") is not None:
            hdr[f"LZMEAN{n}"] = (round(float(L["zmean"]), 3), "[deg] mean zenith distance of the layer")
        if L.get("lam") is not None:
            hdr[f"LLAM{n}"] = (round(float(L["lam"]), 3), "[A] mean wavelength of the layer")


def read_layer_table(hdr, nlayers_in_data: int | None = None) -> list[dict]:
    """Inverse of write_layer_table; copes with old 2-layer cubes (line_sum, cont_sum) without NLAYERS."""
    n = 0
    try:
        n = int(hdr.get("NLAYERS", 0) or 0)
    except Exception:
        n = 0
    layers: list[dict] = []
    if n > 0:
        for i in range(n):
            layers.append({
                "index": i,
                "name": str(hdr.get(f"LAYER{i}", f"layer{i}")),
                "kind": str(hdr.get(f"LKIND{i}", "line" if i == 0 else "cont")),
                "role": str(hdr.get(f"LROLE{i}", "total" if i < 2 else "group")),
                "tag": str(hdr.get(f"LTAG{i}", "") or ""),
                "zspace": str(hdr.get(f"LZSP{i}", "") or ""),
                "texp": hdr.get(f"LTEXP{i}", None),
                "nframes": hdr.get(f"LNFR{i}", None),
                "zmean": hdr.get(f"LZMEAN{i}", None),
                "lam": hdr.get(f"LLAM{i}", None),
            })
        return layers
    # legacy: layer 0 = line_sum, layer 1 = cont_sum
    for i, (name, kind) in enumerate((("line_sum", "line"), ("cont_sum", "cont"))):
        if nlayers_in_data is not None and i >= nlayers_in_data:
            break
        layers.append({"index": i, "name": str(hdr.get(f"LAYER{i}", name)), "kind": kind,
                       "role": "total", "tag": "", "zspace": "", "texp": None})
    return layers


def plan_layers(layers: list[dict]) -> dict:
    """
    What to make of a layer table:
        cont_total  index of the layer holding the combined continuum, or None when the combined
                    continuum must be computed from the continuum layers (exposure-weighted mean)
        lines       [(tag, index), ...] one entry per emission line (net is computed for each)
        conts       [(name, index), ...] individual continua (empty when there is only one)
    Cubes from older versions (with line_sum / cont_sum total layers, or only those two) work too.
    """
    line_grp = [L for L in layers if L["kind"] == "line" and L["role"] == "group"]
    line_tot = [L for L in layers if L["kind"] == "line" and L["role"] != "group"]
    cont_grp = [L for L in layers if L["kind"] == "cont" and L["role"] == "group"]
    cont_tot = [L for L in layers if L["kind"] == "cont" and L["role"] != "group"]

    if line_grp:
        lines = [(L["tag"], L["index"]) for L in line_grp]
    elif line_tot:
        lines = [(line_tot[0]["tag"], line_tot[0]["index"])]
    else:
        lines = []

    if cont_tot:                       # old cubes: a ready combined continuum
        cont_total = cont_tot[0]["index"]
        conts = [(L["name"], L["index"]) for L in cont_grp]
    elif len(cont_grp) == 1:           # a single continuum is the combined one
        cont_total = cont_grp[0]["index"]
        conts = []
    else:                              # several: combined one is computed
        cont_total = None
        conts = [(L["name"], L["index"]) for L in cont_grp]
    return {"cont_total": cont_total, "lines": lines, "conts": conts}
