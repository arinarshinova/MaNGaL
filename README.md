# MaNGaL pipeline

Data reduction pipeline for **MaNGaL** (Mapper of Narrow Galaxy Lines), a tunable-filter photometer
based on a scanning Fabry–Perot interferometer. It turns a night of raw frames into continuum-subtracted,
astrometrically calibrated and flux-calibrated emission-line images (Hα, [N II], [O III] and others).

There are two ways to run it:

- **`mangal_qt6_gui.py`** — interactive graphical interface (recommended to start with);
- **`run_mangal.py`** — command line, the same pipeline step by step.

The GUI only builds a parameter file and a `run_mangal.py` command line, so both give identical results.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/arinarshinova/MaNGaL.git
cd MaNGaL
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

### Astrometry API key

The astrometry stage (`--astro`, *astrometry* in the GUI) uses [nova.astrometry.net](https://nova.astrometry.net)
and needs a **personal API key**: sign in there and copy it from *My Profile → API key*.
Set it in the terminal **before** starting the GUI or `run_mangal.py`:

```bash
export ASTROMETRY_NET_API_KEY="your_key"
python mangal_qt6_gui.py
```

To keep it for all sessions, add the `export` line to `~/.zshrc` (macOS) or `~/.bashrc` (Linux).
On Windows: `set ASTROMETRY_NET_API_KEY=your_key` (cmd) or `$env:ASTROMETRY_NET_API_KEY="your_key"` (PowerShell).
The other stages work without the key.

## Input data

A night directory (`R_DIR`) should contain:

| what | how it is found |
|---|---|
| raw frames `t<YYMMDD>_<CC><NN>.fts` (`CC` = cube/series number, `NN` = frame) | by the observation log |
| observation log `t*.txt` (columns `FILE OBJECT TYPE START Texp FILTERS … Zspace …`) | first `t*.txt` in `R_DIR`, or `--log` |
| ZSPACE log: lines like `Zspace=-1500 Halpha`, `Zspace=-1807 cont` | `ZSPACE_LOG` in the param file, `--zspace-log`, or auto-detected |
| for flux calibration only: `*.calib` | in `R_DIR`, or `--calib-file` |

Filter transmission curves and spectrophotometric standards are shipped with the pipeline in `data/`
(see [Flux calibration](#8-flux-calibration)), so they need not be copied to the night directory.

## Quick start: GUI

```bash
python mangal_qt6_gui.py
```

1. **read_dir → open…** — choose the night directory. Objects are read from the observation log;
   the `AUTHOR` of the raw data is shown under the object line.
2. Choose the **object** and its **cube(s)**; **write_dir** defaults to `<night>/../<object>`.
   All cubes of the object are ticked by default and are merged into one (see [Several cubes](#several-cubes)).
3. Tick the stages to run and adjust their parameters (shift, photometry, continuum subtraction).
4. Press **run**. The log is shown in the window; the photometry plots and the subtraction
   pictures (`*_net_subtract.png`) are opened when the corresponding stages finish.

Every stage runs only when its box is ticked; it takes the products of the earlier stages from `write_dir`,
so any stage can be repeated alone. For a **standard star**, tick *standard* and choose the star: step 1, shift, photometry and line+cont get ticked
(untick what is already done), tick *QE calibration* to get the calibration. *QE calibration* can also run alone:
it uses the existing `<star>_phot.fits`. When it finishes, the QE plot `QE_<star>_<filter>.png` is opened.
*net*, *astrometry* and *apply ergs* do not apply to a standard. When *apply ergs* is ticked for an object, an existing `QE_*.json` of the night is used;
if there is none, it is built first from the standard star selected in the *standard* list.

The parameter file is written to `write_dir/<object>_auto.txt` and the exact command line is printed
in the log, so any run can be repeated from the terminal.

## Command line

```bash
python run_mangal.py --param <param_file> --object <OBJECT> [stages] [options]
```

Full reduction of one object:

```bash
python run_mangal.py --param NGC7803_auto.txt --object NGC7803 \
    --cr --shift --phot --linecont --net --auto-mul-cont --star-match --astro
```

`python run_mangal.py --help` lists all options.

### Parameter file

A plain text file with `KEY = value` lines (`;` starts a comment, strings in single quotes).
The GUI writes one automatically; an example is [`examples/NGC7803_auto.txt`](examples/NGC7803_auto.txt):

```
R_DIR   = '/path/to/data/t261012'
W_DIR   = '/path/to/data/NGC7803'
CCD_CUT = '0,0,1599,1599'
FILTERS = 'SED675'
CUB_OBJ = '04'
CUB_FLAT= '09'
MASK_R  = 800
MASK_XC = 790
MASK_YC = 780
END
```

| key | meaning |
|---|---|
| `R_DIR`, `W_DIR` | raw data directory, output directory (**required**) |
| `FILTERS` | filter name; taken from the log if empty |
| `CUB_OBJ`, `CUB_FLAT` | series number of the object and of the flat; taken from the log if empty |
| `CCD_CUT` | `x1,y1,x2,y2` region of the detector to keep |
| `MASK_XC`, `MASK_YC`, `MASK_R` | centre and radius of the illuminated circular field |
| `DARKS` | extra exposure times for master darks, e.g. `60,120` |
| `ZSPACE_LOG` | name of the ZSPACE log (else auto-detected) |
| `LIM_FLUX`, `LIM_FWHM`, `LIM_RATIO` | default photometry limits (step 5) |
| `MUL_CONT` | default continuum multiplier (step 7) |

Other keys (`BINNING`, `NIGHT`, `NO_SHIFT`, …) are accepted and ignored.

## Pipeline stages

| # | flag | what it does | main output |
|---|---|---|---|
| 1–3 | `--prepare-calib`, `--cr` | master darks and flat, dark/flat correction, cosmic-ray cleaning (L.A.Cosmic), background subtraction | `<obj>_darkflat.fits` → `<obj>_crclean.fits` |
| 4 | `--shift` | registration of the frames by reference stars | `<obj>_shift.fits` |
| 5 | `--phot` | photometry of the stars, rejection of frames with bad transparency / seeing | `<obj>_phot.fits` |
| 6 | `--linecont` | sum of the frames into emission-line and continuum layers | `<obj>_linecont.fits` |
| 7 | `--net` | continuum subtraction `net = line − k·cont` | `<obj>_<line>_net.fits` |
| 7 | `--astro` | astrometry (astrometry.net), rotation to North up / East left | `*_NE.fits` |
| 8 | `--qe-calib` | QE / flux calibration from a standard star | `QE_*.json` |
| 8 | `--apply-ergs` | flux calibration of the object | `*_net_NE_erg.fits` |
| — | `--cr-final` | an extra L.A.Cosmic pass on the final `*_NE` images | (in place) |

Stages can be run together or one by one: each stage takes the newest product of the previous one from `W_DIR`.
`--net`, `--astro` and `--apply-ergs` alone need neither the logs nor the raw data, only the earlier products.

### Several cubes

`CUB_OBJ = '04,05'` (or several cubes ticked in the GUI) merges the cubes into **one** cube at the dark/flat stage:
frames of cube 04 first, then of cube 05 (in the GUI: in increasing order), each in the order of the observation log.
All later stages run once on the merged cube; the first frame of the first cube is the reference for the shift.
Frames with the same ZSPACE from different cubes go into the same line / continuum layer.

One master dark and one master flat are used, so the cubes must have the **same filter and exposure time**;
otherwise the run stops and lists the filter / exposure of every cube. Process such cubes one by one
(tick one cube, or give one value of `CUB_OBJ`) with a different `write_dir` for each.

### 1–3. Dark, flat, cosmic rays, background

Master darks (for the object, flat and `DARKS` exposures) and the master flat are built, every frame is corrected
(`<obj>_darkflat.fits`) and cleaned of cosmic rays (`<obj>_crclean.fits`; L.A.Cosmic inside the `MASK_R` circle,
`--sigclip`, `--objlim`, …).

At the end of this step the background is subtracted from every frame, once
(`--background-mode backgr_sub`, default; window `--background-window`, 30 px; the circle of `--shift-center-radius`
around `MASK_XC/MASK_YC` is kept out of the background estimate). It takes about a second per frame and is done
here, not in the shift stage, so that the shift can be re-run with other parameters without recomputing it.
The cube is marked with `BGMODE` in the header;
`--background-mode none` switches it off. In the GUI: box *background*.

### 4. Shift

Reference stars are detected automatically (`--shift-detect-snr`, `--shift-search-half`; stars within
`--shift-center-radius` px of the mask centre are skipped). An interactive window lets you edit the star list:

- click a star — exclude / restore it;
- click an empty place — add a star (snapped to the nearest peak, shown in orange as `M`);
- right click on an `M` star or `U` — remove the last added star;
- Enter / Esc — continue.

While this window is open the log shows `[shift stars] WAITING …` and the run does not go on:
if nothing seems to happen at the shift stage, look for the window *shift stars* (it may open behind other windows).

`--shift-skip-frames "3,7-9"` excludes frames (1-based) from the shift measurement; their shift is interpolated.
The final star list (`<obj>_shift.stars.xy`) is reused by photometry and continuum scaling.
Other files: `<obj>_shift.shifts.csv` (shift of every frame), `<obj>_shift.stars.png`, `<obj>_shift_frames/`.

### 5. Photometry

The same stars are measured in every frame; frames below the limits are dropped:
`--win`, `--lim-flux`, `--lim-fwhm`, `--lim-ratio` (for standards: `--std-*`).
Report: `<obj>_phot.csv` (good-frame flags) and the plot `<obj>_phot.png`.

### 6. Line / continuum cube

Frames are grouped by the labels of the ZSPACE log:
one layer per emission line (all its ZSPACE values together) and one per continuum position:

```
Zspace=-1807 cont          → cont1
Zspace=-1500 Halpha  ┐
Zspace=-1495 Halpha  ┘     → Ha
Zspace=-1486 [NII]         → NII
Zspace=-1200 cont          → cont2
```

`--lc-group-by label` groups the continuum by label instead of by ZSPACE;
`--no-lc-split` makes only two layers, `line` and `cont`. The layer table is stored in the header.

### 7. Continuum subtraction and astrometry

The combined continuum (exposure-weighted mean of the continuum layers) is subtracted from every line:

- `--mul-cont k` or `--mul-cont-list k1,k2` (one per line) — fixed multiplier;
- `--auto-mul-cont` — `k` from the stars; `--local-mu plane|quadratic|radial|y` — spatially varying `k`;
- the continuum is registered to the line image by an affine fit to the stars (`--no-register-cont` to disable);
- `--star-match` — cosmetic subtraction of each star with its own scale, shift and PSF width;
  `--star-protect X,Y,R` protects the object. Do not use it for photometry of compact emission sources.

`--astro` needs the personal API key in `ASTROMETRY_NET_API_KEY` (see [Astrometry API key](#astrometry-api-key);
another variable can be named by `--astro-env`). The combined continuum is solved and its WCS is given to all
line and net images.

- RA/Dec of the header (coordinates of the date, `EPOCH`) are precessed to J2000 and used as a hint
  (radius 0.5°, pixel scale `PIXSCALE` ±15 %).
- A cleaned crop of the image is uploaded (NaN outside the mask replaced by the sky level); three attempts:
  position + scale, scale only, blind.
- The rotation to North up keeps NaN outside the mask out of the interpolation, so the `_NE` images stay valid.

### 8. Flux calibration

1. Reduce a standard star: `--standard --cr --shift --phot --linecont --qe-calib` → `QE_*.json`.
2. Calibrate the object: `--apply-ergs` (the JSON is searched near `R_DIR`, or given by `--ergcal-json`).

Reference data shipped in `data/`:

- `data/FiltersBTA/` — transmission curves of the filters (`SED675.txt`, `MA500.txt`, …; the name is the `FILTERS` value);
- `data/okestan/` — spectral energy distributions of spectrophotometric standards (`m<star>.dat`, AB magnitudes);
  the GUI uses this list to recognise standard stars in the observation log.

Folders `FiltersBTA/` and `okestan/` (or `okestan.zip`) placed in `R_DIR` or its parent take precedence;
other locations can be given by `--filters-dir` / `MANGAL_FILTERS_DIR` and `--okestan-dir` / `MANGAL_OKESTAN_DIR`.
The `*.calib` file of the night is taken from `R_DIR` (or `--calib-file`).

## Output files

Each intermediate cube carries one stage label, replaced at every stage:

```
<obj>_darkflat → <obj>_crclean → <obj>_shift → <obj>_phot → <obj>_linecont   (.fits)
```

Final 2D images:

| file | content |
|---|---|
| `<obj>_cont.fits`, `<obj>_cont1.fits`, … | combined / individual continuum |
| `<obj>_<line>.fits` | emission line before subtraction, e.g. `NGC7803_NII.fits` |
| `<obj>_<line>_net.fits` | continuum-subtracted line |
| `…_NE.fits` | + astrometry, North up / East left (`NGC7803_NII_NE.fits`, `NGC7803_NII_net_NE.fits`) |
| `…_net_NE_erg.fits` | + flux calibration |
| `<obj>_<line>_net_subtract.png` | QA picture of the subtraction (red — residual emission, blue — over-subtracted) |

`<line>` is `Ha`, `NII`, `5007` for [O III], or the first word of the ZSPACE-log label. Without a line name
(one unrecognised line, or `--no-lc-split`) the images are `<obj>_line.fits` and `<obj>_net.fits`.
Line images of older versions (`<obj>_<line>_line.fits`) are still found as inputs of `--astro`.

Side files of a cube are named after it without `.fits`: `<obj>_darkflat.inputs.csv` (ZSPACE, EXPTIME, … of every
frame, copied along the stages), `<obj>_linecont.csv` (slice report). Side files of older versions
(`<obj>_shift.fits.stars.xy`, `….fits.inputs.csv`, `….fits.phot.csv`) are still read.

Products that are sums of many frames get `EXPTIME`, `NFRAMES`, `ZMEAN` (effective zenith distance),
`TLINE` / `TCONT` (for net images) and a `HISTORY` list of the frames used; frames rejected by photometry are not counted.

## Troubleshooting

**The star-selection window of the shift stage does not appear.** The window is drawn by matplotlib in the
`run_mangal.py` process. If matplotlib uses a backend that cannot open windows (`Agg`, or the *inline* backend that
Jupyter, VS Code or Spyder put into `MPLBACKEND`), the pipeline switches to `macosx` / `QtAgg` / `TkAgg` itself and
says so in the log. If that is not possible, it uses the automatically found stars and prints an error.
Starting the GUI from a plain terminal avoids the problem:

```bash
unset MPLBACKEND          # or: export MPLBACKEND=QtAgg
python mangal_qt6_gui.py
```

**Nothing happens for a while.** Background subtraction at step 1 takes about a second per frame
(progress lines `[bg] N/M`). At the shift stage the log shows `[shift stars] WAITING …` until the star window is
closed with Enter / Esc.

**The shift log says "the input cube has no background subtraction".** The cube was made by an older version or
with `--background-mode none`: run step 1 again to subtract the background.

## Project layout

| file | role |
|---|---|
| `mangal_qt6_gui.py` | GUI; runs `run_mangal.py` with the chosen stages |
| `run_mangal.py` | pipeline driver (command line), calls the stage modules in order |
| `mangal_step3.py` | param file and log parsing, darks, flat, CR cleaning |
| `mangal_step4_shift.py` | frame registration |
| `mangal_step5_photometry.py` | star photometry, frame selection |
| `mangal_step6_linecont.py` | line / continuum cube |
| `mangal_step7_net_astro.py` | continuum subtraction (incl. continuum registration and per-star match), astrometry |
| `mangal_step8_ergcal.py` | QE and flux calibration |
| `mangal_standards.py` | list of spectrophotometric standards |
| `mangal_header.py`, `mangal_names.py` | FITS header keywords, file naming |
| `data/FiltersBTA/`, `data/okestan/` | filter curves, standard-star SEDs |
| `examples/` | example parameter file |
| `LICENSE` | MIT license |

## License

[MIT](LICENSE)
