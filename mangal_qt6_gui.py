#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
from pathlib import Path
import re
import queue
import subprocess
import threading
import time


from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QListWidget,
    QToolButton,
)
from PyQt6.QtCore import QLocale, QSize, Qt
from PyQt6.QtGui import QFont, QStandardItem

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mangal_step3 import parse_obs_log_table, read_author, select_cube_rows, cube_ids_list, object_rows
from run_mangal import find_log_auto
from mangal_standards import load_mangal_standards_for_night, normalize_star_id


def extract_cube_id_from_file(fn: str) -> str:
    m = re.search(r"_([0-9]{2})[0-9]{2}$", fn)
    if not m:
        raise ValueError(f"Cannot extract cube id from filename: {fn}")
    return m.group(1)


def _cube_num_from_file(fn: str) -> int | None:
    try:
        return int(extract_cube_id_from_file(fn))
    except Exception:
        return None


def auto_params_from_log(r_dir: Path, object_name: str, cube_obj: str | None, w_dir: Path | None):
    r_dir = Path(r_dir)
    if w_dir is None:
        w_dir = r_dir.parent / object_name
    w_dir.mkdir(parents=True, exist_ok=True)

    log_path = find_log_auto(r_dir)
    df = parse_obs_log_table(log_path)
    type_col = df["TYPE"].astype(str).str.lower()

    obj_rows = df[(type_col == "obj") & (df["OBJECT"] == object_name)].copy()
    if obj_rows.empty:
        raise RuntimeError(f"No OBJECT={object_name!r} with TYPE='obj' in {log_path}")

    if cube_obj:
        cube_obj = ",".join(cube_ids_list(cube_obj))
        obj_sel = object_rows(df, object_name, cube_obj)  # may include cubes of another OBJECT name
        if not obj_sel.empty:
            obj_rows = obj_sel
    else:
        cube_obj = extract_cube_id_from_file(str(obj_rows.iloc[0]["FILE"]))

    if "FILTERS" in obj_rows.columns:
        exp_filter = str(obj_rows.iloc[0]["FILTERS"]).strip()
    else:
        exp_filter = ""

    flat_rows = df[type_col == "flat"].copy()
    if exp_filter and "FILTERS" in flat_rows.columns:
        fr2 = flat_rows[flat_rows["FILTERS"].astype(str).str.strip() == exp_filter]
        if not fr2.empty:
            flat_rows = fr2
    if flat_rows.empty:
        raise RuntimeError("No flat frames found (TYPE='flat') for this night / filter")

    cube_flat = extract_cube_id_from_file(str(flat_rows.iloc[0]["FILE"]))

    texp_obj = int(obj_rows.iloc[0]["Texp"])
    texp_flat = int(flat_rows.iloc[0]["Texp"])

    params: dict = {}
    params["BINNING"] = "1x1"
    params["NIGHT"] = log_path.stem + "_"
    params["R_DIR"] = str(r_dir)
    params["W_DIR"] = str(w_dir)
    params["CCD_CUT"] = "0,0,1599,1599"
    params["FILTERS"] = exp_filter
    params["CUB_OBJ"] = cube_obj
    params["CUB_FLAT"] = cube_flat
    params["NO_SHIFT"] = 0
    params["FLAT_DX"] = 0
    params["FLAT_DY"] = 0
    params["MASK_R"] = 800
    params["MASK_XC"] = 790
    params["MASK_YC"] = 780
    params["TEXPNORM"] = 1
    params["CCD_CUT_TUPLE"] = (0, 0, 1599, 1599)

    meta = dict(
        log_path=log_path,
        texp_obj=texp_obj,
        texp_flat=texp_flat,
    )
    return params, meta


def write_param_file(params: dict, path: Path):
    lines = []
    lines.append("SIMPLE  =                              T / MaNGaL DATA REDUCTION")
    lines.append("")
    for key in (
        "BINNING", "NIGHT", "R_DIR", "W_DIR", "CCD_CUT", "FILTERS", "CUB_OBJ", "CUB_FLAT",
        "NO_SHIFT", "FLAT_DX", "FLAT_DY", "MASK_R", "MASK_XC", "MASK_YC", "TEXPNORM"
    ):
        if key not in params:
            continue
        v = params[key]
        if isinstance(v, str):
            lines.append(f"{key:<8}= '{v}'")
        else:
            lines.append(f"{key:<8}= {v}")
    lines.append("")
    lines.append("END")
    path.write_text("\n".join(lines), encoding="utf-8")


def _time_to_seconds(s):
    try:
        parts = str(s).strip().split(":")
        if len(parts) != 3:
            return None
        hh, mm, ss = [int(x) for x in parts]
        return 3600 * hh + 60 * mm + ss
    except Exception:
        return None


class CheckableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        if self.lineEdit() is not None:
            self.lineEdit().setReadOnly(True)
        self.view().pressed.connect(self._on_item_pressed)

    def add_check_item(self, text: str, data=None, checked: bool = False):
        item = QStandardItem(text)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        item.setData(data if data is not None else text, Qt.ItemDataRole.UserRole)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.model().appendRow(item)
        self._update_text()

    def clear(self):
        super().clear()
        self._update_text()

    def _on_item_pressed(self, index):
        item = self.model().itemFromIndex(index)
        if item is None:
            return
        item.setCheckState(Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked)
        self._update_text()

    def checked_texts(self) -> list[str]:
        out = []
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                out.append(item.text())
        return out

    def checked_data(self) -> list[str]:
        out = []
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                out.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return out

    def currentText(self) -> str:
        return ', '.join(self.checked_texts())


    def _update_text(self):
        le = self.lineEdit()
        if le is not None:
            le.setText(', '.join(self.checked_texts()))


class CollapsibleBox(QWidget):
    """A titled panel that folds away: click the title to show / hide the contents (folded by default)."""

    def __init__(self, title: str, content: QWidget, expanded: bool = False, parent=None):
        super().__init__(parent)
        self._title = title
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setIconSize(QSize(8, 8))  # small arrow
        self.toggle.setStyleSheet("QToolButton { border: none; }")
        self.content = content
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.toggle)
        lay.addWidget(self.content)
        self.toggle.toggled.connect(self._set_expanded)
        self._set_expanded(expanded)

    def set_title(self, title: str):
        self._title = title
        self.toggle.setText(title)

    def _set_expanded(self, on: bool):
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.toggle.setText(self._title)
        self.content.setVisible(on)


class MangalQt(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MaNGaL Qt6 frontend")
        self.resize(980, 680)
        self.setLocale(QLocale.c())

        self.rdir_edit = QLineEdit()
        self.btn_rdir = QPushButton("open...")
        self.object_combo = QComboBox()
        self.author_label = QLabel("author: —")
        self.author_label.setStyleSheet("font-weight: 600;")
        self.author_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.standard_combo = QComboBox()
        self.cube_combo = CheckableComboBox()
        self.wdir_edit = QLineEdit()

        self.chk_prepare_cube = QCheckBox("masterbias + masterflat + CR-clean")
        self.chk_shift = QCheckBox("shift")
        self.chk_phot_auto = QCheckBox("photometry")
        self.chk_linecont = QCheckBox("line+cont")
        self.chk_net = QCheckBox("net")
        self.chk_astro = QCheckBox("astrometry")
        self.chk_qe = QCheckBox("QE calibration")
        self.chk_apply_ergs = QCheckBox("apply ergs")
        self.chk_standard = QCheckBox("standard")
        self.chk_combine = QCheckBox("combine this night with the nights below")

        # combination of nights: W_DIRs of the same object reduced (up to line+cont) on other nights
        self.combine_list = QListWidget()
        self.combine_list.setMaximumHeight(70)
        self.combine_list.setToolTip("write_dir of the same object on other nights (reduced up to line+cont)")
        self.btn_combine_add = QPushButton("+ night")
        self.btn_combine_del = QPushButton("− remove")
        self.combine_scale_combo = QComboBox()
        self.combine_scale_combo.addItems(["both", "stars", "standards"])
        self.chk_combine_manual = QCheckBox("manual star pairs")

        self.chk_prepare_cube.setChecked(True)
        self.chk_shift.setChecked(True)
        self.chk_phot_auto.setChecked(True)
        self.chk_linecont.setChecked(True)
        self.chk_net.setChecked(True)
        self.chk_astro.setChecked(True)
        self.chk_qe.setChecked(False)
        self.chk_apply_ergs.setChecked(False)

        self.shift_detect_snr = QDoubleSpinBox()
        self.shift_detect_snr.setRange(0.5, 500.0)
        self.shift_detect_snr.setDecimals(1)
        self.shift_detect_snr.setSingleStep(0.5)
        self.shift_detect_snr.setValue(20.0)

        self.shift_search_half = QSpinBox()
        self.shift_search_half.setRange(3, 300)
        self.shift_search_half.setValue(30)

        self.chk_shift_manual = QCheckBox("manual star pairs (between cubes)")
        self.chk_shift_manual.setToolTip("merged cubes: click star pairs to register every other cube to the reference frame")
        self.shift_skip_frames = QLineEdit()
        self.shift_skip_frames.setPlaceholderText("e.g. 3, 7-9")
        self.shift_skip_frames.setToolTip(
            "Frames NOT used when measuring the shifts (numbers as in the '[shift stars] N/M' log lines, "
            "counted from 1). They stay in the cube; their shift is interpolated from the neighbouring frames."
        )
        self.shift_center_r = QDoubleSpinBox()
        self.shift_center_r.setRange(0.0, 2000.0)
        self.shift_center_r.setDecimals(1)
        self.shift_center_r.setSingleStep(5.0)
        self.shift_center_r.setValue(250.0)

        self.win_spin = QSpinBox()
        self.win_spin.setRange(3, 200)
        self.win_spin.setValue(50)

        self.lim_flux_spin = QDoubleSpinBox()
        self.lim_flux_spin.setRange(0.0, 10.0)
        self.lim_flux_spin.setDecimals(2)
        self.lim_flux_spin.setSingleStep(0.01)
        self.lim_flux_spin.setValue(0.00)

        self.lim_fwhm_spin = QDoubleSpinBox()
        self.lim_fwhm_spin.setRange(0.0, 1000.0)
        self.lim_fwhm_spin.setDecimals(1)
        self.lim_fwhm_spin.setValue(20.0)

        self.lim_ratio_spin = QDoubleSpinBox()
        self.lim_ratio_spin.setRange(0.0, 1.0)
        self.lim_ratio_spin.setDecimals(2)
        self.lim_ratio_spin.setSingleStep(0.01)
        self.lim_ratio_spin.setValue(0.00)


        self.std_win_spin = QSpinBox()
        self.std_win_spin.setRange(3, 200)
        self.std_win_spin.setValue(50)

        self.std_lim_flux_spin = QDoubleSpinBox()
        self.std_lim_flux_spin.setRange(0.0, 10.0)
        self.std_lim_flux_spin.setDecimals(2)
        self.std_lim_flux_spin.setSingleStep(0.01)
        self.std_lim_flux_spin.setValue(0.00)

        self.std_lim_fwhm_spin = QDoubleSpinBox()
        self.std_lim_fwhm_spin.setRange(0.0, 1000.0)
        self.std_lim_fwhm_spin.setDecimals(1)
        self.std_lim_fwhm_spin.setValue(20.0)

        self.std_lim_ratio_spin = QDoubleSpinBox()
        self.std_lim_ratio_spin.setRange(0.0, 1.0)
        self.std_lim_ratio_spin.setDecimals(2)
        self.std_lim_ratio_spin.setSingleStep(0.01)
        self.std_lim_ratio_spin.setValue(0.00)

        self.mul_cont_spin1 = QDoubleSpinBox()
        self.mul_cont_spin1.setRange(0.1, 10.0)
        self.mul_cont_spin1.setDecimals(2)
        self.mul_cont_spin1.setSingleStep(0.01)
        self.mul_cont_spin1.setValue(1.00)

        self.mul_cont_spin2 = QDoubleSpinBox()
        self.mul_cont_spin2.setRange(0.1, 10.0)
        self.mul_cont_spin2.setDecimals(2)
        self.mul_cont_spin2.setSingleStep(0.01)
        self.mul_cont_spin2.setValue(1.00)

        self.bg_mode_combo = QComboBox()
        self.bg_mode_combo.addItems(["none", "fon_corners", "backgr_sub"])
        self.bg_mode_combo.setCurrentText("backgr_sub")
        self.bg_window_spin = QSpinBox()
        self.bg_window_spin.setRange(3, 500)
        self.bg_window_spin.setValue(30)

        self.auto_mul_chk = QCheckBox("auto μ from stars")
        self.auto_mul_chk.setChecked(True)
        self.local_mu_combo = QComboBox()
        self.local_mu_combo.addItems(["global", "plane", "quadratic", "radial", "y"])
        self.star_match_chk = QCheckBox("per-star match (each star with its own k)")
        self.star_match_chk.setChecked(True)
        self.star_match_chk.setToolTip("Cosmetic: every point source is subtracted with its own flux ratio, shift and PSF width.\n"
                                       "Stars have different colours, so a single k always leaves +/- residuals.\n"
                                       "An unresolved source with real line emission is kept only if its ratio is far from k "
                                       "(see *_star_match.csv).")

        for spin in (
            self.shift_detect_snr,
            self.shift_center_r,
            self.lim_flux_spin,
            self.lim_fwhm_spin,
            self.lim_ratio_spin,
            self.std_lim_flux_spin,
            self.std_lim_fwhm_spin,
            self.std_lim_ratio_spin,
            self.mul_cont_spin1,
            self.mul_cont_spin2,
        ):
            spin.setLocale(QLocale.c())
            spin.lineEdit().setPlaceholderText("0.0")

        self.btn_run = QPushButton("run")
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)

        self._df_log = None

        self._build_layout()
        self._connect_signals()

    def _build_layout(self):
        layout = QVBoxLayout(self)

        h_rdir = QHBoxLayout()
        h_rdir.addWidget(QLabel("read_dir:"))
        h_rdir.addWidget(self.rdir_edit)
        h_rdir.addWidget(self.btn_rdir)
        layout.addLayout(h_rdir)

        h_obj = QHBoxLayout()
        h_obj.addWidget(QLabel("object:"))
        h_obj.addWidget(self.object_combo, 2)
        h_obj.addWidget(QLabel("standard:"))
        h_obj.addWidget(self.standard_combo, 3)
        h_obj.addWidget(QLabel("cub_obj:"))
        h_obj.addWidget(self.cube_combo, 1)
        layout.addLayout(h_obj)

        layout.addWidget(self.author_label)

        h_wdir = QHBoxLayout()
        h_wdir.addWidget(QLabel("write_dir:"))
        h_wdir.addWidget(self.wdir_edit)
        layout.addLayout(h_wdir)

        steps_layout = QHBoxLayout()
        col1 = QVBoxLayout()
        col2 = QVBoxLayout()
        col3 = QVBoxLayout()
        col1.addWidget(self.chk_prepare_cube)
        col1.addWidget(self.chk_shift)
        col1.addWidget(self.chk_phot_auto)
        col2.addWidget(self.chk_linecont)
        col2.addWidget(self.chk_net)
        col2.addWidget(self.chk_astro)
        col3.addWidget(self.chk_standard)
        col3.addWidget(self.chk_qe)
        col3.addWidget(self.chk_apply_ergs)
        steps_layout.addLayout(col1)
        steps_layout.addLayout(col2)
        steps_layout.addLayout(col3)
        layout.addLayout(steps_layout)

        params_grid = QGridLayout()
        params_grid.setHorizontalSpacing(18)
        params_grid.setVerticalSpacing(10)

        shift_box = QGroupBox("shift parameters")
        grid_shift = QGridLayout(shift_box)
        grid_shift.addWidget(QLabel("threshold"), 0, 0)
        grid_shift.addWidget(self.shift_detect_snr, 0, 1)
        grid_shift.addWidget(QLabel("window"), 1, 0)
        grid_shift.addWidget(self.shift_search_half, 1, 1)
        grid_shift.addWidget(QLabel("center exclusion r"), 2, 0)
        grid_shift.addWidget(self.shift_center_r, 2, 1)
        grid_shift.addWidget(QLabel("skip frames"), 3, 0)
        grid_shift.addWidget(self.shift_skip_frames, 3, 1)
        grid_shift.addWidget(self.chk_shift_manual, 4, 0, 1, 2)

        bg_box = QGroupBox("background")
        grid_bg = QGridLayout(bg_box)
        grid_bg.addWidget(QLabel("mode"), 0, 0)
        grid_bg.addWidget(self.bg_mode_combo, 0, 1)
        grid_bg.addWidget(QLabel("window"), 1, 0)
        grid_bg.addWidget(self.bg_window_spin, 1, 1)
        grid_bg.addWidget(QLabel("center exclusion r: from shift"), 2, 0, 1, 2)

        net_box = QGroupBox("net parameters")
        grid_net = QGridLayout(net_box)
        grid_net.addWidget(QLabel("mul[1]"), 0, 0)
        grid_net.addWidget(self.mul_cont_spin1, 0, 1)
        grid_net.addWidget(QLabel("mul[2]"), 0, 2)
        grid_net.addWidget(self.mul_cont_spin2, 0, 3)
        grid_net.addWidget(self.auto_mul_chk, 1, 0, 1, 2)
        grid_net.addWidget(QLabel("scale mode"), 1, 2)
        grid_net.addWidget(self.local_mu_combo, 1, 3)
        grid_net.addWidget(self.star_match_chk, 2, 0, 1, 4)

        phot_box = QGroupBox()
        grid_phot = QGridLayout(phot_box)
        grid_phot.addWidget(QLabel("win"), 0, 0)
        grid_phot.addWidget(self.win_spin, 0, 1)
        grid_phot.addWidget(QLabel("lim-flux"), 0, 2)
        grid_phot.addWidget(self.lim_flux_spin, 0, 3)
        grid_phot.addWidget(QLabel("lim-fwhm"), 1, 0)
        grid_phot.addWidget(self.lim_fwhm_spin, 1, 1)
        grid_phot.addWidget(QLabel("lim-ratio"), 1, 2)
        grid_phot.addWidget(self.lim_ratio_spin, 1, 3)
        grid_phot.addWidget(QLabel("std-win"), 2, 0)
        grid_phot.addWidget(self.std_win_spin, 2, 1)
        grid_phot.addWidget(QLabel("std-lim-flux"), 2, 2)
        grid_phot.addWidget(self.std_lim_flux_spin, 2, 3)
        grid_phot.addWidget(QLabel("std-lim-fwhm"), 3, 0)
        grid_phot.addWidget(self.std_lim_fwhm_spin, 3, 1)
        grid_phot.addWidget(QLabel("std-lim-ratio"), 3, 2)
        grid_phot.addWidget(self.std_lim_ratio_spin, 3, 3)

        params_grid.addWidget(bg_box, 0, 0)
        params_grid.addWidget(shift_box, 0, 1)
        params_grid.addWidget(net_box, 0, 2)
        # changed rarely: folded away by default (click the title to open)
        self.phot_panel = CollapsibleBox("photometry parameters", phot_box, expanded=False)
        params_grid.addWidget(self.phot_panel, 1, 0, 1, 3)

        combine_box = QGroupBox()
        grid_comb = QGridLayout(combine_box)
        grid_comb.addWidget(self.chk_combine, 0, 0, 1, 4)
        grid_comb.addWidget(QLabel("write_dir of the same object on other nights (reduced up to line+cont):"), 1, 0, 1, 4)
        grid_comb.addWidget(self.combine_list, 2, 0, 3, 1)
        grid_comb.addWidget(self.btn_combine_add, 2, 1)
        grid_comb.addWidget(self.btn_combine_del, 3, 1)
        grid_comb.addWidget(QLabel("scale"), 2, 2)
        grid_comb.addWidget(self.combine_scale_combo, 2, 3)
        grid_comb.addWidget(self.chk_combine_manual, 3, 2, 1, 2)
        grid_comb.setColumnStretch(0, 1)
        # used rarely: folded away; the title shows when it is switched on
        self.combine_panel = CollapsibleBox("combine nights", combine_box, expanded=False)
        params_grid.addWidget(self.combine_panel, 2, 0, 1, 3)
        self.chk_combine.toggled.connect(
            lambda on: self.combine_panel.set_title("combine nights — ON" if on else "combine nights"))
        for col in range(3):
            params_grid.setColumnStretch(col, 1)
        layout.addLayout(params_grid)

        h_buttons = QHBoxLayout()
        h_buttons.addStretch(1)
        h_buttons.addWidget(self.btn_run)
        layout.addLayout(h_buttons)

        layout.addWidget(self.text_log, 1)

    def _connect_signals(self):
        self.btn_rdir.clicked.connect(self.choose_rdir)
        self.object_combo.currentIndexChanged.connect(self._on_object_or_cube_changed)
        self.standard_combo.currentIndexChanged.connect(self._sync_wdir_for_mode)
        self.chk_standard.toggled.connect(self._on_standard_toggled)
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_combine_add.clicked.connect(self._add_combine_night)
        self.btn_combine_del.clicked.connect(self._remove_combine_night)

    def _on_object_or_cube_changed(self):
        self.update_cubes_for_object()
        self._sync_wdir_for_mode()
    def _on_standard_toggled(self, checked: bool):
        if checked:
            # usual stages of a standard-star reduction (each can still be unticked: only ticked stages run)
            self.chk_prepare_cube.setChecked(True)
            self.chk_shift.setChecked(True)
            self.chk_phot_auto.setChecked(True)
            self.chk_linecont.setChecked(True)
            # not applicable to a standard star
            self.chk_combine.setChecked(False)
            self.chk_net.setChecked(False)
            self.chk_astro.setChecked(False)
            self.chk_apply_ergs.setChecked(False)
        self._sync_wdir_for_mode()

    def log(self, msg: str):
        self.text_log.append(msg)
        cursor = self.text_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.text_log.setTextCursor(cursor)

    def choose_rdir(self):
        d = QFileDialog.getExistingDirectory(self, "Select the night directory (read_dir)")
        if d:
            self.rdir_edit.setText(d)
            self.update_objects()

    def _current_object_name(self) -> str:
        return self.object_combo.currentText().strip()

    def _current_standard_selection(self):
        idx = self.standard_combo.currentIndex()
        if idx < 0:
            return None
        return self.standard_combo.itemData(idx)

    def _sync_wdir_for_mode(self):
        rdir_text = self.rdir_edit.text().strip()
        if not rdir_text:
            return
        rdir = Path(rdir_text)
        obj_name = self._current_object_name()
        std_sel = self._current_standard_selection()

        current_text = self.wdir_edit.text().strip()
        current_path = Path(current_text) if current_text else None
        obj_default = rdir.parent / obj_name if obj_name else None
        std_default = (obj_default / str(std_sel["object_name"])) if (obj_default is not None and std_sel) else None

        if self.chk_standard.isChecked() and std_default is not None:
            if current_path is None or current_path == obj_default:
                self.wdir_edit.setText(str(std_default))
        elif not self.chk_standard.isChecked() and obj_default is not None:
            if current_path is None or current_path == std_default:
                self.wdir_edit.setText(str(obj_default))

    def update_objects(self):
        rdir_text = self.rdir_edit.text().strip()
        if not rdir_text:
            return
        rdir = Path(rdir_text)
        if not rdir.is_dir():
            self.log(f"[err] read_dir is not selected: {rdir}")
            return
        try:
            log_path = find_log_auto(rdir)
            df = parse_obs_log_table(log_path)
            self._df_log = df
        except Exception as e:
            self.log(f"[err] cannot read the log: {e}")
            self._df_log = None
            return

        std_ids = load_mangal_standards_for_night(rdir)
        self.log(f"[info] standards loaded: {len(std_ids)}")
        if std_ids:
            preview_std = ", ".join(sorted(std_ids)[:20])
            self.log("[info] okestan ids: " + preview_std + (" ..." if len(std_ids) > 20 else ""))

        type_col = df["TYPE"].astype(str).str.lower()
        mask_obj = type_col.eq("obj")
        norm_object = df["OBJECT"].astype(str).map(normalize_star_id)
        mask_std = norm_object.isin(std_ids)

        all_log_objects = sorted(set(df.loc[mask_obj, "OBJECT"]))
        self.log("[info] log OBJECTs: " + ", ".join(map(str, all_log_objects[:30])) + (" ..." if len(all_log_objects) > 30 else ""))

        recognized_std = sorted(set(df.loc[mask_obj & mask_std, "OBJECT"]))
        if recognized_std:
            self.log("[info] standards in log: " + ", ".join(map(str, recognized_std)))
        else:
            self.log("[warn] no log objects matched okestan standards")

        objs = sorted(set(df.loc[mask_obj & ~mask_std, "OBJECT"]))
        self.object_combo.blockSignals(True)
        self.object_combo.clear()
        for o in objs:
            self.object_combo.addItem(str(o))
        self.object_combo.blockSignals(False)

        if objs:
            obj0 = str(objs[0])
            self.wdir_edit.setText(str(rdir.parent / obj0))
        else:
            self.wdir_edit.clear()
            self.author_label.setText("author: —")

        self.update_cubes_for_object()

    def update_cubes_for_object(self):
        df = self._df_log
        if df is None:
            return
        obj = self._current_object_name()
        if not obj:
            return

        type_col = df["TYPE"].astype(str).str.lower()
        obj_all = df[type_col == "obj"]
        rows = obj_all[obj_all["OBJECT"] == obj]

        def _cubes(r):
            out = {}
            for fn, name in zip(r["FILE"].astype(str), r["OBJECT"].astype(str)):
                try:
                    out.setdefault(extract_cube_id_from_file(fn), name)
                except Exception:
                    continue
            return out

        own = _cubes(rows)
        # cubes of other objects with the same filter can be added (unticked): e.g. PGC45443_dz to PGC45443,
        # a cube observed again under another name; names that start with the object name come first
        filt = set(rows["FILTERS"].astype(str).str.strip()) if "FILTERS" in rows.columns else set()
        others_rows = obj_all[obj_all["OBJECT"] != obj]
        if filt and "FILTERS" in others_rows.columns:
            others_rows = others_rows[others_rows["FILTERS"].astype(str).str.strip().isin(filt)]
        others = {c: n for c, n in _cubes(others_rows).items() if c not in own}
        base = obj.lower()
        others_sorted = sorted(others.items(), key=lambda cn: (not cn[1].lower().startswith(base), cn[0]))

        self.cube_combo.blockSignals(True)
        self.cube_combo.clear()
        for cid in sorted(own):
            self.cube_combo.add_check_item(cid, cid, checked=True)
        for cid, name in others_sorted:
            self.cube_combo.add_check_item(f"{cid} · {name}", cid, checked=False)
        self.cube_combo.blockSignals(False)
        try:
            self.cube_combo._update_text()
        except Exception:
            pass

        rdir_text = self.rdir_edit.text().strip()
        if rdir_text and obj:
            self.wdir_edit.setText(str(Path(rdir_text).parent / obj))

        self.update_standard_candidates()
        self._sync_wdir_for_mode()
        self._update_author(rows["FILE"].astype(str).tolist())

    def _update_author(self, files):
        """Show the applicant (AUTHOR keyword of the raw FITS header) of the selected object."""
        rdir_text = self.rdir_edit.text().strip()
        author = None
        if rdir_text:
            for ext in (".fts", ".fits"):
                author = read_author(rdir_text, files, ext=ext)
                if author:
                    break
        if author:
            self.author_label.setText(f"author: {author}")
            self.log(f"[info] AUTHOR: {author}")
        else:
            self.author_label.setText("author: — (AUTHOR not found in header)")

    def update_standard_candidates(self):
        df = self._df_log
        self.standard_combo.clear()

        rdir_text = self.rdir_edit.text().strip()
        rdir = Path(rdir_text) if rdir_text else None

        if df is None:
            return

        obj = self._current_object_name()
        selected_cubes = [c.strip() for c in self.cube_combo.checked_data() if str(c).strip()]
        cube = selected_cubes[0] if selected_cubes else ""
        if not obj:
            return

        type_col = df["TYPE"].astype(str).str.lower()
        obj_rows = df[(type_col == "obj") & (df["OBJECT"] == obj)].copy()
        if obj_rows.empty:
            return

        if cube:
            sel = select_cube_rows(obj_rows, cube)
            if not sel.empty:
                obj_rows = sel

        exp_filter = ""
        if "FILTERS" in obj_rows.columns:
            exp_filter = str(obj_rows.iloc[0]["FILTERS"]).strip()

        obj_cube_num = None
        if cube:
            try:
                obj_cube_num = int(cube)
            except Exception:
                obj_cube_num = None
        if obj_cube_num is None:
            obj_cube_num = _cube_num_from_file(str(obj_rows.iloc[0]["FILE"]))

        obj_start = str(obj_rows.iloc[0]["START"]).strip() if ("START" in obj_rows.columns and not obj_rows.empty) else None
        obj_t = _time_to_seconds(obj_start)

        std_ids = load_mangal_standards_for_night(rdir)
        norm_object = df["OBJECT"].astype(str).map(normalize_star_id)

        std_rows = df[(type_col == "obj") & norm_object.isin(std_ids)].copy()
        if exp_filter and "FILTERS" in std_rows.columns:
            std_rows = std_rows[std_rows["FILTERS"].astype(str).str.strip() == exp_filter]

        candidates: list[dict] = []
        for _, row in std_rows.iterrows():
            std_name = str(row["OBJECT"]).strip()
            fn = str(row["FILE"])
            try:
                std_cube = extract_cube_id_from_file(fn)
            except Exception:
                continue

            delta_cube = None
            try:
                if obj_cube_num is not None:
                    delta_cube = abs(int(std_cube) - int(obj_cube_num))
            except Exception:
                delta_cube = None

            std_start = str(row["START"]).strip() if "START" in row.index else None
            std_t = _time_to_seconds(std_start)
            delta_time_sec = abs(std_t - obj_t) if (std_t is not None and obj_t is not None) else None

            candidates.append(
                {
                    "object_name": std_name,
                    "cube_id": std_cube,
                    "filter": exp_filter,
                    "delta_cube": delta_cube,
                    "delta_time_sec": delta_time_sec,
                    "start": std_start,
                    "file": fn,
                }
            )

        if not candidates:
            self.log(f"[warn] no standards found for filter={exp_filter!r}")
            self.standard_combo.addItem("no standard in this filter", None)
            return

        candidates.sort(
            key=lambda x: (
                999999 if x["delta_time_sec"] is None else x["delta_time_sec"],
                9999 if x["delta_cube"] is None else x["delta_cube"],
                int(x["cube_id"]),
                x["object_name"].upper(),
            )
        )

        seen = set()
        unique_candidates = []
        for c in candidates:
            key = (c["object_name"], c["cube_id"])
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(c)

        preferred_idx = 0
        for i, c in enumerate(unique_candidates):
            if c["delta_cube"] in (0, 1):
                preferred_idx = i
                break

        for c in unique_candidates:
            delta_txt = "?" if c["delta_cube"] is None else str(c["delta_cube"])
            dt_txt = "?" if c["delta_time_sec"] is None else f"{int(c['delta_time_sec'] // 60)}m"
            label = f'{c["object_name"]} | Δcube={delta_txt} | Δt={dt_txt}'
            if c["filter"]:
                label += f' | {c["filter"]}'
            self.standard_combo.addItem(label, c)

        self.standard_combo.setCurrentIndex(preferred_idx)

        picked = self.standard_combo.itemData(preferred_idx)
        if picked is not None:
            self.log(
                f'[info] auto-standard: {picked["object_name"]}'
                + (f' filter={picked["filter"]}' if picked["filter"] else "")
                + (f' Δcube={picked["delta_cube"]}' if picked["delta_cube"] is not None else "")
                + (f' Δt={int(picked["delta_time_sec"] // 60)}m' if picked.get("delta_time_sec") is not None else "")
            )


    def _build_extra_args(self):
        args: list[str] = []

        # Every stage runs only if its box is ticked: stages are independent and take the products of the earlier
        # ones from write_dir. A standard star only adds --standard and its own photometry limits.
        standard_mode = self.chk_standard.isChecked()

        if self.chk_prepare_cube.isChecked():
            args += [
                "--prepare-calib",
                "--cr",
                # background is subtracted once here (not at every shift run); the center circle is kept out
                # of the background estimate
                "--background-mode", self.bg_mode_combo.currentText(),
                "--background-window", str(self.bg_window_spin.value()),
                "--shift-center-radius", str(self.shift_center_r.value()),
            ]

        if self.chk_shift.isChecked():
            args += [
                "--shift",
                "--shift-mode", "stars",
                "--shift-detect-snr", str(self.shift_detect_snr.value()),
                "--shift-search-half", str(self.shift_search_half.value()),
                "--shift-center-radius", str(self.shift_center_r.value()),
                "--shift-skip-frames", self.shift_skip_frames.text().strip(),
            ]
            if self.chk_shift_manual.isChecked():
                args.append("--shift-manual-pairs")

        if self.chk_phot_auto.isChecked():
            args += [
                "--phot",
                "--win", str(self.win_spin.value()),
                "--lim-flux", str(self.lim_flux_spin.value()),
                "--lim-fwhm", str(self.lim_fwhm_spin.value()),
                "--lim-ratio", str(self.lim_ratio_spin.value()),
            ]

        if standard_mode:
            args += [
                "--standard",
                "--std-win", str(self.std_win_spin.value()),
                "--std-lim-flux", str(self.std_lim_flux_spin.value()),
                "--std-lim-fwhm", str(self.std_lim_fwhm_spin.value()),
                "--std-lim-ratio", str(self.std_lim_ratio_spin.value()),
            ]

        if self.chk_qe.isChecked():
            args.append("--qe-calib")
        if self.chk_apply_ergs.isChecked():
            args.append("--apply-ergs")
        if self.chk_linecont.isChecked():
            args.append("--linecont")
        if self.chk_net.isChecked():
            args += [
                "--net",
                "--mul-cont", str(self.mul_cont_spin1.value()),
                "--mul-cont-list", f"{self.mul_cont_spin1.value():.2f},{self.mul_cont_spin2.value():.2f}",
            ]
            if self.auto_mul_chk.isChecked():
                args += [
                    "--auto-mul-cont",
                    "--local-mu", self.local_mu_combo.currentText(),
                ]
            if self.star_match_chk.isChecked():
                args.append("--star-match")
        if self.chk_astro.isChecked():
            args.append("--astro")

        nights = self._combine_nights()
        if self.chk_combine.isChecked() and not standard_mode:
            if nights:
                args += ["--combine-with", *nights,
                         "--combine-out", str(self._combine_out_dir(Path(self.wdir_edit.text().strip()),
                                                                    self._current_object_name())),
                         "--combine-scale", self.combine_scale_combo.currentText()]
                if self.chk_combine_manual.isChecked():
                    args.append("--combine-manual")
            else:
                self.log("[warn] combine nights is ticked, but no other night is listed: + night")
        return args

    # ---------------------------------------------------------------- combination of nights
    @staticmethod
    def _combine_out_dir(w_dir: Path, object_name: str) -> Path:
        return w_dir.parent / f"{object_name}_combined"

    def _combine_nights(self) -> list[str]:
        return [self.combine_list.item(i).text() for i in range(self.combine_list.count())]

    def _add_combine_night(self):
        start = self.wdir_edit.text().strip() or self.rdir_edit.text().strip() or str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "write_dir of the object on another night", str(Path(start).parent))
        if not d:
            return
        if d in self._combine_nights():
            return
        if Path(d).resolve() == Path(self.wdir_edit.text().strip() or ".").resolve():
            self.log("[warn] this is the write_dir of the current night: choose the folder of ANOTHER night")
            return
        if not list(Path(d).glob("*_linecont.fits")):
            self.log(f"[warn] no *_linecont.fits in {d}: reduce that night up to line+cont first")
        self.combine_list.addItem(d)
        self.chk_combine.setChecked(True)

    def _remove_combine_night(self):
        for it in self.combine_list.selectedItems():
            self.combine_list.takeItem(self.combine_list.row(it))

    def _resolve_target(self):
        rdir_text = self.rdir_edit.text().strip()
        if not rdir_text:
            raise RuntimeError("read_dir is not set")

        standard_mode = self.chk_standard.isChecked()
        if standard_mode:
            std_sel = self._current_standard_selection()
            if std_sel is None:
                raise RuntimeError("no standard selected")
            target_name = str(std_sel["object_name"])
            target_cube = str(std_sel["cube_id"])
        else:
            target_name = self._current_object_name()
            target_cube = ",".join([c.strip() for c in self.cube_combo.checked_data() if str(c).strip()]) or None

        if not target_name:
            raise RuntimeError("no object selected")

        wdir_text = self.wdir_edit.text().strip()
        obj_default_wdir = Path(rdir_text).parent / self._current_object_name() if self._current_object_name() else None
        if standard_mode and obj_default_wdir is not None:
            target_default_wdir = obj_default_wdir / target_name
        else:
            target_default_wdir = Path(rdir_text).parent / target_name

        if wdir_text:
            chosen_wdir = Path(wdir_text)
            if standard_mode and obj_default_wdir is not None and chosen_wdir == obj_default_wdir:
                chosen_wdir = target_default_wdir
        else:
            chosen_wdir = target_default_wdir

        return Path(rdir_text), standard_mode, target_name, target_cube, chosen_wdir

    def _open_file(self, path: Path):
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif os.name == "nt":
                os.startfile(str(path))
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:
            self.log(f"[warn] cannot open {path}: {e}")

    def _find_existing_ergcal_json(self, root: Path, filter_name: str | None = None) -> Path | None:
        cands = sorted(root.rglob("QE_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if filter_name:
            for p in cands:
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if str(data.get("filter", "")).strip() == str(filter_name).strip():
                        return p
                except Exception:
                    continue
        return cands[0] if cands else None

    def _run_command_live(self, cmd: list[str]) -> int:
        try:
            env = os.environ.copy()
            mpl = env.get("MPLBACKEND", "").strip().lower()
            if mpl in ("agg", "pdf", "ps", "svg", "cairo", "pgf", "template") or mpl.startswith("module://"):
                # inherited from an IDE (Jupyter / VS Code / Spyder): run_mangal could not open its windows
                env.pop("MPLBACKEND")
                self.log(f"[info] ignoring MPLBACKEND={mpl} (cannot open windows)")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as e:
            self.log(f"[err] cannot start run_mangal: {e}")
            return 1

        # The output is read in a background thread: a blocking read here would freeze the window
        # while run_mangal is silent (long computations, the interactive star window of the shift stage).
        lines: queue.Queue = queue.Queue()

        def _reader():
            for ln in proc.stdout:
                lines.put(ln)
            lines.put(None)

        threading.Thread(target=_reader, daemon=True).start()
        app = QApplication.instance()
        finished = False
        while not finished:
            try:
                while True:
                    ln = lines.get_nowait()
                    if ln is None:
                        finished = True
                        break
                    ln = ln.rstrip("\n")
                    if ln:
                        self.log(ln)
            except queue.Empty:
                pass
            if app is not None:
                app.processEvents()
            time.sleep(0.05)
        return proc.wait()

    def _ensure_ergcal_json(self, object_rdir: Path, object_wdir: Path) -> Path | None:
        if not self.chk_apply_ergs.isChecked() or self.chk_standard.isChecked():
            return None

        filter_name = None
        try:
            params_obj, _ = auto_params_from_log(object_rdir, self._current_object_name(), ",".join([c.strip() for c in self.cube_combo.checked_data() if str(c).strip()]) or None, object_wdir)
            filter_name = params_obj.get("FILTERS")
        except Exception:
            pass

        existing = self._find_existing_ergcal_json(object_rdir.parent, filter_name=filter_name)
        if existing is not None and existing.exists():
            self.log(f"[auto] using existing QE calibration: {existing}")
            return existing

        std_sel = self._current_standard_selection()
        if std_sel is None:
            self.log("[err] apply ergs requested, but no standard star is selected")
            return None

        std_name = str(std_sel["object_name"])
        std_cube = str(std_sel["cube_id"])
        std_wdir = object_wdir / std_name

        try:
            std_params, _ = auto_params_from_log(object_rdir, std_name, std_cube, std_wdir)
        except Exception as e:
            self.log(f"[err] auto_params_from_log failed for standard {std_name}: {e}")
            return None

        std_wdir = Path(std_params["W_DIR"])
        std_wdir.mkdir(parents=True, exist_ok=True)
        std_param_path = std_wdir / f"{std_name}_auto.txt"
        write_param_file(std_params, std_param_path)

        self.log(
            f"[auto] QE calibration missing for filter={std_params.get('FILTERS', filter_name)}; "
            f"running QE-only standard stage for {std_name} cube {std_params.get('CUB_OBJ', std_cube)}"
        )
        cmd = [
            sys.executable, "-u", str(HERE / "run_mangal.py"),
            "--param", str(std_param_path),
            "--object", std_name,
            "--standard",
            "--qe-calib",
        ]
        self.log(f"[info] stage=auto-qe-calib mode=standard target={std_name} cube={std_params.get('CUB_OBJ', std_cube)}")
        self.log("[cmd] " + " ".join(cmd))
        ret = self._run_command_live(cmd)
        self.log(f"[done] return code = {ret}")
        if ret != 0:
            return None

        existing = self._find_existing_ergcal_json(object_rdir.parent, filter_name=std_params.get("FILTERS"))
        if existing is not None and existing.exists():
            self.log(f"[auto] created QE calibration: {existing}")
            return existing

        self.log("[err] QE calibration run finished, but QE_*.json was not found")
        return None


    def _show_photometry_products(self, w_dir: Path):
        phot_pngs = sorted(list(w_dir.rglob("*_phot.png")) + list(w_dir.rglob("*.phot.png")),
                           key=lambda p: p.stat().st_mtime, reverse=True)

        if phot_pngs:
            self.log(f"[info] photometry plot: {phot_pngs[0]}")
            self._open_file(phot_pngs[0])
        else:
            self.log("[warn] photometry plots were not found in W_DIR")

    def _show_subtraction_products(self, w_dir: Path, since: float):
        """Open the QA pictures of the continuum subtraction written by this run (*_net_subtract.png)."""
        pngs = sorted(
            (p for p in w_dir.rglob("*_net_subtract.png") if p.stat().st_mtime >= since - 1.0),
            key=lambda p: p.name,
        )
        if not pngs:
            self.log("[warn] subtraction pictures were not found in W_DIR")
            return
        for p in pngs:
            self.log(f"[info] continuum subtraction plot: {p}")
            self._open_file(p)

    def _show_qe_products(self, w_dir: Path, since: float):
        """Open the QE plot written by this run (QE_<standard>_<filter>.png, next to the phot cube)."""
        pngs = sorted(
            (p for p in w_dir.rglob("QE_*.png") if p.stat().st_mtime >= since - 1.0),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not pngs:
            self.log("[warn] QE plot was not found in W_DIR")
            return
        self.log(f"[info] QE plot: {pngs[0]}")
        self._open_file(pngs[0])

    def _run_single_pipeline(self, *, rdir: Path, standard_mode: bool, target_name: str, target_cube: str | None, chosen_wdir: Path) -> int:
        try:
            params, meta = auto_params_from_log(rdir, target_name, target_cube, chosen_wdir)
        except Exception as e:
            self.log(f"[err] auto_params_from_log failed: {e}")
            return 1

        w_dir = Path(params["W_DIR"])
        w_dir.mkdir(parents=True, exist_ok=True)
        param_path = w_dir / f"{target_name}_auto.txt"
        write_param_file(params, param_path)

        extra_args = self._build_extra_args()

        if (not standard_mode) and self.chk_apply_ergs.isChecked():
            combining = self.chk_combine.isChecked() and bool(self._combine_nights())
            averaged = sorted(chosen_wdir.glob("QE_combined_*.json"))  # write_dir = a combined folder
            if averaged:
                self.log(f"[info] combined data: averaged QE calibration {averaged[0].name}")
                extra_args += ["--ergcal-json", str(averaged[0])]
            else:
                erg_json = self._ensure_ergcal_json(rdir, chosen_wdir)  # QE of this night exists / is built
                if erg_json is None:
                    self.log("[err] erg calibration was requested, but QE_*.json could not be prepared")
                    return 1
                if combining:  # the averaged calibration is written by the combination and used for it
                    self.log("[info] combined nights: the QE calibration averaged over all nights will be used")
                else:
                    extra_args += ["--ergcal-json", str(erg_json)]

        cmd = [sys.executable, "-u", str(HERE / "run_mangal.py"), "--param", str(param_path), "--object", target_name] + extra_args

        mode_label = "standard" if standard_mode else "object"
        cube_label = params.get("CUB_OBJ", target_cube)
        self.log(f"[info] mode={mode_label} target={target_name} cube={cube_label}")
        self.log("[cmd] " + " ".join(cmd))
        t_start = time.time()
        ret = self._run_command_live(cmd)
        self.log(f"[done] return code = {ret}")

        if self.chk_phot_auto.isChecked():
            self._show_photometry_products(w_dir)
        combined_run = (not standard_mode) and self.chk_combine.isChecked() and bool(self._combine_nights())
        if ret == 0 and (not standard_mode) and (self.chk_net.isChecked() or combined_run):
            prod_dir = w_dir
            if combined_run:  # net is always made from the combined cube
                prod_dir = self._combine_out_dir(w_dir, target_name)
            self._show_subtraction_products(prod_dir, t_start)
        if ret == 0 and standard_mode and self.chk_qe.isChecked():
            self._show_qe_products(w_dir, t_start)
        return ret

    def run_pipeline(self):
        try:
            rdir, standard_mode, target_name, target_cube, chosen_wdir = self._resolve_target()
        except Exception as e:
            self.log(f"[err] {e}")
            return
        self.btn_run.setEnabled(False)  # the window stays responsive during the run: no second start
        self.btn_run.setText("running...")
        try:
            self._run_single_pipeline(
                rdir=rdir,
                standard_mode=standard_mode,
                target_name=target_name,
                target_cube=target_cube,
                chosen_wdir=chosen_wdir,
            )
        finally:
            self.btn_run.setEnabled(True)
            self.btn_run.setText("run")


def main():
    app = QApplication(sys.argv)
    # one font size everywhere (on macOS Qt otherwise uses smaller fonts for some widgets: tool buttons, lists,
    # spin boxes ...)
    base = QFont(app.font())
    size = base.pointSizeF() if base.pointSizeF() > 0 else 13.0
    app.setFont(base)
    app.setStyleSheet(f"* {{ font-size: {size:g}pt; }}")
    w = MangalQt()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
