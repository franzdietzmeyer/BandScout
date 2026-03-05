"""
app.py
======
GelAnalyzer Pro — MVP GUI

Architecture
------------
GelAnalyzerApp (QMainWindow)
│
├── Left sidebar  (QWidget, fixed 300 px)
│   ├── "Load Gel Image"    QPushButton
│   ├── "Image Type"        QComboBox      (disabled until image loaded)
│   ├── ── Lane Detection ──
│   ├── "Number of Wells"   QSpinBox       (default 12)
│   ├── "Detect Lanes"      QPushButton    (disabled until image loaded)
│   ├── ── Band Detection ──
│   ├── "Prominence"        QDoubleSpinBox (default 4.0)
│   ├── "Background Window" QSpinBox       (default 250)
│   ├── "Detect Bands"      QPushButton    (disabled until image loaded)
│   ├── ── (stretch) ──
│   ├── "Export to CSV"     QPushButton    (disabled until bands detected)
│   └── Status label
│
└── Right area (QSplitter, vertical)
    ├── GelCanvas (FigureCanvasQTAgg)
    │   ├── ax_img     — gel image + lane boxes + band markers
    │   └── ax_profile — 1D intensity profile for the selected lane
    └── QTableWidget — band quantification results (read-only)
"""

import csv
import sys
import os
from pathlib import Path

import cv2
import numpy as np
from typing import Optional
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QFont, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QToolBox,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core_engine import Analysis, Band, Lane, VolumeCalcMode


# ===========================================================================
# Ladder configuration dialog
# ===========================================================================


class LadderConfigDialog(QDialog):
    """
    Modal dialog that lets the user assign known MW values (in kDa) to the
    detected bands of the selected ladder lane.

    Layout
    ------
    A three-column table is shown:

    * **Band #**           — read-only, 1-based band index.
    * **Peak Position (px)** — read-only, the band's current ``peak_index``.
    * **Known MW (kDa)**   — editable float; leave blank to exclude a band
      from the calibration.

    ``get_calibration_data()`` harvests the table and returns a dict of
    ``{band_index (0-based): mw_float}``.

    Any MW values already stored on the bands are pre-filled so the dialog
    is non-destructive when opened a second time.
    """

    _DARK_STYLE = """
        QDialog {
            background-color: #1e1e2e;
            color: #cdd6f4;
        }
        QLabel {
            color: #cdd6f4;
            font-size: 11px;
        }
        QTableWidget {
            background-color: #181825;
            color: #cdd6f4;
            gridline-color: #313244;
            border: 1px solid #45475a;
            font-size: 11px;
        }
        QTableWidget::item {
            padding: 4px 8px;
        }
        QTableWidget::item:selected {
            background-color: #45475a;
        }
        QHeaderView::section {
            background-color: #12121f;
            color: #a6adc8;
            border: none;
            border-bottom: 1px solid #313244;
            padding: 5px 8px;
            font-weight: 600;
            font-size: 11px;
        }
        QComboBox {
            background-color: #2a2a3e;
            color: #cdd6f4;
            border: 1px solid #45475a;
            border-radius: 5px;
            padding: 3px 8px;
            font-size: 11px;
        }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView {
            background-color: #2a2a3e;
            color: #cdd6f4;
            selection-background-color: #45475a;
        }
        QDialogButtonBox QPushButton {
            background-color: #89b4fa;
            color: #1e1e2e;
            border: none;
            border-radius: 5px;
            padding: 6px 20px;
            font-weight: 600;
            font-size: 11px;
            min-width: 80px;
        }
        QDialogButtonBox QPushButton:hover  { background-color: #b4d0fc; }
        QDialogButtonBox QPushButton:pressed { background-color: #6494e0; }
        QDialogButtonBox QPushButton[text="Cancel"] {
            background-color: #313244;
            color: #cdd6f4;
        }
        QDialogButtonBox QPushButton[text="Cancel"]:hover  { background-color: #45475a; }
    """

    def __init__(self, lane, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure MW Ladder Lane")
        self.setMinimumWidth(460)
        self.setStyleSheet(self._DARK_STYLE)

        self._lane = lane

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # ---- Instruction label -----------------------------------------
        info = QLabel(
            "Enter the known molecular weight (kDa) for each calibrator band.\n"
            "Leave a row blank to exclude that band from the calibration."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #a6adc8; font-size: 10px;")
        layout.addWidget(info)

        # ---- Preset selector -------------------------------------------
        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)
        preset_label = QLabel("Preset Ladder:")
        preset_label.setFont(QFont("Helvetica Neue", 10, QFont.Weight.Medium))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["Manual", "SeeBlue (Invitrogen)"])
        self.combo_preset.setFixedHeight(28)
        self.combo_preset.currentIndexChanged.connect(self.apply_preset)
        preset_row.addWidget(preset_label)
        preset_row.addWidget(self.combo_preset, stretch=1)
        layout.addLayout(preset_row)

        # ---- Band table ------------------------------------------------
        self.table = QTableWidget(len(lane.bands), 3)
        self.table.setHorizontalHeaderLabels(
            ["Band #", "Peak Position (px)", "Known MW (kDa)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.verticalHeader().setVisible(False)

        for i, band in enumerate(lane.bands):
            # Band # — read-only
            num_item = QTableWidgetItem(str(i + 1))
            num_item.setFlags(num_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            num_item.setTextAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(i, 0, num_item)

            # Peak position — read-only
            pos_item = QTableWidgetItem(str(band.peak_index))
            pos_item.setFlags(pos_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            pos_item.setTextAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(i, 1, pos_item)

            # Known MW — editable; pre-fill if already assigned
            mw_text = (
                f"{band.molecular_weight:.2f}"
                if band.molecular_weight is not None
                else ""
            )
            mw_item = QTableWidgetItem(mw_text)
            mw_item.setTextAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(i, 2, mw_item)

        layout.addWidget(self.table)

        # ---- OK / Cancel -----------------------------------------------
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    # ------------------------------------------------------------------

    _PRESETS: dict = {
        "SeeBlue (Invitrogen)": [198, 98, 62, 49, 38, 28, 17, 14, 6, 3],
    }

    def apply_preset(self) -> None:
        """
        Populate or clear the "Known MW (kDa)" column based on the chosen
        preset.

        * **"Manual"** — clears all entries in the MW column so the user can
          type values freely.
        * Any named preset — fills rows top-to-bottom with the preset MW
          values.  If the lane has fewer bands than the preset list, only the
          available rows are filled; excess preset values are ignored.
        """
        preset_name = self.combo_preset.currentText()
        mw_values   = self._PRESETS.get(preset_name)   # None → "Manual"

        for row in range(self.table.rowCount()):
            item = self.table.item(row, 2)
            if item is None:
                item = QTableWidgetItem()
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(row, 2, item)

            if mw_values is None or row >= len(mw_values):
                item.setText("")
            else:
                item.setText(str(mw_values[row]))

    # ------------------------------------------------------------------

    def get_calibration_data(self) -> dict:
        """
        Return a mapping of ``{band_index (0-based): mw_float}`` for every
        row that contains a valid float in the "Known MW" column.
        Blank or non-numeric rows are silently skipped.
        """
        result: dict = {}
        for row in range(self.table.rowCount()):
            mw_item = self.table.item(row, 2)
            if mw_item is None:
                continue
            text = mw_item.text().strip()
            if not text:
                continue
            try:
                result[row] = float(text)
            except ValueError:
                pass
        return result


# ===========================================================================
# Matplotlib canvas widget
# ===========================================================================


class GelCanvas(FigureCanvasQTAgg):
    """
    Matplotlib canvas with two side-by-side axes:

    ``ax_img``
        Left panel — gel image with lane-rectangle and band-marker overlays.
        Axis decorations (ticks, labels) are hidden; it behaves as a pure
        image viewer.  Mouse clicks on this axes trigger lane selection.

    ``ax_profile``
        Right panel — 1D intensity profile for the currently selected lane.
        Shows smoothed signal, background estimate, corrected signal, and
        per-band filled areas.  Uses a dark-themed styled axes.

    The two axes are housed in a single ``Figure`` with a 3:2 width ratio
    so the image always dominates.  The canvas expands to fill all
    available Qt space.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        fig = Figure(facecolor="#060911")
        super().__init__(fig)
        self.setParent(parent)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.updateGeometry()

        # GridSpec: image column gets 3 units, profile column gets 2.
        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[3, 2],
            left=0.01, right=0.99,
            top=0.96,  bottom=0.07,
            wspace=0.18,
        )
        self.ax_img     = fig.add_subplot(gs[0])
        self.ax_profile = fig.add_subplot(gs[1])

        self._show_placeholder()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _show_placeholder(self) -> None:
        """Render initial placeholder text on both axes."""
        self.ax_img.clear()
        self.ax_img.set_axis_off()
        self.ax_img.set_facecolor("#060911")
        self.ax_img.text(
            0.5, 0.5, "No Image Loaded",
            transform=self.ax_img.transAxes,
            ha="center", va="center",
            fontsize=14, color="#585b70", fontstyle="italic",
        )

        self.ax_profile.clear()
        _style_axes(self.ax_profile)
        self.ax_profile.text(
            0.5, 0.5, "No profile\navailable",
            transform=self.ax_profile.transAxes,
            ha="center", va="center",
            fontsize=9, color="#585b70", fontstyle="italic",
        )
        self.draw()

    def clear(self) -> None:
        """Reset both axes to the placeholder state."""
        self._show_placeholder()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _style_axes(ax) -> None:
    """
    Apply consistent dark-theme styling to a profile/data axes.

    Kept as a module-level function (not a method) so both ``GelCanvas``
    and ``GelAnalyzerApp`` can call it without coupling.
    """
    ax.set_facecolor("#060911")
    for spine in ax.spines.values():
        spine.set_edgecolor("#475569")
    ax.tick_params(colors="#475569", labelsize=6)
    ax.xaxis.label.set_color("#475569")
    ax.yaxis.label.set_color("#475569")


# ===========================================================================
# Main window
# ===========================================================================


class GelAnalyzerApp(QMainWindow):
    """
    Top-level application window.

    State
    -----
    ``self.current_analysis``
        The active :class:`core_engine.Analysis` instance, or ``None`` when
        no image has been loaded.  All downstream operations (lane drawing,
        band picking, calibration) should read/write through this object.
    """

    # Default window geometry
    _DEFAULT_WIDTH  = 1200
    _DEFAULT_HEIGHT = 800
    _SIDEBAR_WIDTH  = 300

    def __init__(self) -> None:
        super().__init__()
        self.current_analysis:    Optional[Analysis] = None
        # Index into current_analysis.lanes; None means no lane is selected.
        self.selected_lane_index: Optional[int]      = None
        # Filesystem path of the currently loaded image (basename shown in header).
        self.image_path: str = ""

        # ---- Band drag state -------------------------------------------
        # Shared by two drag interactions:
        #   • Profile axis (ax_profile) — horizontal drag of start/end lines
        #   • Image axis   (ax_img)     — vertical drag of peak marker lines
        # All variables are reset together by _reset_drag_state().
        self._drag_active: bool          = False
        self._drag_band                  = None   # Band being edited
        self._drag_edge:   Optional[str] = None   # 'start', 'end', or 'peak'
        self._drag_line                  = None   # Matplotlib Line2D handle
        self._drag_axis:   Optional[str] = None   # 'profile' or 'image'

        # Maps id(band) → the vertical peak-indicator Line2D on ax_profile.
        # Populated by redraw_canvas(); used by on_mouse_motion for live sync.
        self._peak_profile_lines: dict = {}

        # 0-based index of the band currently highlighted via the results table.
        # None means no highlight is active.
        self.highlighted_band_index: Optional[int] = None

        # Interaction mode — governs how canvas clicks are interpreted.
        # "NORMAL"   → standard lane-selection behaviour
        # "ADD_LANE" → next click on ax_img inserts a new lane
        # "ADD_BAND" → next click on ax_img or ax_profile inserts a new band
        self.current_mode: str = "NORMAL"

        # Smile (parabolic) correction state.
        # True when the user has enabled the rubber-band smile overlay.
        self.smile_correction_enabled: bool = False

        # Matplotlib Line2D handles for the three draggable smile anchors and
        # the fitted parabola.  Populated by redraw_canvas(); None otherwise.
        self._smile_anchor_artists: list = []
        self._smile_parabola_line        = None

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Assemble the window layout."""
        self.setWindowTitle("GelAnalyzer Pro MVP")
        self.resize(self._DEFAULT_WIDTH, self._DEFAULT_HEIGHT)

        # ---- Menu bar --------------------------------------------------
        self._build_menu_bar()

        # ---- Status bar ------------------------------------------------
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet(
            "QStatusBar { background-color: #12121f; color: #a6adc8; "
            "border-top: 1px solid #313244; font-size: 10px; padding: 2px 8px; }"
        )

        # ---- Central widget + root horizontal layout -------------------
        root_widget = QWidget()
        root_layout = QHBoxLayout(root_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root_widget)

        # ---- Left sidebar ----------------------------------------------
        sidebar = self._build_sidebar()
        root_layout.addWidget(sidebar)

        # ---- Right area: header + vertical splitter --------------------
        right_widget = QWidget()
        right_vbox   = QVBoxLayout(right_widget)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(0)

        # Top info header bar
        header_bar = QWidget()
        header_bar.setFixedHeight(32)
        header_bar.setStyleSheet(
            "background-color: #0b0f1a; border-bottom: 1px solid #1e2a3a;"
        )
        header_hl = QHBoxLayout(header_bar)
        header_hl.setContentsMargins(12, 0, 12, 0)

        self.header_info_label = QLabel("🟢 No Image Loaded | 0 lanes • 0 bands")
        self.header_info_label.setFont(QFont("Helvetica Neue", 9))
        self.header_info_label.setStyleSheet("color: #94a3b8; background: transparent;")
        self.header_info_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_hl.addStretch()
        header_hl.addWidget(self.header_info_label)

        right_vbox.addWidget(header_bar)

        # Matplotlib canvas (image + profile) on top, results table below.
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setHandleWidth(5)
        right_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #1e2a3a; }"
        )

        self.canvas = GelCanvas()
        right_splitter.addWidget(self.canvas)

        self.results_table = QTableWidget()
        self.results_table.setObjectName("dataTablePanel")
        self._configure_results_table()
        right_splitter.addWidget(self.results_table)

        # Canvas starts at ~65 % of vertical space; table at ~35 %.
        right_splitter.setStretchFactor(0, 65)
        right_splitter.setStretchFactor(1, 35)

        right_vbox.addWidget(right_splitter, stretch=1)
        root_layout.addWidget(right_widget, stretch=1)

    def _build_menu_bar(self) -> None:
        """
        Populate the application menu bar.

        Menus
        -----
        File     — Load Image, Export to CSV
        Edit     — Add Lane / Band manually, Delete Selected Band, Cancel Action
        Analysis — Auto-Detect Lanes / Bands
        """
        mb: QMenuBar = self.menuBar()
        mb.setStyleSheet(
            """
            QMenuBar {
                background-color: #12121f;
                color: #cdd6f4;
                border-bottom: 1px solid #313244;
                font-size: 11px;
                padding: 2px 0;
            }
            QMenuBar::item {
                background: transparent;
                padding: 4px 10px;
            }
            QMenuBar::item:selected { background-color: #2a2a3e; }
            QMenu {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                font-size: 11px;
            }
            QMenu::item { padding: 5px 20px; }
            QMenu::item:selected { background-color: #313244; }
            QMenu::separator {
                height: 1px;
                background: #313244;
                margin: 3px 8px;
            }
            """
        )

        # ---- File menu -------------------------------------------------
        file_menu = mb.addMenu("&File")

        act_load = QAction("Load Image", self)
        act_load.triggered.connect(self._on_load_image)
        file_menu.addAction(act_load)

        file_menu.addSeparator()

        act_export = QAction("Export to CSV", self)
        act_export.triggered.connect(self.export_to_csv)
        file_menu.addAction(act_export)

        # ---- Edit menu -------------------------------------------------
        edit_menu = mb.addMenu("&Edit")

        act_add_lane = QAction("Add Lane Manually", self)
        act_add_lane.triggered.connect(self._on_add_lane_mode)
        edit_menu.addAction(act_add_lane)

        act_add_band = QAction("Add Band Manually", self)
        act_add_band.triggered.connect(self._on_add_band_mode)
        edit_menu.addAction(act_add_band)

        act_smile = QAction("Toggle Smile Correction", self)
        act_smile.triggered.connect(self._on_toggle_smile_correction)
        edit_menu.addAction(act_smile)

        edit_menu.addSeparator()

        act_delete_band = QAction("Delete Selected Band", self)
        act_delete_band.setShortcuts([
            QKeySequence(Qt.Key.Key_Delete),
            QKeySequence(Qt.Key.Key_Backspace),
        ])
        act_delete_band.triggered.connect(self._on_delete_selected_band)
        edit_menu.addAction(act_delete_band)

        edit_menu.addSeparator()

        act_cancel = QAction("Cancel Action", self)
        act_cancel.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        act_cancel.triggered.connect(self._on_cancel_action)
        edit_menu.addAction(act_cancel)

        # ---- Analysis menu ---------------------------------------------
        analysis_menu = mb.addMenu("&Analysis")

        act_detect_lanes = QAction("Auto-Detect Lanes", self)
        act_detect_lanes.triggered.connect(self._on_detect_lanes_clicked)
        analysis_menu.addAction(act_detect_lanes)

        act_detect_bands = QAction("Auto-Detect Bands", self)
        act_detect_bands.triggered.connect(self._on_detect_bands_clicked)
        analysis_menu.addAction(act_detect_bands)

        analysis_menu.addSeparator()

        act_recalc = QAction("Recalculate Data", self)
        act_recalc.setShortcut(QKeySequence.StandardKey.Refresh)   # Ctrl+R / Cmd+R
        act_recalc.triggered.connect(self.recalculate_all_data)
        analysis_menu.addAction(act_recalc)

    def _build_sidebar(self) -> QWidget:
        """
        Construct and return the left-hand control panel.

        The controls are organised into a ``QToolBox`` accordion with four
        pages that auto-advance as the user completes each analysis step.
        The "Export to CSV" button and the sidebar status label are pinned
        permanently below the toolbox so they are always visible.
        """
        # ---- Shared slider stylesheet ----------------------------------
        _slider_style = """
            QSlider::groove:horizontal {
                background: #2d3748;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #89b4fa;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #89b4fa;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #b4d0fc;
            }
            QSlider::handle:horizontal:disabled {
                background: #45475a;
            }
            QSlider::sub-page:horizontal:disabled {
                background: #45475a;
            }
        """

        # ---- Outer sidebar container -----------------------------------
        sidebar = QWidget()
        sidebar.setFixedWidth(self._SIDEBAR_WIDTH)
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet(
            "#sidebar { background-color: #12121f; border-right: 1px solid #333355; }"
        )

        outer = QVBoxLayout(sidebar)
        outer.setContentsMargins(12, 16, 12, 12)
        outer.setSpacing(8)

        # ---- App title -------------------------------------------------
        title_label = QLabel("GelAnalyzer Pro")
        title_label.setFont(QFont("Helvetica Neue", 15, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #cdd6f4;")
        outer.addWidget(title_label)

        subtitle_label = QLabel("1D Electrophoresis Analysis")
        subtitle_label.setFont(QFont("Helvetica Neue", 9))
        subtitle_label.setStyleSheet("color: #6c7086;")
        outer.addWidget(subtitle_label)

        outer.addSpacing(6)
        _div = QWidget(); _div.setFixedHeight(1)
        _div.setStyleSheet("background-color: #313244;")
        outer.addWidget(_div)
        outer.addSpacing(4)

        # ================================================================
        # QToolBox
        # ================================================================
        self.toolbox = QToolBox()
        self.toolbox.setObjectName("accordionBody")
        self.toolbox.setStyleSheet(
            """
            QToolBox {
                background-color: #12121f;
            }
            QToolBox::tab {
                background-color: #1e1e2e;
                color: #a6adc8;
                border: 1px solid #313244;
                border-radius: 5px;
                padding: 7px 10px;
                font-size: 10px;
                font-weight: 600;
            }
            QToolBox::tab:selected {
                background-color: #2a2a3e;
                color: #89b4fa;
                border-color: #45475a;
            }
            QToolBox::tab:hover:!selected {
                background-color: #252535;
                color: #cdd6f4;
            }
            """
        )

        # ----------------------------------------------------------------
        # Page 0 — Image Setup
        # ----------------------------------------------------------------
        page_setup = QWidget()
        page_setup.setStyleSheet("background-color: #12121f;")
        setup_layout = QVBoxLayout(page_setup)
        setup_layout.setContentsMargins(4, 8, 4, 8)
        setup_layout.setSpacing(8)

        self.btn_load = QPushButton("Load Gel Image")
        self.btn_load.setFixedHeight(44)
        self.btn_load.setFont(QFont("Helvetica Neue", 11, QFont.Weight.Medium))
        self.btn_load.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load.setStyleSheet(
            """
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover   { background-color: #b4d0fc; }
            QPushButton:pressed { background-color: #6494e0; }
            """
        )
        setup_layout.addWidget(self.btn_load)

        self.grp_image_type = QGroupBox("My Image type is:")
        self.grp_image_type.setFont(QFont("Helvetica Neue", 10))
        self.grp_image_type.setEnabled(False)
        self.grp_image_type.setStyleSheet(
            """
            QGroupBox {
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 10px;
                padding: 8px 8px 6px 8px;
                font-weight: 600;
                font-size: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #cdd6f4;
            }
            QGroupBox:disabled { border-color: #313244; }
            QRadioButton {
                color: #cdd6f4;
                font-size: 10px;
                spacing: 8px;
                padding: 2px 0;
            }
            QRadioButton:disabled { color: #45475a; }
            QRadioButton::indicator {
                width: 13px; height: 13px;
                border: 1px solid #585b70;
                border-radius: 7px;
                background-color: #2a2a3e;
            }
            QRadioButton::indicator:checked {
                background-color: #89b4fa;
                border-color: #89b4fa;
            }
            QRadioButton::indicator:hover { border-color: #89b4fa; }
            """
        )
        grp_layout = QVBoxLayout(self.grp_image_type)
        grp_layout.setSpacing(6)
        self.radio_light_on_dark = QRadioButton("Light bands on dark background")
        self.radio_dark_on_light = QRadioButton("Dark bands on light background")
        self.radio_dark_on_light.setChecked(True)
        grp_layout.addWidget(self.radio_light_on_dark)
        grp_layout.addWidget(self.radio_dark_on_light)
        setup_layout.addWidget(self.grp_image_type)
        setup_layout.addStretch()

        self.toolbox.addItem(page_setup, "1. Image Setup")

        # ----------------------------------------------------------------
        # Page 1 — Detect Lanes
        # ----------------------------------------------------------------
        page_lanes = QWidget()
        page_lanes.setStyleSheet("background-color: #12121f;")
        lanes_layout = QVBoxLayout(page_lanes)
        lanes_layout.setContentsMargins(4, 8, 4, 8)
        lanes_layout.setSpacing(8)

        wells_row = QHBoxLayout()
        wells_row.setContentsMargins(0, 0, 0, 0)
        wells_name = QLabel("Number of Wells")
        wells_name.setFont(QFont("Helvetica Neue", 10, QFont.Weight.Medium))
        wells_name.setStyleSheet("color: #a6adc8;")
        self.lbl_num_wells = QLabel("12")
        self.lbl_num_wells.setFont(QFont("Helvetica Neue", 10, QFont.Weight.Bold))
        self.lbl_num_wells.setStyleSheet("color: #89b4fa;")
        self.lbl_num_wells.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        wells_row.addWidget(wells_name)
        wells_row.addWidget(self.lbl_num_wells)
        lanes_layout.addLayout(wells_row)

        self.slider_num_wells = QSlider(Qt.Orientation.Horizontal)
        self.slider_num_wells.setRange(1, 30)
        self.slider_num_wells.setValue(12)
        self.slider_num_wells.setFixedHeight(24)
        self.slider_num_wells.setStyleSheet(_slider_style)
        self.slider_num_wells.valueChanged.connect(
            lambda v: self.lbl_num_wells.setText(str(v))
        )
        lanes_layout.addWidget(self.slider_num_wells)

        self.btn_detect = QPushButton("Detect Lanes")
        self.btn_detect.setFixedHeight(40)
        self.btn_detect.setFont(QFont("Helvetica Neue", 11, QFont.Weight.Medium))
        self.btn_detect.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_detect.setEnabled(False)
        self.btn_detect.setStyleSheet(
            """
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover   { background-color: #c3f0bf; }
            QPushButton:pressed { background-color: #7ec97a; }
            QPushButton:disabled { background-color: #313244; color: #45475a; }
            """
        )
        lanes_layout.addWidget(self.btn_detect)

        self.chk_add_lane = QCheckBox("Manual Add Lane")
        self.chk_add_lane.setFont(QFont("Helvetica Neue", 10))
        self.chk_add_lane.setStyleSheet(
            "QCheckBox { color: #94a3b8; spacing: 8px; }"
            "QCheckBox::indicator { width: 16px; height: 16px; border-radius: 8px;"
            "  border: 2px solid #475569; background: #0b0f1a; }"
            "QCheckBox::indicator:checked { background: #f97316; border-color: #f97316; }"
            "QCheckBox::indicator:hover { border-color: #f97316; }"
        )
        lanes_layout.addWidget(self.chk_add_lane)
        lanes_layout.addStretch()

        self.toolbox.addItem(page_lanes, "2. Detect Lanes")

        # ----------------------------------------------------------------
        # Page 2 — Detect Bands
        # ----------------------------------------------------------------
        page_bands = QWidget()
        page_bands.setStyleSheet("background-color: #12121f;")
        bands_layout = QVBoxLayout(page_bands)
        bands_layout.setContentsMargins(4, 8, 4, 8)
        bands_layout.setSpacing(8)

        prom_row = QHBoxLayout()
        prom_row.setContentsMargins(0, 0, 0, 0)
        prom_name = QLabel("Prominence")
        prom_name.setFont(QFont("Helvetica Neue", 10, QFont.Weight.Medium))
        prom_name.setStyleSheet("color: #a6adc8;")
        self.lbl_prominence = QLabel("4.0")
        self.lbl_prominence.setFont(QFont("Helvetica Neue", 10, QFont.Weight.Bold))
        self.lbl_prominence.setStyleSheet("color: #89b4fa;")
        self.lbl_prominence.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        prom_row.addWidget(prom_name)
        prom_row.addWidget(self.lbl_prominence)
        bands_layout.addLayout(prom_row)

        # Internal integer range 1–200 maps to 0.1–20.0 (÷10).
        self.slider_prominence = QSlider(Qt.Orientation.Horizontal)
        self.slider_prominence.setRange(1, 200)
        self.slider_prominence.setValue(40)   # = 4.0
        self.slider_prominence.setFixedHeight(24)
        self.slider_prominence.setStyleSheet(_slider_style)
        self.slider_prominence.valueChanged.connect(
            lambda v: self.lbl_prominence.setText(f"{v / 10:.1f}")
        )
        bands_layout.addWidget(self.slider_prominence)

        bg_row = QHBoxLayout()
        bg_row.setContentsMargins(0, 0, 0, 0)
        bg_name = QLabel("Background Window")
        bg_name.setFont(QFont("Helvetica Neue", 10, QFont.Weight.Medium))
        bg_name.setStyleSheet("color: #a6adc8;")
        self.lbl_bg_window = QLabel("250")
        self.lbl_bg_window.setFont(QFont("Helvetica Neue", 10, QFont.Weight.Bold))
        self.lbl_bg_window.setStyleSheet("color: #89b4fa;")
        self.lbl_bg_window.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        bg_row.addWidget(bg_name)
        bg_row.addWidget(self.lbl_bg_window)
        bands_layout.addLayout(bg_row)

        self.slider_bg_window = QSlider(Qt.Orientation.Horizontal)
        self.slider_bg_window.setRange(10, 500)
        self.slider_bg_window.setValue(250)
        self.slider_bg_window.setFixedHeight(24)
        self.slider_bg_window.setStyleSheet(_slider_style)
        self.slider_bg_window.valueChanged.connect(
            lambda v: self.lbl_bg_window.setText(str(v))
        )
        bands_layout.addWidget(self.slider_bg_window)

        self.btn_detect_bands = QPushButton("Detect Bands")
        self.btn_detect_bands.setObjectName("btnPrimary")
        self.btn_detect_bands.setFixedHeight(40)
        self.btn_detect_bands.setFont(QFont("Helvetica Neue", 11, QFont.Weight.Medium))
        self.btn_detect_bands.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_detect_bands.setEnabled(False)
        self.btn_detect_bands.setStyleSheet(
            """
            QPushButton {
                background-color: #cba6f7;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover   { background-color: #dbbcff; }
            QPushButton:pressed { background-color: #a87fe0; }
            QPushButton:disabled { background-color: #313244; color: #45475a; }
            """
        )
        bands_layout.addWidget(self.btn_detect_bands)

        self.chk_add_band = QCheckBox("Manual Add Band")
        self.chk_add_band.setFont(QFont("Helvetica Neue", 10))
        self.chk_add_band.setStyleSheet(
            "QCheckBox { color: #94a3b8; spacing: 8px; }"
            "QCheckBox::indicator { width: 16px; height: 16px; border-radius: 8px;"
            "  border: 2px solid #475569; background: #0b0f1a; }"
            "QCheckBox::indicator:checked { background: #f97316; border-color: #f97316; }"
            "QCheckBox::indicator:hover { border-color: #f97316; }"
        )
        bands_layout.addWidget(self.chk_add_band)
        bands_layout.addStretch()

        self.toolbox.addItem(page_bands, "3. Detect Bands")

        # ----------------------------------------------------------------
        # Page 3 — Calibration
        # ----------------------------------------------------------------
        page_cal = QWidget()
        page_cal.setStyleSheet("background-color: #12121f;")
        cal_layout = QVBoxLayout(page_cal)
        cal_layout.setContentsMargins(4, 8, 4, 8)
        cal_layout.setSpacing(8)

        self.grp_mw_cal = QGroupBox("Molecular Weight Calibration")
        self.grp_mw_cal.setFont(QFont("Helvetica Neue", 10))
        self.grp_mw_cal.setEnabled(False)
        self.grp_mw_cal.setStyleSheet(
            """
            QGroupBox {
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 10px;
                padding: 8px 8px 8px 8px;
                font-weight: 600;
                font-size: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #cdd6f4;
            }
            QGroupBox:disabled { border-color: #313244; }
            QLabel { color: #a6adc8; font-size: 10px; }
            """
        )
        grp_mw_layout = QVBoxLayout(self.grp_mw_cal)
        grp_mw_layout.setSpacing(6)

        model_label = QLabel("Model: ln(MW) = m · distance + c")
        model_label.setFont(QFont("Helvetica Neue", 9))
        model_label.setStyleSheet("color: #6c7086; font-style: italic;")
        model_label.setWordWrap(True)
        grp_mw_layout.addWidget(model_label)
        grp_mw_layout.addSpacing(4)

        self.btn_set_ladder = QPushButton("Set Selected Lane as Ladder")
        self.btn_set_ladder.setFixedHeight(40)
        self.btn_set_ladder.setFont(QFont("Helvetica Neue", 10, QFont.Weight.Medium))
        self.btn_set_ladder.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_set_ladder.setStyleSheet(
            """
            QPushButton {
                background-color: #fab387;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover   { background-color: #fcc9a7; }
            QPushButton:pressed { background-color: #d4905e; }
            QPushButton:disabled { background-color: #313244; color: #45475a; }
            """
        )
        grp_mw_layout.addWidget(self.btn_set_ladder)
        cal_layout.addWidget(self.grp_mw_cal)

        self.chk_smile = QCheckBox("Smile Correction")
        self.chk_smile.setFont(QFont("Helvetica Neue", 10))
        self.chk_smile.setStyleSheet(
            "QCheckBox { color: #94a3b8; spacing: 8px; }"
            "QCheckBox::indicator { width: 16px; height: 16px; border-radius: 8px;"
            "  border: 2px solid #475569; background: #0b0f1a; }"
            "QCheckBox::indicator:checked { background: #eab308; border-color: #eab308; }"
            "QCheckBox::indicator:hover { border-color: #eab308; }"
        )
        cal_layout.addWidget(self.chk_smile)
        cal_layout.addStretch()

        self.toolbox.addItem(page_cal, "4. Calibration")

        outer.addWidget(self.toolbox, stretch=1)

        # ================================================================
        # Global controls — always visible below the toolbox
        # ================================================================
        _div2 = QWidget(); _div2.setFixedHeight(1)
        _div2.setStyleSheet("background-color: #313244;")
        outer.addWidget(_div2)
        outer.addSpacing(6)

        self.btn_export = QPushButton("Export to CSV")
        self.btn_export.setObjectName("btnSecondary")
        self.btn_export.setFixedHeight(44)
        self.btn_export.setFont(QFont("Helvetica Neue", 11, QFont.Weight.Bold))
        self.btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet(
            """
            QPushButton {
                background-color: #94e2d5;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                font-weight: 700;
            }
            QPushButton:hover   { background-color: #b3f0e8; }
            QPushButton:pressed { background-color: #6fcfc2; }
            QPushButton:disabled { background-color: #313244; color: #45475a; }
            """
        )
        outer.addWidget(self.btn_export)
        outer.addSpacing(6)

        self.status_label = QLabel("Ready.")
        self.status_label.setFont(QFont("Helvetica Neue", 9))
        self.status_label.setStyleSheet("color: #585b70;")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        return sidebar

    # ------------------------------------------------------------------
    # Signal / slot connections
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Wire all Qt signals to their handler methods."""
        self.btn_load.clicked.connect(self._on_load_image)

        # Radio buttons — both emit toggled(); the slot ignores the
        # un-check signal so logic runs exactly once per user action.
        self.radio_light_on_dark.toggled.connect(self._on_image_type_toggled)
        self.radio_dark_on_light.toggled.connect(self._on_image_type_toggled)

        self.btn_detect.clicked.connect(self._on_detect_lanes_clicked)
        self.btn_detect_bands.clicked.connect(self._on_detect_bands_clicked)
        self.btn_set_ladder.clicked.connect(self.on_calibrate_ladder_clicked)
        self.btn_export.clicked.connect(self.export_to_csv)
        self.results_table.itemSelectionChanged.connect(self.on_table_row_selected)

        # ---- Sidebar mode toggles -------------------------------------
        self.chk_add_lane.toggled.connect(self._on_chk_add_lane_toggled)
        self.chk_add_band.toggled.connect(self._on_chk_add_band_toggled)
        self.chk_smile.toggled.connect(self._on_chk_smile_toggled)

        # ---- Matplotlib event hooks -----------------------------------
        # Lane selection (image axis click)
        self.canvas.mpl_connect("button_press_event",   self._on_canvas_click)
        # Band-boundary drag (profile axis)
        self.canvas.mpl_connect("pick_event",           self.on_line_pick)
        self.canvas.mpl_connect("motion_notify_event",  self.on_mouse_motion)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_load_image(self) -> None:
        """
        Open a file dialog and load the selected image.

        Steps
        -----
        1.  ``QFileDialog`` — filters to common gel image formats.
        2.  ``cv2.imread`` reads the file; converted BGR → RGB immediately.
        3.  A new :class:`~core_engine.Analysis` is instantiated and stored
            as ``self.current_analysis``.  ``is_dark_on_light`` is set from
            the current combo selection so the two controls stay in sync.
        4.  The processed image is fetched and rendered on the canvas.
        5.  The Image Type combo is enabled.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Gel Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)",
        )

        if not file_path:
            return   # user cancelled — do nothing

        # ---- Load via OpenCV (handles multi-format, incl. 16-bit TIFF) ----
        bgr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if bgr is None:
            self._set_status(f"Failed to load:\n{file_path}", error=True)
            return

        # OpenCV reads colour images as BGR; convert to RGB for numpy/Analysis.
        if bgr.ndim == 3 and bgr.shape[2] >= 3:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = bgr   # already grayscale (2-D) or single-channel

        # ---- Build Analysis object -------------------------------------
        is_dark = self._is_dark_on_light()
        self.current_analysis   = Analysis(
            rgb,
            is_dark_on_light=is_dark,
            volume_calc_mode=VolumeCalcMode.ZERO_CLIPPED,
        )
        self.selected_lane_index = None     # reset selection for the new image

        # ---- Render the processed image --------------------------------
        self.redraw_canvas()

        # ---- Update UI state -------------------------------------------
        self.grp_image_type.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self.btn_detect_bands.setEnabled(True)
        self.btn_export.setEnabled(False)   # no bands yet for the new image
        self.grp_mw_cal.setEnabled(False)   # re-enabled after bands are detected

        self.image_path = file_path
        fname = os.path.basename(file_path)
        self._set_status(f"Loaded: {fname}\n{rgb.shape[1]}×{rgb.shape[0]} px")
        self.update_header_info()
        self.toolbox.setCurrentIndex(1)   # advance to "Detect Lanes" page

    def _on_image_type_toggled(self, checked: bool) -> None:
        """
        Respond to either radio button being toggled.

        Both ``radio_light_on_dark`` and ``radio_dark_on_light`` are
        connected here.  Because Qt emits ``toggled(True)`` for the button
        that is *becoming* checked **and** ``toggled(False)`` for the one
        being unchecked, we process only the ``checked=True`` signal to
        avoid running the logic twice per user click.

        Updates ``Analysis.is_dark_on_light``, invalidates the cached
        processed image, and redraws the canvas so the inversion is
        visible immediately.
        """
        if not checked:
            return   # the unchecked signal — ignore

        if self.current_analysis is None:
            return

        new_flag = self._is_dark_on_light()

        if self.current_analysis.is_dark_on_light == new_flag:
            return   # no actual change; skip the redraw

        self.current_analysis.is_dark_on_light = new_flag
        self.current_analysis.invalidate_image_cache()
        self.redraw_canvas()

    def _on_detect_lanes_clicked(self) -> None:
        """
        Run automatic lane detection and refresh the canvas overlay.

        Reads ``num_wells`` from the slider, delegates the algorithm to
        ``Analysis.auto_detect_lanes()``, then calls ``redraw_canvas()``
        so the lane rectangles appear immediately without blocking the UI.
        """
        if self.current_analysis is None:
            return

        num_wells = self.slider_num_wells.value()
        self.current_analysis.auto_detect_lanes(num_wells=num_wells)
        self.selected_lane_index = None   # lane geometry changed; old selection invalid
        self.redraw_canvas()
        self._set_status(
            f"{num_wells} lanes detected.\n"
            f"Lane width: {self.current_analysis.lanes[0].width} px"
            if self.current_analysis.lanes else "Detection returned no lanes."
        )
        self.update_header_info()
        self.toolbox.setCurrentIndex(2)   # advance to "Detect Bands" page

    def _on_detect_bands_clicked(self) -> None:
        """
        Run ``auto_detect_bands()`` on every lane using the sidebar
        parameters, then refresh the canvas.

        The band detection reads ``_temp_smoothed / _temp_corrected``
        from each Lane for the profile plot.  The status bar reports
        the total band count across all lanes.
        """
        if self.current_analysis is None:
            return

        bg_window  = self.slider_bg_window.value()
        prominence = self.slider_prominence.value() / 10.0

        total = 0
        for lane in self.current_analysis.lanes:
            lane.auto_detect_bands(
                bg_window=bg_window,
                prominence=prominence,
            )
            total += len(lane.bands)

        self.redraw_canvas()
        n_lanes = len(self.current_analysis.lanes)
        self._set_status(
            f"Band detection complete.\n"
            f"{total} band(s) across {n_lanes} lane(s)."
        )
        self.update_header_info()
        if total > 0:
            self.btn_export.setEnabled(True)
            self.grp_mw_cal.setEnabled(True)
            self.toolbox.setCurrentIndex(3)   # advance to "Calibration" page

    def on_calibrate_ladder_clicked(self) -> None:
        """
        Open the :class:`LadderConfigDialog` for the currently selected lane,
        fit the MW calibration curve, and predict MWs for all other bands.

        Workflow
        --------
        1.  Guard: an image and a lane selection must exist.
        2.  Show ``LadderConfigDialog`` for the selected lane.
        3.  On acceptance, tag calibrator bands with ``is_mw_calibrator=True``
            and their entered ``molecular_weight`` values.
        4.  Call ``fit_mw_calibration()`` — uses ``get_flattened_y()``
            internally to build the piecewise semi-log interpolant on
            ``(flattened_y, ln(MW))`` pairs.
        5.  Predict MW for every non-calibrator band in all lanes using
            ``predict_mw(get_flattened_y(...))`` and store in
            ``band.molecular_weight``.
        6.  Refresh the results table.
        """
        if self.current_analysis is None:
            return

        if self.selected_lane_index is None:
            QMessageBox.warning(
                self,
                "No Lane Selected",
                "Please click on a lane in the gel image to select it first,\n"
                "then press 'Set Selected Lane as Ladder'.",
            )
            return

        lanes = self.current_analysis.lanes
        if not (0 <= self.selected_lane_index < len(lanes)):
            return

        selected_lane = lanes[self.selected_lane_index]
        if not selected_lane.bands:
            QMessageBox.warning(
                self,
                "No Bands in Selected Lane",
                "The selected lane has no detected bands.\n"
                "Run 'Detect Bands' first.",
            )
            return

        dialog = LadderConfigDialog(selected_lane, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        cal_data = dialog.get_calibration_data()
        if not cal_data:
            QMessageBox.warning(
                self,
                "No Calibration Points",
                "No MW values were entered.  "
                "Please fill in at least 2 rows.",
            )
            return

        # ---- Apply calibration data to the selected lane ---------------
        for i, band in enumerate(selected_lane.bands):
            if i in cal_data:
                band.is_mw_calibrator = True
                band.molecular_weight = cal_data[i]
            else:
                band.is_mw_calibrator = False
                # Do not clear existing MWs yet

        try:
            # 1. Fit the semi-log curve — get_flattened_y() is called
            #    internally so no separate Rf computation is needed.
            self.current_analysis.fit_mw_calibration()

            # 2. Predict MW for all non-calibrator bands.
            for lane in self.current_analysis.lanes:
                lane_x = lane.path_points[0][0]
                for band in lane.bands:
                    if not getattr(band, "is_mw_calibrator", False):
                        band.molecular_weight = self.current_analysis.predict_mw(
                            self.current_analysis.get_flattened_y(
                                lane_x, band.peak_index
                            )
                        )

            # 3. Refresh UI.
            self.update_results_table()
            QMessageBox.information(
                self,
                "Success",
                "Molecular Weight curve generated and applied!",
            )

        except ValueError as exc:
            QMessageBox.warning(self, "Calibration Error", str(exc))

    def on_table_row_selected(self) -> None:
        """
        Qt ``itemSelectionChanged`` handler for ``results_table``.

        Reads the Band # from the selected row (column 1), converts it to a
        0-based index, stores it in ``highlighted_band_index``, and triggers
        a canvas redraw so both the profile fill and the image tick are
        highlighted in orange.  Deselecting a row clears the highlight.
        """
        selected = self.results_table.selectedItems()
        if not selected:
            self.highlighted_band_index = None
        else:
            row = self.results_table.currentRow()
            band_num_item = self.results_table.item(row, 1)   # "Band #" column
            if band_num_item is not None:
                try:
                    self.highlighted_band_index = int(band_num_item.text()) - 1
                except ValueError:
                    self.highlighted_band_index = None
            else:
                self.highlighted_band_index = None
        self.redraw_canvas()

    # ------------------------------------------------------------------
    # Manual drawing mode slots
    # ------------------------------------------------------------------

    def _on_add_lane_mode(self) -> None:
        """Switch to ADD_LANE mode (from menu); syncs the sidebar checkbox."""
        self.current_mode = "ADD_LANE"
        self.chk_add_lane.blockSignals(True)
        self.chk_add_lane.setChecked(True)
        self.chk_add_lane.blockSignals(False)
        self.chk_add_band.blockSignals(True)
        self.chk_add_band.setChecked(False)
        self.chk_add_band.blockSignals(False)
        self.statusBar().showMessage(
            "ADD LANE — Click on the 2D image to place a new lane center."
        )

    def _on_add_band_mode(self) -> None:
        """Switch to ADD_BAND mode (from menu); syncs the sidebar checkbox."""
        self.current_mode = "ADD_BAND"
        self.chk_add_band.blockSignals(True)
        self.chk_add_band.setChecked(True)
        self.chk_add_band.blockSignals(False)
        self.chk_add_lane.blockSignals(True)
        self.chk_add_lane.setChecked(False)
        self.chk_add_lane.blockSignals(False)
        self.statusBar().showMessage(
            "ADD BAND — Click on the gel image or the 1D profile to place a new peak."
        )

    def _on_toggle_smile_correction(self) -> None:
        """
        Toggle the parabolic smile-correction overlay on or off.

        When turned **on** for the first time, three default anchor points are
        placed at the horizontal thirds of the image at mid-height.  The user
        can then drag them to match the actual smile shape of the gel.

        When turned **off**, the overlay is hidden but ``smile_points`` is kept
        so it can be re-enabled without resetting the anchors.
        """
        if self.current_analysis is None:
            self.statusBar().showMessage("Load an image first.")
            return

        # Drive via the sidebar checkbox so both UI elements stay in sync.
        self.chk_smile.setChecked(not self.smile_correction_enabled)

    def _on_cancel_action(self) -> None:
        """Reset to NORMAL mode, uncheck all manual toggles, clear status prompt."""
        self.current_mode = "NORMAL"
        self._reset_manual_toggles()
        self.statusBar().showMessage("Ready")

    def _reset_manual_toggles(self) -> None:
        """Programmatically uncheck the ADD_LANE / ADD_BAND sidebar checkboxes."""
        for attr in ("chk_add_lane", "chk_add_band"):
            chk = getattr(self, attr, None)
            if chk is not None:
                chk.blockSignals(True)
                chk.setChecked(False)
                chk.blockSignals(False)

    def _on_chk_add_lane_toggled(self, checked: bool) -> None:
        """Sidebar checkbox for ADD_LANE mode."""
        if checked:
            # Deactivate the other mode checkbox first.
            self.chk_add_band.blockSignals(True)
            self.chk_add_band.setChecked(False)
            self.chk_add_band.blockSignals(False)
            self.current_mode = "ADD_LANE"
            self.statusBar().showMessage(
                "ADD LANE — Click on the 2D image to place a new lane center."
            )
        else:
            if self.current_mode == "ADD_LANE":
                self.current_mode = "NORMAL"
                self.statusBar().showMessage("Ready")

    def _on_chk_add_band_toggled(self, checked: bool) -> None:
        """Sidebar checkbox for ADD_BAND mode."""
        if checked:
            self.chk_add_lane.blockSignals(True)
            self.chk_add_lane.setChecked(False)
            self.chk_add_lane.blockSignals(False)
            self.current_mode = "ADD_BAND"
            self.statusBar().showMessage(
                "ADD BAND — Click on the gel image or the 1D profile to place a new peak."
            )
        else:
            if self.current_mode == "ADD_BAND":
                self.current_mode = "NORMAL"
                self.statusBar().showMessage("Ready")

    def _on_chk_smile_toggled(self, checked: bool) -> None:
        """Sidebar checkbox for smile correction — delegates to the shared logic."""
        if self.current_analysis is None:
            # Revert the checkbox; no image to operate on.
            self.chk_smile.blockSignals(True)
            self.chk_smile.setChecked(False)
            self.chk_smile.blockSignals(False)
            self.statusBar().showMessage("Load an image first.")
            return
        # Drive the shared state; avoid calling _on_toggle_smile_correction so
        # we don't flip the state an extra time.
        self.smile_correction_enabled = checked
        if checked:
            if self.current_analysis.smile_points is None:
                img         = self.current_analysis.get_image()
                img_h, img_w = img.shape
                self.current_analysis.smile_points = [
                    (img_w * 0.1, img_h * 0.5),
                    (img_w * 0.5, img_h * 0.5),
                    (img_w * 0.9, img_h * 0.5),
                ]
            self.statusBar().showMessage(
                "Smile Correction ON — Drag the red anchors to match the gel smile."
            )
        else:
            self.statusBar().showMessage("Smile Correction OFF.")
        self.redraw_canvas()
        self.recalculate_all_data()

    def _on_delete_selected_band(self) -> None:
        """
        Remove the currently highlighted band from the selected lane.

        Guards:
        - Image must be loaded.
        - A lane must be selected.
        - A band must be highlighted (via table row selection).
        """
        if self.current_analysis is None:
            return
        if self.selected_lane_index is None:
            return

        lanes = self.current_analysis.lanes
        if not (0 <= self.selected_lane_index < len(lanes)):
            return

        lane = lanes[self.selected_lane_index]

        if self.highlighted_band_index is None:
            self.statusBar().showMessage(
                "No band selected — click a row in the results table first."
            )
            return
        if not (0 <= self.highlighted_band_index < len(lane.bands)):
            return

        del lane.bands[self.highlighted_band_index]
        self.highlighted_band_index = None
        self.statusBar().showMessage("Band deleted.")
        self.redraw_canvas()
        self.update_results_table()
        self.update_header_info()

    def _on_canvas_click(self, event) -> None:
        """
        Matplotlib ``button_press_event`` handler.

        Behaviour depends on ``self.current_mode``:

        ADD_LANE
            Click inside ax_img → create a new straight Lane centred at the
            clicked x-coordinate and append it to the analysis.  Resets mode
            to "NORMAL" afterwards.

        ADD_BAND
            Click inside ax_profile → create a new Band at the clicked profile
            index (±15 px default width) and append it to the selected lane's
            band list.  Bands are re-sorted by peak_index so numbering stays
            top-to-bottom.  Resets mode to "NORMAL" afterwards.

        NORMAL
            Standard lane-selection: click inside ax_img selects the lane
            whose bounding box contains the x-coordinate.

        Note: if a band-boundary line pick is in progress (``_drag_active``),
        this handler exits immediately so the drag is not interrupted.
        """
        if self._drag_active:
            return

        if self.current_analysis is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        # ==============================================================
        # ADD_LANE mode
        # ==============================================================
        if self.current_mode == "ADD_LANE":
            if event.inaxes is not self.canvas.ax_img:
                return

            x = int(event.xdata)
            img_height = self.current_analysis.get_image().shape[0]

            new_lane = Lane(
                self.current_analysis,
                path_points=[(x, 0), (x, img_height)],
            )
            self.current_analysis.lanes.append(new_lane)

            self.current_mode = "NORMAL"
            self._reset_manual_toggles()
            self.statusBar().showMessage(
                f"Lane added at x={x}. Total lanes: "
                f"{len(self.current_analysis.lanes)}."
            )
            self.redraw_canvas()
            self.update_header_info()
            return

        # ==============================================================
        # ADD_BAND mode — accepts clicks on ax_profile OR ax_img
        # ==============================================================
        if self.current_mode == "ADD_BAND":
            ax_img     = self.canvas.ax_img
            ax_profile = self.canvas.ax_profile

            if event.inaxes not in (ax_img, ax_profile):
                return   # click was outside both relevant axes

            lanes = self.current_analysis.lanes

            # ----------------------------------------------------------
            # Determine target_y (profile row index) and resolve lane
            # ----------------------------------------------------------
            if event.inaxes is ax_profile:
                # Profile axis: x-coordinate IS the profile row index.
                if self.selected_lane_index is None:
                    self.statusBar().showMessage(
                        "No lane selected — click a lane in the gel image first."
                    )
                    return
                if not (0 <= self.selected_lane_index < len(lanes)):
                    return
                target_y = int(event.xdata)

            else:  # ax_img click
                # Find the lane whose bounding box contains the click x.
                clicked_x = event.xdata
                found_idx: Optional[int] = None
                for idx, lane in enumerate(lanes):
                    cx     = lane.path_points[0][0]
                    half_w = lane.width / 2.0
                    if (cx - half_w) <= clicked_x <= (cx + half_w):
                        found_idx = idx
                        break

                if found_idx is None:
                    self.statusBar().showMessage(
                        "Click landed outside every lane box — try again."
                    )
                    return

                # Auto-select the lane the user clicked on.
                self.selected_lane_index = found_idx
                target_y = int(event.ydata)

            # ----------------------------------------------------------
            # Build the Band with a guaranteed-safe window around target_y
            # ----------------------------------------------------------
            lane    = lanes[self.selected_lane_index]
            max_y   = len(lane.get_profile()) - 1

            # Clamp peak to the interior so ±1 slots always exist.
            target_y = max(1, min(max_y - 1, target_y))

            start_y = max(0,     target_y - 15)
            end_y   = min(max_y, target_y + 15)

            new_band = Band(
                parent_lane=lane,
                start_index=start_y,
                peak_index=target_y,
                end_index=end_y,
            )
            lane.bands.append(new_band)
            lane.bands.sort(key=lambda b: b.peak_index)

            self.current_mode = "NORMAL"
            self._reset_manual_toggles()
            self.statusBar().showMessage(
                f"Band added at row {target_y} in Lane "
                f"{self.selected_lane_index + 1}."
            )
            self.redraw_canvas()
            self.recalculate_all_data()
            self.update_header_info()
            return

        # ==============================================================
        # NORMAL mode — standard lane selection
        # ==============================================================
        if event.inaxes is not self.canvas.ax_img:
            return

        x = event.xdata

        # Find the lane whose bounding box [center ± half_width] contains x.
        # When multiple lanes overlap (shouldn't happen but is defensive), pick
        # the one whose centre is closest to the click.
        best_idx: Optional[int] = None
        best_dist = float("inf")
        for idx, lane in enumerate(self.current_analysis.lanes):
            x_center = lane.path_points[0][0]
            half_w   = lane.width / 2.0
            if (x_center - half_w) <= x <= (x_center + half_w):
                dist = abs(x - x_center)
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = idx

        if best_idx is not None and best_idx != self.selected_lane_index:
            self.selected_lane_index    = best_idx
            self.highlighted_band_index = None   # clear table highlight on lane change
            self.redraw_canvas()
            lane = self.current_analysis.lanes[best_idx]
            self._set_status(
                f"Lane {best_idx + 1} selected.\n"
                f"{len(lane.bands)} band(s) detected."
            )

    # ------------------------------------------------------------------
    # Band-boundary drag handlers
    # ------------------------------------------------------------------

    def _reset_drag_state(self) -> None:
        """Return all drag state variables to their initial idle values."""
        self._drag_active = False
        self._drag_band   = None
        self._drag_edge   = None
        self._drag_line   = None
        self._drag_axis   = None

    def on_line_pick(self, event) -> None:
        """
        Matplotlib ``pick_event`` handler — fires when the user clicks on
        a pickable artist (here, a band-boundary ``axvline``).

        Only left-click (button 1) initiates a drag.  The custom attributes
        ``band_ref`` and ``edge_type`` that were injected onto the
        ``Line2D`` objects in ``redraw_canvas()`` are read here to identify
        which band boundary was grabbed.
        """
        # Only react to left mouse button presses.
        if event.mouseevent.button != 1:
            return

        artist = event.artist

        # Smile-correction anchor — draggable red marker on ax_img.
        if getattr(artist, "is_smile_anchor", False):
            self._drag_active = True
            self._drag_line   = artist
            self._drag_band   = None
            self._drag_edge   = artist.anchor_idx   # int 0 / 1 / 2
            self._drag_axis   = "smile"
            return

        if not hasattr(artist, "band_ref"):
            return   # not a boundary line we drew

        self._drag_active = True
        self._drag_line   = artist
        self._drag_band   = artist.band_ref
        self._drag_edge   = artist.edge_type
        self._drag_axis   = getattr(artist, "axis_type", "profile")

    def on_mouse_motion(self, event) -> None:
        """
        Matplotlib ``motion_notify_event`` handler.

        Handles two drag types:
        • profile-axis drag (start/end boundaries) — moves line horizontally.
        • image-axis drag  (peak markers)          — moves line vertically.

        ``draw_idle()`` coalesces rapid events into one repaint so the drag
        stays smooth without blocking the Qt event loop.
        """
        if not self._drag_active:
            return

        if self._drag_axis == "smile":
            # Dragging a smile-correction anchor on ax_img.
            if event.inaxes is not self.canvas.ax_img:
                return
            if event.xdata is None or event.ydata is None:
                return

            anchor_idx = self._drag_edge   # int
            new_x, new_y = event.xdata, event.ydata

            # Update the data model live.
            pts = self.current_analysis.smile_points
            pts[anchor_idx] = (new_x, new_y)

            # Move the marker.
            self._drag_line.set_xdata([new_x])
            self._drag_line.set_ydata([new_y])

            # Recompute the parabola and update the dashed line.
            if self._smile_parabola_line is not None:
                X = [p[0] for p in pts]
                Y = [p[1] for p in pts]
                a, b, c = np.polyfit(X, Y, 2)
                x_span  = np.linspace(
                    0,
                    self.current_analysis.get_image().shape[1] - 1,
                    max(300, self.current_analysis.get_image().shape[1]),
                )
                y_span = a * x_span ** 2 + b * x_span + c
                self._smile_parabola_line.set_xdata(x_span)
                self._smile_parabola_line.set_ydata(y_span)

        elif self._drag_axis == "image":
            if event.inaxes is not self.canvas.ax_img:
                return
            if event.ydata is None:
                return
            # Move the blue tick on the gel image vertically.
            self._drag_line.set_ydata([event.ydata, event.ydata])
            # Mirror the change live on the profile panel:
            # the image y-coordinate equals the profile x-index.
            profile_line = self._peak_profile_lines.get(id(self._drag_band))
            if profile_line is not None:
                profile_line.set_xdata([event.ydata, event.ydata])
        else:  # 'profile'
            if event.inaxes is not self.canvas.ax_profile:
                return
            if event.xdata is None:
                return
            self._drag_line.set_xdata([event.xdata, event.xdata])

        self.canvas.draw_idle()

    def on_mouse_release(self, event) -> None:
        """
        Matplotlib ``button_release_event`` handler.

        Commits the dragged boundary position to the ``Band`` data model,
        then triggers a full redraw so the filled AUC area and the results
        table both update instantly.

        Edge cases handled:
        - ``event.xdata is None`` (released outside the axes) → drag is
          cancelled without modifying the band.
        - The new position violates the Band invariant (start < peak < end)
          → the ``Band`` setter raises ``ValueError``; the error is caught,
          reported in the status label, and the drag is cancelled cleanly.
        """
        if not self._drag_active:
            return

        # Capture mutable state before resetting so the try-block can use it.
        drag_band = self._drag_band
        drag_edge = self._drag_edge
        drag_axis = self._drag_axis

        # Always reset drag state first — even if the commit fails, the
        # next redraw_canvas() call will restore the line to its correct position.
        self._reset_drag_state()

        # ---- Smile anchor release ----------------------------------------
        if drag_axis == "smile":
            # smile_points was already updated live in on_mouse_motion;
            # just do a full redraw and recalculate.
            self.redraw_canvas()
            self.recalculate_all_data()
            return

        # ---- Band-boundary release ----------------------------------------
        # Resolve the new coordinate from the appropriate axis.
        new_coord: Optional[int] = None
        if drag_axis == "image":
            if event.ydata is not None:
                new_coord = int(round(event.ydata))
        else:  # 'profile'
            if event.xdata is not None:
                new_coord = int(round(event.xdata))

        if new_coord is not None:
            try:
                if drag_edge == "start":
                    drag_band.start_index = new_coord
                elif drag_edge == "end":
                    drag_band.end_index = new_coord
                elif drag_edge == "peak":
                    drag_band.peak_index = new_coord
            except ValueError as exc:
                self._set_status(
                    f"Invalid position: {exc}",
                    error=True,
                )

        # Redraw the profile (restores correct line positions) and refresh
        # the AUC numbers in the table, re-applying the MW curve if fitted.
        self.redraw_canvas()
        self.recalculate_all_data()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_dark_on_light(self) -> bool:
        """
        Read the radio-button selection and return the corresponding
        ``is_dark_on_light`` flag.

        Returns
        -------
        bool
            ``True``  — "Dark bands on light background" is selected
                        (Coomassie, silver stain, etc.)
            ``False`` — "Light bands on dark background" is selected
                        (fluorescent dyes, ethidium bromide, etc.)
        """
        return self.radio_dark_on_light.isChecked()

    def update_header_info(self) -> None:
        """Refresh the top info bar with current filename / lane / band counts."""
        if self.current_analysis is None or not self.image_path:
            self.header_info_label.setText("🟢 No Image Loaded | 0 lanes • 0 bands")
            return
        fname     = os.path.basename(self.image_path)
        num_lanes = len(self.current_analysis.lanes)
        num_bands = sum(len(lane.bands) for lane in self.current_analysis.lanes)
        self.header_info_label.setText(
            f"🟢 {fname} | {num_lanes} lanes • {num_bands} bands"
        )

    def redraw_canvas(self) -> None:
        """
        Composite all visual layers onto both Matplotlib axes and flush to
        the Qt widget.  This is the single authoritative render call.

        Image axis (``ax_img``)
        -----------------------
        * Gel image (grayscale).
        * Per-lane ``Rectangle`` overlays — **red / thin** for unselected,
          **yellow / thick** for the currently selected lane.
        * Numbered badges at the top of each rectangle.
        * Per-band horizontal blue tick lines at each ``band.peak_index``
          y-coordinate, spanning the full lane width.

        Profile axis (``ax_profile``)
        ------------------------------
        * If a lane is selected *and* its ``_temp_*`` arrays are populated
          (i.e. ``auto_detect_bands`` has been run):

          - Blue solid line   — Gaussian-smoothed raw profile.
          - Red dashed line   — morphological background estimate.
          - Green solid line  — baseline-corrected signal.
          - Blue filled areas — one ``fill_between`` per detected band,
            between ``start_index`` and ``end_index`` under the corrected
            curve.
        * Otherwise a "click a lane" hint message is shown.

        All ``None``-checks prevent crashes when called before detection or
        after resetting state.
        """
        ax_img     = self.canvas.ax_img
        ax_profile = self.canvas.ax_profile

        # ------------------------------------------------------------------
        # Guard: no image loaded
        # ------------------------------------------------------------------
        if self.current_analysis is None:
            self.canvas.clear()
            return

        img = self.current_analysis.get_image()   # (H, W) uint8
        img_height, img_width = img.shape

        # ==================================================================
        # Image axis
        # ==================================================================
        ax_img.clear()
        ax_img.set_axis_off()
        ax_img.set_facecolor("#060911")
        self.canvas.figure.patch.set_facecolor("#060911")
        ax_img.imshow(img, cmap="gray", vmin=0, vmax=255, origin="upper")

        for idx, lane in enumerate(self.current_analysis.lanes):
            x_center = lane.path_points[0][0]
            half_w   = lane.width / 2.0
            is_sel   = (idx == self.selected_lane_index)

            # Selected lane: orange + thicker border; others: red + thin.
            edge_color  = "#f97316" if is_sel else "#ef4444"
            line_width  = 2.0       if is_sel else 0.8
            badge_color = "#f97316" if is_sel else "#ef4444"

            rect = mpatches.Rectangle(
                xy=(x_center - half_w, -0.5),
                width=lane.width,
                height=img_height,
                linewidth=line_width,
                edgecolor=edge_color,
                facecolor=edge_color,
                alpha=0.06,
                zorder=3,
            )
            ax_img.add_patch(rect)

            # Lane-number badge just inside the top edge.
            ax_img.text(
                x_center, 8, str(idx + 1),
                color="white", fontsize=7,
                ha="center", va="top", zorder=4,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor=badge_color,
                    edgecolor="none",
                    alpha=0.85,
                ),
            )

            # Band peak markers — draggable horizontal ticks across the lane.
            # The highlighted band (matching results-table selection) is drawn
            # in orange and thicker so it stands out on the physical gel image.
            for band_i, band in enumerate(lane.bands):
                y             = band.peak_index
                is_highlighted = (
                    is_sel and band_i == self.highlighted_band_index
                )
                (peak_line,) = ax_img.plot(
                    [x_center - half_w, x_center + half_w],
                    [y, y],
                    color="#f97316" if is_highlighted else "#60a5fa",
                    linewidth=3   if is_highlighted else 1.5,
                    solid_capstyle="butt",
                    zorder=6      if is_highlighted else 5,
                    picker=5,
                )
                peak_line.band_ref  = band
                peak_line.edge_type = "peak"
                peak_line.axis_type = "image"

        # ==================================================================
        # Parabolic smile-correction overlay
        # ==================================================================
        self._smile_anchor_artists = []
        self._smile_parabola_line  = None

        if (
            self.smile_correction_enabled
            and self.current_analysis.smile_points is not None
            and len(self.current_analysis.smile_points) >= 3
        ):
            pts = self.current_analysis.smile_points
            X   = [p[0] for p in pts]
            Y   = [p[1] for p in pts]

            # Fit and draw the parabolic curve.
            a, b, c = np.polyfit(X, Y, 2)
            x_span  = np.linspace(0, img_width - 1, max(300, img_width))
            y_span  = a * x_span ** 2 + b * x_span + c
            (self._smile_parabola_line,) = ax_img.plot(
                x_span, y_span,
                color="#eab308", linestyle="--", linewidth=1.5,
                alpha=0.90, zorder=9,
            )

            # Draw the three draggable anchor markers.
            for i, (sx, sy) in enumerate(pts):
                (anchor,) = ax_img.plot(
                    [sx], [sy],
                    marker="o", markersize=8,
                    color="#ef4444", linestyle="none",
                    markeredgecolor="white", markeredgewidth=0.8,
                    zorder=10, picker=5,
                )
                anchor.is_smile_anchor = True
                anchor.anchor_idx      = i
                self._smile_anchor_artists.append(anchor)

        # Lock limits to image extent so added patches don't trigger rescaling.
        ax_img.set_xlim(-0.5, img_width  - 0.5)
        ax_img.set_ylim(img_height - 0.5, -0.5)   # y = 0 at top

        # ==================================================================
        # Profile axis
        # ==================================================================
        ax_profile.clear()
        _style_axes(ax_profile)
        ax_profile.set_xlabel("Profile index (px)", fontsize=7)
        ax_profile.set_ylabel("Intensity",           fontsize=7)

        sel = self.selected_lane_index
        lanes = self.current_analysis.lanes

        if sel is not None and 0 <= sel < len(lanes):
            lane       = lanes[sel]
            smoothed   = getattr(lane, "_temp_smoothed",   None)
            background = getattr(lane, "_temp_background", None)
            corrected  = getattr(lane, "_temp_corrected",  None)

            if smoothed is not None:
                # x-axis = profile index
                x_prof = np.arange(len(smoothed))

                ax_profile.plot(
                    x_prof, smoothed,
                    color="#60a5fa", lw=2.0, label="Smoothed", zorder=3,
                )
                if background is not None:
                    ax_profile.plot(
                        x_prof, background,
                        color="#475569", lw=1.0, ls="--",
                        label="Background", zorder=2,
                    )
                if corrected is not None:
                    ax_profile.plot(
                        x_prof, corrected,
                        color="#34d399", lw=1.2, label="Corrected", zorder=3,
                    )
                    # Rebuild the peak-indicator lookup for live drag sync.
                    self._peak_profile_lines.clear()

                    # Per-band: filled area + draggable boundary lines.
                    for band_i, band in enumerate(lane.bands):
                        xs             = np.arange(band.start_index, band.end_index + 1)
                        is_highlighted = (band_i == self.highlighted_band_index)

                        # Shaded AUC region — orange when highlighted, teal otherwise.
                        ax_profile.fill_between(
                            xs,
                            0,
                            corrected[band.start_index : band.end_index + 1],
                            color="#f97316" if is_highlighted else "#34d399",
                            alpha=0.40      if is_highlighted else 0.20,
                            zorder=3        if is_highlighted else 2,
                        )

                        # Peak-position indicator — mirrors the image tick.
                        peak_vline = ax_profile.axvline(
                            x=band.peak_index,
                            color="#f97316" if is_highlighted else "#ef4444",
                            linestyle=":",
                            linewidth=1.5   if is_highlighted else 1.0,
                            alpha=0.85      if is_highlighted else 0.50,
                            zorder=5        if is_highlighted else 4,
                        )
                        self._peak_profile_lines[id(band)] = peak_vline

                        # ---- Draggable start boundary line ----
                        bnd_color = "#f97316" if is_highlighted else "#ef4444"
                        bnd_lw    = 1.5       if is_highlighted else 1.0
                        line_start = ax_profile.axvline(
                            x=band.start_index,
                            color=bnd_color,
                            linestyle=":",
                            linewidth=bnd_lw,
                            alpha=0.85 if is_highlighted else 0.50,
                            picker=5,
                            zorder=6 if is_highlighted else 5,
                        )
                        line_start.band_ref  = band
                        line_start.edge_type = "start"

                        # ---- Draggable end boundary line ----
                        line_end = ax_profile.axvline(
                            x=band.end_index,
                            color=bnd_color,
                            linestyle=":",
                            linewidth=bnd_lw,
                            alpha=0.85 if is_highlighted else 0.50,
                            picker=5,
                            zorder=6 if is_highlighted else 5,
                        )
                        line_end.band_ref  = band
                        line_end.edge_type = "end"

                ax_profile.set_title(
                    f"1D Profile — Lane {sel + 1}",
                    color="#94a3b8", fontsize=9, pad=4,
                )
                ax_profile.legend(
                    fontsize=6,
                    labelcolor="#94a3b8",
                    facecolor="#060911",
                    edgecolor="#475569",
                    loc="upper right",
                )
            else:
                # Lane selected but bands not yet detected.
                ax_profile.text(
                    0.5, 0.5,
                    f"Lane {sel + 1} selected.\n"
                    "Run 'Detect Bands' to\nview the profile.",
                    transform=ax_profile.transAxes,
                    ha="center", va="center",
                    color="#585b70", fontsize=8, fontstyle="italic",
                )
        else:
            # No lane selected.
            ax_profile.text(
                0.5, 0.5,
                "Click a lane in\nthe image to view\nits 1D profile.",
                transform=ax_profile.transAxes,
                ha="center", va="center",
                color="#585b70", fontsize=8, fontstyle="italic",
            )

        # ------------------------------------------------------------------
        # Single flush at the very end — avoids visible incremental repaints.
        # ------------------------------------------------------------------
        self.canvas.draw()

        # Keep the results table in sync with the current visual state.
        self.update_results_table()

    # ------------------------------------------------------------------
    # Downstream recalculation
    # ------------------------------------------------------------------

    def recalculate_all_data(self) -> None:
        """
        Full downstream recalculation in the correct order of operations.

        Called automatically after any geometry change (drag release, manual
        band / lane insertion, smile-anchor move) so the UI always reflects
        the current state without requiring the user to re-run calibration.

        Order of operations
        -------------------
        1.  **MW curve re-fit** — ``fit_mw_calibration()`` is attempted.
            It calls ``get_flattened_y()`` internally, so smile correction is
            automatically incorporated.  If fewer than 2 calibrator bands
            exist the ``ValueError`` is silently swallowed (the previously
            fitted curve, if any, remains in place).
        2.  **MW prediction** — for every non-calibrator band,
            ``predict_mw(get_flattened_y(...))`` overwrites
            ``band.molecular_weight``.  Left unchanged if no curve is fitted.
        3.  **Table refresh** — ``update_results_table()`` recomputes AUC
            and Relative % for the currently selected lane.
        """
        if self.current_analysis is None:
            self.update_results_table()
            return

        # ---- Step 1: Re-fit the MW calibration curve --------------------
        # fit_mw_calibration() calls get_flattened_y() internally, so no
        # separate Rf pre-computation step is needed.
        try:
            self.current_analysis.fit_mw_calibration()
        except ValueError:
            pass   # not enough calibrators yet — keep any existing curve

        # ---- Step 2: Predict MW for non-calibrator bands ----------------
        if self.current_analysis._mw_interp is not None:
            for lane in self.current_analysis.lanes:
                lane_x = lane.path_points[0][0]
                for band in lane.bands:
                    if not getattr(band, "is_mw_calibrator", False):
                        band.molecular_weight = self.current_analysis.predict_mw(
                            self.current_analysis.get_flattened_y(
                                lane_x, band.peak_index
                            )
                        )

        # ---- Step 3: Refresh the results table --------------------------
        self.update_results_table()

    # ------------------------------------------------------------------
    # Results table — setup and population
    # ------------------------------------------------------------------

    def _configure_results_table(self) -> None:
        """
        Set column headers, resize policy, edit restrictions, and visual
        styling on ``self.results_table``.

        Called once from ``_build_ui()``; never needs to be called again.
        """
        _COLUMNS = [
            "Lane",
            "Band #",
            "Peak Position (px)",
            "Raw Volume (AUC)",
            "Relative Amount (%)",
            "MW (kDa)",
        ]
        self.results_table.setColumnCount(len(_COLUMNS))
        self.results_table.setHorizontalHeaderLabels(_COLUMNS)
        self.results_table.setRowCount(0)

        # Read-only: no in-place editing.
        self.results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        # Clicking a cell selects the whole row.
        self.results_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.results_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )

        # Each column stretches to fill available width equally.
        hdr = self.results_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hdr.setHighlightSections(False)

        # Hide the vertical index column — row numbers are in the data.
        self.results_table.verticalHeader().setVisible(False)

        self.results_table.setMinimumHeight(120)
        self.results_table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        # Dark theme to match the rest of the application.
        self.results_table.setStyleSheet(
            """
            QTableWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                gridline-color: #313244;
                border: none;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 4px 8px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #45475a;
                color: #cdd6f4;
            }
            QHeaderView::section {
                background-color: #181825;
                color: #a6adc8;
                border: none;
                border-bottom: 1px solid #313244;
                padding: 5px 8px;
                font-weight: 600;
                font-size: 11px;
            }
            QScrollBar:vertical {
                background: #1e1e2e;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #45475a;
                border-radius: 4px;
            }
            """
        )

    def update_results_table(self) -> None:
        """
        Repopulate ``results_table`` to reflect the currently selected lane.

        Shows every band in the selected lane with its:
        - Lane number (1-based)
        - Band number within the lane (1-based)
        - Peak position in profile coordinates (px)
        - Raw integrated volume (AUC) formatted to 2 decimal places
        - Volume as a percentage of the lane's total band volume,
          formatted to 1 decimal place

        If no image is loaded, no lane is selected, or the selected lane
        has no bands, the table is simply cleared.

        The ``total_lane_volume`` denominator is the sum of all band
        volumes in the lane (not the global total across all lanes), which
        is the conventional normalisation for relative band quantification.
        """
        # Block itemSelectionChanged while we repopulate the table to avoid
        # triggering on_table_row_selected (→ redraw_canvas) mid-repopulation,
        # which would cause infinite recursion since redraw_canvas calls us.
        self.results_table.blockSignals(True)
        self.results_table.setRowCount(0)   # clear all existing rows

        if self.current_analysis is None:
            self.results_table.blockSignals(False)
            return
        if self.selected_lane_index is None:
            self.results_table.blockSignals(False)
            return

        lanes = self.current_analysis.lanes
        if not (0 <= self.selected_lane_index < len(lanes)):
            self.results_table.blockSignals(False)
            return

        lane = lanes[self.selected_lane_index]
        if not lane.bands:
            self.results_table.blockSignals(False)
            return

        mode = self.current_analysis.volume_calc_mode

        # Pre-calculate all volumes so the denominator is known before
        # we start inserting rows.
        volumes = [band.get_raw_volume(mode) for band in lane.bands]
        total_lane_volume = sum(volumes)

        for band_idx, (band, vol) in enumerate(zip(lane.bands, volumes)):
            pct = (vol / total_lane_volume * 100.0) if total_lane_volume > 0 else 0.0

            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

            mw_text = (
                f"{band.molecular_weight:.2f}"
                if band.molecular_weight is not None
                else "-"
            )
            cells = [
                str(self.selected_lane_index + 1),   # Lane (1-based)
                str(band_idx + 1),                    # Band # (1-based)
                str(band.peak_index),                 # Peak Position (px)
                f"{vol:.2f}",                         # Raw Volume
                f"{pct:.1f}",                         # Relative Amount
                mw_text,                              # MW (kDa)
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
                )
                self.results_table.setItem(row, col, item)

        self.results_table.blockSignals(False)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_to_csv(self) -> None:
        """
        Write quantification results for *all lanes* to a CSV file chosen
        by the user via a save dialog.

        Columns
        -------
        Lane_Number, Band_Number, Peak_Y_px, Start_Y_px, End_Y_px,
        Raw_Volume, Relative_Pct

        ``Relative_Pct`` is the band's volume as a percentage of the
        total volume for its own lane (same normalisation as the table).

        A success dialog is shown after writing.  File-system errors are
        caught and reported in the status label without crashing.
        """
        if self.current_analysis is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results to CSV",
            "gel_results.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return   # user cancelled

        mode = self.current_analysis.volume_calc_mode
        _HEADERS = [
            "Lane_Number", "Band_Number",
            "Peak_Y_px", "Start_Y_px", "End_Y_px",
            "Raw_Volume", "Relative_Pct", "MW_kDa",
        ]

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(_HEADERS)

                for lane_idx, lane in enumerate(self.current_analysis.lanes):
                    if not lane.bands:
                        continue

                    volumes = [b.get_raw_volume(mode) for b in lane.bands]
                    total   = sum(volumes)

                    for band_idx, (band, vol) in enumerate(
                        zip(lane.bands, volumes)
                    ):
                        pct = (vol / total * 100.0) if total > 0 else 0.0
                        mw_str = (
                            f"{band.molecular_weight:.4f}"
                            if band.molecular_weight is not None
                            else ""
                        )
                        writer.writerow([
                            lane_idx + 1,
                            band_idx + 1,
                            band.peak_index,
                            band.start_index,
                            band.end_index,
                            f"{vol:.4f}",
                            f"{pct:.2f}",
                            mw_str,
                        ])

            QMessageBox.information(
                self,
                "Export Successful",
                f"Results exported successfully to:\n\n{file_path}",
            )
            self._set_status(f"Exported: {os.path.basename(file_path)}")

        except OSError as exc:
            self._set_status(f"Export failed:\n{exc}", error=True)

    # ------------------------------------------------------------------
    # Status bar helper
    # ------------------------------------------------------------------

    def _set_status(self, message: str, *, error: bool = False) -> None:
        """Update the sidebar status label."""
        colour = "#f38ba8" if error else "#a6e3a1"
        self.status_label.setStyleSheet(f"color: {colour};")
        self.status_label.setText(message)


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    """Instantiate the QApplication and run the event loop."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")   # consistent cross-platform look

    # Load and apply the Observatory dark stylesheet when it exists alongside
    # this script.  Missing file is silently ignored so the app still runs.
    qss_path = Path(__file__).with_name("gelanalyzer_observatory.qss")
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    window = GelAnalyzerApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
