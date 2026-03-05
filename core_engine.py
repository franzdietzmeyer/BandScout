"""
core_engine.py
==============
Object-Oriented Data Model for 1D PAGE (Polyacrylamide Gel Electrophoresis)
analysis.

Class hierarchy
---------------
    Analysis                  ← top-level session container; owns the raw image
    └── Lane                  ← one vertical (or curved) lane per gel channel
        └── Band              ← one detected band (peak) per lane

Coordinate systems
------------------
* Image coordinates  : (x, y) in pixel space, origin at the top-left corner.
* Profile coordinates: integer index i ∈ [0, N−1] along a Lane's 1D profile.
  Index 0 corresponds to the *top* of the lane; index N−1 to the *bottom*.
  All Band indices refer to profile coordinates, NOT raw image pixels.

Dark-on-light convention
------------------------
Many staining protocols (Coomassie Blue, silver stain, etc.) produce dark
bands on a bright background.  Setting ``Analysis.is_dark_on_light = True``
causes ``get_image()`` to apply a pixel inversion (255 − value), so that
dark band regions become *high-intensity* features.  This is required for
correct volume integration (we always integrate *peaks*, not troughs).
"""

from __future__ import annotations  # Enables forward references in type hints

import warnings
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage
from scipy.interpolate import interp1d, splev, splprep
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# ===========================================================================
# Enums
# ===========================================================================


class VolumeCalcMode(Enum):
    """
    Strategy for integrating band volume against the lane background.

    Choosing the right mode
    -----------------------
    NO_BACKGROUND
        Ignores any background baseline; sums the raw profile intensities
        over the band window.  Valid only when the background is
        demonstrably flat and negligible.  Overestimates volume for bands
        sitting on a sloped baseline.

    ALLOW_NEGATIVE
        Subtracts the interpolated baseline intensity pixel-by-pixel before
        summing.  Mathematically unbiased but can yield negative total volumes
        if the profile dips below the baseline inside the band window (e.g.
        due to noise or a mis-placed baseline).

    ZERO_CLIPPED  *(recommended default)*
        Same as ALLOW_NEGATIVE but each pixel-level contribution is clamped
        to zero before summation: ``max(0, profile − background)``.
        Guarantees volume ≥ 0 and is robust to minor baseline
        over-subtraction artefacts.
    """

    NO_BACKGROUND  = auto()
    ALLOW_NEGATIVE = auto()
    ZERO_CLIPPED   = auto()


class CurveFitType(Enum):
    """
    Mathematical model used when fitting a calibration curve.

    LINEAR      y = a·x + b
    QUADRATIC   y = a·x² + b·x + c
    LINEAR_LOG  y = a·ln(x) + b   ← semi-log; the standard model for MW vs Rf
                                     in SDS-PAGE (Ferguson plot)
    LOG         y = exp(a·x + b)  ← log-linear; inverse of the above
    """

    LINEAR     = auto()
    QUADRATIC  = auto()
    LINEAR_LOG = auto()
    LOG        = auto()


# ===========================================================================
# Private calibration model functions
# (consumed by scipy.optimize.curve_fit; exposed via _FIT_FUNCTIONS)
# ===========================================================================


def _model_linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """y = a·x + b"""
    return a * x + b


def _model_quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """y = a·x² + b·x + c"""
    return a * x ** 2 + b * x + c


def _model_linear_log(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    y = a·ln(x) + b

    This is the standard Ferguson-plot model for SDS-PAGE: log(MW) varies
    linearly with Rf.  A NaN-safe path is used so that downstream code
    receives NaN rather than a hard crash when x ≤ 0 is encountered.
    """
    if np.any(np.asarray(x) <= 0):
        warnings.warn(
            "LINEAR_LOG model received x ≤ 0; those entries will produce NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
    x_safe = np.where(np.asarray(x) > 0, x, np.nan)
    return a * np.log(x_safe) + b


def _model_log(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """y = exp(a·x + b)  — log-linear (inverse of LINEAR_LOG)."""
    return np.exp(a * np.asarray(x, dtype=float) + b)


# Dispatch table: CurveFitType → model callable
_FIT_FUNCTIONS: Dict[CurveFitType, Callable] = {
    CurveFitType.LINEAR:     _model_linear,
    CurveFitType.QUADRATIC:  _model_quadratic,
    CurveFitType.LINEAR_LOG: _model_linear_log,
    CurveFitType.LOG:        _model_log,
}


# ===========================================================================
# Band
# ===========================================================================


class Band:
    """
    A single electrophoretic band (peak) within a Lane's 1D intensity profile.

    Geometry invariant
    ------------------
        start_index  <  peak_index  <  end_index

    All three indices are in *profile coordinates* (positions in the array
    returned by ``Lane.get_profile()``), not raw image pixel coordinates.

    Calibration flags
    -----------------
    ``is_quant_calibrator``
        Mark this band as a known-quantity standard.
        Requires ``calibrated_volume`` to be set.

    ``is_mw_calibrator``
        Mark this band as a known-MW standard (e.g. a molecular-weight ladder
        band).  Requires ``molecular_weight`` to be set.
    """

    def __init__(
        self,
        start_index: int,
        peak_index: int,
        end_index: int,
        parent_lane: "Lane",
        *,
        calibrated_volume: Optional[float] = None,
        molecular_weight: Optional[float] = None,
        is_quant_calibrator: bool = False,
        is_mw_calibrator: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        start_index : int
            Left (upper) boundary in profile coordinates.
        peak_index : int
            Index of the intensity maximum.
        end_index : int
            Right (lower) boundary in profile coordinates.
        parent_lane : Lane
            The Lane object that owns this Band.
        calibrated_volume : float, optional
            Known quantity / concentration (calibrators only).
        molecular_weight : float, optional
            Known MW in kDa (MW calibrators only).
        is_quant_calibrator : bool
            Include this band in the quantity calibration fit.
        is_mw_calibrator : bool
            Include this band in the MW calibration fit.
        """
        # --- Type validation ---
        for name, val in (
            ("start_index", start_index),
            ("peak_index",  peak_index),
            ("end_index",   end_index),
        ):
            if not isinstance(val, int):
                raise TypeError(
                    f"{name} must be an int; got {type(val).__name__!r}."
                )

        # --- Ordering invariant ---
        if not (start_index < peak_index < end_index):
            raise ValueError(
                "Band index invariant violated: expected "
                f"start_index ({start_index}) < peak_index ({peak_index}) "
                f"< end_index ({end_index})."
            )

        # Store in private backing fields; exposed via validated properties.
        self._start_index: int = start_index
        self._peak_index:  int = peak_index
        self._end_index:   int = end_index

        self.parent_lane: Lane = parent_lane

        # Annotation / calibration attributes — mutable by the user.
        self.calibrated_volume: Optional[float] = calibrated_volume
        self.molecular_weight:  Optional[float] = molecular_weight
        self.is_quant_calibrator: bool          = is_quant_calibrator
        self.is_mw_calibrator:    bool          = is_mw_calibrator

    # ------------------------------------------------------------------
    # Properties — cross-validated so the invariant is never broken
    # ------------------------------------------------------------------

    @property
    def start_index(self) -> int:
        """Left (upper) boundary of the band in profile coordinates."""
        return self._start_index

    @start_index.setter
    def start_index(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("start_index must be an int.")
        if value >= self._peak_index:
            raise ValueError(
                f"start_index ({value}) must be strictly less than "
                f"peak_index ({self._peak_index})."
            )
        self._start_index = value

    @property
    def peak_index(self) -> int:
        """Index of the maximum intensity sample within this band."""
        return self._peak_index

    @peak_index.setter
    def peak_index(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("peak_index must be an int.")
        if not (self._start_index < value < self._end_index):
            raise ValueError(
                f"peak_index ({value}) must satisfy "
                f"start_index ({self._start_index}) < peak_index "
                f"< end_index ({self._end_index})."
            )
        self._peak_index = value

    @property
    def end_index(self) -> int:
        """Right (lower) boundary of the band in profile coordinates."""
        return self._end_index

    @end_index.setter
    def end_index(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("end_index must be an int.")
        if value <= self._peak_index:
            raise ValueError(
                f"end_index ({value}) must be strictly greater than "
                f"peak_index ({self._peak_index})."
            )
        self._end_index = value

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def get_peak_value(self) -> float:
        """
        Return the processed intensity at the band's peak position.

        Delegates to the parent Lane (which queries the parent Analysis for
        the grayscale, possibly inverted image).

        Returns
        -------
        float
            Intensity value at ``profile[peak_index]``.
        """
        profile = self.parent_lane.get_profile()
        return float(profile[self._peak_index])

    def get_raw_volume(self, mode: VolumeCalcMode) -> float:
        """
        Integrate the band intensity within ``[start_index, end_index]``
        (inclusive), yielding a scalar *volume* value.

        The integration is a discrete sum equivalent to the area under the
        profile curve between the band boundaries, with optional baseline
        subtraction controlled by *mode*.

        Parameters
        ----------
        mode : VolumeCalcMode
            Determines background-subtraction behaviour:

            ``NO_BACKGROUND``
                ``Σ profile[i]``  for i in window.
            ``ALLOW_NEGATIVE``
                ``Σ (profile[i] − background[i])``  for i in window.
            ``ZERO_CLIPPED``
                ``Σ max(0, profile[i] − background[i])``  for i in window.

        Returns
        -------
        float
            Scalar volume.  Guaranteed ≥ 0 when mode is ``ZERO_CLIPPED``.
        """
        profile    = self.parent_lane.get_profile()
        background = self.parent_lane.get_background_profile()

        # Inclusive slice over the band window
        sl           = slice(self._start_index, self._end_index + 1)
        prof_window  = profile[sl].astype(np.float64)
        bg_window    = background[sl].astype(np.float64)

        if mode is VolumeCalcMode.NO_BACKGROUND:
            return float(prof_window.sum())

        elif mode is VolumeCalcMode.ALLOW_NEGATIVE:
            return float((prof_window - bg_window).sum())

        elif mode is VolumeCalcMode.ZERO_CLIPPED:
            return float(np.maximum(0.0, prof_window - bg_window).sum())

        else:
            raise ValueError(f"Unrecognised VolumeCalcMode: {mode!r}")

    def __repr__(self) -> str:
        return (
            f"Band(start={self._start_index}, peak={self._peak_index}, "
            f"end={self._end_index}, "
            f"cal_vol={self.calibrated_volume}, mw={self.molecular_weight})"
        )


# ===========================================================================
# Lane
# ===========================================================================


class Lane:
    """
    A single vertical (straight or curved) lane on the gel image.

    The lane is defined by its *centreline path* — an ordered list of
    ``(x, y)`` pixel coordinates tracing the lane from top to bottom.

    Profile caching
    ---------------
    ``get_profile()`` and ``get_background_profile()`` cache their results
    after the first computation.  If you mutate ``path_points``, ``width``,
    or ``background_points`` directly, call ``invalidate_cache()`` to force
    recomputation on the next access.
    """

    def __init__(
        self,
        parent_analysis: "Analysis",
        path_points: List[Tuple[float, float]],
        *,
        width: int = 40,
    ) -> None:
        """
        Parameters
        ----------
        parent_analysis : Analysis
            The Analysis object that owns this Lane.
        path_points : list of (x, y) tuples
            Ordered centreline coordinates from top to bottom of the lane.
            Minimum 2 points required.  Subpixel (float) coordinates are
            accepted and handled via rounding at sampling time.
        width : int
            Number of pixels to average *across* the lane at each profile
            step (perpendicular to the centreline direction).  Default 40.
        """
        if len(path_points) < 2:
            raise ValueError("A Lane requires at least 2 path_points.")
        if width < 1:
            raise ValueError("width must be ≥ 1.")

        self.parent_analysis:   "Analysis"                 = parent_analysis
        self.path_points:       List[Tuple[float, float]]  = list(path_points)
        self.width:             int                        = width
        self.bands:             List[Band]                 = []

        # Control points for the piecewise background baseline.
        # Each entry is (profile_index, intensity), where profile_index ∈
        # [0, len(get_profile())-1] — NOT an image pixel row coordinate.
        self.background_points: List[Tuple[float, float]] = []

        # Internal caches — invalidated by invalidate_cache()
        self._profile_cache:    Optional[np.ndarray] = None
        self._background_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Discard the cached profile and background arrays."""
        self._profile_cache    = None
        self._background_cache = None

    # ------------------------------------------------------------------
    # Centreline geometry (private helper)
    # ------------------------------------------------------------------

    def _compute_centreline_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ``(xs, ys)`` float arrays of evenly-spaced centreline sample
        points.

        Two-point lane — straight line
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Linear interpolation between the two endpoints; the number of steps
        equals ``ceil(‖Δr‖₂)`` (one step per pixel of Euclidean distance).

        Multi-point lane — cubic B-spline
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        1.  A parametric cubic B-spline is fitted through all ``path_points``
            using ``scipy.interpolate.splprep`` with ``s=0`` (interpolating,
            not smoothing).  The spline degree is capped at ``min(3, n−1)``
            to handle fewer than 4 control points gracefully.
        2.  A dense parameter sweep is used to compute the cumulative arc
            length of the spline.
        3.  The parameter space is reparametrised to *uniform arc-length
            spacing*, giving ≈ 1 profile sample per pixel of arc length.

        Returns
        -------
        xs : np.ndarray of float, shape (N,)
        ys : np.ndarray of float, shape (N,)
        """
        pts   = np.asarray(self.path_points, dtype=float)  # (M, 2)
        x_pts = pts[:, 0]
        y_pts = pts[:, 1]

        if len(self.path_points) == 2:
            # ---- Straight line ----
            dx        = x_pts[1] - x_pts[0]
            dy        = y_pts[1] - y_pts[0]
            num_steps = max(2, int(np.ceil(np.hypot(dx, dy))))
            t         = np.linspace(0.0, 1.0, num_steps)
            xs        = x_pts[0] + t * dx
            ys        = y_pts[0] + t * dy

        else:
            # ---- Parametric B-spline ----
            # Spline degree k must be < number of control points.
            k = min(3, len(self.path_points) - 1)
            tck, _ = splprep([x_pts, y_pts], s=0, k=k)

            # Dense sweep to measure cumulative arc length along the spline.
            n_dense      = max(1000, len(self.path_points) * 200)
            u_dense      = np.linspace(0.0, 1.0, n_dense)
            xd, yd       = splev(u_dense, tck)
            seg_lengths  = np.hypot(np.diff(xd), np.diff(yd))
            arc_cumul    = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            total_arc    = arc_cumul[-1]

            # Reparametrise: find u values for uniformly-spaced arc positions
            # (≈ 1 sample per pixel of arc length).
            num_steps    = max(2, int(np.ceil(total_arc)))
            arc_uniform  = np.linspace(0.0, total_arc, num_steps)
            u_uniform    = np.interp(arc_uniform, arc_cumul, u_dense)
            xs, ys       = splev(u_uniform, tck)

        return xs, ys

    # ------------------------------------------------------------------
    # Profile extraction
    # ------------------------------------------------------------------

    def get_profile(self) -> np.ndarray:
        """
        Build and return the 1D intensity profile for this lane.

        Algorithm
        ---------
        1.  Obtain the processed (grayscale, possibly inverted) image from
            ``Analysis.get_image()``.
        2.  Compute centreline sample coordinates ``(xs, ys)`` via
            ``_compute_centreline_coords()``.
        3.  At each centreline sample ``i``, collect pixel values at
            horizontal offsets ``j ∈ [−width//2, +width//2]``.
            Out-of-bounds accesses are clamped to the image edge.
        4.  Average the ``width+1`` collected values to yield ``profile[i]``.

        The vectorised implementation constructs integer index grids
        ``ys_grid`` (N, 1) and ``xs_grid`` (N, K) which broadcast to (N, K)
        during the single NumPy indexing call — no Python loops required.

        Returns
        -------
        np.ndarray of shape (N,), dtype float64
            N ≈ arc-length of the lane centreline in pixels.
        """
        if self._profile_cache is not None:
            return self._profile_cache

        img    = self.parent_analysis.get_image()   # 2-D uint8 (H, W)
        H, W   = img.shape
        xs, ys = self._compute_centreline_coords()

        # Horizontal offsets spanning the lane width (across-lane direction).
        half_w  = self.width // 2
        offsets = np.arange(-half_w, half_w + 1, dtype=np.int32)  # (K,)

        # Convert centreline coordinates to integer pixel indices.
        xs_int  = np.round(xs).astype(np.int32)  # (N,)
        ys_int  = np.round(ys).astype(np.int32)  # (N,)

        # Build 2-D index grids, clamped to image boundaries.
        # xs_grid : (N, K)  — x-index for each (step, offset) pair
        # ys_grid : (N, 1)  — y-index; broadcasts with xs_grid to (N, K)
        xs_grid = np.clip(xs_int[:, None] + offsets[None, :], 0, W - 1)
        ys_grid = np.clip(ys_int[:, None],                    0, H - 1)

        # Gather pixel values and average across the lane width.
        pixel_vals = img[ys_grid, xs_grid].astype(np.float64)  # (N, K)
        profile    = pixel_vals.mean(axis=1)                    # (N,)

        self._profile_cache = profile
        return profile

    # ------------------------------------------------------------------
    # Background interpolation
    # ------------------------------------------------------------------

    def get_background_profile(self) -> np.ndarray:
        """
        Interpolate ``background_points`` into a full-length baseline array.

        ``background_points`` is a list of ``(profile_index, intensity)``
        control points.  ``profile_index`` must be in
        ``[0, len(get_profile())−1]`` — these are *profile-coordinate*
        positions, **not** image pixel row indices.

        Interpolation strategy
        ----------------------
        0 control points → zero array (no baseline correction).
        1 control point  → flat array at that intensity level.
        ≥2 control points → piecewise linear interpolation via
                            ``scipy.interpolate.interp1d``; the nearest
                            endpoint intensity is held constant outside the
                            defined range (constant extrapolation).

        Returns
        -------
        np.ndarray of shape (N,), dtype float64
        """
        if self._background_cache is not None:
            return self._background_cache

        N = len(self.get_profile())

        if not self.background_points:
            # No baseline defined — treat as zero everywhere.
            bg = np.zeros(N, dtype=np.float64)
            self._background_cache = bg
            return bg

        bp         = np.asarray(self.background_points, dtype=float)  # (M, 2)
        sort_idx   = np.argsort(bp[:, 0])
        positions  = bp[sort_idx, 0]   # profile-coordinate positions
        intensities = bp[sort_idx, 1]

        if len(positions) == 1:
            # Single control point → constant background.
            bg = np.full(N, intensities[0], dtype=np.float64)
        else:
            # Piecewise linear with constant boundary extrapolation.
            interpolator = interp1d(
                positions,
                intensities,
                kind="linear",
                bounds_error=False,
                fill_value=(intensities[0], intensities[-1]),
            )
            bg = interpolator(np.arange(N, dtype=float))

        self._background_cache = bg
        return bg

    # ------------------------------------------------------------------
    # Rf calculation
    # ------------------------------------------------------------------

    def get_peak_rf(self, band: Band) -> float:
        """
        Compute the default relative-front (Rf) value for a band.

        Definition
        ----------
            Rf = peak_index / (lane_length − 1)

        The result is in ``[0.0, 1.0]``, where 0.0 is the top of the lane
        and 1.0 is the bottom.  In SDS-PAGE, high-MW proteins migrate less
        (small Rf); low-MW proteins migrate further (large Rf).

        Note: Rf-to-distance curve calibration (e.g. using front-marker
        positions) will be implemented in a future extension.

        Parameters
        ----------
        band : Band
            Must belong to this lane (``band.parent_lane is self``).

        Returns
        -------
        float in [0.0, 1.0]

        Raises
        ------
        ValueError
            If the band's ``parent_lane`` is not this Lane object.
        """
        if band.parent_lane is not self:
            raise ValueError(
                "get_peak_rf() called with a Band that does not belong to "
                "this Lane."
            )
        lane_length = len(self.get_profile())
        if lane_length <= 1:
            return 0.0
        return float(band.peak_index) / float(lane_length - 1)

    # ------------------------------------------------------------------
    # Automatic band detection
    # ------------------------------------------------------------------

    def auto_detect_bands(
        self,
        bg_window:    int   = 250,
        prominence:   float = 4.0,
        min_distance: int   = 15,
        top_margin:   int   = 150,
        bottom_margin: int  = 100,
    ) -> None:
        """
        Detect bands automatically from the lane's 1D intensity profile and
        populate ``self.bands``.

        Math pipeline
        -------------
        1.  **Gaussian smoothing** (σ = 3 px) — removes high-frequency noise
            from the raw pixel-averaged profile without significantly broadening
            real peaks.
        2.  **Morphological grey opening** (structuring element = ``bg_window``)
            — erosion followed by dilation traces the slow-varying baseline
            *beneath* all narrow bands.  A large ``bg_window`` (≈ the distance
            between wells) accurately tracks sloped or uneven backgrounds.
        3.  **Background subtraction** — ``corrected = clip(smoothed −
            background, 0, ∞)`` — sets the baseline to zero and ensures that
            integration volumes are never negative.
        4.  **Peak finding** in the margin-trimmed corrected profile via
            ``scipy.signal.find_peaks`` with a ``prominence`` threshold and a
            minimum inter-peak ``distance``.

        Temporary plotting attributes
        ------------------------------
        The intermediate arrays are stored as instance attributes so the GUI
        can overlay them on the profile plot without recomputing:

            ``_temp_smoothed``    — Gaussian-smoothed raw profile
            ``_temp_background``  — morphological rolling baseline
            ``_temp_corrected``   — baseline-corrected signal (≥ 0)

        Parameters
        ----------
        bg_window : int
            Structuring-element size for grey opening.  Should be larger than
            the widest expected band.  Default 250.
        prominence : float
            Minimum peak prominence (intensity units above surroundings).
            Default 4.0.
        min_distance : int
            Minimum number of profile pixels between adjacent peaks.
            Default 15.
        top_margin : int
            Profile pixels excluded from the top of the search region
            (skips the loading wells).  Default 150.
        bottom_margin : int
            Profile pixels excluded from the bottom.  Default 100.
        """
        raw_profile  = self.get_profile()
        profile_len  = len(raw_profile)

        # ---- Step 1: Gaussian smoothing ----
        smoothed = scipy.ndimage.gaussian_filter1d(raw_profile, sigma=3)

        # ---- Step 2: Rolling background via morphological grey opening ----
        background = scipy.ndimage.grey_opening(smoothed, size=bg_window)

        # ---- Step 3: Baseline subtraction, no negatives ----
        corrected = np.clip(smoothed - background, 0, None)

        # Store for GUI plotting (not part of the permanent data model)
        self._temp_smoothed    = smoothed
        self._temp_background  = background
        self._temp_corrected   = corrected

        # ---- Step 4: Peak search in margin-trimmed region ----
        search_end = profile_len - bottom_margin
        if top_margin >= search_end:
            # Nothing to search after applying margins
            self.bands.clear()
            return

        search_area = corrected[top_margin:search_end]

        peaks_local, properties = find_peaks(
            search_area,
            prominence=prominence,
            distance=min_distance,
        )

        # Translate local indices → absolute profile-coordinate indices
        peaks_abs   = peaks_local + top_margin
        # find_peaks with prominence= fills 'left_bases' and 'right_bases'
        left_bases  = properties.get("left_bases",  peaks_local) + top_margin
        right_bases = properties.get("right_bases", peaks_local) + top_margin

        # ---- Build Band objects ----
        self.bands.clear()
        for peak_abs, lb, rb in zip(peaks_abs, left_bases, right_bases):
            start = int(lb)
            peak  = int(peak_abs)
            end   = int(rb)

            # Enforce the Band invariant (start < peak < end) by clamping.
            start = min(start, peak - 1)
            end   = max(end,   peak + 1)
            start = max(0,               start)
            end   = min(profile_len - 1, end)

            if start < peak < end:
                self.bands.append(
                    Band(
                        start_index=start,
                        peak_index=peak,
                        end_index=end,
                        parent_lane=self,
                    )
                )

    def __repr__(self) -> str:
        return (
            f"Lane(n_path_points={len(self.path_points)}, "
            f"width={self.width}, n_bands={len(self.bands)})"
        )


# ===========================================================================
# Analysis
# ===========================================================================


class Analysis:
    """
    Top-level container for a single gel electrophoresis analysis session.

    Ownership
    ---------
    ``Analysis`` owns the (immutable) raw image array and the list of
    ``Lane`` objects.  Each ``Lane`` owns its ``Band`` objects.  Back-
    references (``band.parent_lane``, ``lane.parent_analysis``) are set when
    the child objects are constructed.

    Image pipeline
    --------------
    raw image → grayscale conversion → dynamic-range normalisation
    → optional inversion (if ``is_dark_on_light``) → cached uint8 array.

    Calibration workflow
    --------------------
    1.  On calibrator bands, set ``calibrated_volume`` / ``molecular_weight``
        and raise ``is_quant_calibrator`` / ``is_mw_calibrator``.
    2.  Call ``fit_quantity_calibration(fit_type)`` and/or
        ``fit_mw_calibration(fit_type)``.
    3.  Apply the fitted curves via ``analysis.predict_quantity(raw_vol)``
        and ``analysis.predict_mw(rf)``.
    """

    def __init__(
        self,
        raw_image: np.ndarray,
        *,
        is_dark_on_light: bool = True,
        volume_calc_mode: VolumeCalcMode = VolumeCalcMode.ZERO_CLIPPED,
    ) -> None:
        """
        Parameters
        ----------
        raw_image : np.ndarray
            2-D (grayscale) or 3-D (RGB / RGBA) NumPy array of any dtype.
            An immutable (read-only) copy is stored immediately.
        is_dark_on_light : bool
            True  → dark bands on bright background (Coomassie, silver, etc.).
                    ``get_image()`` returns ``255 − grayscale``.
            False → bright bands on dark background (fluorescent dyes, etc.).
                    No inversion is applied.
        volume_calc_mode : VolumeCalcMode
            Default integration mode passed to ``Band.get_raw_volume()``
            during calibration fitting.  Default: ``ZERO_CLIPPED``.
        """
        if not isinstance(raw_image, np.ndarray):
            raise TypeError(
                f"raw_image must be a numpy ndarray; got {type(raw_image).__name__!r}."
            )

        # Store an immutable copy so external code cannot silently alter pixel
        # values after the Analysis object has been created.
        self._raw_image: np.ndarray = raw_image.copy()
        self._raw_image.flags.writeable = False

        self.is_dark_on_light: bool           = is_dark_on_light
        self.lanes:            List[Lane]      = []
        # Three (x, y) anchor points that define the parabolic smile curve.
        # None means no smile correction has been applied yet.
        self.smile_points:     Optional[list]  = None
        self.volume_calc_mode: VolumeCalcMode  = volume_calc_mode

        # Cached processed image (built lazily by get_image()).
        self._image_cache: Optional[np.ndarray] = None

        # ---- Quantity calibration results (set by fit_quantity_calibration) ----
        self._quant_fit_fn:     Optional[Callable]     = None
        self._quant_fit_params: Optional[np.ndarray]   = None
        self._quant_fit_cov:    Optional[np.ndarray]   = None
        self._quant_fit_type:   Optional[CurveFitType] = None

        # ---- MW calibration results (set by fit_mw_calibration) ----
        # Point-to-point semi-log interpolator on (peak_index, ln(MW)) pairs.
        # predict_mw() evaluates it and converts back with exp().
        self._mw_interp = None   # scipy interp1d object, or None if not yet fitted

    # ------------------------------------------------------------------
    # Immutable raw-image access
    # ------------------------------------------------------------------

    @property
    def raw_image(self) -> np.ndarray:
        """The original image array, stored read-only."""
        return self._raw_image

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def get_image(self) -> np.ndarray:
        """
        Return the processed 8-bit grayscale image used for all profile
        calculations.

        Processing pipeline
        -------------------
        **Step 1 — Grayscale conversion** (only if input is 3-D):
            RGB/RGBA → luminance using ITU-R BT.601 weights:
            ``Y = 0.299·R + 0.587·G + 0.114·B``.
            A single-channel 3-D image (shape H×W×1) is squeezed to 2-D.

        **Step 2 — Dynamic-range normalisation to uint8 [0, 255]**:
            * Float image with all values in [0.0, 1.0] → scaled by 255.
            * All other cases (uint16, float >1, int) → linearly rescaled
              so that the minimum maps to 0 and the maximum maps to 255.
            * Constant (degenerate) images → zero array.

        **Step 3 — Inversion** (if ``is_dark_on_light``):
            ``output = 255 − input``

        The result is cached.  Call ``invalidate_image_cache()`` after
        changing ``is_dark_on_light`` (note: ``raw_image`` is immutable).

        Returns
        -------
        np.ndarray of shape (H, W), dtype uint8
        """
        if self._image_cache is not None:
            return self._image_cache

        img = self._raw_image

        # ---- Step 1: Grayscale conversion ----
        if img.ndim == 3:
            nc = img.shape[2]
            if nc >= 3:
                # ITU-R BT.601 luminosity coefficients
                w    = np.array([0.299, 0.587, 0.114], dtype=np.float64)
                gray = img[:, :, :3].astype(np.float64) @ w
            elif nc == 1:
                gray = img[:, :, 0].astype(np.float64)
            else:
                raise ValueError(
                    f"Unsupported number of image channels: {nc}."
                )
        elif img.ndim == 2:
            gray = img.astype(np.float64)
        else:
            raise ValueError(
                f"raw_image must be 2-D or 3-D; got ndim={img.ndim}."
            )

        # ---- Step 2: Normalise to uint8 [0, 255] ----
        gray_min, gray_max = float(gray.min()), float(gray.max())

        if gray_max == gray_min:
            # Degenerate image (all pixels identical).
            gray_u8 = np.zeros_like(gray, dtype=np.uint8)

        elif (
            img.dtype.kind == "f"
            and gray_min >= 0.0
            and gray_max <= 1.0
        ):
            # Normalised float image in [0, 1] — multiply directly.
            gray_u8 = (gray * 255.0).clip(0, 255).round().astype(np.uint8)

        else:
            # General case: linearly rescale the full dynamic range.
            gray_u8 = (
                (gray - gray_min) / (gray_max - gray_min) * 255.0
            ).clip(0, 255).round().astype(np.uint8)

        # ---- Step 3: Invert for dark-on-light staining ----
        if self.is_dark_on_light:
            # np.uint8 subtraction wraps; subtract from the scalar 255 instead.
            gray_u8 = np.full_like(gray_u8, 255) - gray_u8

        self._image_cache = gray_u8
        return gray_u8

    def invalidate_image_cache(self) -> None:
        """
        Discard the cached processed image and propagate invalidation to all
        child Lane caches (which depend on the processed image).

        Call this after changing ``is_dark_on_light``.  (``raw_image`` is
        immutable; create a new ``Analysis`` to switch the source image.)
        """
        self._image_cache = None
        for lane in self.lanes:
            lane.invalidate_cache()

    # ------------------------------------------------------------------
    # Automatic lane detection
    # ------------------------------------------------------------------

    def auto_detect_lanes(
        self,
        num_wells: int = 12,
        lane_width: int = 45,
    ) -> None:
        """
        Detect lane positions automatically using a 95th-percentile
        column-projection algorithm and populate ``self.lanes``.

        Algorithm
        ---------
        1.  Obtain the processed (possibly inverted) grayscale image.
        2.  **Column projection** — ``np.percentile(img, 95, axis=0)`` takes
            the 95th-percentile intensity across all rows for each column.
            Using the 95th percentile (instead of the mean or max) makes the
            projection robust to horizontal band artefacts and dust specks
            while still capturing where lanes are brightest.
        3.  **Gaussian smoothing** (σ = 4 px) suppresses sub-pixel noise so
            that ``argmax`` reliably finds the ladder bands.
        4.  The **left ladder** centre is the column with the highest smoothed
            value in the first 20 % of the image width.
        5.  The **right ladder** centre is the column with the highest smoothed
            value in the last 20 % of the image width.
        6.  ``np.linspace(left_ladder, right_ladder, num_wells)`` spaces
            ``num_wells`` lane centres evenly between the two ladders.
        7.  ``self.lanes`` is rebuilt from scratch: each lane is a straight
            vertical :class:`Lane` spanning the full image height.

        Parameters
        ----------
        num_wells : int
            Total number of lanes to detect (including the two ladder lanes).
            Default 12.
        lane_width : int
            Width in pixels assigned to each :class:`Lane` for profile
            integration.  Default 45.
        """
        img = self.get_image()                        # (H, W) uint8
        img_height, img_width = img.shape

        # ---- Step 2: 95th-percentile column projection ----
        # axis=0 → collapse rows, keeping one value per column
        projection = np.percentile(img, 95, axis=0).astype(np.float64)

        # ---- Step 3: Gaussian smoothing ----
        smoothed = scipy.ndimage.gaussian_filter1d(projection, sigma=4)

        # ---- Step 4: Left ladder — peak in the first 20 % of columns ----
        left_boundary = max(1, int(img_width * 0.20))
        left_ladder   = int(np.argmax(smoothed[:left_boundary]))

        # ---- Step 5: Right ladder — peak in the last 20 % of columns ----
        right_start  = int(img_width * 0.80)
        right_ladder = right_start + int(np.argmax(smoothed[right_start:]))

        # ---- Step 6: Evenly-spaced lane centres ----
        centers = np.linspace(left_ladder, right_ladder, num_wells)

        # ---- Step 7: Rebuild lanes list ----
        self.lanes.clear()
        for x in centers:
            lane = Lane(
                parent_analysis=self,
                # Straight vertical path from top to bottom of the image.
                path_points=[(float(x), 0.0), (float(x), float(img_height))],
                width=lane_width,
            )
            self.lanes.append(lane)

    # ------------------------------------------------------------------
    # Parabolic smile correction
    # ------------------------------------------------------------------

    def get_flattened_y(self, lane_x: float, raw_y: float) -> float:
        """
        Return the smile-corrected row coordinate for a band.

        The correction is based on a parabola fitted through three anchor
        points (``self.smile_points``) that a user has positioned to follow
        the gel's smile distortion.  All migration distances are expressed
        relative to Lane 1 (the reference lane), so the ladder is always at
        its "true" position and every other lane's bands are shifted
        accordingly.

        Algorithm
        ---------
        1.  Fit ``y = a·x² + b·x + c`` through ``self.smile_points``.
        2.  Evaluate the parabola at the Lane-1 x-coordinate to obtain
            ``ref_curve_y``.
        3.  Evaluate the parabola at ``lane_x`` to obtain ``lane_curve_y``.
        4.  ``offset = lane_curve_y − ref_curve_y``  (positive → lane migrated
            further than the reference; negative → less far).
        5.  Return ``raw_y − offset``, i.e. the row position the band *would*
            have had if the gel had run perfectly flat.

        Fallback
        --------
        Returns ``raw_y`` unchanged when:
        * ``self.smile_points`` is ``None``,
        * fewer than 3 anchor points are stored, or
        * ``self.lanes`` is empty (no reference lane available).

        Parameters
        ----------
        lane_x : float
            Column-pixel coordinate of the lane whose band is being corrected.
        raw_y : float
            Uncorrected row-pixel coordinate (``band.peak_index``).

        Returns
        -------
        float
            Smile-corrected row coordinate.
        """
        if self.smile_points is None or len(self.smile_points) < 3:
            return raw_y
        if not self.lanes:
            return raw_y

        X = [p[0] for p in self.smile_points]
        Y = [p[1] for p in self.smile_points]
        a, b, c = np.polyfit(X, Y, 2)

        ref_x        = self.lanes[0].path_points[0][0]
        ref_curve_y  = a * ref_x ** 2  + b * ref_x  + c
        lane_curve_y = a * lane_x ** 2 + b * lane_x + c
        offset       = lane_curve_y - ref_curve_y

        return raw_y - offset

    # ------------------------------------------------------------------
    # Quantity calibration
    # ------------------------------------------------------------------

    def fit_quantity_calibration(self, fit_type: CurveFitType) -> None:
        """
        Fit a calibration curve mapping raw band volume → calibrated quantity.

        Data collection
        ---------------
        All bands across all lanes where:
            * ``band.is_quant_calibrator`` is ``True``, **and**
            * ``band.calibrated_volume`` is not ``None``

        are collected as calibration points:
            ``x = band.get_raw_volume(self.volume_calc_mode)``
            ``y = band.calibrated_volume``

        Model fitting
        -------------
        ``scipy.optimize.curve_fit`` fits the model function selected by
        ``fit_type`` to the ``(x, y)`` pairs.  The resulting callable is
        stored as ``_quant_fit_fn`` and accessible via the
        ``predict_quantity`` property.

        Parameters
        ----------
        fit_type : CurveFitType
            Mathematical model.

        Raises
        ------
        ValueError
            Fewer than 2 calibrator bands available.
        RuntimeError
            ``scipy.optimize.curve_fit`` failed to converge.
        """
        cal_bands: List[Band] = [
            band
            for lane in self.lanes
            for band in lane.bands
            if band.is_quant_calibrator and band.calibrated_volume is not None
        ]

        if len(cal_bands) < 2:
            raise ValueError(
                f"fit_quantity_calibration requires ≥ 2 calibrator bands; "
                f"found {len(cal_bands)}.  Set is_quant_calibrator=True and "
                f"assign calibrated_volume on at least 2 Band objects."
            )

        x = np.array(
            [b.get_raw_volume(self.volume_calc_mode) for b in cal_bands],
            dtype=np.float64,
        )
        y = np.array(
            [b.calibrated_volume for b in cal_bands],
            dtype=np.float64,
        )

        model_fn = _FIT_FUNCTIONS[fit_type]

        try:
            popt, pcov = curve_fit(model_fn, x, y, maxfev=10_000)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Quantity calibration did not converge for {fit_type!r}: {exc}"
            ) from exc

        # Bind the optimised parameters into a closure so predict_quantity
        # behaves like a plain callable with a single argument.
        self._quant_fit_fn     = lambda v, _p=popt: model_fn(np.asarray(v, dtype=float), *_p)
        self._quant_fit_params = popt
        self._quant_fit_cov    = pcov
        self._quant_fit_type   = fit_type

    @property
    def predict_quantity(self) -> Callable:
        """
        The fitted quantity calibration function.

        Maps a raw volume (scalar or array) to calibrated quantity values.

        Raises
        ------
        RuntimeError
            If ``fit_quantity_calibration()`` has not been called yet.
        """
        if self._quant_fit_fn is None:
            raise RuntimeError(
                "No quantity calibration fitted yet.  "
                "Call fit_quantity_calibration() first."
            )
        return self._quant_fit_fn

    def get_quantity_calibration_info(self) -> Dict:
        """
        Return a diagnostics summary of the fitted quantity calibration.

        Returns
        -------
        dict
            ``fit_type``   : CurveFitType used.
            ``params``     : Optimised model parameters (``popt``).
            ``covariance`` : Covariance matrix of ``popt`` (``pcov``).
            ``std_errors`` : ``sqrt(diag(pcov))`` — 1-σ parameter uncertainties.

        Raises
        ------
        RuntimeError
            If no quantity calibration has been fitted yet.
        """
        if self._quant_fit_params is None:
            raise RuntimeError("No quantity calibration fitted yet.")
        return {
            "fit_type":   self._quant_fit_type,
            "params":     self._quant_fit_params,
            "covariance": self._quant_fit_cov,
            "std_errors": (
                np.sqrt(np.diag(self._quant_fit_cov))
                if self._quant_fit_cov is not None else None
            ),
        }

    # ------------------------------------------------------------------
    # Molecular-weight calibration
    # ------------------------------------------------------------------

    def fit_mw_calibration(self) -> None:
        """
        Fit a point-to-point semi-log interpolation through the ladder bands.

        A global linear regression cannot accurately capture the slight
        non-linearity of real gel migration across the full MW range.  This
        method instead builds a piecewise-linear interpolant directly on the
        ``(flattened_y, ln(MW))`` pairs of every calibrator band, giving an
        exact fit through every ladder point and well-behaved extrapolation
        beyond the calibrated range.

        The x-axis uses **smile-corrected** row positions obtained from
        ``get_flattened_y()``, so any parabolic smile distortion is
        automatically incorporated without requiring a pre-computation step.

        Data collected from all bands across all lanes where:

            * ``band.is_mw_calibrator is True``
            * ``band.molecular_weight is not None``

        The bands are sorted by ascending flattened_y before fitting
        (``interp1d`` requires a monotonically increasing x-axis).

        ``fill_value="extrapolate"`` extends the linear segments at both ends
        so that bands migrating slightly above or below the ladder range still
        receive a meaningful MW estimate.

        Raises
        ------
        ValueError
            Fewer than 2 calibrator bands are available.
        ValueError
            Any calibrator band carries a non-positive MW value
            (``ln`` would be undefined).
        """
        calibrators: List[Band] = [
            band
            for lane in self.lanes
            for band in lane.bands
            if getattr(band, "is_mw_calibrator", False)
            and getattr(band, "molecular_weight", None) is not None
        ]

        if len(calibrators) < 2:
            raise ValueError(
                f"fit_mw_calibration requires ≥ 2 calibrator bands; "
                f"found {len(calibrators)}.  "
                f"Set is_mw_calibrator=True + molecular_weight on ≥ 2 bands."
            )

        # Sort by smile-corrected position so interp1d gets a monotone x-axis.
        calibrators.sort(
            key=lambda b: self.get_flattened_y(
                b.parent_lane.path_points[0][0], b.peak_index
            )
        )

        x_data = np.array(
            [
                self.get_flattened_y(
                    b.parent_lane.path_points[0][0], b.peak_index
                )
                for b in calibrators
            ],
            dtype=np.float64,
        )
        mw_vals = np.array(
            [band.molecular_weight for band in calibrators], dtype=np.float64
        )

        if np.any(mw_vals <= 0):
            raise ValueError(
                "All molecular_weight values must be positive (> 0 kDa) "
                "for semi-log interpolation."
            )

        y_data = np.log(mw_vals)   # work in ln(MW) space

        self._mw_interp = interp1d(
            x_data, y_data,
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )

    def predict_mw(self, flattened_y: float) -> float:
        """
        Predict the molecular weight (kDa) for a band at the given position.

        Evaluates the point-to-point semi-log interpolant fitted by
        ``fit_mw_calibration()``:

            ln(MW) = interp(flattened_y)   →   MW = exp(ln(MW))

        Parameters
        ----------
        flattened_y : float
            Smile-corrected row coordinate of the band, as returned by
            ``get_flattened_y(lane_x, band.peak_index)``.

        Returns
        -------
        float or None
            Predicted molecular weight in the same units as the calibrators
            (typically kDa), or ``None`` if no calibration has been fitted.
        """
        if self._mw_interp is None:
            return None
        try:
            return float(np.exp(self._mw_interp(float(flattened_y))))
        except ValueError:
            return None

    def get_mw_calibration_info(self) -> Dict:
        """
        Return a diagnostics summary of the fitted MW calibration.

        Returns
        -------
        dict
            ``model``      : Human-readable description of the method.
            ``x_points``   : Sorted calibrator ``peak_index`` values used.
            ``ln_mw_points``: Corresponding ``ln(MW)`` values.

        Raises
        ------
        RuntimeError
            If no MW calibration has been fitted yet.
        """
        if self._mw_interp is None:
            raise RuntimeError("No MW calibration fitted yet.")
        return {
            "model":             "Point-to-point semi-log: ln(MW) = interp(flattened_y)",
            "flattened_y_points": self._mw_interp.x.tolist(),
            "ln_mw_points":       self._mw_interp.y.tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"Analysis(image_shape={self._raw_image.shape}, "
            f"is_dark_on_light={self.is_dark_on_light}, "
            f"n_lanes={len(self.lanes)}, "
            f"volume_calc_mode={self.volume_calc_mode.name})"
        )
