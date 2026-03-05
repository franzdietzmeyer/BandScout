# BandScout

**An open-source, cross-platform 1D gel electrophoresis densitometry and molecular weight analysis tool.**

BandScout features automated lane and band detection, interactive 1D profiling, biologically accurate point-to-point semi-log molecular weight calibration, and parabolic "smile" correction.

---

## Key Features

- **Automated Computer Vision** — Instantly detect lanes and bands with customizable sensitivity and background subtraction.
- **Interactive 1D Profiling** — Side-by-side 2D gel and 1D densitometry graphs with bi-directional highlighting.
- **Manual Curation** — Seamlessly add, delete, and drag band boundaries to manually adjust Area Under the Curve (AUC) calculations.
- **Pro-Level Math** — True point-to-point semi-logarithmic molecular weight (MW) interpolation.
- **Physics Correction** — Draggable 3-point "Rubber Band" parabolic smile correction to fix heat-distorted gels.
- **Modern Dark UI** — "Observatory" dark mode theme designed to reduce eye strain in the lab.
- **One-Click Export** — Instantly export all raw volumes, relative percentages, and calculated MWs to CSV.

---

## Installation

BandScout is built on Python, PyQt6, and OpenCV. It runs natively on Windows, macOS, and Linux.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/BandScout.git
cd BandScout
```

### 2. Create a virtual environment

**Windows:**
```bat
python -m venv venv
```

**macOS / Linux:**
```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**Windows:**
```bat
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

---

## Step-by-Step Usage Guide

### Step 1: Image Setup

1. Launch the app and click **"Load Gel Image"** in the left sidebar.
2. Select your gel image (`.jpg`, `.png`, `.tif`).
3. Select whether your image has **light bands on a dark background** (fluorescent) or **dark bands on a light background** (Coomassie/colorimetric). The app will automatically invert the image if necessary for the math engine.

### Step 2: Detect Lanes

1. Open the **"2. Detect Lanes"** drawer in the sidebar.
2. Enter the **Number of Wells** your gel has.
3. Click **"Detect Lanes"**. The software will draw red bounding boxes over the columns.
4. **Manual Override:** If the algorithm misses a skewed lane, check the **"Manual Add Lane"** toggle (or use the top `Edit` menu) and click directly on the 2D image to place a new lane.

### Step 3: Detect Bands & Manual Curation

1. Open the **"3. Detect Bands"** drawer. Adjust the **Prominence** (sensitivity) and **Background Window** if needed, then click **"Detect Bands"**.
2. **View the 1D Profile:** Click on any lane in the 2D image. Its 1D densitometry graph will open on the right, and the data table at the bottom will populate with AUC and Relative % calculations.
3. **Refine Boundaries:** On the 1D graph, click and drag the red dashed lines at the base of each green peak to manually fix integration boundaries. The data table recalculates instantly.
4. **Add / Delete Bands:** Check the **"Manual Add Band"** toggle and click directly on a faint band in the 2D image or on the 1D graph to add it. To delete, select a row in the table and press `Delete` or `Backspace`.

### Step 4: Molecular Weight Calibration

1. Click on the lane containing your protein ladder (e.g., Lane 1).
2. Open the **"4. Calibration"** drawer and click **"Set Selected Lane as Ladder"**.
3. In the dialog, either type the known sizes (in kDa) manually or select a preset (e.g., *SeeBlue Invitrogen*) from the dropdown to auto-fill values.
4. Click **OK**. The software generates a point-to-point semi-log standard curve and calculates the MW for every unknown band on the gel.

### Step 5: Smile Correction *(Optional)*

If your gel ran hot and bands "smile" (curve downward in the middle), raw Y-pixel measurements will produce inaccurate MW calculations.

1. Check the **"Smile Correction"** toggle in the sidebar or top menu.
2. A yellow dashed line with three red control points will appear across the image.
3. Find a single distinct band that spans the entire gel width.
4. Place the **Left** and **Right** dots on the edges of that band. Drag the **Middle** dot down to trace the curve of the smile.
5. The software mathematically flattens the gel in the background and recalculates all molecular weights.

### Step 6: Export Data

Click **"Export to CSV"** at the bottom left. The app generates a clean, formatted spreadsheet containing:

| Column | Description |
|---|---|
| Lane Number | Lane index |
| Band Number | Band index within the lane |
| Pixel Position | Y-pixel centroid of the band |
| Raw Volume | Absolute AUC value |
| Relative % | Band volume as a percentage of the total lane |
| Calculated MW | Interpolated molecular weight (kDa) |

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*© 2026 Franz Dietzmeyer — [franz.dietzmeyer@medizin.uni-leipzig.de](mailto:franz.dietzmeyer@medizin.uni-leipzig.de)*
