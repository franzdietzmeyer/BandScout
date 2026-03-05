# BandScout

**An open-source, cross-platform 1D gel electrophoresis densitometry and molecular weight analysis tool.**

---

## Features

- **Automatic Lane & Band Detection** — Intelligent detection of lanes and bands from gel images with adjustable sensitivity controls.
- **Interactive 1D Intensity Profiling** — Real-time densitometry profiles per lane with peak picking and baseline correction.
- **Semi-Log Molecular Weight Calibration** — Build MW standard curves from ladder lanes and automatically interpolate unknown band sizes.
- **Parabolic Smile Correction** — Correct for gel distortion artifacts with a built-in parabolic warping algorithm.
- **CSV Export** — Export band positions, relative intensities, and estimated molecular weights to CSV for downstream analysis.
- **Dark Mode UI** — A polished, researcher-friendly dark-themed interface built with PyQt6.

---

## Requirements

- Python 3.9 or higher
- pip

---

## Installation

Follow these steps to clone the repository and run the application on **Windows**, **macOS**, or **Linux**.

### 1. Clone the Repository

```bash
git clone https://github.com/franzdietzmeyer/BandScout.git
cd BandScout
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

---

## Usage

1. **Load a gel image** using `File → Open Image` (supports JPEG, PNG, TIFF).
2. **Detect lanes** automatically or draw them manually by clicking and dragging.
3. **Run band detection** to identify peaks in each lane's intensity profile.
4. **Assign a ladder lane** and enter known molecular weights to generate the MW calibration curve.
5. **Apply smile correction** if horizontal band bowing is present.
6. **Export results** via `File → Export CSV`.

---

---

## Project Structure

```
BandScout/
├── app.py              # Main application entry point (PyQt6 UI)
├── core_engine.py      # Image processing and analysis backend
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).
