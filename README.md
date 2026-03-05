Gemini hat gesagt
-----------------

Here is the complete, ready-to-copy text for your GitHub repository. Just click the "Copy code" button in the top right corner of the block below and paste it directly into your `README.md` file!

Markdown

```
# 🧬 GelAnalyzer Pro

**An open-source, cross-platform 1D gel electrophoresis densitometry and molecular weight analysis tool.**

GelAnalyzer Pro replaces expensive, proprietary lab software with a sleek, modern Python desktop application. It features automated lane and band detection, interactive 1D profiling, biologically accurate point-to-point semi-log molecular weight calibration, and parabolic "smile" correction.

## ✨ Key Features
* **Automated Computer Vision:** Instantly detect lanes and bands with customizable sensitivity and background subtraction.
* **Interactive 1D Profiling:** Side-by-side 2D gel and 1D densitometry graphs with bi-directional highlighting.
* **Manual Curation:** Seamlessly add, delete, and drag band boundaries to manually adjust Area Under the Curve (AUC) calculations.
* **Pro-Level Math:** True Point-to-Point Semi-Logarithmic Molecular Weight (MW) interpolation.
* **Physics Correction:** Draggable 3-point "Rubber Band" parabolic smile correction to fix heat-distorted gels.
* **Modern Dark UI:** "Observatory" dark mode theme designed to reduce eye strain in the lab.
* **One-Click Export:** Instantly export all raw volumes, relative percentages, and calculated MWs to CSV.

---

## 🛠 Installation

GelAnalyzer Pro is built on Python, PyQt6, and OpenCV. It runs natively on Windows, macOS, and Linux.

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/GelAnalyzerPro.git](https://github.com/YOUR_USERNAME/GelAnalyzerPro.git)
cd GelAnalyzerPro

```

### 2\. Create a virtual environment

-   **Windows:**

    DOS

    ```
    python -m venv venv

    ```

-   **macOS/Linux:**

    Bash

    ```
    python3 -m venv venv

    ```

### 3\. Activate the virtual environment

-   **Windows:**

    DOS

    ```
    venv\Scripts\activate

    ```

-   **macOS/Linux:**

    Bash

    ```
    source venv/bin/activate

    ```

### 4\. Install dependencies

Bash

```
pip install -r requirements.txt

```

### 5\. Run the application

Bash

```
python app.py

```

* * * * *

📖 Step-by-Step Usage Guide
---------------------------

### Step 1: Image Setup

1.  Launch the app and click **"Load Gel Image"** in the left sidebar.

2.  Select your gel image (`.jpg`, `.png`, `.tif`).

3.  Select whether your image has **light bands on a dark background** (fluorescent) or **dark bands on a light background** (Coomassie/colorimetric). The app will automatically invert the image if necessary for the math engine.

### Step 2: Detect Lanes

1.  Open the **"2. Detect Lanes"** drawer in the sidebar.

2.  Enter the **Number of Wells** your gel has.

3.  Click **"Detect Lanes"**. The software will draw red bounding boxes over the columns.

4.  **Manual Override:** If the algorithm misses a skewed lane, check the **"Manual Add Lane"** toggle (or use the top `Edit` menu) and click directly on the 2D image to drop a new lane perfectly into place.

### Step 3: Detect Bands & Manual Curation

1.  Open the **"3. Detect Bands"** drawer. Adjust the Prominence (sensitivity) and Background Window if needed, then click **"Detect Bands"**.

2.  **View the 1D Profile:** Click on any lane inside the 2D image. Its 1D densitometry graph will instantly open on the right, and the data table at the bottom will populate with Area Under Curve (AUC) and Relative % calculations.

3.  **Refine Boundaries:** On the 1D graph, click and drag the red dashed lines hugging the base of the green peaks to manually fix integration boundaries. The data table will recalculate instantly.

4.  **Add/Delete Bands:** Missed a faint band? Check the **"Manual Add Band"** toggle and click directly on the faint band in the 2D image (or on the 1D graph). To delete, select a row in the table and press the `Delete` or `Backspace` key.

### Step 4: Molecular Weight Calibration

1.  Click on the lane in the 2D image that contains your protein ladder (e.g., Lane 1).

2.  Open the **"4. Calibration"** drawer and click **"Set Selected Lane as Ladder"**.

3.  A dialog box will appear. You can either manually type in the known sizes (in kDa) or select a preset (like *SeeBlue Invitrogen*) from the dropdown menu to auto-fill the values.

4.  Click **OK**. The software will instantly generate a point-to-point semi-log standard curve and calculate the MW for every unknown band on your gel.

### Step 5: Smile Correction (Optional)

If your gel ran hot and the bands "smile" (curve downwards in the middle), absolute Y-pixel measurements will result in inaccurate MW calculations.

1.  Check the **"Smile Correction"** toggle in the sidebar (or top menu).

2.  A yellow dashed line with three red dots will appear across your image.

3.  Find a single, distinct band that spans the entire gel.

4.  Place the Left and Right red dots on the edges of that band. Drag the Middle red dot down to perfectly trace the curve of the "smile".

5.  The software will instantly flatten the gel mathematically in the background and recalculate all Molecular Weights for perfect accuracy.

### Step 6: Export Data

When you are satisfied with your analysis, click the **"Export to CSV"** button at the bottom left. The app will generate a clean, formatted spreadsheet containing Lane Numbers, Band Numbers, Pixel Positions, Raw Volumes, Relative Percentages, and Calculated MWs ready for publication or further statistical analysis.

* * * * *

🤝 Contributing
---------------

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

📄 License
----------

This project is licensed under the MIT License - see the LICENSE file for details.
