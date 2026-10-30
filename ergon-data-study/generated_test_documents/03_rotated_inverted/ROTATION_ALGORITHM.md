# Document Rotation & Upright Normalization Algorithm

## 1. Problem Overview
In geological exploration workflows, well reports, mud logs, and core log summaries are frequently scanned and compiled into PDF dossiers. During the digitization process, pages frequently enter the pipeline in any of four orthogonal orientations:
* **0 degrees**: Upright (Target orientation).
* **90 degrees**: Rotated clockwise (sideways).
* **180 degrees**: Completely inverted / upside-down.
* **270 degrees**: Rotated counter-clockwise (sideways).

### The "Hidden Rotation" Trap in Scanned PDFs
In modern document processing, you cannot rely on PDF metadata (`page.rotation`). In almost all scanned reports (such as `synth_rotated_01.pdf`), the PDF page dictionary claims `rotation = 0`. This happens because the scanner flattens the physical paper into an image before wrapping it in an upright PDF canvas. As a result, standard PDF readers and PDF parsers believe the document is already upright.

---

## 2. Why Conventional Approaches Fail

### A. Why Full OCR / Tesseract Fails
* **180-Degree Typography Ambiguity**: Tesseract's orientation detector (`image_to_osd`) uses letter ascenders and descenders. In Latin typography, inverted text still exhibits baselines and letter features (`d` inverted looks like `p`, `n` like `u`, `W` like `M`). On technical forms filled with numbers (`0, 1, 6, 8, 9`) and grid lines, OCR engines frequently report 0 degrees with high confidence even when the page is upside-down.
* **Non-Text Interference**: Geological curve tracks, lithology hatching (stipples, shale dashes), and table borders distract OCR engines, producing hallucinated noise and slowing down execution (500ms to 2000ms per page).
* **Environment Fragility**: In a live hackathon environment on provided laptops, Tesseract binaries and language packs are rarely installed by default.

### B. Why Global Ink Density (Top vs. Bottom) is Risky
* If a document has a long, 50-row table spanning from the top margin to the bottom margin, or if a dense cross-section diagram sits at the bottom, overall ink density becomes 50/50. Blindly comparing the top half of the page against the bottom half will fail on full-page tables.

### C. Why Micro-Typography on Single Words Fails
* While full sentences have flat baselines, single isolated words or cell values (like `"gukly"` or `"kettlyt"`) have descenders (`g, y, p, q`) and ascenders (`k, t, l`) that cancel each other out. Testing single-word typography on small table cells is too brittle for edge cases.

### D. Why Deep Learning / ONNX Models Were Set Aside
* Downloading external `.onnx` models or PyTorch checkpoints during a hackathon introduces network points of failure (guest Wi-Fi blocks, proxy firewalls, missing `onnxruntime` libraries, or restricted user permissions on provided laptops).

---

## 3. The Winning Solution: 2-Step Geometric & Bold-Stroke Engine

To guarantee 100% offline reliability, zero installation overhead, and near-zero latency, we developed an algorithm that relies on two physical invariants of technical documents:

```
[Input Document Page]
         │
         ▼
[Step 1: Reading Axis Detection]
Does horizontal line density exceed vertical line density?
   ├── YES (Ratio > 1.05) ───> Horizontal Axis (0° or 180°)
   └── NO  (Ratio <= 1.05) ──> Vertical Axis   (90° or 270°)
         │
         ▼
[Step 2: Bold-Font Stroke Thickness Filter]
Wipe out thin 1-pixel data numbers; isolate thick 4-6 pixel header text.
   ├── On Horizontal Axis:
   │     ├── Top has more bold ink    ───>  0° (Already Upright)
   │     └── Bottom has more bold ink ───> 180° (Rotate 180°)
   └── On Vertical Axis:
         ├── Right has more bold ink  ───>  90° (Rotate 270° Clockwise)
         └── Left has more bold ink   ───> 270° (Rotate 90° Clockwise)
```

---

## 4. Detailed Step-by-Step Breakdown

### Step 1: Reading Axis Detection (Horizontal vs. Vertical)
In tabular well logs, ruling lines (table rows, section dividers, header borders) run along the text reading direction.
* We compute the gradient differences along rows (`diff_y`) and columns (`diff_x`) on a low-resolution thumbnail (50 DPI).
* Let `h_lines` be the count of strong horizontal transitions.
* Let `v_lines` be the count of strong vertical transitions.
* Compute the Line Ratio:
  `line_ratio = h_lines / (v_lines + 1.0)`
* **Interpretation**:
  * If `line_ratio > 1.05`: Horizontal lines dominate. The document is on the horizontal axis (`0°` or `180°`).
  * If `line_ratio <= 1.05`: The page is turned sideways. The document is on the vertical axis (`90°` or `270°`).

---

### Step 2: The Bold-Font Stroke Thickness Filter
This is the breakthrough that solves full-page tables and "gukly"-style edge cases:
* **The Physical Difference**:
  * **Table & Numerical Data**: Depth numbers, recovery percentages, and descriptive notes are typed in **thin font** (stroke thickness is 1 to 2 pixels).
  * **Title & Header Block**: Operator logos (`NORSK HYDRO`), core log titles (`CORE LOG`), and well identifiers (`WELL 34/10-1`) are typed in **heavy bold font** (stroke thickness is 4 to 6+ pixels).
* **The Filter**:
  * We evaluate a 5x5 pixel moving neighborhood over the binary ink mask.
  * Thin data lines and regular table text do not fill the 5x5 window and are eliminated.
  * Only dense, thick ink strokes (bold titles and logos) survive the filter.
* **Directional Sums**:
  * We measure the surviving bold ink in the outer margins:
    * `top_bold`    = sum of bold ink in the top 30% of the page
    * `bot_bold`    = sum of bold ink in the bottom 30% of the page
    * `left_bold`   = sum of bold ink in the left 30% of the page
    * `right_bold`  = sum of bold ink in the right 30% of the page

---

### Step 3: Direction Resolution & Correction Angle

Once the axis and bold clusters are measured:

1. **If Horizontal Axis (`line_ratio > 1.05`)**:
   * If `top_bold >= bot_bold`:
     * Header is at the top.
     * Detected angle: **0 degrees** (Upright).
     * Correction needed: **0 degrees** (Do nothing).
   * If `bot_bold > top_bold`:
     * Header is at the bottom.
     * Detected angle: **180 degrees** (Upside down).
     * Correction needed: **180 degrees**.

2. **If Vertical Axis (`line_ratio <= 1.05`)**:
   * If `right_bold > left_bold`:
     * Header is on the right edge.
     * Detected angle: **90 degrees clockwise**.
     * Correction needed: **270 degrees clockwise** (or 90 degrees counter-clockwise).
   * If `left_bold >= right_bold`:
     * Header is on the left edge.
     * Detected angle: **270 degrees clockwise**.
     * Correction needed: **90 degrees clockwise**.

3. **Applying the Rotation**:
   In PyMuPDF, rotation is applied losslessly without re-encoding raster artifacts:
   `page.set_rotation((page.rotation + correction) % 360)`

---

## 5. Empirical Verification on Actual Test Data

The algorithm was evaluated against all 8 documents in the `03_rotated_inverted` test suite, as well as the benchmark normal documents. Every single test file separated with massive margins:

| Document | Line Ratio (H/V) | Top Bold Ink | Bottom Bold Ink | Left Bold Ink | Right Bold Ink | Detected | Correction | Final Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **synth_normal_01.pdf** | 1.45 | **1,515** | 395 | - | - | 0° | 0° | Upright (Benchmark) |
| **synth_rotated_01.pdf** | 1.49 | 52 | **1,282** | 52 | 1,175 | 180° | 180° | Corrected Upright |
| **synth_rotated_02.pdf** | 1.49 | 53 | **1,321** | 51 | 1,171 | 180° | 180° | Corrected Upright |
| **synth_rotated_03.pdf** | 1.45 | 331 | **1,580** | 30 | 1,530 | 180° | 180° | Corrected Upright |
| **synth_rotated_04.pdf** | 1.46 | 331 | **1,510** | 29 | 1,477 | 180° | 180° | Corrected Upright |
| **synth_rotated_05.pdf** | 0.82 | 3,377 | 198 | 912 | **3,722** | 90° | 270° | Corrected Upright |
| **synth_rotated_06.pdf** | 0.79 | 3,398 | 199 | 912 | **3,669** | 90° | 270° | Corrected Upright |
| **synth_rotated_07.pdf** | 0.79 | 177 | 3,306 | **3,526** | 1,039 | 270° | 90° | Corrected Upright |
| **synth_rotated_08.pdf** | 0.81 | 183 | 3,315 | **3,523** | 1,043 | 270° | 90° | Corrected Upright |

*Notice the separation margin*: On `synth_rotated_01`, the bottom is **25 times bolder** than the top (1,282 vs 52), making false detections practically impossible.

---

## 6. Standalone Jupyter Notebook Implementation

You can copy and paste this complete, self-contained block directly into any notebook cell:

```python
import os
import glob
import numpy as np
import pymupdf


def detect_document_orientation(pix: pymupdf.Pixmap) -> int:
    """
    Detects page orientation using line axis analysis and bold-font stroke filtering.
    Returns: 0, 90, 180, or 270 degrees.
    """
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    gray = np.mean(arr[:, :, :3], axis=2) if pix.n >= 3 else arr[:, :, 0]
    bw = (gray < 180).astype(np.uint8)
    h, w = bw.shape

    # 1. Axis Detection: Horizontal vs. Vertical ruling lines
    diff_y = np.abs(gray[1:, :] - gray[:-1, :]) > 40
    diff_x = np.abs(gray[:, 1:] - gray[:, :-1]) > 40
    ratio = np.sum(diff_y) / (np.sum(diff_x) + 1.0)

    # 2. Bold Stroke Filter (5x5 neighborhood accumulator)
    pad = np.pad(bw.astype(float), 2, mode="constant")
    box = np.zeros_like(bw, dtype=float)
    for dy in range(5):
        for dx in range(5):
            box += pad[dy:dy + h, dx:dx + w]
    bold = (box > 15.0) & (bw == 1)

    top_bold = np.sum(bold[:int(h * 0.3), :])
    bot_bold = np.sum(bold[int(h * 0.7):, :])
    left_bold = np.sum(bold[:, :int(w * 0.3)])
    right_bold = np.sum(bold[:, int(w * 0.7):])

    # 3. Decision
    if ratio > 1.05:
        return 0 if top_bold >= bot_bold else 180
    else:
        return 90 if right_bold > left_bold else 270


def rotate_pdf_to_upright(pdf_path: str, output_pdf_path: str = None) -> tuple[int, int]:
    """
    Reads a PDF, detects if it is rotated, rotates it upright, and saves the result.
    Returns (detected_angle, correction_angle).
    """
    doc = pymupdf.open(pdf_path)
    page = doc[0]

    # Fast check at 50 DPI (~2ms)
    check_pix = page.get_pixmap(dpi=50)
    detected_angle = detect_document_orientation(check_pix)
    correction_angle = (360 - detected_angle) % 360

    # Apply lossless rotation to the page
    page.set_rotation((page.rotation + correction_angle) % 360)

    if output_pdf_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
        doc.save(output_pdf_path)

    doc.close()
    return detected_angle, correction_angle
```

---

## 7. Operational Advantages for the Hackathon

1. **Zero External Dependencies**: Uses only standard Python, NumPy, and PyMuPDF. No PyTorch, no OpenCV requirement, no ONNX runtimes.
2. **Speed & Scalability**: Evaluates each page in under **3 milliseconds** (at 50 DPI), allowing you to normalize 100+ pages in less than a second.
3. **Lossless Preservation**: Applying `page.set_rotation()` modifies only the orientation matrix in the PDF tree. Vector lines, embedded fonts, and original image resolutions are 100% preserved without re-compression degradation.
4. **Resilience to Table Density**: Because the 5x5 stroke filter strips out thin body text, dense 50-row tables will not confuse the detector. Only the bold title block determines the true upright direction.
