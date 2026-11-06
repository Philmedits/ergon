# %% [markdown]
# # 📄 Universal Core Log & Strip Log Section Cropper
# ### Interactive Python Notebook (Works in VS Code / Antigravity IDE)
#
# This script extracts ONLY the upper data section:
#   1. Top Title & Header (Operator logo, Core Log title, Well ID, Date, Scale)
#   2. Left Metadata (Country, Field, Licence, Coordinates, Depths, Geologists)
#   3. Right Core Summary Table (Cores described, intervals, recovery %, formations)
#
# Completely excludes:
#   - The "SYMBOL INDEX" banner and lithology legend boxes
#   - Downhole Graphical Log curves (Gamma Ray, sonic, resistivity)
#
# Dual-Engine Detection:
#   - Engine A (Ladder Periodicity): Detects open/frameless tables (0 vertical lines)
#     using rhythmic parallel row spacing (~18-25 px) even with faded ink or overlapping text.
#   - Engine B (Grid Box Contours): Detects standard boxed tables with vertical dividers.

# %% [markdown]
# ### 1. Imports and Environment Setup
# Run this cell first to initialize the libraries.

# %%
import os
import sys
from itertools import groupby
from operator import itemgetter
import cv2
import numpy as np
from PIL import Image
import pymupdf
from IPython.display import display

print(f"[*] Python executable: {sys.executable}")
print(f"[*] PyMuPDF version:   {pymupdf.__version__}")
print(f"[*] OpenCV version:    {cv2.__version__}")

# %% [markdown]
# ### 2. The Universal Cutoff Detection Engine

# %%
def detect_table_cutoff(img_bgr: np.ndarray) -> int:
    """
    Finds where the core summary table ends on the page.
    Returns the Y-pixel coordinate to slice right below the table,
    excluding the SYMBOL INDEX banner and log curves below.

    Supports both:
      1. Open / Frameless tables (horizontal lines only, 0 vertical dividers)
      2. Closed Grid Tables (with vertical dividers and borders)
    """
    height, width = img_bgr.shape[:2]

    # Convert to black-and-white
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # Focus search on Upper-Right Table Region
    roi_top = int(height * 0.15)
    roi_bottom = int(height * 0.75)
    roi_left = int(width * 0.35)
    roi_right = int(width * 0.95)
    roi_bw = bw[roi_top:roi_bottom, roi_left:roi_right]

    # -------------------------------------------------------------------------
    # Engine A: Ladder Periodicity Engine (Open / Frameless Tables)
    # -------------------------------------------------------------------------
    ladder_end_y = None

    # Filter out text: keep only horizontal lines >= 25 px long
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    h_lines = cv2.morphologyEx(roi_bw, cv2.MORPH_OPEN, h_kernel)

    # Heal small gaps or faded line segments
    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    h_lines = cv2.morphologyEx(h_lines, cv2.MORPH_CLOSE, heal_kernel)

    # Project horizontally (count line pixels per row)
    row_sums = np.sum(h_lines, axis=1)
    peaks = np.where(row_sums > 35 * 255)[0]

    # Group adjacent rows that belong to the same physical line
    line_ys = []
    for _, g in groupby(enumerate(peaks), lambda ix: ix[0] - ix[1]):
        line_ys.append(int(np.mean(list(map(itemgetter(1), g)))))

    # Find the rhythmic ladder pattern (stacks of parallel rows spaced ~18-25 px)
    if len(line_ys) >= 4:
        ladders = []
        curr_ladder = [line_ys[0]]
        for i in range(1, len(line_ys)):
            diff = line_ys[i] - line_ys[i - 1]
            if 14 <= diff <= 55:  # standard row height or 1 skipped/faded line
                curr_ladder.append(line_ys[i])
            else:
                if len(curr_ladder) >= 4:
                    ladders.append(curr_ladder)
                curr_ladder = [line_ys[i]]
        if len(curr_ladder) >= 4:
            ladders.append(curr_ladder)

        if ladders:
            longest = max(ladders, key=len)
            ladder_end_y = roi_top + longest[-1]

    # -------------------------------------------------------------------------
    # Engine B: Grid Box Contour Engine (Closed / Boxed Tables)
    # -------------------------------------------------------------------------
    box_end_y = None
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    h_full = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    table_grid = h_full + v_lines

    contours, _ = cv2.findContours(table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)

        is_on_right_half  = x > (width * 0.35)
        is_in_upper_half  = y < (height * 0.50)
        is_table_sized    = (box_w > 150) and (box_h > 100)
        is_not_outer_page = (box_w < width * 0.80) and (box_h < height * 0.80)

        if is_on_right_half and is_in_upper_half and is_table_sized and is_not_outer_page:
            bottom = y + box_h
            if box_end_y is None or bottom > box_end_y:
                box_end_y = bottom

    # -------------------------------------------------------------------------
    # Combine Cutoff Signals (+ 8 px margin)
    # -------------------------------------------------------------------------
    detected_cuts = [y for y in [ladder_end_y, box_end_y] if y is not None]
    if detected_cuts:
        return max(detected_cuts) + 8

    return int(height * 0.50)


def crop_core_section(
    pdf_path: str,
    output_path: str = None,
    dpi: int = 150,
    tight_box: bool = False,
) -> Image.Image:
    """
    Crops the upper data section of a core log PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Cannot find PDF: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    w, h = pix.width, pix.height
    img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, pix.n)[:, :, :3]
    pil_img = Image.frombytes("RGB", [w, h], pix.samples)
    doc.close()

    cut_y = detect_table_cutoff(img_bgr)

    if tight_box:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1)))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100)))
        grid = cv2.add(h_lines, v_lines)
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            fx, fy, fw, fh = cv2.boundingRect(contours[0])
            crop_box = (max(0, fx - 10), max(0, fy - 10), min(w, fx + fw + 10), cut_y)
        else:
            crop_box = (0, 0, w, cut_y)
    else:
        crop_box = (0, 0, w, cut_y)

    cropped = pil_img.crop(crop_box)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cropped.save(output_path, quality=95)

    return cropped

print("[+] Universal Slicing Engine compiled successfully!")

# %% [markdown]
# ### 3. Visual Inspector Tool

# %%
def inspect_cut(pdf_path: str, dpi: int = 150):
    doc = pymupdf.open(pdf_path)
    pix = doc[0].get_pixmap(dpi=dpi)
    img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    doc.close()

    cut_y = detect_table_cutoff(img_bgr)
    cropped = crop_core_section(pdf_path, dpi=dpi)

    preview = img_bgr.copy()
    cv2.line(preview, (0, cut_y), (pix.width, cut_y), (0, 0, 255), 3)
    cv2.putText(preview, f"CUT LINE (Y = {cut_y} px)", (30, cut_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    display_w = 800
    scale = display_w / pix.width
    preview_small = cv2.resize(preview, (display_w, int(pix.height * scale)))
    crop_small = cropped.resize((display_w, int(cropped.size[1] * scale)))

    print(f"\n{'='*75}")
    print(f"File: {os.path.basename(pdf_path)}")
    print(f"Original Size: {pix.width}x{pix.height} px | Cropped Size: {cropped.size[0]}x{cropped.size[1]} px | Cut Y: {cut_y} px")
    print(f"{'='*75}")

    print("\n--- [1] Full Page with Cut Line (Red) ---")
    display(Image.fromarray(cv2.cvtColor(preview_small, cv2.COLOR_BGR2RGB)))

    print("\n--- [2] Cropped Upper Section (Preserving Header & Table, Excluding Symbol Index) ---")
    display(crop_small)

print("[+] Visual inspector ready!")

# %% [markdown]
# ### 4. Test Case 1: Uploaded Open Table Document (`synth_open_table_full.pdf`)
# Open table with 0 vertical dividers, 14 rows, full header, left metadata, and SYMBOL INDEX below.

# %%
open_table_pdf = "ergon-data-study/generated_test_documents/01_normal_straight/synth_open_table_full.pdf"
if not os.path.exists(open_table_pdf):
    open_table_pdf = "generated_test_documents/01_normal_straight/synth_open_table_full.pdf"

if os.path.exists(open_table_pdf):
    inspect_cut(open_table_pdf)
else:
    print(f"[!] File not found: {open_table_pdf}")

# %% [markdown]
# ### 5. Test Case 2: Deep 25-Row Table (`synth_normal_11.pdf`)
# Long 25-row table extending deep down the page (1519 px height).

# %%
norm11_pdf = "ergon-data-study/generated_test_documents/01_normal_straight/synth_normal_11.pdf"
if not os.path.exists(norm11_pdf):
    norm11_pdf = "generated_test_documents/01_normal_straight/synth_normal_11.pdf"

if os.path.exists(norm11_pdf):
    inspect_cut(norm11_pdf)
else:
    print(f"[!] File not found: {norm11_pdf}")

# %% [markdown]
# ### 6. Test Case 3: Standard Boxed Table (`synth_normal_01.pdf`)

# %%
norm01_pdf = "ergon-data-study/generated_test_documents/01_normal_straight/synth_normal_01.pdf"
if not os.path.exists(norm01_pdf):
    norm01_pdf = "generated_test_documents/01_normal_straight/synth_normal_01.pdf"

if os.path.exists(norm01_pdf):
    inspect_cut(norm01_pdf)
else:
    print(f"[!] File not found: {norm01_pdf}")

# %% [markdown]
# ### 7. Batch Process All PDFs in `01_normal_straight`
# Runs the cropper across all test documents and saves crops to `crops_01_normal_straight/`.

# %%
import glob

folder = "ergon-data-study/generated_test_documents/01_normal_straight"
if not os.path.exists(folder):
    folder = "generated_test_documents/01_normal_straight"

output_dir = "ergon-data-study/crops_01_normal_straight"
os.makedirs(output_dir, exist_ok=True)
all_pdfs = sorted(glob.glob(os.path.join(folder, "*.pdf")))

print(f"\n[*] Processing all {len(all_pdfs)} documents in {folder}...")
for p in all_pdfs:
    name = os.path.splitext(os.path.basename(p))[0]
    out_img = os.path.join(output_dir, f"{name}_cropped.png")
    c = crop_core_section(p, output_path=out_img)
    print(f"  -> Cropped {name}: {c.size[0]}x{c.size[1]} px")
print(f"[+] All {len(all_pdfs)} crops saved to: {output_dir}")

