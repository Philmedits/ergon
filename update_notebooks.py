import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 📄 Universal Core Log & Strip Log Section Cropper\n",
                "### Works on **Google Colab**, **Local Jupyter Notebook**, and **Antigravity IDE**\n",
                "\n",
                "This notebook dynamically extracts **ONLY** the upper data section:\n",
                "- **Top Title & Header** (Operator logo, Core Log, Well ID, Date, Scale)\n",
                "- **Left Metadata** (Country, Field, Licence, Coordinates, Depths, Geologists)\n",
                "- **Right Core Tables** (Cores described, intervals, recovery %, formations)\n",
                "\n",
                "**Completely excludes:**\n",
                "- The `SYMBOL INDEX` banner and lithology legend boxes\n",
                "- Downhole graphical log curves (Gamma Ray, sonic, resistivity)\n",
                "\n",
                "**Dual-Engine Architecture:**\n",
                "- **Engine A (Ladder Periodicity):** Detects open/frameless tables (0 vertical lines) using regular row spacing (~18-25 px) even with faded ink or overlapping text.\n",
                "- **Engine B (Grid Box Contours):** Detects standard boxed tables with vertical dividers."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Install & Verify Dependencies\n",
                "Run this cell if running on Google Colab or fresh environment."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install PyMuPDF (OpenCV, Pillow, NumPy are standard)\n",
                "!pip install -q pymupdf opencv-python pillow numpy"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Imports and Environment Setup"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import glob\n",
                "from itertools import groupby\n",
                "from operator import itemgetter\n",
                "import cv2\n",
                "import numpy as np\n",
                "from PIL import Image\n",
                "import pymupdf\n",
                "from IPython.display import display\n",
                "\n",
                "print(f'[*] Python:   {sys.executable}')\n",
                "print(f'[*] PyMuPDF:  {pymupdf.__version__}')\n",
                "print(f'[*] OpenCV:   {cv2.__version__}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. The Universal Cutoff Detection Engine"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def detect_table_cutoff(img_bgr: np.ndarray) -> int:\n",
                "    '''\n",
                "    Detects the bottom Y-coordinate of the core summary table.\n",
                "    Supports both open tables (0 vertical lines) and closed boxed tables.\n",
                "    '''\n",
                "    height, width = img_bgr.shape[:2]\n",
                "    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)\n",
                "    _, bw = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)\n",
                "\n",
                "    # Focus on Upper-Right Table Region\n",
                "    roi_top = int(height * 0.15)\n",
                "    roi_bottom = int(height * 0.75)\n",
                "    roi_left = int(width * 0.35)\n",
                "    roi_right = int(width * 0.95)\n",
                "    roi_bw = bw[roi_top:roi_bottom, roi_left:roi_right]\n",
                "\n",
                "    # Engine A: Ladder Periodicity Engine (Open Tables)\n",
                "    ladder_end_y = None\n",
                "    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))\n",
                "    h_lines = cv2.morphologyEx(roi_bw, cv2.MORPH_OPEN, h_kernel)\n",
                "    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))\n",
                "    h_lines = cv2.morphologyEx(h_lines, cv2.MORPH_CLOSE, heal_kernel)\n",
                "\n",
                "    row_sums = np.sum(h_lines, axis=1)\n",
                "    peaks = np.where(row_sums > 35 * 255)[0]\n",
                "    line_ys = []\n",
                "    for _, g in groupby(enumerate(peaks), lambda ix: ix[0] - ix[1]):\n",
                "        line_ys.append(int(np.mean(list(map(itemgetter(1), g)))))\n",
                "\n",
                "    if len(line_ys) >= 4:\n",
                "        ladders = []\n",
                "        curr_ladder = [line_ys[0]]\n",
                "        for i in range(1, len(line_ys)):\n",
                "            diff = line_ys[i] - line_ys[i - 1]\n",
                "            if 14 <= diff <= 55:\n",
                "                curr_ladder.append(line_ys[i])\n",
                "            else:\n",
                "                if len(curr_ladder) >= 4:\n",
                "                    ladders.append(curr_ladder)\n",
                "                curr_ladder = [line_ys[i]]\n",
                "        if len(curr_ladder) >= 4:\n",
                "            ladders.append(curr_ladder)\n",
                "        if ladders:\n",
                "            longest = max(ladders, key=len)\n",
                "            ladder_end_y = roi_top + longest[-1]\n",
                "\n",
                "    # Engine B: Grid Box Contour Engine (Closed Tables)\n",
                "    box_end_y = None\n",
                "    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))\n",
                "    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)\n",
                "    h_full = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))\n",
                "    table_grid = h_full + v_lines\n",
                "\n",
                "    contours, _ = cv2.findContours(table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)\n",
                "    for contour in contours:\n",
                "        x, y, box_w, box_h = cv2.boundingRect(contour)\n",
                "        if (x > width * 0.35 and y < height * 0.50 and box_w > 150 and box_h > 100\n",
                "            and box_w < width * 0.80 and box_h < height * 0.80):\n",
                "            bottom = y + box_h\n",
                "            if box_end_y is None or bottom > box_end_y:\n",
                "                box_end_y = bottom\n",
                "\n",
                "    detected_cuts = [y for y in [ladder_end_y, box_end_y] if y is not None]\n",
                "    if detected_cuts:\n",
                "        return max(detected_cuts) + 8\n",
                "    return int(height * 0.50)\n",
                "\n",
                "def crop_core_section(pdf_path: str, output_path: str = None, dpi: int = 150) -> Image.Image:\n",
                "    doc = pymupdf.open(pdf_path)\n",
                "    page = doc[0]\n",
                "    pix = page.get_pixmap(dpi=dpi)\n",
                "    w, h = pix.width, pix.height\n",
                "    img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, pix.n)[:, :, :3]\n",
                "    pil_img = Image.frombytes('RGB', [w, h], pix.samples)\n",
                "    doc.close()\n",
                "\n",
                "    cut_y = detect_table_cutoff(img_bgr)\n",
                "    cropped = pil_img.crop((0, 0, w, cut_y))\n",
                "    if output_path:\n",
                "        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)\n",
                "        cropped.save(output_path, quality=95)\n",
                "    return cropped\n",
                "\n",
                "print('[+] Slicing Engine compiled successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Visual Inspector Tool"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def inspect_cut(pdf_path: str, dpi: int = 150):\n",
                "    doc = pymupdf.open(pdf_path)\n",
                "    pix = doc[0].get_pixmap(dpi=dpi)\n",
                "    img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]\n",
                "    doc.close()\n",
                "    cut_y = detect_table_cutoff(img_bgr)\n",
                "    cropped = crop_core_section(pdf_path, dpi=dpi)\n",
                "    preview = img_bgr.copy()\n",
                "    cv2.line(preview, (0, cut_y), (pix.width, cut_y), (0, 0, 255), 3)\n",
                "    cv2.putText(preview, f'CUT LINE (Y = {cut_y} px)', (30, cut_y - 12),\n",
                "                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)\n",
                "    display_w = 800\n",
                "    scale = display_w / pix.width\n",
                "    preview_small = cv2.resize(preview, (display_w, int(pix.height * scale)))\n",
                "    crop_small = cropped.resize((display_w, int(cropped.size[1] * scale)))\n",
                "    print(f'\\n{\"=\"*75}')\n",
                "    print(f'File: {os.path.basename(pdf_path)}')\n",
                "    print(f'Original: {pix.width}x{pix.height} px | Cropped: {cropped.size[0]}x{cropped.size[1]} px | Cut Y: {cut_y} px')\n",
                "    print(f'{\"=\"*75}')\n",
                "    print('\\n--- [1] Full Page with Cut Line (Red) ---')\n",
                "    display(Image.fromarray(cv2.cvtColor(preview_small, cv2.COLOR_BGR2RGB)))\n",
                "    print('\\n--- [2] Cropped Upper Section ---')\n",
                "    display(crop_small)\n",
                "print('[+] Inspector ready!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Test Case 1: Uploaded Open Table Document (`synth_open_table_full.pdf`)\n",
                "Features 0 vertical lines, 14 row lines, header, left metadata, and SYMBOL INDEX below."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pdf1 = 'ergon-data-study/generated_test_documents/01_normal_straight/synth_open_table_full.pdf'\n",
                "if not os.path.exists(pdf1): pdf1 = 'generated_test_documents/01_normal_straight/synth_open_table_full.pdf'\n",
                "inspect_cut(pdf1)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Test Case 2: Deep 25-Row Table (`synth_normal_11.pdf`)\n",
                "Extended 25-row table (1519 px height) with SYMBOL INDEX pushed down."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pdf2 = 'ergon-data-study/generated_test_documents/01_normal_straight/synth_normal_11.pdf'\n",
                "if not os.path.exists(pdf2): pdf2 = 'generated_test_documents/01_normal_straight/synth_normal_11.pdf'\n",
                "inspect_cut(pdf2)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Test Case 3: Standard Boxed Table (`synth_normal_01.pdf`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pdf3 = 'ergon-data-study/generated_test_documents/01_normal_straight/synth_normal_01.pdf'\n",
                "if not os.path.exists(pdf3): pdf3 = 'generated_test_documents/01_normal_straight/synth_normal_01.pdf'\n",
                "inspect_cut(pdf3)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Batch Run Across All Documents in `01_normal_straight/`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "folder = 'ergon-data-study/generated_test_documents/01_normal_straight'\n",
                "if not os.path.exists(folder):\n",
                "    folder = 'generated_test_documents/01_normal_straight'\n",
                "\n",
                "output_dir = 'ergon-data-study/crops_01_normal_straight'\n",
                "os.makedirs(output_dir, exist_ok=True)\n",
                "all_pdfs = sorted(glob.glob(os.path.join(folder, '*.pdf')))\n",
                "\n",
                "print(f'[*] Processing all {len(all_pdfs)} documents in {folder}...')\n",
                "for p in all_pdfs:\n",
                "    name = os.path.splitext(os.path.basename(p))[0]\n",
                "    out_img = os.path.join(output_dir, f'{name}_cropped.png')\n",
                "    c = crop_core_section(p, output_path=out_img)\n",
                "    print(f'  -> Cropped {name}: {c.size[0]}x{c.size[1]} px')\n",
                "print(f'[+] Batch complete! Saved all crops to: {output_dir}')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

for path in ["section_cropper_pipeline.ipynb", "ergon-data-study/section_cropper_pipeline.ipynb"]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"[+] Successfully wrote {path}")
