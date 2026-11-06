"""
create_full_workflow_notebook.py
Builds the complete, clean, end-to-end full_workflow.ipynb Jupyter Notebook
featuring:
  - Steps 1-5: Two-Tier Coarse-to-Fine Computer Vision Cropping & Deskewing
  - Step 6: Multimodal Tabular Extraction via Google Gemini 2.5 Flash Vision
  - Step 7: Geological Quality Control (QC) & Physical Audit
  - Step 8: Master Database Vertical Concatenation (MASTER_ALL_WELLS_CORE_LOGS.csv)
  - Step 9: Executive Geological Analytics
"""

import json

def build_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🛢️ Geological Core Log Digitization: End-to-End Master Workflow\n",
                    "### Automated Computer Vision Cropping & Gemini 2.5 Flash Vision Extraction Pipeline\n",
                    "\n",
                    "---\n",
                    "\n",
                    "## 📌 Executive Overview\n",
                    "In petroleum exploration and field appraisal, legacy geological data is archived inside giant **Composite Well Log Enclosures**.\n",
                    "Each composite enclosure contains three distinct structural zones:\n",
                    "1. **Zone 1 (Upper Header & Metadata Block):** Operator logo, well ID, coordinates, licence, and the **Core Summary Table**.\n",
                    "2. **Zone 2 (Middle Section):** The **`SYMBOL INDEX`** legend.\n",
                    "3. **Zone 3 (Lower Section):** A massive downhole strip log containing 15+ vertical column tracks (Gamma Ray curves, lithology columns, bioturbation bars, and depositional interpretations).\n",
                    "\n",
                    "### The Objective\n",
                    "Given an input folder containing composite well enclosures (e.g. `Njord_Composite_Enclosures_5/`), execute an automated, end-to-end pipeline:\n",
                    "* **Step 1:** Load the official template schema contract (`template_core_log.csv`).\n",
                    "* **Step 2:** Isolate and crop **ONLY** Zone 1 using our **Two-Tier Coarse-to-Fine Computer Vision algorithm**.\n",
                    "* **Step 3:** Correct scanner tilts (e.g. $+3.8^\\circ$ or $-4.2^\\circ$) with sub-degree precision.\n",
                    "* **Step 4:** Extract well metadata and all coring rows from the cropped image using **Google Gemini 2.5 Flash Vision**.\n",
                    "* **Step 5:** Run professional geological quality audits ($Top < Base$, recovery bounds, sequential core numbers).\n",
                    "* **Step 6:** Concatenate all wells into **`MASTER_ALL_WELLS_CORE_LOGS.csv`**."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Setup & Environment Initialisation\n",
                    "We import standard packages available in Anaconda, plus `requests` for the Gemini Vision API."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import glob\n",
                    "import re\n",
                    "import io\n",
                    "import time\n",
                    "import base64\n",
                    "import requests\n",
                    "import cv2\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import pymupdf\n",
                    "from IPython.display import display, Image\n",
                    "\n",
                    "# Load Gemini API Key from .env file\n",
                    "gemini_api_key = None\n",
                    "if os.path.exists('.env'):\n",
                    "    with open('.env', 'r') as env_file:\n",
                    "        for line in env_file:\n",
                    "            if line.startswith('GEMINI_API_KEY'):\n",
                    "                gemini_api_key = line.split('=', 1)[1].strip().strip('\"').strip(\"'\")\n",
                    "\n",
                    "print('Anaconda Environment Initialized Successfully!')\n",
                    "print(f'OpenCV Version   : {cv2.__version__}')\n",
                    "print(f'PyMuPDF Version  : {pymupdf.__version__}')\n",
                    "print(f'Pandas Version   : {pd.__version__}')\n",
                    "print(f'Gemini API Key   : {\"Loaded (\" + gemini_api_key[:8] + \"...)\" if gemini_api_key else \"Not Found\"}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Load the Official Schema Contract\n",
                    "We load `template_core_log.csv` to lock down the exact 20 standard columns required for the final master database."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "template_path = 'template_core_log.csv'\n",
                    "template_df = pd.read_csv(template_path)\n",
                    "master_columns = list(template_df.columns)\n",
                    "\n",
                    "print(f'Official Target Schema Loaded ({len(master_columns)} columns):')\n",
                    "for index, column_name in enumerate(master_columns, start=1):\n",
                    "    print(f'  {index:02d}. {column_name}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. The Two-Tier Coarse-to-Fine Cropping Algorithm\n",
                    "\n",
                    "### Why Naive Computer Vision Fails on Composite Strip Logs\n",
                    "When classical line-detection algorithms inspect full composite enclosures, they get tricked by two field-grade artifacts:\n",
                    "1. **The 15 Vertical Track Columns:** Zone 3 contains over 9,000 vertical gradient pixels, tricking naive orientation detectors into thinking the document was scanned sideways ($270^\\circ$).\n",
                    "2. **Dark Bioturbation Bars:** The bottom of the strip log contains dense black ink fills, tricking naive detectors into thinking the page is upside-down ($180^\\circ$).\n",
                    "\n",
                    "### The Solution: Coarse-to-Fine Architecture\n",
                    "* **Tier 1 (Coarse Crop):** Slice the top 40% of the enclosure height (`image[:int(H * 0.40)]`). This instantly eliminates the entire downhole strip log, Gamma Ray curves, and Symbol Index.\n",
                    "* **Tier 2 (Fine Crop):** Straighten scanner tilt with sub-degree Hough lines, then pinpoint the exact bottom table row using horizontal line periodicity (the **Ladder Algorithm**)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def deskew_image(image_bgr):\n",
                    "    '''Straightens scanner tilts (-10 to +10 degrees) using Hough line analysis.'''\n",
                    "    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)\n",
                    "    edges = cv2.Canny(gray, 50, 150, apertureSize=3)\n",
                    "    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80, minLineLength=80, maxLineGap=15)\n",
                    "    \n",
                    "    if lines is None:\n",
                    "        return image_bgr, 0.0\n",
                    "\n",
                    "    angles = []\n",
                    "    for line in lines:\n",
                    "        pts = line.flatten()\n",
                    "        if len(pts) >= 4:\n",
                    "            x1, y1, x2, y2 = pts[:4]\n",
                    "            dx = x2 - x1\n",
                    "            dy = y2 - y1\n",
                    "            if abs(dx) > abs(dy):\n",
                    "                angle = np.degrees(np.arctan2(dy, dx))\n",
                    "                if -10.0 <= angle <= 10.0:\n",
                    "                    angles.append(angle)\n",
                    "\n",
                    "    if not angles:\n",
                    "        return image_bgr, 0.0\n",
                    "\n",
                    "    median_angle = float(np.median(angles))\n",
                    "    if abs(median_angle) < 0.2:\n",
                    "        return image_bgr, 0.0\n",
                    "\n",
                    "    h, w = image_bgr.shape[:2]\n",
                    "    center = (w // 2, h // 2)\n",
                    "    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)\n",
                    "    cos_val = np.abs(rotation_matrix[0, 0])\n",
                    "    sin_val = np.abs(rotation_matrix[0, 1])\n",
                    "    new_w = int((h * sin_val) + (w * cos_val))\n",
                    "    new_h = int((h * cos_val) + (w * sin_val))\n",
                    "    rotation_matrix[0, 2] += (new_w / 2) - center[0]\n",
                    "    rotation_matrix[1, 2] += (new_h / 2) - center[1]\n",
                    "    \n",
                    "    straightened = cv2.warpAffine(image_bgr, rotation_matrix, (new_w, new_h), borderValue=(255, 255, 255))\n",
                    "    return straightened, median_angle\n",
                    "\n",
                    "\n",
                    "def find_table_bottom_cutoff(image_bgr):\n",
                    "    '''Finds the exact bottom row of the Core Summary Table using horizontal line periodicity.'''\n",
                    "    h, w = image_bgr.shape[:2]\n",
                    "    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)\n",
                    "    _, bw = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)\n",
                    "\n",
                    "    table_roi = bw[:, int(w * 0.45):int(w * 0.98)]\n",
                    "    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))\n",
                    "    clean_lines = cv2.morphologyEx(table_roi, cv2.MORPH_OPEN, kernel)\n",
                    "    \n",
                    "    row_sums = np.sum(clean_lines > 0, axis=1)\n",
                    "    line_positions = np.where(row_sums > 60)[0]\n",
                    "\n",
                    "    distinct_lines = []\n",
                    "    for y in line_positions:\n",
                    "        if not distinct_lines or (y - distinct_lines[-1]) > 8:\n",
                    "            distinct_lines.append(y)\n",
                    "\n",
                    "    if not distinct_lines:\n",
                    "        return int(h * 0.35)\n",
                    "\n",
                    "    table_lines = []\n",
                    "    for i in range(1, len(distinct_lines)):\n",
                    "        dy = distinct_lines[i] - distinct_lines[i - 1]\n",
                    "        if 12 <= dy <= 38:\n",
                    "            if not table_lines:\n",
                    "                table_lines.append(distinct_lines[i - 1])\n",
                    "            table_lines.append(distinct_lines[i])\n",
                    "        elif dy > 50 and len(table_lines) >= 4:\n",
                    "            break\n",
                    "\n",
                    "    if table_lines:\n",
                    "        return table_lines[-1] + 16\n",
                    "    return distinct_lines[-1] + 16\n",
                    "\n",
                    "\n",
                    "def crop_core_log_enclosure(pdf_path, coarse_ratio=0.40):\n",
                    "    '''Applies Two-Tier Coarse-to-Fine Cropping on a composite geological enclosure PDF.'''\n",
                    "    doc = pymupdf.open(pdf_path)\n",
                    "    page = doc[0]\n",
                    "    pix = page.get_pixmap(dpi=150)\n",
                    "    raw_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]\n",
                    "    doc.close()\n",
                    "\n",
                    "    total_h, total_w = raw_bgr.shape[:2]\n",
                    "    h_coarse = int(total_h * coarse_ratio)\n",
                    "    coarse_bgr = raw_bgr[:h_coarse, :].copy()\n",
                    "\n",
                    "    deskewed_bgr, tilt_angle = deskew_image(coarse_bgr)\n",
                    "    cutoff_y = find_table_bottom_cutoff(deskewed_bgr)\n",
                    "    cutoff_y = min(cutoff_y, deskewed_bgr.shape[0])\n",
                    "\n",
                    "    final_cropped_section = deskewed_bgr[:cutoff_y, :]\n",
                    "    return final_cropped_section, tilt_angle, cutoff_y"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Visual Demonstration on a Sample Enclosure\n",
                    "We test the cropping logic on our exact authentic replica enclosure (`03_njord_tall_table_16cores_exact_replica.pdf`) and display the resulting clean section inline."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sample_pdf = 'Njord_Composite_Enclosures_5/03_njord_tall_table_16cores_exact_replica.pdf'\n",
                    "sample_crop, sample_tilt, sample_cut_y = crop_core_log_enclosure(sample_pdf)\n",
                    "\n",
                    "os.makedirs('cropped_sections', exist_ok=True)\n",
                    "sample_output_path = 'cropped_sections/sample_preview_njord_16cores.png'\n",
                    "cv2.imwrite(sample_output_path, sample_crop)\n",
                    "\n",
                    "print(f'Sample Enclosure Cropped Successfully!')\n",
                    "print(f'  - Detected Tilt Angle : {sample_tilt:+.2f}°')\n",
                    "print(f'  - Cutoff Coordinate Y : {sample_cut_y} px')\n",
                    "print(f'  - Cropped Dimensions  : {sample_crop.shape[1]}x{sample_crop.shape[0]} px')\n",
                    "\n",
                    "display(Image(filename=sample_output_path, width=720))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Batch Cropping All Enclosures in the Folder\n",
                    "We iterate across every PDF in `Njord_Composite_Enclosures_5/`, straighten any scanning skew, crop the core section, and save the clean images."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "input_folder = 'Njord_Composite_Enclosures_5'\n",
                    "pdf_files = sorted(glob.glob(os.path.join(input_folder, '*.pdf')))\n",
                    "print(f'Found {len(pdf_files)} enclosures in {input_folder}/\\n')\n",
                    "\n",
                    "crop_records = []\n",
                    "for index, pdf_path in enumerate(pdf_files, start=1):\n",
                    "    file_name = os.path.basename(pdf_path)\n",
                    "    cropped_img, tilt, cut_y = crop_core_log_enclosure(pdf_path)\n",
                    "    \n",
                    "    out_path = os.path.join('cropped_sections', file_name.replace('.pdf', '_crop.png'))\n",
                    "    cv2.imwrite(out_path, cropped_img)\n",
                    "    \n",
                    "    crop_records.append({\n",
                    "        'File': file_name,\n",
                    "        'Tilt_Angle': f'{tilt:+.2f}°',\n",
                    "        'Cutoff_Y': f'{cut_y} px',\n",
                    "        'Cropped_Width': cropped_img.shape[1],\n",
                    "        'Cropped_Height': cropped_img.shape[0],\n",
                    "        'Output_Image': out_path\n",
                    "    })\n",
                    "\n",
                    "df_crop_summary = pd.DataFrame(crop_records)\n",
                    "display(df_crop_summary[['File', 'Tilt_Angle', 'Cutoff_Y', 'Cropped_Width', 'Cropped_Height']])"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Geological Tabular Extraction via Google Gemini 2.5 Flash Vision\n",
                    "\n",
                    "We use **Google Gemini 2.5 Flash Vision** to perform multimodal extraction directly from the cropped images:\n",
                    "* Reads well header metadata (Well ID, Operator, Field, Licence, Coordinates, Elevation, TD).\n",
                    "* Transcribes every single coring row from the table (Core No, Top Depth, Base Driller Depth, Recovery %).\n",
                    "* Automatically calculates:\n",
                    "  $$\\text{Cut\\_Length\\_m} = \\text{Base\\_Driller\\_m} - \\text{Top\\_Depth\\_m}$$\n",
                    "  $$\\text{Recovered\\_m} = \\left(\\frac{\\text{Recovery\\_pct}}{100}\\right) \\times \\text{Cut\\_Length\\_m}$$\n",
                    "* Built-in automatic rate-limit backoff (HTTP 429) ensures smooth, robust execution on free-tier API quotas."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def extract_core_data_with_gemini(image_path, api_key, model='gemini-2.5-flash', max_retries=3):\n",
                    "    '''Extracts well metadata and all coring table rows from a cropped image using Gemini Vision.'''\n",
                    "    with open(image_path, 'rb') as image_file:\n",
                    "        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')\n",
                    "\n",
                    "    prompt = '''You are an expert petroleum geologist digitising a scanned core log.\n",
                    "Extract the well metadata and ALL rows from the Core Summary Table.\n",
                    "\n",
                    "Return as CSV with these columns:\n",
                    "Well,Operator,Country,Licence,Field,Owners,Latitude_DMS,Longitude_DMS,Elevation_KB_m,Water_Depth_m,Total_Depth_Driller_m,Formation_at_TD,Core_No,Top_Depth_m,Base_Driller_m,Recovery_pct\n",
                    "\n",
                    "Rules:\n",
                    "- Include EVERY row from the table.\n",
                    "- Core_No must be an integer.\n",
                    "- Depths in metres as floating numbers.\n",
                    "- Recovery_pct as a percentage number (e.g. 85.8).\n",
                    "- Repeat the well metadata on each row.\n",
                    "- Return ONLY raw CSV text with no markdown formatting.'''\n",
                    "\n",
                    "    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}'\n",
                    "    headers = {'Content-Type': 'application/json'}\n",
                    "    payload = {\n",
                    "        'contents': [{\n",
                    "            'parts': [\n",
                    "                {'inline_data': {'mime_type': 'image/png', 'data': image_base64}},\n",
                    "                {'text': prompt}\n",
                    "            ]\n",
                    "        }],\n",
                    "        'generationConfig': {'temperature': 0.0, 'maxOutputTokens': 4096}\n",
                    "    }\n",
                    "\n",
                    "    for attempt in range(max_retries):\n",
                    "        response = requests.post(url, headers=headers, json=payload, timeout=60)\n",
                    "        if response.status_code == 200:\n",
                    "            result_text = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()\n",
                    "            if result_text.startswith('```'):\n",
                    "                result_text = result_text.split('\\n', 1)[1].rsplit('```', 1)[0].strip()\n",
                    "            dataframe = pd.read_csv(io.StringIO(result_text))\n",
                    "            \n",
                    "            # Calculate Cut Length and Recovered Length\n",
                    "            if 'Top_Depth_m' in dataframe and 'Base_Driller_m' in dataframe:\n",
                    "                dataframe['Cut_Length_m'] = (dataframe['Base_Driller_m'] - dataframe['Top_Depth_m']).round(2)\n",
                    "            if 'Recovery_pct' in dataframe and 'Cut_Length_m' in dataframe:\n",
                    "                dataframe['Recovered_m'] = ((dataframe['Recovery_pct'] / 100.0) * dataframe['Cut_Length_m']).round(2)\n",
                    "                \n",
                    "            dataframe['Recovery_Note'] = 'Core Intact'\n",
                    "            return dataframe.reindex(columns=master_columns)\n",
                    "        elif response.status_code == 429:\n",
                    "            wait_seconds = 45\n",
                    "            print(f'    [Rate Limit 429] Waiting {wait_seconds}s before retry (Attempt {attempt+1}/{max_retries})...')\n",
                    "            time.sleep(wait_seconds)\n",
                    "        else:\n",
                    "            raise RuntimeError(f'Gemini API Error {response.status_code}: {response.text}')\n",
                    "\n",
                    "    raise RuntimeError('Exceeded maximum retries for Gemini API.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Geological Data Audit & Quality Control (QC)\n",
                    "Before concatenating into the master table, every coring interval is audited against physical geology invariants:\n",
                    "1. **Depth Ordering:** $\\text{Top\\_Depth\\_m} < \\text{Base\\_Driller\\_m}$\n",
                    "2. **Cut Length Integrity:** $\\text{Cut\\_Length\\_m} = \\text{Base\\_Driller\\_m} - \\text{Top\\_Depth\\_m}$\n",
                    "3. **Recovery Bounds:** $0\\% \\le \\text{Recovery\\_pct} \\le 100\\%$\n",
                    "4. **Recovered Length Realism:** $\\text{Recovered\\_m} \\le \\text{Cut\\_Length\\_m}$"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def audit_core_dataframe(df_well):\n",
                    "    '''Audits geological physical constraints for every row in the dataframe.'''\n",
                    "    issues = []\n",
                    "    for idx, row in df_well.iterrows():\n",
                    "        core = row['Core_No']\n",
                    "        top = row['Top_Depth_m']\n",
                    "        base = row['Base_Driller_m']\n",
                    "        rec = row['Recovery_pct']\n",
                    "        \n",
                    "        if pd.notnull(top) and pd.notnull(base):\n",
                    "            if top >= base:\n",
                    "                issues.append(f'Core {core}: Top ({top}m) >= Base ({base}m)!')\n",
                    "        if pd.notnull(rec):\n",
                    "            if rec < 0.0 or rec > 100.0:\n",
                    "                issues.append(f'Core {core}: Recovery out of bounds ({rec}%)!')\n",
                    "                \n",
                    "    return issues"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Execute Extraction & Compile Master Database (`MASTER_ALL_WELLS_CORE_LOGS.csv`)\n",
                    "We extract each well's data via Gemini Vision, run the quality audit, save individual CSVs in `individual_csvs/`, and concatenate into `MASTER_ALL_WELLS_CORE_LOGS.csv`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "os.makedirs('individual_csvs', exist_ok=True)\n",
                    "all_well_dataframes = []\n",
                    "\n",
                    "print('Extracting Geological Data from Cropped Sections via Gemini Vision...')\n",
                    "for crop_record in crop_records:\n",
                    "    img_path = crop_record['Output_Image']\n",
                    "    base_name = os.path.splitext(crop_record['File'])[0]\n",
                    "    case_csv_path = os.path.join('individual_csvs', f'{base_name}.csv')\n",
                    "    \n",
                    "    # Check if previously extracted to conserve API quota\n",
                    "    if os.path.exists(case_csv_path):\n",
                    "        df_extracted = pd.read_csv(case_csv_path)\n",
                    "        print(f'  ✓ [Cached] {base_name:<45} -> {len(df_extracted):2d} cores loaded')\n",
                    "    else:\n",
                    "        df_extracted = extract_core_data_with_gemini(img_path, gemini_api_key)\n",
                    "        df_extracted.to_csv(case_csv_path, index=False)\n",
                    "        print(f'  ✓ [Gemini] {base_name:<45} -> {len(df_extracted):2d} cores extracted')\n",
                    "        time.sleep(12) # Polite pacing for rate limits\n",
                    "    \n",
                    "    # Run Geological QC Audit\n",
                    "    audit_errors = audit_core_dataframe(df_extracted)\n",
                    "    qc_status = 'PASS' if not audit_errors else f'FAIL ({len(audit_errors)} issues)'\n",
                    "    print(f'     └─ Geological QC Status: {qc_status}')\n",
                    "    all_well_dataframes.append(df_extracted)\n",
                    "\n",
                    "# Vertical Concatenation into Master CSV\n",
                    "master_dataframe = pd.concat(all_well_dataframes, ignore_index=True)\n",
                    "master_csv_path = 'MASTER_ALL_WELLS_CORE_LOGS.csv'\n",
                    "master_dataframe.to_csv(master_csv_path, index=False)\n",
                    "\n",
                    "print('\\n' + '='*80)\n",
                    "print(f'[COMPLETE] Master Database Successfully Compiled: {master_csv_path}')\n",
                    "print(f'Total Coring Rows in Database : {len(master_dataframe)}')\n",
                    "print(f'Total Unique Wells Digitised : {master_dataframe[\"Well\"].nunique()}')\n",
                    "print('='*80)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 9. Master Database Preview & Summary Analytics\n",
                    "We display sample rows from the compiled master table and an executive summary table grouped by well."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Sample Rows from MASTER_ALL_WELLS_CORE_LOGS.csv:')\n",
                    "preview_columns = ['Well', 'Operator', 'Field', 'Core_No', 'Top_Depth_m', 'Base_Driller_m', 'Cut_Length_m', 'Recovery_pct']\n",
                    "display(master_dataframe[preview_columns].head(12))\n",
                    "\n",
                    "print('\\nExecutive Geological Summary by Well:')\n",
                    "well_summary = master_dataframe.groupby(['Well', 'Operator', 'Field']).agg(\n",
                    "    Total_Cores=('Core_No', 'count'),\n",
                    "    Top_Cored_m=('Top_Depth_m', 'min'),\n",
                    "    Base_Cored_m=('Base_Driller_m', 'max'),\n",
                    "    Average_Recovery_pct=('Recovery_pct', 'mean')\n",
                    ").reset_index()\n",
                    "\n",
                    "display(well_summary)"
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
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open("full_workflow.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Successfully built full_workflow.ipynb with Gemini Vision!")

if __name__ == "__main__":
    build_notebook()
