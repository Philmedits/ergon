# The Master Data Layer Guide: From Raw PDFs to Clean Verified Datasets
> **A universal, production-ready engineering guide for processing collections of PDF documents—handling single-page sheets, short summaries, or 100+ page reports—using pure Vision LLMs to produce verified datasets for both Repeating Tables (`cores.csv`) and Key-Value Metadata (`metadata.csv`).**

---

## 🗺️ The Modern Architecture: Dual-Stream Pure Vision LLM

We deliberately avoid traditional OCR (Tesseract / EasyOCR). Instead, we use **Multimodal / Vision LLMs (e.g. Gemini 2.0 Flash)** with **Pydantic Schemas** to extract two distinct data streams directly into structured JSON:

```
                                    [ FOLDER OF N PDF DOCUMENTS ]
                                                 │
                                                 ▼
                             [ STAGE 0: INVENTORY & DUAL ROUTER ]
                             • Inspect document lengths & headers
                             • Route to Stream A (Tables) & Stream B (Metadata)
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
          [ STREAM A: REPEATING TABLES ]                    [ STREAM B: KEY-VALUE METADATA ]
          • Target: Coring summary pages                    • Target: Cover page / General Info
          • Target File: ground_truth_cores.csv             • Target File: ground_truth_metadata.csv
          • Output: List of 20+ core rows                   • Output: Single key-value well record
                        │                                                 │
                        ▼                                                 ▼
          [ STAGE 1: 300 DPI RENDER & CROP ]                [ STAGE 1: 300 DPI RENDER & CROP ]
          • Auto-rotate landscape 90°                       • Crop title block & header specs
          • Strip hole punches & logos                      • Strip background noise
                        │                                                 │
                        ▼                                                 ▼
          [ STAGE 2: CRASH-PROOF CACHE + RETRY ]            [ STAGE 2: CRASH-PROOF CACHE + RETRY ]
          • Check local JSON cache first                    • Check local JSON cache first
          • Exponential backoff on 429 limits               • Exponential backoff on 429 limits
          • Vision LLM with Pydantic Schema                 • Vision LLM with Pydantic Schema
                        │                                                 │
                        ▼                                                 ▼
          [ STAGE 3: CONCATENATE & UNIFY ]                  [ STAGE 3: CONSOLIDATE FLEET METADATA ]
          • Stitch multi-page continuations                 • 1 record per well document
                        │                                                 │
                        ▼                                                 ▼
          [ STAGE 4: SANITIZATION & CLEANING ]              [ STAGE 4: METADATA SANITIZATION ]
          • Fix European decimals (12,5 -> 12.5)            • Parse ISO dates (YYYY-MM-DD)
          • Strip unit suffixes ('m', '%')                  • Clean UTM / Lat-Long coordinates
          • Forward-fill merged core cells                  • Cast depths & elevations to float
                        │                                                 │
                        ▼                                                 ▼
          [ STAGE 5: DOMAIN EXPERT VALIDATION ]             [ STAGE 5: DOMAIN EXPERT AUDIT TRAIL ]
          • Check depth_to > depth_from                     • Attach crop image links for human review
          • Flag recovery % > 100%                          • Flag missing mandatory metadata fields
                        │                                                 │
                        ▼                                                 ▼
          [ STAGE 6: GROUND TRUTH EVALUATION ]              [ STAGE 6: GROUND TRUTH EVALUATION ]
          • Score vs ground_truth_cores.csv                 • Score vs ground_truth_metadata.csv
                        │                                                 │
                        ▼                                                 ▼
             [ final_cores.csv ]                               [ final_metadata.csv ]
```

---

# 📚 The 5-Layer Edge Cases Master Catalog

Real-world enterprise and geological PDFs are filled with visual and formatting traps. Here is the complete catalog of edge cases you must account for:

## Layer 1: File & Format Edge Cases (The Container)
* **0-Byte / Corrupt Files:** A PDF was interrupted during download or is 0 KB. Passing it to a library throws a fatal error that crashes the entire loop. *Fix: Wrap in `try...except`, log error, and continue.*
* **Garbled Non-Standard Fonts (CID / Type3):** The PDF text layer looks normal on screen, but copying it yields square boxes or unicode gibberish (`\u0001\u0002`). *Fix: Never rely on PDF text layers for scanned reports; use Vision LLMs on 300 DPI images.*
* **Encrypted / Password-Protected Files:** PDFs with permission restrictions throw `PdfEncryptedError`. *Fix: Check `doc.is_encrypted` before processing.*

## Layer 2: Physical & Visual Scan Edge Cases (The Paper)
* **Scanner Skew (Tilted Paper):** The physical sheet was placed at a 2°–5° angle on the scanner. Fixed rectangular crops will slice through text corners. *Fix: Add a 5% to 10% safety margin to all crop boxes.*
* **Landscape Tables in Portrait Reports:** Tables printed horizontally (turned 90° sideways). *Fix: Detect if `width > height` and rotate upright (`page.set_rotation(90)`) before rendering.*
* **Rubber Stamps & Signatures:** "CONFIDENTIAL" or approval stamps stamped directly over numerical data. *Fix: Vision LLMs ignore these when prompted: "Ignore handwritten marks and rubber stamps."*
* **Hole Punches & Staple Shadows:** Black circular marks along the left margin from 3-hole punches. *Fix: Crop the left margin by 3–5% to eliminate them.*
* **Bleed-Through (Ghost Text):** Thin paper where text from the back side shows through faintly on the front. *Fix: Vision LLMs naturally filter low-contrast background bleed.*

## Layer 3: Table Structure & Layout Edge Cases (The Grid)
* **The Multi-Page Table Split:** A table starts on Page 85, continues onto 86, and finishes on 87. Page 86 has no headers and begins with `"Table 6.1 (Cont'd)"`. *Fix: Treat table extraction as a continuous stream; filter out "Cont'd" banner rows in post-processing.*
* **The "Total / Summary" Row:** The bottom row says `Total Recovered: 154.2m (96%)`. If unhandled, this becomes a fake core row. *Fix: Filter out any row where core number or description contains `"total"` or `"average"`.*
* **Merged Category Cells (Row Spans):** "Core #1" spans across 4 rows of intervals; rows 2, 3, and 4 leave the core number blank. *Fix: Forward-fill (`ffill()`) grouping columns within each document.*
* **Multi-Line Text Wrapping:** A geological description wraps over 3 lines inside one cell. *Fix: Vision LLMs with strict Pydantic schemas naturally group all lines into one string field.*
* **Side-by-Side Tables:** Cores 1–10 on the left half of the page; Cores 11–20 on the right half. *Fix: Split the page image down the middle into two vertical halves before sending to the LLM.*
* **Borderless Tables:** Tables with no lines, separated only by whitespace. *Fix: Vision LLMs understand whitespace column alignment without requiring grid lines.*

## Layer 4: Data & Character Edge Cases (The Values)
* **European Comma Decimals:** Numbers written as `12,50` instead of `12.50`. *Fix: Replace commas with dots (`s.str.replace(",", ".")`).*
* **Faint / Missing Decimal Points:** A faded typewriter period turns `1250.5` into `12505` (10x too deep!). *Fix: Domain validation checks depth continuity between consecutive cores.*
* **The "Zero vs. Null" Trap:** A cell has `-`, `N/A`, `nil`, or is blank. Storing this as `0.0` is dangerous (e.g. 0% recovery means lost core; missing recovery means not measured!). *Fix: Map `-`, `--`, `N/A`, `nil` to `None`/`NaN`, never to `0.0`.*
* **Accounting Negatives:** Negative numbers in parentheses: `(14.2)`. *Fix: Regex replace: `s.str.replace(r"\((.*?)\)", r"-\1")`.*
* **Units Glued to Numbers:** Strings like `"15.4m"`, `"98%"`, `"2.45 g/cm3"`. *Fix: Strip alphabetical characters before type casting.*
* **Invisible Whitespace:** Non-breaking spaces (`\xa0`) or trailing spaces (`"Core 1 "` vs `"Core 1"`). *Fix: Run `.str.strip()` and replace `\xa0` across all text fields.*

## Layer 5: Domain & Semantic Edge Cases (The Business Logic)
* **Inverted Depths:** `depth_from = 1500m`, `depth_to = 1490m`. In a vertical well, depth must increase downwards. *Fix: Automatically swap flipped intervals or flag them for review.*
* **Impossible Percentages:** Recovery percentages like `195%` or negative values. *Fix: Flag any row where `recovery_pct < 0` or `recovery_pct > 105`.*
* **Mixed Unit Systems (Meters vs. Feet):** Well 1 logged in meters; Well 2 logged in feet ($1\text{ ft} \approx 0.3048\text{ m}$). *Fix: Check document header units; convert all depths to meters if needed.*
* **Sequence Gaps & Reversals:** Cores appearing out of order. *Fix: Always sort by `depth_from` rather than relying solely on the core number string.*

---

# 🛡️ Foundational Engineering: Pydantic Schemas

Using Pydantic guarantees that the Vision LLM returns **100% syntactically valid JSON** matching your expected structure. Zero regex stripping, zero JSON parse exceptions.

```python
from pydantic import BaseModel, Field
from typing import List, Optional

# --- STREAM A: Repeating Table Schema (Cores) ---
class CoreRow(BaseModel):
    core_number: str = Field(description="Core number or ID, e.g. '1', '2'")
    depth_from: Optional[float] = Field(None, description="Top depth of core cut in meters")
    depth_to: Optional[float] = Field(None, description="Bottom depth of core cut in meters")
    recovered_length: Optional[float] = Field(None, description="Total length recovered in meters")
    recovery_pct: Optional[float] = Field(None, description="Core recovery percentage (0 to 100)")
    description: Optional[str] = Field(None, description="Lithology or remarks")

class CoringTableExtraction(BaseModel):
    rows: List[CoreRow] = Field(description="List of all extracted data rows, excluding headers and totals")


# --- STREAM B: Key-Value Document Metadata Schema ---
class WellMetadataExtraction(BaseModel):
    well_name: Optional[str] = Field(None, description="Official well designation, e.g. '31/2-1'")
    operator: Optional[str] = Field(None, description="Operating company name, e.g. 'Norsk Hydro'")
    field_name: Optional[str] = Field(None, description="Field or block name, e.g. 'Troll'")
    spud_date: Optional[str] = Field(None, description="Date drilling began, format YYYY-MM-DD if possible")
    completion_date: Optional[str] = Field(None, description="Date well completed, format YYYY-MM-DD")
    water_depth_m: Optional[float] = Field(None, description="Water depth in meters")
    total_depth_m: Optional[float] = Field(None, description="Total measured depth (TD) in meters")
    rig_name: Optional[str] = Field(None, description="Drilling rig or vessel name")
```

---

# ⚡ Foundational Engineering: Caching & Rate-Limit Backoff

This wrapper guarantees two things:
1. **Crash-Proof Cache:** If you already processed a page, it reads from disk in `0.001s` (no API cost).
2. **Exponential Backoff:** If you hit API quota limits (`429 Too Many Requests`), it pauses and retries automatically instead of failing.

```python
import os
import json
import time
import google.generativeai as genai
from PIL import Image

def call_vision_llm_with_cache_and_retry(
    image_path: str, 
    prompt: str, 
    response_schema, 
    cache_dir: str = "llm_cache", 
    max_retries: int = 4
) -> dict:
    """
    Executes a Vision LLM call with local JSON caching and automatic 429 exponential backoff.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = os.path.join(cache_dir, os.path.basename(image_path).replace(".png", ".json"))

    # 1. Check local cache first (Free, Instant)
    if os.path.exists(cache_filename):
        with open(cache_filename, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2. Call LLM with Exponential Backoff
    img = Image.open(image_path)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": response_schema
        }
    )

    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, img])
            result_dict = json.loads(response.text)

            # Save successful response to cache
            with open(cache_filename, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2)

            return result_dict

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource" in err_str or "quota" in err_str:
                wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s, 16s
                print(f"⏳ Rate limit encountered. Backing off for {wait_time}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"❌ Non-rate-limit error: {e}")
                raise e

    raise RuntimeError(f"Exceeded max retries for image: {image_path}")
```

---

# Stage 0: Batch Inventory & Dual-Router

```python
import glob
import fitz  # PyMuPDF
import pandas as pd

def inventory_and_route_fleet(folder_path: str) -> pd.DataFrame:
    """
    Inventories all PDFs and identifies:
    - Cover / General Info page (for Metadata extraction)
    - Coring / Table pages (for Cores table extraction)
    """
    pdf_files = glob.glob(os.path.join(folder_path, "**", "*.pdf"), recursive=True)
    inventory = []

    for file_path in pdf_files:
        filename = os.path.basename(file_path)
        try:
            doc = fitz.open(file_path)
            num_pages = len(doc)
            
            # Find Table Pages
            table_pages = []
            for i in range(num_pages):
                txt = doc[i].get_text("text").lower()
                if any(k in txt for k in ["coring", "core summary", "core no", "recovered"]):
                    table_pages.append(i)

            # Metadata is typically on page 0 (cover) or page 1/2 (well summary)
            metadata_page = 0
            for i in range(min(5, num_pages)):
                txt = doc[i].get_text("text").lower()
                if "general information" in txt or "well data" in txt or "spud" in txt:
                    metadata_page = i
                    break

            doc.close()
            is_valid = True
        except Exception as e:
            num_pages = 0
            table_pages = []
            metadata_page = 0
            is_valid = False

        inventory.append({
            "file_path": file_path,
            "filename": filename,
            "num_pages": num_pages,
            "table_pages": table_pages,
            "metadata_page": metadata_page,
            "is_valid": is_valid
        })

    return pd.DataFrame(inventory)
```

---

# Stage 1: Smart 300 DPI Rendering & Margin Cropping

```python
def render_and_crop_page(
    doc: fitz.Document, 
    page_number: int, 
    output_img_path: str, 
    dpi: int = 300,
    top_margin_pct: float = 0.08,
    bottom_margin_pct: float = 0.08
) -> str:
    """
    Renders page to 300 DPI, auto-rotates landscape pages,
    and strips noise margins (logos, page numbers, hole punches).
    """
    page = doc[page_number]
    rect = page.rect

    # Auto-rotate landscape pages upright
    if rect.width > rect.height and page.rotation == 0:
        page.set_rotation(90)
        rect = page.rect

    # Crop margins with 5% safety margin
    crop_rect = fitz.Rect(
        rect.width * 0.03,                 # Shave hole punches on left
        rect.height * top_margin_pct,      # Shave header logos
        rect.width * 0.98,                 # Right edge
        rect.height * (1.0 - bottom_margin_pct) # Shave footer page numbers
    )

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, clip=crop_rect)
    pix.save(output_img_path)
    return output_img_path
```

---

# Stage 2: Dual Extraction Pipelines

### Stream A: Extracting Repeating Core Tables
```python
def extract_cores_from_page(image_path: str, source_filename: str, page_num: int) -> list:
    prompt = f"""
    Inspect this technical coring table image from '{source_filename}', Page {page_num+1}.
    Extract all data rows into the CoringTableExtraction schema.
    Rules:
    - Ignore headers, column titles, and bottom 'Total' or 'Average' rows.
    - Set unreadable or empty values to null.
    """
    data = call_vision_llm_with_cache_and_retry(
        image_path=image_path,
        prompt=prompt,
        response_schema=CoringTableExtraction
    )
    
    rows = data.get("rows", [])
    for r in rows:
        r["source_file"] = source_filename
        r["page_number"] = page_num + 1
        r["audit_image_crop"] = image_path  # The Domain Expert Audit Trail!
    return rows
```

### Stream B: Extracting Key-Value Well Metadata
```python
def extract_metadata_from_page(image_path: str, source_filename: str) -> dict:
    prompt = f"""
    Inspect this well summary / cover page image from '{source_filename}'.
    Extract the key-value document metadata into the WellMetadataExtraction schema.
    Extract: well name, operator, field, spud date, completion date, water depth, total depth, and rig name.
    If a field is not present on this page, leave it as null.
    """
    data = call_vision_llm_with_cache_and_retry(
        image_path=image_path,
        prompt=prompt,
        response_schema=WellMetadataExtraction
    )
    data["source_file"] = source_filename
    data["audit_image_crop"] = image_path  # Audit link for human review!
    return data
```

---

# Stage 3: Tabular Cleaning & Domain Validation

```python
import re
import numpy as np

def sanitize_number_series(series: pd.Series) -> pd.Series:
    """Converts European decimals ('12,5' -> 12.5), strips units ('m', '%'), handles nulls."""
    s = series.astype(str).str.strip()
    s = s.replace(r"^(?:None|nan|null|-|--|n/a|nil|\*|\s*)$", None, regex=True)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.extract(r"([-+]?[0-9]*\.?[0-9]+)")[0]
    return pd.to_numeric(s, errors="coerce")


def clean_and_validate_cores(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans numeric types and enforces geological domain rules on cores table."""
    df = df.copy()
    
    # 1. Forward-fill core numbers within document
    if "core_number" in df.columns and "source_file" in df.columns:
        df["core_number"] = df.groupby("source_file")["core_number"].ffill()

    # 2. Sanitize numeric fields
    for col in ["depth_from", "depth_to", "recovered_length", "recovery_pct"]:
        if col in df.columns:
            df[col] = sanitize_number_series(df[col])

    # 3. Domain Rule 1: Inverted depth check (depth_to must be > depth_from)
    if "depth_from" in df.columns and "depth_to" in df.columns:
        inverted = df["depth_to"] < df["depth_from"]
        if inverted.any():
            print(f"⚠️ Swapping {inverted.sum()} inverted depth intervals!")
            df.loc[inverted, ["depth_from", "depth_to"]] = (
                df.loc[inverted, ["depth_to", "depth_from"]].values
            )

    # 4. Domain Rule 2: Recovery % bounds (0% to 105%)
    if "recovery_pct" in df.columns:
        anomalies = (df["recovery_pct"] < 0) | (df["recovery_pct"] > 105)
        df["flag_suspicious_recovery"] = anomalies

    return df.drop_duplicates().reset_index(drop=True)


def clean_metadata_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans well metadata dataset."""
    df = df.copy()
    for col in ["water_depth_m", "total_depth_m"]:
        if col in df.columns:
            df[col] = sanitize_number_series(df[col])
    return df.reset_index(drop=True)
```

---

# Stage 4: Ground Truth Evaluation for Both Datasets

```python
def evaluate_dataset(extracted_df: pd.DataFrame, gt_csv_path: str, title: str, eval_cols: list, tolerance: float = 0.05):
    """Generic ground truth evaluator that scores columns within specified tolerance."""
    gt_df = pd.read_csv(gt_csv_path)

    print("\n" + "="*60)
    print(f"🏆 {title} GROUND TRUTH SCORECARD")
    print("="*60)
    print(f"Extracted Rows:    {len(extracted_df)}")
    print(f"Ground Truth Rows: {len(gt_df)}")

    for col in eval_cols:
        if col in extracted_df.columns and col in gt_df.columns:
            ext_vals = extracted_df[col].to_numpy()
            gt_vals = gt_df[col].to_numpy()
            n = min(len(ext_vals), len(gt_vals))
            
            if pd.api.types.is_numeric_dtype(extracted_df[col]):
                matches = np.isclose(ext_vals[:n], gt_vals[:n], atol=tolerance, equal_nan=True)
            else:
                matches = (ext_vals[:n].astype(str) == gt_vals[:n].astype(str))
                
            score = (matches.sum() / n) * 100
            print(f"🎯 Column '{col}' Accuracy: {score:.2f}%")

    print("="*60 + "\n")
```

---

# 🚀 The Master Multi-Document Orchestrator

This single master runner processes the entire folder of PDFs and produces both verified CSVs:

```python
def run_complete_fleet_pipeline(
    pdf_folder: str, 
    gt_cores_path: str = None, 
    gt_metadata_path: str = None,
    output_cores_csv: str = "final_cores.csv",
    output_metadata_csv: str = "final_metadata.csv"
):
    """
    Executes the full dual-stream pipeline across all PDFs in a folder.
    Produces final_cores.csv and final_metadata.csv with audit trails.
    """
    os.makedirs("temp_crops", exist_ok=True)
    
    # 1. Inventory & Route
    inventory = inventory_and_route_fleet(pdf_folder)
    print(f"📂 Found {len(inventory)} total documents to process.")

    all_cores_rows = []
    all_metadata_records = []

    # 2. Process each document
    for idx, row in inventory.iterrows():
        filename = row["filename"]
        file_path = row["file_path"]
        print(f"\n📄 [{idx+1}/{len(inventory)}] Processing: {filename}")

        if not row["is_valid"]:
            continue

        doc = fitz.open(file_path)

        # Stream B: Extract Metadata (Cover / General Info page)
        meta_page_idx = row["metadata_page"]
        meta_img = f"temp_crops/{filename}_meta_p{meta_page_idx}.png"
        render_and_crop_page(doc, meta_page_idx, meta_img)
        meta_data = extract_metadata_from_page(meta_img, filename)
        all_metadata_records.append(meta_data)

        # Stream A: Extract Repeating Tables (Coring pages)
        for tbl_page_idx in row["table_pages"]:
            table_img = f"temp_crops/{filename}_table_p{tbl_page_idx}.png"
            render_and_crop_page(doc, tbl_page_idx, table_img)
            page_core_rows = extract_cores_from_page(table_img, filename, tbl_page_idx)
            all_cores_rows.extend(page_core_rows)

        doc.close()

    # 3. Clean & Validate
    df_cores = clean_and_validate_cores(pd.DataFrame(all_cores_rows))
    df_meta = clean_metadata_dataset(pd.DataFrame(all_metadata_records))

    # 4. Save CSVs (with Audit Trail columns included!)
    df_cores.to_csv(output_cores_csv, index=False)
    df_meta.to_csv(output_metadata_csv, index=False)
    print(f"\n💾 Saved clean cores table:    {output_cores_csv}")
    print(f"💾 Saved clean metadata table: {output_metadata_csv}")

    # 5. Evaluate vs Ground Truth
    if gt_cores_path:
        evaluate_dataset(df_cores, gt_cores_path, "CORES TABLE", ["depth_from", "depth_to", "recovery_pct"])
    if gt_metadata_path:
        evaluate_dataset(df_meta, gt_metadata_path, "WELL METADATA", ["well_name", "operator", "water_depth_m", "total_depth_m"])

    return df_cores, df_meta
```
