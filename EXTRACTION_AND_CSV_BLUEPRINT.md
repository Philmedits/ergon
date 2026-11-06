# From Cropped Images to Master CSV: Complete End-to-End Blueprint

> **A structured, step-by-step technical roadmap for transforming clean cropped core log images into validated individual CSVs and a single unified Master CSV.**
>
> *Combines the Industrial Computer Vision foundation of `ergon_master_pipeline.py` with the elegant, token-saving LLM extraction pattern from the NAPE Hackathon.*

---

## 1. High-Level Architecture

```
┌────────────────────────────────────────────────────────┐
│ 100 Clean Cropped Images in 'cropped_images/'          │
│ (Header + Metadata + Core Summary Table isolated)       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STEP 1: Inventory & Grouping                           │
│ • Detect single-page vs. multi-page spillover files    │
│ • Group parts by Well ID (e.g. 061_p1 + 061_p2)        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STEP 2: Standardized Template Schema                   │
│ • Fixed column contract (Well, Operator, Core_No...)   │
│ • Consistent order matching the NAPE target format     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STEP 3: In-Memory Base64 & Vision LLM Extraction       │
│ • Zero temporary disk files (io.BytesIO)               │
│ • Petroleum Geologist system prompt + CSV enforcement  │
│ • Raw text converted to Pandas DataFrame               │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STEP 4: Multi-Page Spillover Stitching                 │
│ • Inherit Page 1 metadata onto Page 2 overflow rows    │
│ • Sequential core renumbering if necessary             │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STEP 5: Automated Quality Auditor & Guardrails         │
│ • Depth check: Top_Depth < Bottom_Depth                │
│ • Math check: Cut_Length = Bottom - Top                │
│ • Recovery check: 0 <= Recovery_pct <= 100             │
│ • Strict numerical type casting                        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STEP 6: Dual-Tier Export                               │
│ • 100 Individual CSVs in 'individual_csvs/'            │
│ • pd.concat() into 'MASTER_ALL_WELLS_CORE_LOGS.csv'    │
└────────────────────────────────────────────────────────┘
```

---

## 2. Step-by-Step Implementation Breakdown

### STEP 1: Inventory & Document Grouping
In our `cropped_images/` folder, files follow two distinct patterns:
1. **Single-Page Logs:** 
   `ergon_log_003_boxed_medium_10cores_cropped.png`
2. **Multi-Page Spillovers:**
   `ergon_log_061_spillover_standard_2p_p1_primary_cropped.png`
   `ergon_log_061_spillover_standard_2p_p2_spillover_cropped.png`

**What we do:**
* Group images by their base well ID (`ergon_log_003`, `ergon_log_061`, etc.).
* If a well has only 1 image $\to$ Process directly.
* If a well has 2 images (`p1_primary` and `p2_spillover`) $\to$ Process both and prepare for stitching.

---

### STEP 2: Define the Fixed Template Schema
To ensure every document produces identical column headers, we define the master column list upfront:

```python
TARGET_COLUMNS = [
    # Well-Level Metadata (repeated on every row)
    "Well",
    "Operator",
    "Country",
    "Licence",
    "Field",
    "Owners",
    "Latitude_DMS",
    "Longitude_DMS",
    "Latitude_DD",
    "Longitude_DD",
    "Elevation_KB_m",
    "Water_Depth_m",
    "Total_Depth_Driller_m",
    "Formation_at_TD",
    "Geologists",

    # Core Table Data (row-specific)
    "Core_No",
    "Interval_Raw",
    "Top_Depth_m",
    "Bottom_Depth_m",
    "Cut_Length_m",
    "Recovery_pct",
    "Recovered_m",

    # File Traceability
    "Source_File",
]

header_line = ",".join(TARGET_COLUMNS)
```

---

### STEP 3: In-Memory Base64 Encoding & Vision LLM Extraction
Instead of creating slow temporary image files on disk, we encode the image in RAM using `io.BytesIO()`:

```python
import base64
import io
from PIL import Image

def encode_image_to_base64(image_path: str) -> str:
    with Image.open(image_path) as img:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
```

#### The Petroleum Geologist System Prompt:
```python
prompt = f"""You are an expert petroleum data engineer and geologist digitising a scanned core log.
Extract all information from the provided image into valid CSV format.
Use EXACTLY this header row in this exact order:
{header_line}

Strict Rules:
1. One row per core run. Repeat all well-level metadata (Well, Operator, Field, etc.) on every row.
2. If interval is '2100.0 - 2114.8', set Top_Depth_m=2100.0, Bottom_Depth_m=2114.8, Cut_Length_m=14.8.
3. Recovery percentage must be a number (e.g. 96.0, not 0.96).
4. If a field is blank or not visible on the image, leave it empty. Never hallucinate or guess.
5. Return ONLY the raw CSV text. No explanations, no markdown fences (no ```csv).
"""
```

---

### STEP 4: Multi-Page Spillover Stitching
On spillover pages (Page 2), the left metadata block does not exist—only the overflow core table rows sit at the top.

**The Stitching Logic:**
```python
if is_spillover:
    # 1. Extract Page 1 (Primary) -> Has Well Metadata + Cores 1 to 15
    df_p1 = extract_from_image(p1_path)
    
    # 2. Extract Page 2 (Spillover) -> Has Cores 16 to 25 (Metadata is blank)
    df_p2 = extract_from_image(p2_path)
    
    # 3. Propagate Page 1 metadata onto Page 2 rows
    for meta_col in ["Well", "Operator", "Country", "Licence", "Field", "Owners"]:
        df_p2[meta_col] = df_p1[meta_col].iloc[0]
        
    # 4. Concatenate both pages into one continuous dataset
    df_well = pd.concat([df_p1, df_p2], ignore_index=True)
else:
    df_well = extract_from_image(single_image_path)
```

---

### STEP 5: Automated Quality Auditor & Guardrails (Column Checks)
Before saving any CSV, the DataFrame passes through 5 automated sanity checks:

```python
def audit_and_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Ensure numeric types
    numeric_cols = ["Core_No", "Top_Depth_m", "Bottom_Depth_m", "Cut_Length_m", "Recovery_pct"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Check Depth Validity: Top must be shallower than Bottom
    invalid_depths = df[df["Top_Depth_m"] >= df["Bottom_Depth_m"]]
    if not invalid_depths.empty:
        # Swap if accidentally inverted
        for idx in invalid_depths.index:
            df.loc[idx, ["Top_Depth_m", "Bottom_Depth_m"]] = df.loc[idx, ["Bottom_Depth_m", "Top_Depth_m"]].values

    # 3. Recalculate Cut_Length_m: Bottom - Top
    df["Cut_Length_m"] = (df["Bottom_Depth_m"] - df["Top_Depth_m"]).round(2)

    # 4. Recovery % Sanity Check
    # If recovery was reported as a fraction (e.g. 0.95), multiply by 100
    df.loc[(df["Recovery_pct"] > 0) & (df["Recovery_pct"] <= 1.0), "Recovery_pct"] *= 100.0

    # 5. Compute Recovered Length: Cut_Length * (Recovery / 100)
    df["Recovered_m"] = (df["Cut_Length_m"] * (df["Recovery_pct"] / 100.0)).round(2)

    # 6. Sort by Core Number sequentially
    df = df.sort_values(by="Core_No").reset_index(drop=True)

    return df
```

---

### STEP 6: Dual-Tier Export (Individual + Master Concatenation)
Once audited:

1. **Tier 1 (Individual CSV):**
   Save each well's data as its own independent CSV file:
   `individual_csvs/ergon_log_003_Core_Log.csv`
2. **Tier 2 (Master CSV):**
   Use `pd.concat` to combine all DataFrames into one master table:
   ```python
   master_df = pd.concat(all_audited_dfs, ignore_index=True)
   master_df.to_csv("MASTER_ALL_WELLS_CORE_LOGS.csv", index=False)
   ```

---

## 3. Summary of Deliverables

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| **Individual Well CSVs** | `individual_csvs/*.csv` | Exactly 1 CSV per unique well dossier containing all its core runs. |
| **Master Unified CSV** | `MASTER_ALL_WELLS_CORE_LOGS.csv` | All 100 wells combined into one massive table (~1,500 rows). |
| **Validation Audit Log** | `extraction_audit_report.json` | Report tracking total rows extracted, missing values, and depth checks. |

