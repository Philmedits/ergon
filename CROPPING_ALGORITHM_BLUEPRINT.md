# Geological Core Log Universal Cropping Algorithm: Master Blueprint

**Project:** Team Ergon AI -- Core Log Document Ingestion  
**Target:** Scanned Well Logs (1970s--1990s historical archives, multi-operator)  
**Goal:** Precision cropping of the **Data Section** (Header + Metadata + Core Summary Table) while excluding **Downhole Curves** and **Symbol Indices** to optimize token usage for Vision LLMs.

---

## 1. Executive Summary & The Golden Rule

When feeding scanned document crops into Multimodal Large Language Models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro):

* **Over-Cropping is Fatal:** Slicing through `LOGGERS - DRILLERS`, `GEOLOGISTS`, or the bottom row of the table permanently destroys data. The LLM cannot extract what is not in the image.
* **Under-Cropping is Harmless:** Leaving 30 to 50 pixels of white space at the bottom (or even a thin boundary line from the next section) costs negligible tokens. Vision models tile images in fixed 512x512 patches anyway. The model easily ignores extra white paper when prompted.

**The Golden Engineering Objective:**  
Never chop real data. Always add a generous safety cushion (25 to 30 pixels) and let the lower of the two data blocks (Table vs. Metadata) govern the crop line.

---

## 2. Post-Mortem: Why Past Approaches Failed

To build a reliable algorithm, we must understand the 5 specific traps that caused previous versions to break:

### Trap 1: Hardcoded Page Height Percentages
* **What went wrong:** The code used fixed ratios like `roi_top = 0.18 * h`, `lines[0] > 0.32 * h`, `clamp = 0.55 * h`.
* **Why it broke:** Documents do not have a uniform aspect ratio. Normal portrait sheets are 1168 x 1600 (ratio 0.73), while synthetic or rotated sheets are rendered on square canvases of 1600 x 1600 (ratio 1.0) with 216-pixel white padding bands. Hardcoded percentages landed in completely different physical zones depending on the canvas.

### Trap 2: The Header-to-Metadata Gap
* **What went wrong:** Naive gap detection scanned downwards from near the top and stopped at the "first gap >= 35px".
* **Why it broke (The `033` / `035` bug):** The company header box ends with a divider line at Y = 403. The first metadata line (`COUNTRY`) does not begin until Y = 468. There is a **65-pixel blank space** between the header box and `COUNTRY`. The naive scanner saw that 65px gap and immediately quit at Y = 418, completely missing `COUNTRY`, `FIELD`, `FORMATION`, `LOGGERS`, and `GEOLOGISTS`!

### Trap 3: The Outer Page Frame Border
* **What went wrong:** A simple horizontal line detector searched for horizontal lines and took the last one (`lines[-1]`) as the table bottom.
* **Why it broke (The `001` / `031` bug):** Boxed logs have a black rectangular border framing the entire sheet down to Y = 1327. The line detector mistook that outer frame border for a table row, causing the crop to grab the entire page down to Y = 1352 including downhole curves and symbol index.

### Trap 4: Continuation / Spillover Pages
* **What went wrong:** Running the same metadata detector on Page 2 or Page 3 of multi-page logs.
* **Why it broke (The `061` bug):** Spillover pages have **NO metadata block** on the left; the table takes up the top, and the Symbol Index or curves sit below. A metadata detector running on Page 2 caught the symbol index boxes and falsely treated them as text.

### Trap 5: Keyword Dependency ("SYMBOL INDEX")
* **What went wrong:** Relying on OCR or template matching to find the exact string `"SYMBOL INDEX"`.
* **Why it broke:** Historical logs use varying terminology: `LEGEND`, `LITHOLOGY KEY`, `CORE SYMBOLS`, or no header at all (just raw symbol boxes). Blurry or low-resolution scans cause OCR to miss the title completely.

---

## 3. Physical Document Anatomy (The Invariants)

Every single core log document follows a consistent structural hierarchy:

```
┌─────────────────────────────────────────────────────────────┐
│ ZONE 1: TOP HEADER BOX                                      │
│   Company Logo, "CORE LOG", Well ID, Scale, Date            │
├──────────────────────────────┬──────────────────────────────┤ <── Header Bottom Line / Table Top
│ ZONE 2A: LEFT METADATA TRAIN │ ZONE 2B: CORE SUMMARY TABLE  │
│   COUNTRY                    │   NO │ INTERVAL  │ RECOVERY  │
│   LICENCE                    │   1  │ 2100-2114 │ 96.0      │
│   FIELD                      │   2  │ 2115-2130 │ 97.0      │
│   OWNERS                     │   ...                        │
│   FORMATION AT TD            │   (Repeating ladder pitch:   │
│   LOGGERS - DRILLERS         │    18 to 26 pixels per row)  │
│   GEOLOGISTS                 │   Last Table Row             │
│   (Repeating line pitch:     ├──────────────────────────────┤ <── Table Bottom (variable)
│    12 to 22 pixels per line) │                              │
├──────────────────────────────┘                              │
│ ZONE 3: THE WHITESPACE CANYON (35px to 80px empty void)     │
├─────────────────────────────────────────────────────────────┤ <── Safe Cut Line = max(Table, Meta) + 25px
│ ZONE 4: EXCLUDED LOWER SECTION                              │
│   Symbol Index Grid / Downhole Seismic & Wireline Curves    │
└─────────────────────────────────────────────────────────────┘
```

### The Two Repeating Rhythms:
1. **The Table Ladder (Right Column):** Rows repeat periodically with a pitch of **18 to 26 pixels**. When the table ends, this repeating rhythm stops.
2. **The Metadata Train (Left Column):** Text lines repeat periodically with a pitch of **12 to 22 pixels**. When the metadata ends, this repeating rhythm stops.

---

## 4. The Universal Algorithm Architecture

The robust pipeline consists of 5 modular steps:

```
Raw Scanned PDF Page
         │
         ▼
[ Step 1: Pre-Normalization ]
  - Correct 90/180/270 degree rotation to 0 deg upright
  - Deskew fractional scanner tilt (-6 to +6 deg)
  - Auto-crop pure white canvas letterbox bands (if square canvas)
         │
         ▼
[ Step 2: Page Role Classification ]
  - If continuation page (is_primary = False):
      Cut = Table Ladder Bottom + 25px cushion (Bypass metadata scan completely)
  - If primary page (is_primary = True):
      Proceed to Steps 3 & 4
         │
         ▼
[ Step 3: Table Ladder Engine ]
  - Filter horizontal lines in right column (X: 45% to 94%)
  - Track the repeating row ladder (pitch: 18-26px)
  - Identify true table_bottom_y (ignores outer frame line at bottom)
         │
         ▼
[ Step 4: Metadata Train Engine ]
  - Horizontal ROI: Left column (X: 4% to 45%)
  - Start strictly AFTER the top header divider line
  - Locate first text line (COUNTRY) where line height >= 6px
  - Walk down line-by-line: continue as long as next line is within pitch (< 30px)
  - Stop at the first true whitespace canyon (gap >= 35px)
  - Result: metadata_bottom_y (captures GEOLOGISTS at Y=701)
         │
         ▼
[ Step 5: Dual-Anchor Integration & Clamping ]
  - Target Cut = max(table_bottom_y, metadata_bottom_y) + 25px cushion
  - If Symbol Index / Curve ceiling exists:
      Final Cut = min(Target Cut, symbol_ceiling_y - 10px)
```

---

## 5. Detailed Step-by-Step Logic

### Step 1: Canvas Auto-Trim (Letterbox Removal)
When square PDFs (like `033` and `035`) are rendered, they have ~216 pixels of pure white borders at the top and bottom.
* Find the first and last rows where `ink_sum > 10`.
* If a document has > 100px of pure white padding at the top, adjust the coordinate offset or crop off the empty border so the header starts at the true document origin.

### Step 2: Continuation vs. Primary Handling
* Continuation/spillover pages (Page 2, 3...) only contain table rows.
* On continuation pages, **never scan the left column** for metadata. The cutoff is simply `table_bottom_y + 25px`.
* This completely eliminates false triggers from symbol boxes on spillover sheets (e.g., `ergon_log_061 P2`).

### Step 3: Table Ladder Detection (Periodic Pitch)
* Use horizontal morphological opening (`cv2.morphologyEx` with a `(30, 1)` kernel) on the right half of the image (`x: 0.45 * w` to `0.94 * w`).
* Find all distinct horizontal line positions: `[y0, y1, y2, ..., yN]`.
* Group lines into a ladder where consecutive line gaps are consistent with row height (`16px <= gap <= 32px`).
* The **last line of that continuous ladder** is `table_bottom_y`.
* Any isolated line hundreds of pixels lower (such as the outer frame border at Y = 1327) has a gap of > 100px from the previous line and is automatically rejected.

### Step 4: Metadata Train Detection (Sequential Ink Tracking)
* Binarize the left column (`x: 0.04 * w` to `0.45 * w`).
* Project horizontally to get row ink sums. Active rows have `ink_sum > 20`.
* Group adjacent active rows into text lines: `(start_y, end_y, line_height)`.
* **Find the entry point:** Identify the first true text line (`line_height >= 6px`) in the upper section (ignoring the thin 1-3px header divider line).
* **Walk the train:** Step through lines sequentially:
  * If `next_line_start - cur_line_end < 30px`: continue down the train.
  * If `next_line_start - cur_line_end >= 35px`: we have reached the whitespace canyon! Stop immediately.
* Return `cur_line_end` as `metadata_bottom_y`.

### Step 5: Final Cut Decision
```python
# Whichever data block extends lower governs the cut
raw_cut = max(table_bottom_y, metadata_bottom_y)

# Add generous safety cushion (guarantees descenders and signatures are preserved)
final_cut = raw_cut + 25

# Hard ceiling guardrail (prevent penetrating symbol index if present)
if symbol_ceiling_y > 0 and final_cut > (symbol_ceiling_y - 10):
    final_cut = symbol_ceiling_y - 10
```

---

## 6. Concrete Case Study: `ergon_log_033` and `035`

Here are the exact pixel measurements from the physical file:

```
Y = 401..403  : Bottom divider of company header box
Y = 403..468  : 65px header gap (Old code crashed here!)
Y = 468..477  : Line 1: COUNTRY: Norway (Gap = 65px from header)
Y = 489..498  : Line 2: LICENCE: PL091 (Gap = 12px)
Y = 510..519  : Line 3: FIELD: Tyrihans (Gap = 12px)
Y = 531..540  : Line 4: OWNERS: Statoil... (Gap = 12px)
Y = 552..561  : Line 5: COORDINATES (Gap = 12px)
Y = 573..582  : Line 6: ELEVATION KB (Gap = 12px)
Y = 594..603  : Line 7: WATER DEPTH (Gap = 12px)
Y = 615..624  : Line 8: TOTAL DEPTH (DRILLER) (Gap = 12px)
Y = 636..645  : Line 9: TOTAL DEPTH (LOGGER) (Gap = 12px)
Y = 657..666  : Line 10: FORMATION AT TD (Gap = 12px)
Y = 670       : [TABLE ENDS HERE - Row 10]
Y = 678..687  : Line 11: LOGGERS - DRILLERS (Gap = 12px)
Y = 699..701  : Line 12: GEOLOGISTS: J.C., J.F.B., T.D. (Gap = 12px)
Y = 701..748  : 47-PIXEL CANYON OF PURE WHITE SPACE
Y = 748..761  : SYMBOL INDEX banner
```

### Old vs. New Behavior:
* **Old Algorithm:** Started scanning at Y = 288. Hit the 65px gap between 403 and 468. Returned Y = 418. Table ended at Y = 670. Cut at Y = 679.
  * **Result:** Sliced through `LOGGERS - DRILLERS`, completely deleted `GEOLOGISTS`.
* **Universal Algorithm:**
  1. Identifies Table Ladder ends at Y = 670.
  2. Enters metadata train at Line 1 (`COUNTRY`, Y = 468).
  3. Steps through 12-pixel gaps until after Line 12 (`GEOLOGISTS`, Y = 701).
  4. Detects the 47px canyon between 701 and 748. Stops at Y = 701.
  5. Takes `max(670, 701) = 701`.
  6. Adds 25px cushion: **Cuts cleanly at Y = 726**.
  * **Result:** All 10 table rows kept, `LOGGERS - DRILLERS` 100% kept, `GEOLOGISTS` 100% kept, `SYMBOL INDEX` 100% excluded.

---

## 7. Audit & Validation Matrix across the 100 Dataset

| Document Category | Key Challenge | Universal Algorithm Behavior |
| :--- | :--- | :--- |
| **Short Table (3-10 cores)** e.g., `033`, `035` | Table ends early; metadata is taller | `metadata_bottom_y` takes priority; cuts under `GEOLOGISTS`. |
| **Long Table (18-25 cores)** e.g., `001`, `034` | Table extends far below metadata | `table_bottom_y` takes priority; captures all core rows. |
| **Boxed Sheet with Outer Frame** e.g., `001`, `031` | Outer frame line at bottom (Y=1327) | Ladder periodicity rejects isolated frame line (>100px gap). |
| **Spillover Continuation Pages** e.g., `061 P2` | No metadata block; symbol index below | Bypasses metadata scan; cuts strictly under last table row. |
| **Tilted Scans (-6 to +6 deg)** e.g., `016`, `030` | Angled text rows | Deskewing straightens lines first; row sums align cleanly. |
| **Faded Carbon Copies** e.g., `090` | Low contrast ink | Adaptive background-level thresholding preserves text ink. |

---

## 8. Summary Checklist for Implementation

When ready to update the master pipeline, verify these points:

1. [ ] **No arbitrary percentage clamps:** Remove all `0.18*h`, `0.32*h`, `0.55*h` hardcoded numbers.
2. [ ] **Table Ladder:** Detect repeating pitch (18-26px) to reject isolated outer frame lines.
3. [ ] **Metadata Train:** Start scan below the header divider, enter at first genuine text line, and follow pitch until the >= 35px whitespace canyon.
4. [ ] **Double-Bottom Rule:** Always use `max(table_bottom_y, metadata_bottom_y)`.
5. [ ] **Safety Cushion:** Always add 25 to 30 pixels below the lower boundary.
6. [ ] **Spillover Guard:** Only run the metadata train on primary pages (`is_primary=True`).
