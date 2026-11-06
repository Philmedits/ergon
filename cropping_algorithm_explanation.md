# Section Cropping: The Complete Master Guide (From ABC to Z)

> **A plain-English, intuitive reference manual explaining the Computer Vision algorithms, Mathematical Morphology, and Table Detection Engines in `ergon_master_pipeline.py`.**

---

# PART 1: Optical Normalization (The "Magnifying Glass" Flow)

## 1. The Core Problem: Why Simple Thresholding Fails

When an old geological well log is scanned or photocopied, lighting is never uniform:
* **The left side (under a book-spine shadow):** Paper is dark gray (brightness ≈ `130`).
* **The right side (under bright light):** Paper is clean white (brightness ≈ `240`).
* **Black printed ink in the bright area:** Has brightness ≈ `80`.

Notice the trap: **The blank paper under the shadow (`130`) is darker than the ink in the light (`80`)!**

If you tell a computer a naive rule like *"Anything darker than 150 is ink"*, the computer turns the **entire shadowed half of the page into a solid black smudge**, destroying the document.

To solve this without AI or OCR, we use a 4-line mathematical trick:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
diff = cv2.subtract(bg, gray)
_, bw = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
```

Here is exactly how it works, line by line.

---

## 2. Line 1: The 41×41 Magnifying Glass (`kernel`)

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
```

* **What it is:** A square window or "magnifying glass" of $41\times 41$ pixels.
* **Why 41 pixels?** 
  A printed letter of text or a table grid line is typically only **2 to 5 pixels wide**. 
  A 41-pixel box is **much larger than any single letter or line** on the page. 
  That means whenever this window is placed over any piece of text, it is guaranteed to capture both the ink **and** the clean paper sitting right beside it.

---

## 3. Line 2: The Magic Eraser (`bg` via Dilation)

```python
bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
```

Forget the complex phrase *"morphological dilation"*. Here is what it does mathematically:

> The computer slides that $41\times 41$ magnifying glass over every pixel on the page. 
> Inside that box, it finds the **BRIGHTEST (highest value) pixel** and replaces the center pixel with that brightness.

### What happens when the box slides over text?
1. The black letter is dark (e.g., `50`).
2. The paper surrounding the letter is bright (e.g., `220`).
3. Because the $41\times 41$ box is wider than the letter, clean paper is always inside the box.
4. The algorithm picks the brightest value (`220`) and **paints over the dark letter**.

**Result:** The black ink is completely erased! 
What remains is a **"ghost sheet" of bare paper**—all text and table lines are gone, but all original scanner shadows, yellow aging, and lighting gradients are preserved. 

That is why this variable is named `bg` (**Background**).

---

## 4. Line 3: The Subtraction Trick (`diff`)

```python
diff = cv2.subtract(bg, gray)
```

Now we subtract the original image (`gray`, which has ink) from our bare paper map (`bg`):

$$\text{Difference} = (\text{Bare Paper}) - (\text{Original Image with Ink})$$

Look at what happens to the math:

### Case A: On Blank Paper (No Text)
$$\text{Paper} - \text{Paper} = 0 \quad (\text{Pure Black})$$
* In the dark shadow: $130 - 130 = \mathbf{0}$
* In the bright light: $240 - 240 = \mathbf{0}$

**The shadows completely cancel themselves out!**

### Case B: On Printed Ink
$$\text{Paper} - \text{Ink} = \text{A Positive Number!}$$
* In the bright zone: $240\text{ (paper)} - 80\text{ (ink)} = \mathbf{160}$
* In the dark shadow: $130\text{ (paper)} - 30\text{ (ink)} = \mathbf{100}$

Across the entire sheet, every stroke of ink now stands out with a positive contrast value between `100` and `160`.

---

## 5. Line 4: The 28 Boundary (`threshold`)

```python
_, bw = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
```

The computer now checks **every single individual pixel** on the page:

> *"Is this pixel at least 28 points darker than the paper around it?"*

* **If Difference $\ge 28$:** It is real printed ink! $\rightarrow$ Turn pixel to **$255$ (Pure White)**.
* **If Difference $< 28$:** It is just paper grain, dirt, or faint scanner dust $\rightarrow$ Turn pixel to **$0$ (Pure Black)**.

### Where does the `0` come from if it's not typed?
The `0` is hardcoded inside OpenCV's `cv2.THRESH_BINARY`. 
"Binary" means only two values exist: **ON (`255`)** or **OFF (`0`)**. Anything that does not meet the $\ge 28$ cutoff is automatically dropped to `0`.

### Why turn ink to White (`255`) and paper to Black (`0`)?
In computer vision mathematics:
* $0$ = **Void / Empty Space / OFF**
* $255$ = **Signal / Target / ON**

Line detectors and contour finders are designed to ignore zeros and track non-zero pixels. 
When paper is `0`, an empty row sums to `0`. When ink is `255`, a table line produces a massive numeric spike that is effortless for the computer to detect.

> **Important Note:** 
> This black-and-white `bw` image is **never shown to the user or to the LLM**. 
> It is an internal scratchpad/ruler used strictly to find the cut coordinates. Once the coordinates are found, the pipeline crops the **original full-color document with real ink**.

---

## 6. The 1.9 Million Pixels Scale

A standard scanned page in our pipeline ($1600 \times 1168$) contains:
$$1600 \times 1168 = \mathbf{1,868,800\text{ pixels}}$$

* The computer inspects **every single one of those 1.9 million pixels** so it never misses tiny details like the dot on the letter `i` or a decimal point.
* Because OpenCV is written in optimized C++ and uses CPU vectorization (SIMD), it processes all 1.9 million pixels in **~3 milliseconds** (0.003 seconds).

---
---

# PART 2: The Building Blocks of Mathematical Morphology

Before looking at the table detection engine, you must understand the two basic operations that power all morphological filters:

Remember: In our binary image, **White (`255`) = Object/Ink**, and **Black (`0`) = Empty Paper**.

### 1. EROSION (`cv2.MORPH_ERODE`) $\rightarrow$ The Shrinker / Sieve
* **The Rule:** The center pixel ONLY stays white if the ENTIRE kernel fits 100% inside white pixels. If even a single pixel of the kernel touches black space, the center pixel is **turned to Black (`0`)**.
* **Effect:** Anything smaller than the kernel is completely erased.

### 2. DILATION (`cv2.MORPH_DILATE`) $\rightarrow$ The Expander / Grower
* **The Rule:** If the kernel touches EVEN ONE white pixel, the center pixel is **turned to White (`255`)**.
* **Effect:** Expands white shapes outwards, fattening lines and filling nearby dark gaps.

### 3. Why Kernels Always Use ODD Numbers
Notice kernel sizes: `(3, 3)`, `(5, 1)`, `(25, 1)`, `(41, 41)`.
In computer vision, kernels are almost always odd numbers because **an odd number has a perfect, unambiguous center pixel**:

$$\text{For } 25\text{ pixels wide:} \quad \underbrace{12\text{ pixels left}}_{\text{left wing}} \quad + \quad \underbrace{\mathbf{1\text{ center pixel}}}_{\text{the target pixel}} \quad + \quad \underbrace{12\text{ pixels right}}_{\text{right wing}} \quad = \quad \mathbf{25\text{ pixels total}}$$

If the width were an even number (like 24), the center would be split awkwardly between two pixels!

---
---

# PART 3: Engine A — The Table Ladder Engine (For Open Tables)

## 1. The Puzzle: What is an "Open Table"?

In **Archetype B** logs, tables have **NO outer box and ZERO vertical divider lines**:

```
          NO     INTERVAL (m)     RECOVERY (%)
Line 0:  ──────────────────────────────────────  ▲
          1      2100.0 - 2114.8     96.0%       │ ~23px pitch
Line 1:  ──────────────────────────────────────  ▼
          2      2115.0 - 2129.8     97.0%       ▲
Line 2:  ──────────────────────────────────────  │ ~23px pitch
          3      2130.0 - 2144.8     98.0%       ▼
Line 3:  ──────────────────────────────────────  <── LAST ROW OF TABLE
                                                 
         ══════ THE WHITESPACE CANYON ═════════  (A 150px empty void!)
                                                 
         [Symbol Index Boxes / Curve Tracks]
```

Because there is no rectangular box, standard contour detectors (`cv2.findContours`) find **nothing**. 
Engine A locates the table by treating the horizontal lines like **rungs on a musical ladder**.

---

## 2. Step 1: Cropping the Table ROI

```python
roi_top = int(height * 0.15)
roi_bottom = int(height * 0.85)
roi_left = int(width * 0.35)
roi_right = int(width * 0.95)
roi_bw = bw[roi_top:roi_bottom, roi_left:roi_right]
```

We ignore the top 15% (logo), bottom 15% (footer), and left 35% (metadata). We focus purely on the right side ($35\%\text{ to }95\%$) where the table lives.

---

## 3. Step 2: The Horizontal Toothpick (`MORPH_OPEN` with `(25, 1)`)

Inside `roi_bw`, we have **both** table lines AND text numbers (`"2100.0"`, `"RECOVERY"`). 
We need to vaporize the text while keeping the horizontal lines:

```python
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
h_lines = cv2.morphologyEx(roi_bw, cv2.MORPH_OPEN, h_kernel)
```

### How `MORPH_OPEN` Works Under the Hood:
$$\mathbf{OPEN} = \mathbf{Erode}\quad \longrightarrow \quad \text{then}\quad \mathbf{Dilate}$$

1. **The 25×1 Toothpick:** It is 25 pixels wide, but strictly **1 pixel tall**.
2. **Why Height = 1 Pixel?** In a digital image, a pixel is indivisible—the thinnest possible line that can exist is 1 pixel tall. Because the kernel height is 1, even the faintest 1-pixel hairline fits 100% inside it vertically and survives!
3. **Pass 1 (Erosion):** 
   * When the $25\times 1$ stick slides over letters and numbers (`"NO"`, `"1"`, `"9"`), they are only 3 to 8 pixels wide. The stick hangs off into the black space on both sides. **They fail the test and get painted Black (`0`)!**
   * A table line is $500\text{ px}$ wide. The stick fits inside it completely. **The line stays White (`255`)!**
4. **Pass 2 (Dilation):** 
   * Expands the surviving lines back to their true original length.
   * Dead letters have $0$ white pixels left, so expanding zero still leaves zero!

**Result:** All letters and numbers dissolve into thin air. Only pure horizontal white lines remain!

---

## 4. Step 3: The Line Welder (`MORPH_CLOSE` with `(5, 1)`)

Scanned documents often have tiny microscopic cracks (1–2 pixel breaks) where ink didn't take to the paper:

```python
heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
h_lines = cv2.morphologyEx(h_lines, cv2.MORPH_CLOSE, heal_kernel)
```

### How `MORPH_CLOSE` Works (The 2-Pass Sequence):
$$\mathbf{CLOSE} = \mathbf{Dilate}\quad \longrightarrow \quad \text{then}\quad \mathbf{Erode}$$

### The "Overflow Mystery" Solved:
You might wonder: *"When Dilation expands a 302px line with a 5px kernel, doesn't it overflow the ends of the line into the black space by 2 pixels?"*

**YES, in Pass 1 it does! But Pass 2 cleans it up:**

1. **Pass 1 (Dilation):**
   * Slides the 5-pixel stick across the entire image.
   * When it reaches a 2-pixel crack inside a line, it touches white on both sides and **fills the crack with White (`255`)**!
   * At the outer ends of the line (pixel 302), it touches white and expands into pixels 303 and 304. The line temporarily overflows by 2 pixels.

2. **Pass 2 (Erosion):**
   * Slides the 5-pixel stick across the dilated image.
   * **At the outer ends (pixels 303, 304):** Beyond pixel 304 is the infinite black abyss of the empty page. The stick touches that black abyss $\rightarrow$ **Erosion shaves pixels 303 and 304 right back to Black (`0`)!** The line shrinks back to its true length of 302 pixels!
   * **Inside the crack:** The crack is now surrounded by white line on both sides. The 5-pixel stick is 100% surrounded by white and touches ZERO black pixels. **The crack stays permanently White (`255`)!**

3. **In empty black space:**
   The 5-pixel stick touches zero white pixels, so Dilation never triggers. **It never draws random lines in empty space.**

---

## 5. Step 4: Line Coordinates (`row_sums`, `peaks` & The Two Runners `groupby`)

At this point, all text has been vaporized, and `h_lines` contains only pure white horizontal lines. 
Now we must find their exact vertical Y pixel coordinates.

```python
row_sums = np.sum(h_lines, axis=1)
peaks = np.where(row_sums > 35 * 255)[0]

line_ys = []
for _, g in groupby(enumerate(peaks), lambda ix: ix[0] - ix[1]):
    line_ys.append(int(np.mean(list(map(itemgetter(1), g)))))
```

Let's break down this entire block layer by layer:

---

### A. `row_sums = np.sum(h_lines, axis=1)` (Summing across Columns)
* Remember: in NumPy, an image is a 2D matrix of shape `(Height, Width)` or `(Rows, Columns)`.
* `axis = 1` tells NumPy: *"For every horizontal row $Y$, sum up all the pixels from left to right."*
* **On empty black rows:** $\text{Sum} = 0 + 0 + 0 \dots = \mathbf{0}$.
* **On rows containing a table line:** Each white pixel is $255$. A 400-pixel wide line produces:
  $$\text{Sum} = 400 \times 255 = \mathbf{102,000}$$
This collapses our 2D image into a **1D vertical profile** of row totals.

---

### B. `peaks = np.where(row_sums > 35 * 255)[0]` (The 35-Pixel Minimum Filter)
Why `35 * 255`?
Because every white pixel is worth $255$, this check means:
> *"Does this row have at least **35 solid white pixels**?"*

* If a tiny 4-pixel speck of scanner noise survived morphology: $\text{Sum} = 4 \times 255 = 1,020 < 35 \times 255$. **It gets thrown in the trash!**
* A real table line is $300\text{–}600\text{ px}$ wide $\rightarrow$ **It passes easily!**

`np.where(...)[0]` returns the list of row numbers ($Y$ coordinates) that passed:
```python
peaks = [210, 211, 212,   233, 234,   256, 257, 258]
```

---

### C. The Problem: Lines have Thickness!
Look at `peaks = [210, 211, 212,   233, 234,   256, 257, 258]`.
In a real scan, a printed line isn't 1 pixel thin; it's usually **2 to 3 pixels thick**:
* Rows `210, 211, 212` are **NOT** three separate table lines! They are the top, middle, and bottom of **ONE line**!
* Rows `233, 234` are **ONE line**!
* Rows `256, 257, 258` are **ONE line**!

We need Python to group consecutive numbers together and find their middle pixel.

---

### D. The "Two Runners" Analogy (Why `ix[0] - ix[1]` works!)

Python has a tool called `itertools.groupby`. It bundles items together as long as they share the **exact same tag**.

How do we create a mathematical tag that stays the same for consecutive numbers, but changes when there is a gap?

Imagine **Two Runners**:
1. **Runner A (The Counter / Index):** Always takes normal $+1$ steps: $0, 1, 2, 3, 4, 5...$
2. **Runner B (The Row Values in `peaks`):** As long as numbers are consecutive, it ALSO takes $+1$ steps: $210, 211, 212...$

Now measure the **distance between the two runners** $(\text{Runner A} - \text{Runner B})$:

| Step / Index (`ix[0]`) | Row Value (`ix[1]`) | Math: `Index - Value` | What `groupby` sees |
| :---: | :---: | :---: | :--- |
| **0** | **210** | $0 - 210 = \mathbf{-210}$ | Same Group! |
| **1** | **211** | $1 - 211 = \mathbf{-210}$ | Same Group! |
| **2** | **212** | $2 - 212 = \mathbf{-210}$ | Same Group! |
| *--- GAP ---* | *--- GAP ---* | *--- JUMP ---* | |
| **3** | **233** | $3 - 233 = \mathbf{-230}$ | **NEW Group!** |
| **4** | **234** | $4 - 234 = \mathbf{-230}$ | Same Group! |
| *--- GAP ---* | *--- GAP ---* | *--- JUMP ---* | |
| **5** | **256** | $5 - 256 = \mathbf{-251}$ | **NEW Group!** |
| **6** | **257** | $6 - 257 = \mathbf{-251}$ | Same Group! |
| **7** | **258** | $7 - 258 = \mathbf{-251}$ | Same Group! |

**Because both runners move at the exact same $+1$ pace, the distance between them NEVER CHANGES (`-210`, `-230`, `-251`)!**
The moment there is a gap, the distance jumps. `groupby` uses that distance as the tag to bundle each thick line into a group `g`!

---

### E. Peeling the Onion: `line_ys.append(int(np.mean(list(map(itemgetter(1), g)))))`

Inside each group `g`, Python is holding pairs of `(counter, row_number)`:
`g = [(0, 210), (1, 211), (2, 212)]`

Here is how that one-liner unpacks from the inside out:

1. **`itemgetter(1)`:** A helper that says: *"Grab the second item at index 1 (the row number `210`, `211`, `212`) and discard the counter (`0`, `1`, `2`)."*
2. **`list(map(itemgetter(1), g)):`** Produces the clean list of row numbers: `[210, 211, 212]`.
3. **`np.mean(...):`** Calculates the mathematical average (the middle pixel):
   $$\text{mean} = \frac{210 + 211 + 212}{3} = \mathbf{211.0}$$
4. **`int(...):`** Chops off any decimal (e.g. for `[233, 234]`, mean is `233.5` $\to$ `int(233.5) = 233`).
5. **`line_ys.append(...):`** Adds `211` to our master list!

**Final Output of Step 4:**
```python
line_ys = [211, 233, 257, 280, 303]
```
All multi-pixel line clusters are now clean, single-integer Y coordinates!

---

## 6. Step 5: The Periodic Pitch Filter (`curr_ladder` vs `ladders`)

```python
if len(line_ys) >= 4:
    ladders = []
    curr_ladder = [line_ys[0]]
    for i in range(1, len(line_ys)):
        diff = line_ys[i] - line_ys[i - 1]
        if 14 <= diff <= 55:
            curr_ladder.append(line_ys[i])
        else:
            if len(curr_ladder) >= 4:
                ladders.append(curr_ladder)
            curr_ladder = [line_ys[i]]
    if len(curr_ladder) >= 4:
        ladders.append(curr_ladder)

    if ladders:
        ladder_end_y = roi_top + ladders[0][-1]
```

### Why do we need BOTH `curr_ladder` AND `ladders`?

This is a critical architectural distinction:
* **A page can have MULTIPLE ladders!**
  1. The **Core Summary Table** at the top (lines spaced ~23px apart).
  2. The **Symbol Index Box** in the middle (lines spaced ~23px apart).
  3. The **Downhole Curve Tracks** at the bottom (lines spaced ~25px apart).

If you had only ONE flat list, all those different sections would get **welded together into one giant soup**! You could never tell where the Core Table stopped and where the Symbol Box started!

Therefore:
* **`curr_ladder`** is the **temporary workbench / staging bucket** where we build ONE ladder at a time.
* **`ladders`** is the **warehouse of completed, confirmed tables**!

---

### Step-by-Step Execution:

1. **`if len(line_ys) >= 4:`**
   A core table must have a header line + at least 3 core rows ($1 + 3 = 4\text{ lines}$). If there are fewer than 4 lines on the whole right side, it cannot be a table.
2. **`curr_ladder = [line_ys[0]]:`**
   Start building our first ladder! Grab the very first line (`211`) as Rung #1.
3. **The Loop: Measuring the Distance (`diff`):**
   * From $211 \to 233$: Gap is **$22\text{ px}$** $\to$ Fits $14 \le \text{diff} \le 55$ $\to$ Add to `curr_ladder`!
   * From $233 \to 257$: Gap is **$24\text{ px}$** $\to$ Fits $\to$ Add to `curr_ladder`!
   * From $257 \to 280$: Gap is **$23\text{ px}$** $\to$ Fits $\to$ Add to `curr_ladder`!
   * From $280 \to 303$: Gap is **$23\text{ px}$** $\to$ Fits $\to$ Add to `curr_ladder`!
   `curr_ladder` is now: `[211, 233, 257, 280, 303]`.
4. **Hitting the Whitespace Canyon (Line 460):**
   * Next line is down at $Y = 460$ (the top border of the Symbol Index box).
   * Gap: $460 - 303 = \mathbf{157\text{ px}}$!
   * $157\text{ px}$ is **NOT** between 14 and 55!
   * The code hits the `else:` branch:
     - The Core Table ladder has ended!
     - It checks: `if len(curr_ladder) >= 4:` (Yes, we have 5 rungs!).
     - **Save it to the warehouse:** `ladders.append(curr_ladder)`.
     - Reset `curr_ladder = [460]` to start fresh for any possible lower ladders.
5. **The Safe Cleanup (`len(curr_ladder) >= 4` after loop):**
   If a table runs all the way to the bottom without hitting a canyon, this line ensures the finished ladder in hand gets saved to the warehouse too.
6. **Picking the Exact Cut Coordinate:**
   ```python
   # ladders[0] is strictly the Core Summary Table!
   first_ladder = ladders[0]        # [211, 233, 257, 280, 303]
   
   # [-1] grabs the last item: the bottom rung!
   bottom_rung = first_ladder[-1]   # 303
   
   # Add roi_top to convert back to full-page physical coordinates:
   ladder_end_y = roi_top + bottom_rung  # e.g. 175 + 303 = 478 px
   ```

**`ladder_end_y` is now the exact bottom of the Core Table!** 
We add a 15-pixel safety margin (`+ 15`), giving the perfect crop line that captures all core rows while leaving the Symbol Index and downhole curves behind!

---

# Summary Cheat Sheet

| Operation | Code | Plain English Meaning |
| :--- | :--- | :--- |
| **Dilation ($41\times 41$)** | `bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)` | Replaces pixels with brightest neighbor $\to$ **Erases text, leaves bare paper with shadows.** |
| **Subtraction** | `diff = cv2.subtract(bg, gray)` | $(\text{Bare Paper}) - (\text{Original with Ink}) \to$ **Cancels shadows, isolates ink.** |
| **Thresholding** | `cv2.threshold(diff, 28, 255, THRESH_BINARY)` | Difference $\ge 28 \to 255$ (White), $< 28 \to 0$ (Black). |
| **`MORPH_OPEN` ($25\times 1$)** | `cv2.morphologyEx(roi, MORPH_OPEN, (25, 1))` | Erode then Dilate $\to$ **Dissolves narrow text, preserves long horizontal lines.** |
| **`MORPH_CLOSE` ($5\times 1$)** | `cv2.morphologyEx(h_lines, MORPH_CLOSE, (5, 1))` | Dilate then Erode $\to$ **Welds 1–4px cracks in lines, cancels edge overflow.** |
| **Pitch Filter** | `14 <= diff <= 55` | Follows the ~23px periodic ladder rungs until it hits the empty canyon. |
