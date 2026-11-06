import os
import glob
import cv2
import numpy as np
import pymupdf
from PIL import Image
from itertools import groupby
from operator import itemgetter

def analyze_and_crop(pdf_path: str, output_dir: str, dpi: int = 150):
    """
    Runs the universal section cropper on a single PDF, saves the crop,
    and returns a diagnostic dictionary.
    """
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    w, h = pix.width, pix.height
    img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, pix.n)[:, :, :3]
    pil_img = Image.frombytes("RGB", [w, h], pix.samples)

    # Search for Symbol Index location in full text
    symbol_index_y = None
    try:
        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            # block: (x0, y0, x1, y1, text, block_no, block_type)
            if "SYMBOL INDEX" in block[4].upper():
                # scale pt to pixels at 150 dpi: 150 / 72 = 2.08333
                symbol_index_y = int(block[1] * (dpi / 72.0))
                break
    except Exception:
        pass
    doc.close()

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    roi_top = int(h * 0.15)
    roi_bottom = int(h * 0.75)
    roi_left = int(w * 0.35)
    roi_right = int(w * 0.95)
    roi_bw = bw[roi_top:roi_bottom, roi_left:roi_right]

    # Engine A: Ladder Periodicity
    ladder_end_y = None
    ladder_line_count = 0
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    h_lines = cv2.morphologyEx(roi_bw, cv2.MORPH_OPEN, h_kernel)
    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    h_lines = cv2.morphologyEx(h_lines, cv2.MORPH_CLOSE, heal_kernel)

    row_sums = np.sum(h_lines, axis=1)
    peaks = np.where(row_sums > 35 * 255)[0]
    line_ys = []
    for _, g in groupby(enumerate(peaks), lambda ix: ix[0] - ix[1]):
        line_ys.append(int(np.mean(list(map(itemgetter(1), g)))))

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
            longest = max(ladders, key=len)
            ladder_line_count = len(longest)
            ladder_end_y = roi_top + longest[-1]

    # Engine B: Grid Box Contours
    box_end_y = None
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    h_full = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    table_grid = h_full + v_lines

    contours, _ = cv2.findContours(table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes_found = []
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        if (x > w * 0.35 and y < h * 0.50 and box_w > 150 and box_h > 100
            and box_w < w * 0.80 and box_h < h * 0.80):
            boxes_found.append((x, y, box_w, box_h))
            bottom = y + box_h
            if box_end_y is None or bottom > box_end_y:
                box_end_y = bottom

    detected_cuts = [y for y in [ladder_end_y, box_end_y] if y is not None]
    if detected_cuts:
        cut_y = max(detected_cuts) + 8
    else:
        cut_y = int(h * 0.50)

    # Perform crop
    cropped = pil_img.crop((0, 0, w, cut_y))
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_img_path = os.path.join(output_dir, f"{base_name}_cropped.png")
    os.makedirs(output_dir, exist_ok=True)
    cropped.save(out_img_path, quality=95)

    # Determine status
    symbol_excluded = True
    if symbol_index_y is not None:
        symbol_excluded = (cut_y < symbol_index_y)
    else:
        # If raster scan without PDF text, check visually below cut_y
        symbol_excluded = (cut_y < int(h * 0.70))

    engines_triggered = []
    if ladder_end_y is not None:
        engines_triggered.append(f"Ladder (lines={ladder_line_count}, y={ladder_end_y})")
    if box_end_y is not None:
        engines_triggered.append(f"GridBox (boxes={len(boxes_found)}, y={box_end_y})")

    return {
        "file": os.path.basename(pdf_path),
        "orig_w": w,
        "orig_h": h,
        "cut_y": cut_y,
        "crop_h": cut_y,
        "ladder_end_y": ladder_end_y,
        "box_end_y": box_end_y,
        "symbol_index_y": symbol_index_y,
        "symbol_excluded": symbol_excluded,
        "engines": " + ".join(engines_triggered) if engines_triggered else "Fallback (50%)",
        "output_path": out_img_path,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "ergon-data-study", "generated_test_documents", "01_normal_straight")
    if not os.path.exists(input_dir):
        input_dir = os.path.join(script_dir, "generated_test_documents", "01_normal_straight")

    output_dir = os.path.join(script_dir, "ergon-data-study", "crops_01_normal_straight")
    pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))

    print("\n" + "=" * 90)
    print(f" BATCH TESTING UNIVERSAL SECTION CROPPER ON ALL {len(pdf_files)} PDFs IN '01_normal_straight'")
    print("=" * 90)

    results = []
    for pdf_path in pdf_files:
        res = analyze_and_crop(pdf_path, output_dir)
        results.append(res)
        sym_str = f"y={res['symbol_index_y']}" if res['symbol_index_y'] else "Raster"
        print(f"[*] {res['file']:<26} | Orig: {res['orig_w']}x{res['orig_h']} | Cut Y: {res['cut_y']:>4} px | Symbol: {sym_str:<8} | Excluded: {res['symbol_excluded']} | {res['engines']}")

    print("\n" + "=" * 90)
    print(" SUMMARY OF BATCH RESULTS")
    print("=" * 90)
    all_passed = True
    for res in results:
        passed = res['symbol_excluded'] and (res['cut_y'] > 400)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {res['file']:<26} -> Cut Y: {res['cut_y']} px, Symbol Excluded: {res['symbol_excluded']}")

    print("-" * 90)
    print(f"TOTAL PROCESSED: {len(results)} | ALL PASSED: {all_passed}")
    print(f"Crops saved to: {output_dir}")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
