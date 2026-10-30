import os
import cv2
import numpy as np
import pymupdf
from itertools import groupby
from operator import itemgetter


def detect_table_cutoff(img_bgr: np.ndarray) -> int:
    """
    Universal Table Cutoff Detector
    -------------------------------
    Detects the bottom Y-coordinate of the core summary table, supporting both:
      1. Open Tables (horizontal lines only, 0 vertical dividers)
      2. Closed Grid Tables (with vertical dividers & borders)
    """
    height, width = img_bgr.shape[:2]

    # Convert to black-and-white
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # Search region: upper 75% of page, right 65% of width
    roi_top = int(height * 0.15)
    roi_bottom = int(height * 0.75)
    roi_left = int(width * 0.35)
    roi_right = int(width * 0.95)
    roi_bw = bw[roi_top:roi_bottom, roi_left:roi_right]

    # 1. Ladder Engine (Horizontal lines only)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    h_lines = cv2.morphologyEx(roi_bw, cv2.MORPH_OPEN, h_kernel)
    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    h_lines = cv2.morphologyEx(h_lines, cv2.MORPH_CLOSE, heal_kernel)

    row_sums = np.sum(h_lines, axis=1)
    peaks = np.where(row_sums > 35 * 255)[0]

    line_ys = []
    for k, g in groupby(enumerate(peaks), lambda ix: ix[0] - ix[1]):
        line_ys.append(int(np.mean(list(map(itemgetter(1), g)))))

    ladder_end_y = None
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

    # 2. Grid Box Engine (Closed tables)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    h_full = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    grid = h_full + v_lines
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    box_end_y = None
    for c in contours:
        x, y, box_w, box_h = cv2.boundingRect(c)
        if (
            x > (width * 0.35)
            and y < (height * 0.50)
            and box_w > 150
            and box_h > 100
            and box_w < (width * 0.80)
            and box_h < (height * 0.80)
        ):
            end = y + box_h
            if box_end_y is None or end > box_end_y:
                box_end_y = end

    detected = [y for y in [ladder_end_y, box_end_y] if y is not None]
    if detected:
        return max(detected) + 8
    return int(height * 0.50)


def crop_and_verify(pdf_path, output_png):
    doc = pymupdf.open(pdf_path)
    pix = doc[0].get_pixmap(dpi=150)
    img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    doc.close()

    cut_y = detect_table_cutoff(img_bgr)
    crop = img_bgr[:cut_y, :]
    cv2.imwrite(output_png, crop)
    print(f"File: {os.path.basename(pdf_path)}")
    print(f"  Original: {pix.width}x{pix.height} px | Cropped: {crop.shape[1]}x{crop.shape[0]} px | Cut Y: {cut_y} px")


if __name__ == "__main__":
    dir_path = r"C:\Users\user\Downloads\ergon-data-study\ergon-data-study\generated_test_documents\01_normal_straight"

    print("\n" + "=" * 75)
    print("TESTING UNIVERSAL CROPPER ENGINE ON TARGET DOCUMENTS")
    print("=" * 75)

    # 1. Test on synth_open_table_full.pdf
    pdf_open = os.path.join(dir_path, "synth_open_table_full.pdf")
    out_open = r"C:\Users\user\Downloads\ergon-data-study\ergon-data-study\test_crop_open_table.png"
    crop_and_verify(pdf_open, out_open)

    # 2. Test on synth_normal_11.pdf (long 25-row boxed table)
    pdf_11 = os.path.join(dir_path, "synth_normal_11.pdf")
    out_11 = r"C:\Users\user\Downloads\ergon-data-study\ergon-data-study\test_crop_normal_11.png"
    crop_and_verify(pdf_11, out_11)

    # 3. Test on synth_normal_01.pdf (standard boxed table)
    pdf_01 = os.path.join(dir_path, "synth_normal_01.pdf")
    out_01 = r"C:\Users\user\Downloads\ergon-data-study\ergon-data-study\test_crop_normal_01.png"
    crop_and_verify(pdf_01, out_01)
    print("=" * 75 + "\n")

