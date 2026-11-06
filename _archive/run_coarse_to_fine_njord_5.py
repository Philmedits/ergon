"""
run_coarse_to_fine_njord_5.py
Implements the Two-Tier Coarse-to-Fine Pipeline:
  Tier 1: Coarse upper-band isolation (top 40% of composite enclosure)
          Eliminates downhole strip logs, wireline curves, and bioturbation ink.
  Tier 2: High-precision deskewing and classical table cutoff
          (Engine A: Ladder Periodicity + Engine B: Closed Grid Boxes)
"""

import os
import glob
import cv2
import pymupdf
import numpy as np
from ergon_master_pipeline import deskew_image, detect_table_cutoff, detect_skew_angle

def process_coarse_to_fine():
    input_files = sorted(glob.glob("Njord_Composite_Enclosures_5/*.pdf"))
    if not input_files:
        print("No files found in Njord_Composite_Enclosures_5/")
        return

    output_dir = "Njord_Composite_Enclosures_5_CoarseToFine"
    os.makedirs(output_dir, exist_ok=True)
    
    artifact_dir = r"C:\Users\Larry Mosh\.gemini\antigravity\brain\917e79e7-8cc4-4f17-b9d2-e87a621eaf54"

    print("=" * 85)
    print("RUNNING TWO-TIER (COARSE-TO-FINE) CROPPING PIPELINE ON 5 COMPOSITE ENCLOSURES")
    print("=" * 85)

    summary_rows = []

    for idx, pdf_path in enumerate(input_files, 1):
        filename = os.path.basename(pdf_path)
        doc_base = os.path.splitext(filename)[0]
        case_out_dir = os.path.join(output_dir, doc_base)
        os.makedirs(case_out_dir, exist_ok=True)

        print(f"\n[{idx}/5] PROCESSING: {filename}")
        print("-" * 70)

        # 1. Render page at 150 DPI
        doc = pymupdf.open(pdf_path)
        pix = doc[0].get_pixmap(dpi=150)
        raw_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
        total_h, total_w = raw_bgr.shape[:2]
        doc.close()

        # 2. Tier 1: Coarse upper-band crop (top 40%)
        # Discards downhole strip log, 15 vertical tracks, and heavy bioturbation ink
        coarse_ratio = 0.40
        h_coarse = int(total_h * coarse_ratio)
        coarse_bgr = raw_bgr[:h_coarse, :].copy()
        print(f"  [Tier 1] Coarse Crop: Top 40% -> {total_w}x{h_coarse} px (Full was {total_w}x{total_h} px)")

        # 3. Tier 2: Sub-degree tilt deskewing on the coarse ROI
        skew_angle = detect_skew_angle(coarse_bgr)
        deskewed_bgr, corrected_angle = deskew_image(coarse_bgr, expand_canvas=True)
        print(f"  [Tier 2] Deskewing  : Detected {skew_angle:+.2f}° -> Corrected {corrected_angle:+.2f}°")

        # 4. Tier 2: Fine Table Cutoff Detection (Dual Engine: Ladder + Closed Grid Box)
        deskew_h, deskew_w = deskewed_bgr.shape[:2]
        cut_y = detect_table_cutoff(deskewed_bgr, is_primary=True)
        final_crop_h = min(cut_y, deskew_h)

        # 5. Extract final high-precision cropped section
        final_cropped_bgr = deskewed_bgr[0:final_crop_h, 0:deskew_w]
        
        # Save cropped PNG
        crop_png_path = os.path.join(case_out_dir, f"{doc_base}_FINE_CROP.png")
        cv2.imwrite(crop_png_path, final_cropped_bgr)

        # Copy to brain artifact directory for instant viewing
        art_png_path = os.path.join(artifact_dir, f"{doc_base}_FINE_CROP.png")
        cv2.imwrite(art_png_path, final_cropped_bgr)

        # Save Final Cropped PDF
        final_pdf_path = os.path.join(case_out_dir, f"{doc_base}_FINAL_CROP.pdf")
        doc_final = pymupdf.open()
        pw = 768.0
        ph = final_crop_h * 72.0 / 150.0
        pg = doc_final.new_page(width=pw, height=ph)
        rect = pymupdf.Rect(0, 0, pw, ph)
        success, enc = cv2.imencode(".png", final_cropped_bgr)
        if success:
            pg.insert_image(rect, stream=enc.tobytes())
        doc_final.save(final_pdf_path)
        doc_final.close()

        print(f"  [Result] Fine Cutoff: Y = {cut_y} px -> Saved {deskew_w}x{final_crop_h} px")
        print(f"  [Output] PNG: {crop_png_path}")
        print(f"  [Output] PDF: {final_pdf_path}")

        summary_rows.append({
            "filename": filename,
            "total_h": total_h,
            "skew": corrected_angle,
            "cut_y": cut_y,
            "crop_w": deskew_w,
            "crop_h": final_crop_h,
            "png_path": crop_png_path,
            "pdf_path": final_pdf_path,
            "art_png": art_png_path,
        })

    print("\n" + "=" * 85)
    print("COARSE-TO-FINE CROPPING SUMMARY TABLE")
    print("=" * 85)
    header = f"{'Document':<46} | {'Total H':<7} | {'Skew':<7} | {'Cut Y':<6} | {'Crop Size':<12}"
    print(header)
    print("-" * len(header))
    for s in summary_rows:
        crop_sz = f"{s['crop_w']}x{s['crop_h']}"
        print(f"{s['filename'][:46]:<46} | {s['total_h']:<7} | {s['skew']:+6.2f}° | {s['cut_y']:<6} | {crop_sz:<12}")

if __name__ == "__main__":
    process_coarse_to_fine()

