"""
Export All Cropped Core Log Images (Token-Optimized for LLMs)
============================================================
Runs the full 5-stage pipeline on all PDFs in Ergon_Core_Logs:
  1. Pre-Normalizes every page (Orientation 0 deg + Deskew).
  2. Multi-Page Pruning: Discards useless seismic curve pages, covers, and symbol-only pages.
  3. Precision Cropping: Slices horizontally right under the Core Summary Table,
     completely excluding the SYMBOL INDEX banner and downhole curves below.
  4. Saves clean, token-saving CROPPED PNG images into a single output folder.
"""

import os
import glob
import sys
import cv2
import numpy as np
import pymupdf

# Ensure data_ergon is on python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from ergon_master_pipeline import (
        pre_normalize_page,
        locate_and_prune_pages,
        detect_table_cutoff,
    )
except ImportError:
    parent_data_ergon = r"c:\Users\Larry Mosh\hackathon-projects\data_ergon"
    if parent_data_ergon not in sys.path:
        sys.path.append(parent_data_ergon)
    from ergon_master_pipeline import (
        pre_normalize_page,
        locate_and_prune_pages,
        detect_table_cutoff,
    )


def export_all_cropped_images(
    input_folder: str,
    output_folder: str,
    dpi: int = 150,
):
    """
    Processes all PDFs in input_folder and exports tight cropped data sections
    containing only the header, metadata, and core table.
    """
    os.makedirs(output_folder, exist_ok=True)
    pdf_files = sorted(glob.glob(os.path.join(input_folder, "*.pdf")))

    total = len(pdf_files)
    print("\n" + "=" * 80)
    print(f" PIPELINE: CROPPING & EXTRACTING TABLE SECTIONS FOR {total} PDF(s)")
    print(f" Input Folder  : {input_folder}")
    print(f" Output Folder : {output_folder}")
    print("=" * 80)

    success_count = 0
    total_crops_generated = 0

    for idx, pdf_path in enumerate(pdf_files, start=1):
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            doc = pymupdf.open(pdf_path)
            num_pages = len(doc)

            # Stage 1: Pre-normalize all pages (Orientation + Deskew)
            aligned_pages = []
            for p_idx in range(num_pages):
                pix = doc[p_idx].get_pixmap(dpi=dpi)
                raw_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
                clean_img, _, _ = pre_normalize_page(raw_bgr)
                aligned_pages.append(clean_img)
            doc.close()

            # Stage 2: Multi-Page Placement & Spillover Pruning
            stage2_res = locate_and_prune_pages(aligned_pages)
            kept_pages = stage2_res["kept_pages"]
            primary_p = stage2_res["primary_page"]

            # Stages 3 & 4: Section Cropping (Cut under table)
            for p_num in kept_pages:
                aligned_img = aligned_pages[p_num - 1]
                is_primary = (p_num == primary_p)
                cut_y = detect_table_cutoff(aligned_img, is_primary=is_primary)

                h_s, w_s = aligned_img.shape[:2]
                crop_h = min(cut_y, h_s)
                cropped_img = aligned_img[0:crop_h, 0:w_s]

                # Name the output cleanly
                if len(kept_pages) == 1:
                    out_name = f"{base_name}_cropped.png"
                else:
                    role = "primary" if p_num == primary_p else "spillover"
                    out_name = f"{base_name}_p{p_num}_{role}_cropped.png"

                out_path = os.path.join(output_folder, out_name)
                cv2.imwrite(out_path, cropped_img)
                total_crops_generated += 1

            success_count += 1
            print(f"[{idx:>3}/{total:>3}] OK: {base_name} ({num_pages}p) -> Kept P{kept_pages} -> Cropped image saved")

        except Exception as e:
            print(f"[{idx:>3}/{total:>3}] ERROR on {base_name}: {e}")

    print("=" * 80)
    print(f" SUCCESS: {success_count}/{total} PDFs processed.")
    print(f" Total Cropped Images Saved: {total_crops_generated}")
    print(f" Location: {output_folder}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    default_input = r"c:\Users\Larry Mosh\hackathon-projects\data_ergon\Ergon_Core_Logs"
    default_output = r"c:\Users\Larry Mosh\hackathon-projects\data_ergon\cropped_images"

    in_dir = sys.argv[1] if len(sys.argv) > 1 else default_input
    out_dir = sys.argv[2] if len(sys.argv) > 2 else default_output

    export_all_cropped_images(in_dir, out_dir, dpi=150)
