"""
Export All Normalized Images
============================
Reads PDFs from a folder (e.g., Ergon_Core_Logs), normalizes every page
(corrects 90/180/270 degree rotations to 0 upright, and straightens tilts),
and saves all normalized PNG images directly into one single output folder.
"""

import os
import glob
import cv2
import numpy as np
import pymupdf
import sys

# Ensure data_ergon is on the Python path to import pipeline modules if available
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from ergon_master_pipeline import pre_normalize_page
except ImportError:
    # If ergon_master_pipeline is in parent or sibling directory
    parent_data_ergon = r"c:\Users\Larry Mosh\hackathon-projects\data_ergon"
    if parent_data_ergon not in sys.path:
        sys.path.append(parent_data_ergon)
    from ergon_master_pipeline import pre_normalize_page


def export_normalized_images(
    input_folder: str,
    output_folder: str,
    dpi: int = 150,
    max_files: int = None,
):
    """
    Processes all PDF files in input_folder and saves normalized PNG images
    directly into output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    pdf_files = sorted(glob.glob(os.path.join(input_folder, "*.pdf")))

    if max_files:
        pdf_files = pdf_files[:max_files]

    total = len(pdf_files)
    print("\n" + "=" * 80)
    print(f" NORMALIZING {total} PDF(s)")
    print(f" Source Folder: {input_folder}")
    print(f" Target Folder: {output_folder}")
    print("=" * 80)

    success_count = 0
    for idx, pdf_path in enumerate(pdf_files, start=1):
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            doc = pymupdf.open(pdf_path)
            num_pages = len(doc)

            for p_idx in range(num_pages):
                pix = doc[p_idx].get_pixmap(dpi=dpi)
                raw_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]

                # Run Stage 1 normalization: orientation correction (0 deg) + deskew tilt
                clean_img, rot_angle, skew_angle = pre_normalize_page(raw_bgr)

                # Name the output PNG cleanly
                if num_pages == 1:
                    out_filename = f"{base_name}_normalized.png"
                else:
                    out_filename = f"{base_name}_p{p_idx + 1}_normalized.png"

                out_path = os.path.join(output_folder, out_filename)
                cv2.imwrite(out_path, clean_img)

            doc.close()
            success_count += 1
            print(f"[{idx:>3}/{total:>3}] OK: {base_name} ({num_pages}p) -> Saved normalized PNG")

        except Exception as e:
            print(f"[{idx:>3}/{total:>3}] ERROR processing {base_name}: {e}")

    print("=" * 80)
    print(f" COMPLETED: {success_count} / {total} PDFs normalized and saved into:")
    print(f" {output_folder}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Default to Ergon_Core_Logs
    default_input = r"c:\Users\Larry Mosh\hackathon-projects\data_ergon\Ergon_Core_Logs"
    default_output = r"c:\Users\Larry Mosh\hackathon-projects\data_ergon\normalized_images"

    # Allow custom arguments if provided
    in_dir = sys.argv[1] if len(sys.argv) > 1 else default_input
    out_dir = sys.argv[2] if len(sys.argv) > 2 else default_output

    export_normalized_images(in_dir, out_dir, dpi=150)
