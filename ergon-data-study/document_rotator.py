"""
Automated Document Rotator & Upright Normalizer
===============================================
Dynamic, Zero-Model Document Orientation Detector & Rotator for Scanned Core Logs & Reports.

Solves 4-way rotation ambiguity (0, 90, 180, 270 degrees) using:
  1. Reading Axis Detection: Distinguishes (0/180) from (90/270) via horizontal vs. vertical line density.
  2. Bold-Font Stroke Filter: Isolates thick title/header text (ignoring thin table numbers) to find true 'Up'.
  3. High-Fidelity Rotation: Corrects the PDF and exports both upright PDFs and PNG previews.
"""

import os
import glob
import numpy as np
import pymupdf


def detect_document_orientation(pix: pymupdf.Pixmap) -> int:
    """
    Detects the current orientation angle of the document page.

    Returns:
      0   : Already upright
      90  : Rotated 90 degrees clockwise (needs 270 deg rotation)
      180 : Inverted / upside-down (needs 180 deg rotation)
      270 : Rotated 270 degrees clockwise (needs 90 deg rotation)
    """
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    gray = np.mean(arr[:, :, :3], axis=2) if pix.n >= 3 else arr[:, :, 0]
    bw = (gray < 180).astype(np.uint8)
    h, w = bw.shape

    # 1. Axis Detection: Horizontal ruling lines vs. Vertical column dividers
    diff_y = np.abs(gray[1:, :] - gray[:-1, :]) > 40
    diff_x = np.abs(gray[:, 1:] - gray[:, :-1]) > 40
    h_lines = np.sum(diff_y)
    v_lines = np.sum(diff_x)
    ratio = h_lines / (v_lines + 1.0)

    # 2. Bold-Font Stroke Thickness Filter (5x5 neighborhood density)
    pad = np.pad(bw.astype(float), 2, mode="constant")
    box = np.zeros_like(bw, dtype=float)
    for dy in range(5):
        for dx in range(5):
            box += pad[dy:dy + h, dx:dx + w]
    bold = (box > 15.0) & (bw == 1)

    top_bold = np.sum(bold[:int(h * 0.3), :])
    bot_bold = np.sum(bold[int(h * 0.7):, :])
    left_bold = np.sum(bold[:, :int(w * 0.3)])
    right_bold = np.sum(bold[:, int(w * 0.7):])

    # 3. Decision Logic
    if ratio > 1.05:
        # Horizontal lines dominate -> 0 deg or 180 deg
        if top_bold >= bot_bold:
            return 0  # Header is at top -> Upright
        else:
            return 180  # Header is at bottom -> Upside-down
    else:
        # Vertical lines dominate -> Turned sideways (90 deg or 270 deg)
        if right_bold > left_bold:
            return 90  # Header is on the right -> Turned 90 deg CW
        else:
            return 270  # Header is on the left -> Turned 270 deg CW


def rotate_pdf_upright(
    pdf_path: str,
    output_pdf: str = None,
    output_png: str = None,
    dpi: int = 150,
) -> tuple[int, int]:
    """
    Detects page orientation and rotates the PDF so that it becomes upright (0 degrees).

    Returns:
      (detected_angle, correction_angle)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    page = doc[0]

    # Fast orientation check on low-res thumbnail (50 DPI)
    check_pix = page.get_pixmap(dpi=50)
    detected_angle = detect_document_orientation(check_pix)

    # Calculate correction angle needed to reach 0 degrees upright
    correction_angle = (360 - detected_angle) % 360

    # Apply rotation directly to the PDF page
    page.set_rotation((page.rotation + correction_angle) % 360)

    # Save rotated PDF if requested
    if output_pdf:
        os.makedirs(os.path.dirname(os.path.abspath(output_pdf)), exist_ok=True)
        doc.save(output_pdf)

    # Save PNG preview if requested
    if output_png:
        os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
        high_res_pix = page.get_pixmap(dpi=dpi)
        high_res_pix.save(output_png)

    doc.close()
    return detected_angle, correction_angle


def batch_rotate_directory(input_dir: str, output_dir: str, dpi: int = 150):
    """
    Rotates all PDF files in input_dir upright and saves them to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))

    print("\n" + "=" * 80)
    print(f" BATCH ROTATING {len(pdf_files)} DOCUMENTS TO UPRIGHT (0 DEG)")
    print(f" Source: {input_dir}")
    print(f" Output: {output_dir}")
    print("=" * 80)

    results = []
    for pdf in pdf_files:
        base = os.path.splitext(os.path.basename(pdf))[0]
        out_pdf = os.path.join(output_dir, f"{base}_upright.pdf")
        out_png = os.path.join(output_dir, f"{base}_upright.png")

        detected, correction = rotate_pdf_upright(
            pdf,
            output_pdf=out_pdf,
            output_png=out_png,
            dpi=dpi,
        )
        print(f"[*] {base:<22} | Detected: {detected:>3} deg -> Rotated by: {correction:>3} deg -> UPRIGHT (0 deg)")
        results.append({
            "file": base,
            "detected": detected,
            "correction": correction,
            "out_pdf": out_pdf,
            "out_png": out_png,
        })

    print("=" * 80)
    print(f" ALL {len(pdf_files)} DOCUMENTS ROTATED SUCCESSFULLY!")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    # Target the 03_rotated_inverted directory
    candidates = [
        r"c:\Users\Larry Mosh\hackathon-projects\data_ergon\ergon-data-study\generated_test_documents\03_rotated_inverted",
        r"c:\Users\Larry Mosh\hackathon-projects\data_ergon\generated_test_documents\03_rotated_inverted",
    ]
    input_folder = next((c for c in candidates if os.path.exists(c)), candidates[0])
    output_folder = r"c:\Users\Larry Mosh\hackathon-projects\data_ergon\ergon-data-study\rotated_documents"

    batch_rotate_directory(input_folder, output_folder, dpi=150)
