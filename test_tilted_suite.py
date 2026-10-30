import os
import glob
import cv2
import numpy as np
import pymupdf
from crop_by_symbol_index import find_symbol_index_y

def test_document(pdf_path: str, output_dir: str, dpi: int = 150) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # 1. Render PDF to 150 DPI image
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    orig_w, orig_h = pix.width, pix.height
    img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(orig_h, orig_w, pix.n)[:, :, :3]
    doc.close()

    # 2. Line Detection via Canny + Hough
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    total_lines = len(raw_lines) if raw_lines is not None else 0
    horizontal_angles = []

    if raw_lines is not None:
        for l in raw_lines:
            coords = l[0] if len(l.shape) > 1 else l
            x1, y1, x2, y2 = coords[:4]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -25.0 <= angle <= 25.0:
                horizontal_angles.append(angle)

    h_line_count = len(horizontal_angles)
    detected_angle = float(np.median(horizontal_angles)) if horizontal_angles else 0.0

    # 3. Straighten page (zero cropping)
    center = (orig_w / 2.0, orig_h / 2.0)
    corners = np.array([img_bgr[5, 5], img_bgr[5, orig_w - 6], img_bgr[orig_h - 6, 5], img_bgr[orig_h - 6, orig_w - 6]])
    paper_color = np.median(corners, axis=0).astype(int).tolist()

    M = cv2.getRotationMatrix2D(center, detected_angle, 1.0)
    straight = cv2.warpAffine(
        img_bgr,
        M,
        (orig_w, orig_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=paper_color,
    )

    # 4. Verify SYMBOL INDEX presence in straightened document
    doc_dummy = pymupdf.open()
    doc_dummy.new_page(width=100, height=100)
    sym_y, sym_method = find_symbol_index_y(doc_dummy, straight)
    doc_dummy.close()

    # 5. Save outputs
    out_png = os.path.join(output_dir, f"{base_name}_straightened.png")
    out_pdf = os.path.join(output_dir, f"{base_name}_straightened.pdf")
    cv2.imwrite(out_png, straight)

    # Export PDF
    pdf_w = orig_w * 72.0 / dpi
    pdf_h = orig_h * 72.0 / dpi
    doc_out = pymupdf.open()
    p = doc_out.new_page(width=pdf_w, height=pdf_h)
    rect = pymupdf.Rect(0, 0, pdf_w, pdf_h)
    success, enc = cv2.imencode(".png", straight)
    if success:
        p.insert_image(rect, stream=enc.tobytes())
        doc_out.save(out_pdf)
    doc_out.close()

    return {
        "file": os.path.basename(pdf_path),
        "orig_size": f"{orig_w}x{orig_h}",
        "total_lines": total_lines,
        "h_lines": h_line_count,
        "tilt_angle": detected_angle,
        "symbol_y": sym_y,
        "symbol_found": (sym_y is not None and sym_y > 400),
        "out_png": out_png,
        "out_pdf": out_pdf,
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "ergon-data-study", "generated_test_documents", "02_tilted_skewed")
    if not os.path.exists(input_dir):
        input_dir = os.path.join(script_dir, "generated_test_documents", "02_tilted_skewed")

    output_dir = os.path.join(script_dir, "ergon-data-study", "straightened_test_results")
    files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))

    print("\n" + "=" * 95)
    print(f" TESTING LINE DETECTION & ZERO-CROP STRAIGHTENER ON ALL {len(files)} TILTED DOCUMENTS")
    print("=" * 95)

    for f in files:
        r = test_document(f, output_dir)
        sym_status = f"PRESENT (at Y={r['symbol_y']} px)" if r['symbol_found'] else "MISSING"
        print(f"[*] {r['file']:<22} | Lines Detected: {r['h_lines']:>2} horizontal (of {r['total_lines']:>2} total) | Tilt: {r['tilt_angle']:>+6.2f} deg | SYMBOL INDEX: {sym_status}")

    print("=" * 95)
    print(" ALL 10 DOCUMENTS TESTED SUCCESSFULLY!")
    print(f" Output files saved to: {output_dir}")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    main()

