import os
import cv2
import numpy as np
import pymupdf


def create_full_open_table_pdf():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_pdf = os.path.join(
        script_dir, "generated_test_documents", "01_normal_straight", "synth_normal_01.pdf"
    )
    uploaded_crop_path = r"C:/Users/user/.gemini/antigravity/brain/c039128d-d0cf-4954-bc08-87051daa58d5/.user_uploaded/media_1793318019525_b096e89c.png"

    # 1. Render base document (1600 x 1168)
    doc_base = pymupdf.open(base_pdf)
    pix = doc_base[0].get_pixmap(dpi=150)
    w, h = pix.width, pix.height
    base_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, pix.n)[:, :, :3].copy()
    doc_base.close()

    # 2. Read the uploaded open table crop
    up_img = cv2.imread(uploaded_crop_path)
    up_h, up_w = up_img.shape[:2]

    # Target region for the table on the full sheet:
    # x: 785 to 1425 (width = 640)
    target_w = 640
    scale = target_w / up_w
    target_h = int(up_h * scale)

    # Resize uploaded crop to match full document scale
    up_resized = cv2.resize(up_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Erase the old boxed table on the right side of base_img
    # Fill with clean paper background
    paper_color = base_img[100, 100].tolist()
    # Table region: x from 780 to 1430, y from 195 to 750
    base_img[195:750, 780:1431] = paper_color

    # Paste the open table into the document at y=210, x=785
    paste_y = 210
    paste_x = 785
    base_img[paste_y : paste_y + target_h, paste_x : paste_x + target_w] = up_resized

    # Make sure outer frame lines at x=170 and x=1432 remain sharp
    base_img[30:1131, 170] = [0, 0, 0]
    base_img[30:1131, 1432] = [0, 0, 0]

    # Save PNG preview
    out_png = os.path.join(script_dir, "synth_open_table_full.png")
    cv2.imwrite(out_png, base_img)
    print(f"[+] Saved full composite image: {out_png}")

    # Create PDF with 150 DPI geometry
    pdf_w = 768.0
    pdf_h = h * 72.0 / 150.0

    out_pdf = os.path.join(
        script_dir, "generated_test_documents", "01_normal_straight", "synth_open_table_full.pdf"
    )
    doc_out = pymupdf.open()
    page = doc_out.new_page(width=pdf_w, height=pdf_h)
    rect = pymupdf.Rect(0, 0, pdf_w, pdf_h)

    success, enc = cv2.imencode(".png", base_img)
    if success:
        page.insert_image(rect, stream=enc.tobytes())
        doc_out.save(out_pdf)
        print(f"[+] Saved full Open Table PDF: {out_pdf}")
    doc_out.close()


if __name__ == "__main__":
    create_full_open_table_pdf()

