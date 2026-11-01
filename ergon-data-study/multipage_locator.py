"""
multipage_locator.py - Classical CV Multi-Page Document Locator & Page Extractor
Identifies the primary Core Log sheet and table spillover continuation pages,
cuts/discards all unneeded pages, and returns the retained pages as a new PDF.

Strict Rules:
- ZERO AI, ZERO OCR
- ZERO CROPPING OF CONTENT, ZERO ROTATION
- Pure deterministic geometric & morphological pattern analysis.
"""

import os
import cv2
import numpy as np
import pymupdf


def extract_table_line_features(img, margin_px=40):
    """
    Extracts horizontal table row dividers on the right half of the sheet (x: 48% to 94%).
    Uses a 60px wide horizontal morphological kernel to dissolve text characters,
    small symbol boxes (which are ~35px wide), and noise.
    
    Excludes the outermost margin_px borders (to ignore outer document frame lines).
    """
    h, w = img.shape[:2]
    roi_top = margin_px
    roi_bottom = h - margin_px
    roi_left = int(0.48 * w)
    roi_right = int(0.94 * w)

    table_roi = img[roi_top:roi_bottom, roi_left:roi_right]
    gray = cv2.cvtColor(table_roi, cv2.COLOR_BGR2GRAY)
    _, bin_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 60px wide horizontal line kernel
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    h_lines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, h_kernel)

    # Project horizontally to find distinct row lines
    h_proj = np.sum(h_lines > 0, axis=1)
    line_ys = np.where(h_proj > 80)[0]

    # Cluster consecutive line pixels (prevent double counting single thick lines)
    distinct_lines = []
    for y in line_ys:
        if not distinct_lines or (y - distinct_lines[-1]) > 6:
            distinct_lines.append(y + roi_top)

    # Top region lines (Y < 35% of page height) for continuation detection
    top_lines = [y for y in distinct_lines if y < int(0.35 * h)]

    bottom_table_y = distinct_lines[-1] if distinct_lines else 0
    top_table_y = distinct_lines[0] if distinct_lines else 0

    return {
        "distinct_lines": distinct_lines,
        "line_count": len(distinct_lines),
        "top_line_count": len(top_lines),
        "top_y": top_table_y,
        "bottom_y": bottom_table_y,
    }


def analyze_page_layout(img):
    """
    Analyzes page zones using classical morphological filtering without OCR or AI.
    Returns geometric metrics: table lines, symbol boxes, metadata asymmetry.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 1. Horizontal lines across the whole page (for header and left zone check)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # 2. Lithology box extraction (Symbol Index legend boxes: clusters of small square boxes)
    box_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    box_features = cv2.morphologyEx(binary, cv2.MORPH_OPEN, box_kernel)

    # Zone 1 (Top 15%): Header title / banner divider
    z1_top = h_lines[0 : int(0.15 * h), :]
    z1_score = int(np.count_nonzero(z1_top))

    # Zone 2 (Mid-Left 50%, y: 15% to 65%): Metadata block (should have sparse lines)
    z2_left = h_lines[int(0.15 * h) : int(0.65 * h), 0 : int(0.48 * w)]
    z2_score = int(np.count_nonzero(z2_left))

    # Zone 4 (Bottom 35%, y: 65% to 95%): Symbol index area
    z4_bottom_boxes = box_features[int(0.65 * h) : int(0.95 * h), int(0.10 * w) : int(0.90 * w)]
    z4_box_score = int(np.count_nonzero(z4_bottom_boxes))

    # Dedicated right-side table features
    table_feat = extract_table_line_features(img, margin_px=40)

    # Primary page scoring:
    # Strong table presence + metadata asymmetry + header line
    primary_score = table_feat["line_count"] * 500 - (z2_score // 10)
    if z4_box_score > 500:
        primary_score += 2000

    return {
        "primary_score": primary_score,
        "table_line_count": table_feat["line_count"],
        "top_table_lines": table_feat["top_line_count"],
        "table_top_y": table_feat["top_y"],
        "table_bottom_y": table_feat["bottom_y"],
        "z4_bottom_boxes": z4_box_score,
        "page_h": h,
        "page_w": w,
    }


def locate_core_log_pages(pdf_path, bottom_margin_threshold=0.78):
    """
    Locates the Primary Core Log page and any Table Spillover continuation pages.

    Parameters:
    - pdf_path: Path to the PDF document
    - bottom_margin_threshold: Fraction of page height (default 0.78 = 78% height)
      beyond which a table is considered to have reached the bottom printable margin.

    Returns dict:
    {
        'pdf_path': str,
        'total_pages': int,
        'primary_page': int (1-based),
        'spillover_pages': list of int (1-based),
        'has_spillover': bool,
        'status': str
    }
    """
    if not os.path.exists(pdf_path):
        return {"status": "ERROR", "error": f"File not found: {pdf_path}"}

    doc = pymupdf.open(pdf_path)
    num_pages = len(doc)
    page_metrics = []

    # Analyze all pages
    for p_idx in range(num_pages):
        pix = doc[p_idx].get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
        metrics = analyze_page_layout(img)
        page_metrics.append(metrics)
    doc.close()

    # Phase 1: Identify Primary Page
    best_primary_idx = -1
    best_primary_score = -1e9
    for p_idx, m in enumerate(page_metrics):
        # Must have at least 6 distinct table lines and start below header (y > 100)
        if m["table_line_count"] >= 6 and m["table_top_y"] >= 120:
            if m["primary_score"] > best_primary_score:
                best_primary_score = m["primary_score"]
                best_primary_idx = p_idx

    # If no valid table page found
    if best_primary_idx == -1:
        return {
            "pdf_path": pdf_path,
            "total_pages": num_pages,
            "primary_page": None,
            "spillover_pages": [],
            "has_spillover": False,
            "status": "NOT_FOUND",
        }

    primary_page_num = best_primary_idx + 1
    spillover_pages = []

    # Phase 2: Check for Table Spillover
    cur_idx = best_primary_idx
    while cur_idx + 1 < num_pages:
        cur_m = page_metrics[cur_idx]
        h = cur_m["page_h"]
        margin_boundary_y = int(bottom_margin_threshold * h)

        table_hit_bottom = cur_m["table_bottom_y"] >= margin_boundary_y
        has_symbol_below = cur_m["z4_bottom_boxes"] > 1000

        # If table hit bottom margin AND there is no symbol index underneath on this page:
        if table_hit_bottom and not has_symbol_below:
            next_idx = cur_idx + 1
            next_m = page_metrics[next_idx]

            # Continuation page must have genuine table rows in its top region (>= 4 lines)
            has_table_continuation = next_m["top_table_lines"] >= 4

            if has_table_continuation:
                # Keep this page as a genuine table continuation!
                spillover_pages.append(next_idx + 1)
                cur_idx = next_idx  # check if next page ALSO hit bottom margin (double spillover)
            else:
                # Top region is empty, curve tracks, or symbol index only -> discard as directed
                break
        else:
            # Table finished cleanly on cur_page, no spillover
            break

    return {
        "pdf_path": pdf_path,
        "total_pages": num_pages,
        "primary_page": primary_page_num,
        "spillover_pages": spillover_pages,
        "has_spillover": len(spillover_pages) > 0,
        "status": "SUCCESS",
    }


def extract_needed_pages_to_pdf(pdf_path, output_path=None, bottom_margin_threshold=0.78):
    """
    Locates the Primary Core Log page and any Table Spillover continuation pages,
    cuts (discards) all unneeded pages (curves, cover sheets, discarded symbol overflow),
    and saves the retained pages into a clean, standalone PDF.

    Preserves exact vector/raster page fidelity losslessly using PyMuPDF.

    Returns dict with file paths, kept pages, and cut pages.
    """
    loc_res = locate_core_log_pages(pdf_path, bottom_margin_threshold=bottom_margin_threshold)
    if loc_res["status"] != "SUCCESS" or loc_res["primary_page"] is None:
        return loc_res

    primary = loc_res["primary_page"]
    spillovers = loc_res["spillover_pages"]
    total = loc_res["total_pages"]

    kept_pages = [primary] + spillovers
    kept_pages = sorted(list(dict.fromkeys(kept_pages)))
    cut_pages = [p for p in range(1, total + 1) if p not in kept_pages]

    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(pdf_path))
        # Default destination: ergon-data-study/extracted_core_pages/
        out_dir = os.path.join(os.path.dirname(base_dir), "extracted_core_pages")
        if not os.path.exists(out_dir):
            out_dir = os.path.join(base_dir, "extracted_core_pages")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.splitext(os.path.basename(pdf_path))[0] + "_extracted.pdf"
        output_path = os.path.join(out_dir, fname)

    # Lossless PDF page extraction
    doc = pymupdf.open(pdf_path)
    out_doc = pymupdf.open()
    for p in kept_pages:
        # PyMuPDF uses 0-based page indices
        out_doc.insert_pdf(doc, from_page=p - 1, to_page=p - 1)

    out_doc.save(output_path)
    out_doc.close()
    doc.close()

    return {
        "status": "SUCCESS",
        "source_pdf": pdf_path,
        "output_pdf": output_path,
        "total_pages_original": total,
        "total_pages_output": len(kept_pages),
        "kept_pages": kept_pages,
        "cut_pages": cut_pages,
        "primary_page": primary,
        "spillover_pages": spillovers,
    }


if __name__ == "__main__":
    import glob

    test_dir = "ergon-data-study/generated_test_documents/04_multi_page_placement"
    out_dir = "ergon-data-study/extracted_core_pages"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(test_dir, "*.pdf")))
    print(f"Executing Multi-Page Extraction on {len(files)} documents in {test_dir}:\n")
    print("=" * 95)
    for f in files:
        res = extract_needed_pages_to_pdf(f)
        fname = os.path.basename(f)
        out_fname = os.path.basename(res["output_pdf"])
        kept_str = str(res["kept_pages"])
        cut_str = str(res["cut_pages"]) if res["cut_pages"] else "None (0 cut)"
        print(f"Input:  {fname:<53}")
        print(f"        -> Kept Pages: {kept_str:<12} | Cut (Discarded) Pages: {cut_str}")
        print(f"        -> Output PDF: {out_fname} ({res['total_pages_output']} page(s))\n")
    print("=" * 95)
