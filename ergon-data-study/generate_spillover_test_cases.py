import os, sys, cv2, numpy as np, pymupdf

def build_spillover_suite():
    base_dir = "ergon-data-study/generated_test_documents"
    multi_dir = os.path.join(base_dir, "04_multi_page_placement")
    normal_dir = os.path.join(base_dir, "01_normal_straight")
    
    # 1. Load components
    # Base straight 01 (header + meta + symbol index)
    doc_n1 = pymupdf.open(os.path.join(normal_dir, "synth_normal_01.pdf"))
    pix_n1 = doc_n1[0].get_pixmap(dpi=150)
    img_n1 = np.frombuffer(pix_n1.samples, dtype=np.uint8).reshape(pix_n1.height, pix_n1.width, pix_n1.n)[:, :, :3].copy()
    doc_n1.close()
    
    # Curve page from synth_multipage_01.pdf page 1
    doc_m1 = pymupdf.open(os.path.join(multi_dir, "synth_multipage_01.pdf"))
    pix_c = doc_m1[0].get_pixmap(dpi=150)
    img_curve = np.frombuffer(pix_c.samples, dtype=np.uint8).reshape(pix_c.height, pix_c.width, pix_c.n)[:, :, :3].copy()
    doc_m1.close()
    
    # Open table crop
    open_table_path = "ergon-data-study/synth_open_table_full.png"
    img_open_full = cv2.imread(open_table_path) if os.path.exists(open_table_path) else None
    
    h_page, w_page = 1168, 1600
    paper_color = [255, 255, 255] # clean paper
    
    # Extract Symbol Index block (y: 750 to 1130, full width)
    symbol_block = img_n1[750:1130, :].copy()
    
    # Helper to create a table grid with N rows
    def draw_table_grid(target_img, start_y, end_y, x1=780, x2=1432, row_h=23, col_splits=[850, 1150, 1300], start_core=1, open_ladder=False):
        # Outer border (unless open_ladder)
        if not open_ladder:
            cv2.rectangle(target_img, (x1, start_y), (x2, end_y), (0, 0, 0), 1)
            for cx in col_splits:
                cv2.line(target_img, (cx, start_y), (cx, end_y), (0, 0, 0), 1)
        
        # Draw header row
        header_h = 30
        cv2.line(target_img, (x1, start_y + header_h), (x2, start_y + header_h), (0, 0, 0), 2)
        # Header text labels
        cv2.putText(target_img, "NO", (x1 + 15, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(target_img, "INTERVAL (m)", (col_splits[0] + 50, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(target_img, "RECOVERY (%)", (col_splits[1] + 15, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        
        # Rows
        cur_y = start_y + header_h + row_h
        core_num = start_core
        while cur_y <= end_y:
            cv2.line(target_img, (x1, cur_y), (x2, cur_y), (0, 0, 0), 1)
            # Row texts
            cv2.putText(target_img, f"{core_num}", (x1 + 20, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
            depth_start = 2100.0 + (core_num - 1) * 15.5
            depth_end = depth_start + 14.8
            cv2.putText(target_img, f"{depth_start:.1f} - {depth_end:.1f}", (col_splits[0] + 35, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
            rec = 95.0 + (core_num % 5)
            cv2.putText(target_img, f"{rec:.1f}", (col_splits[1] + 45, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
            cur_y += row_h
            core_num += 1
        return core_num
    
    # Helper to save multi-page PDF
    def save_pdf(pages, output_path):
        doc = pymupdf.open()
        pdf_w = 768.0
        for p_img in pages:
            h, w = p_img.shape[:2]
            pdf_h = h * 72.0 / 150.0
            page = doc.new_page(width=pdf_w, height=pdf_h)
            rect = pymupdf.Rect(0, 0, pdf_w, pdf_h)
            success, enc = cv2.imencode(".png", p_img)
            if success:
                page.insert_image(rect, stream=enc.tobytes())
        doc.save(output_path)
        doc.close()
        print(f"[+] Saved: {output_path} ({len(pages)} pages)")

    # -------------------------------------------------------------
    # CASE 1: synth_multipage_09_spillover_table_2p.pdf (2 pages)
    # Page 1: Header + Meta + Table running down to y=1110 (no symbol index)
    # Page 2: Table continuation (y: 60 to 520) + Symbol Index (y: 600 to 980)
    # -------------------------------------------------------------
    p1 = img_n1.copy()
    # Wipe out old table and symbol index
    p1[195:1140, 780:1440] = 255
    p1[750:1140, :] = 255
    # Outer frame
    p1[30:1131, 170] = 0; p1[30:1131, 1432] = 0; p1[1131, 170:1432] = 0
    # Draw table filling down to bottom margin
    next_core = draw_table_grid(p1, start_y=195, end_y=1110, start_core=1)
    
    # Page 2: Clean background with outer frame
    p2 = np.ones((h_page, w_page, 3), dtype=np.uint8) * 255
    p2[30:1131, 170] = 0; p2[30:1131, 1432] = 0; p2[30, 170:1432] = 0; p2[1131, 170:1432] = 0
    cv2.putText(p2, "CORE LOG (CONTINUED) - CORES DESCRIBED IN THIS LOG", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    # Table continuation from top y=65 to 520
    draw_table_grid(p2, start_y=65, end_y=520, start_core=next_core)
    # Place symbol index underneath
    sym_h = symbol_block.shape[0]
    p2[580:580+sym_h, :] = symbol_block
    
    out9 = os.path.join(multi_dir, "synth_multipage_09_spillover_table_2p.pdf")
    save_pdf([p1, p2], out9)

    # -------------------------------------------------------------
    # CASE 2: synth_multipage_10_spillover_middle_4p.pdf (4 pages)
    # Page 1: Curves
    # Page 2: Header + Meta + Table running to bottom margin (Primary)
    # Page 3: Table continuation + Symbol index (Spillover)
    # Page 4: Curves
    # -------------------------------------------------------------
    out10 = os.path.join(multi_dir, "synth_multipage_10_spillover_middle_4p.pdf")
    save_pdf([img_curve, p1, p2, img_curve], out10)

    # -------------------------------------------------------------
    # CASE 3: synth_multipage_11_spillover_double_3p.pdf (3 pages)
    # Page 1: Primary (table runs to y=1110)
    # Page 2: Full-page table continuation (y: 65 to 1110)
    # Page 3: Short table continuation (y: 65 to 450) + Symbol Index
    # -------------------------------------------------------------
    p2_full = np.ones((h_page, w_page, 3), dtype=np.uint8) * 255
    p2_full[30:1131, 170] = 0; p2_full[30:1131, 1432] = 0; p2_full[30, 170:1432] = 0; p2_full[1131, 170:1432] = 0
    cv2.putText(p2_full, "CORE LOG (CONTINUED) - CORES DESCRIBED IN THIS LOG", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    next_core_p2 = draw_table_grid(p2_full, start_y=65, end_y=1110, start_core=next_core)
    
    p3_short = np.ones((h_page, w_page, 3), dtype=np.uint8) * 255
    p3_short[30:1131, 170] = 0; p3_short[30:1131, 1432] = 0; p3_short[30, 170:1432] = 0; p3_short[1131, 170:1432] = 0
    cv2.putText(p3_short, "CORE LOG (CONTINUED) - CORES DESCRIBED IN THIS LOG", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    draw_table_grid(p3_short, start_y=65, end_y=450, start_core=next_core_p2)
    p3_short[520:520+sym_h, :] = symbol_block
    
    out11 = os.path.join(multi_dir, "synth_multipage_11_spillover_double_3p.pdf")
    save_pdf([p1, p2_full, p3_short], out11)

    # -------------------------------------------------------------
    # CASE 4: synth_multipage_12_spillover_symbol_only_discard_2p.pdf (2 pages)
    # Page 1: Header + Meta + Normal Table (ends at y=550). No symbol index.
    # Page 2: Starts with SYMBOL INDEX at top (y=65). ZERO table lines!
    # User directive: Discard Page 2! Spillover should be []
    # -------------------------------------------------------------
    p1_short_table = img_n1.copy()
    p1_short_table[750:1140, :] = 255 # no symbol index on page 1
    p1_short_table[30:1131, 170] = 0; p1_short_table[30:1131, 1432] = 0; p1_short_table[1131, 170:1432] = 0
    cv2.putText(p1_short_table, "Remarks: Core summary complete. Symbol index continued on page 2.", (200, 780), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    
    p2_sym_only = np.ones((h_page, w_page, 3), dtype=np.uint8) * 255
    p2_sym_only[30:1131, 170] = 0; p2_sym_only[30:1131, 1432] = 0; p2_sym_only[30, 170:1432] = 0; p2_sym_only[1131, 170:1432] = 0
    # Place symbol index at top y=65
    p2_sym_only[65:65+sym_h, :] = symbol_block
    
    out12 = os.path.join(multi_dir, "synth_multipage_12_spillover_symbol_only_discard_2p.pdf")
    save_pdf([p1_short_table, p2_sym_only], out12)

    # -------------------------------------------------------------
    # CASE 5: synth_multipage_13_spillover_wide_margin_2p.pdf (2 pages)
    # Page 1: Has a wide bottom margin / footer (margin boundary at y=980)
    # Table hits bottom margin boundary at y=980 without Symbol Index.
    # Page 2: Table continuation at top.
    # -------------------------------------------------------------
    p1_margin = img_n1.copy()
    p1_margin[195:1140, 780:1440] = 255
    p1_margin[750:1140, :] = 255
    # Wide footer area from y=980 to 1130
    cv2.rectangle(p1_margin, (170, 980), (1432, 1131), (240, 240, 240), -1)
    cv2.putText(p1_margin, "CONFIDENTIAL & PROPRIETARY - GEOLOGICAL REPORTING FOOTER MARGIN", (450, 1050), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)
    p1_margin[30:1131, 170] = 0; p1_margin[30:1131, 1432] = 0; p1_margin[980, 170:1432] = 0; p1_margin[1131, 170:1432] = 0
    next_core_m = draw_table_grid(p1_margin, start_y=195, end_y=970, start_core=1)
    
    out13 = os.path.join(multi_dir, "synth_multipage_13_spillover_wide_margin_2p.pdf")
    save_pdf([p1_margin, p2], out13)

    # -------------------------------------------------------------
    # CASE 6: synth_multipage_14_spillover_open_ladder_2p.pdf (2 pages)
    # Archetype B: Open frameless ladder table spilling over
    # -------------------------------------------------------------
    p1_open = img_n1.copy()
    p1_open[195:1140, 780:1440] = 255
    p1_open[750:1140, :] = 255
    p1_open[30:1131, 170] = 0; p1_open[30:1131, 1432] = 0; p1_open[1131, 170:1432] = 0
    next_core_open = draw_table_grid(p1_open, start_y=195, end_y=1110, start_core=1, open_ladder=True)
    
    p2_open = np.ones((h_page, w_page, 3), dtype=np.uint8) * 255
    p2_open[30:1131, 170] = 0; p2_open[30:1131, 1432] = 0; p2_open[30, 170:1432] = 0; p2_open[1131, 170:1432] = 0
    cv2.putText(p2_open, "CORE LOG (CONTINUED) - CORES DESCRIBED IN THIS LOG", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    draw_table_grid(p2_open, start_y=65, end_y=520, start_core=next_core_open, open_ladder=True)
    p2_open[580:580+sym_h, :] = symbol_block
    
    out14 = os.path.join(multi_dir, "synth_multipage_14_spillover_open_ladder_2p.pdf")
    save_pdf([p1_open, p2_open], out14)

if __name__ == "__main__":
    build_spillover_suite()

