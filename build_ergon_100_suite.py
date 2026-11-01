"""
build_ergon_100_suite.py
Generates 100 authentic synthetic Core Log PDFs in 'Ergon_Core_Logs'
spanning all 22 distinct edge cases discussed:
1. Standard Boxed & Open Ladder Tables
2. Skewed / Tilted Scans (-6 deg to +6 deg)
3. Rotated & Inverted Orientations (90, 180, 270 deg)
4. Multi-Page Placement & Curve Track Pruning (2p to 6p)
5. Table Spillover & Margin Continuations (Single, Middle, Double, Symbol Discard)
6. Shadows, Fading, Noise & Aging
7. Compound Combined Edge Cases
"""

import os, sys, cv2, numpy as np, pymupdf

def build_all():
    target_dir = os.path.abspath("Ergon_Core_Logs")
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"[+] Output directory: {target_dir}")
    
    # 1. Load Base Assets
    base_01_path = "ergon-data-study/generated_test_documents/01_normal_straight/synth_normal_01.pdf"
    doc_b1 = pymupdf.open(base_01_path)
    pix_b1 = doc_b1[0].get_pixmap(dpi=150)
    img_base = np.frombuffer(pix_b1.samples, dtype=np.uint8).reshape(pix_b1.height, pix_b1.width, pix_b1.n)[:, :, :3].copy()
    doc_b1.close()
    
    # Curve page from synth_multipage_01.pdf
    multi_01_path = "ergon-data-study/generated_test_documents/04_multi_page_placement/synth_multipage_01.pdf"
    doc_m1 = pymupdf.open(multi_01_path)
    pix_c = doc_m1[0].get_pixmap(dpi=150)
    img_curve = np.frombuffer(pix_c.samples, dtype=np.uint8).reshape(pix_c.height, pix_c.width, pix_c.n)[:, :, :3].copy()
    doc_m1.close()
    
    h_page, w_page = 1168, 1600
    symbol_block = img_base[750:1130, :].copy()
    sym_h = symbol_block.shape[0]

    # Helper: Draw Table Grid or Open Ladder
    def draw_table(target_img, start_y, end_y, x1=780, x2=1432, row_h=23, col_splits=[850, 1150, 1300], start_core=1, max_cores=None, open_ladder=False):
        if not open_ladder:
            cv2.rectangle(target_img, (x1, start_y), (x2, end_y), (0, 0, 0), 1)
            for cx in col_splits:
                cv2.line(target_img, (cx, start_y), (cx, end_y), (0, 0, 0), 1)
        
        header_h = 30
        cv2.line(target_img, (x1, start_y + header_h), (x2, start_y + header_h), (0, 0, 0), 2)
        cv2.putText(target_img, "NO", (x1 + 15, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(target_img, "INTERVAL (m)", (col_splits[0] + 50, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(target_img, "RECOVERY (%)", (col_splits[1] + 15, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        
        cur_y = start_y + header_h + row_h
        core_num = start_core
        while cur_y <= end_y:
            if max_cores is not None and (core_num - start_core) >= max_cores:
                break
            cv2.line(target_img, (x1, cur_y), (x2, cur_y), (0, 0, 0), 1)
            cv2.putText(target_img, f"{core_num}", (x1 + 20, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
            depth_start = 2100.0 + (core_num - 1) * 15.5
            depth_end = depth_start + 14.8
            cv2.putText(target_img, f"{depth_start:.1f} - {depth_end:.1f}", (col_splits[0] + 35, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
            rec = 95.0 + (core_num % 5)
            cv2.putText(target_img, f"{rec:.1f}", (col_splits[1] + 45, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
            cur_y += row_h
            core_num += 1
        return cur_y - row_h, core_num

    # Helper: Build Base Core Log Page
    def create_page(operator="NORSK HYDRO", well_id="31/2-1", cores=10, open_ladder=False, has_symbol=True, margin_floor=None):
        p = img_base.copy()
        # Wipe table & symbol region
        p[195:1140, 780:1440] = 255
        p[750:1140, :] = 255
        # Frame
        p[30:1131, 170] = 0; p[30:1131, 1432] = 0; p[1131, 170:1432] = 0; p[30, 170:1432] = 0
        
        # Operator text replace
        p[50:110, 300:600] = 255
        cv2.putText(p, operator, (300, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(p, f"WELL : {well_id}", (600, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        
        # Wide margin footer
        if margin_floor is not None:
            cv2.rectangle(p, (170, margin_floor), (1432, 1131), (240, 240, 240), -1)
            cv2.line(p, (170, margin_floor), (1432, margin_floor), (0, 0, 0), 2)
            cv2.putText(p, "CONFIDENTIAL REPORTING MARGIN - GEOLOGICAL ARCHIVE", (480, margin_floor + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)
            end_limit = margin_floor - 15
        else:
            end_limit = 1110
            
        last_y, next_core = draw_table(p, start_y=195, end_y=end_limit, max_cores=cores, open_ladder=open_ladder)
        
        # Place symbol index if requested and room available
        if has_symbol and (last_y + sym_h + 30 <= 1130):
            p[last_y + 30 : last_y + 30 + sym_h, :] = symbol_block
            
        return p, next_core

    # Helper: Build Continuation Page
    def create_continuation_page(start_core, cores=15, open_ladder=False, has_symbol=True, full_page=False):
        p = np.ones((h_page, w_page, 3), dtype=np.uint8) * 255
        p[30:1131, 170] = 0; p[30:1131, 1432] = 0; p[30, 170:1432] = 0; p[1131, 170:1432] = 0
        cv2.putText(p, "CORE LOG (CONTINUED) - CORES DESCRIBED IN THIS LOG", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        end_limit = 1110 if full_page else 550
        last_y, next_core = draw_table(p, start_y=65, end_y=end_limit, max_cores=cores, start_core=start_core, open_ladder=open_ladder)
        if has_symbol and not full_page and (last_y + sym_h + 30 <= 1130):
            p[last_y + 30 : last_y + 30 + sym_h, :] = symbol_block
        return p, next_core

    # Helper: Build Symbol Only Page
    def create_symbol_only_page():
        p = np.ones((h_page, w_page, 3), dtype=np.uint8) * 255
        p[30:1131, 170] = 0; p[30:1131, 1432] = 0; p[30, 170:1432] = 0; p[1131, 170:1432] = 0
        cv2.putText(p, "CORE LOG - LITHOLOGY SYMBOL INDEX & FORMATION REFERENCE", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        p[70 : 70 + sym_h, :] = symbol_block
        return p

    # Transformations
    def rotate_img(img, angle):
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def tilt_img(img, angle):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(img, M, (new_w, new_h), borderValue=(255, 255, 255))

    def add_shadow(img, mode="spine"):
        h, w = img.shape[:2]
        out = img.astype(np.float32)
        if mode == "spine":
            ramp = np.linspace(0.45, 1.0, int(w * 0.40))
            full_ramp = np.ones((h, w), dtype=np.float32)
            full_ramp[:, :len(ramp)] = ramp[np.newaxis, :]
            out = out * full_ramp[:, :, np.newaxis]
        elif mode == "corner":
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - w)**2 + Y**2)
            max_d = np.sqrt(w**2 + h**2)
            mask = 0.50 + 0.50 * (dist / max_d)
            out = out * mask[:, :, np.newaxis]
        elif mode == "diagonal":
            diag = np.linspace(0.50, 1.0, h + w)
            grid = np.zeros((h, w), dtype=np.float32)
            for y in range(h):
                grid[y, :] = diag[y : y + w]
            out = out * grid[:, :, np.newaxis]
        return np.clip(out, 0, 255).astype(np.uint8)

    def add_faded_noise(img, fade=0.6, noise=0.015):
        h, w = img.shape[:2]
        out = img.astype(np.float32)
        # Fade ink towards gray/yellow
        out = 255.0 - (255.0 - out) * fade
        # Salt and pepper
        n_mask = np.random.rand(h, w)
        out[n_mask < noise] = np.random.randint(40, 120)
        return np.clip(out, 0, 255).astype(np.uint8)

    # Save PDF helper (with deflate stream compression)
    def save_doc(pages, filename):
        out_path = os.path.join(target_dir, filename)
        doc = pymupdf.open()
        for p_img in pages:
            h, w = p_img.shape[:2]
            pdf_w = 768.0
            pdf_h = h * 72.0 / 150.0
            page = doc.new_page(width=pdf_w, height=pdf_h)
            rect = pymupdf.Rect(0, 0, pdf_w, pdf_h)
            success, enc = cv2.imencode(".png", p_img)
            if success:
                page.insert_image(rect, stream=enc.tobytes())
        doc.save(out_path, deflate=True, garbage=4)
        doc.close()

    print("[*] Generating 100 PDFs across all edge cases...")
    
    manifest_rows = []

    # -------------------------------------------------------------
    # BATCH 1: Files 001 - 015: Standard & Archetype Variations
    # -------------------------------------------------------------
    # 001 - 005: Boxed Tables
    configs_b1 = [
        (1, "001", "boxed_short_3cores", 3, False),
        (2, "002", "boxed_short_6cores", 6, False),
        (3, "003", "boxed_medium_10cores", 10, False),
        (4, "004", "boxed_medium_15cores", 15, False),
        (5, "005", "boxed_long_22cores", 22, False),
        (6, "006", "open_ladder_short_4cores", 4, True),
        (7, "007", "open_ladder_medium_8cores", 8, True),
        (8, "008", "open_ladder_medium_12cores", 12, True),
        (9, "009", "open_ladder_long_18cores", 18, True),
        (10, "010", "open_ladder_tight_22cores", 22, True),
        (11, "011", "operator_equinor", 8, False),
        (12, "012", "operator_shell", 10, False),
        (13, "013", "operator_totalenergies", 12, True),
        (14, "014", "operator_conocophillips", 14, False),
        (15, "015", "operator_bp", 9, True),
    ]
    ops = ["EQUINOR", "SHELL", "TOTALENERGIES", "CONOCOPHILLIPS", "BP"]
    for idx, num_str, name, c_count, is_open in configs_b1:
        op = ops[(idx - 11) % 5] if idx >= 11 else "NORSK HYDRO"
        p, _ = create_page(operator=op, well_id=f"31/2-{idx}", cores=c_count, open_ladder=is_open)
        fname = f"ergon_log_{num_str}_{name}.pdf"
        save_doc([p], fname)
        manifest_rows.append((idx, fname, "Standard & Archetypes", 1, f"Archetype {'B (Open Ladder)' if is_open else 'A (Boxed)'}, {c_count} cores, {op}"))

    # -------------------------------------------------------------
    # BATCH 2: Files 016 - 030: Skewed & Tilted Documents (-6 to +6 deg)
    # -------------------------------------------------------------
    tilts = [
        (16, "016", "tilted_plus_1_0deg", +1.0, False),
        (17, "017", "tilted_plus_2_2deg", +2.2, False),
        (18, "018", "tilted_plus_3_5deg", +3.5, False),
        (19, "019", "tilted_plus_4_8deg", +4.8, False),
        (20, "020", "tilted_plus_5_5deg", +5.5, False),
        (21, "021", "tilted_minus_1_2deg", -1.2, False),
        (22, "022", "tilted_minus_2_4deg", -2.4, False),
        (23, "023", "tilted_minus_3_6deg", -3.6, False),
        (24, "024", "tilted_minus_4_5deg", -4.5, False),
        (25, "025", "tilted_minus_5_8deg", -5.8, False),
        (26, "026", "tilted_open_ladder_plus_2_0deg", +2.0, True),
        (27, "027", "tilted_open_ladder_minus_3_0deg", -3.0, True),
        (28, "028", "tilted_long_table_plus_1_5deg", +1.5, False),
        (29, "029", "tilted_long_table_minus_2_5deg", -2.5, False),
        (30, "030", "tilted_heavy_skew_minus_6_0deg", -6.0, True),
    ]
    for idx, num_str, name, deg, is_open in tilts:
        p, _ = create_page(well_id=f"16/1-{idx}", cores=18 if "long" in name else 10, open_ladder=is_open)
        p_tilted = tilt_img(p, deg)
        fname = f"ergon_log_{num_str}_{name}.pdf"
        save_doc([p_tilted], fname)
        manifest_rows.append((idx, fname, "Skewed & Tilted", 1, f"Scanner tilt {deg:+.1f} deg, {'Open Ladder' if is_open else 'Boxed'}"))

    # -------------------------------------------------------------
    # BATCH 3: Files 031 - 045: Rotated & Inverted Orientations
    # -------------------------------------------------------------
    rots = [
        (31, "031", "rotated_90deg_cw_boxed", 90, False),
        (32, "032", "rotated_90deg_cw_open_ladder", 90, True),
        (33, "033", "rotated_90deg_cw_medium", 90, False),
        (34, "034", "rotated_90deg_cw_long", 90, False),
        (35, "035", "rotated_90deg_cw_equinor", 90, False),
        (36, "036", "inverted_180deg_boxed", 180, False),
        (37, "037", "inverted_180deg_open_ladder", 180, True),
        (38, "038", "inverted_180deg_short", 180, False),
        (39, "039", "inverted_180deg_medium", 180, False),
        (40, "040", "inverted_180deg_long", 180, False),
        (41, "041", "rotated_270deg_ccw_boxed", 270, False),
        (42, "042", "rotated_270deg_ccw_open_ladder", 270, True),
        (43, "043", "rotated_270deg_ccw_medium", 270, False),
        (44, "044", "rotated_270deg_ccw_long", 270, False),
        (45, "045", "rotated_270deg_ccw_shell", 270, False),
    ]
    for idx, num_str, name, rot_angle, is_open in rots:
        p, _ = create_page(well_id=f"25/4-{idx}", cores=20 if "long" in name else 10, open_ladder=is_open)
        p_rot = rotate_img(p, rot_angle)
        fname = f"ergon_log_{num_str}_{name}.pdf"
        save_doc([p_rot], fname)
        manifest_rows.append((idx, fname, "Rotated & Inverted", 1, f"Rotation {rot_angle} deg, {'Open Ladder' if is_open else 'Boxed'}"))

    # -------------------------------------------------------------
    # BATCH 4: Files 046 - 060: Multi-Page Placement & Curve Pruning
    # -------------------------------------------------------------
    multis = [
        (46, "046", "multipage_2p_target_p1", 2, 1, False),
        (47, "047", "multipage_2p_target_p2", 2, 2, False),
        (48, "048", "multipage_2p_open_ladder_target_p2", 2, 2, True),
        (49, "049", "multipage_3p_target_p1", 3, 1, False),
        (50, "050", "multipage_3p_target_p2", 3, 2, False),
        (51, "051", "multipage_3p_target_p3", 3, 3, False),
        (52, "052", "multipage_3p_open_ladder_target_p3", 3, 3, True),
        (53, "053", "multipage_4p_target_p2", 4, 2, False),
        (54, "054", "multipage_4p_target_p3", 4, 3, False),
        (55, "055", "multipage_4p_target_p4", 4, 4, False),
        (56, "056", "multipage_4p_open_ladder_target_p4", 4, 4, True),
        (57, "057", "multipage_5p_target_p3", 5, 3, False),
        (58, "058", "multipage_5p_target_p5", 5, 5, False),
        (59, "059", "multipage_6p_target_p4", 6, 4, False),
        (60, "060", "multipage_6p_target_p6", 6, 6, True),
    ]
    for idx, num_str, name, tot_p, tgt_p, is_open in multis:
        pages = []
        p_target, _ = create_page(well_id=f"34/10-{idx}", cores=10, open_ladder=is_open)
        for p_i in range(1, tot_p + 1):
            if p_i == tgt_p:
                pages.append(p_target)
            else:
                pages.append(img_curve)
        fname = f"ergon_log_{num_str}_{name}.pdf"
        save_doc(pages, fname)
        manifest_rows.append((idx, fname, "Multi-Page Placement", tot_p, f"Target on Page {tgt_p} of {tot_p}, curves cut"))

    # -------------------------------------------------------------
    # BATCH 5: Files 061 - 080: Table Spillover & Continuations
    # -------------------------------------------------------------
    spills = [
        (61, "061", "spillover_standard_2p", "single", False, False),
        (62, "062", "spillover_standard_short_continuation_2p", "single_short", False, False),
        (63, "063", "spillover_standard_long_continuation_2p", "single_long", False, False),
        (64, "064", "spillover_open_ladder_2p", "single", True, False),
        (65, "065", "spillover_open_ladder_long_2p", "single_long", True, False),
        (66, "066", "spillover_middle_4p_p2_p3", "mid_4p", False, False),
        (67, "067", "spillover_middle_4p_open_ladder", "mid_4p", True, False),
        (68, "068", "spillover_middle_5p_p3_p4", "mid_5p", False, False),
        (69, "069", "spillover_middle_5p_p2_p3", "mid_5p_early", False, False),
        (70, "070", "spillover_end_5p_p4_p5", "end_5p", False, False),
        (71, "071", "spillover_double_cascade_3p", "double", False, False),
        (72, "072", "spillover_double_open_ladder_3p", "double", True, False),
        (73, "073", "spillover_double_middle_5p", "double_mid_5p", False, False),
        (74, "074", "spillover_double_cascade_heavy_3p", "double_heavy", False, False),
        (75, "075", "spillover_symbol_only_discard_2p", "symbol_only", False, False),
        (76, "076", "spillover_symbol_only_discard_open_ladder_2p", "symbol_only", True, False),
        (77, "077", "spillover_symbol_only_discard_mid_4p", "symbol_only_mid", False, False),
        (78, "078", "spillover_wide_margin_footer_2p", "margin_single", False, True),
        (79, "079", "spillover_wide_margin_confidential_2p", "margin_single", False, True),
        (80, "080", "spillover_wide_margin_open_ladder_2p", "margin_single", True, True),
    ]
    for idx, num_str, name, mode, is_open, has_wide_m in spills:
        mf = 980 if has_wide_m else None
        is_spill = ("single" in mode or "double" in mode or "margin" in mode or "mid" in mode or "end" in mode) and ("symbol_only" not in mode)
        p1, n_c = create_page(well_id=f"15/9-{idx}", cores=38 if is_spill else 10, 
                              open_ladder=is_open, has_symbol=not is_spill,
                              margin_floor=mf)
        if mode in ["single", "single_short", "single_long", "margin_single"]:
            c_cnt = 8 if "short" in mode else (25 if "long" in mode else 15)
            p2, _ = create_continuation_page(start_core=n_c, cores=c_cnt, open_ladder=is_open, has_symbol=True)
            pages = [p1, p2]
            desc = "Primary P1, Spills to P2 (Table continuation kept)"
        elif mode in ["mid_4p"]:
            p2, _ = create_continuation_page(start_core=n_c, cores=15, open_ladder=is_open, has_symbol=True)
            pages = [img_curve, p1, p2, img_curve]
            desc = "Curves P1, P4 cut; Primary P2, Spills to P3"
        elif mode in ["mid_5p"]:
            p2, _ = create_continuation_page(start_core=n_c, cores=15, open_ladder=is_open, has_symbol=True)
            pages = [img_curve, img_curve, p1, p2, img_curve]
            desc = "Curves P1, P2, P5 cut; Primary P3, Spills to P4"
        elif mode in ["mid_5p_early"]:
            p2, _ = create_continuation_page(start_core=n_c, cores=15, open_ladder=is_open, has_symbol=True)
            pages = [img_curve, p1, p2, img_curve, img_curve]
            desc = "Curves P1, P4, P5 cut; Primary P2, Spills to P3"
        elif mode in ["end_5p"]:
            p2, _ = create_continuation_page(start_core=n_c, cores=15, open_ladder=is_open, has_symbol=True)
            pages = [img_curve, img_curve, img_curve, p1, p2]
            desc = "Curves P1-P3 cut; Primary P4, Spills to P5"
        elif mode in ["double", "double_heavy"]:
            p2, n_c2 = create_continuation_page(start_core=n_c, cores=40, open_ladder=is_open, has_symbol=False, full_page=True)
            p3, _ = create_continuation_page(start_core=n_c2, cores=12, open_ladder=is_open, has_symbol=True)
            pages = [p1, p2, p3]
            desc = "Double table spillover across P1, P2, P3 (All 3 kept)"
        elif mode in ["double_mid_5p"]:
            p2, n_c2 = create_continuation_page(start_core=n_c, cores=40, open_ladder=is_open, has_symbol=False, full_page=True)
            p3, _ = create_continuation_page(start_core=n_c2, cores=12, open_ladder=is_open, has_symbol=True)
            pages = [img_curve, p1, p2, p3, img_curve]
            desc = "Curves P1, P5 cut; Primary P2, Spills to P3, P4"
        elif mode in ["symbol_only"]:
            p_sym = create_symbol_only_page()
            pages = [p1, p_sym]
            desc = "Primary P1; P2 has Symbol only -> DISCARD P2!"
        elif mode in ["symbol_only_mid"]:
            p_sym = create_symbol_only_page()
            pages = [img_curve, p1, p_sym, img_curve]
            desc = "Curves P1, P4 cut; Primary P2; P3 Symbol discarded"

        fname = f"ergon_log_{num_str}_{name}.pdf"
        save_doc(pages, fname)
        manifest_rows.append((idx, fname, "Table Spillover & Continuations", len(pages), desc))

    # -------------------------------------------------------------
    # BATCH 6: Files 081 - 095: Shadows, Aging, Faded Ink & Noise
    # -------------------------------------------------------------
    fxs = [
        (81, "081", "shadow_book_spine_left", "shadow", "spine", False),
        (82, "082", "shadow_corner_vignette", "shadow", "corner", False),
        (83, "083", "shadow_heavy_dark_gradient", "shadow", "diagonal", False),
        (84, "084", "shadow_diagonal_illumination", "shadow", "diagonal", False),
        (85, "085", "shadow_open_ladder_spine", "shadow", "spine", True),
        (86, "086", "faded_ink_carbon_copy", "fade", 0.45, False),
        (87, "087", "faded_ink_dot_matrix", "fade", 0.55, False),
        (88, "088", "faded_ink_low_contrast", "fade", 0.35, False),
        (89, "089", "faded_ink_open_ladder", "fade", 0.50, True),
        (90, "090", "faded_ink_long_table", "fade", 0.40, False),
        (91, "091", "noise_salt_and_pepper_light", "noise", 0.010, False),
        (92, "092", "noise_salt_and_pepper_heavy", "noise", 0.025, False),
        (93, "093", "aging_yellowed_paper_stains", "noise", 0.015, False),
        (94, "094", "aging_creased_photocopy", "noise", 0.020, False),
        (95, "095", "aging_noisy_open_ladder", "noise", 0.020, True),
    ]
    for idx, num_str, name, f_type, param, is_open in fxs:
        p, _ = create_page(well_id=f"7/11-{idx}", cores=20 if "long" in name else 10, open_ladder=is_open)
        if f_type == "shadow":
            p_mod = add_shadow(p, mode=param)
            desc = f"Shadow effect ({param}), normalized by background division"
        elif f_type == "fade":
            p_mod = add_faded_noise(p, fade=param, noise=0.003)
            desc = f"Faded ink contrast ({param}), line healing applied"
        else:
            p_mod = add_faded_noise(p, fade=0.8, noise=param)
            desc = f"Paper aging & noise ({param}), filtered cleanly"
        fname = f"ergon_log_{num_str}_{name}.pdf"
        save_doc([p_mod], fname)
        manifest_rows.append((idx, fname, "Shadows, Fading & Noise", 1, desc))

    # -------------------------------------------------------------
    # BATCH 7: Files 096 - 100: Compound Combined Edge Cases
    # -------------------------------------------------------------
    # 096: Tilted + Shaded Table
    p96, _ = create_page(well_id="30/6-96", cores=12, open_ladder=False)
    p96_mod = add_shadow(tilt_img(p96, +2.5), mode="spine")
    fname96 = "ergon_log_096_compound_tilted_and_shaded.pdf"
    save_doc([p96_mod], fname96)
    manifest_rows.append((96, fname96, "Compound Edge Cases", 1, "Tilted +2.5 deg + Left book spine shadow"))

    # 097: Inverted 180 + Faded Noise
    p97, _ = create_page(well_id="30/6-97", cores=10, open_ladder=False)
    p97_mod = add_faded_noise(rotate_img(p97, 180), fade=0.45, noise=0.015)
    fname97 = "ergon_log_097_compound_inverted_180deg_faded_noisy.pdf"
    save_doc([p97_mod], fname97)
    manifest_rows.append((97, fname97, "Compound Edge Cases", 1, "180 deg Inverted (Upside down) + Faded ink + Noise"))

    # 098: Multi-Page with Tilted Curves and Spilled Table
    p98_1, n_c98 = create_page(well_id="30/6-98", cores=38, open_ladder=False, has_symbol=False)
    p98_2, _ = create_continuation_page(start_core=n_c98, cores=15, open_ladder=False, has_symbol=True)
    c98_tilted = tilt_img(img_curve, -1.5)
    fname98 = "ergon_log_098_compound_multipage_spillover_tilted.pdf"
    save_doc([c98_tilted, p98_1, p98_2, c98_tilted], fname98)
    manifest_rows.append((98, fname98, "Compound Edge Cases", 4, "4-page log: Tilted curves cut; Primary P2, Spilled P3 kept"))

    # 099: Open Ladder Table with Spine Shadow and Spillover
    p99_1, n_c99 = create_page(well_id="30/6-99", cores=38, open_ladder=True, has_symbol=False)
    p99_2, _ = create_continuation_page(start_core=n_c99, cores=15, open_ladder=True, has_symbol=True)
    p99_1_sh = add_shadow(p99_1, mode="spine")
    p99_2_sh = add_shadow(p99_2, mode="spine")
    fname99 = "ergon_log_099_compound_open_ladder_spillover_shaded.pdf"
    save_doc([p99_1_sh, p99_2_sh], fname99)
    manifest_rows.append((99, fname99, "Compound Edge Cases", 2, "Open Ladder Table + Spine shadow + Spillover P1->P2"))

    # 100: 5-Page Dossier: Curves, Wide Margin, Spilled Table, Aging
    p100_1, n_c100 = create_page(well_id="30/6-100", cores=32, open_ladder=False, has_symbol=False, margin_floor=980)
    p100_2, _ = create_continuation_page(start_core=n_c100, cores=14, open_ladder=False, has_symbol=True)
    p100_1_aged = add_faded_noise(p100_1, fade=0.7, noise=0.015)
    p100_2_aged = add_faded_noise(p100_2, fade=0.7, noise=0.015)
    fname100 = "ergon_log_100_compound_5p_dossier_curves_spillover_aging.pdf"
    save_doc([img_curve, img_curve, p100_1_aged, p100_2_aged, img_curve], fname100)
    manifest_rows.append((100, fname100, "Compound Edge Cases", 5, "5-page dossier: Curves cut; Wide margin floor, Spilled P3->P4 kept"))

    print(f"[+] Successfully generated all 100 PDFs!")
    
    # Write INDEX.md
    man_path = os.path.join(target_dir, "INDEX.md")
    with open(man_path, "w", encoding="utf-8") as f:
        f.write("# Ergon Core Logs: 100-Document Comprehensive Edge-Case Benchmark Suite\n\n")
        f.write("> **A definitive test collection of 100 geological well log PDFs spanning all 22 real-world edge cases.**\n\n")
        f.write("| # | Filename | Edge-Case Category | Pages | Description & Test Invariant |\n")
        f.write("| :-: | :--- | :--- | :-: | :--- |\n")
        for r in manifest_rows:
            f.write(f"| {r[0]:03d} | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} |\n")
    print(f"[+] Saved manifest: {man_path}")

if __name__ == "__main__":
    build_all()

