import os
import cv2
import numpy as np
import pymupdf

def main():
    suite_dir = os.path.abspath("Njord_EdgeCase_Challenge_10")
    os.makedirs(suite_dir, exist_ok=True)
    print(f"[+] Output directory for challenge suite: {suite_dir}")

    # 1. Load Base Assets
    base_01_path = "ergon-data-study/generated_test_documents/01_normal_straight/synth_normal_01.pdf"
    doc_b1 = pymupdf.open(base_01_path)
    pix_b1 = doc_b1[0].get_pixmap(dpi=150)
    img_base = np.frombuffer(pix_b1.samples, dtype=np.uint8).reshape(pix_b1.height, pix_b1.width, pix_b1.n)[:, :, :3].copy()
    doc_b1.close()

    # Load symbol block
    symbol_block = img_base[750:1130, :].copy()
    sym_h = symbol_block.shape[0]

    # Helper: Save images to multi-page or single-page PDF
    def save_pdf(pages, output_path):
        doc = pymupdf.open()
        for p_img in pages:
            _, png_bytes = cv2.imencode('.png', p_img)
            img_doc = pymupdf.open('png', png_bytes.tobytes())
            rect = img_doc[0].rect
            page = doc.new_page(width=rect.width, height=rect.height)
            page.insert_image(rect, stream=png_bytes.tobytes())
            img_doc.close()
        doc.save(output_path)
        doc.close()
        print(f"  -> Generated: {os.path.basename(output_path)} ({len(pages)} page(s))")

    # Helper: Rotate image
    def rotate_img(img, angle):
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

    # Helper: Draw Table
    def draw_table(target_img, start_y, end_y, x1=780, x2=1432, row_h=23, 
                   col_splits=[850, 1150, 1300], start_core=1, max_cores=None, 
                   open_ladder=False, custom_data=None):
        if not open_ladder:
            cv2.rectangle(target_img, (x1, start_y), (x2, end_y), (0, 0, 0), 1)
            for cx in col_splits:
                cv2.line(target_img, (cx, start_y), (cx, end_y), (0, 0, 0), 1)
        
        header_h = 30
        cv2.line(target_img, (x1, start_y + header_h), (x2, start_y + header_h), (0, 0, 0), 2)
        cv2.putText(target_img, "NO", (x1 + 15, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(target_img, "CORED INTERVAL (m)", (col_splits[0] + 30, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(target_img, "RECOVERY (%)", (col_splits[1] + 15, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        
        cur_y = start_y + header_h + row_h
        core_num = start_core
        idx = 0
        while cur_y <= end_y:
            if max_cores is not None and (core_num - start_core) >= max_cores:
                break
            cv2.line(target_img, (x1, cur_y), (x2, cur_y), (0, 0, 0), 1)
            cv2.putText(target_img, f"{core_num}", (x1 + 20, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
            
            if custom_data and idx < len(custom_data):
                d_str, rec_str = custom_data[idx]
                cv2.putText(target_img, d_str, (col_splits[0] + 35, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
                cv2.putText(target_img, rec_str, (col_splits[1] + 45, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
            else:
                depth_start = 2100.0 + (core_num - 1) * 15.5
                depth_end = depth_start + 14.8
                cv2.putText(target_img, f"{depth_start:.1f} - {depth_end:.1f}", (col_splits[0] + 35, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
                rec = 95.0 + (core_num % 5)
                cv2.putText(target_img, f"{rec:.1f}", (col_splits[1] + 45, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
            
            cur_y += row_h
            core_num += 1
            idx += 1
        return cur_y - row_h, core_num

    # Helper: Base Page Builder
    def create_page(operator="NORSK HYDRO", well_id="6407/7-1", field="Njord", licence="PL 107", 
                    cores=16, open_ladder=False, has_symbol=True, custom_data=None, left_table=False):
        p = img_base.copy()
        # Wipe table & symbol region
        p[195:1140, 780:1440] = 255
        p[750:1140, :] = 255
        # Frame
        p[30:1131, 170] = 0; p[30:1131, 1432] = 0; p[1131, 170:1432] = 0; p[30, 170:1432] = 0
        
        # Header text
        p[50:180, 200:1400] = 255
        cv2.putText(p, "ENCLOSURE 1", (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)
        cv2.putText(p, operator, (220, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
        cv2.putText(p, "CORE LOG", (650, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 0), 2)
        cv2.putText(p, f"WELL: {well_id}", (670, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(p, "DATE: 1986, revised June 1988", (950, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.line(p, (180, 135), (1420, 135), (0, 0, 0), 2)

        # Meta info
        meta_y = 195
        if not left_table:
            meta_x = 220
            t_x1, t_x2 = 780, 1432
            splits = [850, 1150, 1300]
        else:
            meta_x = 850
            t_x1, t_x2 = 180, 800
            splits = [250, 550, 700]

        cv2.putText(p, f"COUNTRY: Norway", (meta_x, meta_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.putText(p, f"LICENCE: {licence}", (meta_x, meta_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.putText(p, f"FIELD: {field}", (meta_x, meta_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.putText(p, "OWNERS: Norsk Hydro, Statoil, Shell, Agip", (meta_x, meta_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.putText(p, "COORDINATES: N 64 16' 12.3\"   E 07 43' 09.5\"", (meta_x, meta_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.putText(p, "ELEVATION KB: 22m   WATER DEPTH: 282m", (meta_x, meta_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.putText(p, "TOTAL DEPTH: 3262m   FORMATION: Triassic, Red beds", (meta_x, meta_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

        last_y, next_core = draw_table(p, start_y=195, end_y=720, x1=t_x1, x2=t_x2, 
                                       col_splits=splits, max_cores=cores, 
                                       open_ladder=open_ladder, custom_data=custom_data)
        
        if has_symbol and (last_y + sym_h + 30 <= 1130):
            p[last_y + 30 : last_y + 30 + sym_h, :] = symbol_block
            
        return p, next_core

    # --- CASE 1: The Exact Njord Archetype (Mirroring user's photo!) ---
    # Custom exact data from the uploaded photo
    njord_photo_data = [
        ("2710.0 - 2724.9", "85.8"),
        ("2750.0 - 2770.8", "86.8"),
        ("2770.0 - 2781.4", "94.6"),
        ("2782.0 - 2801.4", "93.9"),
        ("2804.0 - 2821.5", "100.0"),
        ("2824.0 - 2854.8", "78.0"),
        ("2871.0 - 2883.5", "99.3"),
        ("2888.0 - 2923.0", "89.3"),
        ("2923.0 - 2951.0", "100.0"),
        ("2951.0 - 2970.0", "99.0"),
        ("2970.0 - 2989.7", "100.0"),
        ("2989.7 - 3013.5", "99.0"),
        ("3013.5 - 3044.0", "74.2"),
        ("3052.0 - 3082.0", "78.1"),
        ("3082.0 - 3095.0", "75.1"),
        ("3095.0 - 3119.0", "100.0")
    ]
    p1, _ = create_page(operator="NORSK HYDRO", well_id="6407/7-1", field="Njord", licence="PL 107", 
                        cores=16, custom_data=njord_photo_data)
    save_pdf([p1], os.path.join(suite_dir, "01_njord_archetype_classic_16cores.pdf"))

    # --- CASE 2: The Severe Tilt (+7.5 deg scan) ---
    p2, _ = create_page(operator="EQUINOR", well_id="6407/7-2", field="Njord", cores=12)
    p2_tilted = rotate_img(p2, 7.5)
    save_pdf([p2_tilted], os.path.join(suite_dir, "02_njord_severe_tilt_pos7deg.pdf"))

    # --- CASE 3: Ancient Photocopy Noise & Faded Ink ---
    p3, _ = create_page(operator="SAGA PETROLEUM", well_id="6407/7-3", field="Njord", cores=10)
    # Add gaussian noise & fade
    noise = np.random.normal(0, 15, p3.shape).astype(np.int16)
    p3_noisy = np.clip(p3.astype(np.int16) + noise + 20, 0, 255).astype(np.uint8)
    save_pdf([p3_noisy], os.path.join(suite_dir, "03_njord_photocopy_grain_faded_ink.pdf"))

    # --- CASE 4: The Unboxed Open Ladder (Zero vertical lines) ---
    p4, _ = create_page(operator="NORSK HYDRO", well_id="6407/7-4", field="Njord", cores=14, open_ladder=True)
    save_pdf([p4], os.path.join(suite_dir, "04_njord_open_ladder_no_box.pdf"))

    # --- CASE 5: Dense Micro-Rows (24 Cores tightly packed) ---
    p5 = img_base.copy()
    p5[195:1140, :] = 255
    p5[30:1131, 170] = 0; p5[30:1131, 1432] = 0; p5[1131, 170:1432] = 0; p5[30, 170:1432] = 0
    cv2.putText(p5, "WELL: 6407/7-5 (MICRO-SPACING TEST)", (220, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(p5, "OPERATOR: SHELL EXPRO", (220, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    draw_table(p5, start_y=160, end_y=760, row_h=18, max_cores=24)
    p5[770:770+sym_h, :] = symbol_block
    save_pdf([p5], os.path.join(suite_dir, "05_njord_dense_micro_rows_24cores.pdf"))

    # --- CASE 6: Inverted Layout (Left-Hand Table, Right-Hand Metadata) ---
    p6, _ = create_page(operator="TOTALENERGIES", well_id="6407/7-6", field="Njord", cores=12, left_table=True)
    save_pdf([p6], os.path.join(suite_dir, "06_njord_inverted_left_hand_table.pdf"))

    # --- CASE 7: 3-Page Mega Spillover Dossier ---
    # Page 1: Cores 1-12
    p7_1, next_c = create_page(operator="NORSK HYDRO", well_id="6407/7-7", field="Njord", cores=12)
    # Page 2: Cores 13-24
    p7_2 = img_base.copy()
    p7_2[150:1140, :] = 255
    cv2.putText(p7_2, "WELL: 6407/7-7 (CONTINUATION PAGE 2 OF 3)", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    _, next_c2 = draw_table(p7_2, start_y=160, end_y=600, start_core=next_c, max_cores=12)
    # Page 3: Cores 25-32 + TD completion note
    p7_3 = img_base.copy()
    p7_3[150:1140, :] = 255
    cv2.putText(p7_3, "WELL: 6407/7-7 (FINAL SPILLOVER PAGE 3 OF 3)", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    last_y3, _ = draw_table(p7_3, start_y=160, end_y=450, start_core=next_c2, max_cores=8)
    cv2.putText(p7_3, "TOTAL DEPTH REACHED: 3450.0m KB - WELL COMPLETED AS OIL PRODUCER", (200, last_y3 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    save_pdf([p7_1, p7_2, p7_3], os.path.join(suite_dir, "07_njord_multipage_spillover_3pages.pdf"))

    # --- CASE 8: Official Archive Stamp & Annotation Overprint ---
    p8, _ = create_page(operator="CONOCOPHILLIPS", well_id="6407/7-8", field="Njord", cores=10)
    # Draw a big diagonal red stamp across the table
    overlay = p8.copy()
    cv2.putText(overlay, "CONFIDENTIAL / NAPE 2026", (350, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 180), 4)
    cv2.putText(overlay, "OFFSHORE ARCHIVE - DO NOT COPY", (320, 520), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 3)
    p8 = cv2.addWeighted(overlay, 0.65, p8, 0.35, 0)
    save_pdf([p8], os.path.join(suite_dir, "08_njord_stamp_confidential_overlay.pdf"))

    # --- CASE 9: Heavy Top-Left Corner Scanner Shadow ---
    p9, _ = create_page(operator="BP NORGE", well_id="6407/7-9", field="Njord", cores=12)
    # Gradient shadow
    h, w = p9.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(X**2 + Y**2)
    shadow_mask = np.clip(1.0 - (dist / 700.0) * 0.7, 0.3, 1.0)
    p9 = (p9.astype(np.float32) * shadow_mask[:, :, np.newaxis]).astype(np.uint8)
    save_pdf([p9], os.path.join(suite_dir, "09_njord_heavy_corner_shadow.pdf"))

    # --- CASE 10: The Grand Slam Compound Nightmare ---
    # Tilted -5.2 deg + Open Ladder + Salt-and-pepper noise
    p10, _ = create_page(operator="NORSK HYDRO", well_id="6407/7-10", field="Njord", cores=15, open_ladder=True)
    p10_noisy = np.clip(p10.astype(np.int16) + np.random.normal(0, 12, p10.shape).astype(np.int16), 0, 255).astype(np.uint8)
    p10_compound = rotate_img(p10_noisy, -5.2)
    save_pdf([p10_compound], os.path.join(suite_dir, "10_njord_grand_slam_compound_nightmare.pdf"))

    print("\n[SUCCESS] All 10 Challenge PDFs created successfully in Njord_EdgeCase_Challenge_10/!")

if __name__ == '__main__':
    main()

