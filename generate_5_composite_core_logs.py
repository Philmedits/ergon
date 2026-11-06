import os
import cv2
import numpy as np
import pymupdf

def main():
    out_dir = os.path.abspath("Njord_Composite_Enclosures_5")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[+] Output directory: {out_dir}")

    # Load real components from user photo
    user_img = cv2.imread('C:/Users/Larry Mosh/.gemini/antigravity/brain/917e79e7-8cc4-4f17-b9d2-e87a621eaf54/.user_uploaded/media_1793948932207_5699dfa6.jpg')
    u_h, u_w = user_img.shape[:2]
    
    # Extract authentic Symbol Index & Bottom Strip Log
    symbol_raw = user_img[int(0.26*u_h):int(0.53*u_h), :]
    strip_raw = user_img[int(0.53*u_h):, :]

    # We will build clean high-res canvas at 1000 width
    target_w = 1000
    scale = target_w / u_w
    
    # Resize symbol and strip log to crisp matching width
    symbol_resized = cv2.resize(symbol_raw, (target_w - 40, int(symbol_raw.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
    strip_resized = cv2.resize(strip_raw, (target_w - 40, int(strip_raw.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)

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

    # Helper: Draw Core Table
    def draw_core_table(canvas, x1, y1, x2, cores_count, row_h=22, open_ladder=False, custom_rows=None):
        header_h = 28
        y2 = y1 + header_h + cores_count * row_h
        
        col_splits = [x1 + int((x2 - x1) * 0.16), x1 + int((x2 - x1) * 0.68)]
        
        # Outer border & vertical dividers
        if not open_ladder:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)
            for cx in col_splits:
                cv2.line(canvas, (cx, y1), (cx, y2), (0, 0, 0), 1)
        else:
            # Open ladder: only top and bottom caps or no side lines
            cv2.line(canvas, (x1, y1), (x2, y1), (0, 0, 0), 1)
            cv2.line(canvas, (x1, y2), (x2, y2), (0, 0, 0), 1)
            
        # Header line
        cv2.line(canvas, (x1, y1 + header_h), (x2, y1 + header_h), (0, 0, 0), 2)
        cv2.putText(canvas, "NO", (x1 + 8, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        cv2.putText(canvas, "CORED INTERVAL (m)", (col_splits[0] + 12, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        cv2.putText(canvas, "RECOVERY (%)", (col_splits[1] + 8, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        
        # Rows
        cur_y = y1 + header_h + row_h
        for c in range(1, cores_count + 1):
            cv2.line(canvas, (x1, cur_y), (x2, cur_y), (0, 0, 0), 1)
            cv2.putText(canvas, f"{c}", (x1 + 12, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
            
            if custom_rows and (c - 1) < len(custom_rows):
                int_str, rec_str = custom_rows[c - 1]
            else:
                top = 2100.0 + (c - 1) * 15.2
                base = top + 14.6
                int_str = f"{top:.1f} - {base:.1f}"
                rec_str = f"{92.0 + (c % 7):.1f}"
                
            cv2.putText(canvas, int_str, (col_splits[0] + 15, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
            cv2.putText(canvas, rec_str, (col_splits[1] + 25, cur_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
            cur_y += row_h
            
        return y2

    # Master Builder for a Full Composite Enclosure
    def build_full_enclosure(operator="NORSK HYDRO", well_id="6407/7-1", field="Njord", 
                             licence="PL 107", cores=16, row_h=22, open_ladder=False, 
                             custom_rows=None, angle=0.0):
        # Calculate dynamic canvas height to fit everything with comfortable layout
        header_h = 240
        table_x1, table_x2 = 540, 960
        table_y1 = 80
        
        # Calculate table end Y
        table_y2 = table_y1 + 28 + cores * row_h
        
        # Header bottom is max of table bottom + 25 or 240
        top_zone_h = max(240, table_y2 + 25)
        sym_h = symbol_resized.shape[0]
        strip_h = strip_resized.shape[0]
        
        total_h = top_zone_h + sym_h + strip_h + 80
        canvas = np.full((total_h, target_w, 3), 255, dtype=np.uint8)
        
        # Outer Border
        cv2.rectangle(canvas, (15, 15), (target_w - 15, total_h - 15), (0, 0, 0), 2)
        
        # Top Header Area
        cv2.putText(canvas, "ENCLOSURE 1", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(canvas, "Print Quality Original", (target_w - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        
        # Operator & Title
        cv2.putText(canvas, operator, (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2)
        cv2.putText(canvas, "CORE LOG", (table_x1 - 180, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
        cv2.putText(canvas, f"SCALE : 1 : 200", (table_x1 - 160, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        cv2.putText(canvas, f"WELL : {well_id}", (table_x1 - 170, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(canvas, "DATE : 1986, revised June 1988", (table_x1 - 185, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        
        # Separator line
        cv2.line(canvas, (30, 50), (target_w - 30, 50), (0, 0, 0), 1)
        cv2.line(canvas, (30, 145), (table_x1 - 20, 145), (0, 0, 0), 1)

        # Well Info Block (Left side)
        my = 165
        cv2.putText(canvas, f"COUNTRY     : Norway", (40, my), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
        cv2.putText(canvas, f"LICENCE     : {licence}", (40, my + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
        cv2.putText(canvas, f"FIELD       : {field}", (40, my + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
        cv2.putText(canvas, f"OWNERS      : Norsk Hydro, Statoil, Shell, Agip", (40, my + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        cv2.putText(canvas, f"COORDINATES : N 64 16' 12.3\"   E 07 43' 09.5\"", (40, my + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        cv2.putText(canvas, f"ELEVATION KB: 22m       WATER DEPTH: 282m", (40, my + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        cv2.putText(canvas, f"TOTAL DEPTH : 3262m      FORMATION  : Triassic, Red beds", (40, my + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        cv2.putText(canvas, f"GEOLOGISTS  : T. Dreyer, J.F. Boshnich", (40, my + 126), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)

        # Draw Core Table on the Right!
        table_end_y = draw_core_table(canvas, table_x1, table_y1, table_x2, cores, row_h=row_h, 
                                      open_ladder=open_ladder, custom_rows=custom_rows)
        
        # Zone 2: Paste Symbol Index
        sym_y = max(top_zone_h, table_end_y + 20)
        canvas[sym_y : sym_y + sym_h, 20 : target_w - 20] = symbol_resized
        
        # Zone 3: Paste Strip Log
        strip_y = sym_y + sym_h + 15
        canvas[strip_y : strip_y + strip_h, 20 : target_w - 20] = strip_resized
        
        # Apply Angle / Orientation if specified
        if abs(angle) > 0.01:
            canvas = rotate_img(canvas, angle)
            
        return canvas

    # Helper: Save canvas to clean PDF
    def save_pdf(img, path):
        _, png = cv2.imencode('.png', img)
        img_doc = pymupdf.open('png', png.tobytes())
        rect = img_doc[0].rect
        pdf = pymupdf.open()
        page = pdf.new_page(width=rect.width, height=rect.height)
        page.insert_image(rect, stream=png.tobytes())
        pdf.save(path)
        pdf.close()
        img_doc.close()
        print(f"  -> Generated: {os.path.basename(path)} ({rect.width:.0f} x {rect.height:.0f} pt)")

    # =========================================================================
    # GENERATE 5 DISTINCT COMPOSITE GEOLOGICAL ENCLOSURES
    # =========================================================================

    # 1. Short Table (5 Cores) — Straight Portrait
    print("[1/5] Building PDF 1: Short Table (5 Cores, Compact Height, Straight)...")
    p1 = build_full_enclosure(operator="NORSK HYDRO", well_id="6407/7-1", field="Njord", 
                              licence="PL 107", cores=5, row_h=23, angle=0.0)
    save_pdf(p1, os.path.join(out_dir, "01_njord_short_table_5cores_straight.pdf"))

    # 2. Medium Table (10 Cores) — Tilted Scan (+3.8 deg)
    print("[2/5] Building PDF 2: Medium Table (10 Cores, +3.8 deg Skew)...")
    p2 = build_full_enclosure(operator="SAGA PETROLEUM", well_id="34/4-7", field="Snorre", 
                              licence="PL 057", cores=10, row_h=22, angle=3.8)
    save_pdf(p2, os.path.join(out_dir, "02_snorre_medium_table_10cores_tilted_pos38.pdf"))

    # 3. Exact Photo Clone (16 Cores, Tall Table) — Straight Portrait
    print("[3/5] Building PDF 3: Tall Table (16 Cores, Exact Photo Clone)...")
    njord_exact_data = [
        ("2710.0 - 2724.9", "85.8"), ("2750.0 - 2770.8", "86.8"),
        ("2770.0 - 2781.4", "94.6"), ("2782.0 - 2801.4", "93.9"),
        ("2804.0 - 2821.5", "100.0"), ("2824.0 - 2854.8", "78.0"),
        ("2871.0 - 2883.5", "99.3"), ("2888.0 - 2923.0", "89.3"),
        ("2923.0 - 2951.0", "100.0"), ("2951.0 - 2970.0", "99.0"),
        ("2970.0 - 2989.7", "100.0"), ("2989.7 - 3013.5", "99.0"),
        ("3013.5 - 3044.0", "74.2"), ("3052.0 - 3082.0", "78.1"),
        ("3082.0 - 3095.0", "75.1"), ("3095.0 - 3119.0", "100.0")
    ]
    p3 = build_full_enclosure(operator="NORSK HYDRO", well_id="6407/7-1", field="Njord", 
                              licence="PL 107", cores=16, row_h=21, custom_rows=njord_exact_data, angle=0.0)
    save_pdf(p3, os.path.join(out_dir, "03_njord_tall_table_16cores_exact_replica.pdf"))

    # 4. Open Ladder Table (12 Cores, No Side Borders) — Negative Tilt (-4.2 deg)
    print("[4/5] Building PDF 4: Open Ladder Table (12 Cores, No Box, -4.2 deg Tilt)...")
    p4 = build_full_enclosure(operator="STATOIL", well_id="30/6-14", field="Oseberg", 
                              licence="PL 079", cores=12, row_h=22, open_ladder=True, angle=-4.2)
    save_pdf(p4, os.path.join(out_dir, "04_oseberg_open_ladder_12cores_tilted_neg42.pdf"))

    # 5. Dense Ultra-Tall Table (24 Cores, Micro-Spacing) — Straight Tall Enclosure
    print("[5/5] Building PDF 5: Ultra-Tall Dense Table (24 Cores, Micro-Rows)...")
    p5 = build_full_enclosure(operator="PHILLIPS PETROLEUM", well_id="2/4-14", field="Ekofisk", 
                              licence="PL 018", cores=24, row_h=16, angle=0.0)
    save_pdf(p5, os.path.join(out_dir, "05_ekofisk_ultratall_dense_24cores_micro.pdf"))

    print("\n[SUCCESS] Generated all 5 Composite Geological Enclosure PDFs in Njord_Composite_Enclosures_5/!")

if __name__ == '__main__':
    main()

