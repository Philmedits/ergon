import os
import io
import re
import glob
import base64
import requests
import pandas as pd
from PIL import Image
from collections import defaultdict

def main():
    print("=== STARTING ERGON CORE LOG EXTRACTION PIPELINE (GOOGLE GEMINI VISION) ===")

    # 1. Load GEMINI_API_KEY from .env
    api_key = None
    if os.path.exists('.env'):
        with open('.env', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('GEMINI_API_KEY'):
                    api_key = line.split('=', 1)[1].strip().strip('\"').strip('\'')
                    break
    
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env!")
        return
    print(f"Loaded GEMINI_API_KEY successfully ({api_key[:8]}...).")

    # 2. Load Template
    template_path = 'template_core_log.csv'
    df_temp = pd.read_csv(template_path)
    data_columns = list(df_temp.columns)
    header_line = ','.join(data_columns)
    print(f"Loaded schema template with {len(data_columns)} columns.")

    # 3. Discover Cropped Images
    crops_dir = 'cropped_images'
    all_crop_paths = sorted(glob.glob(os.path.join(crops_dir, '*.png')))
    print(f"Discovered {len(all_crop_paths)} cropped images in {crops_dir}/.")

    # 4. Group by Base Well
    well_groups = defaultdict(list)
    for p in all_crop_paths:
        fname = os.path.basename(p)
        base_match = re.match(r'(ergon_log_\d{3})', fname)
        well_key = base_match.group(1) if base_match else os.path.splitext(fname)[0]
        well_groups[well_key].append(p)
    
    print(f"Total Unique Well Dossiers: {len(well_groups)}")
    multi_page_wells = {k: v for k, v in well_groups.items() if len(v) > 1}
    print(f"Multi-Page Spillover Wells: {len(multi_page_wells)}")

    # 5. Geologist Prompt
    def build_geologist_prompt(header_line: str) -> str:
        return f"""You are a petroleum geologist digitising a scanned core log.
Extract the information in this image as CSV.
Use EXACTLY this header row, in this order:
{header_line}

Rules:
- One row per core. Repeat the well-level values (well, licence, coordinates, etc.) on every row.
- Depths in metres, recovery as a number in percent (e.g. 73.5, not 0.735).
- If an interval is printed as "2100.0 - 2114.8", set Top_Depth_m=2100.0 and Base_Driller_m=2114.8.
- Leave a field empty if it is blank or not visible in the image. Do not guess.
- Return ONLY the CSV: no explanation, no code fences."""

    # 6. Extractor function using Google Gemini Vision API
    def extract_core_data_from_image(img_path: str, model='gemini-2.5-flash') -> pd.DataFrame:
        with open(img_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        prompt = build_geologist_prompt(header_line)
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                    {"text": prompt}
                ]
            }],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096}
        }
        
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API returned error {resp.status_code}: {resp.text}")
            
        res_data = resp.json()
        csv_text = res_data['candidates'][0]['content']['parts'][0]['text'].strip()
        if csv_text.startswith('```'):
            csv_text = csv_text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        df_out = pd.read_csv(io.StringIO(csv_text))
        return df_out.reindex(columns=data_columns)

    # 7. Auditor Function
    def audit_and_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        clean_df = df.copy()
        num_cols = ['Core_No', 'Top_Depth_m', 'Base_Driller_m', 'Cut_Length_m', 'Recovery_pct']
        for c in num_cols:
            if c in clean_df.columns:
                clean_df[c] = pd.to_numeric(clean_df[c], errors='coerce')
        
        swap_mask = clean_df['Top_Depth_m'] > clean_df['Base_Driller_m']
        if swap_mask.any():
            for idx in clean_df[swap_mask].index:
                t = clean_df.loc[idx, 'Top_Depth_m']
                b = clean_df.loc[idx, 'Base_Driller_m']
                clean_df.loc[idx, 'Top_Depth_m'] = b
                clean_df.loc[idx, 'Base_Driller_m'] = t
        
        clean_df['Cut_Length_m'] = (clean_df['Base_Driller_m'] - clean_df['Top_Depth_m']).round(2)
        
        frac_mask = (clean_df['Recovery_pct'] > 0) & (clean_df['Recovery_pct'] <= 1.0)
        clean_df.loc[frac_mask, 'Recovery_pct'] *= 100.0
        
        clean_df = clean_df.sort_values(by='Core_No').reset_index(drop=True)
        return clean_df

    print("\nExtracting sample well using Live Google Gemini Vision API...")
    df_demo = extract_core_data_from_image(all_crop_paths[0])
    df_demo_clean = audit_and_clean_dataframe(df_demo)
    print("Demo extraction successful!")
    print(df_demo_clean[['Well', 'Operator', 'Country', 'Core_No', 'Top_Depth_m', 'Base_Driller_m', 'Cut_Length_m', 'Recovery_pct']])

if __name__ == '__main__':
    main()
