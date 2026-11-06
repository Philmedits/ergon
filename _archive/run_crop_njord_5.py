"""
run_crop_njord_5.py
Directly executes ErgonMasterPipeline without modifying any cropping logic.
Applies the exact, untouched Computer Vision pipeline to the 5 newly generated
composite geological core log enclosure PDFs.
"""

import os
import sys
import glob
from ergon_master_pipeline import ErgonMasterPipeline

def main():
    pipeline = ErgonMasterPipeline()

    pdf_files = sorted(glob.glob("Njord_Composite_Enclosures_5/*.pdf"))
    if not pdf_files:
        print("No PDF files found in Njord_Composite_Enclosures_5/")
        return

    output_base_dir = "Njord_Composite_Enclosures_5_Cropped"
    os.makedirs(output_base_dir, exist_ok=True)

    print("=" * 80)
    print(f"APPLYING UNTOUCHED CROPPING LOGIC TO {len(pdf_files)} NJORD COMPOSITE ENCLOSURES")
    print("=" * 80)

    summary_results = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        doc_name = os.path.splitext(filename)[0]
        case_out_dir = os.path.join(output_base_dir, doc_name)
        os.makedirs(case_out_dir, exist_ok=True)

        print(f"\n[{idx}/{len(pdf_files)}] PROCESSING: {filename}")
        print("-" * 60)

        result = pipeline.process_pdf(pdf_path, output_dir=case_out_dir)
        
        for p_res in result["page_results"]:
            summary_results.append({
                "file": filename,
                "page": p_res["page_number"],
                "rot": p_res["detected_rotation"],
                "skew": p_res["detected_skew_angle"],
                "cutoff_y": p_res["table_cutoff_y"],
                "cropped_size": p_res["cropped_size"],
                "cropped_image": p_res["cropped_image"],
                "final_pdf": result["final_cropped_pdf"]
            })

    print("\n" + "=" * 80)
    print("ALL 5 ENCLOSURES PROCESSED - CROPPING SUMMARY TABLE:")
    print("=" * 80)
    header = f"{'Filename':<48} | {'Rot':<4} | {'Skew':<6} | {'Cut Y':<6} | {'Cropped WxH':<12}"
    print(header)
    print("-" * len(header))
    for r in summary_results:
        size_str = f"{r['cropped_size'][0]}x{r['cropped_size'][1]}"
        print(f"{r['file'][:48]:<48} | {r['rot']:<4} | {r['skew']:+6.2f} | {r['cutoff_y']:<6} | {size_str:<12}")

if __name__ == "__main__":
    main()

