"""
test_all_22_edge_cases.py
Picks one PDF for each of the 22 distinct edge cases from Ergon_Core_Logs
and tests ergon_master_pipeline.py against ground-truth invariants.
"""

import os, sys, pymupdf
sys.path.append(".")
from ergon_master_pipeline import ErgonMasterPipeline

test_cases = [
    {
        "id": 1,
        "category": "Table Archetype A",
        "name": "Standard Closed Boxed Table",
        "file": "Ergon_Core_Logs/ergon_log_003_boxed_medium_10cores.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 2,
        "category": "Table Archetype B",
        "name": "Open Frameless Ladder Table (0 vertical lines)",
        "file": "Ergon_Core_Logs/ergon_log_007_open_ladder_medium_8cores.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 3,
        "category": "Scanner Tilt (+)",
        "name": "Positive Skew / Tilt (+3.5 deg)",
        "file": "Ergon_Core_Logs/ergon_log_018_tilted_plus_3_5deg.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 4,
        "category": "Scanner Tilt (-)",
        "name": "Negative Skew / Tilt (-3.6 deg)",
        "file": "Ergon_Core_Logs/ergon_log_023_tilted_minus_3_6deg.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 5,
        "category": "Rotation 90 deg",
        "name": "90 deg Clockwise (Sideways Scan)",
        "file": "Ergon_Core_Logs/ergon_log_031_rotated_90deg_cw_boxed.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 90,
    },
    {
        "id": 6,
        "category": "Inversion 180 deg",
        "name": "180 deg Inverted (Completely Upside Down)",
        "file": "Ergon_Core_Logs/ergon_log_036_inverted_180deg_boxed.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 180,
    },
    {
        "id": 7,
        "category": "Rotation 270 deg",
        "name": "270 deg Counter-Clockwise (Sideways Scan)",
        "file": "Ergon_Core_Logs/ergon_log_041_rotated_270deg_ccw_boxed.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 270,
    },
    {
        "id": 8,
        "category": "Multi-Page Placement",
        "name": "4-Page Dossier (Target on Page 4, Curves Pruned)",
        "file": "Ergon_Core_Logs/ergon_log_055_multipage_4p_target_p4.pdf",
        "expected_kept": [4],
        "expected_cut": [1, 2, 3],
        "expected_rot": 0,
    },
    {
        "id": 9,
        "category": "Multi-Page Placement",
        "name": "6-Page Dossier (Target on Page 6, Curves Pruned)",
        "file": "Ergon_Core_Logs/ergon_log_060_multipage_6p_target_p6.pdf",
        "expected_kept": [6],
        "expected_cut": [1, 2, 3, 4, 5],
        "expected_rot": 0,
    },
    {
        "id": 10,
        "category": "Table Spillover",
        "name": "Standard Single Overflow (2 Pages Kept)",
        "file": "Ergon_Core_Logs/ergon_log_061_spillover_standard_2p.pdf",
        "expected_kept": [1, 2],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 11,
        "category": "Table Spillover",
        "name": "Mid-Dossier Spillover (4p: Curves Cut, P2-P3 Kept)",
        "file": "Ergon_Core_Logs/ergon_log_066_spillover_middle_4p_p2_p3.pdf",
        "expected_kept": [2, 3],
        "expected_cut": [1, 4],
        "expected_rot": 0,
    },
    {
        "id": 12,
        "category": "Table Spillover",
        "name": "Double Cascade Spillover (3 Pages Kept)",
        "file": "Ergon_Core_Logs/ergon_log_071_spillover_double_cascade_3p.pdf",
        "expected_kept": [1, 2, 3],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 13,
        "category": "Negative Spillover",
        "name": "Symbol Only Overflow (P2 DISCARDED)",
        "file": "Ergon_Core_Logs/ergon_log_075_spillover_symbol_only_discard_2p.pdf",
        "expected_kept": [1],
        "expected_cut": [2],
        "expected_rot": 0,
    },
    {
        "id": 14,
        "category": "Margin Boundary",
        "name": "Wide Footer Margin (Spillover at 78% height)",
        "file": "Ergon_Core_Logs/ergon_log_078_spillover_wide_margin_footer_2p.pdf",
        "expected_kept": [1, 2],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 15,
        "category": "Lighting & Shadow",
        "name": "Book Spine Shadow (Illumination Division)",
        "file": "Ergon_Core_Logs/ergon_log_081_shadow_book_spine_left.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 16,
        "category": "Lighting & Shadow",
        "name": "Corner Vignette Shadow",
        "file": "Ergon_Core_Logs/ergon_log_082_shadow_corner_vignette.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 17,
        "category": "Aging & Fading",
        "name": "Faded Carbon-Copy Ink (Low Contrast)",
        "file": "Ergon_Core_Logs/ergon_log_086_faded_ink_carbon_copy.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 18,
        "category": "Aging & Fading",
        "name": "Faint Dot-Matrix Print (Thin Ruling)",
        "file": "Ergon_Core_Logs/ergon_log_087_faded_ink_dot_matrix.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 19,
        "category": "Noise & Artifacts",
        "name": "Heavy Salt-and-Pepper Noise",
        "file": "Ergon_Core_Logs/ergon_log_092_noise_salt_and_pepper_heavy.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 20,
        "category": "Noise & Artifacts",
        "name": "Photocopy Crease Lines & Yellowing",
        "file": "Ergon_Core_Logs/ergon_log_094_aging_creased_photocopy.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 21,
        "category": "Compound Case",
        "name": "Tilt (+2.5 deg) + Heavy Spine Shadow",
        "file": "Ergon_Core_Logs/ergon_log_096_compound_tilted_and_shaded.pdf",
        "expected_kept": [1],
        "expected_cut": [],
        "expected_rot": 0,
    },
    {
        "id": 22,
        "category": "Compound Case",
        "name": "5-Page Dossier (Curves, Spilled Table, Aging, Footer)",
        "file": "Ergon_Core_Logs/ergon_log_100_compound_5p_dossier_curves_spillover_aging.pdf",
        "expected_kept": [3, 4],
        "expected_cut": [1, 2, 5],
        "expected_rot": 0,
    },
]

def run_tests():
    pipeline = ErgonMasterPipeline(bottom_margin_threshold=0.78, dpi=150)
    out_dir = "test_22_edge_cases_output"
    os.makedirs(out_dir, exist_ok=True)
    
    passed_count = 0
    total_count = len(test_cases)
    
    print("\n" + "=" * 105)
    print("BENCHMARK EXECUTION: 22 REAL-WORLD EDGE CASES ACROSS ERGON_CORE_LOGS")
    print("=" * 105)
    print(f"{'#':<3} | {'Edge Case Description':<38} | {'Kept Pages':<11} | {'Cut Pages':<14} | {'Status':<7} | Result")
    print("-" * 105)
    
    for tc in test_cases:
        fpath = tc["file"]
        if not os.path.exists(fpath):
            print(f"File missing: {fpath}")
            continue
            
        case_out = os.path.join(out_dir, f"case_{tc['id']:02d}")
        res = pipeline.process_pdf(fpath, output_dir=case_out)
        
        kept_match = (res["kept_pages"] == tc["expected_kept"])
        cut_match = (res["cut_pages"] == tc["expected_cut"])
        primary_match = (res["page_results"][0]["detected_rotation"] == tc["expected_rot"])
        is_success = (res["status"] == "SUCCESS") and kept_match and cut_match and primary_match
        
        status_label = "PASS" if is_success else "FAIL"
        if is_success:
            passed_count += 1
            
        desc = f"Case {tc['id']:02d}: {tc['name']}"[:38]
        kept_str = str(res["kept_pages"])
        cut_str = str(res["cut_pages"]) if res["cut_pages"] else "None (0 cut)"
        
        print(f"{tc['id']:02d}  | {desc:<38} | {kept_str:<11} | {cut_str:<14} | {res['status']:<7} | {status_label}")
        
    print("-" * 105)
    pct = (passed_count / total_count) * 100.0
    print(f"FINAL RESULT: {passed_count} / {total_count} PASSED ({pct:.1f}% SUCCESS RATE)")
    print("=" * 105)

if __name__ == "__main__":
    run_tests()

