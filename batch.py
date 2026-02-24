"""Batch runner aligned with the current app.py pipeline.

Pipeline order per protein:
1) Structure search (with auto-retry on prep/ramplot failure)
2) Ramachandran analysis (with automatic SWISS-MODEL fallback)
3) Protein preparation 
4) Binding-site prediction (P2Rank + Fpocket)
5) Ligand analysis (Supports both Protein subdirectories OR flat global folder)
6) Molecular docking
7) ADMET
8) Optional DFT batch

This backend stores every key CSV and generated screenshot under:
<root>/<protein>/<step_folder>/...
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from admet_analysis import run_admet_prediction
from config import DOCKING_RESULTS_DIR, LIGAND_DIR, PRANKWEB_OUTPUT_DIR, RAMPLOT_OUTPUT_DIR, current_pdb_info
from docking import run_molecular_docking
from ligand_analysis import run_ligand_classification
from prankweb import run_prankweb_prediction
from ramachandran import run_ramplot
from utils import find_best_pdb_structure
from visualization import show_structure

PIPELINE_STEPS = [
    ("01_structure_search", "Structure Search"),
    ("02_ramachandran", "Ramachandran Analysis"),
    ("03_protein_preparation", "Protein Preparation"),
    ("04_binding_sites", "Binding Site Prediction"),
    ("05_ligand_analysis", "Ligand Analysis"),
    ("06_docking", "Molecular Docking"),
    ("07_admet", "ADMET Analysis"),
    ("08_dft", "DFT Analysis"),
]


@contextlib.contextmanager
def patch_ligand_dir(new_dir: str):
    """
    Temporarily overrides the global LIGAND_DIR across all modules so they 
    only process ligands belonging to the specific directory structure context.
    """
    import config
    import docking
    import ligand_analysis
    import admet_analysis
    
    # Store original paths
    old_config = getattr(config, 'LIGAND_DIR', None)
    old_docking = getattr(docking, 'LIGAND_DIR', None)
    old_admet = getattr(admet_analysis, 'LIGAND_DIR', None)
    
    # Patch with specific protein subdirectory or base directory
    if hasattr(config, 'LIGAND_DIR'): config.LIGAND_DIR = new_dir
    if hasattr(docking, 'LIGAND_DIR'): docking.LIGAND_DIR = new_dir
    if hasattr(admet_analysis, 'LIGAND_DIR'): admet_analysis.LIGAND_DIR = new_dir
    
    try:
        yield
    finally:
        # Restore original paths
        if hasattr(config, 'LIGAND_DIR'): config.LIGAND_DIR = old_config
        if hasattr(docking, 'LIGAND_DIR'): docking.LIGAND_DIR = old_docking
        if hasattr(admet_analysis, 'LIGAND_DIR'): admet_analysis.LIGAND_DIR = old_admet


def _extract_update_value(item: Any) -> Any:
    if hasattr(item, "value"):
        return item.value
    if isinstance(item, dict):
        return item.get("value", item)
    return item


def _consume_generator(gen: Iterable[Any]) -> Any:
    last = None
    for last in gen:
        pass
    return last


def _safe_copy(src: str | Path, dst: str | Path) -> bool:
    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def _update_dft_script_file(new_pdb_path: str) -> None:
    script_path = "dft.py"
    if not os.path.exists(script_path):
        raise FileNotFoundError("dft.py not found in current directory")

    with open(script_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(script_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip().startswith("PDB_FILE ="):
                f.write(f"PDB_FILE = {repr(new_pdb_path)}\n")
            else:
                f.write(line)


def parse_protein_lines(raw_text: str) -> List[str]:
    proteins: List[str] = []
    for line in raw_text.splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            proteins.append(value)
    return proteins


def _extract_iframe_html(iframe_html: str) -> Optional[str]:
    if not iframe_html:
        return None
    match = re.search(r'src="data:text/html;base64,([^"]+)"', iframe_html)
    if not match:
        return None
    try:
        return base64.b64decode(match.group(1)).decode("utf-8", errors="ignore")
    except Exception:
        return None


def save_3d_viewer_screenshot(iframe_html: str, output_png: Path) -> bool:
    html_content = _extract_iframe_html(iframe_html)
    if not html_content:
        return False

    tmp_html = output_png.with_suffix(".html")
    tmp_html.write_text(html_content, encoding="utf-8")

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")

        driver = webdriver.Chrome(options=options)
        driver.get(f"file://{tmp_html.absolute()}")
        time.sleep(6)

        try:
            canvas = driver.find_element(By.TAG_NAME, "canvas")
            canvas.screenshot(str(output_png))
        except Exception:
            driver.save_screenshot(str(output_png))
        driver.quit()
        return output_png.exists()
    except Exception:
        return False


def _extract_pose_text_from_pdbqt(pdbqt_path: Path, pose_number: int) -> str:
    if not pdbqt_path.exists():
        return ""

    lines = pdbqt_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    in_model = False
    model_lines: List[str] = []
    for line in lines:
        if line.startswith("MODEL"):
            try:
                in_model = int(line.split()[1]) == pose_number
            except Exception:
                in_model = False
        if in_model:
            model_lines.append(line)
        if in_model and line.startswith("ENDMDL"):
            break

    return "".join(model_lines) if model_lines else "".join(lines)


class ProteinPipelineBatch:
    def __init__(self, output_base_dir: str = "batch_results", run_dft: bool = False):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.run_dft = run_dft
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _protein_dir(self, protein_input: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in protein_input.strip())
        protein_dir = self.output_base_dir / safe
        protein_dir.mkdir(parents=True, exist_ok=True)
        for step_key, _ in PIPELINE_STEPS:
            if step_key == "08_dft" and not self.run_dft:
                continue
            (protein_dir / step_key).mkdir(exist_ok=True)
        return protein_dir

    def _save_df(self, df: Optional[pd.DataFrame], out_csv: Path) -> Optional[str]:
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(out_csv, index=False)
            return str(out_csv)
        return None

    def _write_summary(self, protein_dir: Path, result: Dict[str, Any]) -> None:
        with open(protein_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    def _collect_step_files(self, protein_dir: Path) -> Dict[str, List[str]]:
        step_files: Dict[str, List[str]] = {}
        for step_key, _ in PIPELINE_STEPS:
            step_dir = protein_dir / step_key
            if not step_dir.exists():
                continue
            step_files[step_key] = [str(p.relative_to(protein_dir)) for p in sorted(step_dir.rglob("*")) if p.is_file()]
        return step_files

    def process_single_protein(self, protein_input: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"protein_input": protein_input, "start_time": datetime.now().isoformat(), "steps": {}}
        protein_dir = self._protein_dir(protein_input)
        protein_name = protein_input.strip()

        print(f"\n{'='*50}\n🚀 STARTING BATCH PIPELINE FOR: {protein_name}\n{'='*50}")

        # --- PREVENT DATA BLEED: Clear global working directories ---
        for global_dir in [RAMPLOT_OUTPUT_DIR, PRANKWEB_OUTPUT_DIR, DOCKING_RESULTS_DIR]:
            if os.path.exists(global_dir):
                shutil.rmtree(global_dir)
            os.makedirs(global_dir, exist_ok=True)

        # ==========================================
        # STEPS 1-3: WITH AUTO-RETRY LOGIC
        # ==========================================
        max_attempts = 5
        failed_pdbs = []
        prep_success = False

        for attempt in range(max_attempts):
            if attempt > 0:
                print(f"\n⚠️ Retrying {protein_name} (Attempt {attempt+1}/{max_attempts}) - Excluding: {failed_pdbs}")
                # CLEAR FOLDERS FOR RETRY
                for step_key in ["01_structure_search", "02_ramachandran", "03_protein_preparation"]:
                    step_dir = protein_dir / step_key
                    if step_dir.exists():
                        shutil.rmtree(step_dir)
                    step_dir.mkdir(exist_ok=True)

            # --- 1) Structure search ---
            print(f"🔍 [Step 1] Searching for best structure for {protein_name}...")
            search = find_best_pdb_structure(protein_name, max_check=100, excluded_pdbs=failed_pdbs)
            if not search:
                print(f"❌ [Step 1] No suitable structure found for {protein_name}.")
                result["steps"]["structure_search"] = {"status": "failed", "error": "No suitable structure found"}
                break 
            
            pdb_id, pdb_path = search
            print(f"✅ [Step 1] Selected structure: {pdb_id}")
            current_pdb_info.update({
                "pdb_id": pdb_id,
                "pdb_path": pdb_path,
                "prepared_pdbqt": None,
                "search_term": protein_name,
            })

            step1_dir = protein_dir / "01_structure_search"
            _safe_copy(pdb_path, step1_dir / f"{pdb_id}.pdb")
            try:
                with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
                    structure_iframe = show_structure(protein_text=f.read(), ligand_text=None, pdb_id=pdb_id, protein_name=protein_name)
                save_3d_viewer_screenshot(structure_iframe, step1_dir / f"{pdb_id}_structure.png")
            except Exception:
                pass
            result["steps"]["structure_search"] = {"status": "success", "protein_name": protein_name, "pdb_id": pdb_id, "pdb_path": pdb_path}

            # --- 2) Ramachandran ---
            print(f"📈 [Step 2] Running Ramachandran analysis for {pdb_id}...")
            step2_dir = protein_dir / "02_ramachandran"
            try:
                # Passing progress=None to prevent Gradio errors in headless mode
                run_ramplot(progress=None) 
                
                # Verify that files were actually generated before moving on
                if os.path.exists(RAMPLOT_OUTPUT_DIR):
                    plots_generated = list(Path(RAMPLOT_OUTPUT_DIR).rglob("*.png"))
                    if not plots_generated:
                        raise ValueError("No Ramachandran PNG plots generated.")
                        
                    for item in Path(RAMPLOT_OUTPUT_DIR).rglob("*"):
                        if item.is_file():
                            rel = item.relative_to(RAMPLOT_OUTPUT_DIR)
                            target = step2_dir / rel
                            target.parent.mkdir(parents=True, exist_ok=True)
                            _safe_copy(item, target)
                    result["steps"]["ramachandran"] = {"status": "success"}
                    print(f"✅ [Step 2] Ramachandran plots successfully generated.")
                else:
                    raise FileNotFoundError(f"Output directory {RAMPLOT_OUTPUT_DIR} not found.")
            except Exception as exc:
                print(f"❌ [Step 2] Ramachandran failed for {pdb_id}: {exc}")
                failed_pdbs.append(pdb_id)
                continue # Skip Step 3 and loop back to try a new PDB

            # 🛑 CRITICAL FIX: Refresh pdb_path in case run_ramplot swapped it with a SWISS-MODEL!
            pdb_path = current_pdb_info.get("pdb_path", pdb_path)

            # --- 3) Protein preparation ---
            print(f"🛠️ [Step 3] Running Meeko protein preparation on {os.path.basename(pdb_path)}...")
            step3_dir = protein_dir / "03_protein_preparation"
            output_base = step3_dir / f"{pdb_id}_prepared"
            
            cmd = [  
                sys.executable, "mk_prepare_receptor.py", 
                "-i", pdb_path, "-o", str(output_base), "-p",
                "--charge_model", "gasteiger", "--default_altloc", "A"
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                pdbqt_path = f"{output_base}.pdbqt"
                
                if os.path.exists(pdbqt_path):
                    current_pdb_info["prepared_pdbqt"] = pdbqt_path
                    result["steps"]["protein_preparation"] = {"status": "success", "prepared_pdbqt": pdbqt_path}
                    print(f"✅ [Step 3] Protein prepared successfully: {pdbqt_path}")
                    
                    try:
                        with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
                            prep_iframe = show_structure(protein_text=f.read(), ligand_text=None, pdb_id=f"Prepared {pdb_id}", protein_name=protein_name)
                        save_3d_viewer_screenshot(prep_iframe, step3_dir / f"{pdb_id}_prepared.png")
                    except Exception:
                        pass
                        
                    prep_success = True
                    break # ✅ Success! Break out of the retry loop
                else:
                    raise FileNotFoundError("PDBQT file not generated by Meeko.")
            except Exception as e:
                print(f"❌ [Step 3] Preparation failed for {pdb_id}: {e}")
                failed_pdbs.append(pdb_id)
                continue # Loop continues and retries with next best PDB...

        # Check if we exhausted retries without success
        if not prep_success:
            print(f"🚨 FAILED: Exhausted all retries for {protein_name}. Skipping to next protein.")
            result["pipeline_status"] = "failed"
            result["end_time"] = datetime.now().isoformat()
            result["step_files"] = self._collect_step_files(protein_dir)
            self._write_summary(protein_dir, result)
            return result

        # ==========================================
        # STEPS 4-8: POST-PREP PIPELINE
        # ==========================================

        # 4) Binding site prediction
        print(f"🎯 [Step 4] Predicting binding sites using PrankWeb...")
        step4_dir = protein_dir / "04_binding_sites"
        try:
            pocket_final = _consume_generator(run_prankweb_prediction())
            pocket_df = _extract_update_value(pocket_final[1]) if isinstance(pocket_final, tuple) and len(pocket_final) >= 2 else None
            combined_csv = self._save_df(pocket_df, step4_dir / "combined_pockets.csv")
            if os.path.exists(PRANKWEB_OUTPUT_DIR):
                for item in Path(PRANKWEB_OUTPUT_DIR).rglob("*"):
                    if item.is_file():
                        rel = item.relative_to(PRANKWEB_OUTPUT_DIR)
                        target = step4_dir / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        _safe_copy(item, target)
            result["steps"]["binding_sites"] = {"status": "success" if combined_csv or current_pdb_info.get("combined_csv") else "failed", "combined_csv": current_pdb_info.get("combined_csv") or combined_csv}
            print(f"✅ [Step 4] Binding site prediction complete.")
        except Exception as exc:
            print(f"❌ [Step 4] Binding site prediction failed: {exc}")
            result["steps"]["binding_sites"] = {"status": "failed", "error": str(exc)}

        # -----------------------------------------------------------------------------
        # 📂 DIRECTORY MATCHING: Find the specific ligand folder OR use base fallback
        # -----------------------------------------------------------------------------
        base_ligand_dir = Path(LIGAND_DIR) if Path(LIGAND_DIR).exists() else Path("pdbqt")
        target_ligand_dir = None
        
        if base_ligand_dir.exists():
            # Strategy 1: Look for exact protein subdirectory
            for d in base_ligand_dir.iterdir():
                if d.is_dir() and d.name.lower() == protein_name.lower():
                    target_ligand_dir = d
                    print(f"ℹ️ [Steps 5-8] Found specific ligand subdirectory: {d.name}/")
                    break
                    
            # Strategy 2: If no subdirs, check if base dir contains raw PDBQT files
            if not target_ligand_dir:
                has_global_ligands = any(f.suffix.lower() == '.pdbqt' for f in base_ligand_dir.iterdir() if f.is_file())
                if has_global_ligands:
                    target_ligand_dir = base_ligand_dir
                    print(f"ℹ️ [Steps 5-8] No specific folder for '{protein_name}'. Using global ligands directly from {base_ligand_dir}/")

        if not target_ligand_dir:
            print(f"⚠️ [Steps 5-8] No specific ligand folder OR global ligands found in {base_ligand_dir}/. Skipping ligand-dependent steps.")
            result["steps"]["ligand_analysis"] = {"status": "skipped", "message": "No ligand folder/files found"}
            result["steps"]["docking"] = {"status": "skipped", "message": "No ligand folder/files found"}
            result["steps"]["admet"] = {"status": "skipped", "message": "No docking results"}
            result["steps"]["dft"] = {"status": "skipped", "message": "No docking results"}
            
            print(f"🎉 PIPELINE COMPLETED FOR: {protein_name} (with skips)")
            result["pipeline_status"] = "completed_with_skips"
            result["end_time"] = datetime.now().isoformat()
            result["step_files"] = self._collect_step_files(protein_dir)
            self._write_summary(protein_dir, result)
            return result

        # Apply the path monkey-patch for Steps 5-8
        with patch_ligand_dir(str(target_ligand_dir)):
            
            # 5) Ligand analysis
            print(f"🧪 [Step 5] Extracting and analyzing ligands...")
            step5_dir = protein_dir / "05_ligand_analysis"
            try:
                lig_df, lig_csv = run_ligand_classification(str(target_ligand_dir))
                lig_saved = self._save_df(lig_df, step5_dir / "ligand_classification_report.csv")
                if lig_csv and os.path.exists(lig_csv):
                    _safe_copy(lig_csv, step5_dir / Path(lig_csv).name)
                if os.path.exists("ligand_pdb"):
                    ligand_pdb_dir = step5_dir / "ligand_pdb"
                    ligand_pdb_dir.mkdir(exist_ok=True)
                    for pdb_file in Path("ligand_pdb").glob("*.pdb"):
                        _safe_copy(pdb_file, ligand_pdb_dir / pdb_file.name)
                result["steps"]["ligand_analysis"] = {"status": "success" if lig_saved else "failed", "csv": lig_saved}
                print(f"✅ [Step 5] Ligand analysis complete.")
            except Exception as exc:
                print(f"❌ [Step 5] Ligand analysis failed: {exc}")
                result["steps"]["ligand_analysis"] = {"status": "failed", "error": str(exc)}

            # 6) Docking
            print(f"🚀 [Step 6] Running molecular docking...")
            step6_dir = protein_dir / "06_docking"
            try:
                docking_final = _consume_generator(run_molecular_docking())
                docking_df = _extract_update_value(docking_final[1]) if isinstance(docking_final, tuple) and len(docking_final) >= 2 else None
                docking_csv = self._save_df(docking_df, step6_dir / "docking_summary.csv")
                if os.path.exists(DOCKING_RESULTS_DIR):
                    dst_tree = step6_dir / "docking_results"
                    if dst_tree.exists():
                        shutil.rmtree(dst_tree)
                    shutil.copytree(DOCKING_RESULTS_DIR, dst_tree)

                saved_screenshots: List[str] = []
                if isinstance(docking_df, pd.DataFrame) and not docking_df.empty:
                    try:
                        # Save 3D screenshots directly into docking_reports alongside 2D files
                        reports_dir = step6_dir / "docking_results" / "docking_reports"
                        reports_dir.mkdir(parents=True, exist_ok=True)
                        
                        print(f"  -> Generating 3D screenshots for ALL {len(docking_df)} docked poses (this may take a few minutes)...")
                        
                        for idx, row in docking_df.iterrows():
                            ligand_path_str = str(row.get("pdb_file", ""))
                            if not ligand_path_str:
                                continue
                                
                            ligand_path = Path(ligand_path_str)
                            pose_num = int(row.get("pose_number", 1))
                            
                            # Extract base name to PERFECTLY MATCH the 2D interaction reports
                            # e.g., 'Chain_A_SKNKS 1_fpocket_pocket1_out.pdbqt' -> 'Chain_A_SKNKS 1_fpocket_pocket1'
                            base_name = ligand_path.stem
                            if base_name.endswith('_out'): base_name = base_name[:-4]
                            elif base_name.endswith('_docked'): base_name = base_name[:-7]
                            
                            # Final file name prevents overwriting by including pose number
                            screenshot_name = f"{base_name}_pose_{pose_num}_3d.png"
                            
                            receptor_path = str(row.get("receptor_pdb_file") or current_pdb_info.get("pdb_path"))
                            
                            protein_text = Path(receptor_path).read_text(encoding="utf-8", errors="ignore") if receptor_path and os.path.exists(receptor_path) else ""
                            ligand_text = _extract_pose_text_from_pdbqt(ligand_path, pose_num)

                            if protein_text and ligand_text:
                                dock_iframe = show_structure(
                                    protein_text=protein_text,
                                    ligand_text=ligand_text,
                                    pdb_id=f"Pose {pose_num}",
                                    protein_name=base_name,
                                )
                                
                                target_file = reports_dir / screenshot_name
                                if save_3d_viewer_screenshot(dock_iframe, target_file):
                                    saved_screenshots.append(str(target_file))
                                    print(f"     📸 Saved: {screenshot_name}")
                                    
                    except Exception as e:
                        print(f"     ❌ Error during 3D screenshot generation: {e}")

                result["steps"]["docking"] = {
                    "status": "success" if docking_csv else "failed",
                    "csv": docking_csv,
                    "screenshots": saved_screenshots if 'saved_screenshots' in locals() else [],
                }
                print(f"✅ [Step 6] Docking complete.")
            except Exception as exc:
                print(f"❌ [Step 6] Docking failed: {exc}")
                result["steps"]["docking"] = {"status": "failed", "error": str(exc)}

            # 7) ADMET
            print(f"💊 [Step 7] Screening ADMET properties...")
            step7_dir = protein_dir / "07_admet"
            try:
                admet_out = run_admet_prediction()
                if admet_out:
                    msg, admet_df, admet_csv = admet_out
                    admet_saved = self._save_df(admet_df, step7_dir / "admet_results.csv")
                    if admet_csv and os.path.exists(admet_csv):
                        _safe_copy(admet_csv, step7_dir / Path(admet_csv).name)
                    result["steps"]["admet"] = {"status": "success" if admet_saved else "failed", "message": msg}
                    print(f"✅ [Step 7] ADMET screening complete.")
                else:
                    print(f"❌ [Step 7] ADMET screening returned no results.")
                    result["steps"]["admet"] = {"status": "failed"}
            except Exception as exc:
                print(f"❌ [Step 7] ADMET screening failed: {exc}")
                result["steps"]["admet"] = {"status": "failed", "error": str(exc)}

            # 8) Optional DFT
            if self.run_dft:
                print(f"⚛️ [Step 8] Running DFT Analysis...")
                step8_dir = protein_dir / "08_dft"
                result["steps"]["dft"] = self._run_dft_batch(step8_dir)

        print(f"🎉 PIPELINE COMPLETED FOR: {protein_name}")
        result["pipeline_status"] = "completed"
        result["end_time"] = datetime.now().isoformat()
        result["step_files"] = self._collect_step_files(protein_dir)
        self._write_summary(protein_dir, result)
        return result

    def _run_dft_batch(self, step_dir: Path) -> Dict[str, Any]:
        print(f"  -> Locating best docked pose for DFT analysis...")
        
        # 1. Find the docking summary to identify the best pose
        docking_dir = step_dir.parent / "06_docking"
        docking_csv = next(docking_dir.rglob("*summary*.csv"), None)
        
        if not docking_csv or not docking_csv.exists():
            print(f"    ⚠️ [Step 8] No docking summary found in {docking_dir.name}. Cannot determine best pose.")
            return {"status": "skipped", "message": "No docking summary found"}

        try:
            df = pd.read_csv(docking_csv)
            # Find the energy/affinity column
            energy_col = next((col for col in df.columns if 'affinity' in col.lower() or 'energy' in col.lower()), None)
            
            if not energy_col or df.empty:
                print("    ⚠️ Could not determine binding energy column or dataframe is empty. Skipping DFT.")
                return {"status": "skipped", "message": "Missing energy column in docking summary"}

            # Find the row with the lowest energy (most negative affinity)
            best_row = df.loc[df[energy_col].idxmin()]
            
            # Find the complex PDB file path
            best_complex_path_str = best_row.get("Complex_PDB")
            if not best_complex_path_str or pd.isna(best_complex_path_str):
                # Fallback 1: try to find a column with 'complex' in the name
                complex_col = next((col for col in df.columns if 'complex' in col.lower() and 'pdb' in col.lower()), None)
                if complex_col:
                    best_complex_path_str = best_row.get(complex_col)
                    
            # --- NEW LOGIC: Fallback 2: Use pdb_file and replace _ligand with _complex ---
            if not best_complex_path_str or pd.isna(best_complex_path_str):
                pdb_col = next((col for col in df.columns if 'pdb' in col.lower() and 'file' in col.lower()), None)
                if pdb_col:
                    pdb_file_str = best_row.get(pdb_col)
                    if pdb_file_str and not pd.isna(pdb_file_str):
                        best_complex_path_str = str(pdb_file_str).replace("_ligand", "_complex")
            # --------------------------------------------------------------------------------

            if not best_complex_path_str or pd.isna(best_complex_path_str):
                 print("    ⚠️ Could not find Complex_PDB path in the summary CSV.")
                 return {"status": "skipped", "message": "Missing Complex_PDB in summary"}

            # --- NEW LOGIC: Robust Path Resolution ---
            best_complex_path = Path(best_complex_path_str)
            if not best_complex_path.exists():
                # Search recursively inside the docking directory to find the file
                found_files = list(docking_dir.rglob(best_complex_path.name))
                if found_files:
                    best_complex_path = found_files[0]
                else:
                    best_complex_path = Path(best_complex_path_str).resolve()
            # -----------------------------------------
            
            if not best_complex_path.exists():
                print(f"    ⚠️ Best complex PDB file not found on disk: {best_complex_path}")
                return {"status": "skipped", "message": "Best complex PDB file not found"}

            print(f"    🌟 Best pose selected: {best_complex_path.name} | Energy: {best_row[energy_col]}")

        except Exception as e:
            print(f"    ❌ Error parsing docking summary: {e}")
            return {"status": "failed", "message": f"Error reading docking summary: {e}"}

        # 2. Set up and run DFT on the single best pose
        orca_temp = Path("orca_temp")
        
        # Clean up any leftover temp folder from a previous aborted run before starting
        if orca_temp.exists():
            shutil.rmtree(orca_temp, ignore_errors=True)
            
        all_results: List[Dict[str, Any]] = []
        
        try:
            print(f"  -> Running ORCA on {best_complex_path.name}")
            # Update dft.py with the single best file
            _update_dft_script_file(str(best_complex_path))
            
            cmd = 'cmd /c "call activate orca_env && python dft.py"'
            subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # 3. Handle outputs from the isolated temp folder
            if orca_temp.exists():
                calc_folder = step_dir / best_complex_path.stem
                calc_folder.mkdir(exist_ok=True)
                
                # Move everything to the safe result folder for this pose
                for f in orca_temp.glob("*"):
                    try: 
                        shutil.copy2(f, calc_folder / f.name)
                    except Exception as e: 
                        print(f"    ⚠️ Failed to copy {f.name}: {e}")
                
                # Extract data from the temp CSV generated by dft.py
                temp_csv = orca_temp / "dft_batch_results.csv"
                if temp_csv.exists():
                    dft_df = pd.read_csv(temp_csv)
                    if not dft_df.empty:
                        row = dft_df.iloc[-1].to_dict()
                        row["Filename"] = best_complex_path.name
                        row["Binding_Energy"] = best_row[energy_col] # Keep track of the binding affinity!
                        all_results.append(row)
                        print(f"    ✅ Data extracted and outputs saved to {calc_folder.name}/")
                else:
                    print(f"    ❌ ORCA calculation failed (no CSV output) for {best_complex_path.name}")
                    all_results.append({"Filename": best_complex_path.name, "error": "No CSV generated"})
                    
                # Clean up the root temp folder
                shutil.rmtree(orca_temp, ignore_errors=True)
                print(f"    🧹 Cleaned up temporary ORCA files.")
            else:
                print(f"    ❌ ORCA temporary folder not found. Calculation failed.")
                all_results.append({"Filename": best_complex_path.name, "error": "Temp folder not created"})
                
        except Exception as exc:
            print(f"    ❌ ORCA execution error: {exc}")
            all_results.append({"Filename": best_complex_path.name, "error": str(exc)})

        # 4. Save the final single-row batch result for the UI to read
        if all_results:
            out_csv = step_dir / "dft_batch_results.csv"
            pd.DataFrame(all_results).to_csv(out_csv, index=False)
            print(f"✅ [Step 8] DFT step complete for best pose.")
            return {"status": "success", "count": 1, "csv": str(out_csv)}
            
        return {"status": "failed", "message": "No DFT rows collected"}

    def run_batch(self, proteins: List[str]) -> Dict[str, Any]:
        batch_result: Dict[str, Any] = {"start_time": datetime.now().isoformat(), "proteins": {}}
        for index, protein in enumerate(proteins, start=1):
            print(f"\n[{index}/{len(proteins)}] Queuing: {protein}")
            try:
                batch_result["proteins"][protein] = self.process_single_protein(protein)
            except Exception as exc:
                print(f"🚨 FATAL ERROR for {protein}: {exc}")
                batch_result["proteins"][protein] = {"pipeline_status": "fatal_error", "error": str(exc)}

        summary_rows = [{"protein": p, "status": d.get("pipeline_status", "unknown")} for p, d in batch_result["proteins"].items()]
        summary_csv = self.output_base_dir / f"batch_summary_{self.timestamp}.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        batch_result["summary_csv"] = str(summary_csv)

        index_json = self.output_base_dir / "batch_index.json"
        with open(index_json, "w", encoding="utf-8") as f:
            json.dump(batch_result, f, indent=2)
        batch_result["batch_index"] = str(index_json)
        batch_result["end_time"] = datetime.now().isoformat()
        
        print("\n" + "="*50)
        print("✅ ALL BATCH OPERATIONS COMPLETED")
        print("="*50)
        return batch_result


def _load_proteins_from_cli(proteins: List[str], protein_file: Optional[str]) -> List[str]:
    if proteins:
        return proteins
    if protein_file:
        with open(protein_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    raise ValueError("Provide proteins using --proteins or --protein-file")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run current app.py pipeline in batch mode")
    parser.add_argument("--proteins", nargs="*", default=[], help="Protein names")
    parser.add_argument("--protein-file", default=None, help="Text file with one protein per line")
    parser.add_argument("--output-dir", default="batch_results", help="Output folder")
    parser.add_argument("--run-dft", action="store_true", help="Run DFT step after ADMET")
    args = parser.parse_args()

    protein_list = _load_proteins_from_cli(args.proteins, args.protein_file)
    runner = ProteinPipelineBatch(output_base_dir=args.output_dir, run_dft=args.run_dft)
    results = runner.run_batch(protein_list)

    print(json.dumps({"count": len(results.get("proteins", {})), "summary_csv": results.get("summary_csv")}, indent=2))


if __name__ == "__main__":
    main()