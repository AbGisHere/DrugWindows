"""Batch runner aligned with the current app.py pipeline.

Pipeline order per protein:
1) Structure search
2) Ramachandran analysis
3) Protein preparation
4) Binding-site prediction (P2Rank + Fpocket)
5) Ligand analysis
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
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from admet_analysis import run_admet_prediction
from config import DOCKING_RESULTS_DIR, LIGAND_DIR, PRANKWEB_OUTPUT_DIR, RAMPLOT_OUTPUT_DIR, current_pdb_info
from docking import run_molecular_docking
from ligand_analysis import run_ligand_classification
from prankweb import run_prankweb_prediction
from protein_prep import prepare_protein_meeko
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
    """Save a screenshot of the generated 3Dmol HTML via selenium/chromedriver."""
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

        # 1) Structure search
        protein_name = protein_input.strip()
        search = find_best_pdb_structure(protein_name, max_check=100)
        if not search:
            result["steps"]["structure_search"] = {"status": "failed", "error": "No suitable structure found"}
            result["pipeline_status"] = "failed"
            result["end_time"] = datetime.now().isoformat()
            result["step_files"] = self._collect_step_files(protein_dir)
            self._write_summary(protein_dir, result)
            return result

        pdb_id, pdb_path = search
        current_pdb_info.update({
            "pdb_id": pdb_id,
            "pdb_path": pdb_path,
            "prepared_pdbqt": None,
            "docking_results": None,
            "prankweb_csv": None,
            "combined_csv": None,
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

        # 2) Ramachandran
        step2_dir = protein_dir / "02_ramachandran"
        try:
            ram_out = run_ramplot()
            ram_status = _extract_update_value(ram_out[0]) if isinstance(ram_out, tuple) and ram_out else str(ram_out)
            if os.path.exists(RAMPLOT_OUTPUT_DIR):
                for item in Path(RAMPLOT_OUTPUT_DIR).rglob("*"):
                    if item.is_file():
                        rel = item.relative_to(RAMPLOT_OUTPUT_DIR)
                        target = step2_dir / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        _safe_copy(item, target)
            result["steps"]["ramachandran"] = {"status": "success", "message": str(ram_status)}
        except Exception as exc:
            result["steps"]["ramachandran"] = {"status": "failed", "error": str(exc)}

        # 3) Protein preparation
        step3_dir = protein_dir / "03_protein_preparation"
        prep_final = _consume_generator(prepare_protein_meeko())
        prep_download = _extract_update_value(prep_final[2]) if isinstance(prep_final, tuple) and len(prep_final) >= 3 else None

        prepared_pdbqt = current_pdb_info.get("prepared_pdbqt")
        if not prepared_pdbqt or not os.path.exists(prepared_pdbqt):
            result["steps"]["protein_preparation"] = {"status": "failed", "error": "No prepared PDBQT generated"}
            result["pipeline_status"] = "failed"
            result["end_time"] = datetime.now().isoformat()
            result["step_files"] = self._collect_step_files(protein_dir)
            self._write_summary(protein_dir, result)
            return result

        _safe_copy(prepared_pdbqt, step3_dir / Path(prepared_pdbqt).name)
        if prep_download and os.path.exists(prep_download):
            _safe_copy(prep_download, step3_dir / Path(prep_download).name)
        try:
            with open(prepared_pdbqt, "r", encoding="utf-8", errors="ignore") as f:
                prep_iframe = show_structure(protein_text=f.read(), ligand_text=None, pdb_id=f"Prepared {pdb_id}", protein_name=protein_name)
            save_3d_viewer_screenshot(prep_iframe, step3_dir / f"{pdb_id}_prepared.png")
        except Exception:
            pass
        result["steps"]["protein_preparation"] = {"status": "success", "prepared_pdbqt": prepared_pdbqt}

        # 4) Binding site prediction
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
        except Exception as exc:
            result["steps"]["binding_sites"] = {"status": "failed", "error": str(exc)}

        # 5) Ligand analysis
        step5_dir = protein_dir / "05_ligand_analysis"
        try:
            ligand_folder = LIGAND_DIR if os.path.exists(LIGAND_DIR) else "pdbqt"
            lig_df, lig_csv = run_ligand_classification(ligand_folder)
            lig_saved = self._save_df(lig_df, step5_dir / "ligand_classification_report.csv")
            if lig_csv and os.path.exists(lig_csv):
                _safe_copy(lig_csv, step5_dir / Path(lig_csv).name)
            if os.path.exists("ligand_pdb"):
                ligand_pdb_dir = step5_dir / "ligand_pdb"
                ligand_pdb_dir.mkdir(exist_ok=True)
                for pdb_file in Path("ligand_pdb").glob("*.pdb"):
                    _safe_copy(pdb_file, ligand_pdb_dir / pdb_file.name)
            result["steps"]["ligand_analysis"] = {"status": "success" if lig_saved else "failed", "csv": lig_saved}
        except Exception as exc:
            result["steps"]["ligand_analysis"] = {"status": "failed", "error": str(exc)}

        # 6) Docking
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

            # Screenshot top docked pose
            if isinstance(docking_df, pd.DataFrame) and not docking_df.empty:
                try:
                    top = docking_df.sort_values(by="binding_energy", ascending=True).iloc[0]
                    receptor_path = str(top.get("receptor_pdb_file") or current_pdb_info.get("pdb_path"))
                    ligand_path = str(top.get("pdb_file"))
                    pose_num = int(top.get("pose_number", 1))

                    protein_text = Path(receptor_path).read_text(encoding="utf-8", errors="ignore") if os.path.exists(receptor_path) else ""
                    ligand_text = ""
                    if os.path.exists(ligand_path):
                        lines = Path(ligand_path).read_text(encoding="utf-8", errors="ignore").splitlines(True)
                        in_model = False
                        model_lines: List[str] = []
                        for line in lines:
                            if line.startswith("MODEL"):
                                try:
                                    in_model = int(line.split()[1]) == pose_num
                                except Exception:
                                    in_model = False
                            if in_model:
                                model_lines.append(line)
                            if in_model and line.startswith("ENDMDL"):
                                break
                        ligand_text = "".join(model_lines) if model_lines else "".join(lines)

                    if protein_text and ligand_text:
                        dock_iframe = show_structure(protein_text=protein_text, ligand_text=ligand_text, pdb_id="Docking", protein_name=str(top.get("ligand", "Ligand")))
                        save_3d_viewer_screenshot(dock_iframe, step6_dir / "top_pose_3d.png")
                except Exception:
                    pass

            result["steps"]["docking"] = {"status": "success" if docking_csv else "failed", "csv": docking_csv}
        except Exception as exc:
            result["steps"]["docking"] = {"status": "failed", "error": str(exc)}

        # 7) ADMET
        step7_dir = protein_dir / "07_admet"
        try:
            admet_out = run_admet_prediction()
            if admet_out:
                msg, admet_df, admet_csv = admet_out
                admet_saved = self._save_df(admet_df, step7_dir / "admet_results.csv")
                if admet_csv and os.path.exists(admet_csv):
                    _safe_copy(admet_csv, step7_dir / Path(admet_csv).name)
                result["steps"]["admet"] = {"status": "success" if admet_saved else "failed", "message": msg}
            else:
                result["steps"]["admet"] = {"status": "failed"}
        except Exception as exc:
            result["steps"]["admet"] = {"status": "failed", "error": str(exc)}

        # 8) Optional DFT
        if self.run_dft:
            step8_dir = protein_dir / "08_dft"
            result["steps"]["dft"] = self._run_dft_batch(step8_dir)

        result["pipeline_status"] = "completed"
        result["end_time"] = datetime.now().isoformat()
        result["step_files"] = self._collect_step_files(protein_dir)
        self._write_summary(protein_dir, result)
        return result

    def _run_dft_batch(self, step_dir: Path) -> Dict[str, Any]:
        all_pdbs = list(Path(DOCKING_RESULTS_DIR).glob("**/docked_pdb/*.pdb")) if os.path.exists(DOCKING_RESULTS_DIR) else []
        if not all_pdbs:
            return {"status": "skipped", "message": "No docked PDB files found"}

        single_csv = "orca_electronic_metrics.csv"
        if os.path.exists(single_csv):
            os.remove(single_csv)

        all_results: List[Dict[str, Any]] = []
        for pdb_path in all_pdbs:
            try:
                _update_dft_script_file(str(pdb_path))
                cmd = 'cmd /c "call activate orca_env && python dft.py"'
                subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if os.path.exists(single_csv):
                    df = pd.read_csv(single_csv)
                    if not df.empty:
                        row = df.iloc[-1].to_dict()
                        row["Filename"] = pdb_path.name
                        all_results.append(row)
            except Exception as exc:
                all_results.append({"Filename": pdb_path.name, "error": str(exc)})

        if all_results:
            out_csv = step_dir / "dft_batch_results.csv"
            pd.DataFrame(all_results).to_csv(out_csv, index=False)
            return {"status": "success", "count": len(all_results), "csv": str(out_csv)}
        return {"status": "failed", "message": "No DFT rows collected"}

    def run_batch(self, proteins: List[str]) -> Dict[str, Any]:
        batch_result: Dict[str, Any] = {"start_time": datetime.now().isoformat(), "proteins": {}}
        for index, protein in enumerate(proteins, start=1):
            print(f"[{index}/{len(proteins)}] Processing: {protein}")
            try:
                batch_result["proteins"][protein] = self.process_single_protein(protein)
            except Exception as exc:
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
        return batch_result

def _load_proteins_from_cli(proteins: List[str], protein_file: Optional[str]) -> List[str]:
    if proteins:
        return proteins
    if protein_file:
        with open(protein_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    raise ValueError("Provide proteins using --proteins or --protein-file")

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

    print("\nBatch complete")
    print(json.dumps({"count": len(results.get("proteins", {})), "summary_csv": results.get("summary_csv")}, indent=2))


if __name__ == "__main__":
    main()
