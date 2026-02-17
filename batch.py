"""Batch runner for the current app.py pipeline.

This script mirrors the same processing order used by the Gradio app:
1) Structure search
2) Ramachandran analysis
3) Protein preparation (Meeko)
4) Binding-site prediction (P2Rank + Fpocket)
5) Ligand classification
6) Multi-chain docking
7) ADMET
8) DFT batch (optional per protein)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from admet_analysis import run_admet_prediction
from config import (
    DOCKING_RESULTS_DIR,
    LIGAND_DIR,
    PRANKWEB_OUTPUT_DIR,
    RAMPLOT_OUTPUT_DIR,
    current_pdb_info,
)
from docking import run_molecular_docking
from ligand_analysis import run_ligand_classification
from prankweb import run_prankweb_prediction
from protein_prep import prepare_protein_meeko
from ramachandran import run_ramplot
from utils import find_best_pdb_structure, map_disease_to_protein


def _extract_update_value(item: Any) -> Any:
    """Best-effort extraction of value from gr.update-like payloads."""
    if hasattr(item, "value"):
        return item.value
    if isinstance(item, dict):
        return item.get("value", item)
    return item


def _consume_generator(gen: Iterable[Any]) -> Any:
    """Consume a generator and return the last yielded item."""
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
                f.write(f"PDB_FILE = {repr(new_pdb_path)}\\n")
            else:
                f.write(line)


class ProteinPipelineBatch:
    def __init__(self, output_base_dir: str = "batch_results", run_dft: bool = False):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.run_dft = run_dft
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _protein_dir(self, protein_input: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in protein_input.strip())
        d = self.output_base_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_df(self, df: Optional[pd.DataFrame], out_csv: Path) -> Optional[str]:
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(out_csv, index=False)
            return str(out_csv)
        return None

    def process_single_protein(self, protein_input: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "protein_input": protein_input,
            "start_time": datetime.now().isoformat(),
            "steps": {},
        }
        protein_dir = self._protein_dir(protein_input)

        # Step 1: search
        protein_name = map_disease_to_protein(protein_input) or protein_input.strip()
        search = find_best_pdb_structure(protein_name, max_check=100)
        if not search:
            result["steps"]["structure_search"] = {"status": "failed", "error": "No suitable structure found"}
            result["pipeline_status"] = "failed"
            result["end_time"] = datetime.now().isoformat()
            self._write_summary(protein_dir, result)
            return result

        pdb_id, pdb_path = search
        current_pdb_info.update(
            {
                "pdb_id": pdb_id,
                "pdb_path": pdb_path,
                "prepared_pdbqt": None,
                "docking_results": None,
                "prankweb_csv": None,
                "combined_csv": None,
                "search_term": protein_name,
            }
        )
        step1_dir = protein_dir / "01_structure_search"
        step1_dir.mkdir(exist_ok=True)
        _safe_copy(pdb_path, step1_dir / f"{pdb_id}.pdb")
        result["steps"]["structure_search"] = {"status": "success", "pdb_id": pdb_id, "pdb_path": pdb_path}

        # Step 2: ramachandran
        step2_dir = protein_dir / "02_ramachandran"
        step2_dir.mkdir(exist_ok=True)
        ram_out = run_ramplot()
        ram_status = _extract_update_value(ram_out[0]) if isinstance(ram_out, tuple) and ram_out else str(ram_out)
        if os.path.exists(RAMPLOT_OUTPUT_DIR):
            for name in os.listdir(RAMPLOT_OUTPUT_DIR):
                src = os.path.join(RAMPLOT_OUTPUT_DIR, name)
                if os.path.isfile(src):
                    _safe_copy(src, step2_dir / name)
        result["steps"]["ramachandran"] = {"status": "success", "message": str(ram_status)}

        # Step 3: protein prep
        step3_dir = protein_dir / "03_protein_preparation"
        step3_dir.mkdir(exist_ok=True)
        prep_final = _consume_generator(prepare_protein_meeko())
        prep_file = None
        if isinstance(prep_final, tuple) and len(prep_final) >= 3:
            prep_file = _extract_update_value(prep_final[2])
        prepared_pdbqt = current_pdb_info.get("prepared_pdbqt")
        if not prepared_pdbqt or not os.path.exists(prepared_pdbqt):
            result["steps"]["protein_preparation"] = {"status": "failed"}
            result["pipeline_status"] = "failed"
            result["end_time"] = datetime.now().isoformat()
            self._write_summary(protein_dir, result)
            return result
        _safe_copy(prepared_pdbqt, step3_dir / Path(prepared_pdbqt).name)
        if prep_file and os.path.exists(prep_file):
            _safe_copy(prep_file, step3_dir / Path(prep_file).name)
        result["steps"]["protein_preparation"] = {"status": "success", "prepared_pdbqt": prepared_pdbqt}

        # Step 4: pockets
        step4_dir = protein_dir / "04_binding_sites"
        step4_dir.mkdir(exist_ok=True)
        pocket_final = _consume_generator(run_prankweb_prediction())
        pocket_df = None
        if isinstance(pocket_final, tuple) and len(pocket_final) >= 2:
            pocket_df = _extract_update_value(pocket_final[1])
        pockets_csv = self._save_df(pocket_df, step4_dir / "combined_pockets.csv")
        if os.path.exists(PRANKWEB_OUTPUT_DIR):
            for name in os.listdir(PRANKWEB_OUTPUT_DIR):
                src = os.path.join(PRANKWEB_OUTPUT_DIR, name)
                if os.path.isfile(src):
                    _safe_copy(src, step4_dir / name)
        result["steps"]["binding_sites"] = {
            "status": "success" if pockets_csv or current_pdb_info.get("combined_csv") else "failed",
            "combined_csv": current_pdb_info.get("combined_csv") or pockets_csv,
        }

        # Step 5: ligands
        step5_dir = protein_dir / "05_ligand_analysis"
        step5_dir.mkdir(exist_ok=True)
        lig_df, lig_csv = run_ligand_classification(LIGAND_DIR if os.path.exists(LIGAND_DIR) else "pdbqt")
        lig_saved = self._save_df(lig_df, step5_dir / "ligand_classification_report.csv")
        if lig_csv and os.path.exists(lig_csv):
            _safe_copy(lig_csv, step5_dir / Path(lig_csv).name)
        result["steps"]["ligand_analysis"] = {"status": "success" if lig_saved else "failed", "csv": lig_saved}

        # Step 6: docking
        step6_dir = protein_dir / "06_docking"
        step6_dir.mkdir(exist_ok=True)
        docking_final = _consume_generator(run_molecular_docking())
        docking_df = None
        if isinstance(docking_final, tuple) and len(docking_final) >= 2:
            docking_df = _extract_update_value(docking_final[1])
        docking_csv = self._save_df(docking_df, step6_dir / "docking_summary.csv")
        if os.path.exists(DOCKING_RESULTS_DIR):
            dst_tree = step6_dir / "docking_results"
            if dst_tree.exists():
                shutil.rmtree(dst_tree)
            shutil.copytree(DOCKING_RESULTS_DIR, dst_tree)
        result["steps"]["docking"] = {"status": "success" if docking_csv else "failed", "csv": docking_csv}

        # Step 7: admet
        step7_dir = protein_dir / "07_admet"
        step7_dir.mkdir(exist_ok=True)
        admet_out = run_admet_prediction()
        admet_status = "failed"
        if admet_out:
            msg, admet_df, admet_csv = admet_out
            admet_saved = self._save_df(admet_df, step7_dir / "admet_results.csv")
            if admet_csv and os.path.exists(admet_csv):
                _safe_copy(admet_csv, step7_dir / Path(admet_csv).name)
            admet_status = "success" if admet_saved else "failed"
            result["steps"]["admet"] = {"status": admet_status, "message": msg}
        else:
            result["steps"]["admet"] = {"status": "failed"}

        # Step 8: optional dft
        if self.run_dft:
            step8_dir = protein_dir / "08_dft"
            step8_dir.mkdir(exist_ok=True)
            result["steps"]["dft"] = self._run_dft_batch(step8_dir)

        result["pipeline_status"] = "completed"
        result["end_time"] = datetime.now().isoformat()
        self._write_summary(protein_dir, result)
        return result

    def _run_dft_batch(self, step_dir: Path) -> Dict[str, Any]:
        search_path = os.path.join(DOCKING_RESULTS_DIR, "**", "docked_pdb", "*.pdb")
        all_pdbs = list(Path(DOCKING_RESULTS_DIR).glob("**/docked_pdb/*.pdb")) if os.path.exists(DOCKING_RESULTS_DIR) else []
        all_results: List[Dict[str, Any]] = []
        if not all_pdbs:
            return {"status": "skipped", "message": f"No files at {search_path}"}

        single_csv = "orca_electronic_metrics.csv"
        if os.path.exists(single_csv):
            os.remove(single_csv)

        for pdb in all_pdbs:
            try:
                _update_dft_script_file(str(pdb))
                cmd = 'cmd /c "call activate orca_env && python dft.py"'
                subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if os.path.exists(single_csv):
                    df = pd.read_csv(single_csv)
                    if not df.empty:
                        row = df.iloc[-1].to_dict()
                        row["Filename"] = pdb.name
                        all_results.append(row)
            except Exception as exc:
                all_results.append({"Filename": pdb.name, "error": str(exc)})

        if all_results:
            out_csv = step_dir / "dft_batch_results.csv"
            pd.DataFrame(all_results).to_csv(out_csv, index=False)
            return {"status": "success", "csv": str(out_csv), "count": len(all_results)}
        return {"status": "failed"}

    def _write_summary(self, protein_dir: Path, result: Dict[str, Any]) -> None:
        with open(protein_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    def run_batch(self, proteins: List[str]) -> Dict[str, Any]:
        batch = {"start_time": datetime.now().isoformat(), "proteins": {}}
        for i, p in enumerate(proteins, start=1):
            print(f"[{i}/{len(proteins)}] Processing: {p}")
            try:
                batch["proteins"][p] = self.process_single_protein(p)
            except Exception as exc:
                batch["proteins"][p] = {"pipeline_status": "fatal_error", "error": str(exc)}

        rows = []
        for p, r in batch["proteins"].items():
            rows.append({"protein": p, "status": r.get("pipeline_status", "unknown")})
        summary_csv = self.output_base_dir / f"batch_summary_{self.timestamp}.csv"
        pd.DataFrame(rows).to_csv(summary_csv, index=False)
        batch["summary_csv"] = str(summary_csv)
        batch["end_time"] = datetime.now().isoformat()
        return batch


def _load_protein_list(path: Optional[str], proteins: List[str]) -> List[str]:
    if proteins:
        return proteins
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    raise ValueError("Provide proteins via --proteins or --protein-file")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch runner for current app.py pipeline")
    parser.add_argument("--proteins", nargs="*", default=[], help="Protein names/targets")
    parser.add_argument("--protein-file", default=None, help="Text file with one target per line")
    parser.add_argument("--output-dir", default="batch_results", help="Batch output folder")
    parser.add_argument("--run-dft", action="store_true", help="Run DFT step after ADMET")
    args = parser.parse_args()

    proteins = _load_protein_list(args.protein_file, args.proteins)
    runner = ProteinPipelineBatch(output_base_dir=args.output_dir, run_dft=args.run_dft)
    results = runner.run_batch(proteins)

    print("\nBatch completed")
    print(json.dumps({"summary_csv": results.get("summary_csv"), "count": len(results.get("proteins", {}))}, indent=2))


if __name__ == "__main__":
    main()
