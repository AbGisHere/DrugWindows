"""Batch UI with app.py-matching result tabs (read-only, no processing buttons)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import gradio as gr
import pandas as pd

from app import filter_poses_by_chain, render_admet_cards, render_dft_cards, visualize_docking_result
from batch import ProteinPipelineBatch, parse_protein_lines
from config import current_pdb_info
from visualization import show_structure


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())


def _protein_dir(output_dir: str, protein_name: str) -> Path:
    return Path(output_dir) / _safe_name(protein_name)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _find_first(step_dir: Path, patterns: List[str]) -> Path | None:
    for pat in patterns:
        matches = sorted(step_dir.rglob(pat))
        if matches:
            return matches[0]
    return None


def run_batch_ui(protein_lines: str, output_dir: str, run_dft: bool):
    proteins = parse_protein_lines(protein_lines)
    if not proteins:
        return (
            gr.update(value="⚠️ Enter at least one protein (one per line).", visible=True),
            gr.update(value=None, visible=False),
            gr.update(choices=[], value=None),
        )

    runner = ProteinPipelineBatch(output_base_dir=output_dir.strip() or "batch_results", run_dft=run_dft)
    result = runner.run_batch(proteins)

    rows = []
    for protein, data in result.get("proteins", {}).items():
        rows.append(
            {
                "Protein": protein,
                "Status": data.get("pipeline_status", "unknown"),
                "Started": data.get("start_time", ""),
                "Ended": data.get("end_time", ""),
            }
        )

    df = pd.DataFrame(rows)
    selected = proteins[0] if proteins else None
    return (
        gr.update(value="✅ Batch finished", visible=True),
        gr.update(value=df, visible=True),
        gr.update(choices=proteins, value=selected),
    )


def load_overview(protein_name: str, output_dir: str):
    if not protein_name:
        return ""

    summary_path = _protein_dir(output_dir, protein_name) / "pipeline_summary.json"
    if not summary_path.exists():
        return f"⚠️ Summary file missing: {summary_path}"

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = [f"### 🧬 {protein_name}", f"**Pipeline status:** {data.get('pipeline_status', 'unknown')}"]
    for k, v in data.get("steps", {}).items():
        lines.append(f"- **{k.replace('_', ' ').title()}**: {v.get('status', 'n/a')}")
    return "\n".join(lines)


def load_structure_results(protein_name: str, output_dir: str):
    if not protein_name:
        return gr.update(), "", None, gr.update(value="", visible=False)

    pdir = _protein_dir(output_dir, protein_name)
    step = pdir / "01_structure_search"
    pdb_file = _find_first(step, ["*.pdb"])

    if not pdb_file:
        return gr.update(value="", visible=False), "", None, gr.update(value="⚠️ No structure file found", visible=True)

    pdb_id = pdb_file.stem.upper()
    current_pdb_info.update({"pdb_path": str(pdb_file), "pdb_id": pdb_id})
    protein_text = pdb_file.read_text(encoding="utf-8", errors="ignore")
    info_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 24px; border-radius: 16px; color: white;">
        <h3 style='margin-top:0;'>🧬 Structure Loaded</h3>
        <p><strong>Input:</strong> {protein_name}</p>
        <p><strong>Target:</strong> {protein_name}</p>
        <p><strong>PDB ID:</strong> {pdb_id}</p>
    </div>
    """
    structure_html = show_structure(protein_text=protein_text, ligand_text=None, pdb_id=pdb_id, protein_name=protein_name)
    return gr.update(value=info_html, visible=True), structure_html, str(pdb_file), gr.update(value="✅ Structure loaded from batch results!", visible=True)


def load_ramachandran_results(protein_name: str, output_dir: str):
    step = _protein_dir(output_dir, protein_name) / "02_ramachandran"
    imgs = sorted(step.rglob("*.png"))[:4]
    stats = _find_first(step, ["*stats*.txt", "*.md", "*.txt"])
    stats_text = stats.read_text(encoding="utf-8", errors="ignore") if stats and stats.exists() else ""
    updates = [gr.update(value=str(img), visible=True) for img in imgs]
    while len(updates) < 4:
        updates.append(gr.update(value=None, visible=False))
    status = gr.update(value="✅ Ramachandran results loaded from batch.", visible=True) if imgs else gr.update(value="⚠️ No Ramachandran plots found.", visible=True)
    return [status, *updates, gr.update(value=stats_text, visible=bool(stats_text))]


def load_prep_results(protein_name: str, output_dir: str):
    step = _protein_dir(output_dir, protein_name) / "03_protein_preparation"
    pdbqt = _find_first(step, ["*.pdbqt"])
    viewer = ""
    if pdbqt:
        prepared_text = pdbqt.read_text(encoding="utf-8", errors="ignore")
        viewer = show_structure(protein_text=prepared_text, ligand_text=None, pdb_id="Prepared", protein_name=protein_name)
        current_pdb_info.update({"prepared_pdbqt": str(pdbqt)})

    return (
        gr.update(value="✅ Protein preparation results loaded from batch.", visible=True),
        gr.update(value=viewer, visible=True),
        gr.update(value=str(pdbqt) if pdbqt else None, visible=bool(pdbqt)),
    )


def load_binding_site_results(protein_name: str, output_dir: str):
    step = _protein_dir(output_dir, protein_name) / "04_binding_sites"
    combined = _find_first(step, ["combined_pockets.csv", "*.csv"])
    df = _read_csv(combined) if combined else pd.DataFrame()
    return (
        gr.update(value="✅ Binding-site predictions loaded from batch.", visible=True),
        gr.update(value=df, visible=not df.empty),
    )


def load_ligand_results(protein_name: str, output_dir: str):
    step = _protein_dir(output_dir, protein_name) / "05_ligand_analysis"
    csv = _find_first(step, ["ligand_classification_report.csv", "*.csv"])
    df = _read_csv(csv) if csv else pd.DataFrame()
    ligand_pdbs = sorted((step / "ligand_pdb").glob("*.pdb")) if (step / "ligand_pdb").exists() else []
    choices = [p.name for p in ligand_pdbs]
    return (
        gr.update(value="✅ Chemical Class Analysis loaded from batch.", visible=True),
        gr.update(value=df, visible=not df.empty),
        gr.update(value=str(csv) if csv else None, visible=bool(csv)),
        gr.update(visible=True, choices=choices, value=choices[0] if choices else None),
    )


def view_ligand_batch(protein_name: str, output_dir: str, ligand_file: str):
    if not protein_name or not ligand_file:
        return ""
    p = _protein_dir(output_dir, protein_name) / "05_ligand_analysis" / "ligand_pdb" / ligand_file
    if not p.exists():
        return f"❌ File not found: {p}"
    return show_structure(protein_text=None, ligand_text=p.read_text(encoding="utf-8", errors="ignore"), pdb_id="Ligand", protein_name=ligand_file)


def load_docking_results(protein_name: str, output_dir: str):
    step = _protein_dir(output_dir, protein_name) / "06_docking"
    csv = _find_first(step, ["docking_summary.csv", "*.csv"])
    df = _read_csv(csv) if csv else pd.DataFrame()

    prepared_pdbqt = _find_first(_protein_dir(output_dir, protein_name) / "03_protein_preparation", ["*.pdbqt"])
    if prepared_pdbqt:
        current_pdb_info.update({"pdb_path": str(prepared_pdbqt), "prepared_pdbqt": str(prepared_pdbqt)})
        if not df.empty:
            df["receptor_pdb_file"] = str(prepared_pdbqt)

    chains = sorted(df["chain"].dropna().unique().tolist()) if "chain" in df.columns and not df.empty else []
    return (
        gr.update(value="✅ Docking results loaded from batch.", visible=True),
        gr.update(value=df, visible=not df.empty),
        gr.update(choices=chains, value=chains[0] if chains else None, visible=True),
        gr.update(choices=[], value=None, visible=True),
    )


def load_admet_results(protein_name: str, output_dir: str):
    step = _protein_dir(output_dir, protein_name) / "07_admet"
    csv = _find_first(step, ["admet_results.csv", "*.csv"])
    df = _read_csv(csv) if csv else pd.DataFrame()
    cards = render_admet_cards(df) if not df.empty else ""
    return (
        gr.update(value="✅ ADMET results loaded from batch.", visible=True),
        gr.update(value=df, visible=not df.empty),
        gr.update(value=cards, visible=not df.empty),
        gr.update(value=str(csv) if csv else None, visible=bool(csv)),
    )

def load_dft_results(protein_name: str, output_dir: str):
    step = _protein_dir(output_dir, protein_name) / "08_dft"
    csv = _find_first(step, ["dft_batch_results.csv", "*.csv"])
    df = _read_csv(csv) if csv else pd.DataFrame()

    cards = ""
    if not df.empty:
        last = df.iloc[-1]
        # FIX: Updated keys to match exactly what dft.py outputs in the CSV
        homo = last.get("HOMO_eV", "-")
        lumo = last.get("LUMO_eV", "-")
        gap = last.get("Gap_eV", "-")
        cards = render_dft_cards(homo, lumo, gap)

    status = "✅ DFT results loaded from batch." if csv else "⚠️ DFT results not available for this batch."
    return (
        gr.update(value=status, visible=True),
        gr.update(value=cards, visible=bool(cards)),
        gr.update(value=df, visible=not df.empty),
        gr.update(value=str(csv) if csv else None, visible=bool(csv)),
    )

with gr.Blocks(theme=gr.themes.Soft(), title="Batch Protein Structure Finder & Analyzer") as demo:
    gr.HTML("<div class='main-header'><h1>🧬 Batch Protein Structure Finder & Analyzer</h1></div>")

    with gr.Tabs() as tabs:
        with gr.Tab("🔍 Batch Input", id=0):
            gr.Markdown("### Enter proteins for batch execution")
            with gr.Row():
                with gr.Column(scale=1):
                    protein_lines = gr.Textbox(label="Protein List (one per line)", lines=12, placeholder="KRAS\nPI3K\nmTOR")
                    output_dir = gr.Textbox(label="Output Root Folder", value="batch_results")
                    run_dft = gr.Checkbox(label="Run DFT step", value=False)
                    run_btn = gr.Button("🚀 Run Batch", variant="primary")
                    run_status = gr.Markdown(visible=False)
                with gr.Column(scale=2):
                    batch_df = gr.Dataframe(label="Batch Summary", visible=False)
            next_btn_0 = gr.Button("Next: Results Overview →", variant="primary")

        with gr.Tab("📊 Overview", id=1):
            gr.Markdown("### Batch Result Overview")
            protein_selector = gr.Dropdown(label="Select Protein", choices=[], interactive=True)
            protein_summary = gr.Markdown()
            with gr.Row():
                prev_btn_overview = gr.Button("← Previous", variant="secondary")
                next_btn_1 = gr.Button("Next: Structure Search →", variant="primary")

        with gr.Tab("🧬 Structure Search", id=2):
            gr.Markdown("### Step 1: Protein Structure Search")
            info_box = gr.HTML(visible=False)
            structure_viewer = gr.HTML(label="3D Structure Viewer")
            download_file = gr.File(label="Download PDB File", visible=False)
            search_status = gr.HTML(visible=False)
            with gr.Row():
                prev_btn_1 = gr.Button("← Previous", variant="secondary")
                next_btn_2 = gr.Button("Next: Ramachandran Analysis →", variant="primary")

        with gr.Tab("📈 Ramachandran Analysis", id=3):
            gr.Markdown("### Step 2: Ramachandran Analysis")
            ramplot_status = gr.HTML(visible=False)
            with gr.Row():
                plot1 = gr.Image(label="2D Ramachandran")
                plot2 = gr.Image(label="3D Ramachandran")
            with gr.Row():
                plot3 = gr.Image(label="2D with Outliers")
                plot4 = gr.Image(label="3D with Outliers")
            ramplot_stats = gr.Markdown(visible=False)
            with gr.Row():
                prev_btn_2 = gr.Button("← Previous", variant="secondary")
                next_btn_3 = gr.Button("Next: Protein Preparation →", variant="primary")

        with gr.Tab("🛠 Protein Preparation", id=4):
            gr.Markdown("### Step 3: Protein Preparation")
            prepare_status = gr.Markdown(visible=False)
            prepared_viewer = gr.HTML(label="Prepared Structure")
            prepared_download = gr.File(label="Download Prepared Protein", visible=False)
            with gr.Row():
                prev_btn_3 = gr.Button("← Previous", variant="secondary")
                next_btn_4 = gr.Button("Next: Binding Site Prediction →", variant="primary")

        with gr.Tab("🎯 Binding Site Prediction", id=5):
            gr.Markdown("### Step 4: Binding Site Prediction")
            prankweb_status = gr.Markdown(visible=False)
            prankweb_results = gr.Dataframe(label="Combined Pocket Analysis", visible=False)
            with gr.Row():
                prev_btn_4 = gr.Button("← Previous", variant="secondary")
                next_btn_5 = gr.Button("Next: Ligand Analysis →", variant="primary")

        with gr.Tab("🧪 Ligand Analysis", id=6):
            gr.Markdown("### Step 5: Ligand Class Analysis")
            ligand_status = gr.Markdown(visible=False)
            ligand_table = gr.Dataframe(label="Classification Results", visible=False)
            ligand_csv_download = gr.File(label="Download Classification CSV", visible=False)
            ligand_selector = gr.Dropdown(label="Select Ligand for 3D View", choices=[], interactive=True, visible=False)
            ligand_viewer = gr.HTML(label="Ligand 3D Viewer")
            with gr.Row():
                prev_btn_lig = gr.Button("← Previous", variant="secondary")
                next_btn_lig = gr.Button("Next: Docking →", variant="primary")

        with gr.Tab("🚀 Molecular Docking", id=7):
            gr.Markdown("### Molecular Docking (Multi-Chain)")
            docking_status = gr.HTML(visible=False)
            docking_summary = gr.Dataframe(visible=False)
            with gr.Row():
                chain_selector = gr.Dropdown(label="1. Select Chain Results", choices=[], interactive=True, visible=False)
                pose_selector = gr.Dropdown(label="2. Select Pose to View", choices=[], interactive=True, visible=False)
            view_pose_btn = gr.Button("View Pose & Interactions", variant="primary")
            with gr.Row():
                with gr.Column(scale=2):
                    docked_viewer = gr.HTML(label="3D Interaction Viewer")
                with gr.Column(scale=1):
                    docking_report_area = gr.Markdown(label="Pose Details", visible=True)
            with gr.Row():
                prev_btn_5 = gr.Button("← Previous", variant="secondary")
                next_btn_6 = gr.Button("Next: ADMET →", variant="primary")

        with gr.Tab("🧪 ADMET Analysis", id=8):
            gr.Markdown("### Drug-likeness & Safety Screening")
            admet_download = gr.File(label="Download CSV Report", visible=False)
            admet_status = gr.Markdown(visible=False)
            admet_table = gr.Dataframe(label="Detailed Report", visible=True)
            admet_results_view = gr.HTML(label="Analysis Cards", visible=True)
            with gr.Row():
                prev_btn_6 = gr.Button("← Previous", variant="secondary")
                next_btn_7 = gr.Button("Next: DFT Analysis →", variant="primary")

        with gr.Tab("⚛️ DFT Analysis", id=9):
            gr.Markdown("### Electronic Structure Analysis (ORCA)")
            gr.Markdown("Calculates HOMO, LUMO, and Band Gap using r2SCAN-3c DFT.<br>⚠️ **Note:** Processing is disabled here; this tab only displays batch outputs.")
            dft_status = gr.HTML(visible=False)
            dft_results_view = gr.HTML(label="Last Processed Metrics", visible=False)
            dft_table = gr.Dataframe(label="Batch Results Summary", visible=False)
            dft_download = gr.File(label="Download Batch CSV", visible=False)
            with gr.Row():
                prev_btn_dft = gr.Button("← Previous", variant="secondary")
                next_btn_dft = gr.Button("Finish / Start Over", variant="primary")

    # navigation
    next_btn_0.click(lambda: gr.Tabs(selected=1), None, tabs)
    prev_btn_overview.click(lambda: gr.Tabs(selected=0), None, tabs)
    next_btn_1.click(lambda: gr.Tabs(selected=2), None, tabs)
    prev_btn_1.click(lambda: gr.Tabs(selected=1), None, tabs)
    next_btn_2.click(lambda: gr.Tabs(selected=3), None, tabs)
    prev_btn_2.click(lambda: gr.Tabs(selected=2), None, tabs)
    next_btn_3.click(lambda: gr.Tabs(selected=4), None, tabs)
    prev_btn_3.click(lambda: gr.Tabs(selected=3), None, tabs)
    next_btn_4.click(lambda: gr.Tabs(selected=5), None, tabs)
    prev_btn_4.click(lambda: gr.Tabs(selected=4), None, tabs)
    next_btn_5.click(lambda: gr.Tabs(selected=6), None, tabs)
    prev_btn_lig.click(lambda: gr.Tabs(selected=5), None, tabs)
    next_btn_lig.click(lambda: gr.Tabs(selected=7), None, tabs)
    prev_btn_5.click(lambda: gr.Tabs(selected=6), None, tabs)
    next_btn_6.click(lambda: gr.Tabs(selected=8), None, tabs)
    prev_btn_6.click(lambda: gr.Tabs(selected=7), None, tabs)
    next_btn_7.click(lambda: gr.Tabs(selected=9), None, tabs)
    prev_btn_dft.click(lambda: gr.Tabs(selected=8), None, tabs)
    next_btn_dft.click(lambda: gr.Tabs(selected=0), None, tabs)

    run_btn.click(fn=run_batch_ui, inputs=[protein_lines, output_dir, run_dft], outputs=[run_status, batch_df, protein_selector])

    protein_selector.change(fn=load_overview, inputs=[protein_selector, output_dir], outputs=[protein_summary])
    protein_selector.change(fn=load_structure_results, inputs=[protein_selector, output_dir], outputs=[info_box, structure_viewer, download_file, search_status])
    protein_selector.change(fn=load_ramachandran_results, inputs=[protein_selector, output_dir], outputs=[ramplot_status, plot1, plot2, plot3, plot4, ramplot_stats])
    protein_selector.change(fn=load_prep_results, inputs=[protein_selector, output_dir], outputs=[prepare_status, prepared_viewer, prepared_download])
    protein_selector.change(fn=load_binding_site_results, inputs=[protein_selector, output_dir], outputs=[prankweb_status, prankweb_results])
    protein_selector.change(fn=load_ligand_results, inputs=[protein_selector, output_dir], outputs=[ligand_status, ligand_table, ligand_csv_download, ligand_selector])
    protein_selector.change(fn=load_docking_results, inputs=[protein_selector, output_dir], outputs=[docking_status, docking_summary, chain_selector, pose_selector])
    protein_selector.change(fn=load_admet_results, inputs=[protein_selector, output_dir], outputs=[admet_status, admet_table, admet_results_view, admet_download])
    protein_selector.change(fn=load_dft_results, inputs=[protein_selector, output_dir], outputs=[dft_status, dft_results_view, dft_table, dft_download])

    ligand_selector.change(fn=view_ligand_batch, inputs=[protein_selector, output_dir, ligand_selector], outputs=[ligand_viewer])

    chain_selector.change(fn=filter_poses_by_chain, inputs=[chain_selector, docking_summary], outputs=[pose_selector])
    view_pose_btn.click(fn=visualize_docking_result, inputs=[pose_selector, docking_summary], outputs=[docked_viewer, docking_report_area])


if __name__ == "__main__":
    demo.launch()