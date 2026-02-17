"""Gradio UI for batch pipeline execution and result browsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import pandas as pd

from batch import PIPELINE_STEPS, ProteinPipelineBatch, parse_protein_lines


STEP_LABEL_TO_KEY = {label: key for key, label in PIPELINE_STEPS}


def _status_badge(status: str) -> str:
    status = (status or "unknown").lower()
    if status == "success" or status == "completed":
        return "<span style='background:#d4edda;color:#155724;padding:4px 10px;border-radius:12px;font-weight:700;'>SUCCESS</span>"
    if status == "failed" or "error" in status:
        return "<span style='background:#f8d7da;color:#721c24;padding:4px 10px;border-radius:12px;font-weight:700;'>FAILED</span>"
    if status == "skipped":
        return "<span style='background:#fff3cd;color:#856404;padding:4px 10px;border-radius:12px;font-weight:700;'>SKIPPED</span>"
    return "<span style='background:#e2e3e5;color:#383d41;padding:4px 10px;border-radius:12px;font-weight:700;'>UNKNOWN</span>"


def run_batch_from_ui(protein_lines: str, output_dir: str, run_dft: bool):
    proteins = parse_protein_lines(protein_lines)
    if not proteins:
        return (
            gr.update(value="⚠️ Please enter at least one protein (one per line).", visible=True),
            gr.update(value=None, visible=False),
            gr.update(choices=[], value=None, visible=True),
            gr.update(value="", visible=False),
        )

    runner = ProteinPipelineBatch(output_base_dir=output_dir.strip() or "batch_results", run_dft=run_dft)
    result = runner.run_batch(proteins)

    rows = []
    for protein, data in result.get("proteins", {}).items():
        rows.append({
            "protein": protein,
            "status": data.get("pipeline_status", "unknown"),
            "start_time": data.get("start_time", ""),
            "end_time": data.get("end_time", ""),
        })
    df = pd.DataFrame(rows)

    summary_html = f"""
    <div style='padding:16px;border-radius:12px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;'>
        <h3 style='margin:0 0 8px 0;'>✅ Batch Run Complete</h3>
        <p style='margin:0;'><b>Proteins processed:</b> {len(rows)}</p>
        <p style='margin:0;'><b>Summary CSV:</b> {result.get('summary_csv')}</p>
        <p style='margin:0;'><b>Index JSON:</b> {result.get('batch_index')}</p>
    </div>
    """

    choices = list(result.get("proteins", {}).keys())
    return (
        gr.update(value="✅ Batch pipeline finished.", visible=True),
        gr.update(value=df, visible=True),
        gr.update(choices=choices, value=choices[0] if choices else None, visible=True),
        gr.update(value=summary_html, visible=True),
    )


def _step_dir_for(protein_name: str, output_dir: str, step_label: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in protein_name.strip())
    step_key = STEP_LABEL_TO_KEY[step_label]
    return Path(output_dir) / safe / step_key


def load_protein_overview(protein_name: str, output_dir: str):
    if not protein_name:
        return gr.update(value="", visible=False)

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in protein_name.strip())
    summary_path = Path(output_dir) / safe / "pipeline_summary.json"
    if not summary_path.exists():
        return gr.update(value=f"⚠️ Summary not found: {summary_path}", visible=True)

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    map_key = {
        "01_structure_search": "structure_search",
        "02_ramachandran": "ramachandran",
        "03_protein_preparation": "protein_preparation",
        "04_binding_sites": "binding_sites",
        "05_ligand_analysis": "ligand_analysis",
        "06_docking": "docking",
        "07_admet": "admet",
        "08_dft": "dft",
    }
    for step_key, step_label in PIPELINE_STEPS:
        step_data = data.get("steps", {}).get(map_key[step_key], {})
        rows.append(f"<tr><td>{step_label}</td><td>{_status_badge(step_data.get('status', 'N/A'))}</td></tr>")

    html = f"""
    <div style='background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;'>
      <h3 style='margin-top:0;'>🧬 Protein: {protein_name}</h3>
      <p><b>Pipeline status:</b> {data.get('pipeline_status', 'unknown')}</p>
      <table style='width:100%;border-collapse:collapse;'>
        <thead><tr><th style='text-align:left;'>Step</th><th style='text-align:left;'>Status</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """
    return gr.update(value=html, visible=True)


def load_step_details(protein_name: str, step_label: str, output_dir: str):
    if not protein_name or not step_label:
        return None, [], "", None, []

    step_dir = _step_dir_for(protein_name, output_dir, step_label)
    if not step_dir.exists():
        return pd.DataFrame(), [], f"⚠️ Step folder not found: {step_dir}", None, []

    csv_files = sorted(step_dir.rglob("*.csv"))
    image_files = sorted([p for p in step_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    text_files = sorted([p for p in step_dir.rglob("*") if p.suffix.lower() in {".txt", ".md", ".json", ".log"}])
    all_files = [str(p) for p in sorted(step_dir.rglob("*")) if p.is_file()]

    preview_df = pd.DataFrame()
    if csv_files:
        try:
            preview_df = pd.read_csv(csv_files[0]).head(50)
        except Exception:
            preview_df = pd.DataFrame()

    text_preview = ""
    if text_files:
        try:
            text_preview = text_files[0].read_text(encoding="utf-8", errors="ignore")[:5000]
        except Exception as exc:
            text_preview = f"Failed to read text preview: {exc}"

    file_table = pd.DataFrame({"files": all_files}) if all_files else pd.DataFrame({"files": []})
    image_gallery = [(str(path), path.name) for path in image_files]

    first_csv = str(csv_files[0]) if csv_files else None
    return file_table, image_gallery, text_preview, first_csv, preview_df


with gr.Blocks(theme=gr.themes.Soft(), title="Batch Protein Pipeline") as demo:
    gr.HTML("<div class='main-header'><h1>🧪 Batch Protein Structure Finder & Analyzer</h1></div>")

    with gr.Tabs() as tabs:
        with gr.Tab("1️⃣ Batch Input", id=0):
            gr.Markdown("### Enter one protein per line and run the full pipeline batch")
            with gr.Row():
                with gr.Column(scale=1):
                    protein_lines = gr.Textbox(
                        label="Protein List",
                        lines=12,
                        placeholder="KRAS\nPI3K\nmTOR\nEGFR",
                    )
                    output_dir = gr.Textbox(label="Output Directory", value="batch_results")
                    run_dft = gr.Checkbox(label="Run DFT step", value=False)
                    run_btn = gr.Button("🚀 Run Batch Pipeline", variant="primary")
                    run_status = gr.Markdown(visible=False)
                with gr.Column(scale=2):
                    batch_summary_html = gr.HTML(visible=False)
                    batch_summary_df = gr.Dataframe(label="Batch Summary", visible=False)
            next_btn = gr.Button("Next: View Results →", variant="secondary")

        with gr.Tab("2️⃣ Batch Results Browser", id=1):
            gr.Markdown("### Select a protein and inspect results for each pipeline step")
            protein_selector = gr.Dropdown(label="Choose Protein", choices=[], interactive=True)
            protein_overview = gr.HTML(visible=False)

            with gr.Row():
                step_selector = gr.Dropdown(
                    label="Choose Step",
                    choices=[label for _, label in PIPELINE_STEPS],
                    value="Structure Search",
                    interactive=True,
                )
                refresh_btn = gr.Button("Refresh Step View", variant="secondary")

            with gr.Row():
                with gr.Column(scale=1):
                    files_df = gr.Dataframe(label="Files in Step Folder")
                    csv_download = gr.File(label="Primary CSV Download")
                with gr.Column(scale=2):
                    csv_preview = gr.Dataframe(label="CSV Preview (first 50 rows)")
                    text_preview = gr.Markdown(label="Text Preview")
            step_gallery = gr.Gallery(label="Step Images", columns=3, rows=2, height=360)

    next_btn.click(lambda: gr.Tabs(selected=1), inputs=None, outputs=tabs)

    run_btn.click(
        fn=run_batch_from_ui,
        inputs=[protein_lines, output_dir, run_dft],
        outputs=[run_status, batch_summary_df, protein_selector, batch_summary_html],
    )

    protein_selector.change(
        fn=load_protein_overview,
        inputs=[protein_selector, output_dir],
        outputs=[protein_overview],
    )

    refresh_btn.click(
        fn=load_step_details,
        inputs=[protein_selector, step_selector, output_dir],
        outputs=[files_df, step_gallery, text_preview, csv_download, csv_preview],
    )


if __name__ == "__main__":
    demo.launch()
