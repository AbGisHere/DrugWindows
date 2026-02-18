"""Batch UI with app.py-style tabbed result browsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import pandas as pd

from batch import PIPELINE_STEPS, ProteinPipelineBatch, parse_protein_lines

STEP_KEYS = [k for k, _ in PIPELINE_STEPS]
STEP_LABELS = [v for _, v in PIPELINE_STEPS]
STEP_MAP = dict(PIPELINE_STEPS)


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())


def _build_step_choices(output_dir: str, protein_name: str, step_key: str) -> List[str]:
    if not protein_name:
        return []
    step_dir = Path(output_dir) / _safe_name(protein_name) / step_key
    if not step_dir.exists():
        return []
    return [str(p) for p in sorted(step_dir.rglob("*")) if p.is_file()]


def _render_file(path_str: str):
    if not path_str:
        return "", pd.DataFrame(), None, None

    p = Path(path_str)
    if not p.exists() or not p.is_file():
        return f"⚠️ File not found: {path_str}", pd.DataFrame(), None, None

    suf = p.suffix.lower()
    text = ""
    df = pd.DataFrame()
    image = None
    download = str(p)

    if suf == ".csv":
        try:
            df = pd.read_csv(p).head(100)
        except Exception as exc:
            text = f"Failed to read CSV: {exc}"
    elif suf in {".png", ".jpg", ".jpeg"}:
        image = str(p)
    elif suf in {".txt", ".log", ".md", ".json", ".pdb", ".pdbqt"}:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")[:12000]
        except Exception as exc:
            text = f"Failed to read text: {exc}"
    else:
        text = f"Selected file: {p.name}"

    return text, df, image, download


def run_batch_ui(protein_lines: str, output_dir: str, run_dft: bool):
    proteins = parse_protein_lines(protein_lines)
    if not proteins:
        return (
            gr.update(value="⚠️ Enter at least one protein (one per line).", visible=True),
            gr.update(value=None, visible=False),
            gr.update(choices=[], value=None),
            *[gr.update(choices=[], value=None) for _ in STEP_KEYS],
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

    selected = proteins[0]
    per_step_updates = []
    for step_key in STEP_KEYS:
        choices = _build_step_choices(output_dir, selected, step_key)
        per_step_updates.append(gr.update(choices=choices, value=choices[0] if choices else None))

    return (
        gr.update(value="✅ Batch finished", visible=True),
        gr.update(value=df, visible=True),
        gr.update(choices=proteins, value=selected),
        *per_step_updates,
    )


def on_protein_change(protein_name: str, output_dir: str):
    updates = []
    for step_key in STEP_KEYS:
        choices = _build_step_choices(output_dir, protein_name, step_key)
        updates.append(gr.update(choices=choices, value=choices[0] if choices else None))
    return updates


def load_protein_summary(protein_name: str, output_dir: str):
    if not protein_name:
        return ""

    summary_path = Path(output_dir) / _safe_name(protein_name) / "pipeline_summary.json"
    if not summary_path.exists():
        return f"⚠️ Summary file missing: {summary_path}"

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = [f"### 🧬 {protein_name}", f"**Pipeline status:** {data.get('pipeline_status', 'unknown')}"]

    key_map = {
        "01_structure_search": "structure_search",
        "02_ramachandran": "ramachandran",
        "03_protein_preparation": "protein_preparation",
        "04_binding_sites": "binding_sites",
        "05_ligand_analysis": "ligand_analysis",
        "06_docking": "docking",
        "07_admet": "admet",
        "08_dft": "dft",
    }
    for step_key, label in PIPELINE_STEPS:
        step_data = data.get("steps", {}).get(key_map[step_key], {})
        lines.append(f"- **{label}**: {step_data.get('status', 'n/a')}")

    return "\n".join(lines)


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
            next_btn_0 = gr.Button("Next: Results Browser →", variant="primary")

        with gr.Tab("📊 Overview", id=1):
            gr.Markdown("### Batch Result Overview")
            protein_selector = gr.Dropdown(label="Select Protein", choices=[], interactive=True)
            protein_summary = gr.Markdown()
            next_btn_1 = gr.Button("Next: Structure Search →", variant="primary")

        tab_components: Dict[str, Tuple] = {}
        tab_defs = [
            (2, "🧬 Structure Search", "01_structure_search"),
            (3, "📈 Ramachandran", "02_ramachandran"),
            (4, "🛠 Protein Preparation", "03_protein_preparation"),
            (5, "🎯 Binding Sites", "04_binding_sites"),
            (6, "🧪 Ligand Analysis", "05_ligand_analysis"),
            (7, "🚀 Docking", "06_docking"),
            (8, "💊 ADMET", "07_admet"),
            (9, "⚛️ DFT", "08_dft"),
        ]

        for tab_id, tab_title, step_key in tab_defs:
            with gr.Tab(tab_title, id=tab_id):
                gr.Markdown(f"### {STEP_MAP[step_key]} Results")
                selector = gr.Dropdown(label=f"Select file from {STEP_MAP[step_key]}", choices=[], interactive=True)
                with gr.Row():
                    with gr.Column(scale=1):
                        file_download = gr.File(label="Download selected file")
                        text_preview = gr.Markdown(label="Text Preview")
                    with gr.Column(scale=2):
                        csv_preview = gr.Dataframe(label="CSV Preview")
                        img_preview = gr.Image(label="Image Preview", type="filepath")
                tab_components[step_key] = (selector, text_preview, csv_preview, img_preview, file_download)

                selector.change(fn=_render_file, inputs=[selector], outputs=[text_preview, csv_preview, img_preview, file_download])

    # navigation
    next_btn_0.click(lambda: gr.Tabs(selected=1), None, tabs)
    next_btn_1.click(lambda: gr.Tabs(selected=2), None, tabs)

    # run batch
    run_btn.click(
        fn=run_batch_ui,
        inputs=[protein_lines, output_dir, run_dft],
        outputs=[run_status, batch_df, protein_selector] + [tab_components[k][0] for k in STEP_KEYS],
    )

    # protein switch updates all step dropdowns + summary
    protein_selector.change(
        fn=load_protein_summary,
        inputs=[protein_selector, output_dir],
        outputs=[protein_summary],
    )
    protein_selector.change(
        fn=on_protein_change,
        inputs=[protein_selector, output_dir],
        outputs=[tab_components[k][0] for k in STEP_KEYS],
    )


if __name__ == "__main__":
    demo.launch()
