# app.py
"""
Main Gradio interface for Protein Structure Finder & Analyzer
Updated version: 
- DFT Analysis moved to the end (Tab 7).
- Runs DFT on ALL docked compounds in the folder (Batch Processing).
- Uses subprocess to switch environments (orca_env) for calculations.
"""

import os
import gradio as gr
import pandas as pd
import subprocess
import shutil
import glob
import time

# Import modules
from config import current_pdb_info, DOCKING_RESULTS_DIR
from ramachandran import run_ramplot
from prankweb import run_prankweb_prediction
from protein_prep import prepare_protein_meeko
from ligand_analysis import run_ligand_classification
from docking import run_molecular_docking
from admet_analysis import run_admet_prediction
from utils import map_disease_to_protein, find_best_pdb_structure
from visualization import show_structure

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def show_ram_loading():
    """Immediately shows a loading state for the Ramachandran tab."""
    return (
        gr.update(value="""
            <div style='padding: 20px; background: #e7f3ff; border-radius: 8px; color: #004085; border-left: 5px solid #007bff;'>
                <div style='display: flex; align-items: center;'>
                    <span style='margin-right: 10px; font-size: 1.5em;'>⏳</span>
                    <strong>Analyzing Structural Geometry...</strong> 
                </div>
                <p style='margin-top: 10px; font-size: 0.9em;'>Checking for missing residues and generating 2D/3D Ramachandran Map types. This may take a moment if SWISS-MODEL homology modeling is required.</p>
            </div>
        """, visible=True),
        gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False)
    )

def render_dft_cards(homo, lumo, gap):
    """Renders HTML cards for DFT Electronic Metrics."""
    return f"""
    <style>
        .dft-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; text-align: center; font-family: sans-serif; }}
        .dft-card {{ background: white; padding: 20px; border-radius: 12px; border: 1px solid #e5e7eb; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        .dft-val {{ font-size: 1.8em; font-weight: bold; color: #1f2937; margin: 10px 0; }}
        .dft-label {{ color: #6b7280; font-weight: 600; text-transform: uppercase; font-size: 0.85em; }}
        .gap-tag {{ background: #eff6ff; color: #2563eb; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; }}
    </style>
    <div class="dft-grid">
        <div class="dft-card">
            <div class="dft-label">HOMO Energy</div>
            <div class="dft-val">{homo} <span style="font-size:0.5em; color:#999;">eV</span></div>
            <div style="font-size:0.8em; color: #666;">Highest Occupied Molecular Orbital</div>
        </div>
        <div class="dft-card">
            <div class="dft-label">Band Gap</div>
            <div class="dft-val">{gap} <span style="font-size:0.5em; color:#999;">eV</span></div>
            <div class="gap-tag">Chemical Hardness Proxy</div>
        </div>
        <div class="dft-card">
            <div class="dft-label">LUMO Energy</div>
            <div class="dft-val">{lumo} <span style="font-size:0.5em; color:#999;">eV</span></div>
            <div style="font-size:0.8em; color: #666;">Lowest Unoccupied Molecular Orbital</div>
        </div>
    </div>
    """

def render_admet_cards(df):
    """Converts the ADMET DataFrame into a responsive HTML grid of cards."""
    if df is None or df.empty:
        return "<div style='padding:20px; text-align:center; color:#888;'>No data available to render.</div>"

    # --- Filtering Logic: Top 5 per Ligand ---
    filtered_rows = []
    if 'Ligand' in df.columns:
        grouped = df.groupby('Ligand')
        for ligand, group in grouped:
            group_sorted = group.sort_values(by=['Developability Score', 'Docking Score'], ascending=[False, True])
            top_5 = group_sorted.head(5)
            non_rejected = group_sorted[~group_sorted['Final Decision'].astype(str).str.contains("REJECT", case=False, na=False)]
            combined = pd.concat([top_5, non_rejected]).drop_duplicates()
            filtered_rows.append(combined)
        
        df_to_render = pd.concat(filtered_rows).sort_values(by=['Developability Score'], ascending=False) if filtered_rows else df 
    else:
        df_to_render = df

    cards_html = """
    <style>
        .admet-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 24px; padding: 10px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .admet-card { background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); transition: all 0.2s ease; color: #1f2937; display: flex; flex-direction: column; }
        .admet-card:hover { transform: translateY(-4px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); border-color: #d1d5db; }
        .card-header { display: flex; justify-content: space-between; align-items: start; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #f3f4f6; }
        .ligand-info h3 { margin: 0 0 4px 0; font-size: 1.25em; font-weight: 700; color: #111827; }
        .ligand-sub { font-size: 0.85em; color: #6b7280; font-weight: 500; }
        .badge { font-size: 0.75em; padding: 6px 10px; border-radius: 9999px; font-weight: 700; text-transform: uppercase; }
        .badge-green { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
        .badge-red { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
        .badge-yellow { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
        .bar-section { background: #f9fafb; padding: 12px; border-radius: 8px; border: 1px solid #f3f4f6; margin-bottom: 16px; }
        .bar-row { display: flex; align-items: center; margin-bottom: 8px; font-size: 0.85em; }
        .bar-label { width: 95px; color: #4b5563; font-weight: 600; }
        .bar-track { flex-grow: 1; height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden; margin: 0 12px; }
        .bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
        .bar-val { width: 45px; text-align: right; font-weight: bold; color: #1f2937; }
        .props-title { font-size: 0.75em; text-transform: uppercase; color: #9ca3af; font-weight: 700; margin-bottom: 8px; }
        .props-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
        .prop-tag { background: #f3f4f6; padding: 6px 10px; border-radius: 6px; font-size: 0.8em; display: flex; justify-content: space-between; border: 1px solid #e5e7eb; }
        .prop-name { color: #6b7280; font-weight: 500; }
        .prop-value { color: #111827; font-weight: 700; }
        .val-bad { color: #dc2626; }
        .val-good { color: #059669; }
    </style>
    <div class="admet-grid">
    """

    for _, row in df_to_render.iterrows():
        ligand = str(row.get('Ligand', 'Unknown'))
        pocket = str(row.get('Pocket', 'Unknown'))
        chain = str(row.get('Chain', '?'))
        pose = str(row.get('Pose', '1'))
        docking_score = row.get('Docking Score', 0)
        dev_score = row.get('Developability Score', 0)
        decision = str(row.get('Final Decision', 'REVIEW'))

        if "ACCEPT" in decision.upper(): badge_class, badge_text = "badge-green", "ACCEPTED"
        elif "REJECT" in decision.upper(): badge_class, badge_text = "badge-red", "REJECTED"
        else: badge_class, badge_text = "badge-yellow", "REVIEW"

        try:
            ds_val = float(docking_score)
            ds_width = min(100, max(0, (abs(ds_val) - 4) / 8 * 100))
        except: ds_val, ds_width = 0, 0
        
        try:
            dev_val = float(dev_score)
            dev_width = min(100, max(0, dev_val))
        except: dev_val, dev_width = 0, 0

        def get_val(key): return str(row.get(key, '-'))

        props_list = [
            ("SA Score", get_val('SA Score')),
            ("QED", get_val('QED')),
            ("Lipinski", get_val('Lipinski')),
            ("PAINS", get_val('PAINS')),
            ("Brenk", get_val('Brenk')),
            ("hERG", get_val('hERG')),
            ("Ames", get_val('Ames')),
            ("CYP3A4", get_val('CYP3A4 Inhibition')),
        ]

        grid_html = ""
        for name, val in props_list:
            val_class = ""
            if val in ['Yes', 'High', 'Positive', 'Fail']: val_class = "val-bad"
            elif val in ['No', 'Low', 'Negative', 'Pass']: val_class = "val-good"
            grid_html += f'<div class="prop-tag"><span class="prop-name">{name}</span><span class="prop-value {val_class}">{val}</span></div>'

        card = f"""
        <div class="admet-card">
            <div class="card-header">
                <div class="ligand-info"><h3>{ligand}</h3><div class="ligand-sub">Chain {chain} • {pocket} • Pose {pose}</div></div>
                <div class="{badge_class} badge">{badge_text}</div>
            </div>
            <div class="bar-section">
                <div class="bar-row"><span class="bar-label">Binding</span><div class="bar-track"><div class="bar-fill" style="width: {ds_width}%; background: #3b82f6;"></div></div><span class="bar-val">{ds_val:.2f}</span></div>
                <div class="bar-row"><span class="bar-label">Dev Score</span><div class="bar-track"><div class="bar-fill" style="width: {dev_width}%; background: #8b5cf6;"></div></div><span class="bar-val">{int(dev_val)}</span></div>
            </div>
            <div class="props-container"><div class="props-title">Molecular Properties</div><div class="props-grid">{grid_html}</div></div>
        </div>
        """
        cards_html += card

    cards_html += "</div>"
    return cards_html

# ==========================================
# 2. LOGIC HANDLERS
# ==========================================

def process_disease(user_input: str):
    """Returns updates for Search Tab AND clears Ramachandran Tab."""
    clear_ram = (gr.update(value="", visible=False), gr.update(value="", visible=False), gr.update(value=None, visible=False))
    
    if not user_input.strip():
        current_pdb_info.update({"pdb_id": None, "pdb_path": None})
        return (gr.update(visible=False), gr.update(value=""), gr.update(value=None), 
                gr.update(value="⚠️ Please enter a disease or protein name", visible=True)) + clear_ram * 2
    
    protein_name = map_disease_to_protein(user_input) or user_input.strip()
    result = find_best_pdb_structure(protein_name, max_check=100)
    
    if not result:
        current_pdb_info.update({"pdb_id": None, "pdb_path": None})
        return (gr.update(visible=False), gr.update(value=""), gr.update(value=None), 
                gr.update(value=f"❌ No suitable PDB structure found for: {protein_name}", visible=True)) + clear_ram * 2
    
    pdb_id, pdb_path = result
    
    try:
        with open(pdb_path, 'r') as f: pdb_content = f.read()
        current_pdb_info.update({"pdb_id": pdb_id, "pdb_path": pdb_path, "prepared_pdbqt": None, "docking_results": None, "prankweb_csv": None})
        
        info_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 24px; border-radius: 16px; color: white;">
            <h3 style='margin-top:0;'>🧬 Structure Loaded</h3>
            <p><strong>Input:</strong> {user_input}</p>
            <p><strong>Target:</strong> {protein_name}</p>
            <p><strong>PDB ID:</strong> {pdb_id}</p>
        </div>
        """
        structure_html = show_structure(protein_text=pdb_content, ligand_text=None, pdb_id=pdb_id, protein_name=protein_name)
        
        return (gr.update(value=info_html, visible=True), gr.update(value=structure_html), gr.update(value=pdb_path), 
                gr.update(value="✅ Structure loaded successfully!", visible=True)) + clear_ram * 2
        
    except Exception as e:
        return (gr.update(visible=False), gr.update(value=""), gr.update(value=None), 
                gr.update(value=f"❌ Error: {str(e)}", visible=True)) + clear_ram * 2

def process_ligand_analysis():
    """Logic for the new Ligand Analysis tab with dropdown update."""
    try:
        df, csv_path = run_ligand_classification("pdbqt")
        if df is None:
            return {ligand_status: gr.update(value=f"❌ {csv_path}", visible=True), ligand_table: gr.update(visible=False), 
                    ligand_csv_download: gr.update(visible=False), ligand_selector: gr.update(visible=False, choices=[])}
        
        ligand_files = df['Converted_PDB'].dropna().tolist() if 'Converted_PDB' in df.columns else []
        return {
            ligand_status: gr.update(value="✅ Chemical Class Analysis Complete", visible=True),
            ligand_table: gr.update(value=df, visible=True),
            ligand_csv_download: gr.update(value=csv_path, visible=True),
            ligand_selector: gr.update(visible=True, choices=ligand_files, value=ligand_files[0] if ligand_files else None)
        }
    except Exception as e:
        return {ligand_status: gr.update(value=f"❌ Analysis Error: {str(e)}", visible=True), ligand_table: gr.update(visible=False), 
                ligand_csv_download: gr.update(visible=False), ligand_selector: gr.update(visible=False, choices=[])}

def visualize_ligand_only(ligand_file):
    if not ligand_file: return ""
    ligand_path = os.path.join("ligand_pdb", ligand_file)
    if not os.path.exists(ligand_path): return f"❌ File not found: {ligand_path}"
    with open(ligand_path, 'r') as f: content = f.read()
    return show_structure(protein_text=None, ligand_text=content, pdb_id="Ligand", protein_name=ligand_file)

def filter_poses_by_chain(chain_selected, summary_df):
    if summary_df is None or summary_df.empty or not chain_selected: return gr.update(choices=[], value=None)
    chain_df = summary_df[summary_df['chain'] == chain_selected]
    choices = [(f"{r['ligand']} - {r['pocket']} (Pose {r['pose_number']}) | {r['binding_energy']:.2f} kcal", f"{r['pdb_file']}::{r['pose_number']}::{r['chain']}") for _, r in chain_df.iterrows()]
    return gr.update(choices=choices, value=choices[0][1] if choices else None)

def visualize_docking_result(selection_value: str, summary_df: pd.DataFrame):
    if not selection_value: return "⚠️ Please select a pose to view.", None
    try:
        parts = selection_value.split("::")
        if len(parts) < 3: return "❌ Invalid format.", None
        ligand_path, pose_num_str, chain_id = parts
        target_pose_num = int(pose_num_str)
        
        # Extract Metadata
        report_text = "### ⚠️ No detailed report data available."
        receptor_pdb_path = current_pdb_info.get("pdb_path")
        
        if summary_df is not None:
            row = summary_df[(summary_df['pdb_file'].astype(str) == str(ligand_path)) & (summary_df['pose_number'].astype(int) == target_pose_num)]
            if not row.empty:
                info = row.iloc[0]
                if 'receptor_pdb_file' in row.columns and os.path.exists(info['receptor_pdb_file']): receptor_pdb_path = info['receptor_pdb_file']
                
                # Load Report File
                from config import DOCKING_RESULTS_DIR
                report_path = os.path.join(DOCKING_RESULTS_DIR, "docking_reports", f"{info.get('chain')}_{info.get('ligand')}_{info.get('pocket')}_report.txt")
                ext_report = open(report_path, "r").read() if os.path.exists(report_path) else "Report missing."
                
                report_text = f"""
                ### 📄 Pose Details
                | Property | Value |
                | :--- | :--- |
                | **Ligand** | {info.get('ligand')} |
                | **Binding** | **{info.get('binding_energy')} kcal/mol** |
                
                ### 🧬 Interaction Profile
                ```text
                {ext_report}
                ```
                """
        
        # Load Visuals
        if not os.path.exists(ligand_path): return f"❌ Ligand file not found.", report_text
        with open(receptor_pdb_path, 'r') as f: protein_text = f.read()
        with open(ligand_path, 'r') as f: lines = f.readlines()
        
        model_lines, in_model = [], False
        for line in lines:
            if line.startswith("MODEL"):
                if int(line.split()[1]) == target_pose_num: in_model = True
            if in_model: model_lines.append(line)
            if line.startswith("ENDMDL") and in_model: break
            
        ligand_text = "".join(model_lines) if model_lines else "".join(lines)
        return show_structure(protein_text=protein_text, ligand_text=ligand_text, pdb_id="Docking", protein_name=f"Pose {target_pose_num}"), report_text

    except Exception as e: return f"❌ Visualization Error: {str(e)}", None

# ==========================================
# 3. DFT BATCH PROCESSOR (NEW)
# ==========================================

def update_dft_script_file(new_pdb_path):
    """Safely rewrites the PDB_FILE line in dft.py to point to the selected pose."""
    script_path = "dft.py"
    if not os.path.exists(script_path):
        raise FileNotFoundError("dft.py not found in current directory.")
        
    with open(script_path, "r") as f: lines = f.readlines()
    
    with open(script_path, "w") as f:
        for line in lines:
            if line.strip().startswith("PDB_FILE ="):
                f.write(f"PDB_FILE = {repr(new_pdb_path)}\n")
            else:
                f.write(line)

def process_dft_batch():
    """
    Scans the docking results folder for all PDBs and runs DFT on them sequentially.
    """
    # 1. Gather all files
    search_path = os.path.join(DOCKING_RESULTS_DIR, "**", "docked_pdb", "*.pdb")
    all_pdbs = glob.glob(search_path, recursive=True)
    
    if not all_pdbs:
        yield { dft_status: gr.update(value="❌ No docked PDB files found in docking_results.", visible=True) }
        return

    # CSV Cleanup
    batch_csv_path = "dft_batch_results.csv"
    if os.path.exists(batch_csv_path): os.remove(batch_csv_path) 
    
    # SAFETY: Clear the single-run output file to ensure we don't read stale data
    single_csv = "orca_electronic_metrics.csv"
    if os.path.exists(single_csv): os.remove(single_csv)

    all_results = []
    total = len(all_pdbs)
    
    yield { dft_status: gr.update(value=f"⏳ Starting DFT Batch for {total} compounds...", visible=True) }

    for i, pdb_path in enumerate(all_pdbs):
        pdb_name = os.path.basename(pdb_path)
        
        # UI Update for current file
        yield { dft_status: gr.update(value=f"⚙️ Processing ({i+1}/{total}): {pdb_name} ...", visible=True) }
        
        # 1. Update Script
        try:
            update_dft_script_file(pdb_path)
        except Exception as e:
            print(f"Failed to update script for {pdb_name}: {e}")
            continue
            
        # 2. Run Subprocess (Switch Env)
        # This child process runs in orca_env, then dies. Main process stays in drugv.
        command = f'cmd /c "call activate orca_env && python dft.py"'
        try:
            process = subprocess.run(command, capture_output=True, text=True, shell=True)
            if process.returncode != 0:
                print(f"Error on {pdb_name}: {process.stderr}")
                # Continue to next file even if this one fails
        except Exception as e:
            print(f"Execution error on {pdb_name}: {e}")
            
        # 3. Read Single Result
        if os.path.exists(single_csv):
            try:
                # Read the last line (most recent run)
                df = pd.read_csv(single_csv)
                if not df.empty:
                    last_row = df.iloc[-1].to_dict()
                    # Ensure filename is correct in the record
                    last_row['Filename'] = pdb_name
                    all_results.append(last_row)
                    
                    # Append to Batch CSV immediately
                    pd.DataFrame([last_row]).to_csv(batch_csv_path, mode='a', header=not os.path.exists(batch_csv_path), index=False)
            except:
                pass

        # Update Table in UI incrementally
        current_df = pd.DataFrame(all_results)
        yield { dft_table: gr.update(value=current_df, visible=True) }

    # Final Done State
    final_df = pd.DataFrame(all_results)
    if not final_df.empty:
        # Render cards for the LAST item just as an example
        last = final_df.iloc[-1]
        cards = render_dft_cards(last.get('HOMO_eV', '-'), last.get('LUMO_eV', '-'), last.get('Gap_eV', '-'))
        yield {
            dft_status: gr.update(value=f"✅ Batch Processing Complete. Processed {total} files.", visible=True),
            dft_table: gr.update(value=final_df, visible=True),
            dft_results_view: gr.update(value=cards, visible=True),
            dft_download: gr.update(value=batch_csv_path, visible=True)
        }
    else:
        yield { dft_status: gr.update(value="❌ Batch finished but no results generated.", visible=True) }


def process_admet():
    """Run ADMET analysis and render the result as UI Cards."""
    try:
        result = run_admet_prediction()
        if result is None:
            return {admet_status: gr.update(value="❌ Analysis Failed.", visible=True), admet_results_view: gr.update(visible=False), 
                    admet_download: gr.update(visible=False), admet_table: gr.update(visible=False)}
        
        msg, df, csv_path = result
        return {
            admet_status: gr.update(value=f"✅ {msg}", visible=True),
            admet_table: gr.update(value=df, visible=True),
            admet_results_view: gr.update(value=render_admet_cards(df), visible=True),
            admet_download: gr.update(value=csv_path, visible=True)
        }
    except Exception as e:
        return {admet_status: gr.update(value=f"❌ System Error: {str(e)}", visible=True), admet_results_view: gr.update(visible=False), 
                admet_download: gr.update(visible=False), admet_table: gr.update(visible=False)}

# ==========================================
# 4. UI LAYOUT
# ==========================================

with gr.Blocks(theme=gr.themes.Soft(), title="Protein Structure Finder & Analyzer") as demo:
    
    gr.HTML("<div class='main-header'><h1>🧬 Protein Structure Finder & Analyzer</h1></div>")
    
    with gr.Tabs() as tabs:
        # Tab 0: Search
        with gr.Tab("🔍 Structure Search", id=0):
            gr.Markdown("### Protein Identification")
            with gr.Row():
                with gr.Column(scale=1):
                    disease_input = gr.Textbox(label="Enter Disease/Gene/PDB ID", placeholder="e.g., Alzheimer's, Insulin")
                    search_btn = gr.Button("🚀 Search Best Structure", variant="primary")
                    info_box = gr.HTML(visible=False)
                    search_status = gr.Markdown(visible=False)
                    download_file = gr.File(label="Download PDB", visible=True)
                with gr.Column(scale=2):
                    structure_viewer = gr.HTML(label="3D Viewer")
            with gr.Row():
                next_btn_0 = gr.Button("Next: Ramachandran Analysis →", variant="primary")
        
        # Tab 1: Ramachandran
        with gr.Tab("📊 Ramachandran Analysis", id=1):
            gr.Markdown("### Structural Quality Assessment")
            ramplot_btn = gr.Button("🔬 Run Ramachandran Analysis", variant="secondary")
            ramplot_status = gr.HTML(visible=False)
            ramplot_stats = gr.HTML(visible=False) 
            with gr.Row():
                plot1 = gr.Image(label="Map Type 2D", visible=False)
                plot2 = gr.Image(label="Map Type 3D", visible=False)
            with gr.Row():
                plot3 = gr.Image(label="Std Map 2D", visible=False)
                plot4 = gr.Image(label="Std Map 3D", visible=False)
            with gr.Row():
                prev_btn_1 = gr.Button("← Previous", variant="secondary")
                next_btn_1 = gr.Button("Next: Protein Preparation →", variant="primary")
        
        # Tab 2: Protein Prep
        with gr.Tab("⚙️ Protein Preparation", id=2):
            gr.Markdown("### File Preparation for Simulation")
            prepare_btn = gr.Button("🔧 Prepare Protein", variant="secondary")
            prepare_status = gr.HTML(visible=False)
            with gr.Row():
                prepared_viewer = gr.HTML(label="Prepared Viewer")
                prepared_download = gr.File(label="Download PDBQT")
            with gr.Row():
                prev_btn_2 = gr.Button("← Previous", variant="secondary")
                next_btn_2 = gr.Button("Next: Binding Site Prediction →", variant="primary")

        # Tab 3: PrankWeb & Fpocket
        with gr.Tab("🎯 Binding Site Prediction", id=3):
            gr.Markdown("### Active Site Prediction (P2Rank & Fpocket)")
            prankweb_btn = gr.Button("🔮 Run Prediction (P2Rank + Fpocket)", variant="secondary")
            prankweb_status = gr.HTML(visible=False)
            with gr.Row():
                with gr.Column(scale=1):
                    prankweb_results = gr.Dataframe(label="P2Rank Results (CSV)", visible=False)
                with gr.Column(scale=1):
                    fpocket_results = gr.TextArea(label="Fpocket Report (_info.txt)", visible=False, lines=20)
            with gr.Row():
                prev_btn_3 = gr.Button("← Previous", variant="secondary")
                next_btn_3 = gr.Button("Next: Ligand Analysis →", variant="primary")

        # Tab 4: Ligand Analysis
        with gr.Tab("🧪 Ligand Analysis", id=4):
            gr.Markdown("### Chemical Class & Complexity Analysis")
            ligand_analyze_btn = gr.Button("🔍 Analyze Ligands", variant="secondary")
            ligand_status = gr.Markdown(visible=False)
            ligand_table = gr.Dataframe(label="Ligand Properties", visible=False)
            ligand_csv_download = gr.File(label="Download Classification Report", visible=False)
            gr.Markdown("### 3D Ligand Visualization (PDB format)")
            with gr.Row():
                with gr.Column(scale=1):
                    ligand_selector = gr.Dropdown(label="Select Ligand to View", choices=[], interactive=True, visible=False)
                with gr.Column(scale=2):
                    ligand_viewer = gr.HTML(label="Ligand 3D Viewer")
            with gr.Row():
                prev_btn_lig = gr.Button("← Previous", variant="secondary")
                next_btn_lig = gr.Button("Next: Docking →", variant="primary")

        # Tab 5: Docking
        with gr.Tab("🚀 Molecular Docking", id=5):
            gr.Markdown("### Molecular Docking (Multi-Chain)")
            docking_btn = gr.Button("Run Docking (All Chains)", variant="secondary")
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
                prev_btn_4 = gr.Button("← Previous", variant="secondary")
                next_btn_4 = gr.Button("Next: ADMET →", variant="primary")

        # Tab 6: ADMET Analysis (Swapped Position)
        with gr.Tab("🧪 ADMET Analysis", id=6):
            gr.Markdown("### Drug-likeness & Safety Screening")
            with gr.Row():
                admet_btn = gr.Button("🚀 Run ADMET Analysis", variant="primary")
                admet_download = gr.File(label="Download CSV Report", visible=False)
            admet_status = gr.Markdown(visible=False)
            admet_table = gr.Dataframe(label="Detailed Report", visible=True)
            admet_results_view = gr.HTML(label="Analysis Cards", visible=True)
            with gr.Row():
                prev_btn_5 = gr.Button("← Previous", variant="secondary")
                next_btn_5 = gr.Button("Next: DFT Analysis →", variant="primary")

        # Tab 7: DFT Analysis (Final Step)
        with gr.Tab("⚛️ DFT Analysis", id=7):
            gr.Markdown("### Electronic Structure Analysis (ORCA)")
            gr.Markdown("Calculates HOMO, LUMO, and Band Gap using r2SCAN-3c DFT.<br>⚠️ **Note:** Processing all docked compounds sequentially in `orca_env`.")
            
            dft_btn = gr.Button("⚡ Run DFT on ALL Docked Compounds", variant="primary")
            dft_status = gr.HTML(visible=False)
            
            dft_results_view = gr.HTML(label="Last Processed Metrics", visible=False)
            dft_table = gr.Dataframe(label="Batch Results Summary", visible=False)
            dft_download = gr.File(label="Download Batch CSV", visible=False)
            
            with gr.Row():
                prev_btn_dft = gr.Button("← Previous", variant="secondary")
                next_btn_dft = gr.Button("Finish / Start Over", variant="primary")

    # Events
    next_btn_0.click(lambda: gr.Tabs(selected=1), None, tabs)
    prev_btn_1.click(lambda: gr.Tabs(selected=0), None, tabs)
    next_btn_1.click(lambda: gr.Tabs(selected=2), None, tabs)
    prev_btn_2.click(lambda: gr.Tabs(selected=1), None, tabs)
    next_btn_2.click(lambda: gr.Tabs(selected=3), None, tabs) 
    prev_btn_3.click(lambda: gr.Tabs(selected=2), None, tabs)
    next_btn_3.click(lambda: gr.Tabs(selected=4), None, tabs) 
    prev_btn_lig.click(lambda: gr.Tabs(selected=3), None, tabs)
    next_btn_lig.click(lambda: gr.Tabs(selected=5), None, tabs)
    
    # Docking to ADMET
    prev_btn_4.click(lambda: gr.Tabs(selected=4), None, tabs)
    next_btn_4.click(lambda: gr.Tabs(selected=6), None, tabs)
    
    # ADMET to DFT
    prev_btn_5.click(lambda: gr.Tabs(selected=5), None, tabs)
    next_btn_5.click(lambda: gr.Tabs(selected=7), None, tabs)

    # DFT to Start
    prev_btn_dft.click(lambda: gr.Tabs(selected=6), None, tabs)
    next_btn_dft.click(lambda: gr.Tabs(selected=0), None, tabs)

    # Core Connections
    search_btn.click(fn=process_disease, inputs=[disease_input], outputs=[info_box, structure_viewer, download_file, search_status, ramplot_status, ramplot_stats, plot1, plot2, plot3, plot4])
    
    ramplot_btn.click(fn=show_ram_loading, outputs=[ramplot_status, plot1, plot2, plot3, plot4, ramplot_stats]).then(fn=run_ramplot, inputs=[], outputs=[ramplot_status, plot1, plot2, plot3, plot4, ramplot_stats])
    
    prepare_btn.click(fn=prepare_protein_meeko, inputs=[], outputs=[prepare_status, prepared_viewer, prepared_download, info_box, structure_viewer, download_file, search_status, ramplot_status, plot1, plot2, plot3, plot4, ramplot_stats])
    
    prankweb_btn.click(fn=run_prankweb_prediction, inputs=[], outputs=[prankweb_status, prankweb_results, fpocket_results])
    
    ligand_analyze_btn.click(fn=process_ligand_analysis, inputs=[], outputs={ligand_status, ligand_table, ligand_csv_download, ligand_selector})
    ligand_selector.change(fn=visualize_ligand_only, inputs=[ligand_selector], outputs=[ligand_viewer])
    
    docking_btn.click(fn=run_molecular_docking, inputs=[], outputs=[docking_status, docking_summary, chain_selector, pose_selector])
    chain_selector.change(fn=filter_poses_by_chain, inputs=[chain_selector, docking_summary], outputs=[pose_selector])
    view_pose_btn.click(fn=visualize_docking_result, inputs=[pose_selector, docking_summary], outputs=[docked_viewer, docking_report_area])
    
    admet_btn.click(fn=process_admet, inputs=[], outputs={admet_status, admet_table, admet_results_view, admet_download})

    # DFT BATCH TRIGGER
    dft_btn.click(
        fn=process_dft_batch,
        inputs=[],
        outputs={dft_status, dft_results_view, dft_table, dft_download}
    )

if __name__ == "__main__":
    force_dark_mode = """
    function() {
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """
    demo.launch(share=True, js=force_dark_mode)