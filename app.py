# app.py
"""
Main Gradio interface for Protein Structure Finder & Analyzer
Updated version: Wires Protein Prep to update all tabs on retry.
Includes custom White-Themed UI for ADMET Analysis.
"""

import os
import gradio as gr
import pandas as pd

# Import modules
from config import current_pdb_info
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

def render_admet_cards(df):
    """
    Converts the ADMET DataFrame into a responsive HTML grid of cards.
    Style: White background cards with detailed property grids.
    """
    if df is None or df.empty:
        return "<div style='padding:20px; text-align:center; color:#888;'>No data available to render.</div>"

    cards_html = """
    <style>
        .admet-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 24px;
            padding: 10px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .admet-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: all 0.2s ease;
            color: #1f2937;
            display: flex;
            flex-direction: column;
        }
        .admet-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            border-color: #d1d5db;
        }
        
        /* Header Section */
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid #f3f4f6;
        }
        .ligand-info h3 {
            font-size: 1.25em;
            font-weight: 700;
            color: #111827;
            margin: 0 0 4px 0;
            line-height: 1.2;
        }
        .ligand-sub {
            font-size: 0.85em;
            color: #6b7280;
            font-weight: 500;
        }
        
        /* Decision Badge */
        .badge {
            font-size: 0.75em;
            padding: 6px 10px;
            border-radius: 9999px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .badge-green { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
        .badge-red { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
        .badge-yellow { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
        
        /* Progress Bars */
        .bar-section {
            background: #f9fafb;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #f3f4f6;
            margin-bottom: 16px;
        }
        .bar-row {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.85em;
        }
        .bar-row:last-child { margin-bottom: 0; }
        .bar-label {
            width: 95px;
            color: #4b5563;
            font-weight: 600;
        }
        .bar-track {
            flex-grow: 1;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin: 0 12px;
        }
        .bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .bar-val {
            width: 45px;
            text-align: right;
            font-weight: bold;
            color: #1f2937;
        }

        /* Properties Grid */
        .props-container {
            margin-top: auto;
        }
        .props-title {
            font-size: 0.75em;
            text-transform: uppercase;
            color: #9ca3af;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: 0.05em;
        }
        .props-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }
        .prop-tag {
            background: #f3f4f6;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 0.8em;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid #e5e7eb;
        }
        .prop-name { color: #6b7280; font-weight: 500; }
        .prop-value { color: #111827; font-weight: 700; }
        
        /* Conditional Formatting for Property Values */
        .val-bad { color: #dc2626; }
        .val-good { color: #059669; }
        .val-warn { color: #d97706; }
    </style>
    <div class="admet-grid">
    """

    for _, row in df.iterrows():
        # 1. Basic Info
        ligand = str(row.get('Ligand', 'Unknown'))
        pocket = str(row.get('Pocket', 'Unknown'))
        chain = str(row.get('Chain', '?'))
        pose = str(row.get('Pose', '1'))
        
        # 2. Scores & Logic
        docking_score = row.get('Docking Score', 0)
        dev_score = row.get('Developability Score', 0)
        decision = str(row.get('Final Decision', 'REVIEW'))

        # 3. Badge Logic
        if "ACCEPT" in decision.upper():
            badge_class = "badge-green"
            badge_text = "ACCEPTED"
        elif "REJECT" in decision.upper():
            badge_class = "badge-red"
            badge_text = "REJECTED"
        else:
            badge_class = "badge-yellow"
            badge_text = "REVIEW"

        # 4. Bar Visuals
        try:
            ds_val = float(docking_score)
            ds_width = min(100, max(0, (abs(ds_val) - 4) / 8 * 100))
        except: ds_val, ds_width = 0, 0
        
        try:
            dev_val = float(dev_score)
            dev_width = min(100, max(0, dev_val))
        except: dev_val, dev_width = 0, 0

        # 5. Extract Details for Grid
        # Helper to safely get string
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
            ("CYP2D6", get_val('CYP2D6 Inhibition'))
        ]

        # 6. Build Grid HTML
        grid_html = ""
        for name, val in props_list:
            # Simple color coding for obvious yes/no risks
            val_class = ""
            if val in ['Yes', 'High', 'Positive', 'Fail']: val_class = "val-bad"
            elif val in ['No', 'Low', 'Negative', 'Pass']: val_class = "val-good"
            
            grid_html += f"""
            <div class="prop-tag">
                <span class="prop-name">{name}</span>
                <span class="prop-value {val_class}">{val}</span>
            </div>
            """

        # 7. Assemble Card
        card = f"""
        <div class="admet-card">
            <div class="card-header">
                <div class="ligand-info">
                    <h3>{ligand}</h3>
                    <div class="ligand-sub">Chain {chain} • {pocket} • Pose {pose}</div>
                </div>
                <div class="{badge_class} badge">{badge_text}</div>
            </div>
            
            <div class="bar-section">
                <div class="bar-row">
                    <span class="bar-label">Binding</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {ds_width}%; background: #3b82f6;"></div>
                    </div>
                    <span class="bar-val">{ds_val:.2f}</span>
                </div>
                <div class="bar-row">
                    <span class="bar-label">Dev Score</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {dev_width}%; background: #8b5cf6;"></div>
                    </div>
                    <span class="bar-val">{int(dev_val)}</span>
                </div>
            </div>
            
            <div class="props-container">
                <div class="props-title">Molecular Properties</div>
                <div class="props-grid">
                    {grid_html}
                </div>
            </div>
        </div>
        """
        cards_html += card

    cards_html += "</div>"
    return cards_html

# ==========================================
# 2. LOGIC HANDLERS
# ==========================================

def process_disease(user_input: str):
    """
    Main function to process disease/protein input.
    Returns updates for Search Tab AND clears Ramachandran Tab.
    """
    
    # Define "Clear" updates for Ramachandran components
    clear_ram_status = gr.update(value="", visible=False)
    clear_ram_stats = gr.update(value="", visible=False)
    clear_plot = gr.update(value=None, visible=False)

    if not user_input.strip():
        current_pdb_info.update({"pdb_id": None, "pdb_path": None})
        return (
            gr.update(visible=False),           # info_box
            gr.update(value=""),                # structure_viewer
            gr.update(value=None),              # download_file
            gr.update(value="⚠️ Please enter a disease or protein name", visible=True), # search_status
            clear_ram_status, clear_ram_stats, clear_plot, clear_plot, clear_plot, clear_plot
        )
    
    protein_name = map_disease_to_protein(user_input)
    
    if not protein_name:
        protein_name = user_input.strip()
    
    # Search for structure
    result = find_best_pdb_structure(protein_name, max_check=100)
    
    if not result:
        current_pdb_info.update({"pdb_id": None, "pdb_path": None})
        error_msg = f"❌ No suitable PDB structure found for: {protein_name}"
        return (
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(value=error_msg, visible=True),
            clear_ram_status, clear_ram_stats, clear_plot, clear_plot, clear_plot, clear_plot
        )
    
    pdb_id, pdb_path = result
    
    try:
        with open(pdb_path, 'r') as f:
            pdb_content = f.read()
        
        # Update Global Config
        current_pdb_info.update({
            "pdb_id": pdb_id, 
            "pdb_path": pdb_path, 
            "prepared_pdbqt": None, 
            "docking_results": None, 
            "prankweb_csv": None
        })
        
        info_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 24px; border-radius: 16px; color: white;">
            <h3 style='margin-top:0;'>🧬 Structure Loaded</h3>
            <p><strong>Input:</strong> {user_input}</p>
            <p><strong>Target:</strong> {protein_name}</p>
            <p><strong>PDB ID:</strong> {pdb_id}</p>
        </div>
        """
        
        structure_html = show_structure(
            protein_text=pdb_content, 
            ligand_text=None, 
            pdb_id=pdb_id, 
            protein_name=protein_name
        )
        
        return (
            gr.update(value=info_html, visible=True),         # info_box
            gr.update(value=structure_html),                  # structure_viewer
            gr.update(value=pdb_path),                        # download_file
            gr.update(value="✅ Structure loaded successfully!", visible=True), # search_status
            clear_ram_status, clear_ram_stats, clear_plot, clear_plot, clear_plot, clear_plot
        )
        
    except Exception as e:
        return (
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(value=f"❌ Error: {str(e)}", visible=True),
            clear_ram_status, clear_ram_stats, clear_plot, clear_plot, clear_plot, clear_plot
        )

def process_ligand_analysis():
    """Logic for the new Ligand Analysis tab with dropdown update."""
    try:
        df, csv_path = run_ligand_classification("pdbqt")
        if df is None:
            return {
                ligand_status: gr.update(value=f"❌ {csv_path}", visible=True),
                ligand_table: gr.update(visible=False),
                ligand_csv_download: gr.update(visible=False),
                ligand_selector: gr.update(visible=False, choices=[])
            }
        
        # Extract ligand filenames from the 'Converted_PDB' column
        if 'Converted_PDB' in df.columns:
            ligand_files = df['Converted_PDB'].dropna().tolist()
        else:
            ligand_files = []
        
        return {
            ligand_status: gr.update(value="✅ Chemical Class Analysis Complete (Converted to PDB)", visible=True),
            ligand_table: gr.update(value=df, visible=True),
            ligand_csv_download: gr.update(value=csv_path, visible=True),
            ligand_selector: gr.update(visible=True, choices=ligand_files, value=ligand_files[0] if ligand_files else None)
        }
    except Exception as e:
        return {
            ligand_status: gr.update(value=f"❌ Analysis Error: {str(e)}", visible=True),
            ligand_table: gr.update(visible=False),
            ligand_csv_download: gr.update(visible=False),
            ligand_selector: gr.update(visible=False, choices=[])
        }

def visualize_ligand_only(ligand_file):
    """Visualizes a single ligand, reading from the converted PDB folder."""
    if not ligand_file:
        return ""
    
    ligand_path = os.path.join("ligand_pdb", ligand_file)
    
    if not os.path.exists(ligand_path):
        return f"❌ Converted PDB file not found at: {ligand_path}"
    
    try:
        with open(ligand_path, 'r') as f:
            ligand_content = f.read()
            
        html = show_structure(
            protein_text=None,
            ligand_text=ligand_content,
            pdb_id="Ligand",
            protein_name=ligand_file
        )
        return html
    except Exception as e:
        return f"❌ Visualization Error: {str(e)}"

def filter_poses_by_chain(chain_selected, summary_df):
    """Updates the Pose dropdown based on the selected Chain."""
    if summary_df is None or summary_df.empty:
        return gr.update(choices=[], value=None)
    
    if not chain_selected:
        return gr.update(choices=[], value=None)

    # Filter by chain
    chain_df = summary_df[summary_df['chain'] == chain_selected]
    
    choices = []
    for idx, row in chain_df.iterrows():
        label = f"{row['ligand']} - {row['pocket']} (Pose {row['pose_number']}) | {row['binding_energy']:.2f} kcal"
        value = f"{row['pdb_file']}::{row['pose_number']}::{row['chain']}"
        choices.append((label, value))
        
    return gr.update(choices=choices, value=choices[0][1] if choices else None)

def visualize_docking_result(selection_value: str, summary_df: pd.DataFrame):
    if not selection_value:
        return "⚠️ Please select a pose to view.", None
        
    report_text = None
    
    try:
        parts = selection_value.split("::")
        if len(parts) < 3:
             if len(parts) == 2: 
                 ligand_path, pose_num_str = parts
                 chain_id = None 
             else:
                 return f"❌ Invalid format.", None
        else:
             ligand_path, pose_num_str, chain_id = parts

        target_pose_num = int(pose_num_str)
        receptor_pdb_path = None
        
        # 1. Retrieve Metadata & Load External Report
        if summary_df is not None and not summary_df.empty:
            try:
                row = summary_df[
                    (summary_df['pdb_file'].astype(str) == str(ligand_path)) & 
                    (summary_df['pose_number'].astype(int) == target_pose_num)
                ]
                
                if not row.empty:
                    info = row.iloc[0]
                    
                    # Get Receptor Path
                    if 'receptor_pdb_file' in row.columns:
                        rec_p = info['receptor_pdb_file']
                        if rec_p and os.path.exists(rec_p):
                            receptor_pdb_path = rec_p
                    
                    # --- Load External Text Report ---
                    ligand_name = info.get('ligand', 'Unknown')
                    pocket_name = info.get('pocket', 'Unknown')
                    chain_id = info.get('chain', 'Unknown')
                    
                    # Report path construction
                    from config import DOCKING_RESULTS_DIR
                    report_dir = os.path.join(DOCKING_RESULTS_DIR, "docking_reports")
                    report_filename = f"{chain_id}_{ligand_name}_{pocket_name}_report.txt"
                    full_report_path = os.path.join(report_dir, report_filename)
                    
                    external_report_content = "⚠️ Report file not found."
                    if os.path.exists(full_report_path):
                        try:
                            with open(full_report_path, "r", encoding="utf-8") as rf:
                                external_report_content = rf.read()
                        except Exception as e:
                            external_report_content = f"⚠️ Error reading report: {str(e)}"
                    else:
                        external_report_content = f"⚠️ Report file missing at: {report_filename}"

                    # Generate Markdown Report
                    report_text = f"""
                    ### 📄 Pose Details
                    
                    | Property | Value |
                    | :--- | :--- |
                    | **Ligand** | {ligand_name} |
                    | **Binding Pocket** | {pocket_name} |
                    | **Chain** | {info.get('chain', 'N/A')} |
                    | **Pose Number** | {info.get('pose_number', 'N/A')} |
                    | **Binding Affinity** | **{info.get('binding_energy', 'N/A')} kcal/mol** |
                    
                    ---
                    ### 🧬 Interaction Profile Report
                    ```text
                    {external_report_content}
                    ```
                    """
            except Exception as e:
                print(f"Error extracting metadata: {e}")
        
        if not report_text:
            report_text = "### ⚠️ No detailed report data available for this pose."

        # 2. Load Structure for Visualization
        if not receptor_pdb_path:
             receptor_pdb_path = current_pdb_info.get("pdb_path")

        if not os.path.exists(ligand_path):
            return f"❌ Ligand file not found.", report_text
            
        with open(receptor_pdb_path, 'r') as f:
            protein_text = f.read()
            
        with open(ligand_path, 'r') as f:
            lines = f.readlines()
            
        model_lines = []
        in_model = False
        current_model = -1
        found_pose = False
        
        for line in lines:
            if line.startswith("MODEL"):
                try:
                    current_model = int(line.split()[1])
                except: pass
                if current_model == target_pose_num:
                    in_model = True
                    found_pose = True
            if in_model:
                model_lines.append(line)
            if line.startswith("ENDMDL") and in_model:
                in_model = False
                break 
        
        ligand_text_pose = "".join(model_lines) if found_pose else "".join(lines)
        display_name = f"Docked Pose {target_pose_num}"
        if chain_id: display_name += f" ({chain_id})"

        html_viewer = show_structure(
            protein_text=protein_text, 
            ligand_text=ligand_text_pose, 
            pdb_id="Docking", 
            protein_name=display_name
        )
        return html_viewer, report_text

    except Exception as e:
        return f"❌ Visualization Error: {str(e)}", None

def process_admet():
    """Run ADMET analysis and render the result as UI Cards."""
    try:
        result = run_admet_prediction()
        if result is None:
            return {
                admet_status: gr.update(value="❌ Analysis Failed.", visible=True),
                admet_results_view: gr.update(visible=False), 
                admet_download: gr.update(visible=False)
            }
        
        msg, df, csv_path = result
        
        # Render the custom HTML Cards
        html_view = render_admet_cards(df)
        
        return {
            admet_status: gr.update(value=f"✅ {msg}", visible=True),
            admet_results_view: gr.update(value=html_view, visible=True),
            admet_download: gr.update(value=csv_path, visible=True)
        }
    except Exception as e:
        return {
            admet_status: gr.update(value=f"❌ System Error: {str(e)}", visible=True),
            admet_results_view: gr.update(visible=False),
            admet_download: gr.update(visible=False)
        }

# ==========================================
# 3. UI LAYOUT
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

        # Tab 4: Ligand Analysis (NEW)
        with gr.Tab("🧪 Ligand Analysis", id=4):
            gr.Markdown("### Chemical Class & Complexity Analysis")
            ligand_analyze_btn = gr.Button("🔍 Analyze Ligands", variant="secondary")
            ligand_status = gr.Markdown(visible=False)
            ligand_table = gr.Dataframe(label="Ligand Properties", visible=False)
            ligand_csv_download = gr.File(label="Download Classification Report", visible=False)
            
            # New 3D Visualization Section
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
                    # Shows the Report Text
                    docking_report_area = gr.Markdown(label="Pose Details", visible=True)
                    
            with gr.Row():
                prev_btn_4 = gr.Button("← Previous", variant="secondary")
                next_btn_4 = gr.Button("Next: ADMET →", variant="primary")

        # Tab 6: ADMET Analysis
        with gr.Tab("🧪 ADMET Analysis", id=6):
            gr.Markdown("### Drug-likeness & Safety Screening")
            
            with gr.Row():
                admet_btn = gr.Button("🚀 Run ADMET Analysis", variant="primary")
                admet_download = gr.File(label="Download CSV Report", visible=False)
            
            admet_status = gr.Markdown(visible=False)
            
            # Using HTML component for custom White-Card UI
            admet_results_view = gr.HTML(label="Analysis Results", visible=True)
            
            with gr.Row():
                prev_btn_5 = gr.Button("← Previous", variant="secondary")
                next_btn_5 = gr.Button("Back to Start", variant="primary")

    # Events
    next_btn_0.click(lambda: gr.Tabs(selected=1), None, tabs)
    
    prev_btn_1.click(lambda: gr.Tabs(selected=0), None, tabs)
    next_btn_1.click(lambda: gr.Tabs(selected=2), None, tabs)
    
    prev_btn_2.click(lambda: gr.Tabs(selected=1), None, tabs)
    next_btn_2.click(lambda: gr.Tabs(selected=3), None, tabs) 
    
    prev_btn_3.click(lambda: gr.Tabs(selected=2), None, tabs)
    next_btn_3.click(lambda: gr.Tabs(selected=4), None, tabs) 
    
    # Ligand Analysis Events
    prev_btn_lig.click(lambda: gr.Tabs(selected=3), None, tabs)
    next_btn_lig.click(lambda: gr.Tabs(selected=5), None, tabs)
    ligand_analyze_btn.click(fn=process_ligand_analysis, inputs=[], outputs={ligand_status, ligand_table, ligand_csv_download, ligand_selector})
    ligand_selector.change(fn=visualize_ligand_only, inputs=[ligand_selector], outputs=[ligand_viewer])
    
    # Docking Navigation
    prev_btn_4.click(lambda: gr.Tabs(selected=4), None, tabs)
    next_btn_4.click(lambda: gr.Tabs(selected=6), None, tabs)
    
    # ADMET Navigation
    prev_btn_5.click(lambda: gr.Tabs(selected=5), None, tabs)
    next_btn_5.click(lambda: gr.Tabs(selected=0), None, tabs)

    # Core Logic Connections
    
    # Search Button outputs to Search Tab + Ramachandran Tab
    search_btn.click(
        fn=process_disease, 
        inputs=[disease_input], 
        outputs=[
            info_box, structure_viewer, download_file, search_status,   # Search Tab
            ramplot_status, ramplot_stats, plot1, plot2, plot3, plot4   # Ramachandran Tab (Cleared)
        ]
    )
    
    # Updated Ramachandran logic with Loading State
    ramplot_btn.click(
        fn=show_ram_loading, 
        outputs=[ramplot_status, plot1, plot2, plot3, plot4, ramplot_stats]
    ).then(
        fn=run_ramplot, 
        inputs=[], 
        outputs=[ramplot_status, plot1, plot2, plot3, plot4, ramplot_stats]
    )
    
    # Prepare Button
    prepare_btn.click(
        fn=prepare_protein_meeko, 
        inputs=[], 
        outputs=[
            prepare_status, prepared_viewer, prepared_download,
            info_box, structure_viewer, download_file, search_status,
            ramplot_status, plot1, plot2, plot3, plot4, ramplot_stats
        ]
    )
    
    # P2Rank + Fpocket Trigger
    prankweb_btn.click(fn=run_prankweb_prediction, inputs=[], outputs=[prankweb_status, prankweb_results, fpocket_results])
    
    docking_btn.click(fn=run_molecular_docking, inputs=[], outputs=[docking_status, docking_summary, chain_selector, pose_selector])
    chain_selector.change(fn=filter_poses_by_chain, inputs=[chain_selector, docking_summary], outputs=[pose_selector])
    
    # View Pose with Report Output
    view_pose_btn.click(
        fn=visualize_docking_result, 
        inputs=[pose_selector, docking_summary], 
        outputs=[docked_viewer, docking_report_area]
    )
    
    # ADMET Logic Connection
    admet_btn.click(fn=process_admet, inputs=[], outputs={admet_status, admet_results_view, admet_download})

if __name__ == "__main__":
    # JavaScript to force dark mode on load
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