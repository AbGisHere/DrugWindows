# app.py
"""
Main Gradio interface for Protein Structure Finder & Analyzer
Updated version: Includes Fpocket Integration in Tab 3.
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

def process_disease(user_input: str):
    """Main function to process disease/protein input."""
    
    if not user_input.strip():
        current_pdb_info.update({"pdb_id": None, "pdb_path": None})
        return {
            info_box: gr.update(visible=False),
            structure_viewer: gr.update(value=""),
            download_file: gr.update(value=None),
            search_status: gr.update(value="⚠️ Please enter a disease or protein name", visible=True)
        }
    
    protein_name = map_disease_to_protein(user_input)
    is_direct_protein = False
    
    if not protein_name:
        protein_name = user_input.strip()
        is_direct_protein = True
    
    result = find_best_pdb_structure(protein_name, max_check=100)
    
    if not result:
        current_pdb_info.update({"pdb_id": None, "pdb_path": None})
        error_msg = f"❌ No suitable PDB structure found for: {protein_name}"
        return {
            info_box: gr.update(visible=False),
            structure_viewer: gr.update(value=""),
            download_file: gr.update(value=None),
            search_status: gr.update(value=error_msg, visible=True)
        }
    
    pdb_id, pdb_path = result
    
    try:
        with open(pdb_path, 'r') as f:
            pdb_content = f.read()
        
        current_pdb_info.update({
            "pdb_id": pdb_id, 
            "pdb_path": pdb_path, 
            "prepared_pdbqt": None, 
            "docking_results": None, 
            "prankweb_csv": None
        })
        
        info_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 24px; border-radius: 16px; color: white;">
            <h3>Structure Loaded</h3>
            <p><strong>Input:</strong> {user_input}</p>
            <p><strong>Protein:</strong> {protein_name}</p>
            <p><strong>PDB ID:</strong> {pdb_id}</p>
            <p><strong>Species:</strong> Homo-sapiens</p>
        </div>
        """
        
        structure_html = show_structure(
            protein_text=pdb_content, 
            ligand_text=None, 
            pdb_id=pdb_id, 
            protein_name=protein_name
        )
        
        return {
            info_box: gr.update(value=info_html, visible=True),
            structure_viewer: gr.update(value=structure_html),
            download_file: gr.update(value=pdb_path),
            search_status: gr.update(value="✅ Structure loaded successfully!", visible=True)
        }
        
    except Exception as e:
        return {
            info_box: gr.update(visible=False),
            structure_viewer: gr.update(value=""),
            download_file: gr.update(value=None),
            search_status: gr.update(value=f"❌ Error: {str(e)}", visible=True)
        }

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
        # This ensures the dropdown shows .pdb files
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
    
    # ligand_file is now a .pdb name from the dropdown (e.g. "ligand_1.pdb")
    # We access the 'ligand_pdb' directory directly
    ligand_path = os.path.join("ligand_pdb", ligand_file)
    
    if not os.path.exists(ligand_path):
        return f"❌ Converted PDB file not found at: {ligand_path}"
    
    try:
        with open(ligand_path, 'r') as f:
            ligand_content = f.read()
            
        # Pass None as protein_text to visualize only the ligand
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
        
    image_path = None
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
        
        if summary_df is not None and not summary_df.empty:
            try:
                row = summary_df[
                    (summary_df['pdb_file'].astype(str) == str(ligand_path)) & 
                    (summary_df['pose_number'].astype(int) == target_pose_num)
                ]
                if not row.empty:
                    img_p = row.iloc[0]['interaction_image']
                    if img_p and isinstance(img_p, str) and os.path.exists(img_p) and img_p != "N/A":
                        image_path = img_p
                    if 'receptor_pdb_file' in row.columns:
                        rec_p = row.iloc[0]['receptor_pdb_file']
                        if rec_p and os.path.exists(rec_p):
                            receptor_pdb_path = rec_p
            except: pass
        
        if not receptor_pdb_path:
             receptor_pdb_path = current_pdb_info.get("pdb_path")

        if not os.path.exists(ligand_path):
            return f"❌ Ligand file not found.", image_path
            
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
        return html_viewer, image_path

    except Exception as e:
        return f"❌ Visualization Error: {str(e)}", None

def process_admet():
    try:
        result = run_admet_prediction()
        if result is None:
            return {
                admet_status: gr.update(value="❌ Analysis Failed.", visible=True),
                admet_table: gr.update(visible=False),
                admet_download: gr.update(visible=False)
            }
        msg, df, csv_path = result
        return {
            admet_status: gr.update(value=f"✅ {msg}", visible=True),
            admet_table: gr.update(value=df, visible=True),
            admet_download: gr.update(value=csv_path, visible=True)
        }
    except Exception as e:
        return {
            admet_status: gr.update(value=f"❌ System Error: {str(e)}", visible=True),
            admet_table: gr.update(visible=False),
            admet_download: gr.update(visible=False)
        }

# UI Layout
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
                    interaction_image_viewer = gr.Image(label="2D Interaction Map", type="filepath", visible=True)
            with gr.Row():
                prev_btn_4 = gr.Button("← Previous", variant="secondary")
                next_btn_4 = gr.Button("Next: ADMET →", variant="primary")

        # Tab 6: ADMET
        with gr.Tab("🧪 ADMET Analysis", id=6):
            gr.Markdown("### Drug-likeness & Safety")
            admet_btn = gr.Button("Run ADMET", variant="secondary")
            admet_status = gr.Markdown(visible=False)
            admet_download = gr.File(visible=False)
            admet_table = gr.Dataframe(visible=False)
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
    next_btn_3.click(lambda: gr.Tabs(selected=4), None, tabs) # PrankWeb -> Ligand

    # Ligand Analysis Events
    prev_btn_lig.click(lambda: gr.Tabs(selected=3), None, tabs)
    next_btn_lig.click(lambda: gr.Tabs(selected=5), None, tabs)
    ligand_analyze_btn.click(fn=process_ligand_analysis, inputs=[], outputs={ligand_status, ligand_table, ligand_csv_download, ligand_selector})
    ligand_selector.change(fn=visualize_ligand_only, inputs=[ligand_selector], outputs=[ligand_viewer])
    
    # Docking Navigation
    prev_btn_4.click(lambda: gr.Tabs(selected=4), None, tabs) # Docking -> Ligand
    next_btn_4.click(lambda: gr.Tabs(selected=6), None, tabs)
    
    # ADMET Navigation
    prev_btn_5.click(lambda: gr.Tabs(selected=5), None, tabs)
    next_btn_5.click(lambda: gr.Tabs(selected=0), None, tabs)

    # Core Logic Connections
    search_btn.click(process_disease, inputs=[disease_input], outputs={info_box, structure_viewer, download_file, search_status})
    ramplot_btn.click(fn=run_ramplot, inputs=[], outputs=[ramplot_status, plot1, plot2, plot3, plot4, ramplot_stats])
    prepare_btn.click(fn=prepare_protein_meeko, inputs=[], outputs=[prepare_status, prepared_viewer, prepared_download])
    
    # UPDATED: P2Rank + Fpocket Trigger
    prankweb_btn.click(fn=run_prankweb_prediction, inputs=[], outputs=[prankweb_status, prankweb_results, fpocket_results])
    
    docking_btn.click(fn=run_molecular_docking, inputs=[], outputs=[docking_status, docking_summary, chain_selector, pose_selector])
    chain_selector.change(fn=filter_poses_by_chain, inputs=[chain_selector, docking_summary], outputs=[pose_selector])
    view_pose_btn.click(fn=visualize_docking_result, inputs=[pose_selector, docking_summary], outputs=[docked_viewer, interaction_image_viewer])
    
    admet_btn.click(fn=process_admet, inputs=[], outputs={admet_status, admet_table, admet_download})

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