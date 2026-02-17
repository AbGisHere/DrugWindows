# batch.py
"""
Batch Pipeline for Protein Structure Analysis
Processes multiple proteins through the complete workflow and saves all outputs.

UPDATED FEATURES:
- Step 1: Integrated with multi-candidate UniProt search and FASTA auto-retrieval.
- Step 4: Handles Combined P2Rank + Fpocket results.
- Step 5: Ligand Analysis (Chemical classification).
- Step 6: Docking uses the combined pocket list.
"""

import os
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import traceback
import time
import glob
import base64
import re
import requests

# Import modules from your app
from config import current_pdb_info, PROTEINS_DIR, DOCKING_RESULTS_DIR, RAMPLOT_OUTPUT_DIR, LIGAND_DIR
from ramachandran import run_ramplot
from prankweb import run_prankweb_prediction
from protein_prep import prepare_protein_meeko
from ligand_analysis import run_ligand_classification
from docking import run_molecular_docking
from admet_analysis import run_admet_prediction
from utils import map_disease_to_protein, find_best_pdb_structure


# --- HELPER FUNCTIONS ---
def update_dft_script_file(new_pdb_path):
    """Safely rewrites the PDB_FILE line in dft.py to point to the selected pose."""
    script_path = "dft.py"
    if not os.path.exists(script_path):
        raise FileNotFoundError("dft.py not found in current directory.")
        
    with open(script_path, "r") as f: lines = f.readlines()
    
    with open(script_path, "w") as f:
        for line in lines:
            if line.strip().startswith("PDB_FILE ="):
                f.write(f"PDB_FILE = {repr(str(new_pdb_path))}\n")
            else:
                f.write(line)

# --- VISUALIZATION FUNCTION ---
def show_structure(protein_text: str, ligand_text: str = None, pdb_id: str = "Docking Result", protein_name: str = "") -> str:
    """Robust 3D visualization supporting Overlay."""
    prot_json = json.dumps(protein_text)
    lig_json = json.dumps(ligand_text) if ligand_text else "null"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; background-color: white; }}
            #container {{ width: 100vw; height: 100vh; position: relative; }}
            #error-log {{ 
                display: none; position: absolute; top: 10px; left: 10px; 
                background: rgba(255,0,0,0.8); color: white; padding: 10px; 
                z-index: 999; font-family: sans-serif; border-radius: 5px;
            }}
            .legend {{
                position: absolute; bottom: 10px; right: 10px;
                background: rgba(255, 255, 255, 0.9); padding: 8px 12px;
                border-radius: 6px; font-family: sans-serif; font-size: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2); z-index: 100;
                pointer-events: none;
            }}
            .color-box {{ 
                display: inline-block; width: 10px; height: 10px; 
                margin-right: 6px; border-radius: 50%; 
            }}
            .info-overlay {{
                position: absolute; top: 10px; left: 10px;
                background: rgba(255, 255, 255, 0.9); padding: 8px 12px;
                border-radius: 6px; font-family: sans-serif; font-size: 14px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2); z-index: 90;
            }}
        </style>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    </head>
    <body>
        <div id="error-log"></div>
        <div class="info-overlay">
            <strong>{pdb_id}</strong><br>
            <span style="font-size:12px; color:#555">{protein_name}</span>
        </div>
        <div class="legend">
            <div><span class="color-box" style="background: linear-gradient(90deg, blue, green, red);"></span>Protein</div>
            <div style="margin-top:4px"><span class="color-box" style="background: magenta;"></span>Ligand</div>
        </div>
        <div id="container"></div>
        <script>
            function logError(msg) {{
                var el = document.getElementById('error-log');
                el.style.display = 'block';
                el.innerHTML += "Error: " + msg + "<br>";
                console.error(msg);
            }}
            window.onload = function() {{
                try {{
                    if (typeof $3Dmol === 'undefined') throw new Error("3Dmol.js failed to load.");
                    var element = document.getElementById('container');
                    var config = {{ backgroundColor: 'white' }};
                    var viewer = $3Dmol.createViewer(element, config);

                    var proteinData = {prot_json};
                    if (!proteinData) throw new Error("Protein data is empty.");
                    
                    viewer.addModel(proteinData, "pdb");
                    viewer.setStyle({{model: 0}}, {{cartoon: {{color: 'spectrum'}}}});
                    viewer.addStyle({{model: 0, hetflag: true}}, {{stick: {{radius: 0.1, color: 'lightgray'}}}});

                    var ligandData = {lig_json};
                    if (ligandData) {{
                        viewer.addModel(ligandData, "pdb");
                        viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: 'magentaCarbon', radius: 0.4}}}});
                    }} else {{
                        viewer.addStyle({{resn: ["UNL", "LIG", "DRG", "UNK"]}}, {{stick: {{colorscheme: 'magentaCarbon', radius: 0.4}}}});
                    }}

                    viewer.zoomTo();
                    viewer.render();
                }} catch(e) {{ logError(e.message); }}
            }};
        </script>
    </body>
    </html>
    """
    b64 = base64.b64encode(html_content.encode()).decode()
    iframe = f'<iframe src="data:text/html;base64,{b64}" width="100%" height="500" frameborder="0" style="border: 1px solid #ccc; border-radius: 8px;"></iframe>'
    return iframe


class ProteinPipelineBatch:
    """Handles batch processing of proteins through the complete pipeline."""
    
    def __init__(self, output_base_dir="batch_results"):
        """Initialize batch processor with output directory."""
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_protein_folders(self, protein_name):
        """Create folder structure for a protein."""
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                           for c in protein_name).strip()
        protein_dir = self.output_base_dir / safe_name
        protein_dir.mkdir(exist_ok=True)
        
        steps = [
            "01_structure_search", 
            "02_ramachandran_analysis", 
            "03_protein_preparation",
            "04_binding_site_prediction", 
            "05_ligand_analysis",     
            "06_molecular_docking",   
            "07_admet_analysis",
            "08_dft_analysis"       
        ]
        step_dirs = {}
        for step in steps:
            step_dir = protein_dir / step
            step_dir.mkdir(exist_ok=True)
            step_dirs[step] = step_dir
        return protein_dir, step_dirs
    
    def save_3d_viewer_as_image(self, html_content, output_path, title=""):
        """Convert 3D viewer HTML to 2D image using screenshot method."""
        if not html_content: return False

        if isinstance(html_content, str) and "data:text/html;base64," in html_content:
            try:
                match = re.search(r'src="data:text/html;base64,([^"]+)"', html_content)
                if match:
                    b64_str = match.group(1)
                    decoded_bytes = base64.b64decode(b64_str)
                    html_content = decoded_bytes.decode('utf-8')
            except Exception as e:
                print(f"⚠️ Failed to decode base64 iframe: {e}")
        
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            
            chrome_options = Options()
            chrome_options.add_argument("--headless=new") 
            chrome_options.add_argument("--use-gl=swiftshader") 
            chrome_options.add_argument("--enable-webgl")
            chrome_options.add_argument("--ignore-gpu-blocklist")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            
            driver = webdriver.Chrome(options=chrome_options)
            
            style_injection = "<style>body, html, .viewer-container, #viewport, #container { width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden; }</style>"
            if "</head>" in html_content:
                html_content = html_content.replace("</head>", f"{style_injection}</head>")
            else:
                html_content = style_injection + html_content

            temp_html = output_path.parent / "temp_viewer.html"
            with open(temp_html, 'w', encoding='utf-8') as f: f.write(html_content)
            
            driver.get(f"file://{temp_html.absolute()}")
            time.sleep(8)
            
            try: driver.execute_script("if(typeof viewer !== 'undefined') { viewer.render(); }")
            except: pass

            try:
                canvas = driver.find_element(By.TAG_NAME, "canvas")
                canvas.screenshot(str(output_path))
            except:
                driver.save_screenshot(str(output_path))
                
            driver.quit()
            if temp_html.exists(): temp_html.unlink()
            return True
        except Exception as e:
            print(f"⚠️ Screenshot failed: {e}")
            html_path = output_path.with_suffix('.html')
            with open(html_path, 'w', encoding='utf-8') as f: f.write(html_content)
            return False
    
    # --- STEP 1: STRUCTURE SEARCH ---
    def process_structure_search(self, protein_input, step_dir):
        print(f"\n{'='*60}")
        print(f"STEP 1: Structure Search - {protein_input}")
        print(f"{'='*60}")
        results = {"status": "failed", "protein_input": protein_input}
        try:
            protein_name = map_disease_to_protein(protein_input) or protein_input.strip()
            results["protein_name"] = protein_name
            
            # Use updated utils logic (multi-candidate search + auto FASTA download)
            result = find_best_pdb_structure(protein_name, max_check=100)
            
            if not result:
                results["error"] = "No suitable PDB structure found"
                print(f"❌ No structure found")
                return results
            
            pdb_id, pdb_path = result
            pdb_path = os.path.abspath(pdb_path)
            results["pdb_id"], results["pdb_path"] = pdb_id, pdb_path
            
            current_pdb_info.update({
                "pdb_id": pdb_id, 
                "pdb_path": pdb_path, 
                "prepared_pdbqt": None, 
                "docking_results": None, 
                "prankweb_csv": None,
                "combined_csv": None
            })
            
            # Copy PDB to batch folder
            dest_pdb = step_dir / f"{pdb_id}.pdb"
            shutil.copy2(pdb_path, dest_pdb)
            
            # Copy FASTA to batch folder (Downloaded by utils automatically)
            fasta_filename = f"{pdb_id}.fasta"
            # Assuming utils downloads to the same folder as the PDB
            source_fasta_path = os.path.join(os.path.dirname(pdb_path), fasta_filename)
            
            if os.path.exists(source_fasta_path):
                dest_fasta = step_dir / fasta_filename
                shutil.copy2(source_fasta_path, dest_fasta)
                print(f"  ✅ FASTA copied to batch folder.")
            else:
                print(f"  ⚠️ FASTA file not found at expected location: {source_fasta_path}")

            with open(pdb_path, 'r') as f: pdb_content = f.read()
            structure_html = show_structure(protein_text=pdb_content, pdb_id=pdb_id, protein_name=protein_name)
            self.save_3d_viewer_as_image(structure_html, step_dir / f"{pdb_id}_structure.png", f"{protein_name}")
            results["status"] = "success"
            print(f"✅ Structure loaded: {pdb_id}")
        except Exception as e:
            results["error"] = str(e)
            print(f"❌ Error: {e}")
        return results

    # --- STEP 2: RAMACHANDRAN ---
    def process_ramachandran(self, step_dir):
        print(f"\nSTEP 2: Ramachandran Analysis")
        run_ramplot() 
        
        csv_files_copied = 0
        
        if os.path.exists(RAMPLOT_OUTPUT_DIR):
            for csv_file in glob.glob(os.path.join(RAMPLOT_OUTPUT_DIR, "*.csv")):
                try:
                    shutil.copy2(csv_file, step_dir)
                    csv_files_copied += 1
                except Exception as e:
                    print(f"Warning: Could not copy csv {csv_file}: {e}")
            
            plots_subdir = os.path.join(RAMPLOT_OUTPUT_DIR, "Plots")
            target_images = ["MapType2DAll.png", "MapType3DAll.png", "StdMapType2DGeneralGly.png", "StdMapType3DGeneral.png"]
            
            if os.path.exists(plots_subdir):
                for img_name in target_images:
                    src_img = os.path.join(plots_subdir, img_name)
                    if os.path.exists(src_img):
                        try:
                            shutil.copy2(src_img, step_dir)
                        except: pass
        
        print(f"✅ Ramachandran analysis complete.")
        return {"status": "success", "csv_files_count": csv_files_copied}

    # --- STEP 3: PROTEIN PREP ---
    def process_protein_preparation(self, step_dir):
        print(f"\nSTEP 3: Protein Preparation")
        generator = prepare_protein_meeko()
        final_output = None
        for output in generator: final_output = output
        
        results = {"status": "failed"}
        if final_output:
            pdbqt_update = final_output[2] if len(final_output) > 2 else None
            pdbqt_path = None
            
            if hasattr(pdbqt_update, 'value'): pdbqt_path = pdbqt_update.value
            elif isinstance(pdbqt_update, dict): pdbqt_path = pdbqt_update.get('value')
            else: pdbqt_path = pdbqt_update

            if pdbqt_path and os.path.exists(pdbqt_path):
                dest_pdbqt = step_dir / Path(pdbqt_path).name
                shutil.copy2(pdbqt_path, dest_pdbqt)
                current_pdb_info["prepared_pdbqt"] = str(dest_pdbqt)
                
                try:
                    with open(pdbqt_path, 'r') as f: prepared_text = f.read()
                    html_view = show_structure(protein_text=prepared_text, pdb_id="Prepared Structure", protein_name="Auto-Prepared")
                    img_path = step_dir / "prepared_protein_structure.png"
                    self.save_3d_viewer_as_image(html_view, img_path, "Prepared Structure")
                except: pass

                results["status"] = "success"
                print(f"✅ Protein prepared")
        return results

    # --- STEP 4: BINDING SITES (MERGED) ---
    def process_binding_sites(self, step_dir):
        """Step 4: Binding Site Prediction (P2Rank + Fpocket)."""
        print(f"\nSTEP 4: Binding Site Prediction")
        if not current_pdb_info.get("prepared_pdbqt"): return {"status": "skipped"}
        
        try:
            print(f"  Running Local P2Rank & Fpocket...")
            generator = run_prankweb_prediction()
            final_output = None
            for output in generator: final_output = output
            
            results = {"status": "failed"}
            
            if final_output:
                # final_output signature: (status_html, dataframe_update, fpocket_text_update)
                df_update = final_output[1] if len(final_output) > 1 else None
                fpocket_text_update = final_output[2] if len(final_output) > 2 else None
                
                # 1. Handle DataFrame (Combined Pockets)
                df = None
                if hasattr(df_update, 'value'): df = df_update.value
                elif isinstance(df_update, dict): df = df_update.get('value')
                else: df = df_update

                # 2. Handle Fpocket Text
                fpocket_txt = ""
                if hasattr(fpocket_text_update, 'value'): fpocket_txt = fpocket_text_update.value
                elif isinstance(fpocket_text_update, dict): fpocket_txt = fpocket_text_update.get('value')
                else: fpocket_txt = fpocket_text_update

                # Save Results
                if df is not None and not df.empty:
                    # Save Combined CSV (Critical for Docking)
                    combined_csv_path = step_dir / "combined_pockets.csv"
                    df.to_csv(combined_csv_path, index=False)
                    current_pdb_info["combined_csv"] = str(combined_csv_path) # Update global config
                    
                    results["status"] = "success"
                    print(f"✅ {len(df)} pockets found (P2Rank + Fpocket). Saved combined_pockets.csv")
                else:
                    print("  ⚠️ No merged pockets found.")
                
                # Save Fpocket Text Report
                if fpocket_txt:
                    with open(step_dir / "fpocket_report.txt", "w") as f:
                        f.write(fpocket_txt)

                return results

            else:
                print("  ⚠️ No final output received from prediction module.")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            traceback.print_exc()

        return {"status": "failed", "error": "Prediction execution failed"}

    # --- STEP 5: LIGAND ANALYSIS (NEW) ---
    def process_ligand_analysis(self, step_dir):
        """Step 5: Chemical Class Analysis of Ligands."""
        print(f"\nSTEP 5: Ligand Analysis")
        
        # Ensure we are analyzing the configured ligand directory
        ligand_source_dir = LIGAND_DIR if os.path.exists(LIGAND_DIR) else "pdbqt"
        
        try:
            print(f"  Analyzing ligands in: {ligand_source_dir}")
            df, csv_path = run_ligand_classification(ligand_source_dir)
            
            results = {"status": "failed"}
            
            if df is not None and not df.empty:
                # Copy report
                if csv_path and os.path.exists(csv_path):
                    shutil.copy2(csv_path, step_dir / "ligand_classification_report.csv")
                
                # Optionally copy converted PDBs for debugging/viewing
                ligand_pdb_out = step_dir / "ligand_pdb_converted"
                os.makedirs(ligand_pdb_out, exist_ok=True)
                
                source_pdb_folder = "ligand_pdb" # Default output of run_ligand_classification
                if os.path.exists(source_pdb_folder):
                    for f in os.listdir(source_pdb_folder):
                        if f.endswith(".pdb"):
                            shutil.copy2(os.path.join(source_pdb_folder, f), ligand_pdb_out)
                
                results["status"] = "success"
                print(f"✅ Analyzed {len(df)} ligands.")
            else:
                print(f"  ⚠️ No ligands found or analysis failed.")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Ligand Analysis Error: {e}")
            return {"status": "failed", "error": str(e)}

    # --- STEP 6: DOCKING ---
    def process_docking(self, step_dir):
        """Step 6: Molecular docking using Combined Pockets."""
        print(f"\n{'='*60}")
        print(f"STEP 6: Molecular Docking")
        print(f"{'='*60}")
        
        results = {"status": "failed"}
        
        try:
            if not current_pdb_info.get("prepared_pdbqt"): return results
            # Docking now relies on combined_csv
            if not current_pdb_info.get("combined_csv"): 
                print("  ⚠️ No combined pocket data found. Skipping.")
                return results

            generator = run_molecular_docking()
            final_output = None
            print("  ⚙️  Running Vina Docking (Multi-Chain)...")
            for output in generator: final_output = output
            
            if not final_output: return results
            
            # Extract summary dataframe
            df_update = final_output[1] if len(final_output) > 1 else None
            summary_df = None
            if hasattr(df_update, 'value'): summary_df = df_update.value
            elif isinstance(df_update, dict): summary_df = df_update.get('value')
            else: summary_df = df_update
            
            if summary_df is not None and not summary_df.empty:
                csv_path = step_dir / "docking_summary.csv"
                summary_df.to_csv(csv_path, index=False)
                results["csv_file"] = str(csv_path)
                print(f"  ✓ Saved: docking_summary.csv")
                
                summary_df['binding_energy'] = pd.to_numeric(summary_df['binding_energy'], errors='coerce')
                results_to_process = summary_df 
                print(f"  ℹ️  Processing {len(results_to_process)} docked poses...")

                # Pre-load Protein
                protein_path = current_pdb_info.get("pdb_path")
                protein_text = ""
                if protein_path and os.path.exists(protein_path):
                    with open(protein_path, 'r') as pf: protein_text = pf.read()

                for idx, row in results_to_process.iterrows():
                    ligand_name = str(row['ligand']).strip()
                    pocket_name = str(row['pocket']).strip()
                    chain_id = str(row.get('chain', 'A'))
                    energy = float(row['binding_energy'])
                    pose_num = int(row['pose_number'])
                    pdb_file_path = row['pdb_file']
                    
                    safe_ligand = "".join(c if c.isalnum() else '_' for c in ligand_name)
                    energy_str = f"{energy:.1f}kcal"
                    base_filename = f"{safe_ligand}_{pocket_name}_{chain_id}_pose{pose_num}"

                    # 1. Save 2D Interaction Image
                    if 'interaction_image' in row and row['interaction_image'] != "N/A":
                        img_src = row['interaction_image']
                        if os.path.exists(img_src):
                            img_dest = step_dir / f"{base_filename}_2D.png"
                            shutil.copy2(img_src, img_dest)

                    # 2. SAVE COMPLEX PDB
                    # Logic: reconstruct path based on known structure or extract from file system
                    chain_dir = os.path.dirname(os.path.dirname(pdb_file_path)) 
                    complex_src = os.path.join(os.path.dirname(pdb_file_path), f"{ligand_name}_{pocket_name}_complex.pdb")
                    
                    if os.path.exists(complex_src):
                        complex_dest = step_dir / f"{base_filename}_complex.pdb"
                        shutil.copy2(complex_src, complex_dest)

                    # 3. Generate 3D Screenshot
                    ligand_text_pose = ""
                    if os.path.exists(pdb_file_path):
                        with open(pdb_file_path, 'r') as lf:
                            lines = lf.readlines()
                            model_lines = []
                            in_model = False
                            current_model = -1
                            found_pose = False
                            for line in lines:
                                if line.startswith("MODEL"):
                                    try: current_model = int(line.split()[1])
                                    except: pass
                                    if current_model == pose_num:
                                        in_model = True
                                        found_pose = True
                                if in_model: model_lines.append(line)
                                if line.startswith("ENDMDL") and in_model: break
                            
                            ligand_text_pose = "".join(model_lines) if found_pose else "".join(lines)
                    
                    if protein_text and ligand_text_pose:
                        viewer_html = show_structure(
                            protein_text=protein_text, 
                            ligand_text=ligand_text_pose, 
                            pdb_id="Docking Result", 
                            protein_name=f"Chain {chain_id} | {pocket_name} | {energy} kcal"
                        )
                        screenshot_path = step_dir / f"{base_filename}_3D.png"
                        self.save_3d_viewer_as_image(viewer_html, screenshot_path, base_filename)

                results["status"] = "success"
                results["num_poses"] = len(results_to_process)
                print(f"✅ Docking processing complete.")
            else:
                results["error"] = "Docking failed or no results"
                print(f"❌ Docking failed - no poses found")
            
        except Exception as e:
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
            print(f"❌ Error in docking: {e}")
            traceback.print_exc()
        
        return results

    # --- STEP 7: ADMET ---
    def process_admet(self, step_dir):
        print(f"\nSTEP 7: ADMET Analysis")
        result = run_admet_prediction()
        if result:
            msg, df, csv_path = result
            if csv_path and os.path.exists(csv_path):
                shutil.copy2(csv_path, step_dir / "admet_results.csv")
                print(f"✅ ADMET results saved.")
                return {"status": "success", "message": msg}
        return {"status": "failed"}

    # --- STEP 8: DFT ANALYSIS ---
    def process_dft_analysis(self, step_dir, docking_dir):
        """Step 8: Run DFT Analysis on all docked compounds."""
        print(f"\nSTEP 8: DFT Analysis")
        
        # 1. Find all docked PDBs from the docking step
        # docking_dir is the Path object for "06_molecular_docking"
        docked_pdbs = list(docking_dir.glob("*_complex.pdb"))
        
        if not docked_pdbs:
            print("  ⚠️ No docked complex PDB files found for DFT analysis.")
            return {"status": "skipped", "reason": "no_docked_files"}
            
        print(f"  Found {len(docked_pdbs)} complexes to analyze.")
        
        # CSV Cleanup specifically for this run's context, though we are in a batch script.
        # We will collect results in a list and save to the step_dir.
        
        batch_results = []
        dft_script_path = "dft.py"
        
        if not os.path.exists(dft_script_path):
             print(f"  ❌ dft.py not found.")
             return {"status": "failed", "error": "dft.py missing"}

        # Clear single run output to avoid reading stale data from previous manual runs
        single_csv = "orca_electronic_metrics.csv"
        if os.path.exists(single_csv): os.remove(single_csv)

        for i, pdb_path in enumerate(docked_pdbs):
            pdb_name = pdb_path.name
            print(f"  [{i+1}/{len(docked_pdbs)}] Processing DFT for: {pdb_name}")
            
            # 1. Update Script
            try:
                # We must use absolute path for the dft script to find it
                update_dft_script_file(pdb_path.resolve())
            except Exception as e:
                print(f"    ❌ Failed to update script: {e}")
                continue
            
            # 2. Run Subprocess (Switch Env)
            # This child process runs in orca_env, then dies. 
            # We assume 'conda activate' or similar isn't strictly available as a simple executable 
            # unless we use the shell.
            # Mirroring app.py command: 'cmd /c "call activate orca_env && python dft.py"'
            
            command = f'cmd /c "call activate orca_env && python dft.py"'
            try:
                # Run with timeout to prevent hanging? dft can be slow. 
                # Let's assume user wants to wait.
                process = subprocess.run(command, capture_output=True, text=True, shell=True)
                if process.returncode != 0:
                    print(f"    ⚠️ Error executing dft.py: {process.stderr.strip()}")
                else:
                    # check stdout for specific errors if needed
                    pass
            except Exception as e:
                print(f"    ❌ Execution error: {e}")
            
            # 3. Read Single Result
            # dft.py writes to "orca_electronic_metrics.csv" in the CWD (project root).
            if os.path.exists(single_csv):
                try:
                    df = pd.read_csv(single_csv)
                    if not df.empty:
                        last_row = df.iloc[-1].to_dict()
                        # Ensure filename matches what we just ran
                        # dft.py saves the filename, but let's be sure or just use it.
                        batch_results.append(last_row)
                except Exception as e:
                    print(f"    ⚠️ Failed to read results: {e}")
        
        # Save aggregated results
        if batch_results:
            results_df = pd.DataFrame(batch_results)
            csv_path = step_dir / "dft_batch_results.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"  ✅ DFT Batch Analysis complete. Saved {len(results_df)} results.")
            return {"status": "success", "count": len(results_df), "csv_path": str(csv_path)}
        else:
            print("  ⚠️ No DFT results were generated.")
            return {"status": "failed", "reason": "no_results_generated"}
    
    # --- PIPELINE CONTROLLER ---
    def process_single_protein(self, protein_input):
        print(f"\n{'#'*80}")
        print(f"# Processing: {protein_input}")
        print(f"{'#'*80}")
        
        protein_dir, step_dirs = self.create_protein_folders(protein_input)
        pipeline_results = {"protein_input": protein_input, "start_time": datetime.now().isoformat(), "steps": {}}
        
        # 1. Search
        pipeline_results["steps"]["structure_search"] = self.process_structure_search(protein_input, step_dirs["01_structure_search"])
        if pipeline_results["steps"]["structure_search"]["status"] != "success":
            pipeline_results["pipeline_status"] = "failed_at_step_1"
            self.save_pipeline_summary(protein_dir, pipeline_results)
            return pipeline_results
        
        # 2. Ramachandran
        pipeline_results["steps"]["ramachandran"] = self.process_ramachandran(step_dirs["02_ramachandran_analysis"])
        
        # 3. Prep
        pipeline_results["steps"]["protein_preparation"] = self.process_protein_preparation(step_dirs["03_protein_preparation"])
        if pipeline_results["steps"]["protein_preparation"]["status"] != "success":
            pipeline_results["pipeline_status"] = "failed_at_step_3"
            self.save_pipeline_summary(protein_dir, pipeline_results)
            return pipeline_results
        
        # 4. Binding Sites (Combined)
        pipeline_results["steps"]["binding_sites"] = self.process_binding_sites(step_dirs["04_binding_site_prediction"])
        
        # 5. Ligand Analysis (NEW)
        pipeline_results["steps"]["ligand_analysis"] = self.process_ligand_analysis(step_dirs["05_ligand_analysis"])

        # 6. Docking (Multi-Chain, Combined Pockets)
        pipeline_results["steps"]["docking"] = self.process_docking(step_dirs["06_molecular_docking"])
        
        # 7. ADMET
        pipeline_results["steps"]["admet"] = self.process_admet(step_dirs["07_admet_analysis"])

        # 8. DFT Analysis (NEW)
        # We need to pass the docking results from step 6 (or the folder) to this step
        # However, the process_dft_analysis method will likely scan the docking folder itself or receive the list of PDBs.
        # Let's start by modifying create_protein_folders to include step 8.
        pipeline_results["steps"]["dft_analysis"] = self.process_dft_analysis(
            step_dirs["08_dft_analysis"], 
            step_dirs["06_molecular_docking"]
        )
        
        pipeline_results["pipeline_status"] = "completed"
        pipeline_results["end_time"] = datetime.now().isoformat()
        
        self.save_pipeline_summary(protein_dir, pipeline_results)
        print(f"\n{'='*60}\n✅ Pipeline complete for {protein_input}\nResults saved to: {protein_dir}\n{'='*60}\n")
        return pipeline_results
    
    def save_pipeline_summary(self, protein_dir, results):
        with open(protein_dir / "pipeline_summary.json", 'w') as f: json.dump(results, f, indent=2)
    
    def run_batch(self, protein_list):
        print(f"\n{'#'*80}\n# BATCH PROCESSING: {len(protein_list)} proteins\n{'#'*80}\n")
        batch_results = {"proteins": {}}
        for i, protein in enumerate(protein_list, 1):
            print(f"[{i}/{len(protein_list)}]")
            try: batch_results["proteins"][protein] = self.process_single_protein(protein)
            except Exception as e:
                print(f"❌ Fatal error: {e}")
                batch_results["proteins"][protein] = {"status": "fatal_error", "error": str(e)}
        self.generate_batch_csv_summary(batch_results)
        return batch_results
    
    def generate_batch_csv_summary(self, batch_results):
        summary_data = []
        for p, r in batch_results["proteins"].items():
            summary_data.append({"protein": p, "status": r.get("pipeline_status", "unknown")})
        pd.DataFrame(summary_data).to_csv(self.output_base_dir / f"batch_summary_{self.timestamp}.csv", index=False)


if __name__ == "__main__":
    protein_list = [
    # 1. CANCER (PDAC)
    "KRAS",
    "PI3K",
    "mTOR",
    "MEK1",
    "MEK2",
    "RAF1",
    "CDK4",
    "STAT3",
    "NF-κB",
    "SMAD4",
    "MCL-1",
    "PARP1",
    "EGFR",
    "HIF-1α",
    "CDK6",

    # 2. INFLAMMATION TARGETS
    "COX-2",
    "5-LOX",
    "NF-κB",
    "NLRP3",
    "iNOS",
    "Caspase-1",
    "JAK1",
    "JAK2",
    "MAPK14",
    "TNF-α",
    "IL-6",
    "IL-1β",
    "IL-17A",
    "TLR4",
    "PLA2",

    # 3. METABOLIC HEALTH TARGETS
    "ACC",
    "FASN",
    "HMG-CoA reductase",
    "CPT1A",
    "AMPK",
    "PPAR-α",
    "PPAR-γ",
    "SREBP-1",
    "GLP-1 receptor",
    "DPP-4",
    "GSK3β",
    "IRS-1",
    "Leptin receptor",
    "Adiponectin receptor (AdipoR1)",
    "TNF-α"
]


    batch_processor = ProteinPipelineBatch(output_base_dir="New_pipeline")
    results = batch_processor.run_batch(protein_list)