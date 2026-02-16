"""
Ramachandran plot analysis module with SWISS-MODEL integration
"""

import os
import subprocess
import glob
import gradio as gr
import requests
import time
import re
from config import current_pdb_info, RAMPLOT_OUTPUT_DIR, PROTEINS_DIR

# SWISS-MODEL Configuration
SWISS_MODEL_API_TOKEN = "9e8b3ac03b851bb3834cdb311045c78021087d1d"
SWISS_MODEL_BASE_URL = "https://swissmodel.expasy.org"
SWISS_MODEL_HEADERS = {"Authorization": f"Token {SWISS_MODEL_API_TOKEN}"}


def check_remark_465(pdb_path: str) -> bool:
    """
    Check if REMARK 465 (missing residues) is present in PDB file.
    Returns True if missing residues are found, False otherwise.
    """
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('REMARK 465'):
                    # Skip header lines
                    if 'MISSING RESIDUES' in line or 'SSSEQ' in line:
                        continue
                    # If we find actual data lines, missing residues exist
                    if line.strip() and len(line.split()) > 2:
                        return True
        return False
    except Exception as e:
        print(f"Error checking REMARK 465: {e}")
        return False


def extract_favoured_info(csv_path: str):
    """
    Extract the Favoured percentage and format it cleanly.
    Returns a tuple (percentage (float), display_string (str))
    """
    try:
        with open(csv_path, 'r') as f:
            content = f.read()
            
            # Search for pattern: Favoured: ,count,(percent%)
            match = re.search(r'Favoured:.*\((\d+\.?\d*)%\)', content)
            
            if match:
                percentage = float(match.group(1))
                display_string = f"Favoured: {percentage}%"
                return percentage, display_string
            else:
                return 0.0, "Favoured: Not found"
                
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return 0.0, "Error reading stats"


def parse_fasta_file(fasta_path: str) -> list:
    """
    Parse FASTA file and return list of sequences.
    """
    sequences = []
    current_sequence = []
    
    try:
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_sequence:
                        sequences.append("".join(current_sequence))
                        current_sequence = []
                else:
                    if line:
                        current_sequence.append(line)
            if current_sequence:
                sequences.append("".join(current_sequence))
        return sequences
    except Exception as e:
        print(f"Error parsing FASTA file: {e}")
        return []


def run_swiss_model(fasta_path: str, pdb_id: str, progress_callback=None) -> str:
    """
    Run SWISS-MODEL homology modeling on the FASTA sequence.
    Returns the path to the generated PDB file or None if failed.
    """
    if progress_callback:
        progress_callback(0.1, desc="📖 Reading FASTA file...")
    
    sequences = parse_fasta_file(fasta_path)
    if not sequences:
        return None
    
    fasta_input = sequences if len(sequences) > 1 else sequences[0]
    
    if progress_callback:
        progress_callback(0.2, desc="🚀 Submitting job to SWISS-MODEL...")
    
    project_title = f"Homology_Model_{pdb_id}"
    payload = {
        "target_sequences": fasta_input,
        "project_title": project_title
    }
    
    try:
        submit_response = requests.post(
            f"{SWISS_MODEL_BASE_URL}/automodel/", 
            headers=SWISS_MODEL_HEADERS, 
            json=payload,
            timeout=60
        )
        submit_response.raise_for_status()
        project_id = submit_response.json().get("project_id")
        
    except Exception as e:
        print(f"Error submitting SWISS-MODEL job: {e}")
        return None
    
    # Poll for results
    max_wait_time = 1800
    start_time = time.time()
    poll_interval = 30
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            return None
        
        if progress_callback:
            progress_pct = 0.2 + (0.6 * (elapsed / max_wait_time))
            progress_callback(progress_pct, desc=f"⏳ Waiting for SWISS-MODEL (elapsed: {int(elapsed)}s)...")
        
        try:
            status_response = requests.get(
                f"{SWISS_MODEL_BASE_URL}/project/{project_id}/models/summary/", 
                headers=SWISS_MODEL_HEADERS,
                timeout=60
            )
            status_response.raise_for_status()
            status_data = status_response.json()
            job_status = status_data.get("status")
            
            if job_status == "COMPLETED":
                if progress_callback:
                    progress_callback(0.85, desc="📥 Downloading model...")
                
                models = status_data.get("models")
                if not models: return None
                
                model_id = models[0].get("model_id")
                output_filename = os.path.join(PROTEINS_DIR, f"{pdb_id}_swiss_model.pdb")
                
                pdb_response = requests.get(
                    f"{SWISS_MODEL_BASE_URL}/project/{project_id}/models/{model_id}.pdb",
                    headers=SWISS_MODEL_HEADERS,
                    timeout=60
                )
                pdb_response.raise_for_status()
                
                with open(output_filename, "w") as f:
                    f.write(pdb_response.text)
                    
                return output_filename
                
            elif job_status == "FAILED":
                return None
            elif job_status in ["RUNNING", "PENDING"]:
                time.sleep(poll_interval)
            else:
                time.sleep(poll_interval)
        
        except Exception:
            time.sleep(poll_interval)

def run_ramplot(progress=gr.Progress()):
    """
    Run Ramachandran plot analysis.
    """
    # 1. No structure case
    if not current_pdb_info["pdb_id"] or not current_pdb_info["pdb_path"]:
        return (
            gr.update(value="<div style='padding: 20px; background: #fee; border-radius: 8px; color: #c33;'>❌ No structure loaded. Please search for a protein first.</div>", visible=True),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False)
        )

    # IMMEDIATELY show a loading indicator (This is handled by the .click event below, 
    # but we ensure the status box is visible here)
    
    pdb_id = current_pdb_info["pdb_id"]
    pdb_path = current_pdb_info["pdb_path"]
    
    # ... (rest of your existing logic: Swiss-Model, subprocess.run, etc.) ...
    
def run_ramplot(progress=gr.Progress()):
    """
    Run Ramachandran plot analysis.
    Returns: (status_html, plot1, plot2, plot3, plot4, stats_html)
    """
    # 1. No structure case
    if not current_pdb_info["pdb_id"] or not current_pdb_info["pdb_path"]:
        return (
            gr.update(value="<div style='padding: 20px; background: #fee; border-radius: 8px; color: #c33;'>❌ No structure loaded.</div>", visible=True),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False)
        )

    pdb_id = current_pdb_info["pdb_id"]
    pdb_path = current_pdb_info["pdb_path"]
    swiss_model_used = False
    
    # 2. Check for REMARK 465
    progress(0.05, desc="🔍 Checking for missing residues (REMARK 465)...")
    has_missing_residues = check_remark_465(pdb_path)
    
    if has_missing_residues:
        progress(0.1, desc="⚠️ Missing residues detected - Attempting SWISS-MODEL...")
        fasta_path = os.path.join(PROTEINS_DIR, f"{pdb_id}.fasta")
        
        # --- FALLBACK CHECK 1: FASTA Existence ---
        if not os.path.exists(fasta_path):
            print("Warning: FASTA file not found. Skipping SWISS-MODEL and using original PDB.")
            progress(0.15, desc="⚠️ FASTA missing - Using original structure...")
            # We proceed with original pdb_path
        else:
            # Try running SWISS-MODEL
            swiss_model_path = run_swiss_model(fasta_path, pdb_id, progress)
            
            # --- FALLBACK CHECK 2: API Success ---
            if not swiss_model_path:
                print("Warning: SWISS-MODEL failed. Skipping and using original PDB.")
                progress(0.2, desc="⚠️ SWISS-MODEL failed - Reverting to original PDB...")
                # We proceed with original pdb_path
            else:
                # SUCCESS: Switch to new model
                pdb_path = swiss_model_path
                current_pdb_info["pdb_path"] = swiss_model_path
                swiss_model_used = True
                progress(0.9, desc="✅ SWISS-MODEL complete...")
    else:
        progress(0.1, desc="✅ No missing residues - using original...")

    # 3. Run Ramachandran analysis
    progress(0.3, desc="🔬 Running Ramachandran plot analysis...")

    try:
        input_folder = PROTEINS_DIR
        output_folder = RAMPLOT_OUTPUT_DIR
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

        progress(0.5, desc="Executing ramplot command...")

        # We must ensure we run ramplot on the SPECIFIC file we decided on (original or model)
        # ramplot typically takes a folder input ("-i input_folder"). 
        # To ensure it processes the correct file if we switched to a model, 
        # we might need to be careful if both files exist in the folder.
        # However, usually ramplot processes all PDBs in the folder.
        
        cmd = [
            "ramplot", "pdb", "-i", input_folder, "-o", output_folder,
            "-m", "0", "-r", "600", "-p", "png"
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)

        progress(0.8, desc="Loading generated plots...")

        plot_dir = os.path.join(output_folder, "Plots")
        plot_files = {
            'map2d': os.path.join(plot_dir, "MapType2DAll.png"),
            'map3d': os.path.join(plot_dir, "MapType3DAll.png"),
            'std2d': os.path.join(plot_dir, "StdMapType2DGeneralGly.png"),
            'std3d': os.path.join(plot_dir, "StdMapType3DGeneral.png"),
        }

        # 4. Extract Statistics
        csv_files = glob.glob(os.path.join(output_folder, "*.csv"))
        stats_html = ""
        
        if csv_files:
            # Assume the most recently modified CSV or the first one is relevant
            csv_path = csv_files[0]
            pct, clean_display_string = extract_favoured_info(csv_path)
            
            # Determine Color based on quality
            if pct >= 98:
                bg_color = "#d4edda" # Green
                text_color = "#155724"
                status_icon = "🌟 Excellent"
            elif pct >= 90:
                bg_color = "#fff3cd" # Yellow/Orange
                text_color = "#856404"
                status_icon = "✅ Acceptable"
            else:
                bg_color = "#f8d7da" # Red
                text_color = "#721c24"
                status_icon = "⚠️ Low Quality"

            stats_html = f"""
            <div style="margin-top: 15px; padding: 15px; background: {bg_color}; border-left: 5px solid {text_color}; border-radius: 4px; color: {text_color}; font-size: 1.1em;">
                <strong>{status_icon} Structure Quality</strong><br>
                <span style="font-size: 1.4em; font-weight: bold;">{clean_display_string}</span>
            </div>
            """
        else:
            stats_html = "<div style='color: gray;'>No statistics CSV found.</div>"

        # Safety check for images
        for name, path in plot_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing plot: {path}")

        progress(1.0, desc="✅ Complete!")

        success_msg = "✅ Ramachandran plot analysis completed!"
        if swiss_model_used:
            success_msg += "<br>🔧 Used SWISS-MODEL homology model"
        elif has_missing_residues:
             success_msg += "<br>⚠️ Original structure used (SWISS-MODEL skipped/failed)"
        
        return (
            gr.update(value=f"<div style='padding: 20px; background: #d4edda; border-radius: 8px; color: #155724;'>{success_msg}</div>", visible=True),
            gr.update(value=plot_files['map2d'], visible=True),
            gr.update(value=plot_files['map3d'], visible=True),
            gr.update(value=plot_files['std2d'], visible=True),
            gr.update(value=plot_files['std3d'], visible=True),
            gr.update(value=stats_html, visible=True)
        )

    except Exception as e:
        return (
            gr.update(value=f"<div style='padding: 20px; background: #fee; color: #c33;'>❌ Error: {str(e)}</div>", visible=True),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False)
        )