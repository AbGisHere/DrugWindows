"""
Protein preparation module using Meeko
"""

import os
import subprocess
import sys
import tempfile
import gradio as gr
from config import current_pdb_info, PREPARED_PROTEIN_DIR
from visualization import show_structure

# Import Utils for the search logic
import utils

# Import the Ramachandran module (assumed filename: ramachandran.py)
# This module contains the logic to check REMARK 465 and run Swiss-Model
from ramachandran import run_ramplot


def prepare_protein_meeko():
    """
    Prepare protein using Meeko for docking.
    Includes robust retry logic: if preparation fails, it searches for a new PDB,
    runs the validation/Swiss-Model pipeline, and tries again.
    """
    
    # 1. Check for loaded structure
    if not current_pdb_info.get("pdb_id") or not current_pdb_info.get("pdb_path"):
        return (
            gr.update(value="<div style='padding: 20px; background: #fee; border-radius: 8px; color: #c33;'>❌ No structure loaded. Please search for a disease first.</div>", visible=True),
            gr.update(value=""),
            gr.update(value=None)
        )
    
    pdb_path = current_pdb_info["pdb_path"]
    pdb_id = current_pdb_info["pdb_id"]
    
    # --- CHECK FOR MISSING FILE ---
    # If the file is missing, trigger retry immediately
    if not os.path.exists(pdb_path):
        yield from trigger_retry_pipeline(pdb_id, "File missing")
        return

    is_swiss = "swiss_model" in os.path.basename(pdb_path)
    source_msg = "SWISS-MODEL Homology Structure" if is_swiss else "Original Crystal Structure"
    
    # Show processing message
    yield (
        gr.update(value=f"<div style='padding: 20px; background: #fff3cd; border-radius: 8px; color: #856404;'>⚙️ Preparing protein with Meeko...<br>🔍 Source: <b>{source_msg}</b></div>", visible=True),
        gr.update(value=""),
        gr.update(value=None)
    )
    
    output_dir = PREPARED_PROTEIN_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    output_base = os.path.join(output_dir, "prepared_protein")
    
    # 2. Run Meeko
    cmd = [  
        sys.executable, "mk_prepare_receptor.py", 
        "-i", pdb_path,
        "-o", output_base,
        "-p",
        "--charge_model", "gasteiger",
        "--default_altloc", "A"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # The output file will be output_base.pdbqt
        pdbqt_path = f"{output_base}.pdbqt"
        
        if not os.path.exists(pdbqt_path):
            # If Meeko failed to produce output, assume structure is bad -> RETRY
            print(f"Meeko failed to generate PDBQT for {pdb_id}")
            yield from trigger_retry_pipeline(pdb_id, "Meeko output generation failed")
            return
        
        # --- SUCCESS PATH ---
        
        # Store prepared protein path globally
        current_pdb_info["prepared_pdbqt"] = pdbqt_path
        
        # Read PDBQT content
        with open(pdbqt_path, 'r') as f:
            pdbqt_content = f.read()
        
        # Create 3D visualization
        protein_name = f"Prepared: {pdb_id} ({'Swiss-Model' if is_swiss else 'Original'})"
        
        structure_html = show_structure(
            protein_text=pdbqt_content, 
            ligand_text=None, 
            pdb_id=pdb_id, 
            protein_name=protein_name
        )
        
        # Create download file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False)
        temp_file.write(pdbqt_content)
        temp_file.close()
        
        success_msg = "<div style='padding: 20px; background: #d4edda; border-radius: 8px; color: #155724;'>"
        success_msg += f"✅ Protein preparation completed!<br>ℹ️ Used: <b>{source_msg}</b><br>"
        success_msg += f"<small>Output: {pdbqt_path}</small>"
        if result.stdout:
            success_msg += f"<br><small>{result.stdout}</small>"
        success_msg += "</div>"
        
        yield (
            gr.update(value=success_msg, visible=True),
            gr.update(value=structure_html),
            gr.update(value=temp_file.name)
        )
        
    except (subprocess.CalledProcessError, Exception) as e:
        # Catch ANY failure during preparation and retry
        print(f"Preparation failed for {pdb_id}. Reason: {e}")
        yield from trigger_retry_pipeline(pdb_id, str(e))


def trigger_retry_pipeline(failed_pdb_id, reason):
    """
    Helper generator to handle the retry logic:
    1. Exclude failed PDB.
    2. Search for next best PDB (Utils).
    3. Validate and Model (Ramachandran/Swiss).
    4. Recursively call Protein Prep.
    """
    
    # 1. Initialize exclusion list if not present
    if "failed_pdbs" not in current_pdb_info:
        current_pdb_info["failed_pdbs"] = []
    
    # 2. Add current failure to list
    if failed_pdb_id not in current_pdb_info["failed_pdbs"]:
        current_pdb_info["failed_pdbs"].append(failed_pdb_id)
    
    fail_msg = f"<div style='padding: 20px; background: #fee; border-radius: 8px; color: #c33;'>⚠️ Preparation failed for <b>{failed_pdb_id}</b> ({reason}).<br>🔄 <b>Retrying with next best structure...</b></div>"
    
    yield (
        gr.update(value=fail_msg, visible=True),
        gr.update(value=""),
        gr.update(value=None)
    )
    
    # 3. Retrieve original search term
    search_term = current_pdb_info.get("search_term", "")
    if not search_term:
        yield (gr.update(value="❌ Retry failed: Could not recall search term."), gr.update(), gr.update())
        return

    # 4. Search for Next Best Candidate (Utils Step)
    print(f"Retrying search for '{search_term}', excluding: {current_pdb_info['failed_pdbs']}")
    new_result = utils.find_best_pdb_structure(
        search_term, 
        excluded_pdbs=current_pdb_info["failed_pdbs"]
    )
    
    if not new_result:
        yield (gr.update(value="❌ Retry failed: No other suitable structures found."), gr.update(), gr.update())
        return
        
    new_pdb_id, new_pdb_path = new_result
    
    # Update Global Info with New PDB
    current_pdb_info["pdb_id"] = new_pdb_id
    current_pdb_info["pdb_path"] = new_pdb_path
    
    # 5. RERUN RAMACHANDRAN / SWISS MODEL PIPELINE
    # This ensures the new PDB is checked for missing residues and modeled if necessary
    try:
        msg = f"Found new candidate: <b>{new_pdb_id}</b>. Running validation and Swiss-Model check..."
        yield (gr.update(value=msg), gr.update(), gr.update())

        print(f"Triggering Ramachandran/Swiss-Model check for {new_pdb_id}...")
        
        # We call the function from ramchandran.py
        # This function updates current_pdb_info['pdb_path'] internally if Swiss-Model is generated
        run_ramplot() 
        
    except Exception as e:
        yield (gr.update(value=f"❌ Pipeline retry failed during structure validation: {e}"), gr.update(), gr.update())
        return

    # 6. RECURSIVE CALL TO PROTEIN PREP
    # Now that we have a new PDB (and potentially new Swiss model), run prep again
    yield from prepare_protein_meeko()