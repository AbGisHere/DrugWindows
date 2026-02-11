"""
Protein preparation module using Meeko
Updated: Handles UI updates for Search (Tab 1) and Ramachandran (Tab 2) on retry.
"""

import os
import subprocess
import sys
import tempfile
import gradio as gr
from config import current_pdb_info, PREPARED_PROTEIN_DIR
from visualization import show_structure

# Imports for Retry Logic
import utils
from ramachandran import run_ramplot 

# --- Constants for "No Change" updates ---
# We need to return 13 values in total. 
# 3 for Prep Tab + 4 for Search Tab + 6 for Ramachandran Tab = 13
def no_change(count=10):
    return tuple([gr.update() for _ in range(count)])

def generate_search_tab_updates(pdb_id, pdb_path, protein_name):
    """Generates the UI updates for Tab 1 (Search)."""
    try:
        with open(pdb_path, 'r') as f:
            pdb_content = f.read()
            
        info_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 24px; border-radius: 16px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style='margin-top:0;'>🧬 Structure Loaded (Retry)</h3>
            <p><strong>Target:</strong> {protein_name}</p>
            <p><strong>New PDB ID:</strong> {pdb_id}</p>
            <p><strong>Status:</strong> Structure replaced due to quality issues.</p>
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
            gr.update(value=f"✅ Switched to {pdb_id}", visible=True) # search_status
        )
    except Exception as e:
        print(f"Error generating search tab updates: {e}")
        return (gr.update(), gr.update(), gr.update(), gr.update())


def prepare_protein_meeko():
    """
    Prepare protein using Meeko.
    Yields 13 outputs: [3 Prep Outputs] + [4 Search Outputs] + [6 Ramplot Outputs]
    """
    
    # 1. Check for loaded structure
    if not current_pdb_info.get("pdb_id") or not current_pdb_info.get("pdb_path"):
        yield (
            gr.update(value="<div style='padding: 20px; background: #fee; color: #c33;'>❌ No structure loaded.</div>", visible=True),
            gr.update(value=""),
            gr.update(value=None),
        ) + no_change(10)
        return
    
    pdb_path = current_pdb_info["pdb_path"]
    pdb_id = current_pdb_info["pdb_id"]
    
    # --- CHECK MISSING FILE ---
    if not os.path.exists(pdb_path):
        yield from trigger_retry_pipeline(pdb_id, "File missing")
        return

    is_swiss = "swiss_model" in os.path.basename(pdb_path)
    source_msg = "SWISS-MODEL Homology Structure" if is_swiss else "Original Crystal Structure"
    
    # Show processing message (Prep Tab only)
    yield (
        gr.update(value=f"<div style='padding: 20px; background: #fff3cd; color: #856404;'>⚙️ Preparing {pdb_id}...<br>Source: {source_msg}</div>", visible=True),
        gr.update(value=""),
        gr.update(value=None)
    ) + no_change(10)
    
    output_dir = PREPARED_PROTEIN_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_base = os.path.join(output_dir, "prepared_protein")
    
    # 2. Run Meeko
    cmd = [  
        sys.executable, "mk_prepare_receptor.py", 
        "-i", pdb_path, "-o", output_base, "-p",
        "--charge_model", "gasteiger", "--default_altloc", "A"
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        pdbqt_path = f"{output_base}.pdbqt"
        
        if not os.path.exists(pdbqt_path):
            print(f"Meeko failed to generate PDBQT for {pdb_id}")
            yield from trigger_retry_pipeline(pdb_id, "Meeko output generation failed")
            return
        
        # --- SUCCESS ---
        current_pdb_info["prepared_pdbqt"] = pdbqt_path
        
        with open(pdbqt_path, 'r') as f:
            pdbqt_content = f.read()
        
        protein_name = f"Prepared: {pdb_id} ({'Swiss' if is_swiss else 'Original'})"
        structure_html = show_structure(
            protein_text=pdbqt_content, 
            ligand_text=None, 
            pdb_id=pdb_id, 
            protein_name=protein_name
        )
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', delete=False)
        temp_file.write(pdbqt_content)
        temp_file.close()
        
        success_msg = f"<div style='padding: 20px; background: #d4edda; color: #155724;'>✅ Preparation complete for <b>{pdb_id}</b>!</div>"
        
        # Standard Yield: Update Prep Tab, Leave others unchanged
        yield (
            gr.update(value=success_msg, visible=True),
            gr.update(value=structure_html),
            gr.update(value=temp_file.name)
        ) + no_change(10)
        
    except (subprocess.CalledProcessError, Exception) as e:
        print(f"Preparation failed for {pdb_id}: {e}")
        yield from trigger_retry_pipeline(pdb_id, str(e))


def trigger_retry_pipeline(failed_pdb_id, reason):
    """
    Handles retry logic. Updates ALL tabs (Search, Ramachandran, Prep).
    """
    # 1. Update Exclusion List
    if "failed_pdbs" not in current_pdb_info:
        current_pdb_info["failed_pdbs"] = []
    if failed_pdb_id not in current_pdb_info["failed_pdbs"]:
        current_pdb_info["failed_pdbs"].append(failed_pdb_id)
    
    # Yield "Retrying" message on Prep Tab
    fail_msg = f"<div style='padding: 20px; background: #fee; color: #c33;'>⚠️ Failed: {failed_pdb_id} ({reason}).<br>🔄 <b>Searching for next best structure...</b></div>"
    yield (gr.update(value=fail_msg), gr.update(), gr.update()) + no_change(10)
    
    # 2. Find New Structure
    search_term = current_pdb_info.get("search_term", "")
    if not search_term:
        yield (gr.update(value="❌ Retry failed: Search term lost."), gr.update(), gr.update()) + no_change(10)
        return

    print(f"Retrying search for '{search_term}', excluding: {current_pdb_info['failed_pdbs']}")
    new_result = utils.find_best_pdb_structure(search_term, excluded_pdbs=current_pdb_info["failed_pdbs"])
    
    if not new_result:
        yield (gr.update(value="❌ Retry failed: No structure found."), gr.update(), gr.update()) + no_change(10)
        return
        
    new_pdb_id, new_pdb_path = new_result
    
    # Update Global Config
    current_pdb_info["pdb_id"] = new_pdb_id
    current_pdb_info["pdb_path"] = new_pdb_path
    
    # 3. GENERATE UPDATES FOR OTHER TABS
    
    # A. Tab 1 (Search) Updates
    search_updates = generate_search_tab_updates(new_pdb_id, new_pdb_path, search_term)
    
    # B. Tab 2 (Ramachandran) Updates - Run analysis immediately
    try:
        print(f"Running Ramachandran for new structure {new_pdb_id}...")
        # run_ramplot returns (status, p1, p2, p3, p4, stats)
        ram_updates = run_ramplot() 
    except Exception as e:
        print(f"Ramachandran failed during retry: {e}")
        ram_updates = tuple([gr.update() for _ in range(6)])

    # 4. YIELD UPDATES FOR ALL TABS IMMEDIATELY
    # This ensures the user sees the new structure and plots BEFORE prep finishes
    
    prep_msg = f"<div style='padding: 20px; background: #fff3cd; color: #856404;'>✅ Found <b>{new_pdb_id}</b>. Analysis updated.<br>⚙️ Now preparing protein...</div>"
    
    yield (
        gr.update(value=prep_msg), 
        gr.update(), 
        gr.update()
    ) + search_updates + ram_updates

    # 5. RECURSIVE CALL (Run Prep on new file)
    yield from prepare_protein_meeko()