# admet_analysis.py
"""
ADMET Analysis Module
Adapts to chain-specific docking results and includes multiple poses per ligand.
"""

import os
import glob
import pandas as pd
import numpy as np
import subprocess
import re
from rdkit import Chem
from adme_py import ADME
from admet_ai import ADMETModel
from config import DOCKING_RESULTS_DIR

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Updated column list to include "Chain", "Pose" and Safety Flags for UI
DISPLAY_COLUMNS = [
    "Filename",                # Identifier
    "Chain",                   # Chain ID
    "Ligand",                  # Ligand Name
    "Pocket",                  # Pocket Name
    "Pose",                    # NEW: Pose Number
    "Docking Score",           # Docking score (specific to pose)
    "Final Decision",          # Final decision
    "Developability Score",    # Composite score
    "SA Score",                # SA score
    "QED",                     # QED
    "Lipinski",                # Rule of 5 (Needed for UI Warnings)
    "PAINS",                   # PAINS Filter (Needed for UI Warnings)
    "Brenk",                   # Brenk Filter (Needed for UI Warnings)
    "hERG",                    # hERG Toxicity
    "Ames",                    # Ames Mutagenicity
    "CYP3A4 Inhibition",       # CYP Flags (Part 1)
    "CYP2D6 Inhibition",       # CYP Flags (Part 2)
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def file_to_smiles(file_path):
    """
    Robust conversion of PDBQT/PDB to SMILES.
    Uses OpenBabel first as it handles PDBQT better, then RDKit.
    """
    # Method 1: OpenBabel (Preferred for PDBQT)
    try:
        # -ipdbqt or -ipdb auto-detected by extension usually, but being explicit helps
        ext = os.path.splitext(file_path)[1].lower().replace('.', '')
        cmd = ['obabel', file_path, f'-i{ext}', '-osmi']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        # Output format is usually: "SMILES\tFilename"
        if result.returncode == 0 and result.stdout.strip():
            # Get the first SMILES (matches Model 1)
            smiles = result.stdout.strip().splitlines()[0].split()[0]
            return smiles
    except Exception as e:
        print(f"OpenBabel conversion failed for {file_path}: {e}")

    # Method 2: RDKit (Fallback)
    try:
        if file_path.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(file_path, sanitize=False, removeHs=True)
        else:
            return None # RDKit doesn't natively support PDBQT well without extras
            
        if mol:
            try:
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol)
            except: pass
    except: pass

    return None

def extract_poses_data(pdbqt_file):
    """
    Parses a Vina PDBQT file to extract ALL poses and their scores.
    Returns a list of dicts: [{'pose': 1, 'score': -8.5}, ...]
    """
    poses = []
    current_model = None
    
    try:
        with open(pdbqt_file, 'r') as f:
            content = f.read()
        
        # Split by MODEL/ENDMDL logic
        # Vina output usually looks like:
        # MODEL 1
        # REMARK VINA RESULT:   -7.5 ...
        # ...
        # ENDMDL
        
        # Regex to find blocks. Note: Vina output is consistent.
        # We look for "REMARK VINA RESULT:   <score>" lines.
        # They appear in order of poses (Mode 1, Mode 2, etc.)
        
        pattern = re.compile(r"REMARK VINA RESULT:\s+([-\d\.]+)")
        matches = pattern.findall(content)
        
        for idx, score_str in enumerate(matches):
            poses.append({
                'pose': idx + 1,
                'score': float(score_str)
            })
            
    except Exception as e:
        print(f"Error parsing poses from {pdbqt_file}: {e}")
        # Fallback: If no remarks found, assume single pose with default score
        poses.append({'pose': 1, 'score': -7.0})
        
    return poses

# ==========================================
# 3. FILTERING LOGIC
# ==========================================

def apply_primary_filters(row):
    """Stage 1: Primary Filters (Hard Stops)"""
    if row.get('Lipinski') == 'Fail': return 'REJECT', 'Lipinski Fail'
    if row.get('PAINS') == 'Yes': return 'REJECT', 'PAINS Alert'
    if row.get('Brenk') == 'Yes': return 'REJECT', 'Brenk Alert'
    if row.get('Ames') in ['Positive', 1]: return 'REJECT', 'Ames Positive'
    
    herg_val = row.get('hERG', '')
    if herg_val == 'High' or (isinstance(herg_val, (int, float)) and herg_val >= 0.7):
        return 'REJECT', 'hERG High Risk'
    elif herg_val == 'Medium' or (isinstance(herg_val, (int, float)) and 0.3 <= herg_val < 0.7):
        return 'REVIEW', 'hERG Medium Risk'

    if row.get('Carcinogenicity') in ['Yes', 1]: return 'REJECT', 'Carcinogenic'
    
    dili_val = row.get('DILI', '')
    if dili_val == 'High' or (isinstance(dili_val, (int, float)) and dili_val >= 0.7):
        return 'REJECT', 'DILI High Risk'
    
    return 'PASS', None

def apply_developability_filters(row):
    """Stage 2: Developability Filters (Soft Constraints)"""
    reject_reasons = []
    review_reasons = []

    def get_float(key, default):
        try: return float(row.get(key, default))
        except: return default

    sa = get_float('SA Score', 5.0)
    qed = get_float('QED', 0.6)
    mw = get_float('MW', 400)
    tpsa = get_float('TPSA', 100)
    wlogp = get_float('WLogP', 3)

    if sa > 6.0: reject_reasons.append('SA > 6.0')
    elif 5.0 < sa <= 6.0: review_reasons.append('SA 5.0-6.0')
    
    if qed < 0.4: reject_reasons.append('QED < 0.4')
    elif 0.4 <= qed < 0.6: review_reasons.append('QED 0.4-0.6')
    
    if mw > 550: reject_reasons.append('MW > 550')
    if tpsa > 160: reject_reasons.append('TPSA > 160')
    if wlogp > 6: reject_reasons.append('WLogP > 6')
    
    if reject_reasons: return 'REJECT', '; '.join(reject_reasons)
    elif review_reasons: return 'REVIEW', '; '.join(review_reasons)
    return 'ACCEPT', None

def calculate_adme_penalties(row):
    """Stage 3: ADME Risk Flags"""
    penalty = 0
    gi = row.get('GI Absorption', 'High')
    if gi in ['Medium', 'Moderate']: penalty += 5
    
    caco2 = row.get('Caco-2 (Wang)', '')
    if caco2 == 'Moderate': penalty += 5
    
    bbb = row.get('BBB (Martins)', '')
    if bbb == 'Borderline': penalty += 5

    ppb = row.get('PPB (AZ)', 90)
    try: ppb = float(ppb)
    except: ppb = 90
    if ppb >= 95: penalty += 5
    
    if row.get('CYP3A4 Inhibition') in ['Yes', 'Weak', 1]: penalty += 5
    if row.get('CYP2D6 Inhibition') in ['Yes', 'Weak', 1]: penalty += 5
    
    herg = row.get('hERG', '')
    if herg == 'Medium' or (isinstance(herg, (int, float)) and 0.3 <= herg < 0.7):
        penalty += 10
    
    return penalty

def calculate_developability_score(row, docking_score):
    """Stage 4: Composite Score Calculation"""
    base_score = 100
    penalty = 0
    
    sa = row.get('SA Score', 5.0)
    try: sa = float(sa)
    except: sa = 5.0
    if 5.0 <= sa <= 6.0: penalty += 10
    elif sa > 6.0: return 0 
    
    qed = row.get('QED', 0.6)
    try: qed = float(qed)
    except: qed = 0.6
    if qed < 0.6: penalty += 10
    
    # Docking Score Influence (Bonus/Penalty)
    # If score is very good (<-8.0), small bonus. If bad (>-6.0), penalty.
    try:
        ds = float(docking_score)
        if ds < -8.0: base_score += 5
        elif ds > -6.0: penalty += 10
    except: pass

    penalty += calculate_adme_penalties(row)
    return max(0, base_score - penalty)

def make_final_decision(developability_score):
    """Stage 5: Score-based Decision"""
    if developability_score >= 75: return 'ACCEPT'
    elif 60 <= developability_score < 75: return 'REVIEW'
    else: return 'REJECT'

# ==========================================
# 4. MAIN EXECUTION PIPELINE
# ==========================================

def run_admet_prediction():
    print("--- Starting ADMET Prediction Pipeline (Multi-Pose) ---")
    
    # 1. FIND FILES (RECURSIVE SEARCH IN docked_pdbqt)
    # We look in docked_pdbqt because these files contain the SCORES and ALL POSES.
    # Pattern: .../Chain_X/docked_pdbqt/ligand_pocket_poses.pdbqt
    
    pdbqt_files = []
    
    if os.path.exists(DOCKING_RESULTS_DIR):
        for root, dirs, files in os.walk(DOCKING_RESULTS_DIR):
            if os.path.basename(root) == "docked_pdbqt":
                for file in files:
                    if file.endswith("_poses.pdbqt"):
                        full_path = os.path.join(root, file)
                        pdbqt_files.append(full_path)
    
    if not pdbqt_files:
        print(f"No '_poses.pdbqt' files found in {DOCKING_RESULTS_DIR}.")
        return "❌ No docking results found.", None, None

    # 2. PREPARE DATA (Batching Unique Ligands)
    # We want to run ADME only ONCE per unique ligand SMILES, then map to all poses.
    
    print(f"Found {len(pdbqt_files)} docking files. Extracting Data...")
    
    unique_ligands = {} # Map file_path -> {smiles, chain, ligand_name, pocket_name}
    all_rows = []       # Final list to become DataFrame
    
    smiles_for_prediction = []
    
    for file_path in pdbqt_files:
        # Parse Filename: ligand_pocket_poses.pdbqt
        filename = os.path.basename(file_path)
        base_name = filename.replace("_poses.pdbqt", "")
        
        # Try to split Ligand and Pocket (Assuming standard format from docking.py)
        # Format: {ligand_name}_{pocket_name}_poses.pdbqt
        # Note: ligand name might contain underscores, so this is heuristic.
        # But docking.py usually creates distinct ligand/pocket names.
        # We will just store the base_name as identifier if splitting is ambiguous.
        
        parts = base_name.split('_')
        if len(parts) >= 2:
            # Heuristic: Pocket usually starts with 'p2rank' or 'fpocket' or 'pocket'
            # We'll join everything before the pocket keyword as ligand
            pocket_index = -1
            for i, p in enumerate(parts):
                if 'pocket' in p or 'p2rank' in p:
                    pocket_index = i
                    break
            
            if pocket_index > 0:
                ligand_name = "_".join(parts[:pocket_index])
                pocket_name = "_".join(parts[pocket_index:])
            else:
                ligand_name = parts[0]
                pocket_name = "_".join(parts[1:])
        else:
            ligand_name = base_name
            pocket_name = "Unknown"

        # Chain Name
        try:
            path_parts = os.path.normpath(file_path).split(os.sep)
            chain_name = path_parts[-3] # .../Chain_X/docked_pdbqt/file
        except: chain_name = "Unknown"

        # Get SMILES
        smiles = file_to_smiles(file_path)
        if not smiles: continue
        
        # Store Unique Info
        if smiles not in unique_ligands:
            unique_ligands[smiles] = {
                'smiles': smiles,
                'files_mapped': []
            }
            smiles_for_prediction.append(smiles)
            
        # Extract Poses & Scores from this specific file
        poses_data = extract_poses_data(file_path)
        
        # Add to mapping list
        unique_ligands[smiles]['files_mapped'].append({
            'filename': filename,
            'chain': chain_name,
            'ligand': ligand_name,
            'pocket': pocket_name,
            'poses': poses_data # List of {pose: 1, score: -7.5}
        })

    if not smiles_for_prediction:
        return "❌ Failed to extract SMILES from files.", None, None

    # 3. RUN ADME/ADMET PREDICTIONS (ON UNIQUE SMILES ONLY)
    print(f"Running ADMET AI on {len(smiles_for_prediction)} unique structures...")
    
    # Run ADME-Py
    adme_results = {}
    for sm in smiles_for_prediction:
        row = {}
        try:
            res = ADME(sm).calculate()
            p = res.get("physiochemical", {})
            m = res.get("medicinal", {})
            pk = res.get("pharmacokinetics", {})
            
            row["MW"] = p.get("molecular_weight")
            row["TPSA"] = p.get("tpsa")
            row["HBD"] = p.get("num_h_donors")
            row["HBA"] = p.get("num_h_acceptors")
            row["Rotatable Bonds"] = p.get("num_rotatable_bonds")
            row["Fsp3"] = p.get("sp3_carbon_ratio")
            row["WLogP"] = res.get("lipophilicity", {}).get("wlogp")
            row["GI Absorption"] = pk.get("gastrointestinal_absorption")
            row["Lipinski"] = "Pass" if res.get("druglikeness", {}).get("lipinski") else "Fail"
            row["PAINS"] = "Yes" if m.get("pains") else "No"
            row["Brenk"] = "Yes" if m.get("brenk") else "No"
            row["SA Score"] = m.get("synthetic_accessibility")
        except: pass
        adme_results[sm] = row

    # Run ADMET-AI
    ai_results = {}
    try:
        model = ADMETModel()
        preds_df = model.predict(smiles_for_prediction)
        
        rename_map = {
            "Caco2_Wang": "Caco-2 (Wang)", "BBB_Martins": "BBB (Martins)",
            "PPBR_AZ": "PPB (AZ)", "CYP3A4_Veith": "CYP3A4 Inhibition",
            "CYP2D6_Veith": "CYP2D6 Inhibition", "hERG": "hERG",
            "AMES": "Ames", "DILI": "DILI", "Carcinogens_Lagunin": "Carcinogenicity",
            "QED": "QED"
        }
        
        # Map back to SMILES (assuming order is preserved, which it is for lists)
        for idx, sm in enumerate(smiles_for_prediction):
            ai_row = {}
            for col, new_name in rename_map.items():
                if col in preds_df.columns:
                    ai_row[new_name] = preds_df.iloc[idx][col]
            ai_results[sm] = ai_row
            
    except Exception as e:
        print(f"ADMET-AI Failed: {e}")

    # 4. EXPAND TO ALL POSES
    print("Expanding results to all poses...")
    
    for sm, info in unique_ligands.items():
        base_adme = adme_results.get(sm, {})
        base_ai = ai_results.get(sm, {})
        
        # Combine base data
        base_data = {**base_adme, **base_ai}
        base_data['SMILES'] = sm
        
        for file_map in info['files_mapped']:
            # For each file (Ligand/Pocket combo)
            for pose_entry in file_map['poses']:
                # For each POSE in that file
                row = base_data.copy()
                row['Filename'] = file_map['filename']
                row['Chain'] = file_map['chain']
                row['Ligand'] = file_map['ligand']
                row['Pocket'] = file_map['pocket']
                row['Pose'] = pose_entry['pose']
                row['Docking Score'] = pose_entry['score']
                
                all_rows.append(row)

    final_df = pd.DataFrame(all_rows)

    # 5. APPLY DECISION LOGIC
    print("Applying Decision Filters...")
    decisions = []
    dev_scores = []

    for idx, row in final_df.iterrows():
        p_res, p_reason = apply_primary_filters(row)
        if p_res == 'REJECT':
            decisions.append(f"REJECT ({p_reason})")
            dev_scores.append(0)
            continue
        
        d_res, d_reason = apply_developability_filters(row)
        if d_res == 'REJECT':
            decisions.append(f"REJECT ({d_reason})")
            dev_scores.append(0)
            continue
        
        score = calculate_developability_score(row, row.get('Docking Score'))
        dev_scores.append(score)
        
        final_dec = make_final_decision(score)
        
        warnings = []
        if p_res == 'REVIEW': warnings.append(p_reason)
        if d_res == 'REVIEW': warnings.append(d_reason)
        
        if warnings:
            warning_text = "; ".join(warnings)
            if final_dec == 'ACCEPT': decisions.append(f"REVIEW ({warning_text})")
            else: decisions.append(f"{final_dec} ({warning_text})")
        else:
            decisions.append(final_dec)

    final_df['Developability Score'] = dev_scores
    final_df['Final Decision'] = decisions

   # 6. CLEANUP & SAVE
    # Sort by Score (Best first)
    final_df = final_df.sort_values(by=['Developability Score', 'Docking Score'], ascending=[False, True])

    # KEEP ALL COLUMNS (Raw/Unfiltered)
    # We just round the numeric columns so it doesn't look messy in the UI
    numeric_cols = final_df.select_dtypes(include=['float64', 'float32']).columns
    final_df.loc[:, numeric_cols] = final_df[numeric_cols].round(2)

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "final_admet_report.csv")
    final_df.to_csv(csv_path, index=False)
    
    msg = f"Analysis Complete. Generated report for {len(final_df)} poses."
    return msg, final_df, csv_path