"""
Core utility functions for protein structure analysis.
Includes disease mapping, UniProt search, PDB filtering, and structure download.
"""

import requests
import os
import shutil
from typing import Optional, List, Dict, Tuple
from config import DISEASE_PROTEIN_MAP, current_pdb_info # Ensure current_pdb_info is imported to store the name

# --- NEW HELPER FUNCTION ---
def clear_proteins_folder(folder_path: str = "proteins"):
    """
    Deletes all files in the proteins folder to ensure only the 
    current analysis file exists.
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder_path, exist_ok=True)
    
    print(f"✓ Cleared '{folder_path}' directory.")

def map_disease_to_protein(disease_input: str) -> Optional[str]:
    """Map a disease name or condition to its protein target."""
    disease_input = disease_input.replace("-", "").lower().strip()
    
    for category, conditions in DISEASE_PROTEIN_MAP.items():
        if disease_input in conditions:
            return conditions[disease_input]
        if disease_input == category:
            return list(conditions.values())[0]
    
    for category, conditions in DISEASE_PROTEIN_MAP.items():
        for condition_key, protein_name in conditions.items():
            if disease_input in condition_key or condition_key in disease_input:
                return protein_name
    
    return None

def search_uniprot_for_reviewed_human(protein_name: str, limit: int = 5) -> List[str]:
    """Search UniProt for reviewed (Swiss-Prot) human protein entries."""
    url = "https://rest.uniprot.org/uniprotkb/search"
    query_string = f"{protein_name} AND (organism_id:9606) AND (reviewed:true)"

    params = {
        "query": query_string,
        "format": "json",
        "size": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        results = data.get('results', [])
        if not results:
            return []
        
        return [r['primaryAccession'] for r in results]
        
    except requests.exceptions.RequestException as e:
        print(f"UniProt search error: {e}")
        return []

def get_pdb_ids_from_uniprot(uniprot_id: str) -> List[str]:
    """Get all PDB IDs associated with a UniProt entry."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        pdb_ids = []
        cross_refs = data.get('uniProtKBCrossReferences', [])
        
        for ref in cross_refs:
            if ref.get('database') == 'PDB':
                pdb_id = ref.get('id')
                if pdb_id:
                    pdb_ids.append(pdb_id)
        
        return pdb_ids
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching PDB IDs from UniProt: {e}")
        return []

def get_pdb_resolution(pdb_id: str) -> Optional[float]:
    """Get the resolution of a PDB structure."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        experimental_method = data.get('exptl', [{}])[0].get('method', '')
        if 'X-RAY' not in experimental_method.upper():
            return None
        
        resolution = None
        if 'rcsb_entry_info' in data:
            resolution = data['rcsb_entry_info'].get('resolution_combined')
        
        if resolution is None and 'refine' in data:
            refine_list = data['refine']
            if refine_list and len(refine_list) > 0:
                resolution = refine_list[0].get('ls_d_res_high')
        
        if isinstance(resolution, list):
            resolution = resolution[0] if resolution else None
        
        return float(resolution) if resolution is not None else None
        
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError) as e:
        print(f"Error fetching resolution for {pdb_id}: {e}")
        return None

def check_mutations_in_pdb(pdb_id: str) -> bool:
    """Check if a PDB structure contains mutations in ANY chain."""
    try:
        entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        entry_response = requests.get(entry_url, timeout=30)
        entry_response.raise_for_status()
        entry_data = entry_response.json()
        num_entities = entry_data.get('rcsb_entry_info', {}).get('polymer_entity_count', 1)
    except requests.exceptions.RequestException:
        num_entities = 1 
    
    for entity_id in range(1, num_entities + 1):
        url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 404: continue
            response.raise_for_status()
            data = response.json()
            
            mutation_count = 0
            if 'entity_poly' in data:
                mutation_count = data['entity_poly'].get('rcsb_mutation_count', 0)
            if mutation_count == 0 and 'rcsb_polymer_entity' in data:
                mutation_count = data['rcsb_polymer_entity'].get('mutation_count', 0)
                
            if mutation_count > 0:
                return True
        except requests.exceptions.RequestException:
            continue
    
    return False

def download_pdb_file(pdb_id: str, output_dir: str = "proteins") -> Optional[str]:
    """Download a PDB file and save it to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(output_path, 'w') as f:
            f.write(response.text)
        print(f"Downloaded {pdb_id}.pdb to {output_path}")
        return output_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDB file {pdb_id}: {e}")
        return None

def download_fasta_file(pdb_id: str, output_dir: str = "proteins") -> Optional[str]:
    """Download a FASTA file for a PDB structure."""
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    output_path = os.path.join(output_dir, f"{pdb_id}.fasta")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(output_path, 'w') as f:
            f.write(response.text)
        print(f"Downloaded {pdb_id}.fasta to {output_path}")
        return output_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading FASTA file {pdb_id}: {e}")
        return None

def is_pdb_id(text: str) -> bool:
    """Check if the input string looks like a PDB ID."""
    text = text.strip()
    return len(text) == 4 and text[0].isdigit() and text.isalnum()

# --- UPDATED FUNCTION ---
def find_best_pdb_structure(protein_name: str, max_check: int = 100, excluded_pdbs: List[str] = None) -> Optional[Tuple[str, str]]:
    """
    Find the best PDB structure for a protein, ignoring IDs in excluded_pdbs.
    """
    if excluded_pdbs is None:
        excluded_pdbs = []
        
    # Store the search term in global config so we can retry later if needed
    current_pdb_info["search_term"] = protein_name 
    
    # --- STEP 1: CLEAR OLD DATA ---
    clear_proteins_folder() 
    
    clean_input = protein_name.replace("-", "").strip()
    
    # Direct PDB ID Input
    if is_pdb_id(clean_input):
        pdb_id = clean_input.upper()
        # If user explicitly asks for a PDB that failed, we must try it anyway 
        # (or handle specific logic here, but usually manual input overrides exclusion)
        print(f"Input detected as PDB ID: {pdb_id}")
        pdb_path = download_pdb_file(pdb_id)
        if pdb_path:
            download_fasta_file(pdb_id)
            return (pdb_id, pdb_path)
        else:
            print(f"Could not download PDB ID {pdb_id}.")
            return None

    # UniProt Search
    print(f"Searching UniProt for: {clean_input}")
    uniprot_ids = search_uniprot_for_reviewed_human(clean_input, limit=5)
    
    if not uniprot_ids:
        print(f"No reviewed human protein found for: {clean_input}")
        return None
    
    print(f"Found {len(uniprot_ids)} UniProt candidates: {', '.join(uniprot_ids)}")
    
    overall_best_candidate = None 
    
    for uid_idx, uniprot_id in enumerate(uniprot_ids):
        print(f"\n--- Checking UniProt ID [{uid_idx+1}/{len(uniprot_ids)}]: {uniprot_id} ---")
        
        pdb_ids = get_pdb_ids_from_uniprot(uniprot_id)
        if not pdb_ids:
            continue 
        
        current_uniprot_best = None
        checked_count = 0
        
        for i, pdb_id in enumerate(pdb_ids):
            # --- SKIP EXCLUDED PDBS ---
            if pdb_id in excluded_pdbs:
                print(f"Skipping {pdb_id} (previously failed).")
                continue

            if checked_count >= max_check:
                break
            
            resolution = get_pdb_resolution(pdb_id)
            if resolution is None: continue
            
            checked_count += 1
            print(f"[{checked_count}] {pdb_id}: {resolution}Å", end="")
            
            if check_mutations_in_pdb(pdb_id):
                print(" - mutations, skipping")
                continue
            
            print(" - valid ✓")
            
            if current_uniprot_best is None or resolution < current_uniprot_best[1]:
                current_uniprot_best = (pdb_id, resolution)
                print(f"  → Current best for {uniprot_id}: {pdb_id} ({resolution}Å)")
                
                # Golden Ticket (< 2.0 A)
                if resolution < 2.0:
                    print(f"\n✓ Found excellent structure (< 2.0Å): {pdb_id} ({resolution}Å)")
                    pdb_path = download_pdb_file(pdb_id)
                    download_fasta_file(pdb_id)
                    if pdb_path:
                        return (pdb_id, pdb_path)
        
        if current_uniprot_best:
            if overall_best_candidate is None or current_uniprot_best[1] < overall_best_candidate[1]:
                overall_best_candidate = current_uniprot_best
                print(f"  ★ New Global Best Candidate: {overall_best_candidate[0]} ({overall_best_candidate[1]}Å)")
    
    if overall_best_candidate:
        pdb_id, resolution = overall_best_candidate
        print(f"\n✓ Search complete. Returning best available: {pdb_id} ({resolution}Å)")
        
        pdb_path = download_pdb_file(pdb_id)
        download_fasta_file(pdb_id)
        if pdb_path:
            return (pdb_id, pdb_path)
    
    print("\nNo suitable structure found.")
    return None