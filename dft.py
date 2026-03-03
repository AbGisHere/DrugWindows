import os
import subprocess
import shutil
import time
import re
import csv
from pathlib import Path
from datetime import datetime
from Bio import PDB
from Bio.PDB.NeighborSearch import NeighborSearch

# ================= USER CONFIGURATION =================
# 1. ORCA SETUP
BASE_DIR = Path(os.getcwd()).resolve()
ORCA_FOLDER = BASE_DIR / "orca_program"

# Detect ORCA executable
ORCA_EXE = ORCA_FOLDER / "orca.exe" 
if not ORCA_EXE.exists():
    ORCA_EXE = ORCA_FOLDER / "orca_startup_mpi.exe"

# 2. INPUT DATA (This line is overwritten by app.py automatically)
PDB_FILE = 'docking_results\\Chain_A\\docked_pdb\\Alectinib_fpocket_pocket2_complex.pdb'

# (Optional fallback) If auto-detect fails, it will look for this
LIGAND_NAME = "UNL"            

# --- OPTIMIZATION: TIGHTER BUFFER ---
POCKET_RADIUS = 3.2          

# 3. CALCULATION SETTINGS
WORKING_DIR = BASE_DIR / "orca_temp"
WORKING_DIR.mkdir(exist_ok=True) # Ensure the temp folder exists

INPUT_NAME  = "ligand.inp"
OUTPUT_NAME = "ligand.out"

# 4. OUTPUT CSV FILES
METRICS_CSV = WORKING_DIR / "dft_batch_results.csv"
CHARGES_CSV = WORKING_DIR / "orca_mulliken_charges.csv"
# ======================================================

def get_pocket_atoms(pdb_path, radius):
    """Parses PDB and dynamically extracts the ligand + immediate environment."""
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", pdb_path)
    except Exception as e:
        print(f"    ❌ Error reading PDB: {e}")
        return []

    # 1. AUTO-DETECT LIGAND RESIDUE NAME
    # Standard amino acids and common ignorables (water, simple ions)
    std_aas = {'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'}
    ignore_res = {'HOH', 'WAT', 'H2O', 'NA', 'CL', 'MG', 'ZN', 'CA', 'K'}
    
    ligand_resnames = set()
    for res in structure.get_residues():
        res_name = res.get_resname().strip()
        # If it's not a standard amino acid and not water/simple ion
        if res_name not in std_aas and res_name not in ignore_res:
            ligand_resnames.add(res_name)

    # Fallback to the hardcoded name just in case the auto-detect filter was too strict
    if not ligand_resnames:
        ligand_resnames.add(LIGAND_NAME)
        
    detected_ligands = list(ligand_resnames)
    print(f"    -> Auto-detected Ligand Residue Code(s): {detected_ligands}")

    ligand_atoms = []
    all_atoms = list(structure.get_atoms())

    # Extract all atoms belonging to the dynamically detected ligand(s)
    for atom in all_atoms:
        if atom.get_parent().get_resname().strip() in detected_ligands:
            ligand_atoms.append(atom)
    
    if not ligand_atoms:
        print(f"    ❌ ERROR: No atoms found for detected ligands {detected_ligands} in PDB.")
        return []

    # 2. EXTRACT NEIGHBORS (POCKET BUFFER)
    ns = NeighborSearch(all_atoms)
    nearby_residues = set()
    
    for latom in ligand_atoms:
        neighbors = ns.search(latom.get_coord(), radius, level='R')
        nearby_residues.update(neighbors)

    formatted_lines = []
    def format_atom(atom):
        x, y, z = atom.get_coord()
        element = ''.join([i for i in atom.element if not i.isdigit()])
        return f"{element:<2} {x:12.6f} {y:12.6f} {z:12.6f}"

    for atom in ligand_atoms:
        formatted_lines.append(format_atom(atom))
        
    for res in nearby_residues:
        res_name = res.get_resname().strip()
        if res_name not in detected_ligands and res_name not in ignore_res:
            for atom in res.get_atoms():
                formatted_lines.append(format_atom(atom))
                
    return formatted_lines

def write_orca_input(filepath, atom_lines, charge, mult):
    """Writes the .inp file with High-Speed settings."""
    mpi_path = shutil.which("mpiexec")
    
    with open(filepath, "w") as f:
        f.write("! r2SCAN-3c Opt\n\n")  # Efficient composite method
        
        if mpi_path:
            f.write("%pal nprocs 16 end\n\n")
        else:
            print("   [Info] MS-MPI not found. Switching to Serial mode (1 core).")
        
        f.write(f"* xyz {charge} {mult}\n")
        for line in atom_lines:
            f.write(line + "\n")
        f.write("*\n")

def run_orca_attempt(charge, mult, atoms):
    print(f"    --- Attempting Calculation: Charge {charge}, Multiplicity {mult} ---")
    
    inp_path = WORKING_DIR / INPUT_NAME
    write_orca_input(inp_path, atoms, charge, mult)
    
    if not inp_path.exists():
        print("    >> ERROR: Input file creation failed.")
        return False
    
    env = os.environ.copy()
    if str(ORCA_FOLDER) not in env["PATH"]:
        env["PATH"] = str(ORCA_FOLDER) + os.pathsep + env["PATH"]
        
    out_path = WORKING_DIR / OUTPUT_NAME
    start_time = time.time()
    
    with open(out_path, "w") as outfile:
        try:
            result = subprocess.run(
                [str(ORCA_EXE), INPUT_NAME], 
                cwd=str(WORKING_DIR),
                stdout=outfile,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"    >> SUCCESS! Calculation finished in {duration:.1f} seconds.")
                return True
            else:
                print(f"    >> FAILED with code {result.returncode}")
                err_msg = result.stderr.lower()
                if "impossible" in err_msg and "electrons" in err_msg:
                    print("    >> DIAGNOSIS: Electron count mismatch (Odd vs Even). Switching state...")
                elif "mpiexec" in err_msg:
                    print("    >> CRITICAL ERROR: MPI Issue Detected.")
                else:
                    print("    >> ERROR DETAILS (Tail):")
                    print(result.stderr[-500:]) 
                return False

        except Exception as e:
            print(f"    >> EXECUTION ERROR: {e}")
            return False

def extract_and_save_data(output_filepath, pdb_name, charge, mult):
    """Parses ORCA output and saves HOMO/LUMO/Gap and Charges to CSV."""
    
    if not os.path.exists(output_filepath):
        print("    Error: Output file not found for parsing.")
        return

    with open(output_filepath, 'r') as f:
        lines = f.readlines()

    homo_val, lumo_val, gap_val = None, None, None
    mulliken_charges = []
    
    # Regex for orbital energies: Index, Occ, Energy(Eh), Energy(eV)
    orb_pattern = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)")
    
    # 1. PARSE FILE
    extract_mulliken = False
    for i, line in enumerate(lines):
        
        # --- ORBITAL ENERGIES ---
        if "ORBITAL ENERGIES" in line:
            for j in range(i + 4, min(i + 1000, len(lines))):
                match = orb_pattern.match(lines[j])
                if match:
                    occ = float(match.group(2))
                    if occ == 0.0000:
                        prev_match = orb_pattern.match(lines[j-1])
                        if prev_match:
                            homo_val = float(prev_match.group(4))
                            lumo_val = float(match.group(4))     
                            gap_val = round(lumo_val - homo_val, 4)
                        break
        
        # --- MULLIKEN CHARGES ---
        if "MULLIKEN ATOMIC CHARGES" in line:
            mulliken_charges = [] 
            extract_mulliken = True
            continue
        
        if extract_mulliken:
            if "Sum of atomic charges" in line:
                extract_mulliken = False
            elif ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    atom_info = parts[0].strip()
                    try:
                        charge_val = float(parts[1].strip())
                        mulliken_charges.append((atom_info, charge_val))
                    except ValueError:
                        pass

    # 2. SAVE METRICS (HOMO, LUMO, Gap)
    file_exists = METRICS_CSV.exists()
    try:
        with open(METRICS_CSV, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Filename", "Charge", "Multiplicity", "HOMO_eV", "LUMO_eV", "Gap_eV"])
            
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                pdb_name, charge, mult,
                homo_val if homo_val is not None else "NaN",
                lumo_val if lumo_val is not None else "NaN",
                gap_val if gap_val is not None else "NaN"
            ])
        print(f"    >> Metrics saved to: {METRICS_CSV.name}")
    except Exception as e:
        print(f"    Error saving metrics CSV: {e}")

    # 3. SAVE CHARGES (Detailed breakdown)
    if mulliken_charges:
        file_exists = CHARGES_CSV.exists()
        try:
            with open(CHARGES_CSV, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "Filename", "Atom_Index_Element", "Mulliken_Charge"])
                
                for atom_label, q_val in mulliken_charges:
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        pdb_name, atom_label, q_val
                    ])
            print(f"    >> Atomic charges saved to: {CHARGES_CSV.name}")
        except Exception as e:
            print(f"    Error saving charges CSV: {e}")
    else:
        print("    >> Warning: No Mulliken charges found in output.")

def main_workflow():
    if not ORCA_EXE.exists():
        print(f"    CRITICAL: ORCA not found at {ORCA_EXE}")
        return
    
    clean_pdb_path = PDB_FILE.strip("'").strip('"')
    
    if not clean_pdb_path or not Path(clean_pdb_path).exists():
        print(f"    CRITICAL: PDB file not found at {clean_pdb_path}")
        return

    print(f"    Reading PDB: {clean_pdb_path}")
    atoms = get_pocket_atoms(clean_pdb_path, POCKET_RADIUS)
    
    if not atoms: 
        print("    No atoms found. Check PDB format.")
        return
        
    print(f"    Extracted {len(atoms)} atoms (Dynamic Ligand Shape + {POCKET_RADIUS}A Buffer).")

    # ================= SMART STRATEGY =================
    successful_run = False
    final_charge = 0
    final_mult = 1

    if run_orca_attempt(0, 1, atoms):
        successful_run = True
        final_charge, final_mult = 0, 1
    elif run_orca_attempt(1, 1, atoms):
        successful_run = True
        final_charge, final_mult = 1, 1
    elif run_orca_attempt(-1, 1, atoms):
        successful_run = True
        final_charge, final_mult = -1, 1
    elif run_orca_attempt(0, 2, atoms):
        successful_run = True
        final_charge, final_mult = 0, 2

    if successful_run:
        print("\n    --- Parsing Results ---")
        out_path = WORKING_DIR / OUTPUT_NAME
        extract_and_save_data(out_path, os.path.basename(clean_pdb_path), final_charge, final_mult)
        print("    Done.")
    else:
        print("\n    ALL ATTEMPTS FAILED.")
        print("    Please check your PDB file or try manually capping residues.")

if __name__ == "__main__":
    main_workflow()