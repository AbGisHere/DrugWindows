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
PDB_FILE = "" 
LIGAND_NAME = "UNL"            

# --- OPTIMIZATION: TIGHTER BUFFER ---
POCKET_RADIUS = 4.0            

# 3. CALCULATION SETTINGS
WORKING_DIR = BASE_DIR / "orca_calc_folder"
INPUT_NAME  = "ligand.inp"
OUTPUT_NAME = "ligand.out"

# 4. OUTPUT CSV FILES
METRICS_CSV = BASE_DIR / "orca_electronic_metrics.csv"
CHARGES_CSV = BASE_DIR / "orca_mulliken_charges.csv"
# ======================================================

def get_pocket_atoms(pdb_path, lig_name, radius):
    """Parses PDB and extracts the ligand + immediate environment."""
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", pdb_path)
    except Exception as e:
        print(f"Error reading PDB: {e}")
        return []

    ligand_atoms = []
    all_atoms = []

    for atom in structure.get_atoms():
        all_atoms.append(atom)
        if atom.get_parent().get_resname() == lig_name:
            ligand_atoms.append(atom)
    
    if not ligand_atoms:
        print(f"ERROR: Ligand '{lig_name}' not found in PDB.")
        return []

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
        if res.get_resname() != lig_name and res.get_resname() != "HOH":
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
    print(f"\n--- Attempting Calculation: Charge {charge}, Multiplicity {mult} ---")
    
    inp_path = WORKING_DIR / INPUT_NAME
    write_orca_input(inp_path, atoms, charge, mult)
    
    if not inp_path.exists():
        print(" >> ERROR: Input file creation failed.")
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
                print(f" >> SUCCESS! Calculation finished in {duration:.1f} seconds.")
                return True
            else:
                print(f" >> FAILED with code {result.returncode}")
                err_msg = result.stderr.lower()
                if "impossible" in err_msg and "electrons" in err_msg:
                    print(" >> DIAGNOSIS: Electron count mismatch (Odd vs Even). Switching state...")
                elif "mpiexec" in err_msg:
                    print(" >> CRITICAL ERROR: MPI Issue Detected.")
                else:
                    # Print last few lines of error log
                    print(" >> ERROR DETAILS (Tail):")
                    print(result.stderr[-500:]) 
                return False

        except Exception as e:
            print(f" >> EXECUTION ERROR: {e}")
            return False

def extract_and_save_data(output_filepath, pdb_name, charge, mult):
    """Parses ORCA output and saves HOMO/LUMO/Gap and Charges to CSV."""
    
    if not os.path.exists(output_filepath):
        print("Error: Output file not found for parsing.")
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
            # Look ahead for HOMO/LUMO transition
            for j in range(i + 4, min(i + 1000, len(lines))):
                match = orb_pattern.match(lines[j])
                if match:
                    occ = float(match.group(2))
                    # Detect transition from Occupied (NOT 0) to Virtual (0)
                    if occ == 0.0000:
                        prev_match = orb_pattern.match(lines[j-1])
                        if prev_match:
                            homo_val = float(prev_match.group(4)) # eV is group 4
                            lumo_val = float(match.group(4))      # eV is group 4
                            gap_val = round(lumo_val - homo_val, 4)
                        break
        
        # --- MULLIKEN CHARGES ---
        if "MULLIKEN ATOMIC CHARGES" in line:
            mulliken_charges = [] # Reset to capture the final geometry (prevents reading intermediate steps)
            extract_mulliken = True
            continue
        
        if extract_mulliken:
            if "Sum of atomic charges" in line:
                extract_mulliken = False
            elif ":" in line:
                # Format example: "   0 C :   -0.123412"
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
        print(f" >> Metrics saved to: {METRICS_CSV}")
    except Exception as e:
        print(f"Error saving metrics CSV: {e}")

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
            print(f" >> Atomic charges saved to: {CHARGES_CSV}")
        except Exception as e:
            print(f"Error saving charges CSV: {e}")
    else:
        print(" >> Warning: No Mulliken charges found in output.")

def main_workflow():
    if not ORCA_EXE.exists():
        print(f"CRITICAL: ORCA not found at {ORCA_EXE}")
        return
    
    # Clean string path if it came from Windows copy-paste
    clean_pdb_path = PDB_FILE.strip("'").strip('"')
    
    if not clean_pdb_path or not Path(clean_pdb_path).exists():
        print(f"CRITICAL: PDB file not found at {clean_pdb_path}")
        return

    # Clean previous run
    if WORKING_DIR.exists():
        try:
            shutil.rmtree(WORKING_DIR)
        except OSError:
            print("Warning: Could not fully clean working directory (files might be open).")
    
    WORKING_DIR.mkdir(exist_ok=True)

    print(f"Reading PDB: {clean_pdb_path}")
    atoms = get_pocket_atoms(clean_pdb_path, LIGAND_NAME, POCKET_RADIUS)
    
    if not atoms: 
        print("No atoms found. Check Ligand Name (UNL) and PDB format.")
        return
        
    print(f"Extracted {len(atoms)} atoms (Dynamic Ligand Shape + {POCKET_RADIUS}A Buffer).")

    # ================= SMART STRATEGY =================
    # Loop through valid chemical states until one works.
    
    successful_run = False
    final_charge = 0
    final_mult = 1

    # 1. Try Neutral Singlet (Standard State)
    if run_orca_attempt(0, 1, atoms):
        successful_run = True
        final_charge, final_mult = 0, 1
    
    # 2. Try Cationic Singlet (+1 Charge) if step 1 failed
    elif run_orca_attempt(1, 1, atoms):
        successful_run = True
        final_charge, final_mult = 1, 1
        
    # 3. Try Anionic Singlet (-1 Charge) if step 2 failed
    elif run_orca_attempt(-1, 1, atoms):
        successful_run = True
        final_charge, final_mult = -1, 1
        
    # 4. Try Neutral Doublet (Radical) if step 3 failed
    elif run_orca_attempt(0, 2, atoms):
        successful_run = True
        final_charge, final_mult = 0, 2

    if successful_run:
        print("\n--- Parsing Results ---")
        out_path = WORKING_DIR / OUTPUT_NAME
        extract_and_save_data(out_path, os.path.basename(clean_pdb_path), final_charge, final_mult)
        print("Done.")
    else:
        print("\nALL ATTEMPTS FAILED.")
        print("Please check your PDB file or try manually capping residues.")

if __name__ == "__main__":
    main_workflow()