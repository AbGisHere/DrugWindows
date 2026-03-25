import os
import sys
import subprocess
import shutil
import time
import re
import csv
import platform
import hashlib
import importlib.util
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from Bio import PDB
from Bio.PDB.NeighborSearch import NeighborSearch

# ================= HARDWARE DETECTION =================

def get_available_ram_mb() -> int:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // 1024
        elif system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True)
            if out.returncode == 0:
                return int(out.stdout.strip()) // (1024 * 1024)
        elif system == "Windows":
            out = subprocess.run(
                ["wmic", "OS", "get", "FreePhysicalMemory", "/Value"],
                capture_output=True, text=True)
            if out.returncode == 0:
                m = re.search(r"FreePhysicalMemory=(\d+)", out.stdout)
                if m:
                    return int(m.group(1)) // 1024
    except Exception:
        pass
    return 16 * 1024


def detect_gpus() -> list:
    """Returns [(gpu_id, free_vram_mb), ...] or [] if unavailable."""
    if platform.system() == "Darwin":
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        if out.returncode == 0:
            gpus = []
            for line in out.stdout.strip().splitlines():
                parts = line.strip().split(",")
                if len(parts) == 2:
                    gpus.append((int(parts[0].strip()), int(parts[1].strip())))
            return gpus
    except Exception:
        pass
    return []


def get_hardware_specs() -> dict:
    gpus = detect_gpus()
    return {
        "os": platform.system(),
        "cpu_cores": (len(os.sched_getaffinity(0))
                      if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1),
        "gpus": gpus,
        "gpu_available": len(gpus) > 0,
        "ram_mb": get_available_ram_mb(),
        "quick_exe": (shutil.which("quick.cuda") or shutil.which("quick.cuda.MPI")
                      or shutil.which("quick")) if (len(gpus) > 0) else shutil.which("quick"),
        "quick_found": (shutil.which("quick.cuda") or shutil.which("quick.cuda.MPI")
                        or shutil.which("quick")) is not None,
        "pyscf_found": importlib.util.find_spec("gpu4pyscf") is not None,
        "xtb_found": shutil.which("xtb") is not None,
        "mpi_found": (shutil.which("mpiexec") or shutil.which("mpirun")) is not None,
    }


def compute_job_resources(specs: dict, n_jobs: int) -> tuple:
    """Returns (nprocs, max_core_mb) to assign to a single ORCA job."""
    usable_cores = max(1, specs["cpu_cores"] - 2)
    nprocs = max(1, usable_cores // max(1, n_jobs))
    usable_ram = specs["ram_mb"] * 0.70
    ram_per_job = usable_ram / max(1, n_jobs)
    max_core = max(256, int(ram_per_job // nprocs))
    return nprocs, max_core


# ================= PATHS & CONSTANTS =================

BASE_DIR    = Path(os.getcwd()).resolve()
ORCA_FOLDER = BASE_DIR / "orca_program"
if platform.system() == "Windows":
    ORCA_EXE = ORCA_FOLDER / "orca.exe"
else:
    _detected = shutil.which("orca")
    # Guard against Linux screen reader also named "orca" (/usr/bin/orca)
    if _detected and Path(_detected).parent == Path("/usr/bin"):
        _detected = None
    ORCA_EXE  = Path(_detected) if _detected else ORCA_FOLDER / "orca"

PDB_FILE = Path('p2rank_2.5.1') / '4LKS.pdb'
LIGAND_NAME = "UNL"
POCKET_RADIUS = 3.2
WORKING_DIR = BASE_DIR / "orca_temp"
INPUT_NAME = "ligand.inp"
OUTPUT_NAME = "ligand.out"
METRICS_CSV = WORKING_DIR / "dft_batch_results.csv"
JOB_TIMEOUT_S = 14400   # 4 hours max per DFT job
MAX_CPU_WORKERS = 8     # cap for CPU-only parallel runs

# ================= ATOMIC DATA =================

Z_MAP = {
    'H': 1,  'HE': 2,  'LI': 3,  'BE': 4,  'B': 5,  'C': 6,  'N': 7,  'O': 8,  'F': 9,  'NE': 10,
    'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18,
    'K': 19,  'CA': 20, 'FE': 26, 'CU': 29, 'ZN': 30, 'BR': 35, 'I': 53,
}

def get_total_electrons(atom_lines):
    total_z = 0
    for line in atom_lines:
        symbol = line.split()[0].upper()
        total_z += Z_MAP.get(symbol, 6)
    return total_z

# ================= CACHING =================

def get_atom_hash(atom_lines: list) -> str:
    return hashlib.md5("\n".join(sorted(atom_lines)).encode()).hexdigest()[:12]


def is_cached(csv_path: Path, atom_hash: str) -> bool:
    if not csv_path.exists():
        return False
    try:
        return atom_hash in csv_path.read_text(errors="ignore")
    except Exception:
        return False

# ================= POCKET EXTRACTION =================

def get_pocket_atoms(pdb_path, radius):
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", pdb_path)
    except Exception as e:
        print(f"    Error reading PDB: {e}")
        return []

    std_aas = {
        'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
        'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR',
    }
    ignore_res = {'HOH', 'WAT', 'H2O', 'NA', 'CL', 'MG', 'ZN', 'CA', 'K'}

    ligand_resnames = set()
    for res in structure.get_residues():
        resname = res.get_resname().strip()
        if resname not in std_aas and resname not in ignore_res:
            ligand_resnames.add(resname)

    if not ligand_resnames:
        ligand_resnames.add(LIGAND_NAME)
    detected_ligands = list(ligand_resnames)
    print(f"    -> Auto-detected Ligand Residue Code(s): {detected_ligands}")

    all_atoms = list(structure.get_atoms())
    ligand_atoms = [a for a in all_atoms
                    if a.get_parent().get_resname().strip() in detected_ligands]
    if not ligand_atoms:
        return []

    ns = NeighborSearch(all_atoms)
    nearby_residues = set()
    for latom in ligand_atoms:
        nearby_residues.update(ns.search(latom.get_coord(), radius, level='R'))

    def format_atom(atom):
        x, y, z = atom.get_coord()
        element = ''.join(i for i in (atom.element or atom.get_name() or 'C') if not i.isdigit()).strip() or 'C'
        return f"{element:<2} {x:12.6f} {y:12.6f} {z:12.6f}"

    formatted_lines = [format_atom(a) for a in ligand_atoms]
    for res in nearby_residues:
        resname = res.get_resname().strip()
        if resname not in detected_ligands and resname not in ignore_res:
            formatted_lines.extend(format_atom(a) for a in res.get_atoms())

    return formatted_lines

# ================= ORCA INPUT =================

def write_orca_input(filepath, atom_lines, charge, mult, nprocs, max_core,
                     use_mpi, use_gpu):
    with open(filepath, "w") as f:
        if use_gpu:
            f.write("! r2SCAN-3c RIJCOSX TightSCF SOSCF\n")
        else:
            f.write("! r2SCAN-3c SOSCF\n")
        f.write(f"%maxcore {max_core}\n")
        if use_mpi and nprocs > 1:
            f.write(f"%pal nprocs {nprocs} end\n")
        else:
            print("   [Info] Running Serial (1 Core).")
        f.write(f"\n* xyz {charge} {mult}\n")
        for line in atom_lines:
            f.write(line + "\n")
        f.write("*\n")

# ================= ORCA EXECUTION =================

def run_orca_attempt(charge, mult, atoms, orca_exe, work_dir, nprocs,
                     max_core, use_mpi, use_gpu, env):
    print(f"    [ORCA] Attempting: Charge {charge}, Multiplicity {mult}")
    inp_path = work_dir / INPUT_NAME
    out_path = work_dir / OUTPUT_NAME

    write_orca_input(inp_path, atoms, charge, mult, nprocs, max_core,
                     use_mpi, use_gpu)
    start_time = time.time()

    try:
        with open(out_path, "w") as outfile:
            subprocess.run([str(orca_exe), INPUT_NAME], cwd=str(work_dir),
                           stdout=outfile, stderr=subprocess.PIPE,
                           text=True, env=env, timeout=JOB_TIMEOUT_S)
        duration = time.time() - start_time
        content = out_path.read_text(errors="ignore") if out_path.exists() else ""

        if "ORCA TERMINATED NORMALLY" in content:
            print(f"    >> SUCCESS in {duration:.1f}s.")
            return True

        # MPI failure → single fallback attempt (no recursion, no sentinel file)
        mpi_error_keywords = ("Startup", "MPI_Init", "mpirun", "aborting")
        if use_mpi and any(kw in content for kw in mpi_error_keywords):
            print("    [Fallback] MPI failure detected. Dropping to serial mode...")
            write_orca_input(inp_path, atoms, charge, mult, 1,
                             max_core * max(nprocs, 1), False, use_gpu)
            with open(out_path, "w") as outfile:
                subprocess.run([str(orca_exe), INPUT_NAME], cwd=str(work_dir),
                               stdout=outfile, stderr=subprocess.PIPE,
                               text=True, env=env, timeout=JOB_TIMEOUT_S)
            content = out_path.read_text(errors="ignore") if out_path.exists() else ""
            if "ORCA TERMINATED NORMALLY" in content:
                print("    >> SUCCESS (serial fallback).")
                return True
    except subprocess.TimeoutExpired:
        print(f"    [ORCA] Timed out after {JOB_TIMEOUT_S}s.")
    except Exception:
        pass

    return False


def run_quick_attempt(charge, mult, atoms, work_dir, quick_exe="quick"):
    inp_path = work_dir / "quick.inp"
    with open(inp_path, "w") as f:
        f.write(f"PBE0 BASIS=DEF2-SVP CHARGE={charge} MULT={mult} ENERGY\n\n")
        for line in atoms:
            f.write(line + "\n")
    out_path = work_dir / OUTPUT_NAME
    quick_out = work_dir / "quick.out"
    try:
        result = subprocess.run([quick_exe, "quick.inp"], cwd=str(work_dir),
                                capture_output=True, text=True, timeout=JOB_TIMEOUT_S)
        # QUICK writes to quick.out, not stdout — copy it to ligand.out for parsing
        if quick_out.exists():
            out_path.write_text(quick_out.read_text(errors="ignore"))
        content = out_path.read_text(errors="ignore") if out_path.exists() else ""
        return result.returncode == 0 and "THANK YOU FOR USING QUICK!" in content
    except subprocess.TimeoutExpired:
        print(f"    [QUICK] Timed out after {JOB_TIMEOUT_S}s.")
        return False
    except Exception:
        return False

# ================= XTB (GFN2-xTB) =================

def run_xtb_attempt(charge, mult, atoms, work_dir, nprocs):
    """Run GFN2-xTB on CPU. Returns (homo_ev, lumo_ev, gap_ev) or None."""
    xyz_path = work_dir / "xtb_input.xyz"
    out_path = work_dir / "xtb.out"

    with open(xyz_path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"charge={charge} mult={mult}\n")
        for line in atoms:
            f.write(line + "\n")

    uhf = mult - 1
    cmd = ["xtb", str(xyz_path), "--gfn", "2",
           "--chrg", str(charge), "--uhf", str(uhf),
           "--parallel", str(nprocs), "--norestart"]
    try:
        with open(out_path, "w") as outfile:
            subprocess.run(cmd, cwd=str(work_dir),
                           stdout=outfile, stderr=subprocess.STDOUT,
                           timeout=JOB_TIMEOUT_S)

        content = out_path.read_text(errors="ignore") if out_path.exists() else ""
        if "normal termination of xtb" not in content:
            print(f"    [XTB] Did not terminate normally.")
            return None

        # Parse orbital eigenvalue table for HOMO/LUMO
        homo_ev = lumo_ev = None
        last_occ_ev = None
        in_table = False
        for line in content.splitlines():
            if "Occupation" in line and "Energy/eV" in line:
                in_table = True
                continue
            if in_table:
                m = re.match(r'\s+\d+\s+([\d.]+)\s+[-\d.]+\s+([-\d.]+)', line)
                if m:
                    occ = float(m.group(1))
                    ev  = float(m.group(2))
                    if occ > 0:
                        last_occ_ev = ev
                    elif last_occ_ev is not None:
                        homo_ev = last_occ_ev
                        lumo_ev = ev
                        break

        if homo_ev is None or lumo_ev is None:
            print(f"    [XTB] Could not parse HOMO/LUMO from output.")
            return None

        gap_ev = round(lumo_ev - homo_ev, 4)
        return round(homo_ev, 4), round(lumo_ev, 4), gap_ev

    except subprocess.TimeoutExpired:
        print(f"    [XTB] Timed out after {JOB_TIMEOUT_S}s.")
        return None
    except Exception as e:
        print(f"    [XTB] Error: {e}")
        return None


# ================= PYSCF (GPU4PySCF) =================

HARTREE_TO_EV = 27.2114

def run_pyscf_attempt(charge, mult, atoms, work_dir, gpu_id):
    """Run PBE0/def2-SVP via GPU4PySCF with SOSCF. Returns (homo_ev, lumo_ev, gap_ev) or None."""
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        import numpy as np
        from pyscf import gto
        from gpu4pyscf import dft as gpu_dft

        mol = gto.Mole()
        mol.atom = "\n".join(atoms)
        mol.basis = "def2-svp"
        mol.charge = charge
        mol.spin = mult - 1
        mol.verbose = 3
        mol.output = str(work_dir / "pyscf.log")
        mol.build()

        mf = gpu_dft.RKS(mol) if mult == 1 else gpu_dft.UKS(mol)
        mf.xc = "pbe0"
        mf.conv_tol = 1e-8
        mf.max_cycle = 200
        mf = mf.newton()  # SOSCF — handles near-degenerate systems
        mf.kernel()

        if not mf.converged:
            print(f"    [PySCF] SCF did not converge for charge={charge}.")
            return None

        if mult == 1:
            mo_e = np.asarray(mf.mo_energy)
            mo_o = np.asarray(mf.mo_occ)
            homo_ev = float(mo_e[mo_o > 0][-1]) * HARTREE_TO_EV
            lumo_ev = float(mo_e[mo_o == 0][0]) * HARTREE_TO_EV
        else:
            mo_ea, mo_oa = np.asarray(mf.mo_energy[0]), np.asarray(mf.mo_occ[0])
            mo_eb, mo_ob = np.asarray(mf.mo_energy[1]), np.asarray(mf.mo_occ[1])
            homo_ev = max(float(mo_ea[mo_oa > 0][-1]), float(mo_eb[mo_ob > 0][-1])) * HARTREE_TO_EV
            lumo_a  = float(mo_ea[mo_oa == 0][0]) if (mo_oa == 0).any() else float("inf")
            lumo_b  = float(mo_eb[mo_ob == 0][0]) if (mo_ob == 0).any() else float("inf")
            lumo_ev = min(lumo_a, lumo_b) * HARTREE_TO_EV

        gap_ev = round(lumo_ev - homo_ev, 4)
        return round(homo_ev, 4), round(lumo_ev, 4), gap_ev

    except Exception as e:
        print(f"    [PySCF] Error: {e}")
        return None


# ================= RESULT EXTRACTION =================

def extract_result_dict(output_filepath, pdb_name, charge, mult, atom_hash):
    """Parse ORCA/QUICK output. Returns dict or None."""
    out = Path(output_filepath)
    if not out.exists():
        return None
    content = out.read_text(errors="ignore")

    homo_val = lumo_val = gap_val = None
    if "HOMO ENERGY:" in content:
        # QUICK output: HOMO/LUMO reported directly in eV
        homo_match = re.search(r"HOMO ENERGY:\s+[-\d.]+\s+A\.U\.,\s+([-\d.]+)\s+EV", content)
        lumo_match = re.search(r"LUMO ENERGY:\s+[-\d.]+\s+A\.U\.,\s+([-\d.]+)\s+EV", content)
        if homo_match and lumo_match:
            homo_val = float(homo_match.group(1))
            lumo_val = float(lumo_match.group(1))
            gap_val  = round(lumo_val - homo_val, 4)
    elif "ORBITAL ENERGIES" in content:
        # ORCA output: scan occupation table for HOMO/LUMO boundary
        orb_match = re.findall(
            r"(\d+)\s+([0-9\.]+)\s+([-0-9\.]+)\s+([-0-9\.]+)", content)
        for i, match in enumerate(orb_match):
            try:
                if float(match[1]) == 0.0 and i > 0:
                    homo_val = float(orb_match[i - 1][3])
                    lumo_val = float(match[3])
                    gap_val  = round(lumo_val - homo_val, 4)
                    break
            except Exception:
                continue

    engine = "QUICK" if "HOMO ENERGY:" in content else "ORCA"
    return {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename":  pdb_name,
        "Engine":    engine,
        "Charge":    charge,
        "Mult":      mult,
        "HOMO_eV":   homo_val if homo_val is not None else "NaN",
        "LUMO_eV":   lumo_val if lumo_val is not None else "NaN",
        "Gap_eV":    gap_val  if gap_val  is not None else "NaN",
        "AtomHash":  atom_hash,
    }


def write_results_to_csv(csv_path: Path, result_rows: list) -> None:
    if not result_rows:
        return
    fieldnames = ["Timestamp", "Filename", "Engine", "Charge", "Mult",
                  "HOMO_eV", "LUMO_eV", "Gap_eV", "AtomHash"]
    file_exists = csv_path.exists()
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        for row in result_rows:
            writer.writerow(row)

# ================= WORKER (picklable for ProcessPoolExecutor) =================

def process_single_pdb(job_args: tuple):
    """Runs one PDB through DFT. Returns result dict or None."""
    (pdb_path_str, gpu_id, nprocs, max_core, use_mpi,
     specs, orca_exe_str, orca_folder_str, base_dir_str) = job_args

    pdb_path   = Path(pdb_path_str)
    orca_exe   = Path(orca_exe_str)
    orca_folder = Path(orca_folder_str)
    base_dir   = Path(base_dir_str)
    metrics_csv = base_dir / "orca_temp" / "dft_batch_results.csv"

    if not pdb_path.exists():
        print(f"    [Worker] PDB not found: {pdb_path}")
        return None

    work_dir = base_dir / "orca_temp" / pdb_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    atoms = get_pocket_atoms(pdb_path, POCKET_RADIUS)
    if not atoms:
        print(f"    [Worker] No pocket atoms found in {pdb_path.name}")
        return None

    atom_hash = get_atom_hash(atoms)
    if is_cached(metrics_csv, atom_hash):
        print(f"    [Worker] Cache hit for {pdb_path.name} (hash {atom_hash})")
        return None

    env = os.environ.copy()
    if str(orca_folder) not in env.get("PATH", ""):
        env["PATH"] = str(orca_folder) + os.pathsep + env.get("PATH", "")
    if gpu_id >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # RIJCOSX only on Linux/Windows with a real CUDA GPU
    use_gpu = (gpu_id >= 0
               and specs.get("gpu_available", False)
               and specs.get("os") != "Darwin")

    total_z = get_total_electrons(atoms)
    print(f"    -> Total Nuclei Charge (Electrons): {total_z}")

    candidates = []
    for charge in [0, 1, -1]:
        electrons = total_z - charge
        mult = 1 if (electrons % 2 == 0) else 2
        candidates.append((charge, mult))

    use_xtb   = specs.get("xtb_found", False)
    use_quick = (not use_xtb
                 and specs.get("gpu_available", False)
                 and specs.get("quick_found", False))

    for c, m in candidates:
        result_tuple = None

        if use_xtb:
            print(f"    [XTB] Attempting: Charge {c}, Multiplicity {m}")
            result_tuple = run_xtb_attempt(c, m, atoms, work_dir, nprocs)
            if result_tuple is None:
                print(f"    [Fallback] XTB failed for charge={c}. Trying ORCA...")
                success = run_orca_attempt(c, m, atoms, orca_exe, work_dir,
                                           nprocs, max_core, use_mpi, use_gpu, env)
                if success:
                    return extract_result_dict(work_dir / OUTPUT_NAME,
                                               pdb_path.name, c, m, atom_hash)
                continue

        elif use_quick:
            quick_exe = specs.get("quick_exe") or "quick"
            success = run_quick_attempt(c, m, atoms, work_dir, quick_exe)
            if not success:
                print(f"    [Fallback] QUICK failed for charge={c}. Trying ORCA...")
                success = run_orca_attempt(c, m, atoms, orca_exe, work_dir,
                                           nprocs, max_core, use_mpi, use_gpu, env)
            if success:
                return extract_result_dict(work_dir / OUTPUT_NAME,
                                           pdb_path.name, c, m, atom_hash)
            continue

        else:
            success = run_orca_attempt(c, m, atoms, orca_exe, work_dir,
                                       nprocs, max_core, use_mpi, use_gpu, env)
            if success:
                return extract_result_dict(work_dir / OUTPUT_NAME,
                                           pdb_path.name, c, m, atom_hash)
            continue

        if result_tuple is not None:
            homo_ev, lumo_ev, gap_ev = result_tuple
            engine = "XTB" if use_xtb else "PySCF"
            print(f"    >> {engine} SUCCESS: HOMO={homo_ev:.3f} eV, LUMO={lumo_ev:.3f} eV, Gap={gap_ev:.3f} eV")
            return {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Filename":  pdb_path.name,
                "Engine":    engine,
                "Charge":    c,
                "Mult":      m,
                "HOMO_eV":   homo_ev,
                "LUMO_eV":   lumo_ev,
                "Gap_eV":    gap_ev,
                "AtomHash":  atom_hash,
            }

    print(f"    All attempts failed for {pdb_path.name}. Check {work_dir / OUTPUT_NAME}")
    return None

# ================= MAIN WORKFLOW =================

def main_workflow(pdb_files=None):
    WORKING_DIR.mkdir(exist_ok=True)

    specs = get_hardware_specs()
    print(f"    System: {specs['os']} | CPU: {specs['cpu_cores']} "
          f"| RAM: {specs['ram_mb']} MB | GPU: {specs['gpu_available']}")
    if specs['gpus']:
        print(f"    GPUs: {specs['gpus']}")

    # Determine job list
    if pdb_files is None:
        # batch.py mode: single job from the module-level PDB_FILE
        job_paths = [Path(PDB_FILE)]
    else:
        job_paths = [Path(p) for p in pdb_files]

    job_paths = [p for p in job_paths if p.exists()]
    if not job_paths:
        print("    No valid PDB files to process.")
        return

    n_jobs = len(job_paths)
    nprocs, max_core = compute_job_resources(specs, n_jobs)
    # macOS ORCA binaries are serial-only; system mpirun version rarely matches
    use_mpi = specs["mpi_found"] and specs["os"] != "Darwin"

    gpus = specs["gpus"]
    if gpus:
        n_workers = len(gpus)
        gpu_ids = [gpus[i % len(gpus)][0] for i in range(n_jobs)]
    else:
        n_workers = min(max(1, specs["cpu_cores"] // 4), MAX_CPU_WORKERS)
        gpu_ids = [-1] * n_jobs

    job_args = [
        (str(job_paths[i]), gpu_ids[i], nprocs, max_core, use_mpi,
         specs, str(ORCA_EXE), str(ORCA_FOLDER), str(BASE_DIR))
        for i in range(n_jobs)
    ]

    if n_jobs == 1:
        results = [process_single_pdb(job_args[0])]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(process_single_pdb, job_args))

    result_rows = [r for r in results if r is not None]
    if result_rows:
        write_results_to_csv(METRICS_CSV, result_rows)
        print(f"\n    Results written to {METRICS_CSV}")
    else:
        print("\n    No results to write.")


if __name__ == "__main__":
    pdbs = [Path(a) for a in sys.argv[1:] if a.endswith(".pdb")]
    main_workflow(pdbs if pdbs else None)
