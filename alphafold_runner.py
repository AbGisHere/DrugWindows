import os
import argparse
from pathlib import Path
from colabfold.download import download_alphafold_params
from colabfold.utils import setup_logging
from colabfold.batch import get_queries, run, set_model_type

def extract_sequence_from_fasta(fasta_path: str) -> str:
    with open(fasta_path, "r") as f:
        return "".join([line.strip() for line in f if not line.strip().startswith(">")])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--pdb_id", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    jobname = f"{args.pdb_id}_af2"
    result_dir = Path(args.output_dir) / jobname
    result_dir.mkdir(parents=True, exist_ok=True)
    
    query_sequence = extract_sequence_from_fasta(args.fasta)
    queries_path = result_dir / f"{jobname}.csv"

    with open(queries_path, "w") as text_file:
        text_file.write(f"id,sequence\n{jobname},{query_sequence}")

    setup_logging(result_dir / "log.txt")
    queries, is_complex = get_queries(queries_path)
    
    # Logic to set model type
    model_type = set_model_type(is_complex, "auto") 
    download_alphafold_params(model_type, Path("."))

    print(f"Starting AlphaFold prediction for {jobname}...")
    
    # FIXED: Added is_complex=is_complex to the arguments
    run(
        queries=queries,
        result_dir=str(result_dir),
        use_templates=False,
        num_relax=0,
        msa_mode="mmseqs2_uniref_env",
        model_type=model_type,
        num_models=5,
        num_recycles=None,
        is_complex=is_complex, 
        data_dir=Path("."),
        user_agent="colabfold/local-python-script"
    )

if __name__ == "__main__":
    main()