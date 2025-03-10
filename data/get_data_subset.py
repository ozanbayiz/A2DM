import os
import random
import shutil
from pathlib import Path
import argparse

def create_data_subset(input_dir: str, output_dir: str, num_files: int = 64):
    """
    Creates a subset of files from input_dir and copies them to output_dir.
    
    Args:
        input_dir: Path to directory containing source files
        output_dir: Path to directory where subset will be copied
        num_files: Number of files to include in subset (default 64)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of all .npz files in input directory
    all_files = list(input_path.glob("*.npz"))
    
    if len(all_files) < num_files:
        print(f"Warning: Only {len(all_files)} files found in input directory")
        selected_files = all_files
    else:
        # Randomly select num_files files
        selected_files = random.sample(all_files, num_files)
    
    # Copy selected files to output directory
    for file in selected_files:
        shutil.copy2(file, output_path / file.name)
        
    print(f"Copied {len(selected_files)} files to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create subset of data files")
    parser.add_argument("input_dir", type=str, help="Input directory containing source files")
    parser.add_argument("output_dir", type=str, help="Output directory for subset")
    parser.add_argument("--num_files", type=int, default=64, help="Number of files in subset")
    args = parser.parse_args()
    
    create_data_subset(args.input_dir, args.output_dir, args.num_files)

if __name__ == "__main__":
    main()
