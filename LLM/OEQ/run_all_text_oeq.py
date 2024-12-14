#!/usr/bin/env python3

import subprocess
import sys
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('query_all_text_oeq.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

SCRIPTS = [
    "gpqa.py",
    "gsm8k.py",
    "gsmplus.py",
    "imo_geometry.py",
    "math.py",
    "medical_meadow_flashcards.py",
    "medical_meadow_wikidoc_patient.py",
    "medicalquestions.py",
    "medicationqa.py",
    "metamathqa40k.py",
    "metamathqa.py",
    "scibench.py",
    "truthfulqa.py"
]

def run_script(script: str, sample_size: int = None) -> bool:
    """
    Run a single Python script and handle its execution.
    
    Args:
        script: Name of the script to run
        sample_size: Optional sample size (-n parameter)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cmd = [sys.executable, script]
        if sample_size:
            cmd.extend(['-n', str(sample_size)])
            
        logging.info(f"Starting {script}")
        start_time = time.time()
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed_time = time.time() - start_time
        logging.info(f"✓ Completed {script} in {elapsed_time:.2f} seconds")
        
        # Log any output from the script
        if process.stdout:
            logging.debug(f"Output from {script}:\n{process.stdout}")
            
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ Failed to run {script}")
        logging.error(f"Error output:\n{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"✗ Unexpected error running {script}: {str(e)}")
        return False

def run_all_scripts(parallel: bool = False, sample_size: int = None):
    """
    Run all Text OEQ scripts either sequentially or in parallel.
    
    Args:
        parallel: If True, runs scripts in parallel
        sample_size: Optional sample size for each script (-n parameter)
    """
    start_time = time.time()
    total_scripts = len(SCRIPTS)
    successful = 0
    failed = 0

    logging.info(f"Starting to process {total_scripts} text OEQ datasets...")
    
    if parallel:
        # Run scripts in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            future_to_script = {
                executor.submit(run_script, script, sample_size): script 
                for script in SCRIPTS
            }
            
            for future in as_completed(future_to_script):
                script = future_to_script[future]
                if future.result():
                    successful += 1
                else:
                    failed += 1
    else:
        # Run scripts sequentially
        for i, script in enumerate(SCRIPTS, 1):
            logging.info(f"[{i}/{total_scripts}] Processing dataset...")
            if run_script(script, sample_size):
                successful += 1
            else:
                failed += 1

    elapsed_time = time.time() - start_time
    logging.info(f"\nExecution Summary:")
    logging.info(f"Total time: {elapsed_time:.2f} seconds")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")

def main():
    parser = argparse.ArgumentParser(description='Run all text OEQ query scripts')
    parser.add_argument('--parallel', action='store_true', 
                      help='Run scripts in parallel')
    parser.add_argument('-n', '--sample-size', type=int,
                      help='Sample size for each dataset')
    args = parser.parse_args()

    logging.info(f"Starting text OEQ queries at {datetime.now()}")
    run_all_scripts(parallel=args.parallel, sample_size=args.sample_size)
    logging.info(f"Completed all text OEQ queries at {datetime.now()}")

if __name__ == "__main__":
    main()
