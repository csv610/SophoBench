import random
import logging
import os
import json

from datasets import load_dataset

logger = logging.getLogger(__name__)

def load_data(dataset_name, subject, split):
    try:
        if subject:
            ds = load_dataset(dataset_name, subject, trust_remote_code=True)
        else:
            ds = load_dataset(dataset_name, trust_remote_code=True)
        return ds[split]
    except Exception as e:
        return None

def get_sample_indices(dataset, nsamples):
    """
    Get sample indices from the dataset.
    If nsamples is provided and less than the dataset length, select random samples.
    """
    indices = range(len(dataset))
    if nsamples is not None and nsamples < len(dataset):
        indices = random.sample(indices, nsamples)
    return sorted(indices)  # Return the sorted indices

def gen_question_id(subset: str, split: str, index: int) -> str:
    """
    Generate a question ID based on the subset, split, and index.
    Maps full split names to shortened versions for consistency.
    """
    # Map full split names to their shortened versions
    split_map = {
        "test": "tes",
        "train": "tra",
        "validation": "val"
    }

    # Get the shortened split value, default to the original if not found in the map
    short_split = split_map.get(split, split)

    # Construct the combined string based on the subset presence
    if subset:
        result = f"{subset}_{short_split}_{index}"
    else:
        result = f"{short_split}_{index}"

    return result

def save_results(data: dict, bench_name: str, model_name: str):
    """
    Save the benchmark results from a dictionary to a JSON file.

    Parameters:
    data (dict): Dictionary containing the results to save.
    bench_name (str): Name of the benchmark.
    model_name (str): Name of the model used.

    Returns:
    str: File path of the saved JSON file.
    """
    try:
        # Ensure the results directory exists
        os.makedirs('results', exist_ok=True)

        # Replace '/' with '_' in the benchmark name to avoid path issues
        bench_name = bench_name.replace('/', '_')

        # Construct the filename
        filename = os.path.join('results', f'{bench_name}_result_{model_name}.json')

        # Save the dictionary to a JSON file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

        # Log success message
        logger.info(f"Results successfully saved to {filename}")

        return filename
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise
