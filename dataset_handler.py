import argparse
import json
import logging
import os
import random
import time

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import get_dataset_config_names, get_dataset_split_names
from tqdm import tqdm
from utils import gen_question_id, get_sample_indices, load_data, save_results

# Set up logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
log_filename = os.getenv('LOG_FILENAME', 'logfile.log')
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_filename, filemode='w')

logger = logging.getLogger(__name__)

class DatasetHandler(ABC):
    def __init__(self, dataset_name, save_suffix_name, sys_config=None):
        self.dataset_name = dataset_name
        self.save_suffix_name = save_suffix_name
        
        # Set default configuration if None provided
        if sys_config is None:
            sys_config = {}
            
        # Set max_threads with validation and defaults
        self.max_threads = sys_config.get('max_threads', os.cpu_count())
        if not isinstance(self.max_threads, int) or self.max_threads < 1:
            logger.warning(f"Invalid max_threads value: {self.max_threads}. Using system CPU count.")
            self.max_threads = os.cpu_count()
            
        # Set response timeout with validation and defaults
        self.response_timeout = sys_config.get('response_time_out', 300)  # 30 seconds default
        if not isinstance(self.response_timeout, (int, float)) or self.response_timeout <= 0:
            logger.warning(f"Invalid response_timeout value: {self.response_timeout}. Using default 30 seconds.")
            self.response_timeout = 30
        
        # Infer data source from dataset name
        if self.dataset_name.endswith('.csv'):
            self.data_source = 'csv'
        elif self.dataset_name.endswith('.json'):
            self.data_source = 'json'
        else:
            # Default to huggingface if no clear file extension is present
            self.data_source = 'huggingface'

    def process_subset(self, subject, split, nsamples=None):
        print(f"Processing Subject: {subject} | Split: {split}")
        
        dataset = self.get_dataset(subject, split)
        if dataset is None:
            logger.warning(f"Dataset for {subject} - {split} could not be loaded.")
            return None

        indices = get_sample_indices(dataset, nsamples)
        logger.debug(f"Sample indices for {subject}:{split}: {indices}")

        result = {}
        desc = f"{split}" if subject is None else f"{subject}:{split}"
        max_workers = min(self.max_threads, os.cpu_count()) 
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_single_question, dataset, id) for id in indices]
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False):
                try:
                    processed_data = future.result(timeout=self.response_timeout)  # User-specified timeout
                    if processed_data is not None:
                        qid, answer = processed_data
                        if subject not in result:
                            result[subject] = {}
                        if split not in result[subject]:
                            result[subject][split] = {}
                        result[subject][split][qid] = answer
                except Exception as e:
                    logger.error(f"Error processing a question in {subject}:{split}: {e}")
                    logger.exception(e)

        logger.info(f"Finished processing subset {subject}:{split}")
        return result

    def process_single_question(self, dataset, id):
        try:
            answer = self.apply_op(dataset[id])
            return id, answer
        except Exception as e:
            logger.error(f"Error processing row {id}: {e}")
            logger.exception(e)
            return None

    def save_results(self, result):
        save_results(result, self.dataset_name, self.save_suffix_name)

    def get_subjects(self):
        if self.data_source in ['csv', 'json']:
            return ['default']  # Single default subject for CSV and JSON
        subjects = get_dataset_config_names(self.dataset_name)
        if not subjects:
            logger.warning("No subjects found for the dataset.")
        return subjects

    def get_splits(self, subject):
        if self.data_source in ['csv', 'json']:
            return ['train']  # Single default split for CSV and JSON
        splits = get_dataset_split_names(self.dataset_name, subject)
        if not splits:
            logger.warning(f"No splits found for subject {subject}.")
        return splits

    def num_questions(self):
        count = 0
        subjects = self.get_subjects()
        for subject in subjects:
            splits = self.get_splits(subject)
            for split in splits:
                dataset = self.get_dataset(subject, split)
                count   = count + dataset.num_rows
        return count

    def get_dataset(self, subject, split):
        logger.info(f"Loading dataset for {subject} - {split}")
        try:
            if self.data_source == 'csv':
                return self._load_from_csv()
            elif self.data_source == 'json':
                return self._load_from_json()
            else:  # default to huggingface
                return self._load_from_huggingface(subject, split)
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return None

    def _load_from_huggingface(self, subject, split):
        """Load dataset from HuggingFace datasets."""
        try:
            dataset = load_data(self.dataset_name, subject, split)
            if dataset is None:
                logger.warning(f"Dataset for {subject} - {split} could not be loaded from HuggingFace.")
            return dataset
        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset: {str(e)}")
            return None

    def _load_from_csv(self):
        """Load dataset from CSV file."""
        try:
            # Use dataset_name directly as the file path
            file_path = self.dataset_name
            
            if not os.path.exists(file_path):
                logger.warning(f"CSV file not found: {file_path}")
                return None
                
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Convert DataFrame to HuggingFace Dataset
            dataset = Dataset.from_pandas(df)
            
            logger.info(f"Successfully loaded dataset from CSV: {file_path}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading CSV dataset: {str(e)}")
            return None

    def _load_from_json(self):
        """Load dataset from JSON file."""
        try:
            # Use dataset_name directly as the file path
            file_path = self.dataset_name
            
            if not os.path.exists(file_path):
                logger.warning(f"JSON file not found: {file_path}")
                return None
                
            # Read JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert JSON to HuggingFace Dataset
            # Assuming data is a list of dictionaries with consistent keys
            dataset = Dataset.from_list(data if isinstance(data, list) else [data])
            
            logger.info(f"Successfully loaded dataset from JSON: {file_path}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading JSON dataset: {str(e)}")
            return None

    def process_dataset(self, nsamples=None):
        start_time = time.time()
        logger.info("Starting dataset processing")

        subjects = self.get_subjects()
        if not subjects:
            logger.warning("No subjects found for the dataset. Exiting processing.")
            return
        
        results = {}
        for subject in subjects:
            splits = self.get_splits(subject)
            for split in splits:
                result = self.process_subset(subject, split, nsamples)
                if result is not None:
                    if subject not in results:
                        results[subject] = {}
                    if subject in result and split in result[subject]:
                        results[subject][split] = result[subject][split]
                else:
                    logger.warning(f"Failed to load dataset for {subject} - {split}")
            
        self.save_results(results)
        total_time = time.time() - start_time
        logger.info(f"Total dataset processing time: {total_time:.2f} seconds")

    @abstractmethod
    def apply_op(self, row):
        pass

    def run(self, nsamples=None):
        self.process_dataset(nsamples=nsamples)

    @staticmethod
    def has_all_required_keys(row, required_keys):
        if not isinstance(row, dict):
           return None

        missing_keys = required_keys - set(row.keys())
        if missing_keys:
            print(f"Missing required keys: {missing_keys}")
            return None
        return True
