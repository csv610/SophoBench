import os
import sys
import threading
import re
import logging


from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()
initialize_logging(log_file="bigbenchhard_status.log")

from dataset_handler import DatasetHandler
from model_query  import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class BigBenchHardDataset(DatasetHandler):
    HF_DATASET_NAME = "maveriq/bigbenchhard"
    SF_DATASET_NAME = "BigBenchHard"
    REQUIRED_DATA_KEYS = frozenset({"input", "target"})

    @classmethod
    def is_multimodal(cls):
        return False

    def __init__(self, task, models, sys_config=None):
        logging.info("BigBenchHard dataset initializing")
        self.task   = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('text', '')
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
            
        input_text = row.get('input',"")     
        if input_text == "":
            logging.error("Missing input text")
            return None
            
        # Process input text to extract question and options
        match = re.search(r'(.*?)\s*Options:\s*(.*)', input_text, re.DOTALL)
        if not match:
            logging.error("Could not parse question and options from input text")
            return None
            
        question = match.group(1).strip()

        options_text = match.group(2).strip()
        options = [opt.strip() for opt in options_text.split('\n') if opt.strip()]
        
        if not options:
            logging.error("No options found in input text")
            return None
            
        return {
            'question': question,
            'options': options,
            'images': None
        }

    def get_correct_answer(self, row):
        return row['target'].replace('(', '').replace(')', '') if isinstance(row['target'], str) else row['target']

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task

if __name__ == "__main__":
    run_dataset(BigBenchHardDataset)
