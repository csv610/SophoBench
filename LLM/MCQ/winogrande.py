import os
import sys
import threading
import logging

# Import utility functions from local global_setting.py
from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()

initialize_logging(log_file="winogrande_status.log")

import re
from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class WinoGrandeDataset(DatasetHandler):
    HF_DATASET_NAME  = "automated-research-group/winogrande"
    SF_DATASET_NAME  = "WinoGrande"
    REQUIRED_DATA_KEYS = frozenset({'request', 'response'})

    @classmethod
    def is_multimodal(cls):
        return False

    def __init__(self, task, models, sys_config=None):
        logging.info("WinoGrande dataset initializing")
        self.task   = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("text", "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        text = row.get("request", "")
        if not text:
            logging.warning("WinoGrande: Empty text in row")
            return None
            
        # Find the question part
        question_start = text.find('Given the text "')
        question_end = text.find('", choose')
        if question_start == -1 or question_end == -1:
            logging.warning("WinoGrande: Could not find question text")
            return None
        
        question = text[question_start + len('Given the text "'):question_end]
        
        # Extract options
        options = []
        for line in text.split('\n'):
            if ' - "' in line:
                # Extract option number and text
                num, opt_text = line.split(' - "')
                opt_text = opt_text.rstrip(' ."')  # Remove trailing ." and spaces
                options.append(opt_text)
        
        if not options:
            logging.warning("WinoGrande: No options found in text")
            return None
            
        return {
            'question': question,
            'options': options,
            'images': None
        }
        
    def get_correct_answer(self, row):
        return chr(65 + row["response"] )

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  

if __name__ == "__main__":
    run_dataset(WinoGrandeDataset)
