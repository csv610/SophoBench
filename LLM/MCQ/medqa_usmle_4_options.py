import os
import sys
import threading
import logging

# Import utility functions from local global_setting.py
from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()

initialize_logging(log_file="medqa_usmle4opt_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class MedQAUSMLE4OptionsDataset(DatasetHandler):
    HF_DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
    SF_DATASET_NAME = "USMLE-4Opt"
    REQUIRED_DATA_KEYS = frozenset({"question", "options", "answer_idx"})

    @classmethod
    def is_multimodal(cls):
        return False

    def __init__(self, task, models, sys_config=None):
        logging.info("MedQA-USMLE-4-options dataset initializing")
        self.task    = task
        self.models  = models
        self.local_thread = threading.local()
        output_suffix = models.get('text', '')
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
            
        question = row.get('question', '')
        options  = row.get('options', [])
            
        return {
            'question': question,
            'options': options,
            'images': None
        }
        
    def get_correct_answer(self, row):
        answer_idx = row.get('answer_idx', '')
        return 'NA' if answer_idx == '' else answer_idx

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME
  
    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(MedQAUSMLE4OptDataset)
