import os
import sys
import threading
import logging

from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()  
initialize_logging(log_file="gsm8k_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class GSM8KDataset(DatasetHandler):
    """GSM8K (Grade School Math 8K) Dataset handler.
    
    A collection of 8.5K high-quality linguistically diverse grade school math
    word problems. Each problem is provided with a detailed step-by-step solution,
    making it suitable for evaluating mathematical reasoning capabilities.
    
    Dataset: https://huggingface.co/datasets/openai/gsm8k
    """
    HF_DATASET_NAME = "openai/gsm8k"
    SF_DATASET_NAME = "GSM8K"
    REQUIRED_DATA_KEYS = frozenset({'question', 'answer'})

    @classmethod
    def is_multimodal(cls):
        return False
    
    def __init__(self, task, models, sys_config=None):     
        logging.info("GSM8K dataset initializing")
        self.task   = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('text', '')
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        question = row.get('question', '')
        if not question:
            logging.warning("GSM8K: Empty question in row")
            return None
            
        return {
            'question': question,
            'options': None,
            'images': None
        }

    def get_correct_answer(self, row):
        answer = row.get('answer', "")
        return answer

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(GSM8KDataset)
