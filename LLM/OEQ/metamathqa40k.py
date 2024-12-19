import os
import sys
import threading
import logging

from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()  
initialize_logging(log_file="metamathqa40k_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class MetaMath40KQADataset(DatasetHandler):
    """MetaMathQA-40K Dataset handler.
    
    An expanded version of MetaMathQA containing 40,000 mathematical problems.
    Features a diverse range of mathematical questions with varying difficulty levels,
    designed to evaluate advanced mathematical reasoning capabilities.
    
    Dataset: https://huggingface.co/datasets/meta-math/MetaMathQA-40K
    """
    HF_DATASET_NAME = "meta-math/MetaMathQA-40K"
    SF_DATASET_NAME = "MetaMathQA-40K"
    REQUIRED_DATA_KEYS = frozenset({'query', 'response'})

    @classmethod
    def is_multimodal(cls):
        return False

    def __init__(self, task, models, sys_config=None):
        logging.info("MetaMathQA-40K dataset initializing")
        self.task   = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('text', '')
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        question = row.get('query', '')
        if not question:
            return None
            
        return {
            'question': question,
            'options': None,
            'images': None
        }

    def get_correct_answer(self, row):
        return row.get('response',"")

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(MetaMath40KQADataset)
