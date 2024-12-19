import os
import sys
import threading
import logging

from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()  
initialize_logging(log_file="simpleqa_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class SimpleQADataset(DatasetHandler):
    """SimpleQA Dataset handler.
    
    A dataset containing simple questions and answers designed for testing
    language model capabilities on straightforward question answering tasks.
    """
    DATASET_NAME = "./datasimple_qa.json"  # Local dataset
    SF_DATATSET_NAME = "SimpleQA"
    REQUIRED_DATA_KEYS = frozenset({'problem'})
    
    def __init__(self, task, models, sys_config=None):
        self.task   = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('text', '')
        super().__init__(self.DATASET_NAME, output_suffix, sys_config)
        logging.info("SimpleQA dataset initialized")

    @classmethod
    def is_multimodal(cls):
        return False

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        question = row.get('problem', '')
        if not question:
            logging.warning("SimpleQA: Empty question in row")
            return None
            
        return {
            'question': question,
            'options': None,
            'images': None
        }

    def get_correct_answer(self, row):
        # For SimpleQA, we might not have ground truth answers
        # Return empty string or implement based on dataset structure
        return ""

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task

if __name__ == "__main__":
    run_dataset(SimpleQADataset)
