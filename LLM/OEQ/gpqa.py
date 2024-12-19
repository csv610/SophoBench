import os
import sys
import threading
import logging

from global_setting import initialize_logging, add_project_root_to_path

add_project_root_to_path()  
initialize_logging(log_file="gpqa_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class GPQADataset(DatasetHandler):
    """GPQA (General Physics Question Answering) Dataset handler.
    
    A comprehensive dataset focused on physics questions and answers. Contains
    pre-revision and post-revision versions of questions and their correct answers,
    designed to test understanding of physics concepts and problem-solving.
    
    Dataset: https://huggingface.co/datasets/Idavidrein/gpqa
    """
    HF_DATASET_NAME = "Idavidrein/gpqa"
    SF_DATASET_NAME = "GPQA"
    REQUIRED_DATA_KEYS = frozenset({'Pre-Revision Question', 'Pre-Revision Correct Answer'})

    @classmethod
    def is_multimodal(cls):
        return False
    
    def __init__(self, task, models, sys_config=None):
        logging.info("GPQA dataset initializing")
        self.task   = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('text', '')
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        question = row.get('Pre-Revision Question', '')
        if not question:
            logging.warning("GPQA: Empty question in row")
            return None
            
        return {
            'question': question,
            'options': None,
            'images': None
        }

    def get_correct_answer(self, row):
        answer = row.get("Pre-Revision Correct Answer", "")
        return answer

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(GPQADataset)
