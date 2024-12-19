import os
import sys
import threading
import logging

from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()  
initialize_logging(log_file="mathqa_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class MathDataset(DatasetHandler):
    """MATH Dataset handler.
    
    A comprehensive mathematics dataset containing problems from various
    mathematical domains including algebra, counting and probability,
    geometry, intermediate algebra, number theory, precalculus, and more.
    
    Dataset: https://huggingface.co/datasets/lighteval/MATH
    """
    HF_DATASET_NAME = "lighteval/MATH"
    SF_DATASET_NAME = "MATH"
    REQUIRED_DATA_KEYS = frozenset({"problem", "solution"})

    @classmethod
    def is_multimodal(cls):
        return False

    def __init__(self, task, models, sys_config=None):
        logging.info("MATH dataset initializing")
        self.task   = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('text', '')
        super().__init__(self.HF_DATASET_NAMEDATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)
        
    def extract_data(self, row):
        question = row.get('problem', '')
        if not question:
            logging.warning("MATH: Empty problem in row")
            return None
            
        return {
            'question': question,
            'options': None,
            'images': None
        }

    def get_correct_answer(self, row):
        return row.get('solution', "")

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(MathDataset)
