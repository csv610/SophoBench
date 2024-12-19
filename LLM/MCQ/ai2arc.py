import os
import sys
import threading
import logging

from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()
initialize_logging(log_file="ai2arc_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class Ai2ArcDataset(DatasetHandler):
    """AI2's Reasoning Challenge (ARC) dataset handler.
    
    A multiple-choice question-answering dataset containing science exam questions 
    from grade 3 to grade 9. Split into Easy and Challenge partitions, where Challenge
    contains questions requiring more complex reasoning. Most questions have 4 answer 
    choices (<1% have 3 or 5 choices). Includes a 14.3M unstructured text passage KB.
    
    Dataset: https://huggingface.co/datasets/allenai/ai2_arc
    """
    HF_DATASET_NAME = "allenai/ai2_arc"
    SF_DATASET_NAME = "Ai2Arc"
    REQUIRED_DATA_KEYS = frozenset({"question", "choices", "answerKey"})

    @classmethod
    def is_multimodal(cls):
        return False

    def __init__(self, task, models, sys_config=None):
        logging.info("Ai2Arc dataset initializing")
        self.task   = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('text', "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)
  
    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        question = row.get('question', "")
        choices = row.get('choices', None)

        if choices is None:
            logging.warning("Missing choices in data")  
            return None
                
        options = choices.get('text', [])
                
        return {
            'question': question,
            'options': options,
            'images': None
        }

    def get_correct_answer(self, row):
        return row.get('answerKey', "")

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task

if __name__ == "__main__":
    run_dataset(Ai2ArcDataset)
