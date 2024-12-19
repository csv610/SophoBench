import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(log_file="ai2d_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class AI2DDataset(DatasetHandler):
    """AI2D (AI2 Diagrams) Dataset handler.
    
    A comprehensive dataset for diagram question answering, containing science
    diagrams with associated questions and multiple-choice answers. Tests models'
    ability to understand and reason about scientific diagrams and their labels.
    
    Dataset: https://huggingface.co/datasets/lmms-lab/ai2d
    """
    HF_DATASET_NAME = "lmms-lab/ai2d"
    SF_DATASET_NAME = "ai2d"
    REQUIRED_DATA_KEYS = frozenset({"question", "options", "image", "answer"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task, models, sys_config=None):
        logging.info("AI2D Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self.row)

    def extract_data(self, row):
        question = row.get('question')
        options  = row.get('options', [])
        images   = [row.get('image')]

        return {
            'question': question,
            'options': options,
            'images': images
        }

    def get_correct_answer(self, row):
        label = chr(65 + row.get('answer', 0))
        return label
    
    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(AI2DDataset)
