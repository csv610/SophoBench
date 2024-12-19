import os
import sys
import threading
import logging

import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(log_file="ai2d_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class NEJMDataset(DatasetHandler):
    """NEJM (New England Journal of Medicine) Dataset handler.
    
    A dataset containing medical case studies and questions from the New England Journal 
    of Medicine, including images and multiple-choice questions that test medical 
    knowledge and diagnostic reasoning.
    """
    DATASET_NAME = "./data/nejm.json"  # Local dataset
    SF_DATASET_NAME = "NEJM"
    REQUIRED_DATA_KEYS = frozenset({"Question", "Options", "Image", "Answer"})

    @classmethod
    def is_multimodal(cls):
        return True
    
    def __init__(self, task, models, sys_config=None):
        self.models = models
        self.task   = task
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        logging.info(f"NEJM: Initialized with output suffix: {output_suffix}")
        super().__init__(self.DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)
            
    def extract_data(self, row):
        question = row.get('Question', '')
        options = row.get('Options', [])
        image = row.get('Image', '')
        
        if not question:
            logging.warning("NEJM: Empty question in row")
            return None
            
        if not options:
            logging.warning("NEJM: No options provided for question")
            return None
            
        if not image:
            logging.warning("NEJM: No image provided for question")
            return None
            
        return {
            'question': question,
            'options': options,
            'images': [image]  # Wrap single image in list as expected by model
        }

    def get_correct_answer(self, row):
        return row.get("Answer", "")

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(NEJMDataset)
