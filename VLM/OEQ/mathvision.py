import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="mathvision_status.log")

from dataset_handler import DatasetHandler 
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

#URL: https://huggingface.co/datasets/MathLLMs/MathVision

class MathVisionDataset(DatasetHandler):
    """MathVision Dataset handler.
    
    A visual mathematics dataset for evaluating mathematical reasoning capabilities
    of vision-language models. Contains questions with mathematical diagrams, graphs,
    and visual problem-solving scenarios.
    
    Dataset: https://huggingface.co/datasets/MathLLMs/MathVision
    """
    HF_DATASET_NAME = "MathLLMs/MathVision"
    SF_DATASET_NAME = "MathVision"
    REQUIRED_DATA_KEYS = frozenset({"question", "options", "decoded_image", "answer"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task, models, sys_config=None):
        logging.info("MathVision Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)
        
    def extract_data(self, row):
        question = row.get('question', "")
    
        options = row.get('options', [])

        image  = row.get('decoded_image', None)
        
        return {
            "question": question,
            "options":  options,
            "images":   [image]
        }

    def get_correct_answer(self, row):
        ans = row.get('answer', "")  
        return ans

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(MathVisionDataset)
