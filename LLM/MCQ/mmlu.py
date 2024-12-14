import os
import sys
import threading
import logging

# Import utility functions from local global_setting.py
from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()

initialize_logging(log_file="mmlu_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

class MMLUDataset(DatasetHandler):
    DATASET_NAME = "cais/mmlu"
    REQUIRED_DATA_KEYS = frozenset({'question', 'choices', 'answer'})
    
    def __init__(self, task, models, sys_config=None):
        logging.info("MMLU dataset initializing")
        self.task   = task
        self.models  = models
        self.local_thread = threading.local()
        output_suffix = models.get('text', '')
        super().__init__(self.DATASET_NAME, output_suffix, sys_config)

    @classmethod
    def is_multimodal(cls):
        return False

    def apply_op(self, row):
        model_input = self.extract_data(row)
        if model_input is None:
            return "Failed to extract data from question"
        
        if self.task == Tasks.TASK_GENERATE_ANSWER:
            return self.generate_answer(model_input)
        
        if self.task == Tasks.TASK_SAVE_QUESTION:
            return model_input  

        return "Invalid task"

    def generate_answer(self, model_input):
        model = self.get_model()
        if model is None:
            return "Invalid model selection or initialization failed"
            
        response = model.get_response(model_input)
        return response

    def extract_data(self, row):
        
        question = row.get('question', '')
        choices = row.get('choices', [])
                    
        return {
            'question': question,
            'options': choices,
            'images': None
        }
        
    def get_correct_answer(self, row):
        return row.get('answer', '')

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)  

if __name__ == "__main__":
    run_dataset(MMLUDataset)
