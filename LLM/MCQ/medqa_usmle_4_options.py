import os
import sys
import threading
import logging

# Import utility functions from local global_setting.py
from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()

initialize_logging(log_file="medqa_usmle4opt_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

#URL: https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options

class MedQAUSMLE4OptDataset(DatasetHandler):
    DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
    REQUIRED_DATA_KEYS = frozenset({"question", "options", "answer_idx"})

    def __init__(self, task, models, sys_config=None):
        logging.info("MedQA-USMLE-4-options dataset initializing")
        self.task    = task
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
            return "Failed to extract data from the row"

        if self.task == self.TASK_GENERATE_ANSWER:
            return self.generate_answer(model_input)
        
        if self.task == self.TASK_SAVE_QUESTION:
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
        options  = row.get('options', [])
            
        return {
            'question': question,
            'options': options,
            'images': None
        }
        
    def get_correct_answer(self, row):
        answer_idx = row.get('answer_idx', '')
        return 'NA' if answer_idx == '' else answer_idx

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

if __name__ == "__main__":
    run_dataset(MedQAUSMLE4OptDataset)
