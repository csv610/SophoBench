import os
import sys
import threading
import logging

from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()  
initialize_logging(log_file="medqna_version3_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

class MedQnAV3Dataset(DatasetHandler):
    """MedQnA-Version3 dataset handler.
    
    Dataset: https://huggingface.co/datasets/joseagmz/MedQnA_version3
    """
    DATASET_NAME = "joseagmz/MedQnA_version3"

    REQUIRED_DATA_KEYS = frozenset({'question', 'answer'})
    
    def __init__(self, task, models, sys_config=None):
        logging.info("MedQnA-Version3 dataset initializing")
        self.task   = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('text', '')
        super().__init__(self.DATASET_NAME, output_suffix, sys_config)

    @classmethod
    def is_multimodal(cls):
        return False

    def apply_op(self, row):
        model_input = self.extract_data(row)
        if model_input is None:
            error_msg = f"GPQA: Failed to extract data from row: {row.get('Pre-Revision Question', '[No question found]')[:100]}..."
            logging.error(error_msg)
            return error_msg

        if self.task == Tasks.TASK_GENERATE_ANSWER:
            return self.generate_answer(model_input)
        
        if self.task == Tasks.TASK_SAVE_QUESTION:
            return model_input
        return "Invalid task"

    def generate_answer(self, model_input):
        model = self.get_model()
        if model is None:
            error_msg = "GPQA: Model initialization failed - invalid model selection or configuration"
            logging.error(error_msg)
            return error_msg
            
        try:
            response = model.get_response(model_input)
            return response
        except Exception as e:
            error_msg = f"GPQA: Error getting model response: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def extract_data(self, row):
        question = row.get('question', '')
        if not question:
            logging.warning("GPQA: Empty question in row")
            return None
            
        return {
            'question': question,
            'options': None,
            'images': None
        }

    def get_correct_answer(self, row):
        answer = row.get("answer", "")
        return answer

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

if __name__ == "__main__":
    run_dataset(MedQnAV3Dataset)
