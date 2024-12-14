import os
import sys
import threading
import logging

from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()  
initialize_logging(log_file="gsm8k_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

class GSM8KDataset(DatasetHandler):
    """GSM8K (Grade School Math 8K) Dataset handler.
    
    A collection of 8.5K high-quality linguistically diverse grade school math
    word problems. Each problem is provided with a detailed step-by-step solution,
    making it suitable for evaluating mathematical reasoning capabilities.
    
    Dataset: https://huggingface.co/datasets/openai/gsm8k
    """
    DATASET_NAME = "openai/gsm8k"
    REQUIRED_DATA_KEYS = frozenset({'question', 'answer'})
    
    def __init__(self, task, models, sys_config=None):     
        logging.info("GSM8K dataset initializing")
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
            error_msg = f"GSM8K: Failed to extract data from row: {row.get('question', '[No question found]')[:100]}..."
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
            error_msg = "GSM8K: Model initialization failed - invalid model selection or configuration"
            logging.error(error_msg)
            return error_msg
            
        try:
            response = model.get_response(model_input)
            logging.debug(f"GSM8K: Successfully processed question: {model_input['question'][:100]}...")
            return response
        except Exception as e:
            error_msg = f"GSM8K: Error getting model response: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def extract_data(self, row):
        question = row.get('question', '')
        if not question:
            logging.warning("GSM8K: Empty question in row")
            return None
            
        return {
            'question': question,
            'options': None,
            'images': None
        }

    def get_correct_answer(self, row):
        answer = row.get('answer', "")
        return answer

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

if __name__ == "__main__":
    run_dataset(GSM8KDataset)
