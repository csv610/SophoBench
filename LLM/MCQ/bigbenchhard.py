import os
import sys
import threading
import re
import logging


from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()
initialize_logging(log_file="bigbenchhard_status.log")

from dataset_handler import DatasetHandler
from model_query  import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

class BigBenchHardDataset(DatasetHandler):
    DATASET_NAME = "maveriq/bigbenchhard"
    REQUIRED_DATA_KEYS = frozenset({"input", "target"})

    def __init__(self, task, models, sys_config=None):
        logging.info("BigBenchHard dataset initializing")
        slef.task   = task
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
            return "Failed to extract question and options from input"

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
            
        input_text = row.get('input',"")     
        if input_text == "":
            logging.error("Missing input text")
            return None
            
        # Process input text to extract question and options
        match = re.search(r'(.*?)\s*Options:\s*(.*)', input_text, re.DOTALL)
        if not match:
            logging.error("Could not parse question and options from input text")
            return None
            
        question = match.group(1).strip()

        options_text = match.group(2).strip()
        options = [opt.strip() for opt in options_text.split('\n') if opt.strip()]
        
        if not options:
            logging.error("No options found in input text")
            return None
            
        return {
            'question': question,
            'options': options,
            'images': None
        }

    def get_correct_answer(self, row):
        return row['target'].replace('(', '').replace(')', '') if isinstance(row['target'], str) else row['target']

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

if __name__ == "__main__":
    run_dataset(BigBenchHardDataset)
