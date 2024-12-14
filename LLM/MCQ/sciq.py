import os
import sys
import threading

# Import utility functions from local global_setting.py
from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()
initialize_logging(log_file="sciq_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

class SciqDataset(DatasetHandler):
    DATASET_NAME  = "allenai/sciq"
    REQUIRED_DATA_KEYS = frozenset({'question', 'correct_answer', 'distractor1', 'distractor2', 'distractor3'})

    def __init__(self, task, models, sys_config=None):
        logging.info("Sciq dataset initializing")    
        self.task   = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("text", "")
        super().__init__(self.DATASET_NAME, output_suffix, sys_config)

    @classmethod
    def is_multimodal(cls):
        return False

    def process_question(self, row):
        model = self.get_model()
        if model is None:
            return "Invalid model selection or initialization failed"
            
        model_input = self.extract_data(row)
        if model_input is None:
            return "Failed to extract data from question"
            
        response = model.get_response(model_input)
        return response

    def extract_data(self, row):
        
        question = row.get('question', '')
        
        opt1 = row.get('correct_answer', '')
        opt2 = row.get('distractor1', '')
        opt3 = row.get('distractor2', '')
        opt4 = row.get('distractor3', '')
               
        options = [opt1, opt2, opt3, opt4]
                    
        return {
            'question': question,
            'options': options,
            'images': None
        }
       
    def get_correct_answer(self, row):
        return "A"

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)    

if __name__ == "__main__":
    run_dataset(SciqDataset)
