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
from task_list import Tasks

#URL: https://huggingface.co/datasets/MathLLMs/MathVision

class MathVisionDataset(DatasetHandler):
    """MathVision Dataset handler.
    
    A visual mathematics dataset for evaluating mathematical reasoning capabilities
    of vision-language models. Contains questions with mathematical diagrams, graphs,
    and visual problem-solving scenarios.
    
    Dataset: https://huggingface.co/datasets/MathLLMs/MathVision
    """
    DATASET_NAME = "MathLLMs/MathVision"
    REQUIRED_DATA_KEYS = frozenset({"question", "options", "decoded_image", "answer"})

    def __init__(self, task, models, sys_config=None):
        logging.info("MathVision Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        super().__init__(self.DATASET_NAME, output_suffix, sys_config)

    @classmethod
    def is_multimodal(cls):
        return True

    def apply_op(self, row):
        model_input = self.extract_data(row)
        if model_input is None:
            logging.error("MathVision: Failed to extract data from row")
            return "Invalid or missing input data"
        
        if self.task == Tasks.TASK_GENERATE_ANSWER:
            return self.generate_answer(model_input)

        if self.task == Tasks.TASK_SAVE_QUESTION:
            logging.debug("Executing save question task")
            return model_input

        return "Invalid task"
        
    def generate_answer(self, model_input):
        model = self.get_model()
        if model is None:
            logging.error("MathVision: Invalid model selection or initialization failed")
            return "Invalid model selection or initialization failed"
            
        response = model.get_response(model_input)
        return response

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

if __name__ == "__main__":
    run_dataset(MathVisionDataset)
