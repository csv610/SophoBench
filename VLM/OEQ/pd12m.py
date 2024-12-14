import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="pd12mdataset_status.log")

from dataset_handler import DatasetHandler 
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

class PD12MDataset(DatasetHandler):
    """PD12M (Public Domain 12M) Dataset handler.
    
    A large-scale dataset containing 12 million image-caption pairs from public domain
    sources. Designed for training and evaluating vision-language models on diverse,
    real-world content with high-quality annotations.
    
    Dataset: https://huggingface.co/datasets/Spawning/PD12M
    """
    DATASET_NAME = "Spawning/PD12M"
    REQUIRED_DATA_KEYS = frozenset({"caption", "url", "hash"})

    def __init__(self, task,models, sys_config=None):
        logging.info("PD12M Dataset initializing")
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
            logging.error("PD12M: Failed to extract data from row - missing image URL")
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
            logging.error("PD12M: Invalid model selection or initialization failed")
            return "Invalid model selection or initialization failed"
            
        response = model.get_response(model_input)
        return response

    def extract_data(self, row):

        image = row.get('url', None)
        if image is None:
            return None
       
        return {
            'question': "Give dense caption for this image", 
            'options': None,
            'images': [image]
        }

    def get_correct_answer(self, row):
        answer = row.get('caption', None)
        if answer is None:
            return None
        return answer

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

if __name__ == "__main__":
    run_dataset(PD12MDataset)
