import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="cauldron_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_list

class CauldronDataset(DatasetHandler):
    """The Cauldron Dataset handler.
    
    A diverse multimodal dataset for evaluating vision-language models on various
    tasks. Contains multiple-choice questions paired with images, testing models'
    ability to understand and reason about visual content in different contexts.
    
    Dataset: https://huggingface.co/datasets/HuggingFaceM4/the_cauldron
    """
    HF_DATASET_NAME = "HuggingFaceM4/the_cauldron"
    SF_DATASET_NAME = "Cauldron"
    REQUIRED_DATA_KEYS = frozenset({"question", "options", "image", "answer"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task, models, sys_config=None):
        logging.info("Cauldron Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        logging.info(f"Cauldron: Initialized with output suffix: {output_suffix}")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        question = row.get('question',"")
        if not question:
            logging.warning("Cauldron: Empty question in row")
            return None
            
        options = row.get('options', [])
        if not options:
            logging.warning("Cauldron: Empty options in row")
            return None
            
        image = row.get('image', None)
        if image is None:
            logging.warning("Cauldron: Missing image in row")
            return None
            
        answer = row.get('answer', None)
        if answer is None:
            logging.warning("Cauldron: Missing answer in row")
            return None
            
        return {
            'question': question,
            'options': options,
            'images': [image],
            'answer': answer
        }

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)  

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(CauldronDataset)
