import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="blink_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class BLINKDataset(DatasetHandler):
    """BLINK (Benchmark for Language-Image Navigation Knowledge) Dataset handler.
    
    A comprehensive benchmark for evaluating vision-language models on navigation
    and spatial reasoning tasks. Contains questions with multiple image choices,
    testing models' ability to understand spatial relationships and navigation
    instructions.
    
    Dataset: https://huggingface.co/datasets/BLINK-Benchmark/BLINK
    """
    HF_DATASET_NAME = "BLINK-Benchmark/BLINK"
    SF_DATASET_NAME = "BLINK"
    REQUIRED_DATA_KEYS = frozenset({"question", "choices", "image_1", "image_2", "image_3", "image_4", "answer"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task, models, sys_config=None):
        logging.info("BLINK Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        question = row.get('question',"")
        if not question:
            logging.warning("BLINK: Empty question in row")
            return None
            
        options  = row.get('options', [])
        if not options:
            logging.warning("BLINK: Empty options in row")
            return None
        
        images  = []
        for j in range(1, 5):
            image_key = f'image_{j}'
            image = row.get(image_key, None)
            if image:
               images.append(image)
               
        return {
            'question': question,
            'options': options, 
            'images': images
        }

    def get_correct_answer(self, row):
        answer = row.get('answer')
        if answer is None or answer == "":
            return "NA"
        label = chr(65 + answer)
        return label

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(BLINKDataset)
