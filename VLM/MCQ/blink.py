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
from task_list import Tasks

class BLINKDataset(DatasetHandler):
    """BLINK (Benchmark for Language-Image Navigation Knowledge) Dataset handler.
    
    A comprehensive benchmark for evaluating vision-language models on navigation
    and spatial reasoning tasks. Contains questions with multiple image choices,
    testing models' ability to understand spatial relationships and navigation
    instructions.
    
    Dataset: https://huggingface.co/datasets/BLINK-Benchmark/BLINK
    """
    DATASET_NAME = "BLINK-Benchmark/BLINK"
    REQUIRED_DATA_KEYS = frozenset({"question", "choices", "image_1", "image_2", "image_3", "image_4", "answer"})

    def __init__(self, task, models, sys_config=None):
        logging.info("BLINK Dataset initializing")
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
            logging.error("BLINK: Failed to extract data from row")
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
            logging.error("BLINK: Invalid model selection or initialization failed")
            return "Invalid model selection or initialization failed"
            
        response = model.get_response(model_input)
        return response

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

if __name__ == "__main__":
    run_dataset(BLINKDataset)
