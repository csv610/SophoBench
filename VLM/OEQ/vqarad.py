import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="vqarad_status.log")

from dataset_handler import DatasetHandler 
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

class VqaRadDataset(DatasetHandler):
    """VQA-RAD (Visual Question Answering in Radiology) Dataset handler.
    
    A specialized dataset for visual question answering in the medical domain,
    focusing on radiology images. Contains question-answer pairs about medical
    imaging findings, anatomical structures, and diagnostic interpretations.
    
    Dataset: https://huggingface.co/datasets/flaviagiammarino/vqa-rad
    """
    DATASET_NAME = "flaviagiammarino/vqa-rad"
    REQUIRED_DATA_KEYS = frozenset({"image", "question", "answer"})

    def __init__(self, task, models, sys_config=None):
        logging.info("VQA-RAD Dataset initializing")
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
            logging.error("Failed to extract data from row: invalid or missing input data")
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
            logging.error("Model initialization failed or invalid model selection")
            return "Invalid model selection or initialization failed"

        response = model.get_response(model_input)
        return response

    def extract_data(self, row):

        question = row['question']
        if question is None:
            return None

        image = row.get('image', None)
        if image is None:
            return None

        return {
            'question': question,
            'options': None,
            'images': [image]
        }

    def get_correct_answer(self, row):
        return row.get('answer', "")

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)
        
if __name__ == "__main__":
    run_dataset(VqaRadDataset)
