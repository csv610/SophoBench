import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="olympiadbenchdataset_status.log")

from sympy import sympify
from sympy.parsing.latex import parse_latex 

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset

class OlympiadBenchDataset(DatasetHandler):
    """OlympiadBench Dataset handler.
    
    A challenging dataset of mathematical olympiad problems with visual elements.
    Contains high-level mathematical problems that require advanced reasoning,
    geometric understanding, and mathematical proof capabilities.
    
    Dataset: https://huggingface.co/datasets/lmms-lab/OlympiadBench
    """
    DATASET_NAME = "lmms-lab/OlympiadBench"
    REQUIRED_DATA_KEYS = frozenset({"question", "images"})

    def __init__(self, task, models, sys_config=None):
        logging.info("OlympiadBench Dataset initializing")
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
            logging.error("OlympiadBench: Failed to extract data from row")
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
            logging.error("OlympiadBench: Invalid model selection or initialization failed")
            return "Invalid model selection or initialization failed"

        response = model.get_response(model_input)
        return response
  
    def extract_data(self, row):
        question = row.get('question', "")
        images = row.get('images', [])

        return {
            "question": question,
            "options": None,
            "images": images
        }

    def get_correct_answer(self, row):
        text = row.get('final_answer', "")
        if text != "":
            answer = parse_latex(text)
            return float(answer)
        return "NA"

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

if __name__ == "__main__":
    run_dataset(OlympiadBenchDataset)
