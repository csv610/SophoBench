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
from task_list import execute_task

class OlympiadBenchDataset(DatasetHandler):
    """OlympiadBench Dataset handler.
    
    A challenging dataset of mathematical olympiad problems with visual elements.
    Contains high-level mathematical problems that require advanced reasoning,
    geometric understanding, and mathematical proof capabilities.
    
    Dataset: https://huggingface.co/datasets/lmms-lab/OlympiadBench
    """
    HF_DATASET_NAME = "lmms-lab/OlympiadBench"
    SF_DATASET_NAME = "OlympiadBench"
    REQUIRED_DATA_KEYS = frozenset({"question", "images"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task, models, sys_config=None):
        logging.info("OlympiadBench Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

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

    def get_task(self):
        return self.task

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

if __name__ == "__main__":
    run_dataset(OlympiadBenchDataset)
