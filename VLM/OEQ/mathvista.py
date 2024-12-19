import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="mathvistadataset_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

#https://huggingface.co/datasets/AI4Math/MathVista

class MathVistaDataset(DatasetHandler):
    """MathVista Dataset handler.
    
    A comprehensive visual mathematics dataset for evaluating mathematical reasoning
    in vision-language models. Includes a diverse range of mathematical problems
    presented through diagrams, charts, and visual representations.
    
    Dataset: https://huggingface.co/datasets/AI4Math/MathVista
    """
    HF_DATASET_NAME = "AI4Math/MathVista"
    SF_DATASET_NAME = "MathVista"
    REQUIRED_DATA_KEYS = frozenset({"question", "choices", "decoded_image", "answer"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task, models, sys_config=None):
        logging.info("Math Vista Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        question = row.get('question', "")

        image = row.get('decoded_image', None)
        if image is None:
            return None
    
        return {
            'question': question,
            'options': None,
            'images': [image]
        }

    def get_correct_answer(self, row):
        ans = row.get('answer', "")
        return ans

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self.):
        return self.task
  
if __name__ == "__main__":
    run_dataset(MathVistaDataset)
