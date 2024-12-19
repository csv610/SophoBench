import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="theoremqadataset_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class TheoremQADataset(DatasetHandler):
    """TheoremQA Dataset handler.
    
    A specialized dataset for mathematical theorem question answering with visual
    elements. Contains questions about mathematical theorems, proofs, and concepts,
    often accompanied by diagrams, graphs, or mathematical visualizations.
    
    Dataset: https://huggingface.co/datasets/TIGER-Lab/TheoremQA
    """
    HF_DATASET_NAME = "TIGER-Lab/TheoremQA"
    SF_DATASET_NAME = "TheoremQA"
    REQUIRED_DATA_KEYS = frozenset({"Question", "Answer", "Picture"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task,models, sys_config=None):
        logging.info("TheoremQA Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        question = row.get('Question', '')
        if question is None:
            return None

        image = row.get('Picture', None)
        if image is None:
            return None

        return {
            'question': question,
            'options': None,
            'images': [image]
        }

    def get_correct_answer(self, row):
        return row.get('Answer', "")

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(TheoremQADataset)
