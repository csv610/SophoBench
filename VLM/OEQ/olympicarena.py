import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="olympicarenadataset_status.log")

from dataset_handler import DatasetHandler 
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class OlympicArenaDataset(DatasetHandler):
    """OlympicArena Dataset handler.
    
    A multilingual dataset of mathematical olympiad problems with visual components.
    Features complex mathematical problems that combine geometric reasoning, visual
    understanding, and advanced mathematical concepts across different languages.
    
    Dataset: https://huggingface.co/datasets/GAIR/OlympicArena
    """
    HF_DATASET_NAME = "GAIR/OlympicArena"
    SF_DATASET_NAME = "OlympicArena"
    REQUIRED_DATA_KEYS = frozenset({"problem", "figure_urls", "answer", "language"})

    @classmethod
    def is_multimodal(cls):
         return True

    def __init__(self, task, models, sys_config=None):
        logging.info("OlympicArena Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        
        language = row.get('language')
        if language != 'EN':
            return None

        question = row.get('problem',"")
        images   = row.get('figure_urls', [])
        
        return {
            'question': question,
            'options': None,
            'images': images
        }

    def get_correct_answer(self, row):
        answer = row.get('answer', '')
        return answer

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  

if __name__ == "__main__":
    run_dataset(OlympicArenaDataset)
