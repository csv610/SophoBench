import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="kvasirvqadataset_status.log")   

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class KvasirvqaDataset(DatasetHandler):
    """Kvasir Visual Question Answering dataset handler.
    
    A medical VQA dataset focused on gastrointestinal endoscopy images. Contains
    question-answer pairs about anatomical landmarks, pathological findings, and
    procedural steps in endoscopic examinations.
    
    Dataset: https://huggingface.co/datasets/SimulaMet-HOST/Kvasir-VQA
    """

    HF_DATASET_NAME = "SimulaMet-HOST/Kvasir-VQA"
    SF_DATASET_NAME = "Kvasir-VQA"
    REQUIRED_DATA_KEYS = frozenset({"image", "question", "answer"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task,models, sys_config=None):
        logging.info("Kvasirvqa Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('vision', "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_datsset_row(self, row):
        return execute_task( self, row)

    def extract_data(self, row):
        return {
            'question': row.get('question', ''),
            'options': None,
            'images': [row.get('image', None)]
        }

    def get_correct_answer(self, row):
        return row.get('answer', '')

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(KvasirvqaDataset)
