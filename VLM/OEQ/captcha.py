import os
import sys
import threading

import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="captchadataset_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class CaptchaDataset(DatasetHandler):
    """CAPTCHA Dataset handler.
    
    A visual question answering dataset focused on CAPTCHA recognition.
    Contains image-text pairs for evaluating model's ability to read
    and interpret text from CAPTCHA images.
    
    Dataset: https://huggingface.co/datasets/hammer888/captcha-data
    """
    HF_DATASET_NAME = "hammer888/captcha-data"
    SF_DATASET_NAME = "Captcha"
    REQUIRED_DATA_KEYS = frozenset({"image", "text"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task.models, sys_config=None):
        logging.info("Captcha Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('vision', "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        image = row.get('image', None)
        if image is None:
            return None

        return {
            'question': "What is the text in this CAPTCHA image?",
            'options': None,
            'images': [image]
        }

    def get_correct_answer(self, row):
        return row.get('text', None)

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(CaptchaDataset)
