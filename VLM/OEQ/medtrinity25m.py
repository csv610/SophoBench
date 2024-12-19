import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="medtrinity25mdataset_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class MedTrinity25MDataset(DatasetHandler):
    """MedTrinity-25M Dataset handler.
    
    A large-scale medical vision-language dataset containing 25M medical image-text pairs.
    Designed for medical visual question answering and image captioning tasks.
    
    Dataset: https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M
    """
    HF_DATASET_NAME = "UCSC-VLAA/MedTrinity-25M"
    SF_DATASET_NAME = "MedTrinity-25M"
    REQUIRED_DATA_KEYS = frozenset({"image", "caption"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task, models, sys_config=None):
        logging.info("MedTrinity-25M Dataset initializing")
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
            'question': "Provide detail decription of the image",
            'options': None,
            'images': [image]
        }

    def get_correct_answer(self, row):
        ans = row.get('caption', "")
        return ans

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(MedTrinity25MDataset)
