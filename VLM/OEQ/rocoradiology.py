import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="rocoradiology_status.log")

from model_query_base import ModelQueryBase
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class RocoRadiologyDataset(ModelQueryBase):
    """ROCO (Radiology Objects in COntext) Dataset handler.
    
    A large-scale dataset of radiology images with associated captions and annotations.
    Designed for training and evaluating vision-language models on medical imaging
    tasks, focusing on radiological findings and descriptions.
    
    Dataset: https://huggingface.co/datasets/mdwiratathya/ROCO-radiology
    """
    HF_DATASET_NAME = "mdwiratathya/ROCO-radiology"
    SF_DATASET_NAME = "ROCO-radiology"
    REQUIRED_DATA_KEYS = frozenset({"image", "caption"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task,  models, sys_config=None):
        logging.info("ROCO Radiology Dataset initializing")
        self.task = task  
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        image = row.get('image', None)
        if image is None:
            return None
        
        return {
            'question': "Describe the radiology image",
            'options': None,
            'images': [image]
        }

    def get_correct_answer(self, row):
        answer = row.get('caption', "")
        return answer

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(RocoRadiologyDataset)
