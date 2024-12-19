import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="slakedataset_status.log")

from dataset_handler import DatasetHandler 
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class SlakeDataset(DatasetHandler):
    """SLAKE (Semantic Language Annotations for Knowledge Extraction) Dataset handler.
    
    A multilingual medical visual question answering dataset. Contains medical images
    paired with questions and answers in multiple languages, focusing on clinical
    understanding and medical knowledge extraction.
    
    Dataset: https://huggingface.co/datasets/BoKelvin/SLAKE
    """
    HF_DATASET_NAME = "BoKelvin/SLAKE"
    SF_DATASET_NAME = "SLAKE"
    REQUIRED_DATA_KEYS = frozenset({"question", "img_name", "answer", "q_lang"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task, models, sys_config=None):
        logging.info("SLAKE Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        self.image_directory = "."
        output_suffix = models.get("vision", "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        question = row.get('question', "")
    
        if row.get('q_lang', "") != 'en':
           return None  

        image = row.get('img_name', None)
        if image is None:
            return None

        return {
            'question': question,
            'options' : None,
            'images'  : [image]
        }

    def get_correct_answer(self, row):
        answer = row.get('answer', "")
        return answer

    def set_image_directory(self, img_dir):
        self.image_directory = img_dir

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(SlakeDataset)
