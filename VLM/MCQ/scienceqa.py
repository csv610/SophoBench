import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="scienceqadataset_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class ScienceQADataset(DatasetHandler):
    """ScienceQA Dataset handler.
    
    A comprehensive science question answering dataset that combines text and
    visual elements. Contains multiple-choice questions across various science
    topics, often accompanied by diagrams, graphs, or other visual aids to test
    scientific understanding.
    
    Dataset: https://huggingface.co/datasets/derek-thomas/ScienceQA
    """
    HF_DATASET_NAME = "derek-thomas/ScienceQA"
    SF_DATASET_NAME = "ScienceQA"
    REQUIRED_DATA_KEYS = frozenset({"question", "options", "image", "answer"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task, models, sys_config=None):
        logging.info("ScienceQA Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        question = row.get('question', '')
        if question == "":
            logging.error("Empty question received in row data")
            return None
            
        options = row.get('choices', [])
        image = row.get('image', None)
                
        return {
            'question': question,
            'options': options,
            'images': [image]
        }

    def get_correct_answer(self, row):
        ans = row.get('answer',"")
        if ans:
            return chr(65+ans)
        return "NA"

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(ScienceQADataset)
