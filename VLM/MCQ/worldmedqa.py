import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="worldmedqadataset_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class WorldMedQADataset(DatasetHandler):
    """WorldMedQA Dataset handler.
    
    A global medical visual question answering dataset. Contains multiple-choice
    questions about medical images, designed to test clinical knowledge and
    visual understanding in medical contexts across different languages and
    healthcare systems.
    
    Dataset: https://huggingface.co/datasets/WorldMedQA/V
    """
    HF_DATASET_NAME = "WorldMedQA/V"
    SF_DATASET_NAME = "WorldMedQAV"
    REQUIRED_DATA_KEYS = frozenset({"question", "A", "B", "C", "D", "image", "correct_option"})
    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task, models, sys_config=None):
        logging.info("WorldMedQA Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
            
        question = row.get('question',"")
            
        options = []
        for opt in ['A', 'B', 'C', 'D']:
            val = row.get(opt, "")
            if val is not None:
                options.append(val)
                
        images = [row.get('image')]
        return {
            'question': question,
            'options': options,
            'images': images
        }

    def get_correct_answer(self, row):
        ans = row.get('correct_option',"")
        return ans

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(WorldMedQADataset)
