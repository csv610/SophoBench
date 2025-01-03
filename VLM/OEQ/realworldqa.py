import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="realworldqadataset_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class RealWorldQADataset(DatasetHandler):
    """RealWorldQA Dataset handler.
    
    A visual question answering dataset focused on real-world scenarios and practical
    applications. Contains diverse image-question pairs that test models' ability to
    understand and reason about everyday situations and objects.
    
    Dataset: https://huggingface.co/datasets/xai-org/RealworldQA
    """
    HF_DATASET_NAME = "xai-org/RealworldQA"
    SF_DATASET_NAME = "RealworldQA"
    REQUIRED_DATA_KEYS = frozenset({"question", "image", "answer"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task,models, sys_config=None):
        logging.info("RealWorldQA Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        text = row.get('question')  
        if text is None:
            return None

        question, options = self.split_text(text)

        image = row.get('image')
        if image is None:
            return None

        return {
            'question': question,
            'options': options,
            'images': [image]
        }

    def get_correct_answer(self, row):
        answer = row.get('answer')
        if answer is None:
            return None
        return answer

    def split_text(self, text: str):
        """Split the text into question and options."""
        if "Please" in text:
            text = text[:text.rfind("Please")].strip()

        parts = text.split("?", 1)
        if len(parts) == 2:
            question = parts[0] + "?"
            options = re.split(r'(?=\b[A-Z]\.\s)', parts[1].strip())
            options = [opt.strip() for opt in options if opt.strip()]
        else:
            question = text
            options = []

        return question, options

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(RealWorldQADataset)
