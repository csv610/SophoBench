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
from task_list import Tasks

class ScienceQADataset(DatasetHandler):
    """ScienceQA Dataset handler.
    
    A comprehensive science question answering dataset that combines text and
    visual elements. Contains multiple-choice questions across various science
    topics, often accompanied by diagrams, graphs, or other visual aids to test
    scientific understanding.
    
    Dataset: https://huggingface.co/datasets/derek-thomas/ScienceQA
    """
    DATASET_NAME = "derek-thomas/ScienceQA"
    REQUIRED_DATA_KEYS = frozenset({"question", "options", "image", "answer"})

    def __init__(self, task, models, sys_config=None):
        logging.info("ScienceQA Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        super().__init__(self.DATASET_NAME, output_suffix, sys_config)

    @classmethod
    def is_multimodal(cls):
        return True

    def apply_op(self, row):
        model_input = self.extract_data(row)
        if model_input is None:
            logging.error("Failed to process input row data")
            return None

        if self.task == Tasks.TASK_GENERATE_ANSWER:
            return self.generate_answer(model_input)

        if self.task == Tasks.TASK_SAVE_QUESTION:
            logging.debug("Executing save question task")
            return model_input
        
        return "Invalid task"

    def generate_answer(self, model_input):
        model = self.get_model()
        if model is None:
            logging.error("Invalid model selection or initialization failed")
            return None
        response = model.get_response(model_input)
        return response

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

if __name__ == "__main__":
    run_dataset(ScienceQADataset)
