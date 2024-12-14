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
from task_list import Tasks

class OlympicArenaDataset(DatasetHandler):
    """OlympicArena Dataset handler.
    
    A multilingual dataset of mathematical olympiad problems with visual components.
    Features complex mathematical problems that combine geometric reasoning, visual
    understanding, and advanced mathematical concepts across different languages.
    
    Dataset: https://huggingface.co/datasets/GAIR/OlympicArena
    """
    DATASET_NAME = "GAIR/OlympicArena"
    REQUIRED_DATA_KEYS = frozenset({"problem", "figure_urls", "answer", "language"})

    def __init__(self, task, models, sys_config=None):
        logging.info("OlympicArena Dataset initializing")
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
            logging.error("OlympicArena: Failed to extract data from row")
            return "Invalid or missing input data"

        if self.task == Tasks.TASK_GENERATE_ANSWER:
            return self.generate_answer(model_input)

        if self.task == Tasks.TASK_SAVE_QUESTION:
            logging.debug("Executing save question task")
            return model_input
        
        return "Invalid task"

    def generate_answer(self, model_input):
        model = self.get_model()
        if model is None:
            logging.error("OlympicArena: Invalid model selection or initialization failed")
            return "Invalid model selection or initialization failed"

        response = model.get_response(model_input)
        return response
  
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

if __name__ == "__main__":
    run_dataset(OlympicArenaDataset)
