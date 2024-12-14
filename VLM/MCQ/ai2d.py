import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="ai2d_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

class AI2DDataset(DatasetHandler):
    """AI2D (AI2 Diagrams) Dataset handler.
    
    A comprehensive dataset for diagram question answering, containing science
    diagrams with associated questions and multiple-choice answers. Tests models'
    ability to understand and reason about scientific diagrams and their labels.
    
    Dataset: https://huggingface.co/datasets/lmms-lab/ai2d
    """
    DATASET_NAME = "lmms-lab/ai2d"
    REQUIRED_DATA_KEYS = frozenset({"question", "options", "image", "answer"})

    def __init__(self, task, models, sys_config=None):
        logging.info("AI2D Dataset initializing")
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
            logging.warning("Failed to extract data from row: %s", row)
            return "Data extraction failed"
        
        if self.task == Tasks.TASK_GENERATE_ANSWER:
            return self.generate_answer(model_input)

        if self.task == Tasks.TASK_SAVE_QUESTION:
            logging.debug("Executing save question task")
            return model_input

        return "Invalid task"

    def generate_answer(self, model_input):
        model = self.get_model()
        if model is None:
            logging.error("Model is not available for processing question")
            return "Invalid model selection or initialization failed"

        response = model.get_response(model_input)
        return response

    def extract_data(self, row):
        question = row.get('question')
        options  = row.get('options', [])
        images   = [row.get('image')]

        return {
            'question': question,
            'options': options,
            'images': images
        }

    def get_correct_answer(self, row):
        label = chr(65 + row.get('answer', 0))
        return label
    
    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

if __name__ == "__main__":
    run_dataset(AI2DDataset)
