import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="mathvistadataset_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

#https://huggingface.co/datasets/AI4Math/MathVista

class MathVistaDataset(DatasetHandler):
    """MathVista Dataset handler.
    
    A comprehensive visual mathematics dataset for evaluating mathematical reasoning
    in vision-language models. Includes a diverse range of mathematical problems
    presented through diagrams, charts, and visual representations.
    
    Dataset: https://huggingface.co/datasets/AI4Math/MathVista
    """
    DATASET_NAME = "AI4Math/MathVista"
    REQUIRED_DATA_KEYS = frozenset({"question", "choices", "decoded_image", "answer"})

    def __init__(self, task, models, sys_config=None):
        logging.info("Math Vista Dataset initializing")
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
            logging.error("MathVista: Failed to extract data from row")
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
            logging.error("MathVista: Invalid model selection or initialization failed")
            return "Invalid model selection or initialization failed"
            
        response = model.get_response(model_input)
        return response

    def extract_data(self, row):
        question = row.get('question', "")

        image = row.get('decoded_image', None)
        if image is None:
            return None
    
        return {
            'question': question,
            'options': None,
            'images': [image]
        }

    def get_correct_answer(self, row):
        ans = row.get('answer', "")
        return ans

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

if __name__ == "__main__":
    run_dataset(MathVistaDataset)
