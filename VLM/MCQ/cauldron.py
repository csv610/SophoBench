import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="cauldron_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

class CauldronDataset(DatasetHandler):
    """The Cauldron Dataset handler.
    
    A diverse multimodal dataset for evaluating vision-language models on various
    tasks. Contains multiple-choice questions paired with images, testing models'
    ability to understand and reason about visual content in different contexts.
    
    Dataset: https://huggingface.co/datasets/HuggingFaceM4/the_cauldron
    """
    DATASET_NAME = "HuggingFaceM4/the_cauldron"
    REQUIRED_DATA_KEYS = frozenset({"question", "options", "image", "answer"})

    def __init__(self, task, models, sys_config=None):
        logging.info("Cauldron Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        logging.info(f"Cauldron: Initialized with output suffix: {output_suffix}")
        super().__init__(self.DATASET_NAME, output_suffix, sys_config)

    @classmethod
    def is_multimodal(cls):
        return True

    def apply_op(self, row):
        model_input = self.extract_data(row)
        if model_input is None:
            error_msg = f"Cauldron: Failed to extract data from row: {row.get('question', '[No question found]')[:100]}..."
            logging.error(error_msg)
            return error_msg
        
        if self.task == Tasks.TASK_GENERATE_ANSWER:
            return self.generate_answer(model_input)

        if self.task == Tasks.TASK_SAVE_QUESTION:
            logging.debug("Executing save question task")
            return model_input

        return "Invalid task"

    def generate_answer(self, model_input):
        model = self.get_model()
        if model is None:
            error_msg = "Cauldron: Model initialization failed - invalid model selection or configuration"
            logging.error(error_msg)
            return error_msg

        try:
            response = model.get_response(model_input)
            return response
        except Exception as e:
            error_msg = f"Cauldron: Error getting model response: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def extract_data(self, row):
        question = row.get('question',"")
        if not question:
            logging.warning("Cauldron: Empty question in row")
            return None
            
        options = row.get('options', [])
        if not options:
            logging.warning("Cauldron: Empty options in row")
            return None
            
        image = row.get('image', None)
        if image is None:
            logging.warning("Cauldron: Missing image in row")
            return None
            
        answer = row.get('answer', None)
        if answer is None:
            logging.warning("Cauldron: Missing answer in row")
            return None
            
        return {
            'question': question,
            'options': options,
            'images': [image],
            'answer': answer
        }

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)  

if __name__ == "__main__":
    run_dataset(CauldronDataset)
