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
from task_list import Tasks

class SlakeDataset(DatasetHandler):
    """SLAKE (Semantic Language Annotations for Knowledge Extraction) Dataset handler.
    
    A multilingual medical visual question answering dataset. Contains medical images
    paired with questions and answers in multiple languages, focusing on clinical
    understanding and medical knowledge extraction.
    
    Dataset: https://huggingface.co/datasets/BoKelvin/SLAKE
    """
    DATASET_NAME = "BoKelvin/SLAKE"
    REQUIRED_DATA_KEYS = frozenset({"question", "img_name", "answer", "q_lang"})

    def __init__(self, task, models, sys_config=None):
        logging.info("SLAKE Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        self.image_directory = "."
        output_suffix = models.get("vision", "")
        super().__init__(self.DATASET_NAME, output_suffix, sys_config)

    @classmethod
    def is_multimodal(cls):
        return True

    def apply_op(self, row):
        model_input = self.extract_data(row)
        if model_input is None:
            logging.error("SLAKE: Failed to extract data from row")
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
            logging.error("SLAKE: Invalid model selection or initialization failed")
            return "Invalid model selection or initialization failed"

        response = model.get_response(model_input)
        return response

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

if __name__ == "__main__":
    run_dataset(SlakeDataset)
