import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="medtrinity25mdataset_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset

class MedTrinity25MDataset(DatasetHandler):
    """MedTrinity-25M Dataset handler.
    
    A large-scale medical vision-language dataset containing 25M medical image-text pairs.
    Designed for medical visual question answering and image captioning tasks.
    
    Dataset: https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M
    """
    DATASET_NAME = "UCSC-VLAA/MedTrinity-25M"
    REQUIRED_DATA_KEYS = frozenset({"image", "caption"})

    def __init__(self, task, models, sys_config=None):
        logging.info("MedTrinity-25M Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('vision', "")
        super().__init__(self.DATASET_NAME, output_suffix, sys_config)

    @classmethod
    def is_multimodal(cls):
        return True

    def apply_op(self, row):
        model_input = self.extract_data(row)
        if model_input is None:
            logging.error("MedTrinity-25M: Failed to extract data from row - missing image")
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
            logging.error("MedTrinity-25M: Invalid model selection or initialization failed")
            return "Invalid model selection or initialization failed"

        response = model.get_response(model_input)
        return response

    def extract_data(self, row):
        image = row.get('image', None)
        if image is None:
            return None
        
        return {
            'question': "Provide detail decription of the image",
            'options': None,
            'images': [image]
        }

    def get_correct_answer(self, row):
        ans = row.get('caption', "")
        return ans

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

if __name__ == "__main__":
    run_dataset(MedTrinity25MDataset)
