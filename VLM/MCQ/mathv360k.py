import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="mathv360k_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

class MathV360KDataset(DatasetHandler):
    """MathV360K Dataset handler.
    
    A large-scale visual mathematics dataset containing 360,000 math problems
    with images. Problems cover various mathematical topics and difficulty levels,
    testing models' ability to understand and solve mathematical problems with
    visual context.
    
    Dataset: https://huggingface.co/datasets/Zhiqiang007/MathV360K
    """
    DATASET_NAME = "Zhiqiang007/MathV360K"
    REQUIRED_DATA_KEYS = frozenset({"image", "conversations"})

    def __init__(self, task, models, sys_config=None):
        logging.info("MathV360K Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        self.image_folder = None
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
            logging.error("Model initialization failed")
            return "Invalid model selection or initialization failed"
 
        response = model.get_response(model_input)
        return response
  
    def extract_data(self, row):
        conversations = row.get('conversations', [])
        image = row.get('image', None)
        
        if not conversations or image is None:
            return None
            
        # Extract question from the first conversation
        question = conversations[0].get('value', '') if conversations else ''
        
        return {
            'question': question,
            'images': [image]
        }

    def get_correct_answer(self, row):
        data = row['conversation']
        for item in data:
            if item.get("from") == "gpt":
               raw_answer = item.get("value")
               # Remove "The answer is" and strip any extra whitespace
               return raw_answer.replace("The answer is", "").strip()
        return "NA"

    def set_image_folder(imdir):
        self.image_folder = imdir

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

if __name__ == "__main__":
    run_dataset(MathV360KDataset)
