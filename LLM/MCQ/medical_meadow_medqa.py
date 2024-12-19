import os
import sys
import threading
import logging

from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()  
initialize_logging(log_file="medical_meadow_medqa_status.log")

import re
import ast
from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import execute_task

class MedicalMeadowMedQADataset(DatasetHandler):
    """Medical Meadow MedQA dataset handler.
    
    A comprehensive medical question-answering dataset focused on healthcare knowledge.
    Contains multiple-choice questions covering various medical topics and scenarios.
    
    Dataset: https://huggingface.co/datasets/medalpaca/medical_meadow_medqa
    """
    HF_DATASET_NAME = "medalpaca/medical_meadow_medqa"
    SF_DATASET_NAME = "MedMeadowQA"  #Short form 
    REQUIRED_DATA_KEYS = frozenset({"input", "output"})
    
    @classmethod
    def is_multimodal(cls):
        return False

    def __init__(self, task, models, sys_config=None):
        logging.info("Medical Meadow MedQA dataset initializing")
        self.task   = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('text', "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

    def extract_data(self, row):
        input_text = row.get('input', "")
        if input_text == "":
            logging.warning("Missing input text")
            return None
        
        # Find the last dictionary in the text
        dict_start = input_text.rfind('{')
        if dict_start == -1:
            logging.warning("Could not find options dictionary")
            return None
        
        # Split into question and options
        # Remove the 'Q:' prefix from the question
        question = input_text[:dict_start].strip()
        if question.startswith('Q:'):
            question = question[2:].strip()
        options_dict_str = input_text[dict_start:].strip()
        
        try:
            # Parse the options dictionary
            parsed = ast.literal_eval(options_dict_str)
            # Handle case where it's a tuple with one dictionary due to trailing comma
            choices_dict = parsed[0] if isinstance(parsed, tuple) else parsed
            if not isinstance(choices_dict, dict):
                logging.warning("Parsed options is not a dictionary")
                return None
            options = list(choices_dict.values())
            if not options:
                logging.warning("No valid options found")
                return None
        except Exception as e:
            logging.error(f"Error parsing options: {str(e)}")
            return None
        
        return {
            'question': question,
            'options': options,
            'images': None
        }

    def get_correct_answer(self, row):
        output = row.get('output', "")            
        return output[0]

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models) 

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_dataset(MedicalMeadowMedQADataset)
