import os
import sys
import threading
import logging

from global_setting import initialize_logging, add_project_root_to_path
add_project_root_to_path()  
initialize_logging(log_file="medical_meadow_wikidoc_patient_status.log")

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

class MedicalMeadowWikiDocPatientDataset(DatasetHandler):
    """Medical Meadow WikiDoc Patient Information Dataset handler.
    
    A collection of medical information from WikiDoc focused on patient care and education.
    Contains detailed medical information written for patients, covering various conditions,
    treatments, and healthcare topics in an accessible format.
    
    Dataset: https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information
    """
    DATASET_NAME = "medalpaca/medical_meadow_wikidoc_patient_information"
    REQUIRED_DATA_KEYS = frozenset({'input', 'output'})

    def __init__(self, task, models, sys_config=None):
        logging.info("Medical Meadow WikiDoc Patient Information dataset initializing")
        self.task   = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('text', '')
        super().__init__(self.DATASET_NAME, output_suffix, sys_config)

    @classmethod
    def is_multimodal(cls):
        return False

    def apply_op(self, row):
        model_input = self.extract_data(row)
        if model_input is None:
            error_msg = f"WikiDocPatient: Failed to extract data from row: {row.get('input', '[No question found]')[:100]}..."
            logging.error(error_msg)
            return error_msg

        if self.task == Tasks.TASK_GENERATE_ANSWER:
            return self.generate_answer(model_input)
        
        if self.task == Tasks.TASK_SAVE_QUESTION:
            return model_input

        return "Invalid task"

    def generate_answer(self, model_input):
        model = self.get_model()
        if model is None:
            error_msg = "WikiDocPatient: Model initialization failed - invalid model selection or configuration"
            logging.error(error_msg)
            return error_msg
            
        try:
            response = model.get_response(model_input)
            return response
        except Exception as e:
            error_msg = f"WikiDocPatient: Error getting model response: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def extract_data(self, row):
        question = row.get('input', '')
        if not question:
            logging.warning("WikiDocPatient: Empty question in row")
            return None
            
        return {
            'question': question,
            'options': None,
            'images': None
        }

    def get_correct_answer(self, row):
        answer = row.get('output', "")
        return answer

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

if __name__ == "__main__":
    run_dataset(MedicalMeadowWikiDocPatientDataset)
