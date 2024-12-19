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
from task_list import execute_task

class MedicalMeadowWikiDocPatientDataset(DatasetHandler):
    """Medical Meadow WikiDoc Patient Information Dataset handler.
    
    A collection of medical information from WikiDoc focused on patient care and education.
    Contains detailed medical information written for patients, covering various conditions,
    treatments, and healthcare topics in an accessible format.
    
    Dataset: https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information
    """
    HF_DATASET_NAME = "medalpaca/medical_meadow_wikidoc_patient_information"
    SF_DATASET_NAME = "MedMeadowWikidocPatient"
    REQUIRED_DATA_KEYS = frozenset({'input', 'output'})

    @classmethod
    def is_multimodal(cls):
        return False

    def __init__(self, task, models, sys_config=None):
        logging.info("Medical Meadow WikiDoc Patient Information dataset initializing")
        self.task   = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('text', '')
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)

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

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(MedicalMeadowWikiDocPatientDataset)
