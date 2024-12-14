import os
import sys
import threading
import logging

# Initialize logging only if not already initialized
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='datalog.log',
        filemode='w'
    )

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from dataset_handler import DatasetHandler
from model_query import ModelQuery
from dataset_run_util import run_dataset

class NEJMDataset(DatasetHandler):
    """NEJM (New England Journal of Medicine) Dataset handler.
    
    A dataset containing medical case studies and questions from the New England Journal 
    of Medicine, including images and multiple-choice questions that test medical 
    knowledge and diagnostic reasoning.
    """
    DATASET_NAME = "./data/nejm.json"  # Local dataset
    REQUIRED_DATA_KEYS = frozenset({"Question", "Options", "Image", "Answer"})
    
    def __init__(self, models, sys_config=None):
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get("vision", "")
        logging.info(f"NEJM: Initialized with output suffix: {output_suffix}")
        super().__init__(self.DATASET_NAME, output_suffix, sys_config)

    @classmethod
    def is_multimodal(cls):
        return True

    def process_question(self, row):
        model = self.get_model()
        if model is None:
            error_msg = "NEJM: Model initialization failed - invalid model selection or configuration"
            logging.error(error_msg)
            return error_msg
            
        model_input = self.extract_data(row)
        if model_input is None:
            error_msg = f"NEJM: Failed to extract data from row: {row.get('Question', '[No question found]')[:100]}..."
            logging.error(error_msg)
            return error_msg
            
        try:
            response = model.get_response(model_input)
            return response
        except Exception as e:
            error_msg = f"NEJM: Error getting model response: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def extract_data(self, row):
        question = row.get('Question', '')
        options = row.get('Options', [])
        image = row.get('Image', '')
        
        if not question:
            logging.warning("NEJM: Empty question in row")
            return None
            
        if not options:
            logging.warning("NEJM: No options provided for question")
            return None
            
        if not image:
            logging.warning("NEJM: No image provided for question")
            return None
            
        return {
            'question': question,
            'options': options,
            'images': [image]  # Wrap single image in list as expected by model
        }

    def get_correct_answer(self, row):
        return row.get("Answer", "")

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)

if __name__ == "__main__":
    run_dataset(NEJMDataset)
