import os
import sys
import threading
import logging

from global_setting import add_project_root_to_path, initialize_logging
add_project_root_to_path()
initialize_logging(logfile="animalsdataset_status.log")

from dataset_handler import DatasetHandler 
from model_query import ModelQuery
from dataset_run_util import run_dataset
from task_list import Tasks

class AnimalsDataset(DatasetHandler):
    """Animals Dataset handler.
    
    A dataset for visual question answering about animals in images.
    Supports multiple animal categories for classification and identification.
    
    Dataset: https://huggingface.co/datasets/Fr0styKn1ght/Animals
    """
    # Class-level constants
    ANIMALS = [
        "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar",
        "butterfly", "cat", "caterpillar", "chimpanzee", "cockroach", "cow",
        "coyote", "crab", "crow", "deer", "dog", "dolphin", "donkey", "dragonfly",
        "duck", "eagle", "elephant", "flamingo", "fly", "fox", "goat", "goldfish",
        "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog",
        "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish",
        "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard", "lobster",
        "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter",
        "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes", "penguin",
        "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer",
        "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake",
        "sparrow", "squid", "squirrel", "starfish", "swan", "tiger", "turkey",
        "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"
    ]

    DATASET_NAME = "Fr0styKn1ght/Animals"
    REQUIRED_DATA_KEYS = frozenset({"image", "label"})

    def __init__(self, task,models, sys_config=None):
        logging.info("Animals Dataset initializing")
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
            logging.error("Animals: Failed to extract data from row")
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
            logging.error("Animals: Invalid model selection or initialization failed")
            return "Invalid model selection or initialization failed"
        
        response = model.get_response(model_input)
        return response

    def extract_data(self, row):
        image = row.get('image', None)
        if image is None:
            return None
   
        return {
            "question": "What is the animal in the image?",
            "options": None, 
            "images": [image]
        }

    def get_correct_answer(self, row):
        return self.ANIMALS[row.get('label', 0)]

    def get_model(self):
        return ModelQuery.get_thread_model(self.local_thread, self.models)
        
if __name__ == "__main__":
    run_dataset(AnimalsDataset)
