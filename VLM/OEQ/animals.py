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
from task_list import execute_task

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

    HF_DATASET_NAME = "Fr0styKn1ght/Animals"
    SF_DATASET_NAME = "Animals"
    REQUIRED_DATA_KEYS = frozenset({"image", "label"})

    @classmethod
    def is_multimodal(cls):
        return True

    def __init__(self, task,models, sys_config=None):
        logging.info("Animals Dataset initializing")
        self.task = task
        self.models = models
        self.local_thread = threading.local()
        output_suffix = models.get('vision', "")
        super().__init__(self.HF_DATASET_NAME, output_suffix, sys_config)

    def process_dataset_row(self, row):
        return execute_task(self, row)
        
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

    def get_dataset_name(self):
        return self.SF_DATASET_NAME

    def get_assigned_task(self):
        return self.task
  
if __name__ == "__main__":
    run_dataset(AnimalsDataset)
