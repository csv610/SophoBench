import ollama
import sys
import threading
import argparse
import logging
import os
import torch
import base64
import requests
from queue import Queue
from PIL import Image
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='query.log', filemode='w')

class ModelQuery:
    DEFAULT_TIMEOUT = 100
    DEFAULT_MODELS  = {'text': "llama3.2", 'vision': "llama3.2-vision"}
    VALID_INPUT_KEYS = {'question', 'options', 'images'}

    def __init__(self, models = DEFAULT_MODELS):
        # Validate model keys and use defaults for invalid keys
        validated_models = {}
        for key in ['text', 'vision']:
            if key not in models:
                logging.warning(f"Missing {key} key in models. Using default value: {self.DEFAULT_MODELS[key]}")
                validated_models[key] = self.DEFAULT_MODELS[key]
            else:
                validated_models[key] = models[key]
                logging.info(f"Using {key} model: {models[key]}")

        self.text_model   = validated_models['text']
        self.vision_model = validated_models['vision']
        
        # Validate models are available
        if not self._validate_models():
            raise RuntimeError("Models not properly initialized. Check if models are installed in Ollama.")
            
        if self.text_model and self.vision_model:
            torch.cuda.empty_cache()
    

    def chat(self, model, messages):
        try:
            response = ollama.chat(model=model, messages=messages)
            return response['message']['content'].strip()
        except Exception as e:
            logging.error(f"Exception occurred while interacting with the model: {e}")
            raise e

    def ensure_list(self, items):
        if isinstance(items, str) or isinstance(items, Image.Image):
            return [items]
        return items

    def validate_inputs(self, question, options=None):
        """Validate question and options.
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not question.strip():
            logging.error("Question cannot be empty")
            return False
        if options is not None and len(options) <= 1:
            logging.error("There must be more than one option")
            return False
        return True

    def validate_image_files(self, images):
        """Validate image files and formats.
        
        Returns:
            bool: True if valid, False otherwise
        """
        images = self.ensure_list(images)
        
        valid_extensions = ['.jpg', '.jpeg', '.png']
        for image in images:
            if isinstance(image, Image.Image):
                continue
            if not (os.path.isfile(image) or image.startswith(('http', 'https')) or isinstance(image, bytes)):
                logging.error(f"Image file {image} does not exist")
                return False
            if isinstance(image, str) and not any(image.lower().endswith(ext) for ext in valid_extensions):
                logging.error(f"Unsupported file extension for {image}. Please use one of the following: .jpg, .jpeg, .png")
                return False
        return True

    def _validate_models(self):
        """Validate that required models are available.
        
        Returns:
            bool: True if valid, False otherwise
        """
        
        try:
            # Check if models are installed in Ollama
            models = ollama.list()
            available_models = {model.model.replace(':latest', '') for model in models['models']}
            
            if self.text_model not in available_models:
                logging.error(f"Text model '{self.text_model}' not installed in Ollama")
                return False
            if self.vision_model not in available_models:
                logging.error(f"Vision model '{self.vision_model}' not installed in Ollama")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Failed to check Ollama models: {str(e)}")
            return False

    def is_valid_mcq_response(self, response, num_options):
        response = response.strip().upper()
        if len(response) != 1:
            return False

        # Ensure the answer is a valid choice
        valid_choices = [chr(65 + i) for i in range(num_options)]
        if response in valid_choices:
            return True

        return False

    def execute_with_timeout(self, target, args, timeout):
        resultQ = Queue()
        thread = threading.Thread(target=target, args=(*args, resultQ))
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return "Error: Request timed out."
        elif resultQ.empty():
            return "Error: No response received."
        return resultQ.get()

    def format_text_mcq(self, question, options):
        options_text = "\n".join([f"({chr(65 + i)}) {option}" for i, option in enumerate(options)])
        complete_question = f"{question}\n{options_text}\nPlease answer with one of the following: {', '.join([f'({chr(65 + i)})' for i in range(len(options))])}. Do not include any explanation or additional text, just respond with the letter. "
        return complete_question

    def text_mcq_ollama(self, question, options, resultQ):
        complete_question = self.format_text_mcq(question, options)
        
        messages = [{'role': 'user', 'content': complete_question}]
        
        try:
            response_content = self.chat(self.text_model, messages).upper()
            response_content = response_content.replace("(", "").replace(")", "").replace(".", "")
            if not response_content:
                resultQ.put("Error: Received empty response from the model.")
            elif len(response_content) < 1:
                resultQ.put("Error: Response content is too short.")
            else:
                resultQ.put(response_content[0])
        except Exception as e:
            resultQ.put(f"Error: Exception occurred during model interaction - {str(e)}")

    def get_text_mcq_answer(self, question, options, timeout=100):
        """Process text MCQ queries."""
        logging.info("Processing Text MCQ")
        
        response_content = self.execute_with_timeout(self.text_mcq_ollama, (question, options), timeout)
        if response_content.startswith("Error"):
            logging.error(response_content)
            return response_content

        valid_response = self.is_valid_mcq_response(response_content, len(options))

        if valid_response:
            return response_content[0]
        else:
            return f"Invalid mcq response: {response_content}"

    def text_oeq_ollama(self, question, resultQ):
        messages = [{'role': 'user', 'content': question}]
        try:
            response = self.chat(self.text_model, messages)
            resultQ.put(response)
        except Exception as e:
            resultQ.put(f"Error: Exception occurred during model interaction - {str(e)}")

    def get_text_oeq_answer(self, question, timeout=100):
        logging.info("Processing Text OEQ")
        # Validate inputs
        self.validate_inputs(question)
        
        response = self.execute_with_timeout(self.text_oeq_ollama, (question,), timeout)
        if isinstance(response, str) and response.startswith("Error"):
            return response

        return response

    def format_image_mcq(self, question, images, options):
        options_text = "\n".join([f"({chr(65 + i)}) {option}" for i, option in enumerate(options)])
        complete_question = f"{question}\n{options_text}\nPlease answer with one of the following: {', '.join([f'({chr(65 + i)})' for i in range(len(options))])}. Do not include any explanation or additional text, just respond with the letter."
        return complete_question

    def encode_images(self, images):
        encoded_images = []
        for image in images:
            if isinstance(image, bytes):
                encoded_images.append(base64.b64encode(image).decode('utf-8'))
            elif isinstance(image, Image.Image):
                temp_image_path = f"{uuid.uuid4().hex}.png"
                with open(temp_image_path, "wb") as img_file:
                    image.save(img_file, format="PNG")
                with open(temp_image_path, "rb") as img_read:
                    encoded_images.append(base64.b64encode(img_read.read()).decode('utf-8'))
                os.remove(temp_image_path)
            elif os.path.isfile(image):
                with open(image, "rb") as img_file:
                    encoded_images.append(base64.b64encode(img_file.read()).decode('utf-8'))
            elif image.startswith(('http', 'https')):
                try:
                    response = requests.get(image)
                    response.raise_for_status()
                    encoded_images.append(base64.b64encode(response.content).decode('utf-8'))
                except requests.RequestException as e:
                    raise ValueError(f"Failed to fetch image from URL {image}: {e}")
            else:
                raise ValueError("Unsupported image format.")
        return encoded_images

    def image_mcq_ollama(self, question, images, options, resultQ):
        images = self.ensure_list(images)
        complete_question = self.format_image_mcq(question, images, options)

        try:
            encoded_images = self.encode_images(images)
        except ValueError as e:
            logging.warning(f"Image validation failed: {e}. Falling back to text model.")
            return self.text_mcq_ollama(question, options, resultQ)
        
        messages = [
            {
                'role': 'user',
                'content': complete_question,
                'images': encoded_images
            }
        ]
        try:
            result = self.chat(self.vision_model, messages)
            resultQ.put(result)
        except Exception as e:
            resultQ.put(f"Error: Exception occurred during model interaction - {str(e)}")

    def get_image_mcq_answer(self, question, images, options, timeout=100):
        """Process image MCQ queries."""
        logging.info("Processing Image MCQ")
        
        response_content = self.execute_with_timeout(self.image_mcq_ollama, (question, images, options), timeout)
        if isinstance(response_content, str) and response_content.startswith("Error"):
            logging.error(response_content)
            return response_content

        response_content = response_content.upper().replace("(", "").replace(")", "").replace(".", "")
        if not response_content:
            return "Error: Empty response from the model."

        valid_response = self.is_valid_mcq_response(response_content, len(options))

        if valid_response:
            return response_content[0]
        else:
            return f"Invalid mcq response: {response_content}"

    def image_oeq_llama(self, question, images, resultQ):
        images = self.ensure_list(images)
        try:
            encoded_images = self.encode_images(images)
        except ValueError as e:
            logging.warning(f"Image validation failed: {e}. Falling back to text model.")
            return self.text_oeq_ollama(question, resultQ)
        
        messages = [{'role': 'user', 'content': question, 'images': encoded_images}]
        
        try:
            response = self.chat(self.vision_model, messages)
            resultQ.put(response)
        except Exception as e:
            resultQ.put(f"Error: Exception occurred during model interaction - {str(e)}")

    def get_image_oeq_answer(self, question, images, timeout=100):
        logging.info("Processing Image OEQ")
        # Validate inputs
        self.validate_inputs(question)
        
        # Validate image files
        try:
            self.validate_image_files(images)
        except (FileNotFoundError, ValueError) as e:
            logging.warning(f"Image validation failed: {e}. Falling back to text model.")
            return self.get_text_oeq_answer(question, timeout)
        
        response = self.execute_with_timeout(self.image_oeq_llama, (question, images), timeout)
        if isinstance(response, str) and response.startswith("Error"):
            return response
        
        return response

    def _validate_timeout(self, timeout):
        """Validate timeout parameter.
        
        Args:
            timeout: The timeout value to validate
            
        Returns:
            bool: True if timeout is valid, False otherwise
        """
        if not isinstance(timeout, int):
            logging.error("Invalid timeout type: timeout must be an integer")
            return False
        if timeout <= 0:
            logging.error("Invalid timeout value: timeout must be positive")
            return False
        return True

    def _validate_input(self, model_input):
        """Validate the model input dictionary to ensure only valid keys are present.
        
        Args:
            model_input (dict): The input dictionary to validate
            
        Returns:
            bool: True if input is valid, False otherwise
        """
        if not isinstance(model_input, dict):
            logging.error("Invalid input type: model_input must be a dictionary")
            return False
            
        provided_keys = set(model_input.keys())
        
        # Check for invalid keys
        invalid_keys = provided_keys - self.VALID_INPUT_KEYS
        if invalid_keys:
            logging.error(f"Invalid keys detected in model_input: {invalid_keys}")
            return False
            
        # Check for required keys
        if 'question' not in model_input:
            logging.error("Required key 'question' missing from model_input")
            return False
            
        # Validate question is not empty
        if not model_input['question'].strip():
            logging.error("Empty question provided in model_input")
            return False
        
        # Validate options if present
        if 'options' in model_input:
            options = model_input.get('options', [])
            if options:  # Only validate if options list is non-empty
                if any(not str(option).strip() for option in options):
                    logging.error("Empty option detected in options list")
                    return False
        
        # For image processing, validate images key
        if 'images' in model_input:
            images = model_input.get('images', [])
            if images:  # Only validate if images list is non-empty
                if any(not str(image).strip() for image in images):
                    logging.error("Empty image path detected in images list")
                    return False
                try:
                    self.validate_image_files(images)
                except (ValueError, FileNotFoundError) as e:
                    logging.error(f"Image validation failed: {str(e)}")
                    return False
            
        return True
    
    def _validate_arguments(self, model_input, timeout):
        """Validate all arguments required for processing queries.
        
        Args:
            model_input (dict): The input dictionary to validate
            timeout (int): The timeout value to validate
            
        Returns:
            tuple: (bool, str) - (True, None) if valid, (False, error_message) if invalid
        """
        # Check input
        if not self._validate_input(model_input):
            return False, "Error: Invalid input"
            
        # Check timeout
        if not self._validate_timeout(timeout):
            return False, "Error: Invalid timeout"
            
        return True, None

    def get_text_response(self, model_input, timeout=DEFAULT_TIMEOUT):
        """Process text-based queries."""
        
        question = model_input['question'].strip()
        options = model_input.get('options', [])
        
        if options:
            return self.get_text_mcq_answer(
                question,
                options,
                timeout
            )
        return self.get_text_oeq_answer(
            question,
            timeout
        )

    def get_image_response(self, model_input, timeout=DEFAULT_TIMEOUT):
        """Process image-based queries."""
        
        question = model_input['question'].strip()
        images = model_input['images']
        options = model_input.get('options', [])
        
        if options:
            return self.get_image_mcq_answer(
                question,
                images,
                options,
                timeout
            )
        return self.get_image_oeq_answer(
            question,
            images,
            timeout
        )

    def get_response(self, model_input, timeout=DEFAULT_TIMEOUT):
        """Process queries with either text or image model."""
        is_valid, error = self._validate_arguments(model_input, timeout)
        if not is_valid:
            return error
        
        if not model_input.get('images') or len(model_input.get('images', [])) == 0:
            return self.get_text_response(model_input, timeout)
        
        return self.get_image_response(model_input, timeout)

    @staticmethod
    def get_thread_model(local_thread, models):
        """Static method to get or create a model instance for a thread.
        
        Args:
            local_thread: threading.local instance
            models: dict of model configurations
            
        Returns:
            ModelQuery instance or None if initialization fails
        """
        if not hasattr(local_thread, 'model'):
            try:
                local_thread.model = ModelQuery(models)
            except Exception as e:
                return None
        return local_thread.model
