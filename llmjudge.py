import openai
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

class LLMJudge:
    def __init__(self):
        """
        Initialize the LLMJudge class and load the OpenAI API key from the environment.
        """
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Set it as the environment variable 'OPENAI_API_KEY'.")
        self.blip_processor = None
        self.blip_model = None
        self.default_rubric = """
        1. Accuracy: Is the information correct and free from errors?
        2. Completeness: Does the response fully address all aspects of the question?
        3. Clarity: Is the response well-articulated and easy to understand?
        4. Depth of Understanding: Does the response demonstrate a thorough understanding of the topic?
        5. Use of Terminology: Are technical terms used correctly and appropriately?
        6. Structure and Organization: Is the response logically structured and coherent?
        7. Relevance: Does the response stay focused on the question without unnecessary deviation?
        8. Integration of Image (if applicable): If an image is provided, does the response effectively utilize it in the explanation?
        """

    def generate_dense_caption(self, image_path):
        """
        Generate a dense caption for an image using the BLIP-2 model.
        """
        try:
            if not self.blip_processor or not self.blip_model:
                print("Loading BLIP-2 model...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip2-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-image-captioning-base")

                # Check if GPU is available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.blip_model = self.blip_model.to(device)

            print("Processing the image...")
            image = Image.open(image_path).convert("RGB")
            inputs = self.blip_processor(images=image, return_tensors="pt").to(self.blip_model.device)

            print("Generating dense caption...")
            caption_ids = self.blip_model.generate(
                **inputs, max_length=100, num_beams=5, repetition_penalty=1.2
            )
            caption = self.blip_processor.decode(caption_ids[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print("Error generating dense caption with BLIP-2:", e)
            return "Could not generate caption."

    def evaluate_pointwise(self, question, answer, rubric=None, image_path=None):
        """
        Evaluate a single response, optionally using an image.
        """
        rubric = rubric or self.default_rubric
        image_description = ""
        if image_path:
            print("Generating caption for the provided image...")
            image_description = self.generate_dense_caption(image_path)

        prompt = f"""
        You are an expert evaluator. Below is the question, the answer, and the evaluation rubric. 
        {f'Additionally, the image description is provided: {image_description}' if image_path else ''}
        
        Question: {question}
        Answer: {answer}

        Evaluation Rubric:
        {rubric}

        Your response should include:
        - A score (0-10) for each criterion with an explanation.
        - Additional feedback for the answer.
        """
        return self._call_openai(prompt)

    def evaluate_pairwise(self, question, answer_1, answer_2, rubric=None, image_path=None):
        """
        Compare two responses, optionally using an image.
        """
        rubric = rubric or self.default_rubric
        image_description = ""
        if image_path:
            print("Generating caption for the provided image...")
            image_description = self.generate_dense_caption(image_path)

        prompt = f"""
        You are a highly skilled evaluator. Below is a question and two responses. 
        {f'Additionally, the image description is provided: {image_description}' if image_path else ''}

        Question: {question}

        Response 1: {answer_1}
        Response 2: {answer_2}

        Evaluation Rubric:
        {rubric}

        Evaluate both responses based on the rubric, and then:
        1. Provide scores (0-10) for each response based on the rubric.
        2. Explain the strengths and weaknesses of each response.
        3. Clearly state which response is better and why.
        """
        return self._call_openai(prompt)

    def _call_openai(self, prompt):
        """
        Helper method to call the OpenAI API with a prompt.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Use gpt-4 or another preferred model
                messages=[
                    {"role": "system", "content": "You are a professional evaluator of academic and creative responses."},
                    {"role": "user", "content": prompt},
                ]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print("Error during OpenAI evaluation:", e)
            return "Could not evaluate responses."


# Example usage
if __name__ == "__main__":
    judge = LLMJudge()

    # Define the question, responses, and rubric
    question = "Explain the concept of photosynthesis."
    answer_1 = "Photosynthesis is the process where plants use sunlight, carbon dioxide, and water to create glucose and oxygen."
    answer_2 = "Plants make their own food using sunlight, water, and carbon dioxide, producing glucose and oxygen as a result."

    # Case 1: Pointwise evaluation with an image
    print("\nCase 1: Pointwise evaluation with an image")
    print(judge.evaluate_pointwise(question, answer_1, image_path="path_to_image.jpg"))

    # Case 2: Pointwise evaluation without an image
    print("\nCase 2: Pointwise evaluation without an image")
    print(judge.evaluate_pointwise(question, answer_1))

    # Case 3: Pairwise evaluation with an image
    print("\nCase 3: Pairwise evaluation with an image")
    print(judge.evaluate_pairwise(question, answer_1, answer_2, image_path="path_to_image.jpg"))

    # Case 4: Pairwise evaluation without an image
    print("\nCase 4: Pairwise evaluation without an image")
    print(judge.evaluate_pairwise(question, answer_1, answer_2))

