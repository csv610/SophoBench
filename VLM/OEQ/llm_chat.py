from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

import time

class LLMChat:
    def __init__(self, model_name):
        self.model = OllamaLLM(model=model_name)
        self.mcq_prompt_template = """
            Question: {question}
            Choices:  {choices}

            Answer: Let's carefully consider the subject matter and each of the provided choices before determining the most accurate answer. After evaluating the options, the answer is:
        """
        self.explain_prompt_template = ChatPromptTemplate.from_template("""
            Question: {question}

            Let's think step by step and provide a brief explanation.
        """)

        self.default_prompt_template = ChatPromptTemplate.from_template("""
            Question: {question}
        """)

    def count_words(self, sentence: str) -> int:
        """Counts the number of words in a sentence."""
        words = sentence.split()
        return len(words)

    def get_mcq_answer(self, question, choices):
        # Dynamically create the choices string
        choices_str = "\n".join([f"{chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])

        num_input_words = self.count_words(question) + sum(self.count_words(choice) for choice in choices)
        
        prompt = self.mcq_prompt_template.format(
            question=question,
            choices=choices_str
        )
        t0 = time.time()
        response = (ChatPromptTemplate.from_template(prompt) | self.model).invoke({})

        t1 = time.time()

        num_output_words = self.count_words(response)
        elapsed_time = t1 - t0
        answer, explanation = response.split("\n", 1)
        retval = {
               "num_input_words": num_input_words,
               "num_output_words": num_output_words,
               "response_time": elapsed_time,
               "response": answer,
               "explanation": explanation
        }
        
        return answer, explanation

    def get_answer(self, question, cot_prompt=False):

        num_input_words = self.count_words(question)

        t0 = time.time()

        if cot_prompt:
           response = (self.explain_prompt_template | self.model).invoke({ "question": question })
        else:
           response = (self.default_prompt_template | self.model).invoke({ "question": question })

        t1 = time.time()

        num_output_words = self.count_words(response)
        elapsed_time = t1 - t0
        retval = {
               "num_input_words": num_input_words,
               "num_output_words": num_output_words,
               "response_time": elapsed_time,
               "response": response,
               "explanation": []
        }
        
        return retval
    




