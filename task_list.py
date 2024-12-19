import logging

class Tasks:
    GENERATE_ANSWERS = "generate_answers"
    SAVE_QUESTIONS   = "save_questions"
    VALID_TASKS = frozenset({GENERATE_ANSWERS, SAVE_QUESTIONS})

def generate_answer(obj, model_input):
    model = obj.get_model()
    if model is None:
       logging.error("Invalid model selection or initialization failed")  
       return "Invalid model selection or initialization failed"
    
    response = model.get_response(model_input)
    return response

def execute_task(obj, row):
    task  = obj.get_assigned_task()
    dname = obj.get_dataset_name()

    model_input = obj.extract_data(row)
    if model_input is None:
       error_msg = "Failed to extract data from the row"
       logging.error(error_msg)
       return error_msg

    if task == Tasks.GENERATE_ANSWERS:
       return generate_answer(obj, model_input)
        
    if task == Tasks.SAVE_QUESTIONS:
       model_input['answer'] = obj.get_correct_answer(row)
       return model_input

    return "Unknown task"
