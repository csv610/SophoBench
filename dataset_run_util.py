import argparse
from typing import NamedTuple, Type
from task_list import Tasks

class ExecutionArgs(NamedTuple):
    nsamples: int | None
    models: dict
    sys_config: dict
    task: str

def get_execution_args(description: str = "Process dataset") -> ExecutionArgs:
    """
    Common argument parser for benchmark scripts.
    
    Args:
        description: Custom description for the benchmark.
        
    Returns:
        ExecutionArgs containing parsed arguments
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-n", 
        "--nsamples", 
        type=int, 
        default=None,
        help="Number of random samples to process per subset and split. If not provided, process all samples."
    )
    parser.add_argument(
        "--text_model", 
        type=str, 
        default="llama3.2",
        help="Name of the text model to use for processing."
    )
    parser.add_argument(
        "--vision_model", 
        type=str, 
        default="llama3.2-vision",
        help="Name of the vision model to use for processing."
    )
    parser.add_argument(
        "-t",
        "--max_threads",
        type=int,
        default=None,
        help="Maximum number of threads to use for processing. If not provided, uses system default."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=100,
        help="Timeout in seconds for model response. Default is 100 seconds."
    )
    parser.add_argument(
        '-task', 
        type=str, 
        default=Tasks.GENERATE_ANSWERS, 
        choices=list(Tasks.VALID_TASKS), 
        help='Specify the task to run (default: generate_answer).'
    )
    args = parser.parse_args()

    models = {
        'text': args.text_model,
        'vision': args.vision_model
    }
    
    sys_config = {
        'max_threads': args.max_threads,
        'response_timeout': args.timeout
    }

    return ExecutionArgs(
        nsamples=args.nsamples,
        models=models,
        sys_config=sys_config,
        task=args.task
    )

def run_dataset(execute_class: Type) -> None:
    """
    Execute a benchmark using the provided benchmark class.
    
    Args:
        execute_class: The benchmark class to instantiate (e.g., Ai2arc_Query, GPQA_Query)
    """
    description = f"Executing {execute_class.__name__} dataset"
    args = get_execution_args(description=description)

    bench = execute_class(task=args.task, models=args.models, sys_config=args.sys_config)
   
    bench.run(nsamples=args.nsamples)
