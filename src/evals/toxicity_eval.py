# Evaluate model on hate speech generation

from typing import Dict
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_package_available

if is_package_available("accelerate"):
    from datetime import timedelta
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

def evaluate_llm(config: Dict):
    """ Evaluate the perfromance of the LLM on hate speech generation"""
    
    task = "toxigen"
    evaluation_tracker = EvaluationTracker(
        output_dir=config['log_dir'],
        save_details=True,
        push_to_hub=False,
        use_wandb=True,
        hub_results_org=config['hf_username'], 
    )
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory=None,
    )
    model_config = VLLMModelConfig(
        model_name=config['model_path'],
        dtype="float16",
    )
    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )
    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()
