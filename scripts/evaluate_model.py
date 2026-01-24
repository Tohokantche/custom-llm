
from src.utils.helpers import load_config, set_seed
from src.evals.evaluate import evaluate_llm
import argparse
import logging
import wandb
import os
from lighteval.tasks.registry import Registry

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="An automated and memory-efficient domain specific fine-tuning script for open-source LLMs.")
parser.add_argument("--config", required= True, default="./configs/default_config.yaml", help="The fine-tuning configuration file.")
args = parser.parse_args()
config_data = load_config(args.config)
if not config_data:
    exit(1)
set_seed(config_data['app_seed'])
os.environ["WANDB_PROJECT"] = config_data['wandb_config']['wandb_project'] 
os.environ["WANDB_API_KEY"] = config_data['wandb_config']['wandb_api_key']
wandb.init(entity=config_data['wandb_config']['entity'], project=os.environ["WANDB_PROJECT"])

def evaluate_model():
    
    all_tasks = Registry.load_all_task_configs(load_multilingual=True)
    temp_config = config_data['hf_hub_config'].update(config_data['saved_model_config'])

    # Standard task evaluation pipeline
    for task_name in config_data['eval_config']['std_tasks']:
        try:
            evaluate_llm({"task": all_tasks[task_name].name, "task_path": None,
                           "model_path": os.path.join(config_data['checkpoint_dir'], "ft_model_16bit_vllm")}.update(temp_config))
        except KeyError as error_message:
            logger.error(f"Cannot find the task : {error_message}")
        except Exception as error_message:
            logger.error(f"An unexpected error ocuurs : {error_message}")
    
     # Customized task evaluation pipeline
    for id, task_name in enumerate(config_data['eval_config']['custom_tasks']['task_name']):
        try:
            evaluate_llm({"task": task_name, "task_path":config_data['eval_config']['custom_tasks']['task_path'][id],
                          "model_path": os.path.join(config_data['checkpoint_dir'], "ft_model_16bit_vllm")}.update(temp_config))
        except Exception as error_message:
            logger.error(f"An unexpected error ocuurs : {error_message}")


if __name__ == "__main__":
    evaluate_model()