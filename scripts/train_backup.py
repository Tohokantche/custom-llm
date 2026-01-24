
from src.models.models import get_peft_model, get_model
from src.data.dataset import GenDataset
from src.data.dataloader import ProcessDataset
from src.utils.helpers import MemStats, load_config, saved_model, set_seed
from src.evals.evaluate import evaluate_llm
from trl import SFTTrainer, SFTConfig
import argparse
import logging
import wandb
import os
import sys
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


def main():
    if not os.path.exists(config_data['saved_model_config']['log_dir']):
        os.makedirs(config_data['saved_model_config']['log_dir'])
    logging.basicConfig(
        level=logging.INFO,
        # {%(pathname)s:%(lineno)d}
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(
                config_data['saved_model_config']['log_dir'], "app.log"), mode='w'), 
            logging.StreamHandler(sys.stdout)         
        ],
        force=True 
    )
    logger.info("Start the fine-tuning process.")

    if config_data:
        ########################## DATA GENERATION FROM SEED DOCUMENT ##############################
        gen_dataset = GenDataset(config_data['gen_data_config'])
        try:
            chunch_filenames = gen_dataset.gen_chunch_doc()
        except RuntimeError as error_message:
            logger.error(f"Unable to generate the data : {error_message}")
            return
        
        ########################## DATA FILTERING AND FORMATING ####################################
        model, tokenizer = get_model(config_data['unsloth_model_config'])
        model = get_peft_model(model, config_data['unsloth_peft_config'])

        process_data = ProcessDataset(gen_dataset.generator, chunch_filenames, tokenizer,
                                       config_data['gen_data_config'].update(
                                       {"app_command_timeout": config_data['app_command_timeout']}))
        try:
            qa_dataset = process_data.qa_pairs_transform()
        except RuntimeError as error_message:
            logger.error(f"Unable to generate the data : {error_message}")
            return
        final_dataset = process_data.chat_template_transform(qa_dataset)

        split_dataset = final_dataset.train_test_split(
            test_size=config_data['gen_data_config']['data_transform_config']['eval_size'])

        ########################## MODEL TRAINING AND SAVING #######################################
        mem_stats = MemStats()
        logger.info(config_data['hf_sft_config'])
        trainer = SFTTrainer(
            model = model,
            processing_class = tokenizer,
            train_dataset = split_dataset['train'],
            eval_dataset = split_dataset['test'],
            args = SFTConfig(**config_data['hf_sft_config']),
            )
        trainer_stats = trainer.train()
        mem_stats.get_final_mem_stats(trainer_stats)
        try:
            saved_model(model, tokenizer, config_data['saved_model_config'])
        except ValueError as error_message:
            logger.error(f"Can't save the model : {error_message}")

        logger.info("Completed the fine-tuning process.")

        ########################## MODEL EVALUATION ################################################
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

