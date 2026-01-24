
from src.models.models import get_peft_model, get_model
from src.data.dataloader import ProcessDataset
from src.utils.helpers import MemStats, load_config, saved_model, set_seed
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
import argparse
import logging
import wandb
import os


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

def train_model():
    
    model, tokenizer = get_model(config_data['unsloth_model_config'])
    model = get_peft_model(model, config_data['unsloth_peft_config'])
    process_data = ProcessDataset(None, None, tokenizer,
                                       config_data['gen_data_config'].update(
                                       {"app_command_timeout": config_data['app_command_timeout']}))
    split_dataset = load_from_disk(os.path.join("./src/data/",config_data['app_seed']))
    train_dataset = process_data.chat_template_transform(split_dataset['train'])
    test_dataset = process_data.chat_template_transform(split_dataset['test'])
    mem_stats = MemStats()
    logger.info(config_data['hf_sft_config'])
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset =  train_dataset,
        eval_dataset = test_dataset,
        args = SFTConfig(**config_data['hf_sft_config']),
        )
    trainer_stats = trainer.train()
    mem_stats.get_final_mem_stats(trainer_stats)
    try:
        saved_model(model, tokenizer, config_data['saved_model_config'])
    except ValueError as error_message:
        logger.error(f"Can't save the model : {error_message}")


if __name__== "__main__":
    train_model()