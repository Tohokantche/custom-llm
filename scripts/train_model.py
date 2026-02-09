
from src.models.models import get_peft_model, get_model
from src.data.dataloader import ProcessDataset
from src.utils.helpers import MemStats, load_config, saved_model, set_seed, GradientGuard, clean_up_ddp
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
import torch
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

    ###############################  DDP bootstrap ###########################################
    LOCAL_RANK  = int(os.environ.get("LOCAL_RANK", 0))
    RANK        = int(os.environ.get("RANK", 0))
    WORLD_SIZE  = int(os.environ.get("WORLD_SIZE", 1))
    IS_DIST     = WORLD_SIZE > 1

    if IS_DIST:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(LOCAL_RANK)
    device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
    logger.info(f"RANK={RANK} LOCAL_RANK={LOCAL_RANK} WORLD_SIZE={WORLD_SIZE} device={device}")
    
    ###############################  Training #################################################
    
    # Data-parallel load: each process places the model on its own GPU.
    device_map = {"": f"cuda:{LOCAL_RANK}"} if torch.cuda.is_available() else None
    config_data['unsloth_model_config']['device_map'] = device_map
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
        args = SFTConfig(**config_data['hf_sft_config']['sfttrainer']),
        callbacks = [GradientGuard()],
        )
    trainer_stats = trainer.train(**config_data['hf_sft_config']['train'])
    mem_stats.get_final_mem_stats(trainer_stats)
    try:
        saved_model(model, tokenizer, config_data['saved_model_config'])
    except ValueError as error_message:
        logger.error(f"Can't save the model : {error_message}")
        clean_up_ddp(IS_DIST)


if __name__== "__main__":
    train_model()