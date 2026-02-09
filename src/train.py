from src.models.models import get_peft_model, get_model
from src.data.dataset import GenDataset
from src.data.dataloader import ProcessDataset
from src.utils.helpers import MemStats, saved_model, GradientGuard, clean_up_ddp
from typing import Dict
from src.evals.evaluate import evaluate_llm
from lighteval.tasks.registry import Registry
import logging
from trl import SFTTrainer, SFTConfig
import torch
import os

logger = logging.getLogger(__name__)

def train(config_data: Dict):

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


    ########################## DATA GENERATION FROM SEED DOCUMENT ##############################
    gen_dataset = GenDataset(config_data['gen_data_config'])
    try:
        chunch_filenames = gen_dataset.gen_chunch_doc()
    except RuntimeError as error_message:
        logger.error(f"Unable to generate the data : {error_message}")
        clean_up_ddp(IS_DIST)
        return
    
    ########################## DATA FILTERING AND FORMATING ####################################
    
    # Data-parallel load: each process places the model on its own GPU.
    device_map = {"": f"cuda:{LOCAL_RANK}"} if torch.cuda.is_available() else None
    config_data['unsloth_model_config']['device_map'] = device_map
    model, tokenizer = get_model(config_data['unsloth_model_config'])
    model = get_peft_model(model, config_data['unsloth_peft_config'])

    process_data = ProcessDataset(gen_dataset.generator, chunch_filenames, tokenizer,
                                   config_data['gen_data_config'].update(
                                       {"app_command_timeout": config_data['app_command_timeout']}))
    try:
        qa_dataset = process_data.qa_pairs_transform()
    except RuntimeError as error_message:
        logger.error(f"Unable to generate the data : {error_message}")
        clean_up_ddp(IS_DIST)
        return 
    
    final_dataset = process_data.chat_template_transform(qa_dataset)
    final_dataset = final_dataset.shuffle(seed=config_data['app_seed'])
    split_dataset = final_dataset.train_test_split(
        test_size=config_data['gen_data_config']['data_transform_config']['split_ratio'])
    

    ########################## MODEL TRAINING AND SAVING #######################################
    mem_stats = MemStats()
    logger.info(config_data['hf_sft_config'])

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = split_dataset['train'],
        eval_dataset = split_dataset['test'],
        args = SFTConfig(**config_data['hf_sft_config']['sfttrainer']),
        callbacks = [GradientGuard()],
        )
    trainer_stats = trainer.train(**config_data['hf_sft_config']['train'])
    mem_stats.get_final_mem_stats(trainer_stats)

    try:
        saved_model(model, tokenizer, config_data['saved_model_config'].update(config_data['hf_hub_config']))
    except ValueError as error_message:
        logger.error(f"Can't save the model : {error_message}")
        clean_up_ddp(IS_DIST)
        return

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
            clean_up_ddp(IS_DIST)
            
    clean_up_ddp(IS_DIST)
    
    


