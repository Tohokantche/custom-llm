from src.models.models import get_peft_model, get_model
from src.data.dataset import GenDataset
from src.data.dataloader import ProcessDataset
from src.utils.helpers import  load_config, set_seed
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

def generate_data():
  
    gen_dataset = GenDataset(config_data['gen_data_config'])
    try:
        chunch_filenames = gen_dataset.gen_chunch_doc()
    except RuntimeError as error_message:
        logger.error(f"Unable to generate the data : {error_message}")
        return
    
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
    #final_dataset = process_data.chat_template_transform(qa_dataset)
    final_dataset = qa_dataset.shuffle(seed=config_data['app_seed'])
    split_dataset = final_dataset.train_test_split(
        test_size=config_data['gen_data_config']['data_transform_config']['split_ratio'])
    split_dataset.save_to_disk(os.path.join("./src/data/",config_data['app_seed']))
    

if __name__== "__main__":
    generate_data()