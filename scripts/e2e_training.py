# TO-DO Add model merging script to revover potential performance loss on standard benchmark
from src.train import train
from src.utils.helpers import load_config, set_seed
import argparse
import logging
import wandb
import os
import sys

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="An automated and efficient domain specific fine-tuning script for Chat LLMs.")
parser.add_argument("--config", required= True, default="./configs/default_config.yaml", 
                    help="The fine-tuning configuration file.")
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
    logger.info("Starting the fine-tuning process.")
    train(config_data)
    logger.info("Completed the fine-tuning process.")
