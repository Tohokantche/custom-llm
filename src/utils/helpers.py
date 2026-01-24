"""
Module docstring: This module's contains all supportive functions use in the app.
"""

# --- Imports ---
from typing import Dict, Union, List
from matplotlib import pyplot as plt
import numpy as np
import time
import torch
from transformers import TextStreamer
from pathlib import Path
from datasets import Dataset
import yaml
import glob
import logging
import subprocess
import os
import random
import sys
import re

logger = logging.getLogger(__name__)

# --- Helper Methods/Functions ---
# Place all supporting functions here.

class MemStats:
    def __init__(self):
        self.get_initial_mem_stats()

    def get_initial_mem_stats(self) -> None:
        """Return memory statistics before start model training"""
        gpu_stats = torch.cuda.get_device_properties(0)
        self.start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        self.max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {self.max_memory} GB.")
        logger.info(f"{self.start_gpu_memory} GB of memory reserved.")

    def get_final_mem_stats(self, trainer_stats) -> None:
        """ Show memory consuption during training"""
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - self.start_gpu_memory, 3)
        used_percentage = round(used_memory / self.max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / self.max_memory * 100, 3)
        logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        logger.info(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        logger.info(f"Peak reserved memory = {used_memory} GB.")
        logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
        logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


def execute_command(command: List[str], timeout: int) -> Union[subprocess.CompletedProcess[str], None]:
    """ Method for executing a shell command """
    result = None
    try:
        logger.info(f"Running : {command}")
        result = subprocess.run(
        command,
        timeout=timeout,
        check=True,
        capture_output=True,
        text=True 
        )
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out after {e.timeout} seconds.")
        logger.error(f"Captured stdout: {e.stdout}")
        logger.error(f"Captured stderr: {e.stderr}")
    except FileNotFoundError as e:
        logger.error(f"Error: Command '{command[0]}' not found - {e}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Process failed with exit code {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return result


def test_inference(tokenizer, model, querry_msg):
    """Simple inference method for quick model inspection"""
    messages = [
    {"role": "user", "content": querry_msg},]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    response = model.generate(input_ids = inputs, streamer = text_streamer,
                    max_new_tokens = 256, temperature = 0.1)
    return response.split("<|eot")[0] if "<|eot" in response else response


def saved_dataset(dataset: Dataset, config:Dict):
    # Replace 'your_user_name/your_dataset_name' with your target repo name
    dataset.push_to_hub(f"{config['hf_username']}/dataset")


def saved_model(model, tokenizer, config:Dict) -> None:
    """Save trained model and/or push it on hub."""

    checkpoint_path = Path(config['checkpoint_dir'])
    if not checkpoint_path.is_dir():
        os.makedirs(config['checkpoint_dir'])
    if ("vllm" or "GGUF") not in config['saving_format'] :
        raise ValueError("Saving format should be in : ['vllm', 'GGUF'] ")

    model.save_pretrained(os.path.join(config['checkpoint_dir'], "ft_lora_adapter"))
    tokenizer.save_pretrained(os.path.join(config['checkpoint_dir'], "ft_lora_adapter"))

    if "vllm" in config['saving_format']:
        model.save_pretrained_merged(os.path.join(config['checkpoint_dir'], "ft_model_16bit_vllm"), 
                                     tokenizer, save_method = "merged_16bit",)
        logger.info(f"Model saved locally to : {os.path.join(config['checkpoint_dir'], "ft_model_16bit_vllm")} ")
        if config['saving_format']['push_hub']:
            model.push_to_hub_merged(f"{config['hf_username']}/ft_model_16bit_vllm", tokenizer, save_method = "merged_16bit", 
                                     token = config['hf_token'])
            logger.info(f"Model push to HuggingFace hub at : {f"{config['hf_username']}/ft_model_16bit_vllm"}")

    if "GGUF" in config['saving_format']:
        model.save_pretrained_gguf(os.path.join(config['checkpoint_dir'], "ft_model_f16_GGUF"), 
                                   tokenizer, quantization_method = "f16")
        logger.info(f"Model saved locally to : {os.path.join(config['checkpoint_dir'], "ft_model_f16_GGUF")} ")
        if config['saving_format']['push_hub']:
            model.push_to_hub_gguf(f"{config['hf_username']}/ft_model_f16_GGUF", tokenizer, quantization_method = "f16", 
                                   token = config['hf_token'])
            logger.info(f"Model push to HuggingFace hub at : {f"{config['hf_username']}/ft_model_f16_GGUF"}")


def get_latest_file(directory_path, extension='*.txt', 
                    pattern :re.Pattern=None) -> Union[str, None]:
    """
    Finds the most recently modified file in a directory matching an extension.
    
    Args:
        directory_path (str): The path to the directory.
        extension (str): File extension pattern (e.g., '*.csv', or '*' for all files).
        
    Returns:
        str: The path to the latest file.
    """
    search_pattern = os.path.join(directory_path, extension)
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    if pattern:
       return latest_file if pattern.search(latest_file) else None
    return latest_file


def write_config(config_to_write: Dict, config_file_name: str):
    """Write configuration to a YAML file."""

    with open(config_file_name, 'w') as file:
        yaml.dump(config_to_write, file, sort_keys=False)


def load_config(config_path: str ='config.yaml') -> Union[Dict, None]:
    """Loads configuration from a YAML file."""
    
    try:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
        return config
    except FileNotFoundError:
        logger.error(f"Error: The configuration file '{config_path}' was not found.")
        return None
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file: {exc}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None
    
def set_seed(seed_value=42) -> None:
    """Set seeds for reproducibility across Python, NumPy, and PyTorch."""

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    logger.info(f"App random seed set to {seed_value} for reproducibility.")


# def benchmark_eval():
#     # !lighteval vllm  "model_name=/content/Qwen3-0.6B" mmlu_pro
#     command = ["lighteval","vllm", '"model_name=/content/Qwen3-0.6B"', "ifeval" ]
#     result = execute_command(command)

def start_vllm_server(model_name: str ="OPENAI/gpt-oss", port=8000):
    """Starts the vLLM OpenAI-compatible server in a subprocess."""
    logger.info(f"Starting vLLM server for model: {model_name} on port {port}...")
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port)
    ]
    # Start the server as a non-blocking child process
    process = subprocess.Popen(command)
    time.sleep(7) 
    logger.info("vLLM server started.")
    return process

def stop_vllm_server(process):
    """Stops the vLLM server process."""
    logger.info("Stopping vLLM server...")
    try:
        process.terminate()
        process.wait() 
        logger.info("vLLM server stopped.")
    except Exception as e:
        logger.error(f"Error stopping server: {e}")

def install_package(package) -> None:
    """Installs a specific Python package using pip as a subprocess."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    

def plot_loss(log_history, config:Dict) -> None:
    """Plot manually the history of the train and validation loss"""
    plot_id = 0
    saving_path = f"./logs/sft_loss_{plot_id}.png"
    while os.path.exists(saving_path):
        saving_path = f"./logs/sft_loss_{plot_id}.png"
        plot_id+=1
    train_loss_logs = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
    eval_loss_logs = [log for log in log_history if 'eval_loss' in log]

    train_steps = [log['step'] for log in train_loss_logs]
    train_losses = [log['loss'] for log in train_loss_logs]

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label='Training Loss', marker='o')

    if eval_loss_logs:
        eval_steps = [log['step'] for log in eval_loss_logs]
        eval_losses = [log['eval_loss'] for log in eval_loss_logs]
        plt.plot(eval_steps, eval_losses, label='Validation Loss', marker='x')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(saving_path , dpi=300, bbox_inches='tight', transparent=True)
    logger.info(f"Saved plot in : {saving_path}")
    plt.show()
        

    __all__ = [MemStats,
               execute_command,
               start_vllm_server,
               test_inference,
               saved_model,
               get_latest_file,
               write_config,
               load_config,
               set_seed,
               start_vllm_server,
               stop_vllm_server,
               install_package,
               plot_loss,        
        ]