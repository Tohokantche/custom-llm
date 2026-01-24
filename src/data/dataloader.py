from logging import config
from src.utils.helpers import execute_command, get_latest_file
from datasets import Dataset
from typing import Dict
import pandas as pd
import logging
import os
import re

logger = logging.getLogger(__name__)

class ProcessDataset:
    def __init__(self, generator, filenames, tokenizer, config:Dict):
        self.generator = generator
        self.filenames = filenames
        self.tokenizer = tokenizer
        self.config = config

    # Generate questions answers pairs 
    def qa_pairs_transform(self) -> Dataset:

        qa_pairs_filenames = []
        filter_qa_pairs_filenames = []
        format_qa_pairs_filenames = []

        # To-DO: parallelise this step with respect to the ressources available
        for filename in self.filenames:
            # Generate QA pairs from files
            command = ["synthetic-data-kit", "-c", self.config['synthetic_data_kit_config'], "create", filename, 
                       "--num-pairs", str(self.config['data_transform_config']['num_pairs']), "--type", 
                       self.config['data_transform_config']['ft_data_format']]
            result_1 = execute_command(command, int(self.config['app_command_timeout']))
            if not(("saved to " in result_1.stdout) and result_1):
                raise RuntimeError("Unable to tranform the data into QA pairs format.")
            qa_pairs_filenames.append(result_1.stdout.split("saved to ")[1].strip())
            
            if not os.path.exists(qa_pairs_filenames[-1]):
                qa_pairs_filenames[-1] = get_latest_file(directory_path="./data/generated", extension="qa_pairs.json")
            if not qa_pairs_filenames[-1]:
                raise RuntimeError("Unable to locate the QA pairs files")
            logger.info(f"Current QA pairs file: {qa_pairs_filenames[-1]}")

            # Filter the data
            command = ["synthetic-data-kit", "-c", self.config['synthetic_data_kit_config'], "curate", 
                    f"--threshold {self.config['data_transform_config']['filter_threshold']}",
                qa_pairs_filenames[-1]]
            result_2 = execute_command(command, int(self.config['app_command_timeout']))

            if not(("saved to " in result_2.stdout) and result_2):
                raise RuntimeError("Unable to filter the generated data.")
            filter_qa_pairs_filenames.append(result_2.stdout.split("saved to ")[1].strip())
            
            if not os.path.exists(filter_qa_pairs_filenames[-1]):
                filter_qa_pairs_filenames[-1] = get_latest_file(directory_path="./data/cleaned", extension="qa_pairs_cleaned.json")
            if not filter_qa_pairs_filenames[-1]:
                raise RuntimeError("Unable to locate the filtered files")
            logger.info(f"Current filtered QA pairs file: {filter_qa_pairs_filenames[-1]}")
            
            # Format the data
            command = ["synthetic-data-kit", "-c", self.config['synthetic_data_kit_config'], "save-as", filter_qa_pairs_filenames[-1], "-f", "ft"]
            result_3 = execute_command(command, int(self.config['app_command_timeout']))

            if not(("saved to " in result_3.stdout) and result_3):
                raise RuntimeError("Unable to tranform QA data into SFT format.")
            format_qa_pairs_filenames.append(result_3.stdout.split("saved to ")[1].strip())
            
            if not os.path.exists(format_qa_pairs_filenames[-1]):
                format_qa_pairs_filenames[-1] = get_latest_file(directory_path="./data/final", extension="qa_pairs_ft.json")
            if not format_qa_pairs_filenames[-1]:
                raise RuntimeError("Unable to locate the SFT formated QA files.")
            logger.info(f"Current SFT formated QA pairs file: {format_qa_pairs_filenames[-1]}")
            
        conversations = pd.concat([
            pd.read_json(name) for name in format_qa_pairs_filenames ]).reset_index(drop = True)
        dataset = Dataset.from_pandas(conversations)
        logger.info(f" 'QA Sample 0:', {dataset[0]}")
        return dataset
    
    def formatting_prompts_func(self, examples):
        """  Defining a chat template """

        convos = examples["messages"]
        texts = [self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    
    def chat_template_transform(self, task_dataset: Dataset) -> Dataset:
        """  Applying a chat template """

        dataset = task_dataset.map(self.formatting_prompts_func, batched = True,)
        logger.info(f" 'Formatted QA Sample 0:', {dataset[0]}") 
        if self.generator:
            self.generator.cleanup()
        return dataset


