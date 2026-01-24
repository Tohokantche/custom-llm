from unsloth.dataprep import SyntheticDataKit
from src.utils.helpers import execute_command, get_latest_file
from typing import List, Dict
import logging
import glob
import os

logger = logging.getLogger(__name__)

class GenDataset:
    """ Generate SFT Data from a given user specified file  """
    def __init__(self, config:Dict):
        self.config = config
        self.set_generator()

        files = glob.glob(os.path.join("./data/output", "*.txt"))
        for filename in files:
            try:
                os.remove(filename)
                print(f"Removed: {filename}")
            except OSError as e:
                print(f"Error removing {filename}: {e}")
        
    def set_generator (self):   
        self.generator = SyntheticDataKit.from_pretrained(**self.config['from_pretrained'])  
        self.generator.prepare_qa_generation(**self.config['prepare_qa_generation'])

    def gen_chunch_doc(self) -> List[str]:
        if not GenDataset.is_server_up():
            raise RuntimeError(f"Failed to start the VLLM server for data generation.")
        parse_doc_path = None

        command = ["synthetic-data-kit", "-c", self.config['synthetic_data_kit_config'], "ingest", 
                   f"{self.config['doc_url']}"]
        result = execute_command(command)
        if not(result and ("successfully extracted" in result.stdout)):
            raise RuntimeError(f"Failed to parse file from seed doc at: {self.config['doc_url']}")
        parse_doc_path = result.stdout.split("successfully extracted to ")[1].strip()
        
        # Handling error during parsing
        if not os.path.exists(parse_doc_path):
            parse_doc_path = get_latest_file(directory_path='./data/output', extension='*.txt')
        if not parse_doc_path:
            raise RuntimeError(f"Unable to locate the path of the parsed doc from {self.config['doc_url']}")
        logger.info(f"PARSE_DOC_PATH : {parse_doc_path}")
        filenames = self.generator.chunk_data(parse_doc_path)
        logging.info(f"{len(filenames)}, {filenames[:3]}")
        return filenames
        
    @staticmethod
    def is_server_up() -> bool:
        command = ["synthetic-data-kit", "system-check"]
        result = execute_command(command)
        return True if result and ("VLLM server is running at" in result.stdout) else False
        

    


        
        
        
