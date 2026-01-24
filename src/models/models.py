from unsloth import FastLanguageModel
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def get_model(config: Dict):
    model, tokenizer = FastLanguageModel.from_pretrained(
        **config
        )
    return model, tokenizer

def get_peft_model(plain_model, config: Dict):
    if hasattr(plain_model, "enable_input_require_grads"):
        plain_model.enable_input_require_grads()
        logger.info("****Enabling  computational graph****")
    model = FastLanguageModel.get_peft_model(
        plain_model,
        **config
        )
    return model