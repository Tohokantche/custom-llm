# Efficiently train, evaluate and serve a customised LLM
## Use cases

> This repo is an automated, easy-to-customize and end-to-end LLMs post-training pipeline that consists of the following components : 
* :white_check_mark: Data generation and pre-processing (<a href="https://github.com/meta-llama/synthetic-data-kit">Synthetic Data Kit</a>),
* :white_check_mark: Model fine-tuning (<a href="https://huggingface.co/docs/trl/en/index">TRL</a>),
* :white_check_mark: Model evaluation (<a href="https://github.com/huggingface/lighteval">LightEval</a> from <a href="https://huggingface.co/">HugginFace</a>), 
* :white_check_mark: Model inference and serving (<a href="https://github.com/vllm-project/vllm">vLLM</a> and <a href="https://github.com/open-webui/open-webui">Open-WebUI</a> using <a href="https://github.com/docker">Docker</a>).
  
> Pipeline configurations are stored in a YAML file (<a href="configs/default_config.yaml">configs/default_config.yaml </a>) to enable easy hyperparameter tuning, model changes (i.e., gpt-oss, Qwen, DeepSeek, Gemma, LLaMA ...), and data source updates.


## Getting started
After cloning this repo, install dependencies
```yaml
# 1. Clone repository 
git clone https://github.com/Tohokantche/custom-llm.git
cd custom-llm

# 2. Install requirements
pip install -r requirements.txt
```

Generate data, train and evaluate your LLMs model with the default configuration
```yaml
# Train on 1 GPU using the default shell script (for linux environment)
chmod +x run_train.sh
./run_training.sh

# Train on multiple GPUs using the default shell script (for linux environment)
chmod +x run_distributed_train.sh
./run_distributed_training.sh

# Train on 1 GPU using the Python script and your configuration file (for windows and linux environment)
python scripts/main.py  --config configs/default_config.yaml

# Train on multiple GPUs (i.e., 5) using the Python script and your configuration (for windows and linux environment)
torchrun --nproc_per_node 5 scripts/main.py  --config configs/default_config.yaml
```

Deployment and inference using Docker
```yaml
# Deploy your trained LLM using vLLM as inference engine and open-webui for the user interface
cd configs && docker-compose up -d

```
## GPU memory requirement for training
![VRAM requirements](src/assets/Fine-tuning-requirements-on-Unsloth.jpg "VRAM requirements")

## Acknowledgments 

This repo is inspired by <a href="https://huggingface.co/">HugginFace</a> and  <a href="https://github.com/unslothai/unsloth">Unsloth AI</a>. They are doing an amazing job for efficient LLMs post-training. Make sure to check them out and buy them a coffee!

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute it in either commercial or academic projects under the terms of this license.






