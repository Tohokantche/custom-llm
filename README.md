# Efficiently train, evaluate and serve a customised LLM
## Use cases

> This repo is an automated, easy-to-customize and end-to-end LLMs post-training pipeline that consists of the following components : 
* :white_check_mark: Data generation and pre-processing (<a href="https://github.com/meta-llama/synthetic-data-kit">Synthetic Data Kit</a>),
* :white_check_mark: Model fine-tuning (<a href="https://huggingface.co/docs/trl/en/index">TRL</a>),
* :white_check_mark: Model evaluation (<a href="https://github.com/huggingface/lighteval">LightEval</a> of <a href="https://huggingface.co/">HugginFace</a>), 
* :white_check_mark: Model inference and serving (<a href="https://github.com/vllm-project/vllm">vLLM</a> and <a href="https://github.com/open-webui/open-webui">Open-WebUI</a> using <a href="https://github.com/docker">Docker</a>).
  
> Pipeline configurations are stored in a YAML file (<a href="configs/default_config.yaml">configs/default_config.yaml </a>) to enable easy hyperparameter tuning, model changes (i.e., gpt-oss, Qwen, DeepSeek, Gemma, LLaMA ...), and data source updates.


## Getting started
After cloning this repo, install dependencies on a machine with Nvidia-GPU
```yaml
# 1. Clone repository 
git clone https://github.com/Tohokantche/custom-llm.git
cd custom-llm

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# 3. Install requirements
pip install -r requirements.txt
```

End-to-end data generation, training and evaluation of your LLM with the default configuration
```yaml
# End-to-end training on 1 GPU using the default shell script (for linux environment)
chmod +x run_train.sh
./run_training.sh

# End-to-end training on multiple GPUs using the default shell script (for linux environment)
chmod +x run_distributed_train.sh
./run_distributed_training.sh

# End-to-end training on 1 GPU using the Python script and your configuration file (for windows and linux environment)
python scripts/e2e_training.py  --config configs/default_config.yaml

# End-to-end training on multiple GPUs (i.e., 5) using the Python script and your configuration (for windows and linux environment)
torchrun --nproc_per_node 5 scripts/e2e_training.py  --config configs/default_config.yaml
```

Deployment and inference of your trained LLM using <a href="https://docs.docker.com/desktop/">Docker</a>
```yaml
# Deploy your trained LLM using vLLM as inference engine and open-webui for the user interface
cd configs && docker-compose up -d

```
## GPU memory and data requirement for efficient training
![VRAM requirements](src/assets/Fine-tuning-requirements-on-Unsloth.jpg "VRAM requirements")

## File structure
```text
.
├── configs
│   ├── default_config.yaml
│   ├── docker-compose.yml
│   └── synthetic_data_kit_config.yaml
├── LICENSE
├── README.md
├── requirements.txt
├── run_distributed_training.sh
├── run_training.sh
├── scripts
│   ├── __pycache__
│   │   └── main.cpython-39.pyc
│   ├── e2e_training.py
│   ├── evaluate_model.py
│   ├── generate_data.py
│   └── train_model.py
└── src
    ├── __init__.py
    ├── __pycache__
    │   └── __init__.cpython-39.pyc
    ├── assets
    │   └── Fine-tuning-requirements-on-Unsloth.jpg
    ├── checkpoints
    ├── data
    │   ├── __init__.py
    │   ├── dataloader.py
    │   └── dataset.py
    ├── evals
    │   ├── __init__.py
    │   └── evaluate.py
    ├── logs
    ├── models
    │   ├── __init__.py
    │   └── models.py
    ├── notebooks
    │   └── Simple_Inference.ipynb
    ├── tasks
    │   ├── __init__.py
    │   └── tasks_template.py
    ├── train.py
    └── utils
        ├── __init__.py
        ├── __pycache__
        │   ├── __init__.cpython-313.pyc
        │   ├── __init__.cpython-39.pyc
        │   ├── helpers.cpython-313.pyc
        │   └── helpers.cpython-39.pyc
        ├── helpers.py
        └── metrics.py
```


## Acknowledgments 

This repo is inspired by <a href="https://huggingface.co/">HugginFace</a> and  <a href="https://github.com/unslothai/unsloth">Unsloth AI</a>. They are doing amazing jobs for an efficient LLMs post-training. Make sure to check them out and buy them a coffee!

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute it in either commercial or academic projects under the terms of this license.






