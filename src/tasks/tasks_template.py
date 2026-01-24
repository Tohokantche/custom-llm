# A simple template to define a custom task and its corresponding metric

from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.metrics.metrics_sample import F1_score
import numpy as np

#standard_metric = Metrics.ACCURACY

class custom_sample_level_fn(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs):
        response = model_response.final_text[0]
        return {"accuracy": response == doc.choices[doc.gold_index], "F1_score": 
                F1_score().compute(doc, model_response, **kwargs)}

custom_metric_1 = SampleLevelMetric(
    metric_name="my_custom_metric_name",
    higher_is_better=True,
    category="accuracy",
    sample_level_fn= custom_sample_level_fn(),  
    corpus_level_fn=np.mean, 
)

def prompt_fn(line: dict, task_name: str):
    """Defines how to go from a dataset line to a doc object.
    Follow examples in src/lighteval/tasks/default_prompts.py.
    """
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["gold"],
    )

custom_task_1 = LightevalTaskConfig(
    name="my_other_task",
    prompt_function=prompt_fn, 
    hf_repo="your_dataset_repo_on_hf",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    metrics=[custom_metric_1], 
    generation_size=256,
    stop_sequence=["\n", "Question:"],
)
TASKS_TABLE = [custom_task_1]

