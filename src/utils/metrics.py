# Defining a custom metric

from aenum import extend_enum
import numpy as np
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.models.model_output import ModelResponse
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.metrics_sample import (
    BLEU,
    BLEURT,
    MRR,
    ROUGE,
    AccGoldLikelihood,
    AvgAtN,
    BertScore,
    ExactMatches,
    Extractiveness,
    F1_score,
    Faithfulness,
    GPassAtK,
    JudgeLLMSimpleQA,
    LoglikelihoodAcc,
    MajAtN,
    PassAtK,
    Recall,
    StringDistance,
    JudgeLLM,
)

class custom_metric_1_fn(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs):
        response = model_response.final_text[0]
        return {"custom_accuracy": response == doc.choices[doc.gold_index], "response_lengh": len(response) , 
                "F1_score": F1_score().compute(doc, model_response, **kwargs)}

class custom_metric_2_fn(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs):
        response = model_response.final_text[0]
        return {"custom_accuracy": response == doc.choices[doc.gold_index]}


custom_metric_1 =SampleLevelMetric(
    metric_name="custom_accuracy",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=custom_metric_1_fn(),  # compute score for one sample
    corpus_level_fn=np.mean,  # aggregate the sample metrics
)

custom_metric_2 = SampleLevelMetricGrouping(
    metric_name=["custom_accuracy", "response_length", "F1_score"],
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=custom_metric_2_fn(),
    corpus_level_fn={
        "custom_accuracy": np.mean,
        "response_length": np.mean,
        "F1_score": np.mean,
    },
    higher_is_better = {
        "custom_accuracy": True,
        "response_length": False,
        "F1_score": True,
    },
)
extend_enum(Metrics, "CUSTOM_ACCURACY_1", custom_metric_1)
extend_enum(Metrics, "CUSTOM_ACCURACY_2", custom_metric_2)

if __name__ == "__main__":
    print("Imported metric")
    print(Metrics.CUSTOM_ACCURACY_1)
    print(Metrics.CUSTOM_ACCURACY_2)