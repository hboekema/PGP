from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from vod.eval.prediction.config import PredictionConfig
from vod.eval.prediction.data_classes import Prediction


def compute_metrics(
    predictions: List[Dict[str, Any]], ground_truths, config: PredictionConfig
) -> Dict[str, Any]:
    """
    Computes metrics from a set of predictions.
    :param predictions: List of prediction JSON objects.
    :param helper: Instance of PredictHelper that wraps the VOD val set.
    :param config: Config file.
    :return: Metrics. Nested dictionary where keys are metric names and value is a dictionary
        mapping the Aggregator name to the results.
    """
    n_preds = len(predictions)
    containers = {
        metric.name: np.zeros((n_preds, metric.shape)) for metric in config.metrics
    }
    for i, (prediction_str, ground_truth) in enumerate(zip(predictions, ground_truths)):
        prediction = Prediction.deserialize(prediction_str)
        for metric in config.metrics:
            containers[metric.name][i] = metric(ground_truth, prediction)
    aggregations: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for metric in config.metrics:
        for agg in metric.aggregators:
            aggregations[metric.name][agg.name] = agg(containers[metric.name])
    return aggregations
