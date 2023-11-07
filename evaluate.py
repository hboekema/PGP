import argparse
import collections
import json
import os
import random

import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from vod.prediction.helper import PredictHelper

from train_eval.evaluator import Evaluator
from train_eval.initialization import (get_specific_args, initialize_dataset,
                                       initialize_prediction_model)
from train_eval.visualizer import GeneralVisualizer

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config", help="Config file with dataset parameters", required=True
)
parser.add_argument("-r", "--data_root", help="Root directory with data", required=True)
parser.add_argument("-d", "--data_dir", help="Directory to extract data", required=True)
parser.add_argument(
    "-m", "--metrics_config", help="Config file with metric parameters", required=True
)
parser.add_argument(
    "-o", "--output_dir", help="Directory to save results", required=True
)
parser.add_argument(
    "-w",
    "--checkpoint",
    help="Path to pre-trained or intermediate checkpoint",
    required=True,
)
parser.add_argument("-s", "--seed", help="Random seed", type=int, default=0)
args = parser.parse_args()

mode = "display"

# Make directories
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
if not os.path.isdir(os.path.join(args.output_dir, "results")):
    os.mkdir(os.path.join(args.output_dir, "results"))
if not os.path.isdir(os.path.join(args.output_dir, "vis")):
    os.mkdir(os.path.join(args.output_dir, "vis"))
if not os.path.isdir(os.path.join(args.output_dir, "vis", mode)):
    os.mkdir(os.path.join(args.output_dir, "vis", mode))


# Load config
with open(args.config, "r") as yaml_file:
    cfg = yaml.safe_load(yaml_file)

with open(args.metrics_config, "r") as f:
    metrics_cfg = json.load(f)


# Initialize dataset
ds_type = (
    cfg["dataset"] + "_" + cfg["agent_setting"] + "_" + cfg["input_representation"]
)
if "VOD" in cfg["dataset"]:
    # spec_args = get_specific_args(cfg["dataset"], args.data_root, "v1.0-test", cfg)
    spec_args = get_specific_args(
        cfg["dataset"], args.data_root, "v1.0-visualisation", cfg
    )
    # spec_args = get_specific_args(cfg["dataset"], args.data_root, "v1.0-mini", cfg)
elif "nuScenes" in cfg["dataset"]:
    spec_args = get_specific_args(cfg["dataset"], args.data_root, "v1.0-trainval", cfg)
# test_set = initialize_dataset(
#    ds_type,
#    ["load_data", args.data_dir, cfg["test_set_args"]] + spec_args,
# )
test_set = initialize_dataset(
    ds_type,
    ["load_data", args.data_dir, cfg["vis_set_args"]] + spec_args,
)

# Initialize model
model = initialize_prediction_model(
    cfg["encoder_type"],
    cfg["aggregator_type"],
    cfg["decoder_type"],
    cfg["encoder_args"],
    cfg["aggregator_args"],
    cfg["decoder_args"],
)


visualizer = GeneralVisualizer(test_set.helper)


def get_metrics_by_class(metrics, tokens, vod):
    # TODO generalise
    K = [5, 10]
    class_metrics = collections.defaultdict(lambda: collections.defaultdict(list))
    for metric_name, metric_dict in metrics.items():
        for aggregator, values in metric_dict.items():
            for value, (i_t, s_t) in zip(values, tokens):
                instance = vod.get("instance", i_t)
                category = vod.get("category", instance["category_token"])

                if category["name"] == "vehicle.ego":
                    class_name = "vehicle.car"
                else:
                    class_name = category["name"]

                for i in range(len(K)):
                    class_metrics[class_name][(metric_name, str(K[i]))].append(value[i])

    return class_metrics


def display_class_metric_stats(metric_stats):
    max_name_length = 30
    for class_name, metrics in metric_stats.items():
        print()
        print(class_name)
        print("metric".ljust(max_name_length), "mean", "std.")
        for metric_name, metric_values in metrics.items():
            metric_avg = metric_values["mean"]
            metric_std = metric_values["std"]
            print(
                str(metric_name).ljust(max_name_length),
                # metric_name,
                f"{metric_avg:.02f}",
                f"{metric_std:.02f}",
            )


def get_class_metric_stats(class_metrics, stats=["mean", "std"]):
    metric_stats = collections.defaultdict(lambda: collections.defaultdict(dict))
    for class_name, metrics in class_metrics.items():
        for metric_name, metric_values in metrics.items():
            if "mean" in stats:
                metric_mean = np.mean(metric_values)
                metric_stats[class_name][metric_name]["mean"] = metric_mean
            if "std" in stats:
                metric_std = np.std(metric_values)
                metric_stats[class_name][metric_name]["std"] = metric_std

    return metric_stats


evaluator = Evaluator(
    cfg, metrics_cfg, test_set, model, args.output_dir, args.checkpoint, visualizer
)
evaluator.log_visualizations(mode)
exit()

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# Evaluate
metrics, tokens = evaluator.evaluate()
class_metrics = get_metrics_by_class(metrics, tokens, test_set.data)

metric_names = class_metrics.keys()
max_name_length = np.max([len(name) for name in metric_names])

metric_stats = collections.defaultdict(dict)
agg_metric_values = collections.defaultdict(list)
for class_name, metrics in class_metrics.items():
    print()
    print(class_name)
    print("metric".ljust(max_name_length), "mean", "std.")
    for metric_name, metric_values in metrics.items():
        # agg_metric_values[metric_name].extend(metric_values)
        metric_avg = np.mean(metric_values)
        metric_std = np.std(metric_values)
        print(
            str(metric_name).ljust(max_name_length),
            f"{metric_avg:.02f}",
            f"{metric_std:.02f}",
        )
        metric_stats[class_name][str(metric_name)] = {
            "mean": metric_avg,
            "std": metric_std,
        }

# for class_name, metrics in class_metrics.items():
#    print(class_name)
#    metric_name = ("MinADEK", "5")
#    metric_values = metrics[metric_name]
#    # np.histogram(metric_values, bins=20)
#    plt.hist(metric_values, bins=50)
#    plt.show()

with open(os.path.join(args.output_dir, "metric_stats.json"), "w") as f:
    json.dump(metric_stats, f, indent=2)

exit()

seeds = np.arange(0, 10)
# seeds = np.arange(0, 2)
# seeds = np.array([0])
seed_metrics = {}

for n, seed in enumerate(seeds):
    print(f"Seed {n+1}/{len(seeds)}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Evaluate
    metrics, tokens = evaluator.evaluate()
    class_metrics = get_metrics_by_class(metrics, tokens, test_set.data)

    # metrics_stats = get_class_metric_stats(class_metrics, ["mean", "std"])
    # display_class_metric_stats(metrics_stats)

    seed_metrics[seed] = class_metrics


metric_collection = collections.defaultdict(lambda: collections.defaultdict(list))
metric_names = []
for seed, class_metrics in seed_metrics.items():
    for class_name, metric_dict in class_metrics.items():
        for metric_name, metric_value in metric_dict.items():
            metric_collection[class_name][metric_name].append(np.mean(metric_value))
            metric_names.append(str(metric_name))

max_name_length = np.max([len(name) for name in metric_names])

metric_stats = collections.defaultdict(dict)
agg_metric_values = collections.defaultdict(list)
for class_name, metrics in metric_collection.items():
    print()
    print(class_name)
    print("metric".ljust(max_name_length), "mean", "std.")
    for metric_name, metric_values in metrics.items():
        agg_metric_values[metric_name].extend(metric_values)
        metric_avg = np.mean(metric_values)
        metric_std = np.std(metric_values)
        print(
            str(metric_name).ljust(max_name_length),
            f"{metric_avg:.02f}",
            f"{metric_std:.02f}",
        )
        metric_stats[class_name][str(metric_name)] = {
            "mean": metric_avg,
            "std": metric_std,
        }


for metric_name, metric_values in agg_metric_values.items():
    metric_stats["aggregated"][str(metric_name)] = {
        "mean": np.mean(metric_values),
        "std": np.std(metric_values),
    }

metric_stats["seeds"] = seeds.tolist()
with open(os.path.join(args.output_dir, "metric_stats.json"), "w") as f:
    json.dump(metric_stats, f, indent=2)
