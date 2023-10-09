import collections
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.utils.data as torch_data
from PIL import Image
from vod.eval.prediction.compute_metrics import compute_metrics
from vod.eval.prediction.config import PredictionConfig
from vod.eval.prediction.data_classes import Prediction
from vod.prediction.helper import convert_local_coords_to_global

import train_eval.utils as u
from train_eval.initialization import (get_specific_args, initialize_dataset,
                                       initialize_metric,
                                       initialize_prediction_model)

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def index_dict(dict_, idx):
    return {key: elem[idx : idx + 1] for key, elem in dict_.items()}


class Evaluator:
    """
    Class for evaluating trained models
    """

    def __init__(
        self,
        cfg: Dict,
        metric_cfg: Dict,
        dataset,
        output_dir: str,
        checkpoint_path: str = None,
        visualizer=None,
    ):
        """
        Initialize evaluator object
        :param cfg: Configuration parameters
        :param data_root: Root directory with data
        :param data_dir: Directory with extracted, pre-processed data
        :param checkpoint_path: Path to checkpoint with trained weights
        """

        self.ds = dataset
        self.helper = self.ds.helper
        self.visualizer = visualizer
        self.output_dir = output_dir

        # Initialize dataloader
        self.dl = torch_data.DataLoader(
            self.ds, cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"]
        )

        # Initialize model
        self.model = initialize_prediction_model(
            cfg["encoder_type"],
            cfg["aggregator_type"],
            cfg["decoder_type"],
            cfg["encoder_args"],
            cfg["aggregator_args"],
            cfg["decoder_args"],
        )
        self.model = self.model.float().to(device)
        self.model.eval()

        if checkpoint_path is not None:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # Initialize metrics
        self.metric_config = PredictionConfig.deserialize(metric_cfg, self.helper)

    def evaluate_return(self):
        """
        Main function to evaluate trained model
        """

        # Initialize aggregate metrics
        agg_metrics = self.initialize_aggregate_metrics()
        class_metrics = self.initialize_class_metrics()

        with torch.no_grad():
            for i, data in enumerate(self.dl):
                # Load data
                data = u.send_to_device(u.convert_double_to_float(data))
                class_encoding = data["inputs"]["target_agent_representation"][
                    :, 0, -len(self.class_encodings) :
                ]
                class_idx = torch.argmax(class_encoding, dim=-1).detach().cpu().numpy()
                classes = [self.idx_to_class[idx] for idx in class_idx]

                # Forward pass
                predictions = self.model(data["inputs"])

                # Aggregate metrics
                agg_metrics = self.aggregate_metrics(
                    agg_metrics, predictions, data["ground_truth"]
                )

                # Get metrics by class
                class_metrics = self.calculate_class_metrics(
                    class_metrics,
                    classes,
                    predictions,
                    data["ground_truth"],
                )

                self.print_progress(i)

        self.print_progress(len(self.dl))

        # compute and print average metrics
        avg_metrics = collections.defaultdict(dict)
        for metric in self.metrics:
            avg_metric = agg_metrics[metric.name] / agg_metrics["sample_count"]
            avg_metrics["aggregate"][metric.name] = avg_metric

        for agent_type in class_metrics.keys():
            for metric in self.metrics:
                class_avg_metric = class_metrics[agent_type][metric.name] / (
                    class_metrics[agent_type]["sample_count"] + 1e-5
                )

                avg_metrics[agent_type][metric.name] = class_avg_metric

        return avg_metrics

    def visualize(self):
        print("Logging visualizations...")
        self.log_visualizations(self.ds, "test")
        print("Done.")

    def evaluate(self):
        """
        Main function to evaluate trained model
        :param output_dir: Output directory to store results
        """
        metrics = collections.defaultdict(lambda: collections.defaultdict(list))
        tokens = []
        with torch.no_grad():
            for i, data in enumerate(self.dl):

                # Load data
                data = u.send_to_device(u.convert_double_to_float(data))

                # Forward pass
                predictions = self.model(data["inputs"])

                instance_tokens = data["inputs"]["instance_token"]
                sample_tokens = data["inputs"]["sample_token"]
                tokens.extend(list(zip(instance_tokens, sample_tokens)))

                minibatch_metrics = self.get_metrics(
                    instance_tokens, sample_tokens, predictions
                )
                for metric_name, metric_dict in minibatch_metrics.items():
                    for metric_agg, values in metric_dict.items():
                        metrics[metric_name][metric_agg].extend(values)

                self.print_progress(i)

        self.print_progress(len(self.dl))

        return metrics, tokens

    def _evaluate(self):
        """
        Main function to evaluate trained model
        :param output_dir: Output directory to store results
        """
        # Initialize aggregate metrics
        agg_metrics = self.initialize_aggregate_metrics()
        class_metrics = self.initialize_class_metrics()

        with torch.no_grad():
            for i, data in enumerate(self.dl):

                # Load data
                data = u.send_to_device(u.convert_double_to_float(data))
                class_encoding = data["inputs"]["target_agent_representation"][
                    :, 0, -len(self.class_encodings) :
                ]
                class_idx = torch.argmax(class_encoding, dim=-1).detach().cpu().numpy()
                classes = [self.idx_to_class[idx] for idx in class_idx]

                # Forward pass
                predictions = self.model(data["inputs"])

                # Aggregate metrics
                agg_metrics = self.aggregate_metrics(
                    agg_metrics, predictions, data["ground_truth"]
                )

                # Get metrics by class
                class_metrics = self.calculate_class_metrics(
                    class_metrics,
                    classes,
                    predictions,
                    data["ground_truth"],
                )

                self.print_progress(i)

        self.print_progress(len(self.dl))

        # compute and print average metrics
        with open(
            os.path.join(self.output_dir, "results", "results.txt"), "w"
        ) as out_file:
            for metric in self.metrics:
                avg_metric = agg_metrics[metric.name] / agg_metrics["sample_count"]
                output = metric.name + ": " + format(avg_metric, "0.2f")
                print(output)
                out_file.write(output + "\n")

            print()
            for agent_type in class_metrics.keys():
                print(agent_type)
                for metric in self.metrics:
                    class_avg_metric = class_metrics[agent_type][metric.name] / (
                        class_metrics[agent_type]["sample_count"] + 1e-5
                    )
                    output = (
                        agent_type
                        + " "
                        + metric.name
                        + ": "
                        + format(class_avg_metric, "0.2f")
                    )
                    print(output)
                    out_file.write(output + "\n")

        # Make visualizations
        self.log_visualizations(self.ds, "test")

        # Debug
        # all_metrics = {metric.name: 0.0 for metric in self.metrics}
        # all_metrics["sample_count"] = 0
        # for agent_type in class_metrics.keys():
        #    all_metrics["sample_count"] += class_metrics[agent_type]["sample_count"]
        #    for metric in self.metrics:
        #        all_metrics[metric.name] += class_metrics[agent_type][metric.name]

        # for metric in self.metrics:
        #    avg_metric = all_metrics[metric.name] / all_metrics["sample_count"]
        #    output = metric.name + ": " + format(avg_metric, "0.2f")
        #    print(output)
        # print(agg_metrics["sample_count"])
        # print(all_metrics["sample_count"])
        # print(class_metrics)

    def log_visualizations(self, ds, mode):
        max_idx = len(ds)
        assert self.num_vis <= max_idx
        assert self.num_vis > 0
        idcs = range(0, max_idx)

        for idx in idcs:
            data = ds[idx]
            # image_array = self.visualizer.generate_frame(
            #    data, self.model, ds.helper, ds.map_extent
            # )
            instance_token = data["inputs"]["instance_token"]
            sample_token = data["inputs"]["sample_token"]
            image_array = self.visualizer.generate_frame(
                data, self.model, ds.helper, [-20, 20, -10, 30]
            )
            image = Image.fromarray(image_array)
            image.save(
                os.path.join(
                    self.output_dir,
                    "vis",
                    mode,
                    f"{instance_token}_{sample_token}.png",
                )
            )
            image.save(
                os.path.join(
                    self.output_dir,
                    "vis",
                    mode,
                    f"{instance_token}_{sample_token}.pdf",
                ),
                "PDF",
            )

    def initialize_aggregate_metrics(self):
        """
        Initialize aggregate metrics for test set.
        """
        agg_metrics = {"sample_count": 0}
        for metric in self.metrics:
            agg_metrics[metric.name] = 0

        return agg_metrics

    def initialize_class_metrics(self):
        """
        Initialize class metrics for test set.
        """
        class_metrics = collections.defaultdict(dict)
        for agent_type, _ in self.class_encodings.items():
            class_metrics[agent_type]["sample_count"] = 0
            for metric in self.metrics:
                class_metrics[agent_type][metric.name] = 0

        return class_metrics

    def calculate_metrics(self, model_outputs, ground_truth):
        samples_metrics = []
        for i in range(len(model_outputs["traj"])):
            sample_metrics = {}
            for metric in self.metrics:
                sample_metrics[metric.name] = metric.compute(
                    index_dict(model_outputs, i), index_dict(ground_truth, i)
                ).item()
            samples_metrics.append(sample_metrics)

        return samples_metrics

    def calculate_class_metrics(
        self,
        class_metrics: Dict,
        classes,
        model_outputs: Dict,
        ground_truth: Dict,
    ):
        """
        Calculates class metrics for evaluation
        """
        samples_metrics = self.calculate_metrics(model_outputs, ground_truth)

        for i, sample_metrics in enumerate(samples_metrics):
            class_name = classes[i]
            class_metrics[class_name]["sample_count"] += 1
            for metric in self.metrics:
                class_metrics[class_name][metric.name] += sample_metrics[metric.name]

        return class_metrics

    def get_metrics(
        self,
        instance_tokens: List,
        sample_tokens: List,
        predictions: Dict,
    ):
        """
        Computes metrics for evaluation using VOD devkit
        """
        # Forward pass
        traj = predictions["traj"]
        probs = predictions["probs"]

        # Create prediction object and add to list of predictions
        preds = []
        for n in range(traj.shape[0]):
            traj_local = traj[n].detach().cpu().numpy()
            probs_n = probs[n].detach().cpu().numpy()
            starting_annotation = self.helper.get_sample_annotation(
                instance_tokens[n], sample_tokens[n]
            )
            traj_global = np.zeros_like(traj_local)
            for m in range(traj_local.shape[0]):
                traj_global[m] = convert_local_coords_to_global(
                    traj_local[m],
                    starting_annotation["translation"],
                    starting_annotation["rotation"],
                )

            preds.append(
                Prediction(
                    instance=instance_tokens[n],
                    sample=sample_tokens[n],
                    prediction=traj_global,
                    probabilities=probs_n,
                ).serialize()
            )

        minibatch_metrics = compute_metrics(preds, self.helper, self.metric_config)
        # batch_size = ground_truth["traj"].shape[0]
        # agg_metrics["sample_count"] += batch_size

        # for metric in self.metrics:
        #    agg_metrics[metric.name] += minibatch_metrics[metric.name] * batch_size

        # return agg_metrics
        return minibatch_metrics

    # def aggregate_metrics(
    #    self, agg_metrics: Dict, model_outputs: Dict, ground_truth: Dict
    # ):
    #    """
    #    Aggregates metrics for evaluation
    #    """
    #    minibatch_metrics = {}
    #
    #    for metric in self.metrics:
    #        minibatch_metrics[metric.name] = metric.compute(
    #            model_outputs, ground_truth
    #        ).item()

    #    batch_size = ground_truth["traj"].shape[0]
    #    agg_metrics["sample_count"] += batch_size

    #    for metric in self.metrics:
    #        agg_metrics[metric.name] += minibatch_metrics[metric.name] * batch_size

    #    return agg_metrics

    def print_progress(self, minibatch_count: int):
        """
        Prints progress bar
        """
        epoch_progress = minibatch_count / len(self.dl) * 100
        print("\rEvaluating:", end=" ")
        progress_bar = "["
        for i in range(20):
            if i < epoch_progress // 5:
                progress_bar += "="
            else:
                progress_bar += " "
        progress_bar += "]"
        print(
            progress_bar,
            format(epoch_progress, "0.2f"),
            "%",
            end="\n" if epoch_progress == 100 else " ",
        )

    def generate_nuscenes_benchmark_submission(self):
        """
        Sets up list of Prediction objects for the nuScenes benchmark.
        """

        # NuScenes prediction helper
        helper = self.dl.dataset.helper

        # List of predictions
        preds = []

        with torch.no_grad():
            for i, data in enumerate(self.dl):

                # Load data
                data = u.send_to_device(u.convert_double_to_float(data))

                # Forward pass
                predictions = self.model(data["inputs"])
                traj = predictions["traj"]
                probs = predictions["probs"]

                # Load instance and sample tokens for batch
                instance_tokens = data["inputs"]["instance_token"]
                sample_tokens = data["inputs"]["sample_token"]

                # Create prediction object and add to list of predictions
                for n in range(traj.shape[0]):

                    traj_local = traj[n].detach().cpu().numpy()
                    probs_n = probs[n].detach().cpu().numpy()
                    starting_annotation = helper.get_sample_annotation(
                        instance_tokens[n], sample_tokens[n]
                    )
                    traj_global = np.zeros_like(traj_local)
                    for m in range(traj_local.shape[0]):
                        traj_global[m] = convert_local_coords_to_global(
                            traj_local[m],
                            starting_annotation["translation"],
                            starting_annotation["rotation"],
                        )

                    preds.append(
                        Prediction(
                            instance=instance_tokens[n],
                            sample=sample_tokens[n],
                            prediction=traj_global,
                            probabilities=probs_n,
                        ).serialize()
                    )

                # Print progress bar
                self.print_progress(i)

            # Save predictions to json file
            json.dump(
                preds,
                open(
                    os.path.join(self.output_dir, "results", "evalai_submission.json"),
                    "w",
                ),
            )
            self.print_progress(len(self.dl))
