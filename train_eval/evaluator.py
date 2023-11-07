import collections
import json
import os
from typing import Dict, List

import matplotlib
# matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as torch_data
# from nuscenes.eval.prediction.compute_metrics import compute_metrics
# from nuscenes.eval.prediction.config import PredictionConfig
# from nuscenes.eval.prediction.data_classes import Prediction
# from nuscenes.prediction.helper import convert_local_coords_to_global
from PIL import Image
from vod.eval.prediction.config import PredictionConfig
from vod.eval.prediction.data_classes import Prediction
from vod.prediction.helper import convert_local_coords_to_global

import train_eval.utils as u
from train_eval.compute_metrics import compute_metrics

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
        model,
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

        self.model = model.float().to(device)
        self.model.eval()

        if checkpoint_path is not None:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # Initialize metrics
        self.metric_config = PredictionConfig.deserialize(metric_cfg, self.helper)

    def log_visualizations(self, mode="test"):
        print("Logging visualizations...")
        max_idx = len(self.ds)
        idcs = range(0, max_idx)

        # vis_tokens = [
        #    "16iapzfxofuerrobtdew2je4pui8wcnt_3cwqryg2ouozd06ss5a15taekybj63db",
        #    "3y87hc7eowhavplig9op9o6wfhjgugdv_vi2um5abqhclmoyk9ysfezi54dp9oeo1",
        #    "lx2k5hlhd35no2boqqszzszvkpuibbzy_26qkcdesfrriqza4o5kp81ewxk7ur3u4",
        # ]
        vis_instance_tokens = [
            "16iapzfxofuerrobtdew2je4pui8wcnt",
            "3y87hc7eowhavplig9op9o6wfhjgugdv",
            # "lx2k5hlhd35no2boqqszzszvkpuibbzy",
        ]
        for idx in idcs:
            data = self.ds[idx]
            # image_array = self.visualizer.generate_frame(
            #    data, self.model, ds.helper, ds.map_extent
            # )
            instance_token = data["inputs"]["instance_token"]
            sample_token = data["inputs"]["sample_token"]

            # token = instance_token + "_" + sample_token
            # if token not in vis_tokens:
            #    continue
            if instance_token not in vis_instance_tokens:
                continue

            image_array = self.visualizer.generate_frame(
                data, self.model, self.ds.helper, [-20, 20, -10, 30]
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
                    instance_tokens,
                    sample_tokens,
                    predictions,
                    data["ground_truth"]["traj"].detach().cpu().numpy(),
                )
                for metric_name, metric_dict in minibatch_metrics.items():
                    for metric_agg, values in metric_dict.items():
                        metrics[metric_name][metric_agg].extend(values)

                self.print_progress(i)

        self.print_progress(len(self.dl))

        return metrics, tokens

    def get_metrics(
        self,
        instance_tokens: List,
        sample_tokens: List,
        predictions: Dict,
        ground_truths: List,
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
            # traj_global = np.zeros_like(traj_local)
            # for m in range(traj_local.shape[0]):
            #    traj_global[m] = convert_local_coords_to_global(
            #        traj_local[m],
            #        starting_annotation["translation"],
            #        starting_annotation["rotation"],
            #    )
            traj_global = traj_local

            preds.append(
                Prediction(
                    instance=instance_tokens[n],
                    sample=sample_tokens[n],
                    prediction=traj_global,
                    probabilities=probs_n,
                ).serialize()
            )

        minibatch_metrics = compute_metrics(preds, ground_truths, self.metric_config)
        # batch_size = ground_truth["traj"].shape[0]
        # agg_metrics["sample_count"] += batch_size

        # for metric in self.metrics:
        #    agg_metrics[metric.name] += minibatch_metrics[metric.name] * batch_size

        # return agg_metrics
        return minibatch_metrics

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
