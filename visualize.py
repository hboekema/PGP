import argparse
import os

import torch

torch.manual_seed(42)
import numpy as np
import torch.utils.data as torch_data
import yaml

import train_eval.utils as u
from train_eval.initialization import (get_specific_args, initialize_dataset,
                                       initialize_metric,
                                       initialize_prediction_model)
from train_eval.visualizer import GeneralVisualizer, Visualizer

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config", help="Config file with dataset parameters", required=True
)
parser.add_argument("-r", "--data_root", help="Root directory with data", required=True)
parser.add_argument("-d", "--data_dir", help="Directory to extract data", required=True)
parser.add_argument(
    "-o", "--output_dir", help="Directory to save results", required=True
)
parser.add_argument(
    "-w",
    "--checkpoint",
    help="Path to pre-trained or intermediate checkpoint",
    required=True,
)
args = parser.parse_args()


# Make directories
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
if not os.path.isdir(os.path.join(args.output_dir, "results")):
    os.mkdir(os.path.join(args.output_dir, "results"))


# Load config
with open(args.config, "r") as yaml_file:
    cfg = yaml.safe_load(yaml_file)


data_root = args.data_root
data_dir = args.data_dir
checkpoint_path = args.checkpoint


# Initialize dataset
ds_type = (
    cfg["dataset"] + "_" + cfg["agent_setting"] + "_" + cfg["input_representation"]
)
spec_args = get_specific_args(cfg["dataset"], data_root, "v1.0-test")
test_set = initialize_dataset(
    ds_type, ["load_data", data_dir, cfg["test_set_args"]] + spec_args
)

helper = test_set.helper

# Visualize
vis = GeneralVisualizer(helper)


# Initialize dataloader
dl = torch_data.DataLoader(
    test_set, cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"]
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
model = model.float().to(device)
model.eval()

if checkpoint_path is not None:
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

idcs = np.arange(0, 300, 10)
for idx in idcs:
    # print(data["inputs"].keys())
    # print(data["ground_truth"].keys())
    data = test_set[idx]

    # node_seq = data["inputs"]["node_seq_gt"][0].detach().cpu().numpy()
    # map_representation = data["inputs"]["map_representation"]
    # node_feats = map_representation["lane_node_feats"][0].detach().cpu().numpy()
    # s_next = map_representation["s_next"][0].detach().cpu().numpy()
    # edge_type = map_representation["edge_type"][0].detach().cpu().numpy()

    # evf_gt = data["ground_truth"]["evf_gt"][0].detach().cpu().numpy()
    # fut_xy = data["ground_truth"]["traj"][0].detach().cpu().numpy()

    vis.generate_frame(data, model, helper, map_extent=[-20, 20, -10, 30])
    # vis.visualize_graph(
    #    node_feats,
    #    s_next,
    #    edge_type,
    #    evf_gt,
    #    node_seq,
    #    fut_xy,
    #    map_extent=[-25, 25, -10, 40],
    # )
# vis = Visualizer(cfg, args.data_root, args.data_dir, args.checkpoint)
# vis.visualize(output_dir=args.output_dir, dataset_type=cfg["dataset"])
