import abc
from typing import Dict, Union

import torch
import torch.nn as nn

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Identity(nn.Module):
    """
    Base class for context aggregators for single agent prediction.
    Aggregates a set of context (map, surrounding agent) encodings and outputs either a single aggregated context vector
    or 'K' selectively aggregated context vectors for multimodal prediction.
    """

    def __init__(self):
        super().__init__()

    def forward(self, encodings: Dict) -> Union[Dict, torch.Tensor]:
        """
        Forward pass for prediction aggregator
        :param encodings: Dictionary with target agent and context encodings
        :return agg_encoding: Aggregated context encoding
        """
        return encodings["feats"]
