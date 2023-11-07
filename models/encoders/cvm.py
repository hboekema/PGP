import abc
from typing import Dict

import torch.nn as nn


class CVMEncoder(nn.Module):
    """
    Base class for encoders for single agent prediction.
    """

    def __init__(self, args: Dict):
        super().__init__()

    def forward(self, inputs: Dict) -> Dict:
        """
        :param inputs: Dictionary with
            'target_agent_representation': target agent history
            'surrounding_agent_representation': surrounding agent history
            'map_representation': HD map representation
        :return encodings: Dictionary with input encodings
        """
        target_agent_feats = inputs["target_agent_representation"]

        return {"feats": target_agent_feats}
