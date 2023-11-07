import abc
from typing import Dict, Union

import torch
import torch.nn as nn

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CVMDecoder(nn.Module):
    """
    Base class for decoders for single agent prediction.
    Outputs K trajectories and/or their probabilities
    """

    def __init__(self, args: Dict):
        super().__init__()
        self.op_len = args["op_len"]
        self.k = args["k"]

    def forward(
        self, agg_encoding: Union[torch.Tensor, Dict]
    ) -> Union[torch.Tensor, Dict]:
        """
        Forward pass for prediction decoder
        :param agg_encoding: Aggregated context encoding
        :return outputs: K Predicted trajectories and/or their probabilities/scores
        """
        curr, prev = agg_encoding[:, -1, :2], agg_encoding[:, -2, :2]
        vec = curr - prev
        vec = vec.unsqueeze(1).repeat(1, self.op_len, 1)

        steps = (
            torch.arange(1, self.op_len + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(vec.shape[0], 1, 2)
        ).float()
        # steps = steps.permute(0, 2, 1)
        steps = steps.to(vec.get_device())

        # print(steps.shape)
        # print(vec.shape)
        out = vec * steps
        out = out.unsqueeze(1).repeat(1, self.k, 1, 1)

        probs = torch.zeros(vec.shape[0], self.k)
        probs[:,0] = 1.0

        predictions = {"traj": out, "probs": probs}

        return predictions 

