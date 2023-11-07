import abc
import os
import pickle
from typing import Dict, Union

import numpy as np
from datasets.interface import SingleAgentDataset
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper


class NuScenesTrajectories10Hz(SingleAgentDataset):
    """
    NuScenes dataset class for single agent prediction
    """

    def __init__(self, mode: str, data_dir: str, args: Dict, helper: PredictHelper):
        """
        Initialize predict helper, agent and scene representations
        :param mode: Mode of operation of dataset, one of {'compute_stats', 'extract_data', 'load_data'}
        :param data_dir: Directory to store extracted pre-processed data
        :param helper: NuScenes PredictHelper
        :param args: Dataset arguments
        """
        super().__init__(mode, data_dir)
        self.helper = helper

        # nuScenes sample and instance tokens for prediction challenge
        self.token_list = get_prediction_challenge_split(
            args["split"], dataroot=helper.data.dataroot
        )

        # Past and prediction horizons
        self.t_h = args["t_h"]
        self.t_f = args["t_f"]

        self.freq = 10  # Hz
        self.ts_h = int(self.freq * self.t_h)
        self.ts_f = int(self.freq * self.t_f)

        self.max_t = 100

        self.data = self.helper.data

    def __len__(self):
        """
        Size of dataset
        """
        return len(self.token_list)

    def get_inputs(self, idx: int) -> Dict:
        """
        Gets model inputs for nuScenes single agent prediction
        :param idx: data index
        :return inputs: Dictionary with input representations
        """
        i_t, s_t = self.token_list[idx].split("_")
        map_representation = self.get_map_representation(idx)
        surrounding_agent_representation = self.get_surrounding_agent_representation(
            idx
        )
        target_agent_representation = self.get_target_agent_representation(idx)

        assert len(target_agent_representation) == self.ts_h + 1
        inputs = {
            "instance_token": i_t,
            "sample_token": s_t,
            "map_representation": map_representation,
            "surrounding_agent_representation": surrounding_agent_representation,
            "target_agent_representation": target_agent_representation,
        }
        return inputs

    def get_ground_truth(self, idx: int) -> Dict:
        """
        Gets ground truth labels for nuScenes single agent prediction
        :param idx: data index
        :return ground_truth: Dictionary with grund truth labels
        """
        target_agent_future = self.get_target_agent_future(idx)
        ground_truth = {"traj": target_agent_future}
        return ground_truth

    def save_data(self, idx: int, data: Dict):
        """
        Saves extracted pre-processed data
        :param idx: data index
        :param data: pre-processed data
        """
        filename = os.path.join(self.data_dir, self.token_list[idx] + ".pickle")
        with open(filename, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, idx: int) -> Dict:
        """
        Function to load extracted data.
        :param idx: data index
        :return data: Dictionary with batched tensors
        """
        filename = os.path.join(self.data_dir, self.token_list[idx] + ".pickle")

        if not os.path.isfile(filename):
            raise Exception(
                "Could not find data. Please run the dataset in extract_data mode"
            )

        with open(filename, "rb") as handle:
            data = pickle.load(handle)

        # print("inputs")
        # for key, value in data["inputs"].items():
        #     if isinstance(value, np.ndarray):
        #         print(key, value.shape)
        # print("ground_truth")
        # for key, value in data["ground_truth"].items():
        #     if isinstance(value, np.ndarray):
        #         print(key, value.shape)

        return data

    def get_target_agent_future(self, idx: int) -> np.ndarray:
        """
        Extracts future trajectory for target agent
        :param idx: data index
        :return fut: future trajectory for target agent, shape: [t_f * freq, 2]
        """
        i_t, s_t = self.token_list[idx].split("_")
        _, fut = self.upsample_track(i_t, s_t)
        if len(fut) > self.ts_f:
            fut = fut[: self.ts_f]

        assert len(fut) == self.ts_f
        return fut

    @abc.abstractmethod
    def get_target_agent_representation(self, idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts target agent representation
        :param idx: data index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_map_representation(self, idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts map representation
        :param idx: data index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_surrounding_agent_representation(self, idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts surrounding agent representation
        :param idx: data index
        """
        raise NotImplementedError()
