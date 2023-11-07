import abc
import itertools
import os
import pickle
from typing import Dict, List, Union

import numpy as np
from datasets.interface import SingleAgentDataset
from vod.eval.prediction.splits import get_prediction_challenge_split
from vod.prediction import PredictHelper, create_prediction_split


class VODTrajectories(SingleAgentDataset):
    """
    VOD dataset class for single agent prediction
    """

    def __init__(
        self,
        mode: str,
        data_dir: str,
        args: Dict,
        helper: PredictHelper,
        output_mode="vod",
    ):
        """
        Initialize predict helper, agent and scene representations
        :param mode: Mode of operation of dataset, one of {'compute_stats', 'extract_data', 'load_data'}
        :param data_dir: Directory to store extracted pre-processed data
        :param helper: VOD PredictHelper
        :param args: Dataset arguments
        """
        super().__init__(mode, data_dir)
        self.helper = helper
        self.data = self.helper.data

        # VOD sample and instance tokens for split
        split = args["split"]
        if split == "train" or split == "val" or split == "train_val":
            splitname = "trainval"
        else:
            splitname = split

        tokens = helper.get_tokens("prediction_scenes_h5f30.json")[splitname]
        if split == "train" or split == "val":
            train_tokens, val_tokens = create_prediction_split(
                tokens, nbr=int(len(tokens) * 0.8)
            )
            split_tokens = train_tokens if split == "train" else val_tokens
        else:
            split_tokens = tokens

        self.token_list = list(
            itertools.chain(*[values for _, values in split_tokens.items()])
        )

        self.agent_types = [
            "vehicle.car",
            "vehicle.bicycle",
            "vehicle.motorcycle",
            "human.pedestrian.adult",
        ]

        # Past and prediction horizons
        self.t_h = args["t_h"]
        self.t_f = args["t_f"]

        self.output_mode = output_mode
        assert output_mode in ["vod", "nuscenes"]

    def __len__(self):
        """
        Size of dataset
        """
        return len(self.token_list)

    def get_inputs(self, idx: int) -> Dict:
        """
        Gets model inputs for VOD single agent prediction
        :param idx: data index
        :return inputs: Dictionary with input representations
        """
        i_t, s_t = self.token_list[idx].split("_")
        map_representation = self.get_map_representation(idx)
        surrounding_agent_representation = self.get_surrounding_agent_representation(
            idx, self.agent_types
        )
        target_agent_representation = self.get_target_agent_representation(idx)
        inputs = {
            "instance_token": i_t,
            "sample_token": s_t,
            "map_representation": map_representation,
            "surrounding_agent_representation": surrounding_agent_representation,
            "target_agent_representation": target_agent_representation,
            "class_encoding": target_agent_representation[0, -len(self.agent_types) :],
        }
        return inputs

    def get_ground_truth(self, idx: int) -> Dict:
        """
        Gets ground truth labels for VOD single agent prediction
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

        if self.output_mode == "nuscenes":
            data = self.translate_to_nuscenes(data)
        return data

    def translate_to_nuscenes(self, data):
        del data["inputs"]["class_encoding"]
        inputs = data["inputs"]

        # edit class encoding
        target_agent = data["inputs"]["target_agent_representation"]
        old_class_enc = target_agent[:, -len(self.agent_types) :]
        new_class_enc = list(map(lambda x: [1] if x[0] == 1 else [0], old_class_enc))
        target_agent = target_agent[:, : -len(self.agent_types)]
        # target_agent = np.concatenate([target_agent, new_class_enc], axis=-1)
        data["inputs"]["target_agent_representation"] = target_agent

        # Merge surrounding agents
        vehicles = []
        pedestrians = []
        vehicle_masks = []
        pedestrian_masks = []
        vehicle_node_masks = []
        pedestrian_node_masks = []

        surrounding_agents = inputs["surrounding_agent_representation"]
        node_masks = inputs["agent_node_masks"]
        # mask_length = max([value.shape[-1] for _, value in node_masks.items()])
        for agent_type in self.agent_types:
            agent_data = surrounding_agents[agent_type]
            agent_masks = surrounding_agents[agent_type + "_masks"]
            agent_node_masks_unpadded = node_masks[agent_type]
            agent_node_masks = agent_node_masks_unpadded
            # agent_node_masks = np.ones(
            #    (agent_node_masks_unpadded.shape[0], mask_length)
            # )
            # agent_node_masks[
            #    :,
            #    : agent_node_masks_unpadded.shape[1],
            # ] = agent_node_masks_unpadded
            if "vehicle" in agent_type:
                vehicles.extend(agent_data)
                vehicle_masks.extend(agent_masks)
                vehicle_node_masks.append(agent_node_masks)
            elif "pedestrian" in agent_type:
                pedestrians.extend(agent_data)
                pedestrian_masks.extend(agent_masks)
                pedestrian_node_masks.extend(agent_node_masks)

        vehicle_node_masks = np.concatenate(vehicle_node_masks, axis=-1)
        vehicle_node_masks = np.array(vehicle_node_masks)
        # vehicle_node_masks = np.all(vehicle_node_masks, axis=0).astype(int)
        pedestrian_node_masks = np.array(pedestrian_node_masks)

        data["inputs"]["surrounding_agent_representation"] = {
            "vehicles": np.array(vehicles),
            "vehicle_masks": np.array(vehicle_masks),
            "pedestrians": np.array(pedestrians),
            "pedestrian_masks": np.array(pedestrian_masks),
        }
        # print(np.array(vehicles).shape)
        # print(np.array(pedestrians).shape)
        # print(np.array(vehicle_node_masks).shape)
        # print(np.array(pedestrian_node_masks).shape)
        # exit()
        # print(np.array(vehicles))
        # print(np.array(vehicles).shape)
        data["inputs"]["agent_node_masks"] = {
            "vehicles": vehicle_node_masks,
            "pedestrians": pedestrian_node_masks,
        }
        return data

    def get_target_agent_future(self, idx: int) -> np.ndarray:
        """
        Extracts future trajectory for target agent
        :param idx: data index
        :return fut: future trajectory for target agent, shape: [t_f * freq, 2]
        """
        i_t, s_t = self.token_list[idx].split("_")
        fut = self.helper.get_future_for_agent(
            i_t, s_t, seconds=self.t_f, in_agent_frame=True
        )
        # print(fut.shape)
        # assert fut.shape == (20, 2), fut.shape

        return fut

    @abc.abstractmethod
    def get_target_agent_representation(self, idx: int) -> Union[np.ndarray, Dict]:
        """target_agent_future
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
    def get_surrounding_agent_representation(
        self, idx: int, agent_types: List[str]
    ) -> Union[np.ndarray, Dict]:
        """
        Extracts surrounding agent representation
        :param idx: data index
        """
        raise NotImplementedError()


if __name__ == "__main__":
    from vod import VOD

    vod = VOD("v1.0-trainval", "/home/hjhboekema/Projects/view-of-delft-prediction")
    helper = PredictHelper(vod)

    mode = "compute_stats"
    data_dir = "/home/hjhboekema/Projects/PGP/data/preprocessed"

    ds = VODTrajectories(mode, data_dir, {}, helper)
