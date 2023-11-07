import os
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from datasets.nuScenes10Hz.nuScenes import NuScenesTrajectories10Hz
from geometry_utils.interpolate import upsample1d, upsample2d
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import correct_yaw
from pyquaternion import Quaternion
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class NuScenesVector10Hz(NuScenesTrajectories10Hz):
    """
    NuScenes dataset class for single agent prediction, using the vector representation for maps and agents
    """

    def __init__(self, mode: str, data_dir: str, args: Dict, helper: PredictHelper):
        """
        Initialize predict helper, agent and scene representations
        :param mode: Mode of operation of dataset, one of {'compute_stats', 'extract_data', 'load_data'}
        :param data_dir: Directory to store extracted pre-processed data
        :param helper: NuScenes PredictHelper
        :param args: Dataset arguments
        """
        super().__init__(mode, data_dir, args, helper)

        # Initialize helper and maps
        self.map_locs = [
            "singapore-onenorth",
            "singapore-hollandvillage",
            "singapore-queenstown",
            "boston-seaport",
        ]
        self.maps = {
            i: NuScenesMap(map_name=i, dataroot=self.helper.data.dataroot)
            for i in self.map_locs
        }

        # Vector map parameters
        self.map_extent = args["map_extent"]
        self.polyline_resolution = args["polyline_resolution"]
        self.polyline_length = args["polyline_length"]

        # Load dataset stats (max nodes, max agents etc.)
        if self.mode == "extract_data":
            stats = self.load_stats()
            self.max_nodes = stats["num_lane_nodes"]
            self.max_vehicles = stats["num_vehicles"]
            self.max_pedestrians = stats["num_pedestrians"]

        # Whether to add random flips for data augmentation
        elif self.mode == "load_data":
            self.random_flips = args["random_flips"]

    def compute_stats(self, idx: int) -> Dict[str, int]:
        """
        Function to compute statistics for a given data point
        """
        num_lane_nodes = self.get_map_representation(idx)
        num_vehicles, num_pedestrians = self.get_surrounding_agent_representation(idx)
        stats = {
            "num_lane_nodes": num_lane_nodes,
            "num_vehicles": num_vehicles,
            "num_pedestrians": num_pedestrians,
        }

        return stats

    def load_data(self, idx: int) -> Dict:
        """
        Perform random flips if lag is set to true.
        """
        data = super().load_data(idx)

        if self.random_flips:
            if torch.randint(2, (1, 1)).squeeze().bool().item():
                data = self.flip_horizontal(data)

        return data

    def upsample_track(self, i_t, s_t, factor=5, in_agent_frame=True):
        hist = self.helper.get_past_for_agent(
            i_t, s_t, seconds=self.max_t, in_agent_frame=in_agent_frame
        )
        len_h = len(hist)
        hist = hist[::-1]  # flip to have correct order of timestamps
        if in_agent_frame:
            present = np.zeros((1, 2))
        else:
            annotation = self.helper.get_sample_annotation(i_t, s_t)
            present = np.asarray(annotation["translation"][0:2]).reshape(1, 2)

        # hist = np.concatenate([hist, present], axis=0)
        fut = self.helper.get_future_for_agent(
            i_t, s_t, seconds=self.max_t, in_agent_frame=in_agent_frame
        )
        len_f = len(fut)

        if len_f > 0:
            track = np.concatenate([hist, present, fut], axis=0)

            track_upsampled = upsample2d(track, factor=factor)
            hist_upsampled, fut_upsampled = (
                track_upsampled[: len_h * 5],
                track_upsampled[-len_f * 5 :],
            )
        else:
            track = np.concatenate([hist, present], axis=0)

            track_upsampled = upsample2d(track, factor=factor)
            hist_upsampled = track_upsampled[: len(hist)]
            fut_upsampled = None

        return hist_upsampled, fut_upsampled

    def get_target_agent_representation(self, idx: int) -> np.ndarray:
        """
        Extracts target agent representation
        :param idx: data index
        :return hist: track history for target agent, shape: [t_h * freq, 5]
        """
        i_t, s_t = self.token_list[idx].split("_")

        # x, y co-ordinates in agent's frame of reference
        # hist = self.helper.get_past_for_agent(
        #    i_t, s_t, seconds=int(self.freq * self.t_h), in_agent_frame=True
        # )
        hist, fut = self.upsample_track(i_t, s_t)
        # print("hist ups:", hist)
        if len(hist) > self.ts_h:
            hist = hist[-self.ts_h :]
        # print("hist cropped:", hist)
        # exit()

        # Zero pad for track histories shorter than t_h
        hist_zeropadded = np.zeros((self.ts_h + 1, 2))
        hist_zeropadded[-hist.shape[0] - 1 : -1] = hist
        hist = hist_zeropadded

        # Get velocity, acc and yaw_rate over past t_h sec
        # print(hist.shape)
        motion_states = self.get_past_motion_states(i_t, s_t, len(hist))
        # print(motion_states.shape)
        hist = np.concatenate((hist, motion_states), axis=1)

        return hist

    def get_map_representation(self, idx: int) -> Union[int, Dict]:
        """
        Extracts map representation
        :param idx: data index
        :return: Returns an ndarray with lane node features, shape [max_nodes, polyline_length, 5] and an ndarray of
            masks of the same shape, with value 1 if the nodes/poses are empty,
        """
        i_t, s_t = self.token_list[idx].split("_")
        map_name = self.helper.get_map_name_from_sample_token(s_t)
        map_api = self.maps[map_name]

        # Get agent representation in global co-ordinates
        global_pose = self.get_target_agent_global_pose(idx)

        # Get lanes around agent within map_extent
        lanes = self.get_lanes_around_agent(global_pose, map_api)

        # Get relevant polygon layers from the map_api
        polygons = self.get_polygons_around_agent(global_pose, map_api)

        # Get vectorized representation of lanes
        lane_node_feats, _ = self.get_lane_node_feats(global_pose, lanes, polygons)

        # Discard lanes outside map extent
        lane_node_feats = self.discard_poses_outside_extent(lane_node_feats)

        # Add dummy node (0, 0, 0, 0, 0) if no lane nodes are found
        if len(lane_node_feats) == 0:
            lane_node_feats = [np.zeros((1, 5))]

        # While running the dataset class in 'compute_stats' mode:
        if self.mode == "compute_stats":
            return len(lane_node_feats)

        # Convert list of lane node feats to fixed size numpy array and masks
        lane_node_feats, lane_node_masks = self.list_to_tensor(
            lane_node_feats, self.max_nodes, self.polyline_length, 5
        )

        map_representation = {
            "lane_node_feats": lane_node_feats,
            "lane_node_masks": lane_node_masks,
        }

        return map_representation

    def get_surrounding_agent_representation(
        self, idx: int
    ) -> Union[Tuple[int, int], Dict]:
        """
        Extracts surrounding agent representation
        :param idx: data index
        :return: ndarrays with surrounding pedestrian and vehicle track histories and masks for non-existent agents
        """

        # Get vehicles and pedestrian histories for current sample
        vehicles = self.get_agents_of_type(idx, "vehicle")
        pedestrians = self.get_agents_of_type(idx, "human")

        # Discard poses outside map extent
        vehicles = self.discard_poses_outside_extent(vehicles)
        pedestrians = self.discard_poses_outside_extent(pedestrians)

        # While running the dataset class in 'compute_stats' mode:
        if self.mode == "compute_stats":
            return len(vehicles), len(pedestrians)

        # Convert to fixed size arrays for batching
        vehicles, vehicle_masks = self.list_to_tensor(
            vehicles, self.max_vehicles, self.ts_h + 1, 5
        )
        pedestrians, pedestrian_masks = self.list_to_tensor(
            pedestrians, self.max_pedestrians, self.ts_h + 1, 5
        )

        surrounding_agent_representation = {
            "vehicles": vehicles,
            "vehicle_masks": vehicle_masks,
            "pedestrians": pedestrians,
            "pedestrian_masks": pedestrian_masks,
        }

        return surrounding_agent_representation

    def get_target_agent_global_pose(self, idx: int) -> Tuple[float, float, float]:
        """
        Returns global pose of target agent
        :param idx: data index
        :return global_pose: (x, y, yaw) or target agent in global co-ordinates
        """
        i_t, s_t = self.token_list[idx].split("_")
        sample_annotation = self.helper.get_sample_annotation(i_t, s_t)
        x, y = sample_annotation["translation"][:2]
        yaw = quaternion_yaw(Quaternion(sample_annotation["rotation"]))
        yaw = correct_yaw(yaw)
        global_pose = (x, y, yaw)

        return global_pose

    def get_lanes_around_agent(
        self, global_pose: Tuple[float, float, float], map_api: NuScenesMap
    ) -> Dict:
        """
        Gets lane polylines around the target agent
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param map_api: nuScenes map api
        :return lanes: Dictionary of lane polylines
        """
        x, y, _ = global_pose
        radius = max(self.map_extent)
        lanes = map_api.get_records_in_radius(x, y, radius, ["lane", "lane_connector"])
        lanes = lanes["lane"] + lanes["lane_connector"]
        lanes = map_api.discretize_lanes(lanes, self.polyline_resolution)

        return lanes

    def get_polygons_around_agent(
        self, global_pose: Tuple[float, float, float], map_api: NuScenesMap
    ) -> Dict:
        """
        Gets polygon layers around the target agent e.g. crosswalks, stop lines
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param map_api: nuScenes map api
        :return polygons: Dictionary of polygon layers, each type as a list of shapely Polygons
        """
        x, y, _ = global_pose
        radius = max(self.map_extent)
        record_tokens = map_api.get_records_in_radius(
            x, y, radius, ["stop_line", "ped_crossing"]
        )
        polygons = {k: [] for k in record_tokens.keys()}
        for k, v in record_tokens.items():
            for record_token in v:
                polygon_token = map_api.get(k, record_token)["polygon_token"]
                polygons[k].append(map_api.extract_polygon(polygon_token))

        return polygons

    def get_lane_node_feats(
        self,
        origin: Tuple,
        lanes: Dict[str, List[Tuple]],
        polygons: Dict[str, List[Polygon]],
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Generates vector HD map representation in the agent centric frame of reference
        :param origin: (x, y, yaw) of target agent in global co-ordinates
        :param lanes: lane centerline poses in global co-ordinates
        :param polygons: stop-line and cross-walk polygons in global co-ordinates
        :return:
        """

        # Convert lanes to list
        lane_ids = [k for k, v in lanes.items()]
        lanes = [v for k, v in lanes.items()]

        # Get flags indicating whether a lane lies on stop lines or crosswalks
        lane_flags = self.get_lane_flags(lanes, polygons)

        # Convert lane polylines to local coordinates:
        lanes = [
            np.asarray([self.global_to_local(origin, pose) for pose in lane])
            for lane in lanes
        ]

        # Concatenate lane poses and lane flags
        lane_node_feats = [
            np.concatenate((lanes[i], lane_flags[i]), axis=1) for i in range(len(lanes))
        ]

        # Split lane centerlines into smaller segments:
        lane_node_feats, lane_node_ids = self.split_lanes(
            lane_node_feats, self.polyline_length, lane_ids
        )

        return lane_node_feats, lane_node_ids

    def get_agents_of_type(self, idx: int, agent_type: str) -> List[np.ndarray]:
        """
        Returns surrounding agents of a particular class for a given sample
        :param idx: data index
        :param agent_type: 'human' or 'vehicle'
        :return: list of ndarrays of agent track histories.
        """
        i_t, s_t = self.token_list[idx].split("_")

        # Get agent representation in global co-ordinates
        origin = self.get_target_agent_global_pose(idx)

        # Load all agents for sample
        agent_details = self.helper.get_past_for_sample(
            s_t, seconds=self.t_h, in_agent_frame=False, just_xy=False
        )

        # print("agent_hist cropped:", agent_hist)
        # exit()

        # Filter for agent type
        agent_list = []
        agent_i_ts = []
        for k, v in agent_details.items():
            if (
                v
                and agent_type in v[0]["category_name"]
                and v[0]["instance_token"] != i_t
            ):
                ann_i_t = v[0]["instance_token"]
                agent_hist, _ = self.upsample_track(ann_i_t, s_t, in_agent_frame=False)
                if len(agent_hist) > self.ts_h:
                    agent_hist = agent_hist[-self.ts_h :]

                # Add present time to agent histories
                annotation = self.helper.get_sample_annotation(ann_i_t, s_t)
                present_pose = np.asarray(annotation["translation"][0:2]).reshape(1, 2)
                if len(agent_hist) > 0:
                    try:
                        agent_hist = np.concatenate((agent_hist, present_pose))
                    except ValueError as e:
                        print(present_pose.shape)
                        print(agent_hist.shape)
                        raise e
                else:
                    agent_hist = present_pose

                agent_list.append(agent_hist)
                agent_i_ts.append(ann_i_t)

        # Convert to target agent's frame of reference
        for agent in agent_list:
            for n, pose in enumerate(agent):
                local_pose = self.global_to_local(origin, (pose[0], pose[1], 0))
                agent[n] = np.asarray([local_pose[0], local_pose[1]])

        # Flip history to have most recent time stamp last and extract past motion states
        for n, agent in enumerate(agent_list):
            xy = agent
            motion_states = self.get_past_motion_states(agent_i_ts[n], s_t, len(xy))
            # motion_states = motion_states[-len(xy) :]
            agent_list[n] = np.concatenate((xy, motion_states), axis=1)

        return agent_list

    def discard_poses_outside_extent(
        self, pose_set: List[np.ndarray], ids: List[str] = None
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[str]]]:
        """
        Discards lane or agent poses outside predefined extent in target agent's frame of reference.
        :param pose_set: agent or lane polyline poses
        :param ids: annotation record tokens for pose_set. Only applies to lanes.
        :return: Updated pose set
        """
        updated_pose_set = []
        updated_ids = []

        for m, poses in enumerate(pose_set):
            flag = False
            for n, pose in enumerate(poses):
                if (
                    self.map_extent[0] <= pose[0] <= self.map_extent[1]
                    and self.map_extent[2] <= pose[1] <= self.map_extent[3]
                ):
                    flag = True

            if flag:
                updated_pose_set.append(poses)
                if ids is not None:
                    updated_ids.append(ids[m])

        if ids is not None:
            return updated_pose_set, updated_ids
        else:
            return updated_pose_set

    def load_stats(self) -> Dict[str, int]:
        """
        Function to load dataset statistics like max surrounding agents, max nodes, max edges etc.
        """
        filename = os.path.join(self.data_dir, "stats.pickle")
        if not os.path.isfile(filename):
            raise Exception(
                "Could not find dataset statistics. Please run the dataset in compute_stats mode"
            )

        with open(filename, "rb") as handle:
            stats = pickle.load(handle)

        return stats

    def get_past_motion_states(self, i_t, s_t, ts, factor=5):
        past_anns = self.helper.get_past_for_agent(
            i_t, s_t, seconds=self.max_t, in_agent_frame=False, just_xy=False
        )
        past_anns = past_anns[::-1]  # flip timestamps
        # print("past_anns len", len(past_anns))
        # print("hist len", len(hist))

        velocities = np.zeros(ts)
        accelerations = np.zeros(ts)
        yaw_rates = np.zeros(ts)
        for k, ann in enumerate(past_anns):
            if k >= ts:
                break

            ann_s_t = ann["sample_token"]

            velocity = self.helper.get_velocity_for_agent(i_t, ann_s_t)
            velocities[-(k + 1)] = velocity

            acceleration = self.helper.get_acceleration_for_agent(i_t, ann_s_t)
            accelerations[-(k + 1)] = acceleration

            yaw_rate = self.helper.get_heading_change_rate_for_agent(i_t, ann_s_t)
            yaw_rates[-(k + 1)] = yaw_rate

        velocities = np.nan_to_num(velocities)
        accelerations = np.nan_to_num(accelerations)
        yaw_rates = np.nan_to_num(yaw_rates)

        if len(past_anns) > 1:
            velocities_intp = upsample1d(velocities, factor=factor)
            accelerations_intp = upsample1d(accelerations, factor=factor)
            yaw_rates_intp = upsample1d(yaw_rates, factor=factor)
        else:
            velocities_intp = np.repeat(velocities, factor)
            accelerations_intp = np.repeat(accelerations, factor)
            yaw_rates_intp = np.repeat(yaw_rates, factor)

        motion_states = np.concatenate(
            [
                velocities_intp[:, None],
                accelerations_intp[:, None],
                yaw_rates_intp[:, None],
            ],
            axis=1,
        )

        motion_states = motion_states[-ts:]
        return motion_states

    def _get_past_motion_states(self, i_t, s_t):
        """
        Returns past motion states: v, a, yaw_rate for a given instance and sample token over self.t_h seconds
        """
        motion_states = np.zeros((self.ts_h + 1, 3))
        motion_states[-1, 0] = self.helper.get_velocity_for_agent(i_t, s_t)
        motion_states[-1, 1] = self.helper.get_acceleration_for_agent(i_t, s_t)
        motion_states[-1, 2] = self.helper.get_heading_change_rate_for_agent(i_t, s_t)
        hist, fut = self.upsample_track(i_t, s_t)
        # print("hist shape", hist.shape)
        # print("fut shape", fut.shape)
        # exit()

        for k in range(len(hist)):
            motion_states[-(k + 2), 0] = self.helper.get_velocity_for_agent(
                i_t, hist[k]["sample_token"]
            )
            motion_states[-(k + 2), 1] = self.helper.get_acceleration_for_agent(
                i_t, hist[k]["sample_token"]
            )
            motion_states[-(k + 2), 2] = self.helper.get_heading_change_rate_for_agent(
                i_t, hist[k]["sample_token"]
            )

        motion_states = np.nan_to_num(motion_states)
        return motion_states

    @staticmethod
    def global_to_local(origin: Tuple, global_pose: Tuple) -> Tuple:
        """
        Converts pose in global co-ordinates to local co-ordinates.
        :param origin: (x, y, yaw) of origin in global co-ordinates
        :param global_pose: (x, y, yaw) in global co-ordinates
        :return local_pose: (x, y, yaw) in local co-ordinates
        """
        # Unpack
        global_x, global_y, global_yaw = global_pose
        origin_x, origin_y, origin_yaw = origin

        # Translate
        local_x = global_x - origin_x
        local_y = global_y - origin_y

        # Rotate
        global_yaw = correct_yaw(global_yaw)
        theta = np.arctan2(
            -np.sin(global_yaw - origin_yaw), np.cos(global_yaw - origin_yaw)
        )

        r = np.asarray(
            [
                [np.cos(np.pi / 2 - origin_yaw), np.sin(np.pi / 2 - origin_yaw)],
                [-np.sin(np.pi / 2 - origin_yaw), np.cos(np.pi / 2 - origin_yaw)],
            ]
        )
        local_x, local_y = np.matmul(r, np.asarray([local_x, local_y]).transpose())

        local_pose = (local_x, local_y, theta)

        return local_pose

    @staticmethod
    def split_lanes(
        lanes: List[np.ndarray], max_len: int, lane_ids: List[str]
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Splits lanes into roughly equal sized smaller segments with defined maximum length
        :param lanes: list of lane poses
        :param max_len: maximum admissible length of polyline
        :param lane_ids: list of lane ID tokens
        :return lane_segments: list of smaller lane segments
                lane_segment_ids: list of lane ID tokens corresponding to original lane that the segment is part of
        """
        lane_segments = []
        lane_segment_ids = []
        for idx, lane in enumerate(lanes):
            n_segments = int(np.ceil(len(lane) / max_len))
            n_poses = int(np.ceil(len(lane) / n_segments))
            for n in range(n_segments):
                lane_segment = lane[n * n_poses : (n + 1) * n_poses]
                lane_segments.append(lane_segment)
                lane_segment_ids.append(lane_ids[idx])

        return lane_segments, lane_segment_ids

    @staticmethod
    def get_lane_flags(
        lanes: List[List[Tuple]], polygons: Dict[str, List[Polygon]]
    ) -> List[np.ndarray]:
        """
        Returns flags indicating whether each pose on lane polylines lies on polygon map layers
        like stop-lines or cross-walks
        :param lanes: list of lane poses
        :param polygons: dictionary of polygon layers
        :return lane_flags: list of ndarrays with flags
        """

        lane_flags = [np.zeros((len(lane), len(polygons.keys()))) for lane in lanes]
        for lane_num, lane in enumerate(lanes):
            for pose_num, pose in enumerate(lane):
                point = Point(pose[0], pose[1])
                for n, k in enumerate(polygons.keys()):
                    polygon_list = polygons[k]
                    for polygon in polygon_list:
                        if polygon.contains(point):
                            lane_flags[lane_num][pose_num][n] = 1
                            break

        return lane_flags

    @staticmethod
    def list_to_tensor(
        feat_list: List[np.ndarray], max_num: int, max_len: int, feat_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of sequential features (e.g. lane polylines or agent history) to fixed size numpy arrays for
        forming mini-batches

        :param feat_list: List of sequential features
        :param max_num: Maximum number of sequences in List
        :param max_len: Maximum length of each sequence
        :param feat_size: Feature dimension
        :return: 1) ndarray of features of shape [max_num, max_len, feat_dim]. Has zeros where elements are missing,
            2) ndarray of binary masks of shape [max_num, max_len, feat_dim]. Has ones where elements are missing.
        """
        feat_array = np.zeros((max_num, max_len, feat_size))
        mask_array = np.ones((max_num, max_len, feat_size))
        for n, feats in enumerate(feat_list):
            feat_array[n, : len(feats), :] = feats
            mask_array[n, : len(feats), :] = 0

        return feat_array, mask_array

    @staticmethod
    def flip_horizontal(data: Dict):
        """
        Helper function to randomly flip some samples across y-axis for data augmentation
        :param data: Dictionary with inputs and ground truth values.
        :return: data: Dictionary with inputs and ground truth values fligpped along y-axis.
        """
        # Flip target agent
        hist = data["inputs"]["target_agent_representation"]
        hist[:, 0] = -hist[:, 0]  # x-coord
        hist[:, 4] = -hist[:, 4]  # yaw-rate
        data["inputs"]["target_agent_representation"] = hist

        # Flip lane node features
        lf = data["inputs"]["map_representation"]["lane_node_feats"]
        lf[:, :, 0] = -lf[:, :, 0]  # x-coord
        lf[:, :, 2] = -lf[:, :, 2]  # yaw
        data["inputs"]["map_representation"]["lane_node_feats"] = lf

        # Flip surrounding agents
        vehicles = data["inputs"]["surrounding_agent_representation"]["vehicles"]
        vehicles[:, :, 0] = -vehicles[:, :, 0]  # x-coord
        vehicles[:, :, 4] = -vehicles[:, :, 4]  # yaw-rate
        data["inputs"]["surrounding_agent_representation"]["vehicles"] = vehicles

        peds = data["inputs"]["surrounding_agent_representation"]["pedestrians"]
        peds[:, :, 0] = -peds[:, :, 0]  # x-coord
        peds[:, :, 4] = -peds[:, :, 4]  # yaw-rate
        data["inputs"]["surrounding_agent_representation"]["pedestrians"] = peds

        # Flip groud truth trajectory
        fut = data["ground_truth"]["traj"]
        fut[:, 0] = -fut[:, 0]  # x-coord
        data["ground_truth"]["traj"] = fut

        return data
