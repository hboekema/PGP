# Import datasets
from typing import Dict, List, Union

from datasets.interface import TrajectoryDataset
from datasets.nuScenes10Hz.nuScenes_graphs import NuScenesGraphs10Hz
from datasets.nuScenes10Hz.nuScenes_raster import NuScenesRaster10Hz
from datasets.nuScenes10Hz.nuScenes_vector import NuScenesVector10Hz
from datasets.nuScenes.nuScenes_graphs import NuScenesGraphs
from datasets.nuScenes.nuScenes_raster import NuScenesRaster
from datasets.nuScenes.nuScenes_vector import NuScenesVector
from datasets.VOD.vod_graphs import VODGraphs
from datasets.VOD.vod_raster import VODRaster
from datasets.VOD.vod_traj import VODTrajectories
from datasets.VOD.vod_vector import VODVector
from metrics.covernet_loss import CoverNetLoss
from metrics.goal_pred_nll import GoalPredictionNLL
from metrics.min_ade import MinADEK
from metrics.min_fde import MinFDEK
from metrics.miss_rate import MissRateK
# Import metrics
from metrics.mtp_loss import MTPLoss
from metrics.pi_bc import PiBehaviorCloning
from models.aggregators.concat import Concat
from models.aggregators.global_attention import GlobalAttention
from models.aggregators.goal_conditioned import GoalConditioned
from models.aggregators.identity import Identity
from models.aggregators.pgp import PGP
from models.decoders.covernet import CoverNet
from models.decoders.cvm import CVMDecoder
from models.decoders.lvm import LVM
from models.decoders.lvm_ca import LVM_CA
from models.decoders.mtp import MTP
from models.decoders.multipath import Multipath
from models.encoders.cvm import CVMEncoder
from models.encoders.pgp_encoder import PGPEncoder
from models.encoders.pgpca_encoder import PGPCAEncoder
from models.encoders.pgpvod_encoder import PGPVODEncoder
from models.encoders.polyline_subgraph import PolylineSubgraphs
from models.encoders.raster_encoder import RasterEncoder
# Import models
from models.model import PredictionModel
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper as NSPredictHelper
from vod import VOD
from vod.prediction import PredictHelper


# Datasets
def initialize_dataset(dataset_type: str, args: List) -> TrajectoryDataset:
    """
    Helper function to initialize appropriate dataset by dataset type string
    """
    # TODO: Add more datasets as implemented
    dataset_classes = {
        "nuScenes10Hz_single_agent_raster": NuScenesRaster10Hz,
        "nuScenes10Hz_single_agent_vector": NuScenesVector10Hz,
        "nuScenes10Hz_single_agent_graphs": NuScenesGraphs10Hz,
        "nuScenes_single_agent_raster": NuScenesRaster,
        "nuScenes_single_agent_vector": NuScenesVector,
        "nuScenes_single_agent_graphs": NuScenesGraphs,
        "VOD_single_agent_traj": VODTrajectories,
        "VOD_single_agent_raster": VODRaster,
        "VOD_single_agent_vector": VODVector,
        "VOD_single_agent_graphs": VODGraphs,
    }
    return dataset_classes[dataset_type](*args)


def get_specific_args(
    dataset_name: str, data_root: str, version: str = None, cfg: Dict = None
) -> List:
    """
    Helper function to get dataset specific arguments.
    """
    # TODO: Add more datasets as implemented
    specific_args = []
    if "nuScenes" in dataset_name:
        ns = NuScenes(version, dataroot=data_root)
        pred_helper = NSPredictHelper(ns)
        specific_args.append(pred_helper)
    elif dataset_name == "VOD":
        vod_ds = VOD(version, dataroot=data_root)
        pred_helper = PredictHelper(vod_ds)
        specific_args.extend([pred_helper, cfg["mode"]])

    return specific_args


# Models
def initialize_prediction_model(
    encoder_type: str,
    aggregator_type: str,
    decoder_type: str,
    encoder_args: Dict,
    aggregator_args: Union[Dict, None],
    decoder_args: Dict,
):
    """
    Helper function to initialize appropriate encoder, aggegator and decoder models
    """
    encoder = initialize_encoder(encoder_type, encoder_args)
    aggregator = initialize_aggregator(aggregator_type, aggregator_args)
    decoder = initialize_decoder(decoder_type, decoder_args)
    model = PredictionModel(encoder, aggregator, decoder)

    return model


def initialize_encoder(encoder_type: str, encoder_args: Dict):
    """
    Initialize appropriate encoder by type.
    """
    # TODO: Update as we add more encoder types
    encoder_mapping = {
        "raster_encoder": RasterEncoder,
        "polyline_subgraphs": PolylineSubgraphs,
        "pgp_encoder": PGPEncoder,
        "pgpvod_encoder": PGPVODEncoder,
        "pgpca_encoder": PGPCAEncoder,
        "cvm": CVMEncoder,
    }

    return encoder_mapping[encoder_type](encoder_args)


def initialize_aggregator(aggregator_type: str, aggregator_args: Union[Dict, None]):
    """
    Initialize appropriate aggregator by type.
    """
    # TODO: Update as we add more aggregator types
    aggregator_mapping = {
        "concat": Concat,
        "global_attention": GlobalAttention,
        "gc": GoalConditioned,
        "pgp": PGP,
        "identity": Identity,
    }

    if aggregator_args:
        return aggregator_mapping[aggregator_type](aggregator_args)
    else:
        return aggregator_mapping[aggregator_type]()


def initialize_decoder(decoder_type: str, decoder_args: Dict):
    """
    Initialize appropriate decoder by type.
    """
    # TODO: Update as we add more decoder types
    decoder_mapping = {
        "mtp": MTP,
        "multipath": Multipath,
        "covernet": CoverNet,
        "lvm": LVM,
        "lvm-ca": LVM_CA,
        "cvm": CVMDecoder,
    }

    return decoder_mapping[decoder_type](decoder_args)


# Metrics
def initialize_metric(metric_type: str, metric_args: Dict = None):
    """
    Initialize appropriate metric by type.
    """
    # TODO: Update as we add more metrics
    metric_mapping = {
        "mtp_loss": MTPLoss,
        "covernet_loss": CoverNetLoss,
        "min_ade_k": MinADEK,
        "min_fde_k": MinFDEK,
        "miss_rate_k": MissRateK,
        "pi_bc": PiBehaviorCloning,
        "goal_pred_nll": GoalPredictionNLL,
    }

    if metric_args is not None:
        return metric_mapping[metric_type](metric_args)
    else:
        return metric_mapping[metric_type]()
