# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import pickle
import numpy as np
import nibabel as nib
from pathlib import Path
from skimage.measure import block_reduce
from typing import Tuple

CTP_DB_MEAN = 86.47372173768187
CTP_DB_STD = 320.94952673512597


def load_ctp_array_and_curves(
    ctp_path: Path, spatial_down_factor: Tuple[int, int, int, int],
    gt_path: Path, data_aug: bool = False, ct0_path: Path = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads the ctp, performs data augmentation if required and provides
        the curves of aif and vof, according to the gt file.
    Args:
        ctp_path (Path): Path to the ctp image.
        spatial_down_factor (Tuple[int, int, int, int]):
            Downsampling factors to use in each dimesion.
        gt_path (Path): Path to the gt pickle file.
        data_aug (bool, optional): Whether to do data augmentation ot not.
            Defaults to False.
        ct0_path (Path, optional): If data augmentation is required, the
            path to the baseline image must be provided. Defaults to None.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            (np.ndarray): Augmented 4d image.
            (np.ndarray): AIF curve.
            (np.ndarray): VOF curve.
    """

    # upload case and get volume
    myvol = nib.load(ctp_path)
    vol_data = myvol.get_fdata()

    # pre-processing
    if data_aug:
        # peak concentration scaling
        vol_data = da_concentr_scaling_mimic(vol_data, ct0_path, [.4, 2])
        # random bolus delay
        vol_data = da_bolus_delay_mimic(vol_data, 6)

    # donwsample volumes in time
    vol_data = downsample_vol_time(vol_data, 43)
    # z-score normalization
    vol_data = normalize_vol(vol_data)

    # get AIF & VOF ground truth curves
    aif_curve, vof_curve = get_aif_vof_gt(vol_data, gt_path)
    if spatial_down_factor != (1, 1, 1, 1):
        vol_data = downsample_vol_space(volume=vol_data, factor=spatial_down_factor)

    return vol_data, aif_curve, vof_curve


def downsample_vol_time(volume: np.ndarray, frames: int) -> np.ndarray:
    """ Takes first frames of the 4D volume.
    Args:
        volume (np.ndarray): CTP 4d volume
        frames (int): Frames to consider
    Returns:
        np.ndarray: Downsampled volume
    """
    return volume[:, :, :, 0:frames]


def da_bolus_delay_mimic(volume: np.ndarray, frames: int) -> np.ndarray:
    """ Data Augmentation. Simulate bolus injection delay by shifting frames.
    Args:
        volume (np.ndarray): CTP 4d volume.
        frames (int): Frames to consider.
    Returns:
        np.ndarray: Augmented version of the volume.
    """
    frames2shift = np.round(np.random.uniform(-frames, frames+1, 1)[0])
    if frames2shift > 0:
        rep_initvol = np.repeat(
            np.expand_dims(volume[:, :, :, 0], axis=3), frames2shift, axis=3)
        shifted_vol = volume[:, :, :, 0:(volume.shape[3] - frames2shift.astype(int))]
        volume = np.concatenate((rep_initvol, shifted_vol), axis=3)
    elif frames2shift < 0:
        rep_lastvol = np.repeat(
            np.expand_dims(volume[:, :, :, -1], axis=3), abs(frames2shift), axis=3)
        shifted_vol = volume[:, :, :, abs(frames2shift.astype(int)):volume.shape[3]]
        volume = np.concatenate((shifted_vol, rep_lastvol), axis=3)
    return volume


def normalize_vol(volume: np.ndarray) -> np.ndarray:
    """ Normalize 4D volumes to 0 mean and std 1.
        Use mean (std) of whole db.
    Args:
        volume (np.ndarray): CTP 4D volume.
    Returns:
        np.ndarray: Normalized volume.
    """
    return (volume - CTP_DB_MEAN) / CTP_DB_STD


def da_concentr_scaling_mimic(
    volume: np.ndarray, ct0_path: Path, scale_int: list
) -> np.ndarray:
    """ Simulate peak concentration changes by random scaling.
    Args:
        volume (np.ndarray): CTP 4D volume.
        ct0_path (Path): Path to the baseline image
        scale_int (list): Scale factor to use in the augmentation
    Returns:
        np.ndarray: _description_
    """
    myvol0 = nib.load(ct0_path)
    vol_data0 = myvol0.get_fdata()
    # generate scaling factor.
    scaling_factor = np.random.uniform(scale_int[0], scale_int[1], 1)[0]
    # remove baseline and scale.
    vol_data0 = np.repeat(np.expand_dims(vol_data0, axis=3), volume.shape[3], axis=3)
    volume = (volume - vol_data0) * scaling_factor
    volume += vol_data0
    return volume


def get_aif_vof_gt(
    ctp_array: np.ndarray, gt_pkl_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Given a ground truth pickle file containing the location in the image where
        AIF and VIF can be measured, returns the AIF and VIF curves as arrays.
    Args:
        ctp_array (np.ndarray): 4D array of the CTP image.
        gt_pkl_path (Path): ground truth file containing the venous and arterial locations.
    Returns:
        Tuple[np.ndarray, np.ndarray]: aif_curve and vof curve
    """
    # get AIF and VOF ground truth curves from a pkl file.
    with open(gt_pkl_path, "rb") as f:
        gt = pickle.load(f)
    aif_voxel_coord = tuple(int(c) for c in gt["aif_voxel_coordinate"])
    vof_voxel_coord = tuple(int(c) for c in gt["vof_voxel_coordinate"])

    # get AIF and VOF coordinates
    aif_curve = ctp_array[aif_voxel_coord[0], aif_voxel_coord[1], aif_voxel_coord[2], :]
    vof_curve = ctp_array[vof_voxel_coord[0], vof_voxel_coord[1], vof_voxel_coord[2], :]

    return aif_curve, vof_curve


def downsample_vol_space(
    volume: np.ndarray, factor: Tuple[int, int, int, int]
) -> np.ndarray:
    """Downsample volume spatially.
    Args:
        volume (np.ndarray): 4D array of the CTP image.
        factor (Tuple[int, int, int, int]):
            Downsample factor in each dimension.
    Returns:
        (np.ndarray): Downsampled version of the image.
    """
    return block_reduce(volume, block_size=factor, func=np.mean)
