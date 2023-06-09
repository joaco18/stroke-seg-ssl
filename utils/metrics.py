# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import logging
import numpy as np
from medpy.metric.binary import hd

logging.basicConfig(level=logging.INFO)


def mutual_information(vol1: np.ndarray, vol2: np.ndarray) -> float:
    """Computes the mutual information between two images/volumes
    Args:
        vol1 (np.ndarray): First of two image/volumes to compare
        vol2 (np.ndarray): Second of two image/volumes to compare
    Returns:
        (float): Mutual information
    """
    # Get the histogram
    hist_2d, x_edges, y_edges = np.histogram2d(
        vol1.ravel(), vol2.ravel(), bins=255)
    # Get pdf
    pxy = hist_2d / float(np.sum(hist_2d))
    # Marginal pdf for x over y
    px = np.sum(pxy, axis=1)
    # Marginal pdf for y over x
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def dice_score(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute dice across classes. The corresponding labels should be
    previously matched.
    Args:
        gt (np.ndarray): Grounth truth
        pred (np.ndarray): Labels
    Returns:
        (np.ndarray): Dice scores per tissue [CSF, GM, WM]
    """
    classes = np.unique(gt[gt != 0]).astype(int)
    dice = np.zeros((len(classes)))
    for i, lab in enumerate(classes):
        bin_pred = np.where(pred == lab, 1, 0)
        bin_gt = np.where(gt == lab, 1, 0)
        if np.sum(bin_gt) == 0:
            dice[i] = None
        else:
            dice[i] = (np.sum(bin_pred[bin_gt == 1]) * 2.0) / (np.sum(bin_pred) + np.sum(bin_gt))
    return dice


def hausdorff(gt: np.ndarray, pred: np.ndarray, voxelspacing: tuple):
    """Compute relative absolute volume difference across classes. The corresponding labels should be
    previously matched.
    Args:
        gt (np.ndarray): Grounth truth
        pred (np.ndarray): Labels
        voxelspacing (tuple): voxel_spacing
    Returns:
        list: Dice scores per tissue [CSF, GM, WM]
    """
    classes = np.unique(gt[gt != 0]).astype(int)
    hd_values = np.zeros((len(classes)))
    for i in classes:
        bin_pred = np.where(pred == i, 1, 0)
        bin_gt = np.where(gt == i, 1, 0)
        try:
            hd_values[i-1] = hd(bin_pred, bin_gt, voxelspacing=voxelspacing)
        except Exception as e:
            logging.warning(e)
            raise Exception('Hausroff failed')
            hd_values[i-1] = np.nan
    return hd_values.tolist()
