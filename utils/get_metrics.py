# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import json
import pandas as pd

import numpy as np
import multiprocessing as mp
import medpy.metric.binary as mpy_metrics
import SimpleITK as sitk

from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
from scipy.ndimage import label

from utils.utils import remove_small_lesions_array


def get_dataset_from_subject(subject: str) -> str:
    if 'train' in subject:
        return 'apis'
    else:
        return 'aisd'


def get_metrics_volume(gt_path: Path, pred_path: Path, min_les_vol: int, remove_small_preds: bool):
    gt = sitk.ReadImage(str(gt_path))
    spacing = gt.GetSpacing()
    gt = sitk.GetArrayFromImage(gt)
    gt = np.where(gt > 0, 1, 0).astype(int)
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
    pred = np.where(pred > 0, 1, 0).astype(int)
    voxel_vol = np.prod(np.asarray(spacing))
    gt = remove_small_lesions_array(gt, min_les_vol=min_les_vol, unit_volume=voxel_vol)
    if remove_small_preds:
        pred = remove_small_lesions_array(pred, min_les_vol=3000, unit_volume=voxel_vol)
    metrics = get_metrics(gt, pred)
    metrics.update({
        'min_les_vol': min_les_vol,
        'remove_small_pred': remove_small_preds
    })
    return metrics


def get_metrics(gt: np.ndarray, pred: np.ndarray):
    gt_vol = np.sum(gt)
    pred_vol = np.sum(pred)
    se = np.ones((3, 3, 3))
    if gt_vol == 0:
        if pred_vol == 0:
            metrics = {
                'dice': 1, 'jaccard': 1, 'asd': None, 'assd': None,
                'hd': None, 'hd95': None, 'ppv': None, 'tnr': None,
                'tpr': None, 'pred_vol': 0, 'gt_vol': 0, 'ravd': None,
                'avd': 0, 'lesion': False, 'n_cc_gt': 0, 'n_cc_pred': 0,
                'alcd': 0
            }
        else:
            label_pred, _ = label(pred, se)
            n_cc_pred = len(np.unique(label_pred)) - 1
            metrics = {
                'dice': 0, 'jaccard': 0, 'asd': None, 'assd': None,
                'hd': None, 'hd95': None, 'ppv': None, 'tnr': None,
                'tpr': None, 'pred_vol': pred_vol, 'gt_vol': 0,
                'ravd': None, 'avd': pred_vol, 'lesion': False,
                'n_cc_gt': 0, 'n_cc_pred': n_cc_pred, 'alcd': n_cc_pred,
            }
    elif pred_vol == 0:
        label_gt, _ = label(gt, se)
        n_cc_gt = len(np.unique(label_gt)) - 1
        metrics = {
            'dice': 0, 'jaccard': 0, 'asd': None, 'assd': None,
            'hd': None, 'hd95': None, 'ppv': None, 'tnr': None,
            'tpr': None, 'pred_vol': 0, 'gt_vol': gt_vol,
            'ravd': None, 'avd': gt_vol, 'lesion': True,
            'n_cc_gt': n_cc_gt, 'n_cc_pred': 0, 'alcd': n_cc_gt,
        }
    else:
        label_pred, _ = label(pred, se)
        n_cc_pred = len(np.unique(label_pred)) - 1
        label_gt, _ = label(gt, se)
        n_cc_gt = len(np.unique(label_gt)) - 1
        metrics = {
            'dice': mpy_metrics.dc(pred, gt),
            'jaccard': mpy_metrics.jc(pred, gt),
            'asd': mpy_metrics.asd(pred, gt),
            'assd': mpy_metrics.assd(pred, gt),
            'hd': mpy_metrics.hd(pred, gt),
            'hd95': mpy_metrics.hd95(pred, gt),
            'ppv': mpy_metrics.positive_predictive_value(pred, gt),
            'tnr': mpy_metrics.true_negative_rate(pred, gt),
            'tpr': mpy_metrics.true_positive_rate(pred, gt),
            'pred_vol': np.sum(pred),
            'gt_vol': np.sum(gt),
            'ravd': mpy_metrics.ravd(pred, gt),
            'avd': np.abs(np.sum(pred) - np.sum(gt)),
            'lesion': True,
            'n_cc_gt': n_cc_gt,
            'n_cc_pred': n_cc_pred,
            'alcd': np.abs(n_cc_gt - n_cc_pred),
        }
    return metrics


def get_metrics_df_mp(exp_cfg: Dict) -> pd.DataFrame:
    return get_metrics_df(**exp_cfg)


def get_metrics_df(task_id: str, folds: List[int], epochs: List[str],
                   training: str, input_kind: str, exp_name: str, force_compute: bool,
                   trained_models_path: Path, preprocessed_path: Path, raw_path: Path,
                   used_dataset: str, run: int, freezing: int, train_perc: int,
                   symmetry_aug: bool = False, old: bool = False) -> pd.DataFrame:

    kind = f'_{exp_name}' if exp_name != '' else exp_name
    dataset = f'Dataset{task_id}_AIS'
    results_path = trained_models_path
    if old:
        results_path = results_path / 'old_experiments' / \
            f'{dataset}{kind}/nnUNetTrainerCfg__nnUNetPlansSSL__3d_fullres'
    else:
        results_path = results_path / \
            f'{dataset}{kind}/nnUNetTrainerCfg__nnUNetPlansSSL__3d_fullres'
    label_path = raw_path / f'Dataset{task_id}_AIS/labelsTr'
    splits_path = preprocessed_path / f'{dataset}/raw_splits_test.json'
    with open(splits_path, 'r') as file:
        splits = json.load(file)
    with open(splits_path, 'r') as file:
        splits = json.load(file)
    splits = [{'val': [Path(i[0]).name.split('_')[0] for i in splits[0]]}]

    dataframe = []
    for fold in folds:
        csv_path = results_path / f'test_metrics_fold{fold}.csv'
        for epoch in epochs:
            local_results_path = results_path / f'fold_{fold}/test_{epoch}/'
            local_csv_path = local_results_path/f'test_metrics_fold{fold}_epoch{epoch}.csv'
            if (not local_csv_path.exists()) or force_compute:
                val_set = splits[fold]['val']
                if local_results_path.exists():
                    rows = []
                    nii_files = [file for file in local_results_path.iterdir()
                                 if file.name.endswith('.nii.gz')]
                    nii_files = [file for file in nii_files
                                 if file.name.replace('.nii.gz', '') in val_set]
                    for pred_path in tqdm(nii_files, desc="SUB: ", position=1,
                                          leave=False, total=len(nii_files)):
                        subject = pred_path.name.replace('.nii.gz', '')
                        gt_path = label_path / pred_path.name
                        base = {'subject': subject,
                                'dataset': get_dataset_from_subject(subject),
                                'used_dataset': used_dataset,
                                'exp_name': exp_name,
                                'long': True if 'long' in exp_name else False,
                                'training': training,
                                'fold': fold,
                                'epoch': epoch,
                                'task_id': task_id,
                                'epoch': epoch,
                                'input_kind': input_kind,
                                'freezing': freezing,
                                'run': 'ensemble' if exp_name == 'ensemble' else run,
                                'sym': 'SA' if symmetry_aug else 'NSA',
                                'train_perc': train_perc}
                        for remove_small in [True, False]:
                            for min_vol in [0, 3000, 7500, 15000, 30000, 50000, 70000]:
                                metrics = get_metrics_volume(
                                    gt_path, pred_path, min_vol, remove_small)
                                row = deepcopy(base)
                                row.update(metrics)
                                rows.append(row)
                    df = pd.DataFrame(rows)
                    df.to_csv(local_csv_path)
                    dataframe.append(df)
                else:
                    raise Exception(f'{local_results_path}\nExperiment with dataset: {dataset},'
                                    f' fold: {fold}, epoch: {epoch} does not exist. Avoiding')
            else:
                df = pd.read_csv(local_csv_path, index_col=0)
                dataframe.append(df)
    df = pd.concat(dataframe, ignore_index=True)
    df.to_csv(csv_path)
    return df


def main():
    epochs = ['best']
    constats = {
        'force_compute': True,
        'trained_models_path': Path('<nnUNet_TRAINED_MODELS_PATH>'),
        'preprocessed_path': Path('<nnUNet_PREPROCESSED_PATH>'),
        'raw_path': Path('<nnUNet_RAW_PATH>'),
        'folds': [0],
        'epochs': epochs,
        'old': False
    }

    experiments_cfgs = [
        {'task_id': '038', 'exp_name': 'all_ncct_004_long_fr_00_ensemble', 'training': 'SSL',
         'input_kind': 'ncct', 'used_dataset': 'all', 'run': 0, 'freezing': 0,
         'symmetry_aug': True, 'train_perc': 100},
    ]
    full_df = []
    for exp_cfg in experiments_cfgs:
        exp_cfg.update(constats)
    with mp.Pool(mp.cpu_count()) as pool:
        for result in tqdm(pool.imap(get_metrics_df_mp, experiments_cfgs),
                           total=len(experiments_cfgs)):
            full_df.append(result)


if __name__ == '__main__':
    main()
