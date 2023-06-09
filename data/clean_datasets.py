# -*-coding:utf-8 -*-
'''
@Time    :   2023/02/08 16:35:14
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import argparse
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
from typing import List

from clean_aisd import clean_aisd_data
from clean_apis import clean_apis_data
from clean_isles18 import clean_isles18_data
from clean_tbi import clean_center_tbi_data
from data_utils import FILE_TYPES, METADATA_FIELDS

logging.basicConfig(level=logging.INFO)
repository_base_path = Path().resolve()


def quantify_lesion_sizes(df: pd.DataFrame, bidsdata_path: Path) -> pd.DataFrame:
    # Quantify lesion sizes
    selection = (df['discard'] == 'n') & ((df['msk'] != '-') | (df['msk-tilt'] != '-'))
    new_columns = [
        'n_lesions', 'min_lesion_vol', 'mean_lesion_vol', 'median_lesion_vol', 'max_lesion_vol'
    ]
    for col in new_columns:
        df[col] = '-'
    df.loc[(df.dataset_name == 'tbi') | (~df.ais), 'n_lesions'] = 0

    logging.info('Getting number of lesions and sizes')
    for _, row in tqdm(df.loc[selection].iterrows(), total=sum(selection)):
        msk_path = row['msk-tilt'] if (row['msk-tilt'] != '-') else row['msk']
        msk_path = bidsdata_path / row['dataset_name'] / msk_path
        img = sitk.ReadImage(str(msk_path))
        vox_dims = img.GetSpacing()
        unit_volume = vox_dims[0] * vox_dims[1] * vox_dims[2]
        msk_array = sitk.GetArrayFromImage(img)
        label_img, n_components = ndimage.label(msk_array, structure=np.ones((3, 3, 3)))
        _, counts = np.unique(label_img, return_counts=True)
        volumes = counts[1:] * unit_volume
        df.loc[df.subject == row['subject'], 'n_lesions'] = n_components - 1
        df.loc[df.subject == row['subject'], 'min_lesion_vol'] = np.min(volumes)
        df.loc[df.subject == row['subject'], 'median_lesion_vol'] = np.mean(volumes)
        df.loc[df.subject == row['subject'], 'mean_lesion_vol'] = np.median(volumes)
        df.loc[df.subject == row['subject'], 'max_lesion_vol'] = np.max(volumes)
    return df


def get_partitions_for_normals(
    df: pd.DataFrame, selection: np.ndarray, partition_str: str, seed: int, splits: list
) -> pd.DataFrame:
    for db_name in df.loc[selection, 'dataset_name'].unique():
        sampling_pool = selection & (df.dataset_name == db_name)
        sub_df = df.loc[sampling_pool, :]
        sub_df = sub_df.drop_duplicates(subset=['patient']).copy()
        if len(splits) == 3:
            train_subjs = sub_df.patient.sample(frac=0.7, random_state=seed).tolist()
            df.loc[df.patient.isin(train_subjs), partition_str] = 'train'

            sampling_pool = sampling_pool & (~df.patient.isin(train_subjs))
            sub_df = df.loc[sampling_pool, :]
            sub_df = sub_df.drop_duplicates(subset=['patient']).copy()
            val_subjs = sub_df.patient.sample(frac=2/3, random_state=seed).tolist()
            df.loc[df.patient.isin(val_subjs), partition_str] = 'validation'

            test_leftovers = sampling_pool & (~df.patient.isin(val_subjs))
            sub_df = df.loc[test_leftovers, :]
            sub_df = sub_df.drop_duplicates(subset=['patient']).copy()
            test_subjs = sub_df.patient.tolist()
            df.loc[df.patient.isin(test_subjs), partition_str] = 'test'
        else:
            train_subjs = sub_df.patient.sample(frac=7/9, random_state=seed).tolist()
            df.loc[df.patient.isin(train_subjs), partition_str] = 'train'

            sampling_pool = sampling_pool & (~df.patient.isin(train_subjs))
            sub_df = df.loc[sampling_pool, :]
            sub_df = sub_df.drop_duplicates(subset=['patient']).copy()
            val_subjs = sub_df.patient.tolist()
            df.loc[df.patient.isin(val_subjs), partition_str] = 'validation'
    return df


def get_partitions_for_ais(
    df: pd.DataFrame, selection: np.ndarray, partition_str: str, seed: int, splits: list
) -> pd.DataFrame:
    for dataset in df.loc[selection, 'dataset_name'].unique():
        dataset_selection = (selection & (df.dataset_name == dataset))
        for hemisphere in df.loc[dataset_selection, 'hemisphere'].unique():
            hemisphere_selection = dataset_selection & (df.hemisphere == hemisphere)
            # in one dataset there are only 3 bilateral caes, we place them
            # one in each partition
            if (sum(hemisphere_selection) == 3) and (len(splits) == 3):
                df.loc[hemisphere_selection, partition_str] = ['train', 'validation', 'test']
                continue
            if (sum(hemisphere_selection) == 2) and (len(splits) == 2):
                df.loc[hemisphere_selection, partition_str] = ['train', 'validation']
                continue
            # force diversity on lesion size
            sub_df = df.loc[hemisphere_selection, :]
            min_les_vol = sub_df.min_lesion_vol.tolist()
            min_les_vol = sorted(min_les_vol)
            n_cases = len(min_les_vol)
            bin_lims = np.floor(np.linspace(0, n_cases-1, 4)).astype(int)
            bin_lims[-1] = n_cases if bin_lims[-1] != n_cases-1 else bin_lims[-1]
            bin_lims = [min_les_vol[i] for i in bin_lims]
            for i in range(3):
                if i == 2:
                    size_selection = (sub_df.min_lesion_vol >= bin_lims[i])
                else:
                    size_selection = (
                        (sub_df.min_lesion_vol >= bin_lims[i]) &
                        (sub_df.min_lesion_vol < bin_lims[i+1])
                    )
                temp_df = sub_df.loc[size_selection, :]
                temp_df = temp_df.drop_duplicates(subset=['patient']).copy()
                if len(splits) == 3:
                    train_subjs = temp_df.patient.sample(frac=0.7, random_state=seed).tolist()
                    df.loc[df.patient.isin(train_subjs), partition_str] = 'train'

                    size_selection = size_selection & (~sub_df.patient.isin(train_subjs))
                    temp_df = sub_df.loc[size_selection, :]
                    temp_df = temp_df.drop_duplicates(subset=['patient']).copy()
                    val_subjs = temp_df.patient.sample(frac=2/3, random_state=seed).tolist()
                    df.loc[df.patient.isin(val_subjs), partition_str] = 'validation'

                    test_leftovers = size_selection & (~sub_df.patient.isin(val_subjs))
                    temp_df = sub_df.loc[test_leftovers, :]
                    temp_df = temp_df.drop_duplicates(subset=['patient']).copy()
                    test_subjs = temp_df.patient.tolist()
                    df.loc[df.patient.isin(test_subjs), partition_str] = 'test'
                else:
                    train_subjs = temp_df.patient.sample(frac=7/9, random_state=seed).tolist()
                    df.loc[df.patient.isin(train_subjs), partition_str] = 'train'

                    size_selection = size_selection & (~sub_df.patient.isin(train_subjs))
                    temp_df = sub_df.loc[size_selection, :]
                    temp_df = temp_df.drop_duplicates(subset=['patient']).copy()
                    val_subjs = temp_df.patient.tolist()
                    df.loc[df.patient.isin(val_subjs), partition_str] = 'validation'
    return df


def generate_stratified_partitions(df: pd.DataFrame) -> pd.DataFrame:
    seeds = [420, 38, 45, 29, 9]
    fold = 0
    splits = ['train', 'validation', 'test']
    # Generate partitions
    partition_str = f'partition_{fold}'
    seed = seeds[fold]

    df[partition_str] = '-'
    logging.info('Getting standarized partitions')
    # Cases without ground truth
    selection = (df.discard == 'n') & (df.label == 'n')
    get_partitions_for_normals(df, selection, partition_str, seed, splits=splits)
    # Cases with ground truth
    selection = (df.discard == 'n') & (df.label == 'y')
    df.loc[selection, partition_str] = '-'
    get_partitions_for_ais(df, selection, partition_str, seed, splits=splits)

    splits = ['train', 'validation']
    for fold in range(1, 5):
        # Generate 5 different partitions from train and val sets
        partition_str = f'partition_{fold}'
        seed = seeds[fold]
        df[partition_str] = '-'
        df.loc[df.partition_0 == 'test', partition_str] = 'test'
        # Cases without ground truth
        selection = (df.discard == 'n') & (df.label == 'n') & (df.partition_0.isin(splits))
        get_partitions_for_normals(df, selection, partition_str, seed, splits=splits)
        # Cases with ground truth
        selection = (df.discard == 'n') & (df.label == 'y') & (df.partition_0.isin(splits))
        df.loc[selection, partition_str] = '-'
        get_partitions_for_ais(df, selection, partition_str, seed, splits=splits)
    return df


def unify_and_refactor_csv(
    dataframes_list: List[pd.DataFrame], bidsdata_path: Path
) -> pd.DataFrame:

    # Join dataframes
    logging.info('Joining dataframes...')
    df = pd.concat(dataframes_list, ignore_index=True)

    # drop unnecesary columns
    cols_to_keep = [
        'patient_sex', 'patient_age', 'slice_thickness', 'pixel_spacing_x',
        'pixel_spacing_y', 'original_partition', 'original_space', 'tilt_corr_needed',
        'dataset_name', 'ais', 'gt_space', 'n_pncct', 'subject', 'patient', 'standard'
    ]
    modalities = [
        'ctp', 'ncct', 'ncct-tilt', 'adc', 'dwi', 'flair', 'msk',
        'msk-tilt', 'pncct', 'cbf', 'cbv', 'tmax', 'mtt', 'bm'
    ]
    cols_to_drop = [
        name for name in (FILE_TYPES + METADATA_FIELDS) if name not in (cols_to_keep + modalities)
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    # read the manual annotations file:
    mseg = pd.read_csv(repository_base_path/'data/datasets_manual_label.csv', index_col=0)
    new_columns = [
        'label', 'brain', 'cerebellum', 'stem', 'bilateral', 'hemisphere', 'discard', 'comment'
    ]
    # pair the datasets
    for col in new_columns:
        df[col] = '-'
    #   tbi cases:
    df.loc[df.dataset_name == 'tbi', 'label'] = 'n'
    df.loc[df.dataset_name == 'tbi', 'discard'] = 'n'
    #   other datasets
    other_subjects = df.loc[df.dataset_name != 'tbi', 'subject'].tolist()
    for subj in other_subjects:
        if (df.loc[df.subject == subj, 'dataset_name'].values[0] == 'aisd'):
            sel = df.subject == subj
            subj = (str(subj)).rjust(7, '0')
            df.loc[sel, 'subject'] = subj
        df.loc[df.subject == subj, new_columns] = \
            mseg.loc[mseg.subject == subj, new_columns].values
    df.loc[df.label == 'h', 'ais'] = False
    df.loc[df.label == 'h', 'msk'] = '-'
    df.loc[df.label == 'h', 'label'] = 'n'
    df.fillna('-', inplace=True)

    df = quantify_lesion_sizes(df, bidsdata_path)
    df = generate_stratified_partitions(df)
    return df


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-sdp', dest='sourcedata_path', help='Path to the source data')
    parser.add_argument('-bdp', dest='bidsdata_path', help='Path to the bids data')
    parser.add_argument(
        '-rdf', action='store_true', dest='read_df',
        help='Avoid reprocessing each dataset individually'
    )
    args = parser.parse_args()

    sourcedata_path = Path(args.sourcedata_path)
    bidsdata_path = Path(args.bidsdata_path)

    # Clean APIS dataset
    logging.info('Cleaning APIS dataset...')
    if args.read_df:
        apis_df = pd.read_csv(
            repository_base_path/'data/metadata_files/metadata_apis.csv', index_col=0
        )
    else:
        apis_df = clean_apis_data(sourcedata_path, bidsdata_path)

    # Clean AISD:
    logging.info('Cleaning AISD dataset...')
    if args.read_df:
        aisd_df = pd.read_csv(
            repository_base_path/'data/metadata_files/metadata_aisd.csv', index_col=0
        )
    else:
        aisd_df = clean_aisd_data(sourcedata_path, bidsdata_path)

    # Clean ISLES18:
    logging.info('Cleaning ISLES18 dataset...')
    if args.read_df:
        isles18_df = pd.read_csv(
            repository_base_path/'data/metadata_files/metadata_isles18.csv', index_col=0
        )
    else:
        isles18_df = clean_isles18_data(sourcedata_path, bidsdata_path)

    # Clean CENTER_TBI:
    logging.info('Cleaning CENTER TBI dataset...')
    if args.read_df:
        tbi_df = pd.read_csv(
            repository_base_path/'data/metadata_files/metadata_tbi.csv', index_col=0
        )
    else:
        tbi_df = clean_center_tbi_data(sourcedata_path, bidsdata_path)

    df = unify_and_refactor_csv(
        [apis_df, aisd_df, isles18_df, tbi_df], bidsdata_path)
    logging.info('Saving dataset...')
    df.to_csv(repository_base_path / 'data/dataset.csv')
