# -*-coding:utf-8 -*-
'''
@Time    :   2023/02/08 16:36:02
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import argparse
import logging
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from data_utils import FILE_TYPES, METADATA_FIELDS


logging.basicConfig(level=logging.INFO)
repository_base_path = Path().resolve()


def clean_center_tbi_data(source_data_path: Path, bids_data_path: Path) -> pd.DataFrame:
    """Clean CENTER TBI dataset (in house)
    Args:
        source_data_path (Path): Path where the uncompressed TRAINING files are stored.
        bids_data_path (Path): Path where the bids-reorganized files will be stored.
    Returns:
        pd.DataFrame: Dataset csv
    """
    # define necessary paths
    source_datapath = source_data_path/'tbi/anomaly_detection_healthy_CT/'

    # static variables
    session = 'ses-0000'
    partitions = ['train', 'test', 'valid', 'extra']
    ncct_filename = 'ct_in_mni_inter_1.nii.gz'
    bm_filename = 'brain_mask_MNI.nii.gz'

    # define placeholder for dataset
    df = []
    for partition in partitions:
        partition_path = source_datapath / partition if partition != 'extra' else source_datapath
        total = len(list(partition_path.iterdir()))
        for k, subject_path in tqdm(enumerate(partition_path.iterdir()), total=total):
            subject = subject_path.name
            if subject in partitions:
                continue
            df_row = {
                'subject': subject,
                'original_space': 'mni',
                'original_partition': (partition if partition != 'extra' else '-'),
                'tilt_corr_needed': 'no',
                'ais': False,
                'gt_space': '-',
                'base_ses_zero_path': f'{subject}/{session}/sub-{subject}_{session}',
                'patient': subject,
                'standard': '-'
            }
            bids_session_path = bids_data_path / 'tbi' / subject / session
            bids_session_path.mkdir(exist_ok=True, parents=True)

            # NCCT
            ncct_filepath = subject_path / ncct_filename
            bids_ncct_filename = f'sub-{subject}_{session}_ncct.nii.gz'
            bids_ncct_filepath = bids_session_path / bids_ncct_filename
            shutil.copyfile(ncct_filepath, bids_ncct_filepath)
            df_row['ncct'] = f'{subject}/{session}/{bids_ncct_filename}'

            # BRAIN MASK
            bm_filepath = subject_path / bm_filename
            bids_bm_filename = f'sub-{subject}_{session}_bm.nii.gz'
            bids_bm_filepath = bids_session_path / bids_bm_filename
            shutil.copyfile(bm_filepath, bids_bm_filepath)
            df_row['bm'] = f'{subject}/{session}/{bids_bm_filename}'

            # fill remaining fieds in the df to be consistent
            for field in (METADATA_FIELDS + FILE_TYPES):
                if field in list(df_row.keys()):
                    continue
                df_row[field] = '-'

            df.append(df_row)
    df = pd.DataFrame(df)
    df['dataset_name'] = 'tbi'
    df.to_csv(repository_base_path / 'data/metadata_files/metadata_tbi.csv')
    return df


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-sdp', dest='sourcedata_path', help='Path to the source data')
    parser.add_argument('-bdp', dest='bidsdata_path', help='Path to the bids data')
    args = parser.parse_args()

    sourcedata_path = Path(args.sourcedata_path)
    bidsdata_path = Path(args.bidsdata_path)

    # Clean CENTER_TBI:
    logging.info('Cleaning CENTER TBI dataset...')
    tbi_df = clean_center_tbi_data(sourcedata_path, bidsdata_path)
