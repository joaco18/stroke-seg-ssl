# -*-coding:utf-8 -*-
'''
@Time    :   2023/02/08 16:35:30
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import argparse
import logging
import shutil
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import data_utils
from data_utils import FILE_TYPES, METADATA_FIELDS

logging.basicConfig(level=logging.INFO)
repository_base_path = Path().resolve()


def clean_apis_data(source_data_path: Path, bids_data_path: Path) -> pd.DataFrame:
    """Clean "A paired dataset of CT-MRI for Ischemic Stroke segmentation"
        (https://bivl2ab.uis.edu.co/challenges/apis#data)
    Args:
        source_data_path (Path): Path where the uncompressed files are stored.
        bids_data_path (Path): Path where the bids-reorganized files will be stored.
    Returns:
        pd.DataFrame: Dataset csv
    """
    # Define necessary data paths
    apis_sourcedata_path = source_data_path / 'apis'
    bids_apis_sourcedata_path = bids_data_path / 'apis'

    # Define static variables
    session = 'ses-0000'
    extension = '.nii.gz'

    df = []
    total = len(list(apis_sourcedata_path.iterdir()))
    for subj in tqdm(apis_sourcedata_path.iterdir(), total=total):
        if subj.is_file():
            continue
        subject = ('').join(subj.name.split('_'))
        df_row = {
            'subject': subject,
            'original_space': 'patient',
            'original_partition': 'train',
            'tilt_corr_needed': 'no',
            'ais': True,
            'gt_space': 'adc',
            'base_ses_zero_path': f'{subject}/{session}/sub-{subject}_{session}',
            'patient': subject,
            'standard': 'gold'
        }
        for img_filepath in subj.iterdir():
            if not img_filepath.is_file():
                seq = 'msk'
                img_filepath = list(img_filepath.iterdir())[0]
            elif 'ncct' in img_filepath.name:
                seq = 'ncct'
                img_array = sitk.GetArrayFromImage(sitk.ReadImage(str(img_filepath)))
                n_slices = img_array.shape[0]
                df_row['ncct_n_slices'] = n_slices
                ncct_meta = data_utils.get_ncct_metadata_from_json('-')
                df_row.update(ncct_meta)
            else:
                seq = 'adc'

            # generate bids naming and reorganize files
            bids_filename = f'sub-{subject}_{session}_{seq}{extension}'
            bids_filepath = bids_apis_sourcedata_path / subject / session
            bids_filepath.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(img_filepath, bids_filepath/bids_filename)

            df_row[seq] = f'{subject}/{session}/{bids_filename}'

        # fill remaining fieds in the df to be consistent
        for field in (FILE_TYPES + METADATA_FIELDS):
            if field in list(df_row.keys()):
                continue
            df_row[field] = '-'
        df.append(df_row)

    # store standard dataset csv
    df = pd.DataFrame(df)
    df['dataset_name'] = 'apis'
    df.to_csv(repository_base_path / 'data/metadata_files/metadata_apis.csv')
    return df


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-sdp', dest='sourcedata_path', help='Path to the source data')
    parser.add_argument('-bdp', dest='bidsdata_path', help='Path to the bids data')
    args = parser.parse_args()

    sourcedata_path = Path(args.sourcedata_path)
    bidsdata_path = Path(args.bidsdata_path)

    # Clean APIS dataset
    logging.info('Cleaning APIS dataset...')
    apis_df = clean_apis_data(sourcedata_path, bidsdata_path)
