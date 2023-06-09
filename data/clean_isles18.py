# -*-coding:utf-8 -*-
'''
@Time    :   2023/02/08 16:35:53
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import argparse
import json
import logging
import numpy as np
import shutil
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import data_utils
from data_utils import FILE_TYPES, METADATA_FIELDS
from utils import utils


logging.basicConfig(level=logging.INFO)
repository_base_path = Path().resolve()


def clean_isles18_data(source_data_path: Path, bids_data_path: Path) -> pd.DataFrame:
    """Clean ISLES28 dataset (https://www.smir.ch/ISLES/Start2018)
    Args:
        source_data_path (Path): Path where the uncompressed TRAINING files are stored.
        bids_data_path (Path): Path where the bids-reorganized files will be stored.
    Returns:
        pd.DataFrame: Dataset csv
    """
    source_datapath = source_data_path / 'isles18/TRAINING'
    session = 'ses-0000'
    with open(repository_base_path/'data/metadata_files/metadata_isles18.json', 'r') as jfile:
        metadata_isles18 = json.load(jfile)

    # placeholder for metadata
    df = []
    total = len(list(source_datapath.iterdir()))
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    for k, subject_path in tqdm(enumerate(source_datapath.iterdir()), total=total):
        subject = subject_path.name
        if subject == '.DS_Store':
            continue

        # get the content of the directory
        subject_name = ''.join((subject).split('_'))
        subj_dir_content = list(subject_path.iterdir())
        bids_session_path = bids_data_path / 'isles18' / subject_name / session
        bids_session_path.mkdir(exist_ok=True, parents=True)

        df_row = {
            'subject': subject_name,
            'original_space': 'patient',
            'original_partition': metadata_isles18[subject]['partition'],
            'tilt_corr_needed': 'no',
            'ais': True,
            'gt_space': 'ncct',
            'base_ses_zero_path': f'{subject}/{session}/sub-{subject}_{session}',
            'patient': metadata_isles18[subject]['patient'],
            'standard': 'gold'
        }

        # CTP
        # get ctp nifti and ctp json paths
        ctp_img_path, ctp_json_path = data_utils.isles_get_nii_and_json_filenames(
            subj_dir_content, '4DPWI')
        # organize in bids format
        bids_ctp_name = f'sub-{subject_name}_{session}_ctp'
        shutil.copyfile(ctp_json_path, bids_session_path/f'{bids_ctp_name}.json')
        ctp_img = sitk.ReadImage(str(ctp_img_path))
        sitk.WriteImage(ctp_img, str(bids_session_path/f'{bids_ctp_name}.nii.gz'))
        ctp_img = sitk.GetArrayFromImage(ctp_img)
        # add metadata for csv
        df_row['ctp'] = f'{subject_name}/{session}/{bids_ctp_name}.nii.gz'
        df_row['ctp_json'] = f'{subject_name}/{session}/{bids_ctp_name}.json'
        df_row['ctp_n_tpts'] = ctp_img.shape[0]
        df_row['ctp_n_slices'] = ctp_img.shape[1]

        # BRAIN MASK
        bids_bm_name = f'sub-{subject_name}_{session}_bm.nii.gz'
        bm_path = ctp_img_path.parent/'intra_cranial_mask.nii.gz'
        if (bids_session_path/bids_bm_name).exists():
            shutil.copyfile(bm_path, bids_session_path/bids_bm_name)
            df_row['bm'] = f'{subject_name}/{session}/{bids_bm_name}'

        # PSEUDO NCCTs
        if metadata_isles18[subject]['pseudo_ncct_timepoints'] is not None:
            pncct_timepts = metadata_isles18[subject]['pseudo_ncct_timepoints']
            # limit the number of pseudo nccts to store separately
            pncct_timepts = 4 if pncct_timepts > 4 else pncct_timepts
            # get the base ctp image and use it as reference for the timepoints volumes
            base_ctp_img_path, _ = data_utils.isles_get_nii_and_json_filenames(
                subj_dir_content, '.CT.')
            base_ctp_img = sitk.ReadImage(str(base_ctp_img_path))
            # save a 3d volume for each 'safe' timepoint
            mean_pncct = np.zeros_like(ctp_img[0, :, :, :])
            for tpt in range(pncct_timepts+1):
                pncct_bids_name = f'sub-{subject_name}_{session}_pncct{tpt}.nii.gz'
                pncct_bids_path = bids_session_path / pncct_bids_name
                utils.save_img_from_array_using_referece(
                    ctp_img[tpt, :, :, :], base_ctp_img, pncct_bids_path)
                mean_pncct += ctp_img[tpt, :, :, :]
            mean_pncct /= (pncct_timepts + 1)

            # save mean volume (increased snr)
            pncct_bids_name = f'sub-{subject_name}_{session}_pncct.nii.gz'
            pncct_bids_path = bids_session_path / pncct_bids_name
            utils.save_img_from_array_using_referece(
                mean_pncct, base_ctp_img, pncct_bids_path)

            df_row['pncct'] = \
                f'{subject_name}/{session}/sub-{subject_name}_{session}_pncct.nii.gz'
            df_row['n_pncct'] = pncct_timepts + 1
            df_row['ncct_n_slices'] = ctp_img.shape[1]

        # PERFUSION MAPS
        for modality in ['CBF', 'CBV', 'MTT', 'Tmax', 'OT']:
            # cbf
            if (modality == 'OT') and df_row['original_partition'] != 'train':
                continue
            img_path, json_path = data_utils.isles_get_nii_and_json_filenames(
                subj_dir_content, modality)
            modality = modality.lower()
            modality = 'msk' if modality == 'ot' else modality
            # organize in bids format
            bids_name = f'sub-{subject_name}_{session}_{modality}'
            shutil.copyfile(json_path, bids_session_path/f'{bids_name}.json')
            img = sitk.ReadImage(str(img_path))
            sitk.WriteImage(img, str(bids_session_path/f'{bids_name}.nii.gz'))
            # add metadata for csv
            df_row[modality] = f'{subject_name}/{session}/{bids_name}.nii.gz'
            df_row[f'{modality}_json'] = f'{subject_name}/{session}/{bids_name}.json'

        # fill remaining fieds in the df to be consistent
        for field in (METADATA_FIELDS + FILE_TYPES):
            if field in list(df_row.keys()):
                continue
            df_row[field] = '-'
        df.append(df_row)
    sitk.ProcessObject_SetGlobalWarningDisplay(True)
    df = pd.DataFrame(df)
    df['dataset_name'] = 'isles18'
    df.to_csv(repository_base_path / 'data/metadata_files/metadata_isles18.csv')
    return df


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-sdp', dest='sourcedata_path', help='Path to the source data')
    parser.add_argument('-bdp', dest='bidsdata_path', help='Path to the bids data')
    args = parser.parse_args()

    sourcedata_path = Path(args.sourcedata_path)
    bidsdata_path = Path(args.bidsdata_path)

    # Clean ISLES18:
    logging.info('Cleaning ISLES18 dataset...')
    isles18_df = clean_isles18_data(sourcedata_path, bidsdata_path)
