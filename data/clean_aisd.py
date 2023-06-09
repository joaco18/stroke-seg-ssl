# -*-coding:utf-8 -*-
'''
@Time    :   2023/02/08 16:35:41
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import argparse
import logging
import shutil
import yaml
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

import data_utils
from data_utils import FILE_TYPES, METADATA_FIELDS
from utils.utils import save_img_from_array_using_referece, split_4d_img_into_3d_imgs

logging.basicConfig(level=logging.INFO)
repository_base_path = Path().resolve()

CUT_SLICES = {
    '0091519': 24,
    '0091551': 24,
    '0091580': 30,
}

TEST_SET_IDS = [
    '0073410', '0072723', '0226290', '0537908', '0538058', '0091415', '0538780',
    '0073540', '0226188', '0226258', '0226314', '0091507', '0226298', '0538975',
    '0226257', '0226142', '0072681', '0091538', '0538983', '0537961', '0091646',
    '0072765', '0226137', '0091621', '0091458', '0021822', '0538319', '0226133',
    '0091657', '0537925', '0073489', '0538502', '0091476', '0226136', '0538532',
    '0073312', '0539025', '0226309', '0226307', '0091383', '0021092', '0537990',
    '0226299', '0073060', '0538505', '0073424', '0091534', '0226125', '0072691',
    '0538425', '0226199', '0226261'
]

with open(repository_base_path / 'data/metadata_files/metadata_dwi_aisd.json', 'r') as yfile:
    time_pts = yaml.safe_load(yfile)


def clean_aisd_data(source_data_path: Path, bids_data_path: Path) -> pd.DataFrame:
    """Clean Acute Ischemic stroke dataset (https://github.com/GriffinLiang/AISD)
    Args:
        source_data_path (Path): Path where the uncompressed files are stored.
        bids_data_path (Path): Path where the bids-reorganized files will be stored.
    Returns:
        pd.DataFrame: Dataset csv
    """
    # define necessary paths
    source_data_path = source_data_path / 'aisd'
    dcm_mask_path = source_data_path / 'dcm_mask'
    masks_path = source_data_path / 'mask'
    session = 'ses-0000'
    missing_files_from_dcm0 = ['0091505', '0073366', '0091468', '0091519', '0091465']

    # placeholder for metadata
    df = []

    # go through the four dicom data directories
    for i in range(4):
        dcm_dir = f'dicom-{i}'
        dcm_dir_path = source_data_path / dcm_dir
        # go through the subjects in that folder
        total = len(list(dcm_dir_path.iterdir()))
        for subject_path in tqdm(dcm_dir_path.iterdir(), total=total):
            subject = subject_path.name
            if subject == '.DS_Store':
                continue
            if (subject in missing_files_from_dcm0) and (dcm_dir == 'dicom-0'):
                continue
            df_row = {
                'subject': subject,
                'original_space': 'patient',
                'original_partition': 'test' if (subject in TEST_SET_IDS) else 'train',
                'tilt_corr_needed': 'no',
                'ais': True,
                'gt_space': 'ncct',
                'base_ses_zero_path': f'{subject}/{session}/sub-{subject}_{session}',
                'patient': subject,
                'standard': 'gold'
            }

            # create bids path
            bids_filepath = bids_data_path / 'aisd' / subject / session
            bids_filepath.mkdir(exist_ok=True, parents=True)

            # convert ncct dicom series to nifti image
            ncct_bids_filename = f'sub-{subject}_{session}_ncct.nii.gz'
            ncct_tilt_bids_filename = f'sub-{subject}_{session}_ncct-tilt.nii.gz'
            ncct_json_filename = f'sub-{subject}_{session}_ncct.json'
            series_path = subject_path/'CT'
            ncct_filepath = bids_filepath / ncct_bids_filename
            data_utils.dcm2niix(series_path, ncct_filepath)
            if not(ncct_filepath.exists()):
                logging.warning(f'Check manually case: {subject}')

            # fix tilt image filename
            non_standard_tilt_a = bids_filepath / f'sub-{subject}_{session}_ncct_Tilt_1.nii.gz'
            non_standard_tilt_b = bids_filepath / f'sub-{subject}_{session}_ncct_Tilt_1_Eq_1.nii.gz'
            non_standard_tilt_c = bids_filepath / f'sub-{subject}_{session}_ncct_Tilt_Eq_1.nii.gz'
            non_standards = [non_standard_tilt_a, non_standard_tilt_b, non_standard_tilt_c]
            for non_standart_tilt in non_standards:
                if non_standart_tilt.exists():
                    non_standart_tilt.rename(bids_filepath/ncct_tilt_bids_filename)
                    df_row['ncct-tilt'] = f'{subject}/{session}/{ncct_tilt_bids_filename}'
                    df_row['tilt_corr_needed'] = 'done'

            # update metadata dataframe
            ncct_meta = data_utils.get_ncct_metadata_from_json(bids_filepath/ncct_json_filename)
            df_row.update(ncct_meta)
            df_row['ncct'] = f'{subject}/{session}/{ncct_bids_filename}'
            df_row['ncct_json'] = f'{subject}/{session}/{ncct_json_filename}'
            ncct_img = sitk.ReadImage(str(ncct_filepath))
            df_row['ncct_n_slices'] = sitk.GetArrayFromImage(ncct_img).shape[0]

            # convert dwi dicom series to nifti image
            dwi_bids_filename = f'sub-{subject}_{session}_dwi.nii.gz'
            dwi_tilt_bids_filename = f'sub-{subject}_{session}_dwi-tilt.nii.gz'
            dwi_json_filename = f'sub-{subject}_{session}_dwi.json'
            series_path = subject_path/'DWI'
            series_path = series_path if (series_path.exists()) else subject_path/'MRI'
            dwi_filepath = bids_filepath / dwi_bids_filename
            data_utils.dcm2niix(series_path, dwi_filepath)
            if not(dwi_filepath.exists()):
                logging.warning(f'Check manually case: {subject}')

            # fix tilt image filename
            for non_standard_tilt in non_standards:
                non_standard_tilt = Path(str(non_standard_tilt).replace('ncct', 'dwi'))
                if non_standard_tilt.exists():
                    non_standard_tilt.rename(dwi_tilt_bids_filename)
                    df_row['dwi-tilt'] = f'{subject}/{session}/{dwi_tilt_bids_filename}'

            # fix dwi images (some are 4D sequences)
            if subject == '0538549':
                dwi_b_filepath = Path(str(dwi_filepath).replace('dwi', 'dwib'))
                dwi_b_json_filename = Path(str(dwi_json_filename).replace('dwi', 'dwib'))
                shutil.copyfile(dwi_b_filepath, dwi_filepath)
                shutil.copyfile(dwi_b_json_filename, dwi_json_filename)

            dwi_o_path = Path(str(dwi_filepath).replace('.nii.gz', '-orig.nii.gz'))
            img = sitk.ReadImage(str(dwi_filepath))
            if len(img.GetSize()) > 3:
                time_pt = time_pts[subject]
                shutil.copyfile(dwi_filepath, dwi_o_path)
                t_pt = split_4d_img_into_3d_imgs(img, time_pt-1)[0]
                sitk.WriteImage(t_pt, str(dwi_filepath))

            # update metadata dataframe
            df_row['dwi'] = f'{subject}/{session}/{dwi_bids_filename}'
            df_row['dwi_json'] = f'{subject}/{session}/{dwi_json_filename}'

            # read mask and reorganise in bids format
            dcm_mask_subject_path = dcm_mask_path / subject
            dcm_mask_subject_path.mkdir(exist_ok=True, parents=True)
            data_utils.generate_dcm_from_pngs(
                subject_path/'CT', dcm_mask_subject_path, masks_path/subject)
            gt_bids_filename = f'sub-{subject}_{session}_msk.nii.gz'
            gt_tilt_bids_filename = f'sub-{subject}_{session}_msk-tilt.nii.gz'
            data_utils.dcm2niix(dcm_mask_subject_path, bids_filepath/gt_bids_filename)
            df_row['msk'] = f'{subject}/{session}/{gt_bids_filename}'

            # fix tilt mask image filename
            for non_standard_tilt in non_standards:
                non_standard_tilt = Path(str(non_standard_tilt).replace('ncct', 'msk'))
                if non_standard_tilt.exists():
                    non_standard_tilt.rename(bids_filepath/gt_tilt_bids_filename)
                    df_row['msk-tilt'] = f'{subject}/{session}/{gt_tilt_bids_filename}'

            if subject == '0073465':
                msk = sitk.ReadImage(str(bids_filepath/gt_bids_filename))
                orig_dir = msk.GetDirection()
                right_dir = [
                    -(np.pi-abs(orig_dir[0])), (np.pi-abs(orig_dir[1])), abs(orig_dir[2]),
                    (np.pi-abs(orig_dir[3])), -(np.pi-abs(orig_dir[4])), abs(orig_dir[5]),
                    abs(orig_dir[6]), abs(orig_dir[7]), abs(orig_dir[8])
                ]
                msk.SetDirection(right_dir)
                sitk.WriteImage(msk, str(bids_filepath/gt_bids_filename))
                img = sitk.ReadImage(str(ncct_filepath))
                sitk.WriteImage(img, str(ncct_filepath))
                roi_file = bids_filepath / f'sub-{subject}_{session}_ncct_ROI1.nii.gz'
                if roi_file.exists():
                    roi_file.unlink()
                roi_file = Path(str(roi_file).replace('ncct', 'msk'))
                if roi_file.exists():
                    roi_file.unlink()
            if subject in CUT_SLICES.keys():
                img = sitk.ReadImage(str(ncct_filepath))
                img_array = sitk.GetArrayFromImage(img)
                save_img_from_array_using_referece(
                    img_array[CUT_SLICES[subject]:, :, :], img, str(ncct_filepath)
                )
                msk_array = sitk.GetArrayFromImage(
                    sitk.ReadImage(str(bids_filepath/gt_bids_filename)))
                save_img_from_array_using_referece(
                    msk_array[CUT_SLICES[subject]:, :, :], img,
                    str(bids_filepath/gt_bids_filename)
                )

            # fill remaining fieds in the df to be consistent
            for field in (FILE_TYPES + METADATA_FIELDS):
                if field in list(df_row.keys()):
                    continue
                df_row[field] = '-'
            df.append(df_row)
    df = pd.DataFrame(df)
    missing_age = df.patient_age.isnull() | (df.patient_age == '-')
    df.loc[~missing_age, 'patient_age'] = df.loc[~missing_age, 'patient_age'].str.rstrip('Y')
    df.loc[~missing_age, 'patient_age'] = df.loc[~missing_age, 'patient_age'].astype('int')
    df['dataset_name'] = 'aisd'
    df.to_csv(repository_base_path / 'data/metadata_files/metadata_aisd.csv')
    return df


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-sdp', dest='sourcedata_path', help='Path to the source data')
    parser.add_argument('-bdp', dest='bidsdata_path', help='Path to the bids data')
    args = parser.parse_args()

    sourcedata_path = Path(args.sourcedata_path)
    bidsdata_path = Path(args.bidsdata_path)

    # Clean AISD:
    logging.info('Cleaning AISD dataset...')
    aisd_df = clean_aisd_data(sourcedata_path, bidsdata_path)
