# -*-coding:utf-8 -*-
'''
@Time    :   2023/02/08 16:34:24
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import cv2
import json
import pydicom
import subprocess
import time
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import List, Tuple

from utils.utils import to_snake_case


FILE_TYPES = [
    'ctp', 'ctp_json', 'ncct', 'ncct_json', 'ncct-tilt', 'ncct-tilt_json', 'adc',
    'adc_json', 'dwi', 'dwi_json', 'flair', 'flair_json', 'msk', 'msk_json', 'msk-tilt', 'snp',
    'pncct', 'pncct_json', 'n_pncct', 'cbf', 'cbf_json', 'cbv', 'cbv_json', 'tmax',
    'tmax_json', 'mtt', 'mtt_json', 'bm'
]

METADATA_FIELDS = [
    'patient_birth_date', 'patient_sex', 'patient_age', 'slice_thickness', 'spacing_between_slices',
    'protocol_name', 'photometric_interpretation', 'rows', 'columns', 'pixel_spacing',
    'acquisition_date', 'acquisition_time', 'manufacturer', 'institution_name', 'model_name',
    'patient_id', 'image_position', 'image_orientation', 'original_partition', 'original_space',
    'ctp_n_tpts', 'ctp_n_slices', 'ncct_n_slices', 'tilt_corr_needed', 'ais', 'patient',
    'base_ses_zero_path', 'base_ses_one_path', 'gt_space'
]


def get_ncct_metadata_from_json(json_filepath: Path) -> dict:
    """ Extracs useful fields from the ncct json file
    Args:
        json_filepath (Path): path to the json file.
    Returns:
        (dict): containing the desired metadata
    """
    useful_fields = [
        'PatientBirthDate', 'PatientSex', 'PatientAge', 'SliceThickness', 'SpacingBetweenSlices',
        'ProtocolName', 'PhotometricInterpretation', 'Rows', 'Columns', 'PixelSpacing',
        'AcquisitionDate', 'AcquisitionTime', 'Manufacturer', 'InstitutionName', 'ModelName',
        'PatientID', 'ImagePosition', 'ImageOrientation'
    ]

    metadata = {}
    if str(json_filepath).endswith('-'):
        for field in useful_fields:
            if field == 'PixelSpacing':
                metadata[f'{to_snake_case(field)}_x'] = None
                metadata[f'{to_snake_case(field)}_y'] = None
            else:
                metadata[to_snake_case(field)] = None
        return metadata

    with open(json_filepath, 'r') as jfile:
        ncct_meta = json.load(jfile)

    for field in useful_fields:
        if field in ncct_meta.keys():
            if field == 'PixelSpacing':
                metadata[f'{to_snake_case(field)}_x'] = ncct_meta[field][0]
                metadata[f'{to_snake_case(field)}_y'] = ncct_meta[field][1]
            else:
                metadata[f'{to_snake_case(field)}'] = ncct_meta[field]
        else:
            metadata[to_snake_case(field)] = '-'
    return metadata


def generate_dcm_from_pngs(
    dicom_dir: Path, output_dir: Path, mask_path: Path
) -> None:
    """Generates a dcm slice for each mask png using the original
    ncct metadata of each of its dcms.
    Args:
        dicom_dir (Path): Path to the original slice-dcm images of the ncct
        output_dir (Path): Path where the masks are going to be stored
        mask_path (Path): Path where the png images are stored
    """

    # get the length of the filename to match zeros in the begining
    example_filename = next(dicom_dir.iterdir()).stem
    dcm_file_len = len(example_filename)

    # read the dicom series
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        str(dicom_dir), series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image = series_reader.Execute()

    # for each slice read the png, and repopulate importante dcm tags
    for i in range(image.GetDepth()):
        # read the dicom tags
        tags_dict = {}
        for j, key in enumerate(series_reader.GetMetaDataKeys(i)):
            tags_dict[key] = series_reader.GetMetaData(i, key)

        # read the pngs and write them as dcm
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        png = np.expand_dims(
            cv2.imread(str(mask_path/f'{(i):03}.png'), cv2.IMREAD_GRAYSCALE), axis=0)
        image_slice = sitk.GetImageFromArray(np.where(png != 0, 1, 0).astype('int16'))
        for j, key in enumerate(tags_dict):
            image_slice.SetMetaData(key, tags_dict[key])

        # force some dicom tags
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        image_slice.SetMetaData(
            "0020|0032", '\\'.join(map(str, image.TransformIndexToPhysicalPoint((0, 0, i)))))
        image_slice.SetMetaData("0020,0013", str(i))
        writer.SetFileName(str(output_dir/f'{i:03}.dcm'))
        writer.Execute(image_slice)
        # force some other dicom fields in the easies way
        ncct_dcm = pydicom.dcmread(str(dicom_dir/f'{(i+1):0{dcm_file_len}}.dcm'))
        mask_dcm = pydicom.dcmread(str(output_dir/f'{i:03}.dcm'))
        if 0x181120 in ncct_dcm.keys():
            mask_dcm[0x181120].value = ncct_dcm[0x181120].value
        mask_dcm[0x180050].value = ncct_dcm[0x180050].value
        mask_dcm[0x200032].value = ncct_dcm[0x200032].value
        mask_dcm[0x280030].value = ncct_dcm[0x280030].value
        pydicom.dcmwrite(str(output_dir/f'{i:03}.dcm'), mask_dcm)


def dcm2niix(
    series_path: Path, out_filepath: Path, verbose: bool = False
) -> None:
    """Wrapper for dcm2niix comand line function.
    Args:
        series_path (Path): path to the series folder.
        out_filepath (Path): output path of the nifti image.
        verbose (bool, optional): Whether to print logs of dcm2niix or not.
            Defaults to False.
    """
    filename = out_filepath.name.replace(''.join(out_filepath.suffixes), '')
    out_path = out_filepath.parent
    command = [
        'dcm2niix', '-f', filename, '-o', str(out_path), '-z', 'y', str(series_path)
    ]
    if verbose:
        print(command)
        subprocess.call(command)
    else:
        subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def isles_get_nii_and_json_filenames(
    subj_dir_content: List[Path], key_word: str
) -> Tuple[Path, Path]:
    """Explore the availabe files and find the desired modality image and json files"""
    path = [path for path in subj_dir_content if (key_word in str(path))][0]
    img_name = f'{path.name}.nii'
    img_path = path / img_name
    json_name = f'{path.name}.json'
    json_path = path / json_name
    return img_path, json_path
