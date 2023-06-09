# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import argparse
import yaml
import pickle
import multiprocessing as mp
import pandas as pd
import tensorflow as tf

from pathlib import Path

from preprocessing.preprocessing_utils import check_available_modalities
from preprocessing.resample.resampling_utils import resolution_and_size_match
import preprocessing.tissue_segmentation.tissue_seg_utils as ts_utils

GPU_AVAILABLE = len(tf.config.list_physical_devices('GPU')) != 0
N_JOBS = mp.cpu_count()


class TissueSegmenter():
    def __init__(
        self,
        cfg: dict,
    ) -> None:
        self.cfg = cfg

        self.suffix = self.cfg['suffix']

        self.gpu = True if (GPU_AVAILABLE and self.cfg['gpu']) else False
        self.device = 'GPU' if self.gpu else 'CPU'

        condition = ('n_threads' not in self.cfg.keys()) or (self.cfg['n_threads'] is None)
        self.n_threads = N_JOBS-1 if condition else self.cfg['n_threads']

    def __call__(self, ncct_path: Path, mri_path: Path) -> None:
        # check the images available
        ncct_path, adc_path, dwi_path = check_available_modalities(ncct_path, mri_path)
        basis = 'ncct-tilt' if 'tilt' in str(ncct_path) else 'ncct'
        # initiate empty variables for storing the metrics
        new_files = []
        metrics = {}

        if self.cfg['over_image'] is not None:
            for modality, img_type in self.cfg['over_image'].items():
                # Check if the image modaities exist
                if (modality == 'ncct') and (ncct_path is not None):
                    if ('tilt' in basis) and ('tilt' not in img_type):
                        img_type = img_type.replace('ncct', 'ncct-tilt')
                    img_path = Path(str(ncct_path).replace(f'{basis}.nii.gz', f'{img_type}.nii.gz'))
                elif (modality == 'adc') and (adc_path is not None):
                    img_path = Path(str(adc_path).replace('adc.nii.gz', f'{img_type}.nii.gz'))
                elif (modality == 'dwi') and (dwi_path is not None):
                    img_path = Path(str(dwi_path).replace('dwi.nii.gz', f'{img_type}.nii.gz'))

                if not img_path.exists():
                    raise Exception(
                        'The following image doesn\'t exist check Tissue Segmentation'
                        f' config:\n{str(img_path)}\n'
                    )

                # define output paths
                qc_path = None
                if self.cfg['get_qc_df']:
                    qc_path = Path(str(img_path).replace('.nii.gz', f'-{self.suffix}-qc.csv'))
                out_img_path = Path(str(img_path).replace('.nii.gz', f'-{self.suffix}.nii.gz'))

                # run segmentation if required
                if (not out_img_path.exists()) or self.cfg['force']:
                    ts_utils.synthseg_wrapper(
                        input_img_path=img_path,
                        output_img_path=out_img_path,
                        parc=False,
                        robust=True,
                        vol=None,
                        qc=qc_path,
                        post=None,
                        resample=None,
                        cpu=not(self.gpu),
                        n_threads=self.n_threads,
                        crop=None,
                        fast=False,
                        v1=False
                    )

                # check that the resolution matches the one from the resampled image
                if self.cfg['check_resolution']:
                    if modality == 'ncct':
                        img_path_check = Path(str(img_path).replace('-sr.nii.gz', '-res.nii.gz'))

                    assert resolution_and_size_match(img_path_check, out_img_path), \
                        'Either the size or the resolution of the following images doesn\'t ' \
                        f'match:\n\t- {str(img_path_check)}\n\t- {str(out_img_path)}'
                new_files.append(out_img_path)

                # Obtain the simplified segementation
                if self.cfg['save_simplified']:
                    simple_out_img_path = Path(
                        str(img_path).replace('.nii.gz', f'-{self.cfg["simplified_sufix"]}.nii.gz'))
                    ts_utils.simplify_synthseg_labels(
                        tseg_path=out_img_path,
                        save=True,
                        output_path=simple_out_img_path
                    )
                    new_files.append(simple_out_img_path)

                # store the qc details in the returned metrics
                if self.cfg['get_qc_df']:
                    out_dict = pd.read_csv(qc_path).to_dict()
                    out_dict['subject'] = out_dict['subject'][0]
                    for key, val in out_dict.items():
                        metrics[f'{modality}_{key}'] = val

        return metrics, new_files


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ncct', dest='ncct_path')
    parser.add_argument('-mri', dest='mri_path')
    parser.add_argument('-cfg', dest='cfg_path')
    parser.add_argument('-temp', dest='temp_path')
    args = parser.parse_args()
    ncct_path = None if (args.ncct_path == 'None') else Path(args.ncct_path)
    mri_path = None if (args.mri_path == 'None') else Path(args.mri_path)
    cfg_path = Path(args.cfg_path)
    with open(cfg_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # instantiate and call the tissue segmenter
    tissue_segmenter = TissueSegmenter(cfg['tissue_segmentation'])
    metric, files = tissue_segmenter(ncct_path, mri_path)

    # dump the outputs in a temporary file
    res = (metric, files)
    with open(args.temp_path, 'wb') as pfile:
        pickle.dump(res, pfile)


# if __name__ == '__main__':
#     # parse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-sr', dest='sr_path')
#     parser.add_argument('-out', dest='out_img_path')
#     parser.add_argument('-qc', dest='qc_path')
#     args = parser.parse_args()
#     ts_utils.synthseg_wrapper(
#         input_img_path=Path(args.sr_path),
#         output_img_path=Path(args.out_img_path),
#         parc=False,
#         robust=True,
#         vol=None,
#         qc=Path(args.qc_path),
#         post=None,
#         resample=None,
#         cpu=False,
#         n_threads=8,
#         crop=None,
#         fast=False,
#         v1=False
#     )