# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import argparse
import pickle
import yaml
import multiprocessing as mp
import tensorflow as tf

from pathlib import Path
from typing import List

from preprocessing.preprocessing_utils import check_available_modalities
from preprocessing.resample.resampling_utils import resolution_and_size_match
from preprocessing.super_resolution.super_res_utils import synthsr_wrapper


GPU_AVAILABLE = len(tf.config.list_physical_devices('GPU')) != 0
N_JOBS = mp.cpu_count()


class SuperResolutor():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

        self.suffix = self.cfg['suffix']
        # define the resampled suffix
        self.res_suffix = ''
        if self.cfg['apply_over_resampled']:
            self.res_suffix = f'-{self.cfg["resampled_suffix"]}'

        self.cpu = False if (GPU_AVAILABLE and self.cfg['gpu']) else True
        condition = ('n_threads' not in self.cfg.keys()) or (self.cfg['n_threads'] is None)
        self.n_threads = N_JOBS-1 if condition else self.cfg['n_threads']

    def __call__(self, ncct_path: Path, mri_path: Path) -> List[Path]:

        # define the files to process and the output filenames
        ncct_path, adc_path, dwi_path = check_available_modalities(ncct_path, mri_path)
        modalities_paths, new_files = [], []
        if ('ncct' in self.cfg['modalities']) and (ncct_path is not None):
            ncct_path = Path(str(ncct_path).replace('.nii.gz', f'{self.res_suffix}.nii.gz'))
            modalities_paths.append(ncct_path)
        if ('adc' in self.cfg['modalities']) and (adc_path is not None):
            adc_path = Path(str(adc_path).replace('.nii.gz', f'{self.res_suffix}.nii.gz'))
            if adc_path.exists():
                modalities_paths.append(adc_path)
        if ('dwi' in self.cfg['modalities']) and (dwi_path is not None):
            dwi_path = Path(str(dwi_path).replace('.nii.gz', f'{self.res_suffix}.nii.gz'))
            if dwi_path.exists():
                modalities_paths.append(dwi_path)

        # process the files
        for img_path in modalities_paths:
            out_img_path = Path(str(img_path).replace('.nii.gz', f'-{self.suffix}.nii.gz'))
            if not out_img_path.exists() or self.cfg['force']:
                synthsr_wrapper(
                    input_img_path=img_path,
                    output_img_path=out_img_path,
                    n_threads=self.n_threads,
                    cpu=self.cpu,
                    ct=True if ('ncct' in str(img_path)) else False
                )

            # check if the resampled image and the SR one have same size
            if self.cfg['check_resolution']:
                if not self.cfg['apply_over_resampled']:
                    img_path = Path(str(img_path).replace(
                        '.nii.gz', f'-{self.cfg["resampled_suffix"]}.nii.gz'))
                    assert resolution_and_size_match(img_path, out_img_path), \
                        'Either the size or the resolution of the following images' \
                        f' doesn\'t match:\n\t- {str(img_path)}\n\t- {str(out_img_path)}'

            new_files.append(out_img_path)

        return new_files


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

    # instantiate and call super resolutor
    super_resolutor = SuperResolutor(cfg['super_resolution'])
    new_files = super_resolutor(ncct_path, mri_path)

    # store outputs in temporary file
    with open(args.temp_path, 'wb') as pfile:
        pickle.dump(new_files, pfile)


# if __name__ == '__main__':
#     # parse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-ncct', dest='ncct_path')
#     parser.add_argument('-out', dest='out_img_path')
#     args = parser.parse_args()

#     synthsr_wrapper(
#         input_img_path=Path(args.ncct_path),
#         output_img_path=Path(args.out_img_path),
#         n_threads=8,
#         cpu=False,
#         ct=True
#     )
