# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import SimpleITK as sitk
from pathlib import Path
from typing import List

from preprocessing.preprocessing_utils import check_available_modalities
import preprocessing.resample.resampling_utils as res_utils


class Resampler():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def resample_img(self, img_path: Path, out_path: Path, mask: bool = False) -> None:
        if ('size' in self.cfg.keys()) and (self.cfg['size'] is not None):
            res_utils.resample_img_with_size_and_tolerance(
                img_path,
                resample_img_path=out_path,
                interpolator='nearest' if mask else self.cfg['interpolator'],
                new_img_size=tuple(self.cfg['size']),
                new_resolution=tuple(self.cfg['resolution']),
                up_margin=self.cfg['tolerance'][1],
                low_margin=self.cfg['tolerance'][0],
                force=True,
                apply_gaussian_smoothing=False if mask else self.cfg['gaussian_smooth']
            )
        else:
            res_utils.resample_img_with_tolerance(
                img_path,
                resample_img_path=out_path,
                interpolator='nearest' if mask else self.cfg['interpolator'],
                new_resolution=tuple(self.cfg['resolution']),
                up_margin=self.cfg['tolerance'][1],
                low_margin=self.cfg['tolerance'][0],
                force=True,
                apply_gaussian_smoothing=False if mask else self.cfg['gaussian_smooth']
            )

    def __call__(self, ncct_path, mri_path) -> List[Path]:
        new_files = []

        # Resample images (ncct, adc, dwi)
        ncct_path, adc_path, dwi_path = check_available_modalities(ncct_path, mri_path)
        for img_path in [ncct_path, adc_path, dwi_path]:
            if img_path is not None:
                # define output name
                suffix = self.cfg['suffix']
                img_res_path = Path(str(img_path).replace(
                    '.nii.gz', f'-{suffix}.nii.gz'))
                # check if image exists already
                if not img_res_path.exists() or self.cfg['force']:
                    self.resample_img(img_path, img_res_path)
                new_files.append(img_res_path)

        # Resample mask
        if 'msk' in self.cfg['modalities']:
            ncct_path, adc_path, dwi_path = check_available_modalities(ncct_path, mri_path)
            for name, img_path in zip(['ncct', 'adc', 'dwi'], [ncct_path, adc_path, dwi_path]):
                if img_path is None:
                    continue
                msk_path = Path(str(img_path).replace(name, 'msk'))
                if not msk_path.exists():
                    continue

                # get the size of the grayscale image to match while resizing
                res_path = Path(str(img_path).replace('.nii.gz', f'-{suffix}.nii.gz'))
                o_size = self.cfg['size']
                self.cfg['size'] = (sitk.ReadImage(str(res_path))).GetSize()

                # get the output name
                img_res_path = Path(str(msk_path).replace(
                    '.nii.gz', f'-{self.cfg["suffix"]}.nii.gz'))

                # resize if necessary
                if not img_res_path.exists() or self.cfg['force']:
                    self.resample_img(msk_path, img_res_path, mask=True)

                new_files.append(img_res_path)
                self.cfg['size'] = o_size
                break
        return new_files
