# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

from pathlib import Path
from typing import List
from preprocessing.preprocessing_utils import check_available_modalities
from utils.utils import intensity_clipping


class IntensityClipper():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.modalities = [key for key in self.cfg]

    def __call__(self, ncct_path: Path, mri_path: Path) -> List[Path]:
        new_files = []
        ncct_path, adc_path, dwi_path = check_available_modalities(ncct_path, mri_path)
        for modality in self.modalities:
            mod_cfg = self.cfg[modality]
            # Define the image paths
            if ('ncct' in modality) and (ncct_path is not None):
                if ('tilt' in ncct_path.name) and ('ncct-tilt' not in modality):
                    modality = modality.replace('ncct', 'ncct-tilt')
                new_name = f'{("_").join(ncct_path.name.split("_")[:2])}_{modality}.nii.gz'
                img_path = ncct_path.parent / new_name
                assert img_path.exists(), \
                    f'The required image to clip intensities does not exist:\n{str(img_path)}'
            elif ('adc' in modality) and (adc_path is not None):
                new_name = f'{("_").join(adc_path.name.split("_")[:2])}_{modality}.nii.gz'
                img_path = adc_path.parent / new_name
                assert img_path.exists(), \
                    f'The required image to clip intensities does not exist:\n{str(img_path)}'
            elif ('dwi' in modality) and (dwi_path is not None):
                new_name = f'{("_").join(dwi_path.name.split("_")[:2])}_{modality}.nii.gz'
                img_path = dwi_path.parent / new_name
                assert img_path.exists(), \
                    f'The required image to clip intensities does not exist:\n{str(img_path)}'
            sfx = self.cfg["suffix"]
            out_path = img_path.parent / img_path.name.replace('.nii.gz', f'-{sfx}.nii.gz')

            intensity_clipping(
                img_path=img_path, out_path=out_path, v_min=mod_cfg['min'], v_max=mod_cfg['max']
            )
            new_files.append(out_path)
        return new_files
