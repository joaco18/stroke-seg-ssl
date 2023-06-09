# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

from pathlib import Path
from typing import Tuple
import SimpleITK as sitk


def check_available_modalities(
    ncct_path: Path,
    mri_path: Path,
    msk: bool = False
) -> Tuple[Path, Path, Path]:
    """Checks if the ncct and mri paths exist. Checks if the mri path is
    ADC or DWI, and checks if other option exists.
    Returns the NCCT, ADC, and DWI paths, if any of the mri ones doesn't exist
    it returns None insted.
    Args:
        ncct_path (Path): NCCT image path
        adc_path (Path): ADC image path
        dwi_path (Path): DWI image path
        msk (bool, optional): To check if msk file exists. Defauls to False.
    Returns:
        Tuple[Path, Path, Path]: NCCT, ADC and DWI paths.
        Tuple[Path, Path, Path, Path]: NCCT, ADC, DWI and MSK paths.
    """
    assert ncct_path.exists(), \
        'Ncct image path: \n\t {ncct_path} doesn\'t exists.'

    # Check which of the two mri modalities exist
    if mri_path is not None:
        if 'adc' in str(mri_path):
            assert mri_path.exists(), f'ADC image path: \n\t {ncct_path} doesn\'t exists.'
            adc_path = mri_path
            dwi_path = Path(str(mri_path).replace('adc', 'dwi'))
            if not dwi_path.exists():
                dwi_path = None

        elif 'dwi' in str(mri_path):
            assert mri_path.exists(), f'DWI image path: \n\t {ncct_path} doesn\'t exists.'
            dwi_path = mri_path
            adc_path = Path(str(mri_path).replace('dwi', 'adc'))
            if not adc_path.exists():
                adc_path = None
    else:
        dwi_path, adc_path = None, None

    # Check which of the two mri modalities exist
    if msk:
        msk_path = None
        paths = [ncct_path, adc_path, dwi_path]
        names = ['ncct', 'adc', 'dwi']
        for name, path in zip(names, paths):
            if path is not None:
                mask_path = Path(str(path).replace(name, 'msk'))
                if mask_path.exists():
                    msk_path = mask_path
                    break
        return ncct_path, adc_path, dwi_path, msk_path
    return ncct_path, adc_path, dwi_path


def read_write_sitk(ncct_path: Path, mri_path: Path) -> None:
    """
    Silly function to load and write images so SimpleITK doesn't complain.
    """
    paths = list(check_available_modalities(ncct_path, mri_path, msk=True))
    for path in paths:
        if (path is not None):
            img = sitk.ReadImage(str(path))
            sitk.WriteImage(img, str(path))
