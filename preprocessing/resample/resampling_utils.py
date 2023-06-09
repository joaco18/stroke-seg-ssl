# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import logging
import shutil

import numpy as np
import SimpleITK as sitk

from pathlib import Path
from typing import Tuple
import preprocessing.freesurfer_utils as fs


def resolution_and_size_match(
    img1_path: Path, img2_path: Path,
    resolution_tolerance: float = 0.05
) -> bool:
    """Checks if the resolution (with a certain tolerance) and the shape of
    the images is the same.
    Args:
        img1_path (Path): Path of one of the images to compare.
        img2_path (Path): Path of other image to compare.
        resolution_tolerance (float, optional): Tolerance to
            check if the resolution matches. Defaults to 0.05.
    Returns:
        bool: True if the images match, False if not.
    """
    img1 = sitk.ReadImage(str(img1_path))
    img2 = sitk.ReadImage(str(img2_path))
    shape1 = np.asarray(img2.GetSize())
    shape2 = np.asarray(img2.GetSize())
    if not all(shape1 == shape2):
        return False
    spacing1 = np.asarray(img1.GetSpacing())
    spacing2 = np.asarray(img2.GetSpacing())
    if not all(np.abs(spacing1 - spacing2) < resolution_tolerance):
        return False
    return True


def resample_img(
    img_path: Path, resample_img_path: Path, interpolator: str,
    new_vox_size: Tuple[float, float, float] = (1., 1., 1.),
    apply_gaussian_smoothing: bool = False, new_img_size: Tuple[int] = None
) -> None:
    assert interpolator in ['nearest', 'linear'], \
        f'Interpolator "{interpolator}" not supported'
    im, aff, head = fs.load_volume(str(img_path), im_only=False)
    im, aff = fs.resample_volume(
        im, aff, np.array(new_vox_size),
        interpolation=interpolator,
        apply_gaussian_smoothing=apply_gaussian_smoothing,
        new_img_size=new_img_size
    )
    fs.save_volume(im, aff, head, str(resample_img_path))


def resample_img_with_tolerance(
    img_path: Path, resample_img_path: Path, interpolator: str,
    new_resolution: Tuple[float, float, float] = (1., 1., 1.),
    up_margin: float = 1.05, low_margin: float = 0.95,
    force: bool = False, apply_gaussian_smoothing: bool = False
) -> None:
    """Check the image spacing, and if it is outside the specified resolution
    margins, resamples it to the specified resolution. If force is True, even
    when the image is inside the margins, it creates a copy of the image with
    the specified filename.
    Args:
        img_path (Path): Path to the image to resample.
        resample_img_path (Path): Path where the image resampled image is saved.
        interpolator (str): Interpolator to use in the resampling. Options are:
            ['nearest', 'linear', 'bs1', 'bs2', 'bs3'].
        new_resolution (Tuple[float, float, float], optional):
            Desired resolution in mm. Defaults to (1., 1., 1.).
        up_margin (float, optional): Margin of resolution above the desired one.
            Defaults to 0.05.
        low_margin (float, optional): Margin of resolution below the desired one.
            Defaults to 0.95.
        force (bool, optional): If True, creates a copy of the image with
            the specified filename when the image has a resolution inside the
            indicated limits. Defaults to False.
        apply_gaussian_smoothing (bool): Apply gaussian smoothing before resampling.
            Defaults to False.
    """
    img_resolution = np.array(sitk.ReadImage(str(img_path)).GetSpacing())
    # resample image if necessary
    top_limit = np.asarray(new_resolution) + up_margin
    bottom_limit = np.asarray(new_resolution) - low_margin
    if np.any((img_resolution > top_limit) | (img_resolution < bottom_limit)):
        resample_img(
            img_path, resample_img_path, interpolator,
            new_resolution, apply_gaussian_smoothing
        )
    elif force:
        logging.warning(
            f'Image already in the desired resolution range '
            f'-({new_resolution}) + {up_margin} - {low_margin}-'
            'copying the image with the desired name only.'
        )
        shutil.copyfile(str(img_path), str(resample_img_path))


def resample_img_with_size_and_tolerance(
    img_path: Path, resample_img_path: Path, interpolator: str,
    new_img_size: Tuple[int, int, int],
    new_resolution: Tuple[float, float, float] = (1., 1., 1.),
    up_margin: float = 1.05, low_margin: float = 0.95,
    force: bool = False, apply_gaussian_smoothing: bool = False
) -> None:
    """Resample the image to match certain image size and check that the output
    resolution falls inside a desired margin. If this condition is not matched,
    just rescale the image and print a warning.
    Args:
        img_path (Path): Path to the image to resample.
        resample_img_path (Path): Path where the image resampled image is saved.
        interpolator (str): Interpolator to use in the resampling. Options are:
            ['nearest', 'linear', 'bs1', 'bs2', 'bs3'].
        new_img_size (Tuple[int, int, int]): Size of the resampled image to generate.
        new_resolution (Tuple[float, float, float], optional):
            Desired resolution in mm. Defaults to (1., 1., 1.).
        up_margin (float, optional): Margin of resolution above the desired one.
            Defaults to 0.05.
        low_margin (float, optional): Margin of resolution below the desired one.
            Defaults to 0.95.
        force (bool, optional): If True, creates a copy of the image with
            the specified filename when the image has a resolution inside the
            indicated limits. Defaults to False.
        apply_gaussian_smoothing (bool): Apply gaussian smoothing before resampling.
            Defaults to False.
    """
    im, aff, _ = fs.load_volume(str(img_path), im_only=False)
    original_spacing = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    original_size = im.shape
    empirical_vox_size = (
        ((original_spacing[0] * original_size[0]) / new_img_size[0]),
        ((original_spacing[1] * original_size[1]) / new_img_size[1]),
        ((original_spacing[2] * original_size[2]) / new_img_size[2])
    )
    top_limit = np.asarray(new_resolution) + up_margin
    bottom_limit = np.asarray(new_resolution) - low_margin
    if np.any((empirical_vox_size > top_limit) | (empirical_vox_size < bottom_limit)):
        logging.warning(
            f'Resizing to shape {new_img_size} generated unallowed voxel resolutions: '
            f'{empirical_vox_size}. \n'
            f'Resizing to the desired resultion ({new_resolution}) only.\nImages:'
            f'\n\t{img_path}\n\t{resample_img_path}'
        )
        resample_img(
            img_path, resample_img_path, interpolator,
            new_resolution, apply_gaussian_smoothing
        )
    else:
        resample_img(
            img_path, resample_img_path, interpolator,
            empirical_vox_size, apply_gaussian_smoothing, new_img_size
        )


def resample_img_with_size(
    img_path: Path, resample_img_path: Path, interpolator: str,
    new_img_size: Tuple[int, int, int],
    apply_gaussian_smoothing: bool = False
) -> None:
    """Resample the image to match certain image size.
    Args:
        img_path (Path): Path to the image to resample.
        resample_img_path (Path): Path where the image resampled image is saved.
        interpolator (str): Interpolator to use in the resampling. Options are:
            ['nearest', 'linear', 'bs1', 'bs2', 'bs3'].
        new_img_size (Tuple[int, int, int]): Size of the resampled image to generate.
        apply_gaussian_smoothing (bool): Apply gaussian smoothing before resampling.
            Defaults to False.
    """
    im, aff, _ = fs.load_volume(str(img_path), im_only=False)
    original_spacing = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    original_size = im.shape
    empirical_vox_size = (
        ((original_spacing[0] * original_size[0]) / new_img_size[0]),
        ((original_spacing[1] * original_size[1]) / new_img_size[1]),
        ((original_spacing[2] * original_size[2]) / new_img_size[2])
    )
    resample_img(
        img_path, resample_img_path, interpolator,
        empirical_vox_size, apply_gaussian_smoothing, new_img_size
    )
