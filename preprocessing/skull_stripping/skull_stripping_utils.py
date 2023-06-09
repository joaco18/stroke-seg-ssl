# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import logging
import subprocess
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from scipy.spatial import ConvexHull, Delaunay

from preprocessing.skull_stripping import synthstrip
# from utils.utils import save_img_from_array_using_referece

# logging.basicConfig(level=logging.INFO)
PREPROCESSING_PATH = Path(__file__).parent.parent.resolve()


def hdbet_wrapper(
    input_img_path: Path,
    output_img_path: Path,
    device: str = 'cpu',
    mode: str = 'accurate',
    tta: bool = True,
    post_proc: bool = True,
    verbose: bool = False,
    save_mask: bool = False,
    output_bm_path: Path = None,
) -> bool:
    """TODO: Add docstring """
    # Fix filenames and create folders
    output_img_path.parent.mkdir(exist_ok=True, parents=True)

    tta = 1 if tta else 0
    post_proc = 1 if post_proc else 0

    command = \
        f'hd-bet -i {input_img_path} -o {output_img_path} '\
        f'-device {device} -mode {mode} -tta {tta} -pp {post_proc}'

    if save_mask:
        command = f'{command} --save_mask 1'

    if verbose:
        subprocess.run(command, shell=True)
    else:
        subprocess.run(
            command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not output_img_path.exists():
            logging.warning(f'Synthseg on file {input_img_path} failed.')
    default_mask_name = output_img_path.name.replace('.nii.gz', '_mask.nii.gz')
    default_mask_path = output_img_path.parent / default_mask_name
    if save_mask:
        if not default_mask_path.exists():
            return False
        default_mask_path.rename(output_bm_path)
    return True


def synthstrip_wrapper(
    input_img_path: Path,
    output_img_path: Path,
    output_bm_path: Path,
    gpu: bool = True,
    border: int = 1,
    no_csf: bool = False,
    model_file: Path = None
) -> bool:
    """TODO: Add docstring """
    # Fix filenames and create folders
    output_img_path.parent.mkdir(exist_ok=True, parents=True)

    if model_file is None:
        model = 'synthstrip.nocsf.1.pt' if no_csf else 'synthstrip.1.pt'
        model_path = PREPROCESSING_PATH / 'skull_stripping/models' / model
    else:
        model_path = model_file

    # Get the brain mask / stripping
    synthstrip.predict(
        str(input_img_path),
        str(output_img_path),
        str(output_bm_path),
        gpu=gpu,
        no_csf=no_csf,
        border=border,
        model_path=str(model_path),
    )
    if not output_img_path.exists():
        logging.warning(f'Synth Strip on file {input_img_path} failed.')
        return False
    return True


def freesurfer_synthstrip_wrapper(
    input_img_path: Path,
    output_img_path: Path,
    output_bm_path: Path,
    gpu: bool = True,
    border: int = 1,
    no_csf: bool = False,
    verbose: bool = False
) -> bool:
    """TODO: Add docstring """
    # Fix filenames and create folders
    output_img_path.parent.mkdir(exist_ok=True, parents=True)
    skullstrip_path = Path(__file__).resolve().parent

    command = \
        f'mri_synthstrip --i {str(input_img_path)} '\
        f' --o {str(output_img_path)} --mask {output_bm_path} --border {border}'

    if no_csf:
        command = f'{command} --no-csf'
    if gpu:
        command = f'{command} --gpu'

    synthstrip_sh_file = \
        '#!/bin/bash\n' \
        'export FREESURFER_HOME=$HOME/freesurfer-dev\n' \
        'source $FREESURFER_HOME/SetUpFreeSurfer.sh\n' \
        f'{command}'

    bash_file_path = skullstrip_path/'synthsstrip.sh'
    with open(bash_file_path, 'w') as sh_file:
        sh_file.write(synthstrip_sh_file)

    if verbose:
        print(command)
        subprocess.run(f'bash {str(bash_file_path)}', shell=True)
    else:
        subprocess.run(
            f'bash {str(bash_file_path)}', shell=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    if not output_img_path.exists():
        logging.warning(f'Synthseg on file {input_img_path} failed.')
        return False
    return True


def fill_hull(array: np.ndarray):
    """ Computes the convex hull of the given binary array and returns a mask
    of the filled hull.
    """
    assert (np.array(array.shape) <= np.iinfo(np.int16).max).all(), \
        f"This function assumes your image is smaller than {2**15} in each dimension"

    points = np.argwhere(array).astype(np.int16)
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices])

    # Instead of allocating a giant array for all indices in the volume,
    # just iterate over the slices one at a time.
    idx_2d = np.indices(array.shape[1:], np.int16)
    idx_2d = np.moveaxis(idx_2d, 0, -1)

    idx_3d = np.zeros((*array.shape[1:], array.ndim), np.int16)
    idx_3d[:, :, 1:] = idx_2d

    mask = np.zeros_like(array, dtype=bool)
    for z in range(len(array)):
        idx_3d[:, :, 0] = z
        s = deln.find_simplex(idx_3d)
        mask[z, (s != -1)] = 1

    return mask


def check_convexity_criteria_over_imgs(mask_path: Path, threshold: float) -> bool:
    """ Checks if the percetual difference between the area of the convex hull of the
    mask and the mask itself is bigger than a specified threshold. If so, return False
    (meaning 'doesn't not verify the criteria'), else True.
    Args:
        mask_path (Path): Path where the mask is stored.
        threshold (float): Area percentage difference threshold.
    Returns:
        bool: False if the difference in areas is bigger than the threshold.
    """
    mask = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask)
    mask_array = np.where(mask_array > 0, 1, 0).astype('uint8')
    convex_hull_filled = np.where(fill_hull(mask_array), 1, 0).astype('uint8')
    convex_hull_filled_volume = np.sum(convex_hull_filled)
    mask_volume = np.sum(mask_array)
    # save_img_from_array_using_referece(
    #   convex_hull_filled, mask, str(mask_path.parent/'temp.nii.gz'))
    perc_area_diff = (convex_hull_filled_volume - mask_volume) / mask_volume
    if perc_area_diff > threshold:
        return perc_area_diff, False
    return perc_area_diff, True
