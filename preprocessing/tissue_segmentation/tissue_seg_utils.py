# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import logging
import multiprocessing
import subprocess

import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from pathlib import Path
from typing import Tuple

import preprocessing.tissue_segmentation.synthseg as synthseg
from preprocessing.tissue_segmentation.constants import SIMPLIFY_SYNTHSEG
from utils.utils import save_img_from_array_using_referece

N_JOBS = multiprocessing.cpu_count()
PREPROCESSING_PATH = Path(__file__).parent.parent.resolve()


def freesurfer_synthseg_wrapper(
    input_img_path: Path,
    output_img_path: Path,
    parc: bool = False,
    robust: bool = True,
    vol: Path = None,
    qc: Path = None,
    post: Path = None,
    resample: Path = None,
    cpu: bool = True,
    n_threads: int = N_JOBS,
    crop: Tuple[int, int, int] = None,
    fast: bool = False,
    v1: bool = False,
    verbose: bool = False
) -> None:
    # Fix filenames and create folders
    output_img_path.parent.mkdir(exist_ok=True, parents=True)

    tissue_seg_path = Path().resolve().parent.parent
    # synthseg_path = tissue_seg_path / 'SynthSeg/scripts/commands/SynthSeg_predict.py'
    # command = \
    #     f'python {synthseg_path} --i {str(input_img_path)} --o {str(output_img_path)}'
    command = \
        f'mri_synthseg --i {str(input_img_path)} --o {str(output_img_path)}'

    if parc:
        command = f'{command} --parc'
    if robust:
        command = f'{command} --robust'
    if vol is not None:
        command = f'{command} --vol {vol}'
    if qc is not None:
        command = f'{command} --qc {qc}'
    if post is not None:
        command = f'{command} --post {post}'
    if resample is not None:
        command = f'{command} --resample {resample}'
    if cpu:
        command = f'{command} --cpu'
    if n_threads is not None:
        command = f'{command} --threads {n_threads}'
    if crop is not None:
        command = f'{command} --crop {crop}'
    if fast:
        command = f'{command} --fast'
    if v1:
        command = f'{command} --v1'

    super_resolution_sh_file = \
        '#!/bin/bash\n' \
        'export FREESURFER_HOME=$HOME/freesurfer-dev\n' \
        'source $FREESURFER_HOME/SetUpFreeSurfer.sh\n' \
        'rm -rf ~/.nv\n' \
        f'{command}'

    # super_resolution_sh_file = \
    #     '#!/bin/bash\n' \
    #     'source ~/anaconda3/bin/activate\n' \
    #     'conda activate stroke\n' \
    #     'export CPATH=$CPATH:$HOME/local/include \n' \
    #     'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/local/lib \n' \
    #     f'{command}'

    bash_file_path = tissue_seg_path/'synthseg_segmentation.sh'
    with open(bash_file_path, 'w') as sh_file:
        sh_file.write(super_resolution_sh_file)

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
            # raise Exception(f'Synthseg on file {input_img_path} failed.')
    return


def synthseg_wrapper(
    input_img_path: Path,
    output_img_path: Path,
    parc: bool = False,
    robust: bool = True,
    vol: Path = None,
    qc: Path = None,
    post: Path = None,
    resample: Path = None,
    cpu: bool = True,
    n_threads: int = N_JOBS,
    crop: Tuple[int, int, int] = None,
    fast: bool = False,
    v1: bool = False
) -> bool:

    synthseg_home = Path(__file__).resolve().parent
    models_path = synthseg_home/'models'

    # impose running constraints according to arguments
    if robust:
        fast = True
        if v1:
            logging.warning(
                'The flag --v1 cannot be used with --robust since '
                'SynthSeg-robust only came out with 2.0.'
            )

    # path models
    if robust:
        path_model_segmentation = models_path/'synthseg_robust_2.0.h5'
    else:
        path_model_segmentation = models_path/'synthseg_2.0.h5'
    path_model_parcellation = models_path/'synthseg_parc_2.0.h5'
    path_model_qc = models_path/'synthseg_qc_2.0.h5'

    # path labels
    labels_segmentation = models_path/'synthseg_segmentation_labels_2.0.npy'
    labels_denoiser = models_path/'synthseg_denoiser_labels_2.0.npy'
    labels_parcellation = models_path/'synthseg_parcellation_labels.npy'
    labels_qc = models_path/'synthseg_qc_labels_2.0.npy'
    names_segmentation_labels = models_path/'synthseg_segmentation_names_2.0.npy'
    names_parcellation_labels = models_path/'synthseg_parcellation_names.npy'
    names_qc_labels = models_path/'synthseg_qc_names_2.0.npy'
    topology_classes = models_path/'synthseg_topological_classes_2.0.npy'
    n_neutral_labels = 19

    # use v1 model if needed
    if v1:
        path_model_segmentation = models_path/'synthseg_1.0.h5'  # to modify
        labels_segmentation = Path(str(labels_segmentation).replace('_2.0.npy', '.npy'))
        labels_qc = Path(str(labels_qc).replace('_2.0.npy', '.npy'))
        names_segmentation_labels = Path(str(names_segmentation_labels).replace('_2.0.npy', '.npy'))
        names_qc_labels = Path(str(names_qc_labels).replace('_2.0.npy', '.npy'))
        topology_classes = Path(str(topology_classes).replace('_2.0.npy', '.npy'))
        n_neutral_labels = 18

    # configure threading options
    tf.config.threading.set_inter_op_parallelism_threads(n_threads)
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)

    # run prediction
    synthseg.predict(
        path_images=str(input_img_path),
        path_segmentations=str(output_img_path),
        path_model_segmentation=str(path_model_segmentation),
        labels_segmentation=str(labels_segmentation),
        robust=robust,
        fast=fast,
        v1=v1,
        do_parcellation=parc,
        n_neutral_labels=n_neutral_labels,
        names_segmentation=str(names_segmentation_labels),
        labels_denoiser=str(labels_denoiser),
        path_posteriors=post,
        path_resampled=resample,
        path_volumes=vol,
        path_model_parcellation=str(path_model_parcellation),
        labels_parcellation=str(labels_parcellation),
        names_parcellation=str(names_parcellation_labels),
        path_qc_scores=qc,
        path_model_qc=str(path_model_qc),
        labels_qc=str(labels_qc),
        names_qc=str(names_qc_labels),
        cropping=crop,
        topology_classes=str(topology_classes),
        cpu=cpu,
        threads=n_threads
    )
    if not output_img_path.exists():
        logging.warning(f'Synthseg on file {input_img_path} failed.')
        return False
    return True


def simplify_synthseg_labels(
    tseg_path: Path,
    save: bool = False,
    output_path: Path = None
) -> np.ndarray:
    """Simplify synthseg output to coarser structure labels.
    Check utils/constants.py to know which number corresponds to which structure.
    Args:
        tseg_path (Path): Path to the synthseg segmentation mask image.
        save (bool, optional): Whether to save the image in addition to returning
            the array. Defaults to False.
        output_path (Path, optional): If saving is indicated an output path should
            be provided. Defaults to None.
    Returns:
        np.ndarray: Simplified segmentation mask.
    """
    tseg = sitk.ReadImage(str(tseg_path))
    tseg_array = sitk.GetArrayFromImage(tseg)
    tseg_array_replace = np.zeros_like(tseg_array)
    for key, val in SIMPLIFY_SYNTHSEG.items():
        tseg_array_replace[tseg_array == key] = val
    if save:
        assert output_path is not None, \
            'An output image path should be provided in synthseg labels' \
            ' simplification if saving is indicated.'
        save_img_from_array_using_referece(
            tseg_array_replace.astype('uint8'), tseg, output_path
        )
    return tseg_array_replace
