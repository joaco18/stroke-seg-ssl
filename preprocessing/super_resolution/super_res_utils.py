# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import logging
import multiprocessing
import subprocess
from preprocessing.super_resolution import synthsr
from pathlib import Path


# logging.basicConfig(level=logging.INFO)
N_JOBS = multiprocessing.cpu_count()
PREPROCESSING_PATH = Path(__file__).parent.parent.resolve()


def synthsr_wrapper(
    input_img_path: Path,
    output_img_path: Path,
    n_threads: int = N_JOBS,
    v1: bool = False,
    cpu: bool = True,
    ct: bool = True,
    model_file: Path = None,
    disable_sharpening: bool = False,
    disable_flipping: bool = False
) -> None:

    if model_file is None:
        model = 'synthsr_v10_210712.h5' if v1 else 'synthsr_v20_230130.h5'
        path_model = PREPROCESSING_PATH/'super_resolution/models'/model
    else:
        path_model = model_file

    synthsr.predict(
        str(input_img_path),
        str(output_img_path),
        str(path_model),
        ct,
        disable_sharpening,
        disable_flipping,
        threads=n_threads,
        cpu=cpu
    )
    if not output_img_path.exists():
        logging.warning(f'SynthSR on file {input_img_path} failed.')
    return None


def freesurfer_synthsr_wrapper(
    input_img_path: Path,
    output_img_path: Path,
    n_threads: int = N_JOBS,
    v1: bool = False,
    cpu: bool = True,
    ct: bool = True,
    disable_sharpening: bool = False,
    disable_flipping: bool = False,
    verbose: bool = False
) -> None:
    # Fix filenames and create folders
    output_img_path.parent.mkdir(exist_ok=True, parents=True)
    synthsr_file_path = Path(__file__).resolve().parent

    command = \
        f'mri_synthsr --i {str(input_img_path)} --o {str(output_img_path)} --threads {n_threads}'
    if ct:
        command = f'{command} --ct'
    if disable_sharpening:
        command = f'{command} --disable_sharpening'
    if disable_flipping:
        command = f'{command} --disable_flipping'
    if cpu:
        command = f'{command} --cpu'

    model_filename = 'SynthSR_v10_210712.h5' if v1 else 'synthsr_v20_230130.h5'
    model_path = synthsr_file_path / 'models' / model_filename
    command = f'{command} --model {str(model_path)}'

    synthstrip_sh_file = \
        '#!/bin/bash\n' \
        'export FREESURFER_HOME=$HOME/freesurfer-dev\n' \
        'source $FREESURFER_HOME/SetUpFreeSurfer.sh\n' \
        'export CPATH=$CPATH:$HOME/local/include \n' \
        'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/local/lib \n' \
        f'{command}'

    bash_file_path = synthsr_file_path/'synthsr.sh'
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
            logging.warning(f'SynthSR on file {input_img_path} failed.')
    return
