# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import logging
import shutil
import subprocess
import SimpleITK as sitk
from pathlib import Path
from typing import List

# logging.basicConfig(level=logging.INFO)


def elastix_wrapper(
    fix_img_path: Path,
    mov_img_path: Path,
    result_path: Path,
    parameters_paths: List[Path],
    fix_mask_path: Path = None,
    mov_mask_path: Path = None,
    keep_just_useful_files: bool = True,
    write_image: bool = True,
    verbose: bool = False,
    transformation_file_path: Path = None
) -> Path:
    # TODO: Check docstring
    """Wraps Elastix command line interface into a python function
    Args:
        fix_img_path (Path): Path to the fix image
        mov_img_path (Path): Path to the moving image
        result_path (Path): Path where to store the register image and transformation parameters
        fix_mask_path (Path, optional): Path to the fix image mask.
            Defaults to None which means no mask is used
        mov_mask_path (Path, optional): Path to the moving image mask.
            Defaults to None which means no mask is used
        parameters_paths (List[Path]): List of Paths to the parameters map file to use
        keep_just_useful_files (bool, optional): Wheter to delete the rubish Elastix outputs.
            Defaults to True.
        verbose (bool, optional): Wheter to the print the logs of elastix
            Defaults to False.
    Returns:
        (Path): Path where the transformation matrix is stored
    """
    # Fix filenames and create folders
    mov_img_name = mov_img_path.name.replace(''.join(mov_img_path.suffixes), '')
    if '.' in result_path.name:
        if '.nii' not in result_path.name:
            raise Exception('Output file should have nifti extension')
        else:
            res_filename = f'{result_path.name.replace("".join(result_path.suffixes), "")}.nii.gz'
        result_path = result_path.parent / 'res_tmp'
    else:
        res_filename = f'{mov_img_name}_reg.nii.gz'
        result_path = result_path / 'res_tmp'
    result_path.mkdir(exist_ok=True, parents=True)

    # Run elastix
    if (fix_mask_path is not None) and (mov_mask_path is not None):
        command = [
            'elastix', '-out', str(result_path), '-f', str(fix_img_path), '-m', str(mov_img_path),
            '-fMask', str(fix_mask_path), '-mMask', str(mov_mask_path)
        ]
    elif (fix_mask_path is not None):
        command = [
            'elastix', '-out', str(result_path), '-f', str(fix_img_path), '-m', str(mov_img_path),
            '-fMask', str(fix_mask_path)
        ]
    elif (mov_mask_path is not None):
        command = [
            'elastix', '-out', str(result_path), '-f', str(fix_img_path), '-m', str(mov_img_path),
            '-mMask', str(mov_mask_path)
        ]
    else:
        command = [
            'elastix', '-out', str(result_path), '-f', str(fix_img_path), '-m', str(mov_img_path)]

    for i in parameters_paths:
        command.extend(['-p', i])

    if verbose:
        logging.info(command)
        subprocess.call(command)
    else:
        # logging.info(command)
        subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    n = len(parameters_paths) - 1
    # Fix resulting filenames
    if write_image:
        (result_path/f'result.{n}.nii.gz').rename(result_path.parent/res_filename)
    if transformation_file_path is None:
        transformation_file_name = f'TransformParameters_{mov_img_name}.txt'
        transformation_file_path = result_path.parent/transformation_file_name
    shutil.copyfile(
        str(result_path/f'TransformParameters.{n}.txt'),
        str(transformation_file_path)
    )
    if keep_just_useful_files:
        shutil.rmtree(result_path)
    return transformation_file_path


def transformix_wrapper(
    mov_img_path: Path,
    result_path: Path,
    transformation_path: Path,
    keep_just_useful_files: bool = True,
    points: bool = False,
    verbose: bool = False
):
    """Wraps elastix command line interfase into a python function
    Args:
        mov_img_path (Path): Path to the moving image
        result_path (Path): Path where to store the register image and transformation parameters
        transformation_path (Path): Path to the transformation map file
        keep_just_useful_files (bool, optional): Wheter to delete the rubish Elastix outputs.
            Defaults to True.
        points (bool, optional): Wheter to the things to transform are points or img
            Defaults to False.
        verbose (bool, optional): Wheter to the print the logs of elastix
            Defaults to False.
    """
    # TODO: Check docstring
    # Fix filenames and create folders
    if points:
        if result_path.name.endswith('.txt'):
            res_filename = f'{result_path.name.split(".")[0]}.txt'
            result_path = result_path.parent / 'res_tmp'
        else:
            mov_img_name = mov_img_path.name.split(".")[0]
            res_filename = f'{mov_img_name}.txt'
            result_path = result_path / 'res_tmp'
        result_path.mkdir(exist_ok=True, parents=True)
    else:
        if result_path.name.endswith('.img') or ('.nii' in result_path.name):
            res_filename = f'{result_path.name.split(".")[0]}.nii.gz'
            result_path = result_path.parent / 'res_tmp'
        else:
            mov_img_name = mov_img_path.name.split(".")[0]
            res_filename = f'{mov_img_name}.nii.gz'
            result_path = result_path / 'res_tmp'
        result_path.mkdir(exist_ok=True, parents=True)

    # Run transformix
    command = ['transformix', '-out', str(result_path), '-tp', str(transformation_path)]
    if points:
        command.extend(['-def', str(mov_img_path)])
    else:
        command.extend(['-in', str(mov_img_path)])

    if verbose:
        logging.warning(command)
        subprocess.call(command)
    else:
        subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Fix resulting filenames
    if points:
        shutil.copyfile(str(result_path/'outputpoints.txt'), str(result_path.parent/res_filename))
    else:
        (result_path/'result.nii.gz').rename(result_path.parent/res_filename)
    if keep_just_useful_files:
        shutil.rmtree(result_path)


def modify_field_parameter_map(
    field_value_list: List[tuple], in_par_map_path: Path, out_par_map_path: Path = None
):
    """Modifies the parameter including/overwriting the Field/Value pairs passed
    Args:
        field_value_list (List[tuple]): List of (Field, Value) pairs to modify
        in_par_map_path (Path): Path to the original parameter file
        out_par_map_path (Path, optional): Path to the destiny parameter file
            if None, then the original is overwritten. Defaults to None.
    """
    pm = sitk.ReadParameterFile(str(in_par_map_path))
    for [field, value] in field_value_list:
        if isinstance(value, list):
            pm[field] = (val for val in value)
        elif isinstance(value, bool):
            value = 'true' if value else 'false'
            pm[field] = (value, )
        else:
            pm[field] = (value, )
    out_par_map_path = in_par_map_path if out_par_map_path is None else out_par_map_path
    sitk.WriteParameterFile(pm, str(out_par_map_path))
