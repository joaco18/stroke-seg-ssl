# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import json
import multiprocessing as mp

from copy import copy
from pathlib import Path
from typing import List, Dict
import numpy as np
import SimpleITK as sitk

import preprocessing.registration.elastix_utils as e_utils
from preprocessing.preprocessing_utils import check_available_modalities
from utils.utils import mask_image, save_img_from_array_using_referece, extend_image
from utils.utils import intensity_clipping


N_JOBS = mp.cpu_count()
PREPROCESSING_PATH = Path(__file__).parent.parent.resolve()


def sagital_flip(img_path: Path, out_path: Path) -> None:
    """Given an image it flips it from left to right (sagital flip)
    Args:
        img_path (Path): Path to the image to flip (should be in MNI space)
        out_path (Path): Path where to save the image
    """
    img = sitk.ReadImage(str(img_path))
    img_array = sitk.GetArrayFromImage(img)
    img_array = np.flip(img_array, 2)
    save_img_from_array_using_referece(img_array, img, out_path)


def get_diff(img1_path: Path, img2_path: Path, out_path: Path) -> None:
    """Given two images it computes the difference between them and saves it.
    Img1 - Img2 = Output
    Args:
        img1_path (Path): Path to the image to substract from
        img2_path (Path): Path to the image beeing substracted
        out_path (Path): Path where to save the difference image
    """
    img1 = sitk.ReadImage(str(img1_path))
    img2 = sitk.ReadImage(str(img2_path))

    img1_array = sitk.GetArrayFromImage(img1)
    img2_array = sitk.GetArrayFromImage(img2)

    diff = img1_array - img2_array
    save_img_from_array_using_referece(diff, img1, out_path)


def extend_image_file(
    img_path: Path, out_path: Path, ref_path: Path, bkgd_val: int, bbox_meta: Dict
) -> None:
    ref = sitk.ReadImage(str(ref_path))
    cropped_array = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
    extended = extend_image(bbox_meta=bbox_meta,
                            cropped_array=cropped_array,
                            bkgd_val=bkgd_val)
    save_img_from_array_using_referece(extended, ref, out_path)


class DifferenceImageGenerator():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.atlas_path = Path(self.cfg['mni_template_path'])
        self.vm_path = Path(self.cfg['mni_vascular_map_path'])
        self.vm_t1_path = Path(self.cfg['mni_vascular_map_t1_path'])
        pmaps_path = PREPROCESSING_PATH/'registration/parameter_maps'
        self.pmaps = [pmaps_path/path for path in self.cfg['parameter_maps']]
        # self.transf_map_path = None
        self.diff_suffix = self.cfg['diff_suffix']
        self.cntrlt_suffix = self.cfg['cntrlt_suffix']
        self.over_crop = '-c' if self.cfg['apply_over_img'].endswith('-c') else ''

    def __call__(self, ncct_path: Path, mri_path: Path) -> List[Path]:
        ncct_path, adc_path, dwi_path = check_available_modalities(ncct_path, mri_path)
        new_files = []

        # Define NCCT path
        ncct_img_name = self.cfg['apply_over_img']
        assert (ncct_path is not None), 'Ncct path not defined in difference image generation'
        if ('tilt' in ncct_path.name) and ('ncct-tilt' not in ncct_img_name):
            ncct_img_name = ncct_img_name.replace('ncct', 'ncct-tilt')
        base_name = ("_").join(ncct_path.name.split("_")[:2])
        ncct_path = ncct_path.parent / f'{base_name}_{ncct_img_name}.nii.gz'
        ref_ncct_path = ncct_path.parent / ncct_path.name.replace('-c.nii.gz', '.nii.gz')
        assert ncct_path.exists(), \
            f'The required NCCT img to generate the difference does not exist:\n{str(ncct_path)}'

        # Define BM path
        bm_name = self.cfg['brain_mask']
        if 'ncct' in bm_name:
            if ('tilt' in ncct_path.name) and ('ncct-tilt' not in bm_name):
                bm_name = bm_name.replace('ncct', 'ncct-tilt')
            bm_path = ncct_path.parent / f'{base_name}_{bm_name}.nii.gz'
        elif 'adc' in bm_name:
            new_name = f'{("_").join(adc_path.name.split("_")[:2])}_{bm_name}.nii.gz'
            bm_path = adc_path.parent / new_name
        elif 'dwi' in bm_name:
            new_name = f'{("_").join(dwi_path.name.split("_")[:2])}_{bm_name}.nii.gz'
            bm_path = dwi_path.parent / new_name
        else:
            raise Exception(
                f'The required brain mask in diff image does not exist:\n{str(bm_name)}')
        assert bm_path.exists(), \
            f'The required brain mask in difference does not exist:\n{str(bm_path)}'

        # Define outputs paths
        vm_reg_path = ncct_path.parent / f'{base_name}_vm-pp{self.over_crop}.nii.gz'
        diff_path = ncct_path.parent / f'{base_name}_{self.diff_suffix}-pp{self.over_crop}.nii.gz'
        flip_path = ncct_path.parent / ncct_path.name.replace('.nii.gz',
                                                              f'{self.cntrlt_suffix}.nii.gz')
        bm_flip_path = bm_path.parent / bm_path.name.replace('.nii.gz', '-flip.nii.gz')
        reg_path = ncct_path.parent / ncct_path.name.replace('.nii.gz', '-mni.nii.gz')
        t_map_path = ncct_path.parent / 'rough_mni.txt'

        files_not_exists = self.cfg['register_vascular_map'] and (not vm_reg_path.exists())
        files_not_exists = files_not_exists or (not flip_path.exists()) or (not diff_path.exists())

        if files_not_exists or self.cfg['force']:
            # Register NCCT to MNI
            e_utils.elastix_wrapper(fix_img_path=self.atlas_path,
                                    mov_img_path=ncct_path,
                                    result_path=reg_path,
                                    parameters_paths=self.pmaps,
                                    transformation_file_path=t_map_path,
                                    verbose=self.cfg['verbose'],
                                    write_image=False)

            # Register NCCT to MNI
            field_value_pairs = [('ResultImageFormat', 'nii.gz'), ('ResultImagePixelType', "float"),
                                 ('FinalBSplineInterpolationOrder', '3')]
            e_utils.modify_field_parameter_map(field_value_pairs, t_map_path)
            e_utils.transformix_wrapper(mov_img_path=ncct_path,
                                        result_path=reg_path,
                                        transformation_path=t_map_path)

            # Register BM to MNI
            field_value_pairs = [('ResultImagePixelType', "int"),
                                 ('FinalBSplineInterpolationOrder', '0')]
            e_utils.modify_field_parameter_map(field_value_pairs, t_map_path)
            e_utils.transformix_wrapper(mov_img_path=bm_path,
                                        result_path=bm_flip_path,
                                        transformation_path=t_map_path,
                                        verbose=self.cfg['verbose'])
            # Mask the image after registration to avoid background alteration
            mask_image(reg_path, bm_flip_path, reg_path, -100)

            # FLIP IMAGES
            flip_path = Path(str(ncct_path).replace('.nii.gz', '-flip.nii.gz'))
            sagital_flip(reg_path, flip_path)
            sagital_flip(bm_flip_path, bm_flip_path)

            # Register Flipped to original space
            e_utils.elastix_wrapper(fix_img_path=ncct_path,
                                    mov_img_path=flip_path,
                                    result_path=flip_path,
                                    parameters_paths=self.pmaps,
                                    transformation_file_path=t_map_path,
                                    verbose=self.cfg['verbose'],
                                    write_image=False)

            # REGISTER FLIPPED NCCT
            field_value_pairs = [('ResultImageFormat', 'nii.gz'), ('ResultImagePixelType', "float"),
                                 ('FinalBSplineInterpolationOrder', '3')]
            e_utils.modify_field_parameter_map(field_value_pairs, t_map_path)
            e_utils.transformix_wrapper(mov_img_path=flip_path,
                                        result_path=flip_path,
                                        transformation_path=t_map_path,
                                        verbose=self.cfg['verbose'])
            # Register flipped brain mask
            field_value_pairs = [('ResultImagePixelType', "int"),
                                 ('FinalBSplineInterpolationOrder', '0')]
            e_utils.modify_field_parameter_map(field_value_pairs, t_map_path)
            e_utils.transformix_wrapper(mov_img_path=bm_flip_path,
                                        result_path=bm_flip_path,
                                        transformation_path=t_map_path,
                                        verbose=self.cfg['verbose'])
            # Mask the image to avoid wrong background values
            mask_image(flip_path, bm_flip_path, flip_path, -100)
            intensity_clipping(img_path=flip_path, out_path=flip_path, v_min=-100, v_max=400)

            # GET DIFFERENCE MAP
            get_diff(flip_path, ncct_path, diff_path)

            # REGISTER VASCULAR MAPS
            if self.cfg['register_vascular_map']:
                e_utils.elastix_wrapper(fix_img_path=ncct_path,
                                        mov_img_path=self.vm_t1_path,
                                        result_path=reg_path,
                                        parameters_paths=self.pmaps,
                                        transformation_file_path=t_map_path,
                                        verbose=self.cfg['verbose'],
                                        write_image=False)
                field_value_pairs = [('ResultImageFormat', 'nii.gz'),
                                     ('ResultImagePixelType', "int"),
                                     ('FinalBSplineInterpolationOrder', '0')]
                e_utils.modify_field_parameter_map(field_value_pairs, t_map_path)
                e_utils.transformix_wrapper(mov_img_path=self.vm_path,
                                            result_path=vm_reg_path,
                                            transformation_path=t_map_path)
                new_files.append(vm_reg_path)
            reg_path.unlink()
            new_files = new_files + [bm_flip_path, diff_path, flip_path]

        if (self.cfg['save_original_size']) and (self.over_crop):
            subject = (base_name.split('_')[0]).replace('sub-', '')
            with open(self.cfg['bboxes_info_path'], 'r') as jfile:
                bbxes_info = json.load(jfile)
            assert (subject in [key for key in bbxes_info]), \
                f'Check bbox metadata file subject -{subject}- is missing'
            files_to_extend = copy(new_files)
            for file_path in files_to_extend:
                out_path = file_path.parent / file_path.name.replace('-c.nii.gz', '.nii.gz')
                out_path = out_path.parent / out_path.name.replace('-c-', '-')
                if (not out_path.exists()) or self.cfg['force']:
                    discrete_files = [diff_path.name, vm_reg_path.name,
                                      bm_flip_path.name, bm_path.name]
                    bkgd = 0 if file_path.name in discrete_files else -100
                    extend_image_file(
                        file_path, out_path, ref_ncct_path, bkgd, bbxes_info[subject])
                    new_files.append(out_path)
        return new_files
