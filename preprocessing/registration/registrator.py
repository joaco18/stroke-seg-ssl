# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import copy
import multiprocessing as mp
import SimpleITK as sitk

from pathlib import Path
from typing import Tuple, List, Dict

from preprocessing.preprocessing_utils import check_available_modalities
from utils.metrics import mutual_information, dice_score
import preprocessing.registration.elastix_utils as e_utils
from utils.utils import mask_image

N_JOBS = mp.cpu_count()
PREPROCESSING_PATH = Path(__file__).parent.parent.resolve()


class Registrator():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.method = self.cfg['method']['name']
        self.method_cfg = self.cfg['method']['config']
        if self.method == 'elastix':
            self.pmaps = self.method_cfg['parameter_maps']
            self.transf_map_path = None
        self.suffix = self.cfg['suffix']
        self.f_img = self.cfg['fixed_img']
        self.af_img = self.cfg['alternative_fixed_img']
        self.m_img = self.cfg['moving_img']

    def check_and_define_fix_and_moving_imgs(
        self, ncct_path: Path, mri_path: Path
    ) -> Tuple[Path, Path]:
        base_fpath = ncct_path if ('ncct' in self.f_img) else mri_path
        f_img = copy.copy(self.f_img)
        if ('ncct' in self.f_img):
            base_fname = 'ncct'
            if ('tilt' in str(base_fpath)):
                base_fname = 'ncct-tilt'
                f_img = f_img.replace('ncct', 'ncct-tilt')
        else:
            base_fname = 'adc' if ('adc' in str(base_fpath)) else 'dwi'

        base_mpath = ncct_path if ('ncct' in self.m_img) else mri_path
        m_img = copy.copy(self.m_img)
        if ('ncct' in self.m_img):
            base_mname = 'ncct'
            if ('tilt' in str(base_mpath)):
                base_mname = 'ncct-tilt'
                m_img = m_img.replace('ncct', 'ncct-tilt')
        else:
            base_mname = 'adc' if ('adc' in str(base_mpath)) else 'dwi'

        mov_img_path = Path(str(base_mpath).replace(base_mname, m_img))
        if not mov_img_path.exists():
            raise Exception(f'Required moving path doesn\'t exist:\n\t{str(mov_img_path)}')
        fix_img_path = Path(str(base_fpath).replace(base_fname, f_img))

        if not fix_img_path.exists():
            if self.af_img is not None:
                afix_img_path = Path(str(base_fpath).replace(base_fname, self.af_img))
                if not afix_img_path.exists():
                    raise Exception(
                        'Both the fix image and the alternative fix images indicated don\'t exist'
                        f'Check the files: \n\t{str(fix_img_path)}\n\t{str(afix_img_path)}'
                    )
                afix_img_path
                self.f_img = self.af_img
                return afix_img_path, mov_img_path
            else:
                raise Exception(f'Required fix path doesn\'t exist:\n\t{str(fix_img_path)}')
        return fix_img_path, mov_img_path

    def get_fix_and_mov_masks_filepaths(
        self, fix_path: Path, mov_path: Path
    ) -> Tuple[Path, Path]:
        bm_suffix = self.method_cfg['brain_mask_suffix']
        res_suffix = self.method_cfg['resampled_suffix']
        fix_bm_name = f'{(self.f_img).split("-")[0]}-{res_suffix}-{bm_suffix}'
        fix_mask_path = Path(str(fix_path).replace(self.f_img, fix_bm_name))
        mov_bm_name = f'{(self.m_img).split("-")[0]}-{res_suffix}-{bm_suffix}'
        mov_mask_path = Path(str(mov_path).replace(self.m_img, mov_bm_name))
        if not fix_mask_path.exists():
            raise Exception(f'Brain mask file for fix image doesn\'t exist:\n\t{fix_mask_path}')
        if not mov_mask_path.exists():
            raise Exception(f'Brain mask file for mov image doesn\'t exist:\n\t{mov_mask_path}')
        return fix_mask_path, mov_mask_path

    def correct_param_map_with_cfg(
        self, param_file_paths: List[Path], temp_path: Path,
        field_value_pairs: List[Tuple] = None
    ) -> List[Path]:
        if field_value_pairs is None:
            field_value_pairs = [[] for _ in param_file_paths]
        field_value_pairs[-1].extend([
            ('WriteResultImage', self.method_cfg['write_img']),
            ('ResultImageFormat', 'nii.gz')
        ])
        modified_param_paths = []
        for fv_pair, pm_path in zip(field_value_pairs, param_file_paths):
            temp_pmap_path = temp_path / pm_path.name
            modified_param_paths.append(temp_pmap_path)
            e_utils.modify_field_parameter_map(fv_pair, pm_path, temp_pmap_path)
        return modified_param_paths

    def correct_transf_map_for_non_bin_imgs(self):
        field_value_pairs = [
            ('ResultImageFormat', 'nii.gz'),
            ('ResultImagePixelType', "float"),
            ('FinalBSplineInterpolationOrder', '3')
        ]
        e_utils.modify_field_parameter_map(field_value_pairs, self.transf_map_path)

    def correct_transf_map_for_bin_imgs(self):
        field_value_pairs = [
            ('ResultImageFormat', 'nii.gz'),
            ('ResultImagePixelType', "int"),
            ('FinalBSplineInterpolationOrder', '0')
        ]
        e_utils.modify_field_parameter_map(field_value_pairs, self.transf_map_path)

    def project_image(self, img_name, ncct_path, mri_path):
        mri_type = 'adc' if ('adc' in str(mri_path)) else 'dwi'
        base = 'ncct' if ('tilt' not in str(ncct_path)) else 'ncct-tilt'
        if 'ncct' in img_name:
            img_path = Path(str(ncct_path).replace(base, img_name))
        elif ('dwi' in img_name) or ('adc' in img_name):
            img_path = Path(str(mri_path).replace(mri_type, img_name))
        elif 'msk' in img_name:
            img_path = Path(str(mri_path).replace(mri_type, img_name))
            if not img_path.exists():
                img_name = img_name.replace('msk', 'msk-tilt')
                img_path = Path(str(mri_path).replace(base, img_name))
        else:
            raise Exception(f'Image to propagate not supported: {img_name}')

        reg_img_path = Path(str(img_path).replace('.nii.gz', f'-{self.suffix}.nii.gz'))

        if (not reg_img_path.exists()) or self.cfg['force']:
            e_utils.transformix_wrapper(
                mov_img_path=img_path,
                result_path=reg_img_path,
                transformation_path=self.transf_map_path,
                points=False,
                verbose=self.cfg['verbose'],
                keep_just_useful_files=True
            )
        if not reg_img_path.exists():
            raise Exception(f'Image projection failed, check file: \n\t{str(reg_img_path)}')
        return reg_img_path

    def check_metric(
        self, img1_path: Path, img2_path: Path, metric_name: str, threshold: float = None
    ) -> Tuple[float, bool]:
        img1 = sitk.GetArrayFromImage(sitk.ReadImage(str(img1_path)))
        img2 = sitk.GetArrayFromImage(sitk.ReadImage(str(img2_path)))
        if metric_name == 'mi':
            value = mutual_information(img1, img2)
        elif metric_name == 'dice':
            value = dice_score(img1, img2)
        else:
            raise Exception(f'Metric "{metric_name}" not supported.')

        if threshold is not None:
            return value, value > threshold
        return value, None

    def check_mutual_info(self, ncct_path: Path, mri_path: Path) -> Tuple[Dict, bool]:
        paths = list((mri_path.parent).iterdir())
        mi_cfg = self.cfg['success_metric']['mutual_info']
        th = mi_cfg['threshold']
        img1_name = mi_cfg['mri_img']
        img1_path = [
            path for path in paths
            if ((img1_name in str(path)) and (not str(path).endswith('txt')))
        ]
        # print(img1_path)
        if len(img1_path) == 0:
            if 'adc' in img1_name:
                img1_name = img1_name.replace('adc', 'dwi')
            else:
                img1_name = img1_name.replace('dwi', 'adc')
            img1_path = [path for path in paths if img1_name in str(path)]
            assert len(img1_path) != 0

        paths = list((ncct_path.parent).iterdir())
        img2_name = mi_cfg['ncct_img']
        img2_name = img2_name.replace('ncct', 'ncct-tilt') \
            if 'tilt' in str(ncct_path) else img2_name
        img2_path = [
            path for path in paths
            if ((img2_name in str(path)) and (not str(path).endswith('txt')))
        ]
        assert len(img2_path) != 0
        mi_fix_mov, success = \
            self.check_metric(img1_path[0], img2_path[0], 'mi', th)
        return mi_fix_mov, success

    def check_bm_dice(self, ncct_path: Path, mri_path: Path) -> Tuple[Dict, bool]:
        paths = list((mri_path.parent).iterdir())
        dice_cfg = self.cfg['success_metric']['brain_mask_dice']
        th = dice_cfg['threshold']
        img1_name = dice_cfg['mri_img']
        img1_path = [
            path for path in paths
            if ((img1_name in str(path)) and (not str(path).endswith('txt')))
        ]
        if len(img1_path) == 0:
            if 'adc' in img1_name:
                img1_name = img1_name.replace('adc', 'dwi')
            else:
                img1_name = img1_name.replace('dwi', 'adc')
            img1_path = [path for path in paths if img1_name in str(path)]
            assert len(img1_path) != 0

        paths = list((ncct_path.parent).iterdir())
        img2_name = dice_cfg['ncct_img']
        img2_name = img2_name.replace('ncct', 'ncct-tilt') \
            if 'tilt' in str(ncct_path) else img2_name
        img2_path = [
            path for path in paths
            if ((img2_name in str(path)) and (not str(path).endswith('txt')))
        ]
        assert len(img2_path) != 0
        dice, success = \
            self.check_metric(img1_path[0], img2_path[0], 'dice', th)
        return dice, success

    def __call__(self, ncct_path: Path, mri_path: Path) -> Dict:
        # determine the available images and define fix and moving
        ncct_path, adc_path, dwi_path = check_available_modalities(ncct_path, mri_path)
        mri_path = dwi_path if (adc_path is None) else adc_path
        fix_path, mov_path = \
            self.check_and_define_fix_and_moving_imgs(ncct_path, mri_path)
        reg_path = Path(str(mov_path).replace('.nii.gz', f'-{self.suffix}.nii.gz'))

        # initiate variables for outputs
        metrics = {}
        new_files = []

        if self.method == 'elastix':
            pmaps_path = PREPROCESSING_PATH/'registration/parameter_maps'
            pmaps_paths = [pmaps_path/pmap for pmap in self.pmaps]
            fix_mask_path, mov_mask_path = None, None
            if self.method_cfg['use_brain_masks']:
                fix_mask_path, mov_mask_path = \
                    self.get_fix_and_mov_masks_filepaths(fix_path, mov_path)
            temp_path = fix_path.parent
            pmaps_paths = self.correct_param_map_with_cfg(pmaps_paths, temp_path)

            self.transf_map_path = None
            mov_img_name = mov_path.name.replace(''.join(mov_path.suffixes), '')
            transformation_file_name = f'TransformParameters_{mov_img_name}.txt'
            self.transf_map_path = mov_path.parent/transformation_file_name

            if (not self.transf_map_path.exists()) or self.cfg['force']:
                self.transf_map_path = e_utils.elastix_wrapper(
                    fix_img_path=fix_path,
                    mov_img_path=mov_path,
                    result_path=reg_path,
                    parameters_paths=pmaps_paths,
                    fix_mask_path=fix_mask_path,
                    mov_mask_path=mov_mask_path,
                    keep_just_useful_files=True,
                    write_image=self.method_cfg['write_img'],
                    verbose=self.cfg['verbose'],
                    transformation_file_path=self.transf_map_path
                )

            if self.method_cfg['write_img']:
                new_files.append(reg_path)

            # propagate/project gray scale images
            if self.cfg['images_to_propagate'] is not None:
                self.correct_transf_map_for_non_bin_imgs()
                for img_name in self.cfg['images_to_propagate']:
                    if ('tilt' in str(ncct_path)) and ('ncct' in img_name):
                        img_name = img_name.replace('ncct', 'ncct-tilt')
                    reg_path = self.project_image(img_name, ncct_path, mri_path)
                    new_files.append(reg_path)

            # propagate/project masks and segmentations
            if self.cfg['masks_to_propagate'] is not None:
                self.correct_transf_map_for_bin_imgs()
                for img_name in self.cfg['masks_to_propagate']:
                    if ('tilt' in str(ncct_path)) and ('ncct' in img_name):
                        img_name = img_name.replace('ncct', 'ncct-tilt')
                    reg_path = self.project_image(img_name, ncct_path, mri_path)
                    new_files.append(reg_path)

            if self.cfg['final_skull_stripping'] is not None:
                for img_name, mask_name in self.cfg['final_skull_stripping'].items():
                    if 'adc' in img_name and adc_path is None:
                        continue
                    if 'dwi' in img_name and dwi_path is None:
                        continue
                    if 'ncct' in img_name:
                        base_name = 'ncct'
                        if ('tilt' in str(ncct_path)):
                            base_name = 'ncct-tilt'
                            img_name = img_name.replace('ncct', 'ncct-tilt')
                        img_path = Path(str(ncct_path).replace(base_name, img_name))
                    else:
                        if adc_path is not None:
                            img_path = Path(str(adc_path).replace('adc', img_name))
                        else:
                            img_path = Path(str(dwi_path).replace('dwi', img_name))
                    if not img_path.exists() and 'msk' in img_path.name:
                        img_path = Path(str(img_path).replace('msk', 'msk-tilt'))

                    if 'ncct' in mask_name:
                        base_name = 'ncct'
                        if ('tilt' in str(ncct_path)):
                            base_name = 'ncct-tilt'
                            mask_name = mask_name.replace('ncct', 'ncct-tilt')
                        mask_path = Path(str(ncct_path).replace(base_name, mask_name))
                    else:
                        if adc_path is not None:
                            mask_path = Path(str(adc_path).replace('adc', mask_name))
                        else:
                            mask_path = Path(str(dwi_path).replace('dwi', mask_name))

                    if (img_path.exists() and mask_path.exists()):
                        res_path = Path(str(img_path).replace('.nii.gz', '-ss.nii.gz'))
                        bkgd_v = -1000 if 'ncct' in str(res_path) else 0
                        mask_image(img_path, mask_path, res_path, bkgd_v)
                        new_files.append(res_path)
                    else:
                        raise Exception(f'Images missing:\n\t{img_path}\n\t{mask_path}')

            # check if the registrations are "correct"
            if self.cfg['success_metric'] is not None:
                if 'mutual_info' in self.cfg['success_metric'].keys():
                    metrics['mi_fix_mov'], success = \
                        self.check_mutual_info(ncct_path, mri_path)
                    if (success is not None) and not success:
                        raise Exception('Mutual Information between images registered failed.')

                if 'brain_mask_dice' in self.cfg['success_metric'].keys():
                    metrics['bm_dice'], success = \
                        self.check_bm_dice(ncct_path, mri_path)
                    if (success is not None) and not success:
                        raise Exception('Brain masks dice failed.')

                if 'tissues_dice' in self.cfg['success_metric'].keys():
                    th = self.cfg['success_metric']['tissues_dice']
                    metrics['stseg_dice'], success = \
                        self.check_metric(fix_path, mov_path, 'dice', th)
                    if (success is not None) and not success:
                        raise Exception('Simple tissue segmentation dice failed.')

            return metrics, new_files
        else:
            raise Exception(f'Registration method {self.method} not supported.')
