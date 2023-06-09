# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import argparse
import logging
import pickle
import yaml
import torch
import shutil

from pathlib import Path
from typing import Dict, List, Tuple

from preprocessing.preprocessing_utils import check_available_modalities
from preprocessing.resample.resampling_utils import resolution_and_size_match
from utils.utils import mask_image
import preprocessing.skull_stripping.skull_stripping_utils as ss_utils


GPU_AVAILABLE = torch.cuda.is_available()


class SkullStripper():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

        self.bm_suffix = self.cfg['brain_mask_suffix']
        self.ss_suffix = self.cfg['skull_striped_suffix']
        self.alt_suffix = self.cfg['alternative_suffix']

        self.gpu = True if (GPU_AVAILABLE and self.cfg['gpu']) else False
        self.device = 'GPU' if self.gpu else 'CPU'

    def skull_stripping_wrapper(
        self, img_path: Path, ss_path: Path, bm_path: Path,
        modality: str, use_alt: bool = False
    ):
        method = 'alternative_method' if use_alt else 'method'
        if self.cfg[modality][method] == 'synth-strip':
            success = ss_utils.synthstrip_wrapper(
                input_img_path=img_path,
                output_img_path=ss_path,
                output_bm_path=bm_path,
                gpu=self.gpu,
                border=self.cfg[modality]['border'],
                no_csf=False
            )

            # If image doesn't fit in GPU memory, try cpu.
            if (not success):
                if self.gpu:
                    logging.warning(
                        'GPU attept to perform skull stripping with synth-strip on'
                        f'image {str(img_path)} failed, trying CPU version...'
                    )
                    success = ss_utils.synthstrip_wrapper(
                        input_img_path=img_path,
                        output_img_path=ss_path,
                        output_bm_path=bm_path,
                        gpu=False,
                        border=self.cfg[modality]['border'],
                        no_csf=False
                    )
                else:
                    logging.warning(
                        'Skull stripping with synth-strip on image '
                        f'{str(img_path)} failed.'
                    )

        elif self.cfg[modality][method] == 'hd-bet':
            success = ss_utils.hdbet_wrapper(
                input_img_path=img_path,
                output_img_path=ss_path,
                device=0 if (self.device == 'GPU') else 'cpu',
                mode=self.cfg[modality]['mode'],
                tta=self.cfg[modality]['tta'],
                post_proc=True,
                verbose=True,
                save_mask=True if (bm_path is not None) else False,
                output_bm_path=bm_path
            )
            # If image doesn't fit in GPU memory, try cpu.
            if (not success):
                if self.gpu:
                    logging.warning(
                        'GPU attept to perform skull stripping with hd-bet on'
                        f'image {str(img_path)} failed, trying CPU version...'
                    )
                    success = ss_utils.synthstrip_wrapper(
                        input_img_path=img_path, output_img_path=ss_path,
                        output_bm_path=bm_path, gpu=False,
                        border=self.cfg[modality]['border'], no_csf=False
                    )
                else:
                    logging.warning(
                        'Skull stripping with hd-bet on image '
                        f'{str(img_path)} failed.'
                    )
        else:
            raise Exception(
                f'Skull stripping method {self.cfg[modality][method]}'
                f' for modality {modality} not supported.'
            )

    def __call__(self, ncct_path: Path, mri_path: Path) -> Tuple[Dict, List[Path]]:

        # get the images paths
        ncct_path, adc_path, dwi_path = check_available_modalities(ncct_path, mri_path)
        modalities_paths, modalities_names = [], []
        # initialize empty variable for metrics
        metrics = {}
        new_files = []
        for modality, path in zip(['ncct', 'adc', 'dwi'], [ncct_path, adc_path, dwi_path]):
            if (path is not None) and (modality in self.cfg.keys()):
                over_image_suffix = self.cfg[modality]['over_image']
                over_image_suffix = f'-{over_image_suffix}' if over_image_suffix is not None else ''
                path = Path(str(path).replace('.nii.gz', f'{over_image_suffix}.nii.gz'))
                modalities_paths.append(path)
                modalities_names.append(modality)

        for img_path, modality in zip(modalities_paths, modalities_names):
            # Check if skull stripping and/or brain mask should be saved
            bm_path = None
            if self.cfg['save_brain_mask']:
                bm_path = Path(str(img_path).replace('.nii.gz', f'-{self.bm_suffix}.nii.gz'))
            ss_path = None
            if self.cfg['save_skull_stripped']:
                ss_path = Path(str(img_path).replace('.nii.gz', f'-{self.ss_suffix}.nii.gz'))

            # Avoid computing the brain masks if they are already computed
            condition1 = (bm_path is not None) and ((not bm_path.exists()) or self.cfg['force'])
            condition2 = (ss_path is not None) and ((not ss_path.exists()) or self.cfg['force'])

            # Use the config information to decide the method to perform skull stripping
            if (condition1 or condition2):
                self.skull_stripping_wrapper(img_path, ss_path, bm_path, modality)
                new_files.append(ss_path)
                new_files.append(bm_path)

            # Check if the resultion of the output images are correct
            if self.cfg['check_resolution']:
                assert resolution_and_size_match(img_path, bm_path)
                assert resolution_and_size_match(img_path, ss_path)

            # Check if the brain mask was not SUPER wrong
            try_alt_method = False
            if self.cfg['success_metric'] is not None:
                if 'convexity' in self.cfg['success_metric']:
                    score, success = ss_utils.check_convexity_criteria_over_imgs(
                        bm_path, self.cfg['convexity_threshold'])
                    metrics[f'{modality}_convexity_score'] = score
                    if not success:
                        method = self.cfg[modality]['method']
                        logging.warning(
                            f'Brain mask extraction with method {method} failed for img:\n'
                            f'{str(img_path)}\n Trying alternative method...'
                        )
                        try_alt_method = True

            # If the brain mask was super bad, try the altern. method and pick to the best
            if try_alt_method:
                alt_bm_path_2, alt_ss_path_2 = None, None
                if bm_path is not None:
                    alt_bm_path_1 = Path(
                        str(bm_path).replace('.nii.gz', f'-{self.alt_suffix}1.nii.gz'))
                    shutil.copyfile(bm_path, alt_bm_path_1)
                    alt_bm_path_2 = Path(
                        str(bm_path).replace('.nii.gz', f'-{self.alt_suffix}2.nii.gz'))
                if ss_path is not None:
                    alt_ss_path_1 = Path(
                        str(ss_path).replace('.nii.gz', f'-{self.alt_suffix}1.nii.gz'))
                    shutil.copyfile(ss_path, alt_ss_path_1)
                    alt_ss_path_2 = Path(
                        str(ss_path).replace('.nii.gz', f'-{self.alt_suffix}2.nii.gz'))

                condition1 = (alt_bm_path_2 is not None) and \
                    ((not alt_bm_path_2.exists()) or self.cfg['force'])
                condition2 = (alt_ss_path_2 is not None) and \
                    ((not alt_ss_path_2.exists()) or self.cfg['force'])

                if (condition1 or condition2):
                    self.skull_stripping_wrapper(
                        img_path, alt_ss_path_2, alt_bm_path_2, modality, use_alt=True
                    )
                # Check if the resultion of the output images are correct
                if self.cfg['check_resolution']:
                    assert resolution_and_size_match(img_path, alt_bm_path_2)
                    assert resolution_and_size_match(img_path, alt_ss_path_2)

                # Check if the brain mask was not a piece of shit
                if self.cfg['success_metric'] is not None:
                    if 'convexity' in self.cfg['success_metric']:
                        alt_score, success = ss_utils.check_convexity_criteria_over_imgs(
                            alt_bm_path_2, self.cfg['convexity_threshold'])
                        if not success:
                            method = self.cfg[modality]['method']
                            logging.warning(
                                f'Brain mask extraction with method {method} failed'
                                f' for img:\n{str(img_path)}\n best one is saved...'
                            )
                            if alt_score < score:
                                shutil.copyfile(alt_bm_path_2, bm_path)
                                shutil.copyfile(alt_ss_path_2, ss_path)
                                metrics[f'{modality}_convexity_score'] = alt_score
                new_files = new_files + [alt_bm_path_1, alt_ss_path_1, alt_bm_path_2, alt_ss_path_2]

        # Apply the brain mask to additional images if necessary
        if self.cfg['skull_strip_imgs'] is not None:
            for img_name, bm_name in self.cfg['skull_strip_imgs'].items():
                if 'ncct' in img_name:
                    name = 'ncct'
                    if 'tilt' in str(ncct_path):
                        name = 'ncct-tilt'
                        img_name = img_name.replace('ncct', 'ncct-tilt')
                        bm_name = bm_name.replace('ncct', 'ncct-tilt')
                    img_path = Path(str(ncct_path).replace(name, img_name))
                    bm_path = Path(str(ncct_path).replace(name, bm_name))
                    bkg_value = -1000
                elif 'adc' in img_name:
                    img_path = Path(str(adc_path).replace('adc', img_name))
                    bm_path = Path(str(adc_path).replace('adc', bm_name))
                    bkg_value = 0
                else:
                    img_path = Path(str(dwi_path).replace('dwi', img_name))
                    bm_path = Path(str(dwi_path).replace('dwi', bm_name))
                    bkg_value = 0
                out_path = Path(str(img_path).replace('.nii.gz', '-ss.nii.gz'))
                mask_image(img_path, bm_path, out_path, bkg_value)
                new_files.append(out_path)

        return metrics, new_files


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ncct', dest='ncct_path')
    parser.add_argument('-mri', dest='mri_path')
    parser.add_argument('-cfg', dest='cfg_path')
    parser.add_argument('-temp', dest='temp_path')
    args = parser.parse_args()
    ncct_path = None if (args.ncct_path == 'None') else Path(args.ncct_path)
    mri_path = None if (args.mri_path == 'None') else Path(args.mri_path)
    cfg_path = Path(args.cfg_path)
    with open(cfg_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # instantiate and call the skull stripper.
    skull_stripper = SkullStripper(cfg['skull_stripping'])
    metric, files = skull_stripper(ncct_path, mri_path)

    # store the outputs in a temporary file
    res = (metric, files)
    with open(args.temp_path, 'wb') as pfile:
        pickle.dump(res, pfile)


# if __name__ == '__main__':
#     # parse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-ncct', dest='ncct_sr_path')
#     parser.add_argument('-ss', dest='ss_path')
#     parser.add_argument('-bm', dest='bm_path')
#     args = parser.parse_args()

#     ss_utils.synthstrip_wrapper(
#         input_img_path=Path(args.ncct_sr_path),
#         output_img_path=Path(args.ss_path),
#         output_bm_path=Path(args.bm_path),
#         gpu=True,
#         border=0,
#         no_csf=False
#     )
