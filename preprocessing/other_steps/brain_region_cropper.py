# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import numpy as np
import SimpleITK as sitk
import json

from pathlib import Path
from typing import List, Dict

from preprocessing.preprocessing_utils import check_available_modalities
from preprocessing.skull_stripping.skull_stripping_utils import fill_hull
from utils.utils import save_img_from_array_using_referece


def find_brain_bbox(bm_path: Path, margin: int = 3) -> Dict:
    bm_array = sitk.GetArrayFromImage(sitk.ReadImage(str(bm_path)))
    mask_array = np.where(bm_array > 0, 1, 0).astype('uint8')
    convex_hull_filled = np.where(fill_hull(mask_array), 1, 0).astype('uint8')
    size = mask_array.shape
    a = np.where(convex_hull_filled == 1)
    crop_init = [
        np.maximum(0, np.min(a[0])-margin),
        np.maximum(0, np.min(a[1])-margin),
        np.maximum(0, np.min(a[2])-margin)
    ]
    crop_end = [
        np.minimum(np.max(a[0])+margin, size[0]),
        np.minimum(np.max(a[1])+margin, size[1]),
        np.minimum(np.max(a[2])+margin, size[2])
    ]
    bbox = {
        'origin_x': crop_init[0],
        'origin_y': crop_init[1],
        'origin_z': crop_init[2],
        'end_x': crop_end[0],
        'end_y': crop_end[1],
        'end_z': crop_end[2],
        'shape_x': crop_end[0] - crop_init[0],
        'shape_y': crop_end[1] - crop_init[1],
        'shape_z': crop_end[2] - crop_init[2],
        'o_shape_x': size[0],
        'o_shape_y': size[1],
        'o_shape_z': size[2]
    }
    return bbox


def crop_bbox(img_path: Path, out_path: Path, bbox_info: Dict) -> None:
    ox, oy, oz = bbox_info['origin_x'], bbox_info['origin_y'], bbox_info['origin_z']
    ex, ey, ez = bbox_info['end_x'], bbox_info['end_y'], bbox_info['end_z']
    img = sitk.ReadImage(str(img_path))
    arr = sitk.GetArrayFromImage(img)
    crop_arr = arr[ox:ex, oy:ey, oz:ez]
    save_img_from_array_using_referece(crop_arr, img, out_path)
    return None


class BrainRegionCropper():
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def __call__(self, ncct_path: Path, mri_path: Path) -> List[Path]:
        new_files = []
        ncct_path, adc_path, dwi_path, msk_path = check_available_modalities(
            ncct_path, mri_path, msk=True)
        if 'ncct' in self.cfg['brain_mask']:
            if ('tilt' in ncct_path.name) and ('ncct-tilt' not in self.cfg['brain_mask']):
                self.cfg['brain_mask'] = self.cfg['brain_mask'].replace('ncct', 'ncct-tilt')
            new_name = ("_").join(ncct_path.name.split("_")[:2])
            new_name = f'{new_name}_{self.cfg["brain_mask"]}.nii.gz'
            img_path = ncct_path.parent / new_name
        else:
            raise Exception(
                f'Modality not supported in brain region cropping: {self.cfg["brain_mask"]}')

        bbox_info = find_brain_bbox(img_path, margin=self.cfg['margin'])
        for img_name in self.cfg['over_img']:
            if ('ncct' in img_name) and (ncct_path is not None):
                if ('tilt' in ncct_path.name) and ('ncct-tilt' not in img_name):
                    img_name = img_name.replace('ncct', 'ncct-tilt')
                new_name = ("_").join(ncct_path.name.split("_")[:2])
                new_name = f'{new_name}_{img_name}.nii.gz'
                img_path = ncct_path.parent / new_name
                assert img_path.exists(), \
                    f'The required image to crop does not exist:\n{str(img_path)}'
            elif ('adc' in img_name) and (adc_path is not None):
                new_name = ("_").join(adc_path.name.split("_")[:2])
                new_name = f'{new_name}_{img_name}.nii.gz'
                img_path = adc_path.parent / new_name
            elif ('dwi' in img_name) and (dwi_path is not None):
                new_name = ("_").join(dwi_path.name.split("_")[:2])
                new_name = f'{new_name}_{img_name}.nii.gz'
                img_path = dwi_path.parent / new_name
            elif ('msk' in img_name) and (msk_path is not None):
                if ('tilt' in str(ncct_path)) and ('ses-0000' in str(img_path)):
                    img_name = img_name.replace('msk', 'msk-tilt')
                new_name = ("_").join(msk_path.name.split("_")[:2])
                new_name = f'{new_name}_{img_name}.nii.gz'
                img_path = msk_path.parent / new_name
            if not img_path.exists():
                raise Exception(
                    f'The required image to crop does not exist:\n{str(img_path)}'
                )
            sfx = self.cfg["suffix"]
            out_path = img_path.parent / img_path.name.replace('.nii.gz', f'-{sfx}.nii.gz')
            if (not out_path.exists()) or self.cfg['force']:
                crop_bbox(img_path, out_path, bbox_info)
            new_files.append(out_path)
        subject = img_path.name.split('_')[0].replace('sub-', '')
        file_path = Path(self.cfg["bboxes_info_path"]) / f'{subject}.json'
        meta = {subject: {k: int(v) for k, v in bbox_info.items()}}
        with open(file_path, 'w') as jfile:
            json.dump(meta, jfile, indent=4)
        return new_files
