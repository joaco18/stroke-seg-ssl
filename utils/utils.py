# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import re
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from scipy import ndimage
from typing import List, Union, Dict
from dataset.dataset import StrokeDataset


def to_snake_case(name: str) -> str:
    """Turns a string from cammel case to snake case
    Args:
        name (str): string to convert
    Returns:
        str: snake case string
    """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


def extract_metadata(img: sitk.Image) -> dict:
    header = {
        'direction': img.GetDirection(),
        'origin': img.GetOrigin(),
        'spacing': img.GetSpacing(),
        'metadata': {}
    }
    for key in img.GetMetaDataKeys():
        header['metadata'][key] = img.GetMetaData(key)
    return header


def copy_metadata_from_reference(
    img: sitk.Image, reference: sitk.Image
) -> sitk.Image:
    """Copies metadata from one image to the other.
    Args:
        img (sitk.Image): sitk image to be modified
        reference (sitk.Image): sitk image to use as reference
    Returns:
        sitk.Image: modified image.
    """
    img.SetDirection(reference.GetDirection())
    img.SetOrigin(reference.GetOrigin())
    img.SetSpacing(reference.GetSpacing())
    for key in reference.GetMetaDataKeys():
        img.SetMetaData(key, reference.GetMetaData(key))
    return img


def save_img_from_array_using_referece(
    volume: Union[np.ndarray, List[np.ndarray]], reference: sitk.Image, filepath: Path,
    ref_3d_for_4d: bool = False
) -> None:
    """Stores the volume in nifty format using the spatial parameters coming
        from a reference image
    Args:
        volume (np.ndarray | List[np.ndarray]): Volume to store in Nifty format.
            If a list of 3d volumes is passed, then a single 4d nifty is stored.
        reference (sitk.Image): Reference image to get the spatial parameters from.
        filepath (Path): Where to save the volume.
        ref_3d_for_4d (bool, optional): Whether to use 3d metadata and store a 4d image.
            Defaults to False
    """
    meta_ready = False

    # determine if the image is a 3d or 4d image
    if (type(volume) == list) or (len(volume.shape) > 3):
        # if the image is 4d determine the right reference and store
        if type(volume[0]) == sitk.Image:
            vol_list = [vol for vol in volume]
        else:
            vol_list = [sitk.GetImageFromArray(vol) for vol in volume]
            if ref_3d_for_4d:
                vol_list = [copy_metadata_from_reference(vol, reference) for vol in vol_list]
                meta_ready = True
        joiner = sitk.JoinSeriesImageFilter()
        img = joiner.Execute(*vol_list)
    else:
        if isinstance(volume, np.ndarray):
            img = sitk.GetImageFromArray(volume)
        elif not isinstance(volume, sitk.Image):
            raise Exception(
                'Error in save_img_from_array_using_referece, '
                'passed image is neither an array or a sitk.Image'
            )

    # correct the metadata using reference image
    if not meta_ready:
        copy_metadata_from_reference(img, reference)
    # write
    sitk.WriteImage(img, str(filepath))


def save_img_using_reference_metadata(
    volume: np.ndarray, metadata: dict, filepath: Path
) -> None:
    """Stores the volume in nifty format using the spatial parameters coming
        from a reference image
    Args:
        volume (np.ndarray): Volume to store as in Nifty format
        metadata (dict): Metadata from the reference image to store the volumetric image.
        filepath (Path): Where to save the volume.
    """
    # Save image
    if (type(volume) == list) or (len(volume.shape) > 3):
        if type(volume[0]) == sitk.Image:
            vol_list = [vol for vol in volume]
        else:
            vol_list = [sitk.GetImageFromArray(vol) for vol in volume]
        joiner = sitk.JoinSeriesImageFilter()
        img = joiner.Execute(*vol_list)
    else:
        img = sitk.GetImageFromArray(volume)
    img.SetDirection(metadata['direction'])
    img.SetOrigin(metadata['origin'])
    img.SetSpacing(metadata['spacing'])
    for key, val in metadata['metadata'].items():
        img.SetMetaData(key, val)
    sitk.WriteImage(img, filepath)


def mask_img_array(
    img_array: np.ndarray, mask_array: np.ndarray, bkg_value: int = -1000
) -> np.ndarray:
    """Masks an image, setting all the values outside the mask in the
    specified background value.
    Args:
        img_array (np.ndarray): Image to mask.
        mask (np.ndarray): Mask to use.
        bkg_value (int, optional): Value to fill the background with.
            Defaults to -1000.
    Returns:
        np.ndarray: Masked image.
    """
    mask_array = np.where(mask_array > 0, 1, 0).astype('uint8')
    img_array[mask_array == 0] = bkg_value
    return img_array


def mask_image(
    img_path: Path, mask_path: Path, out_path: Path, bkg_value: int = -1000
) -> None:
    """Masks an image, setting all the values outside the mask in the
    specified background value.
    Args:
        img_path (Path): Path to the image to mask.
        mask_path (Path): Path of the mask to use.
        out_path (Path): Path where the masked iamge will be saved.
        bkg_value (int, optional): Value to fill the background with.
            Defaults to -1000.
    """
    img = sitk.ReadImage(str(img_path))
    img_array = sitk.GetArrayFromImage(img)
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    assert len(np.unique(mask_array)) == 2, \
        'Skull stripping failed. Brain mask is not binary.'
    assert (mask_array.shape == img_array.shape), \
        f'Image ({img_array.shape}) and mask ({mask_array.shape}) shapes not equal.'\
        f'\n Check: \n\t- {img_path}\n\t- {mask_path}'

    img_array = mask_img_array(img_array, mask_array, bkg_value)
    save_img_from_array_using_referece(img_array, img, out_path)


def split_4d_img_into_3d_imgs(img: sitk.Image, time_pt: int = None) -> List[sitk.Image]:
    """Given a 4D sitk image, it split it in a list of 3D sitk images.
    Args:
        img (sitk.Image): 4D sitk image to split.
        time_pt (int, optional): If an specific time point is requiered it
            can be directly selected. Defaults to None (all volumes are returned)
    Returns:
        List[sitk.Image]: List of 3D sitk images, one for each time point.
    """
    img_array = sitk.GetArrayFromImage(img)
    assert img_array.shape, 'Provided image is not 4D'

    time_pts = img_array.shape[0]

    assert time_pt < time_pts, \
        f'The 4D image has {time_pts} time points, {time_pt} out of range'

    time_pts = np.arange(time_pts) if (time_pt is None) else [time_pt]
    imgs_list = []
    for time_pt in time_pts:
        pt_array = img_array[time_pt, ...]
        pt_img = sitk.GetImageFromArray(pt_array)
        direction = np.asarray(img.GetDirection())
        direction = tuple((direction.reshape((4, 4))[:3, :3]).flatten())
        pt_img.SetDirection(direction)
        pt_img.SetOrigin(img.GetOrigin()[:3])
        pt_img.SetSpacing(img.GetSpacing()[:3])
        for key in img.GetMetaDataKeys():
            if key == 'dim[4]':
                pt_img.SetMetaData(key, '1')
            elif key == 'pixdim[4]':
                pt_img.SetMetaData(key, '0')
            elif key == 'xyzt_units':
                pt_img.SetMetaData(key, '2')
            else:
                pt_img.SetMetaData(key, img.GetMetaData(key))
        imgs_list.append(pt_img)
    return imgs_list


def intensity_clipping(
    img_path: Path, out_path: Path, v_min: int = -100, v_max: int = 400
) -> Path:
    """Clip instensities between v_min and v_max.
    Args:
        img_path (Path): path to the image to clip
        out_path (Path): path where to save the clipped image
        v_min (int, optional): Lower limit in the clipping rage. Defaults to -100.
        v_max (int, optional): Upper limit in the clipping rage. Defaults to 400.
    Returns:
        (Path): Path to the clipped image
    """
    img = sitk.ReadImage(str(img_path))
    img_array = sitk.GetArrayFromImage(img)

    img_array = np.clip(img_array, v_min, v_max)
    if out_path is None:
        out_path = Path(str(img_path).replace('.nii.gz', '-ic.nii.gz'))
    save_img_from_array_using_referece(img_array, img, str(out_path))
    return out_path


def replace_value(img_path: Path, out_path: Path, value: int, replace: int) -> Path:
    """Replace an intensity value in an image for another one.
    Args:
        img_path (Path): Path to the image to modify
        out_path (Path): Path where to save the modified image
        value (int): value to replace
        replace (int): replacement value
    Returns:
        (Path): Path to the clipped image
    """
    img = sitk.ReadImage(str(img_path))
    img_array = sitk.GetArrayFromImage(img)
    img_array = np.where(img_array == value, replace, img_array).astype(img_array.dtype)
    save_img_from_array_using_referece(img_array, img, str(out_path))
    return out_path


def remove_small_lesions_array(mask_array: np.ndarray, min_les_vol: float = 3000,
                               unit_volume: float = 1) -> np.ndarray:
    """Removes lesions with volume smaller than a given threshold.
    Args:
        mask_array (np.ndarray): Image to filter small lesions from
        min_les_vol (float, optional): Volume threshold in cubic milimiters.
            Defaults to 3000.
        unit_volume (float, optional): volume of one voxel. Defaults to 1mm3
    Returns:
        (np.ndarray): Filtered mask
    """
    # print(mask_array.shape)
    label_img, _ = ndimage.label(mask_array, structure=np.ones((3, 3, 3)))
    label, counts = np.unique(label_img, return_counts=True)
    volumes = counts * unit_volume
    labels_to_ignore = label[volumes <= min_les_vol]
    if len(labels_to_ignore) != 0:
        for i in labels_to_ignore:
            label_img[label_img == i] = 0
        mask_array = np.where(label_img != 0, 1, 0).astype('int')
    return mask_array


def remove_small_lesions(mask_path: Path, output_path: Path, min_les_vol: float = 3.5) -> None:
    """Removes lesions with volume smaller than a given threshold.
    Args:
        mask_path (Path): Path to the mask image
        output_path (Path): Path where to save the resulting image
        min_les_vol (float, optional): Volume threshold. Defaults to 3.5.
    """
    mask = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask)
    voxel_size = mask.GetSize()
    unit_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]

    label_img, _ = ndimage.label(mask_array, structure=np.ones((3, 3, 3)))
    label, counts = np.unique(label_img, return_counts=True)
    volumes = counts * unit_volume
    labels_to_ignore = label[volumes <= min_les_vol]
    if len(labels_to_ignore) != 0:
        for i in labels_to_ignore:
            label_img[label_img == i] = 0
        mask_array = np.where(label_img != 0, 1, 0).astype('int')
    save_img_from_array_using_referece(mask_array, mask, output_path)


def hardcore_fix_images(img_path: Path, ref_path: Path, out_path: Path):
    """Make to SITK images have the same metadata"""
    img = sitk.ReadImage(str(img_path))
    img_array = sitk.GetArrayFromImage(img)
    ref_img = sitk.ReadImage(str(ref_path))
    save_img_from_array_using_referece(img_array, ref_img, out_path)


def extend_image(bbox_meta: Dict, cropped_array: np.ndarray, bkgd_val: int = 0) -> np.ndarray:
    """Get an image from the original size using the cropped version, and the metadata dictionary.
    Args:
        bbox_meta (Dict): Metadata ditionary that should include the following keys:
            ['o_shape_x', 'o_shape_y', 'o_shape_z', 'origin_x', 'origin_y', 'origin_z',
             'end_x', 'end_y', 'end_z', 'shape_x', 'shape_y', 'shape_z']
        cropped_array (np.ndarray): Cropped array to extend.
        bkgd_val (int, optional): . Defaults to 0.
    Returns:
        np.ndarray: Extended/padded version of the cropped array.
    """
    crop_dims, dims, oris, ends = [], [], [], []
    osh_keys = ['o_shape_x', 'o_shape_y', 'o_shape_z']
    ori_keys = ['origin_x', 'origin_y', 'origin_z']
    end_keys = ['end_x', 'end_y', 'end_z']
    crop_keys = ['shape_x', 'shape_y', 'shape_z']
    for osh, ori, end, csh in zip(osh_keys, ori_keys, end_keys, crop_keys):
        dims.append(bbox_meta[osh])
        oris.append(bbox_meta[ori])
        ends.append(bbox_meta[end])
        crop_dims.append(bbox_meta[csh])
    assert cropped_array.shape == tuple(crop_dims), \
        f'the metadata and the crop shape don\'t match {cropped_array.shape} {tuple(crop_dims)}'
    extended = np.zeros(dims) + bkgd_val
    extended[oris[0]:ends[0], oris[1]:ends[1], oris[2]:ends[2]] = cropped_array
    return extended


def get_datasets(datapath: Path, datasets: List[str], standard: List[str],
                 pathology: List[str], ssl: bool, fold: int = 0, kwargs: Dict = {}):
    """Get train val and test datasets separately"""
    modalities = ['ncct', 'ncct-tilt']
    modalities = modalities+['msk', 'msk-tilt'] if (not ssl) else modalities
    args = {
        'datapath': datapath,
        'pathology': pathology,
        'datasets': datasets,
        'modalities': modalities,
        'partitions': ['train'],
        'standard': standard,
        'fold': fold,
        'filter_discard': True
    }
    args.update(kwargs)
    train = StrokeDataset(**args)
    args['partitions'] = ['validation']
    validation = StrokeDataset(**args)
    args['partitions'] = ['test']
    test = StrokeDataset(**args)
    return train, validation, test
