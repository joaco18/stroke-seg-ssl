# -*-coding:utf-8 -*-
'''
Code based on https://github.com/freesurfer/freesurfer
@Author  :   Freesurfer
'''


import logging
import glob
import os
import time
import numpy as np
import nibabel as nib
import tensorflow as tf

from datetime import timedelta
from itertools import combinations
from scipy.ndimage import label as scipy_label
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from scipy.interpolate import RegularGridInterpolator
from tensorflow import keras
from keras import backend as K


# ================================================================================================
#                       Neurite Utilities - See github.com/adalca/neurite
# ================================================================================================


def unet(nb_features,
         input_shape,
         nb_levels,
         conv_size,
         nb_labels,
         name='unet',
         prefix=None,
         feat_mult=1,
         pool_size=2,
         padding='same',
         dilation_rate_mult=1,
         activation='elu',
         skip_n_concatenations=0,
         use_residuals=False,
         final_pred_activation='softmax',
         nb_conv_per_level=1,
         layer_nb_feats=None,
         conv_dropout=0,
         batch_norm=None,
         input_model=None):
    """
    Unet-style keras model with an overdose of parametrization. Copied with permission
    from github.com/adalca/neurite.
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # volume size data
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # get encoding model
    enc_model = conv_enc(nb_features,
                         input_shape,
                         nb_levels,
                         conv_size,
                         name=model_name,
                         prefix=prefix,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         padding=padding,
                         dilation_rate_mult=dilation_rate_mult,
                         activation=activation,
                         use_residuals=use_residuals,
                         nb_conv_per_level=nb_conv_per_level,
                         layer_nb_feats=layer_nb_feats,
                         conv_dropout=conv_dropout,
                         batch_norm=batch_norm,
                         input_model=input_model)

    # get decoder
    # use_skip_connections=True makes it a u-net
    lnf = layer_nb_feats[(nb_levels * nb_conv_per_level):] if layer_nb_feats is not None else None
    dec_model = conv_dec(nb_features,
                         None,
                         nb_levels,
                         conv_size,
                         nb_labels,
                         name=model_name,
                         prefix=prefix,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         use_skip_connections=True,
                         skip_n_concatenations=skip_n_concatenations,
                         padding=padding,
                         dilation_rate_mult=dilation_rate_mult,
                         activation=activation,
                         use_residuals=use_residuals,
                         final_pred_activation=final_pred_activation,
                         nb_conv_per_level=nb_conv_per_level,
                         batch_norm=batch_norm,
                         layer_nb_feats=lnf,
                         conv_dropout=conv_dropout,
                         input_model=enc_model)
    final_model = dec_model

    return final_model


def conv_enc(nb_features,
             input_shape,
             nb_levels,
             conv_size,
             name=None,
             prefix=None,
             feat_mult=1,
             pool_size=2,
             dilation_rate_mult=1,
             padding='same',
             activation='elu',
             layer_nb_feats=None,
             use_residuals=False,
             nb_conv_per_level=2,
             conv_dropout=0,
             batch_norm=None,
             input_model=None):
    """
    Fully Convolutional Encoder. Copied with permission from github.com/adalca/neurite.
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # first layer: input
    name = '%s_input' % prefix
    if input_model is None:
        input_tensor = keras.layers.Input(shape=input_shape, name=name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.inputs
        last_tensor = input_model.outputs
        if isinstance(last_tensor, list):
            last_tensor = last_tensor[0]

    # volume size data
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # prepare layers
    convL = getattr(keras.layers, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation, 'data_format': 'channels_last'}
    maxpool = getattr(keras.layers, 'MaxPooling%dD' % ndims)

    # down arm:
    # add nb_levels of conv + ReLu + conv + ReLu. Pool after each of first nb_levels - 1 layers
    lfidx = 0  # level feature index
    for level in range(nb_levels):
        lvl_first_tensor = last_tensor
        nb_lvl_feats = np.round(nb_features * feat_mult ** level).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult ** level

        for conv in range(nb_conv_per_level):
            # does several conv per level, max pooling applied at the end
            if layer_nb_feats is not None:  # None or List of all the feature numbers
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1

            name = '%s_conv_downarm_%d_%d' % (prefix, level, conv)
            if conv < (nb_conv_per_level - 1) or (not use_residuals):
                last_tensor = \
                    convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(last_tensor)
            else:  # no activation
                last_tensor = \
                    convL(nb_lvl_feats, conv_size, padding=padding, name=name)(last_tensor)

            if conv_dropout > 0:
                # conv dropout along feature space only
                name = '%s_dropout_downarm_%d_%d' % (prefix, level, conv)
                noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                last_tensor = keras.layers.Dropout(
                    conv_dropout, noise_shape=noise_shape, name=name)(last_tensor)

        if use_residuals:
            convarm_layer = last_tensor

            # the "add" layer is the original input
            # However, it may not have the right number of features to be added
            nb_feats_in = lvl_first_tensor.get_shape()[-1]
            nb_feats_out = convarm_layer.get_shape()[-1]
            add_layer = lvl_first_tensor
            if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
                name = '%s_expand_down_merge_%d' % (prefix, level)
                last_tensor = convL(
                    nb_lvl_feats, conv_size, **conv_kwargs, name=name)(lvl_first_tensor)
                add_layer = last_tensor

                if conv_dropout > 0:
                    name = '%s_dropout_down_merge_%d_%d' % (prefix, level, conv)
                    noise_shape = [None, *[1] * ndims, nb_lvl_feats]

            name = '%s_res_down_merge_%d' % (prefix, level)
            last_tensor = keras.layers.add([add_layer, convarm_layer], name=name)

            name = '%s_res_down_merge_act_%d' % (prefix, level)
            last_tensor = keras.layers.Activation(activation, name=name)(last_tensor)

        if batch_norm is not None:
            name = '%s_bn_down_%d' % (prefix, level)
            last_tensor = keras.layers.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

        # max pool if we're not at the last level
        if level < (nb_levels - 1):
            name = '%s_maxpool_%d' % (prefix, level)
            last_tensor = maxpool(pool_size=pool_size, name=name, padding=padding)(last_tensor)

    # create the model and return
    model = keras.Model(inputs=input_tensor, outputs=[last_tensor], name=model_name)
    return model


def conv_dec(nb_features,
             input_shape,
             nb_levels,
             conv_size,
             nb_labels,
             name=None,
             prefix=None,
             feat_mult=1,
             pool_size=2,
             use_skip_connections=False,
             skip_n_concatenations=0,
             padding='same',
             dilation_rate_mult=1,
             activation='elu',
             use_residuals=False,
             final_pred_activation='softmax',
             nb_conv_per_level=2,
             layer_nb_feats=None,
             batch_norm=None,
             conv_dropout=0,
             input_model=None):
    """
    Fully Convolutional Decoder. Copied with permission from github.com/adalca/neurite.
    Parameters:
        ...
        use_skip_connections (bool): if true, turns an Enc-Dec to a U-Net.
            If true, input_tensor and tensors are required.
            It assumes a particular naming of layers. conv_enc...
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # if using skip connections, make sure need to use them.
    if use_skip_connections:
        assert input_model is not None, "is using skip connections, tensors dictionary is required"

    # first layer: input
    input_name = '%s_input' % prefix
    if input_model is None:
        input_tensor = keras.layers.Input(shape=input_shape, name=input_name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.input
        last_tensor = input_model.output
        input_shape = last_tensor.shape.as_list()[1:]

    # vol size info
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        if ndims > 1:
            pool_size = (pool_size,) * ndims

    # prepare layers
    convL = getattr(keras.layers, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation}
    upsample = getattr(keras.layers, 'UpSampling%dD' % ndims)

    # up arm:
    # nb_levels - 1 layers of Deconvolution3D
    #    (approx via up + conv + ReLu) + merge + conv + ReLu + conv + ReLu
    lfidx = 0
    for level in range(nb_levels - 1):
        nb_lvl_feats = np.round(nb_features * feat_mult ** (nb_levels - 2 - level)).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult ** (nb_levels - 2 - level)

        # upsample matching the max pooling layers size
        name = '%s_up_%d' % (prefix, nb_levels + level)
        last_tensor = upsample(size=pool_size, name=name)(last_tensor)
        up_tensor = last_tensor

        # merge layers combining previous layer
        if use_skip_connections & (level < (nb_levels - skip_n_concatenations - 1)):
            conv_name = \
                '%s_conv_downarm_%d_%d' % (prefix, nb_levels - 2 - level, nb_conv_per_level - 1)
            cat_tensor = input_model.get_layer(conv_name).output
            name = '%s_merge_%d' % (prefix, nb_levels + level)
            last_tensor = keras.layers.concatenate(
                [cat_tensor, last_tensor], axis=ndims + 1, name=name)

        # convolution layers
        for conv in range(nb_conv_per_level):
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1

            name = '%s_conv_uparm_%d_%d' % (prefix, nb_levels + level, conv)
            if conv < (nb_conv_per_level - 1) or (not use_residuals):
                last_tensor = convL(
                    nb_lvl_feats, conv_size, **conv_kwargs, name=name)(last_tensor)
            else:
                last_tensor = convL(
                    nb_lvl_feats, conv_size, padding=padding, name=name)(last_tensor)

            if conv_dropout > 0:
                name = '%s_dropout_uparm_%d_%d' % (prefix, level, conv)
                noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                last_tensor = keras.layers.Dropout(
                    conv_dropout, noise_shape=noise_shape, name=name)(last_tensor)

        # residual block
        if use_residuals:

            # the "add" layer is the original input
            # However, it may not have the right number of features to be added
            add_layer = up_tensor
            nb_feats_in = add_layer.get_shape()[-1]
            nb_feats_out = last_tensor.get_shape()[-1]
            if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
                name = '%s_expand_up_merge_%d' % (prefix, level)
                add_layer = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(add_layer)

                if conv_dropout > 0:
                    name = '%s_dropout_up_merge_%d_%d' % (prefix, level, conv)
                    noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                    last_tensor = keras.layers.Dropout(
                        conv_dropout, noise_shape=noise_shape, name=name)(last_tensor)

            name = '%s_res_up_merge_%d' % (prefix, level)
            last_tensor = keras.layers.add([last_tensor, add_layer], name=name)

            name = '%s_res_up_merge_act_%d' % (prefix, level)
            last_tensor = keras.layers.Activation(activation, name=name)(last_tensor)

        if batch_norm is not None:
            name = '%s_bn_up_%d' % (prefix, level)
            last_tensor = keras.layers.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

    # Compute likelyhood prediction (no activation yet)
    name = '%s_likelihood' % prefix
    last_tensor = convL(nb_labels, 1, activation=None, name=name)(last_tensor)
    like_tensor = last_tensor

    # output prediction layer
    # we use a softmax to compute P(L_x|I) where x is each location
    if final_pred_activation == 'softmax':
        name = '%s_prediction' % prefix
        pred_tensor = keras.layers.Lambda(
            lambda x: keras.activations.softmax(x, axis=ndims + 1),
            name=name
        )(last_tensor)

    # otherwise create a layer that does nothing.
    else:
        name = '%s_prediction' % prefix
        pred_tensor = keras.layers.Activation('linear', name=name)(like_tensor)

    # create the model and retun
    model = keras.Model(inputs=input_tensor, outputs=pred_tensor, name=model_name)
    return model


# ================================================================================================
#                                        Lab2Im Utilities
# ================================================================================================

# ---------------------------------------------- loading/saving functions ------------------------


def load_volume(path_volume, im_only=True, squeeze=True, dtype=None, aff_ref=None):
    """
    Load volume file.
    :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
    If npz format, 1) the variable name is assumed to be 'vol_data',
    2) the volume is associated with a identity affine matrix and blank header.
    :param im_only: (optional) if False, the function also returns the affine matrix and
        header of the volume.
    :param squeeze: (optional) whether to squeeze the volume when loading.
    :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
        The returned affine matrix is also given in this new space. Must be a numpy array of
        dimension 4x4.
    :return: the volume, with corresponding affine matrix and header if im_only is False.
    """
    assert path_volume.endswith(
        ('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

    if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(path_volume)
        if squeeze:
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
    else:  # npz
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        if 'int' in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)

    # align image to reference affine matrix
    if aff_ref is not None:
        n_dims, _ = get_dims(list(volume.shape), max_channels=10)
        volume, aff = align_volume_to_ref(
            volume, aff, aff_ref=aff_ref, return_aff=True, n_dims=n_dims)

    if im_only:
        return volume
    else:
        return volume, aff, header


def save_volume(volume, aff, header, path, res=None, dtype=None, n_dims=3):
    """
    Save a volume.
    :param volume: volume to save
    :param aff: affine matrix of the volume to save. If aff is None, the volume is saved
        with an identity affine matrix. aff can also be set to 'FS', in which case the
        volume is saved with the affine matrix of FreeSurfer outputs.
    :param header: header of the volume to save. If None, the volume is saved with a blank header.
    :param path: path where to save the volume.
    :param res: (optional) update the resolution in the header before saving the volume.
    :param dtype: (optional) numpy dtype for the saved volume.
    :param n_dims: (optional) number of dimensions, to avoid confusion in multi-channel case.
        Default is None, where
    n_dims is automatically inferred.
    """

    mkdir(os.path.dirname(path))
    if '.npz' in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == 'FS':
                aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        elif aff is None:
            aff = np.eye(4)
        nifty = nib.Nifti1Image(volume, aff, header)
        if dtype is not None:
            if 'int' in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)
            nifty.set_data_dtype(dtype)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = reformat_to_list(res, length=n_dims, dtype=None)
            nifty.header.set_zooms(res)
        nib.save(nifty, path)


def get_volume_info(path_volume, return_volume=False, aff_ref=None, max_channels=10):
    """
    Gather information about a volume: shape, affine matrix, number of dimensions and channels,
        header, and resolution.
    :param path_volume: path of the volume to get information form.
    :param return_volume: (optional) whether to return the volume along with the information.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
        All info relative to the volume is then given in this new space. Must be a numpy array
        of dimension 4x4.
    :return: volume (if return_volume is true), and corresponding info.
        If aff_ref is not None, the returned aff is the original one, i.e. the affine of the
        image before being aligned to aff_ref.
    """
    # read image
    im, aff, header = load_volume(path_volume, im_only=False)

    # understand if image is multichannel
    im_shape = list(im.shape)
    n_dims, n_channels = get_dims(im_shape, max_channels=max_channels)
    im_shape = im_shape[:n_dims]

    # get labels res
    if '.nii' in path_volume:
        data_res = np.array(header['pixdim'][1:n_dims + 1])
    elif '.mgz' in path_volume:
        data_res = np.array(header['delta'])  # mgz image
    else:
        data_res = np.array([1.0] * n_dims)

    # align to given affine matrix
    if aff_ref is not None:
        ras_axes = get_ras_axes(aff, n_dims=n_dims)
        ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
        im = align_volume_to_ref(im, aff, aff_ref=aff_ref, n_dims=n_dims)
        im_shape = np.array(im_shape)
        data_res = np.array(data_res)
        im_shape[ras_axes_ref] = im_shape[ras_axes]
        data_res[ras_axes_ref] = data_res[ras_axes]
        im_shape = im_shape.tolist()

    # return info
    if return_volume:
        return im, im_shape, aff, n_dims, n_channels, header, data_res
    else:
        return im_shape, aff, n_dims, n_channels, header, data_res


def get_list_labels(label_list=None, labels_dir=None, save_label_list=None, FS_sort=False):
    """This function reads or computes a list of all label values used in a set of label maps.
    It can also sort all labels according to FreeSurfer lut.
    :param label_list: (optional) already computed label_list.
        Can be a sequence, a 1d numpy array, or the path to
    a numpy 1d array.
    :param labels_dir: (optional) if path_label_list is None, the label list is computed by
        reading all the label maps
    in the given folder. Can also be the path to a single label map.
    :param save_label_list: (optional) path where to save the label list.
    :param FS_sort: (optional) whether to sort label values according to
        the FreeSurfer classification.
    If true, the label values will be ordered as follows: neutral labels first
        (i.e. non-sided), left-side labels,
    and right-side labels. If FS_sort is True, this function also returns the number of
        neutral labels in label_list.
    :return: the label list (numpy 1d array), and the number of neutral (i.e. non-sided)
        labels if FS_sort is True.
    If one side of the brain is not represented at all in label_list, all labels are
        considered as neutral, and
    n_neutral_labels = len(label_list).
    """

    # load label list if previously computed
    if label_list is not None:
        label_list = np.array(reformat_to_list(label_list, load_as_numpy=True, dtype='int'))

    # compute label list from all label files
    elif labels_dir is not None:
        # print('Compiling list of unique labels')
        # go through all labels files and compute unique list of labels
        labels_paths = list_images_in_folder(labels_dir)
        label_list = np.empty(0)
        loop_info = LoopInfo(len(labels_paths), 10, 'processing', print_time=True)
        for lab_idx, path in enumerate(labels_paths):
            loop_info.update(lab_idx)
            y = load_volume(path, dtype='int32')
            y_unique = np.unique(y)
            label_list = np.unique(np.concatenate((label_list, y_unique))).astype('int')

    else:
        raise Exception('either label_list, path_label_list or labels_dir should be provided')

    # sort labels in neutral/left/right according to FS labels
    n_neutral_labels = 0
    if FS_sort:
        neutral_FS_labels = [
            0, 14, 15, 16, 21, 22, 23, 24, 72, 77, 80, 85, 100, 101, 102, 103, 104, 105, 106, 107,
            108, 109, 165, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 251, 252, 253,
            254, 255, 258, 259, 260, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 502, 506,
            507, 508, 509, 511, 512, 514, 515, 516, 517, 530, 531, 532, 533, 534, 535, 536, 537
        ]
        neutral = list()
        left = list()
        right = list()
        for la in label_list:
            if la in neutral_FS_labels:
                if la not in neutral:
                    neutral.append(la)
            elif ((0 < la < 14) | (16 < la < 21) | (24 < la < 40) | (135 < la < 139) |
                  (1000 <= la <= 1035) | (la == 865) | (20100 < la < 20110)):
                if la not in left:
                    left.append(la)
            elif ((39 < la < 72) | (162 < la < 165) | (2000 <= la <= 2035) |
                  (20000 < la < 20010) | (la == 139) | (la == 866)):
                if la not in right:
                    right.append(la)
            else:
                raise Exception('label {} not in our current FS classification, '
                                'please update get_list_labels in utils.py'.format(la))
        label_list = np.concatenate([sorted(neutral), sorted(left), sorted(right)])
        if ((len(left) > 0) & (len(right) > 0)) | ((len(left) == 0) & (len(right) == 0)):
            n_neutral_labels = len(neutral)
        else:
            n_neutral_labels = len(label_list)

    # save labels if specified
    if save_label_list is not None:
        np.save(save_label_list, np.int32(label_list))

    if FS_sort:
        return np.int32(label_list), n_neutral_labels
    else:
        return np.int32(label_list), None


def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the
        path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var


# ----------------------------------------------- reformatting functions --------------------------


def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this funtion returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a
        list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformated list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    var_list = (
        int, float, np.int8, np.int16, np.int32, np.int64, np.float16,
        np.float32, np.float64, np.float128
    )
    if isinstance(var, var_list):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError(
                    'if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                    'had {1}'.format(length, var))
    else:
        raise TypeError(
            'var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise ValueError(
                "dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return var


# ----------------------------------------------- path-related functions --------------------------


def list_images_in_folder(path_dir, include_single_image=True, check_if_empty=True):
    """List all files with extension nii, nii.gz, mgz, or npz whithin a folder."""
    basename = os.path.basename(path_dir)
    if include_single_image & (
            ('.nii.gz' in basename) | ('.nii' in basename) |
            ('.mgz' in basename) | ('.npz' in basename)
    ):
        assert os.path.isfile(path_dir), 'file %s does not exist' % path_dir
        list_images = [path_dir]
    else:
        if os.path.isdir(path_dir):
            list_images = sorted(glob.glob(os.path.join(path_dir, '*nii.gz')) +
                                 glob.glob(os.path.join(path_dir, '*nii')) +
                                 glob.glob(os.path.join(path_dir, '*.mgz')) +
                                 glob.glob(os.path.join(path_dir, '*.npz')))
        else:
            raise Exception('Folder does not exist: %s' % path_dir)
        if check_if_empty:
            assert len(list_images) > 0, \
                'no .nii, .nii.gz, .mgz or .npz image could be found in %s' % path_dir
    return list_images


def mkdir(path_dir):
    """Recursively creates the current dir as well as its parent folders
    if they do not already exist."""
    if path_dir[-1] == '/':
        path_dir = path_dir[:-1]
    if not os.path.isdir(path_dir):
        list_dir_to_create = [path_dir]
        while not os.path.isdir(os.path.dirname(list_dir_to_create[-1])):
            list_dir_to_create.append(os.path.dirname(list_dir_to_create[-1]))
        for dir_to_create in reversed(list_dir_to_create):
            os.mkdir(dir_to_create)

# ---------------------------------------------- shape-related functions --------------------


def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
        The number of dimensions is assumed to be the length of the shape, as long as the shape
        of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels


def add_axis(x, axis=0):
    """Add axis to a numpy array.
    :param axis: index of the new axis to add. Can also be a list of indices to add
        several axes at the same time."""
    axis = reformat_to_list(axis)
    for ax in axis:
        x = np.expand_dims(x, axis=ax)
    return x


# --------------------------------------------------- miscellaneous --------------------


def infer(x):
    ''' Try to parse input to float. If it fails, tries boolean, and otherwise keep it as string '''
    try:
        x = float(x)
    except ValueError:
        if x == 'False':
            x = False
        elif x == 'True':
            x = True
        elif not isinstance(x, str):
            raise TypeError('input should be an int/float/boolean/str, had {}'.format(type(x)))
    return x


class LoopInfo:
    """
    Class to print the current iteration in a for loop, and optionally the estimated remaining time.
    Instantiate just before the loop, and call the update method at the start of the loop.
    The printed text has the following format:
    processing i/total    remaining time: hh:mm:ss
    """

    def __init__(self, n_iterations, spacing=10, text='processing', print_time=False):
        """
        :param n_iterations: total number of iterations of the for loop.
        :param spacing: frequency at which the update info will be printed on screen.
        :param text: text to print. Default is processing.
        :param print_time: whether to print the estimated remaining time. Default is False.
        """

        # loop parameters
        self.n_iterations = n_iterations
        self.spacing = spacing

        # text parameters
        self.text = text
        self.print_time = print_time
        self.print_previous_time = False
        self.align = len(str(self.n_iterations)) * 2 + 1 + 3

        # timing parameters
        self.iteration_durations = np.zeros((n_iterations,))
        self.start = time.time()
        self.previous = time.time()

    def update(self, idx):

        # time iteration
        now = time.time()
        self.iteration_durations[idx] = now - self.previous
        self.previous = now

        # print text
        if idx == 0:
            logging.debug(self.text + ' 1/{}'.format(self.n_iterations))
        elif idx % self.spacing == self.spacing - 1:
            iteration = str(idx + 1) + '/' + str(self.n_iterations)
            if self.print_time:
                # estimate remaining time
                max_duration = np.max(self.iteration_durations)
                average_duration = np.mean(
                    self.iteration_durations[self.iteration_durations > .01 * max_duration])
                remaining_time = int(average_duration * (self.n_iterations - idx))
                # print total remaining time only if it is greater than 1s or
                # if it was previously printed
                if (remaining_time > 1) | self.print_previous_time:
                    eta = str(timedelta(seconds=remaining_time))
                    logging.debug(self.text + ' {:<{x}} remaining time: {}'.format(
                        iteration, eta, x=self.align))
                    self.print_previous_time = True
                else:
                    logging.debug(self.text + ' {}'.format(iteration))
            else:
                pass
                logging.debug(self.text + ' {}'.format(iteration))


def get_mapping_lut(source, dest=None):
    """This functions returns the look-up table to map a list of N values (source)
        to another list (dest).
    If the second list is not given, we assume it is equal to [0, ..., N-1]."""

    # initialise
    source = np.array(reformat_to_list(source), dtype='int32')
    n_labels = source.shape[0]

    # build new label list if neccessary
    if dest is None:
        dest = np.arange(n_labels, dtype='int32')
    else:
        assert len(source) == len(dest), 'label_list and new_label_list should have the same length'
        dest = np.array(reformat_to_list(dest, dtype='int'))

    # build look-up table
    lut = np.zeros(np.max(source) + 1, dtype='int32')
    for source, dest in zip(source, dest):
        lut[source] = dest

    return lut


def find_closest_number_divisible_by_m(n, m, answer_type='lower'):
    """
    Return the closest integer to n that is divisible by m. answer_type can either be 'closer',
        'lower' (only returns values lower than n), or 'higher (only returns values higher than m).
    """
    if n % m == 0:
        return n
    else:
        q = int(n / m)
        lower = q * m
        higher = (q + 1) * m
        if answer_type == 'lower':
            return lower
        elif answer_type == 'higher':
            return higher
        elif answer_type == 'closer':
            return lower if (n - lower) < (higher - n) else higher
        else:
            raise Exception(
                'answer_type should be lower, higher, or closer, had : %s' % answer_type)


def build_binary_structure(connectivity, n_dims, shape=None):
    """Return a dilation/erosion element with provided connectivity"""
    if shape is None:
        shape = [connectivity * 2 + 1] * n_dims
    else:
        shape = reformat_to_list(shape, length=n_dims)
    dist = np.ones(shape)
    center = tuple([tuple([int(s / 2)]) for s in shape])
    dist[center] = 0
    dist = distance_transform_edt(dist)
    struct = (dist <= connectivity) * 1
    return struct


# ---------------------------------------------------- edit volume ----------------------------

def mask_volume(
    volume, mask=None, threshold=0.1, dilate=0, erode=0, fill_holes=False,
    masking_value=0, return_mask=False, return_copy=True
):
    """Mask a volume, either with a given mask, or by keeping only the values above a threshold.
    :param volume: a numpy array, possibly with several channels
    :param mask: (optional) a numpy array to mask volume with.
    Mask doesn't have to be a 0/1 array, all strictly positive values of mask are considered for
        masking volume.
    Mask should have the same size as volume. If volume has several channels, mask can either be
        uni- or multi-channel. In the first case, the same mask is applied to all channels.
    :param threshold: (optional) If mask is None, masking is performed by keeping thresholding
        the input.
    :param dilate: (optional) number of voxels by which to dilate the provided or computed mask.
    :param erode: (optional) number of voxels by which to erode the provided or computed mask.
    :param fill_holes: (optional) whether to fill the holes in the provided or computed mask.
    :param masking_value: (optional) masking value
    :param return_mask: (optional) whether to return the applied mask
    :return: the masked volume, and the applied mask if return_mask is True.
    """

    # get info
    new_volume = volume.copy() if return_copy else volume
    vol_shape = list(new_volume.shape)
    n_dims, n_channels = get_dims(vol_shape)

    # get mask and erode/dilate it
    if mask is None:
        mask = new_volume >= threshold
    else:
        assert list(mask.shape[:n_dims]) == vol_shape[:n_dims], \
            'mask should have shape {0}, or {1}, had {2}'.format(
                vol_shape[:n_dims], vol_shape[:n_dims] + [n_channels], list(mask.shape)
            )
        mask = mask > 0
    if dilate > 0:
        dilate_struct = build_binary_structure(dilate, n_dims)
        mask_to_apply = binary_dilation(mask, dilate_struct)
    else:
        mask_to_apply = mask
    if erode > 0:
        erode_struct = build_binary_structure(erode, n_dims)
        mask_to_apply = binary_erosion(mask_to_apply, erode_struct)
    if fill_holes:
        mask_to_apply = binary_fill_holes(mask_to_apply)

    # replace values outside of mask by padding_char
    if mask_to_apply.shape == new_volume.shape:
        new_volume[np.logical_not(mask_to_apply)] = masking_value
    else:
        new_volume[np.stack([np.logical_not(mask_to_apply)] * n_channels, axis=-1)] = masking_value

    if return_mask:
        return new_volume, mask_to_apply
    else:
        return new_volume


def rescale_volume(
    volume, new_min=0, new_max=255, min_percentile=2., max_percentile=98.,
    use_positive_only=False
):
    """This function linearly rescales a volume between new_min and new_max.
    :param volume: a numpy array
    :param new_min: (optional) minimum value for the rescaled image.
    :param new_max: (optional) maximum value for the rescaled image.
    :param min_percentile: (optional) percentile for estimating robust minimum of volume
        (float in [0,...100]), where 0 = np.min
    :param max_percentile: (optional) percentile for estimating robust maximum of volume
        (float in [0,...100]), where 100 = np.max
    :param use_positive_only: (optional) whether to use only positive values when estimating
        the min and max percentile.
    :return: rescaled volume
    """

    # select only positive intensities
    new_volume = volume.copy()
    intensities = new_volume[new_volume > 0] if use_positive_only else new_volume.flatten()

    # define min and max intensities in original image for normalisation
    robust_min = np.min(intensities) \
        if min_percentile == 0 else np.percentile(intensities, min_percentile)
    robust_max = np.max(intensities) \
        if max_percentile == 100 else np.percentile(intensities, max_percentile)

    # trim values outside range
    new_volume = np.clip(new_volume, robust_min, robust_max)

    # rescale image
    if robust_min != robust_max:
        return new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
    else:  # avoid dividing by zero
        return np.zeros_like(new_volume)


def crop_volume(
    volume, cropping_margin=None, cropping_shape=None, aff=None,
    return_crop_idx=False, mode='center'
):
    """Crop volume by a given margin, or to a given shape.
    :param volume: 2d or 3d numpy array (possibly with multiple channels)
    :param cropping_margin: (optional) margin by which to crop the volume.
        The cropping margin is applied on both sides. Can be an int, sequence or 1d numpy array
        of size n_dims. Should be given if cropping_shape is None.
    :param cropping_shape: (optional) shape to which the volume will be cropped.
        Can be an int, sequence or 1d numpy  array of size n_dims.
        Should be given if cropping_margin is None.
    :param aff: (optional) affine matrix of the input volume.
        If not None, this function also returns an updated version of the
         affine matrix for the cropped volume.
    :param return_crop_idx: (optional) whether to return the cropping indices
         used to crop the given volume.
    :param mode: (optional) if cropping_shape is not None, whether to extract
         the centre of the image (mode='center'),
    or to randomly crop the volume to the provided shape (mode='random'). Default is 'center'.
    :return: cropped volume, corresponding affine matrix if aff is not None,
         and cropping indices if return_crop_idx is
    True (in that order).
    """

    assert (cropping_margin is not None) | (cropping_shape is not None), \
        'cropping_margin or cropping_shape should be provided'
    assert not ((cropping_margin is not None) & (cropping_shape is not None)), \
        'only one of cropping_margin or cropping_shape should be provided'

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, _ = get_dims(vol_shape)

    # find cropping indices
    if cropping_margin is not None:
        cropping_margin = reformat_to_list(cropping_margin, length=n_dims)
        do_cropping = np.array(vol_shape[:n_dims]) > 2 * np.array(cropping_margin)
        min_crop_idx = [cropping_margin[i] if do_cropping[i] else 0 for i in range(n_dims)]
        max_crop_idx = [
            (vol_shape[i] - cropping_margin[i]
                if do_cropping[i] else vol_shape[i] for i in range(n_dims))
        ]
    else:
        cropping_shape = reformat_to_list(cropping_shape, length=n_dims)
        if mode == 'center':
            min_crop_idx = np.maximum(
                [int((vol_shape[i] - cropping_shape[i]) / 2) for i in range(n_dims)], 0)
            max_crop_idx = np.minimum(
                [min_crop_idx[i] + cropping_shape[i] for i in range(n_dims)],
                np.array(vol_shape)[:n_dims]
            )
        elif mode == 'random':
            crop_max_val = np.maximum(
                np.array([vol_shape[i] - cropping_shape[i] for i in range(n_dims)]), 0)
            min_crop_idx = np.random.randint(0, high=crop_max_val + 1)
            max_crop_idx = np.minimum(
                min_crop_idx + np.array(cropping_shape), np.array(vol_shape)[:n_dims])
        else:
            raise ValueError('mode should be either "center" or "random", had %s' % mode)
    crop_idx = np.concatenate([np.array(min_crop_idx), np.array(max_crop_idx)])

    # crop volume
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]: crop_idx[2], crop_idx[1]: crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[
            crop_idx[0]: crop_idx[3], crop_idx[1]: crop_idx[4], crop_idx[2]: crop_idx[5], ...]

    # sort outputs
    output = [new_volume]
    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ np.array(min_crop_idx)
        output.append(aff)
    if return_crop_idx:
        output.append(crop_idx)
    return output[0] if len(output) == 1 else tuple(output)


def crop_volume_with_idx(volume, crop_idx, aff=None, n_dims=None, return_copy=True):
    """Crop a volume with given indices.
    :param volume: a 2d or 3d numpy array
    :param crop_idx: croppping indices, in the order
        [lower_bound_dim_1, ..., upper_bound_dim_1, ...].
    Can be a list or a 1d numpy array.
    :param aff: (optional) if aff is specified, this function returns an updated
        affine matrix of the volume after
    cropping.
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume.
        If not provided, n_dims will be
    inferred from the input volume.
    :return: the cropped volume, and the updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy() if return_copy else volume
    n_dims = int(np.array(crop_idx).shape[0] / 2) if n_dims is None else n_dims

    # crop image
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[
            crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], ...]
    else:
        raise Exception('cannot crop volumes with more than 3 dimensions')

    if aff is not None:
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ crop_idx[:3]
        return new_volume, aff
    else:
        return new_volume


def pad_volume(volume, padding_shape, padding_value=0, aff=None, return_pad_idx=False):
    """Pad volume to a given shape
    :param volume: volume to be padded
    :param padding_shape: shape to pad volume to. Can be a number, a sequence or a 1d numpy array.
    :param padding_value: (optional) value used for padding
    :param aff: (optional) affine matrix of the volume
    :return: padded volume, and updated affine matrix if aff is not None.
    """

    # get info
    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, n_channels = get_dims(vol_shape)
    padding_shape = reformat_to_list(padding_shape, length=n_dims, dtype='int')

    # check if need to pad
    if np.any(np.array(padding_shape, dtype='int32') > np.array(vol_shape[:n_dims], dtype='int32')):

        # get padding margins
        min_margins = np.maximum(
            np.int32(np.floor((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        max_margins = np.maximum(
            np.int32(np.ceil((np.array(padding_shape) - np.array(vol_shape)[:n_dims]) / 2)), 0)
        pad_idx = np.concatenate([min_margins, min_margins + np.array(vol_shape[:n_dims])])
        pad_margins = tuple([(min_margins[i], max_margins[i]) for i in range(n_dims)])
        if n_channels > 1:
            pad_margins = tuple(list(pad_margins) + [(0, 0)])

        # pad volume
        new_volume = np.pad(new_volume, pad_margins, mode='constant', constant_values=padding_value)

        if aff is not None:
            if n_dims == 2:
                min_margins = np.append(min_margins, 0)
            aff[:-1, -1] = aff[:-1, -1] - aff[:-1, :-1] @ min_margins

    else:
        pad_idx = np.concatenate([np.array([0] * n_dims), np.array(vol_shape[:n_dims])])

    # sort outputs
    output = [new_volume]
    if aff is not None:
        output.append(aff)
    if return_pad_idx:
        output.append(pad_idx)
    return output[0] if len(output) == 1 else tuple(output)


def flip_volume(volume, axis=None, direction=None, aff=None, return_copy=True):
    """Flip volume along a specified axis.
    If unknown, this axis can be inferred from an affine matrix with
         a specified anatomical direction.
    :param volume: a numpy array
    :param axis: (optional) axis along which to flip the volume. Can either be an int or a tuple.
    :param direction: (optional) if axis is None, the volume can be flipped along
         an anatomical direction:
    'rl' (right/left), 'ap' anterior/posterior), 'si' (superior/inferior).
    :param aff: (optional) please provide an affine matrix if direction is not None
    :return: flipped volume
    """

    new_volume = volume.copy() if return_copy else volume
    assert (axis is not None) | ((aff is not None) & (direction is not None)), \
        'please provide either axis, or an affine matrix with a direction'

    # get flipping axis from aff if axis not provided
    if (axis is None) & (aff is not None):
        volume_axes = get_ras_axes(aff)
        if direction == 'rl':
            axis = volume_axes[0]
        elif direction == 'ap':
            axis = volume_axes[1]
        elif direction == 'si':
            axis = volume_axes[2]
        else:
            raise ValueError("direction should be 'rl', 'ap', or 'si', had %s" % direction)

    # flip volume
    return np.flip(new_volume, axis=axis)


def resample_volume(
        volume, aff, new_vox_size, interpolation='linear', apply_gaussian_smoothing=True,
        new_img_size: tuple = None
):
    """This function resizes the voxels of a volume to a new provided size,
        while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    if apply_gaussian_smoothing:
        volume_filt = gaussian_filter(volume, sigmas)
    else:
        volume_filt = volume

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = RegularGridInterpolator(
        (x, y, z), volume_filt, method=interpolation)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    if new_img_size is None:
        stop = start + step * np.ceil(volume_filt.shape * factor)
        xi = np.arange(start=start[0], stop=stop[0], step=step[0])
        yi = np.arange(start=start[1], stop=stop[1], step=step[1])
        zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    else:
        stop = start + step * np.asarray(new_img_size)
        xi = np.linspace(start=start[0], stop=stop[0], num=new_img_size[0])
        yi = np.linspace(start=start[1], stop=stop[1], num=new_img_size[1])
        zi = np.linspace(start=start[2], stop=stop[2], num=new_img_size[2])

    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2


def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of
        a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims,
        n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding
        to the provided affine matrix.
    :return: two numpy 1d arrays of lengtn n_dims, one with the axes corresponding
        to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i

    return img_ras_axes


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None, return_copy=True):
    """This function aligns a volume to a reference orientation (axis and direction)
        specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume.
        If not provided, n_dims will be
    inferred from the input volume.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    new_volume = volume.copy() if return_copy else volume
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = \
                ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)

    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume


# --------------------------------------------------- edit label map ------------------------------


def get_largest_connected_component(mask, structure=None):
    """Function to get the largest connected component for a given input.
    :param mask: a 2d or 3d label map of boolean type.
    :param structure: numpy array defining the connectivity.
    """
    components, n_components = scipy_label(mask, structure)
    return components == np.argmax(
        np.bincount(components.flat)[1:]) + 1 if n_components > 0 else mask.copy()


# ---------------------------------------------------- keras layers- ------------------------------

class RandomFlip(keras.layers.Layer):

    """This function flips the input tensors along the specified axes with a probability of 0.5.
    The input tensors are expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    If specified, this layer can also swap corresponding values, such that the flip tensors stay
         consistent with the native spatial orientation
         (especially when flipping in the righ/left dimension).
    :param flip_axis: integer, or list of integers specifying the dimensions along which to flip.
        The values exclude the batch dimension (e.g. 0 will flip the tensor along the first axis
         after the batch dimension). Default is None, where
    the tensors can be flipped along any of the axes (except batch and channel axes).
    :param swap_labels: list of booleans to specify wether to swap the values of each input.
        All the inputs for which the values need to be swapped must have a int32 ot int64 dtype.
    :param label_list: if swap_labels is True, list of all labels contained in labels.
        Must be ordered as follows, first the neutral labels (i.e. non-sided), then left labels
        and right labels.
    :param n_neutral_labels: if swap_labels is True, number of non-sided labels

    example 1:
    if input is a tensor of shape (batchsize, 10, 100, 200, 3)
    output = RandomFlip()(input) will randomly flip input along one of the 1st, 2nd, or 3rd axis
        (i.e. those with shape 10, 100, 200).

    example 2:
    if input is a tensor of shape (batchsize, 10, 100, 200, 3)
    output = RandomFlip(flip_axis=1)(input) will randomly flip input along the 3rd axis
        (with shape 100), i.e. the axis
    with index 1 if we don't count the batch axis.

    example 3:
    input = tf.convert_to_tensor(np.array([[1, 0, 0, 0, 0, 0, 0],
                                           [1, 0, 0, 0, 2, 2, 0],
                                           [1, 0, 0, 0, 2, 2, 0],
                                           [1, 0, 0, 0, 2, 2, 0],
                                           [1, 0, 0, 0, 0, 0, 0]]))
    label_list = np.array([0, 1, 2])
    n_neutral_labels = 1
    output = RandomFlip(flip_axis=1, swap_labels=True, label_list=label_list,
        n_neutral_labels=n_neutral_labels)(input)
    where output will either be equal to input (bear in mind the flipping
        occurs with a 0.5 probability), or:
    output = [[0, 0, 0, 0, 0, 0, 2],
              [0, 1, 1, 0, 0, 0, 2],
              [0, 1, 1, 0, 0, 0, 2],
              [0, 1, 1, 0, 0, 0, 2],
              [0, 0, 0, 0, 0, 0, 2]]
    Note that the input must have a dtype int32 or int64 for its values to be swapped,
        otherwise an error will be raised

    example 4:
    if labels is the same as in the input of example 3, and image is a float32 image,
        then we can swap consistently both
    the labels and the image with:
    labels, image = RandomFlip(flip_axis=1, swap_labels=[True, False], label_list=label_list,
                               n_neutral_labels=n_neutral_labels)([labels, image]])
    Note that the labels must have a dtype int32 or int64 to be swapped, otherwise an
        error will be raised.
    This doesn't concern the image input, as its values are not swapped.
    """

    def __init__(
        self, flip_axis=None, swap_labels=False, label_list=None,
        n_neutral_labels=None, prob=0.5, **kwargs
    ):

        # shape attributes
        self.several_inputs = True
        self.n_dims = None
        self.list_n_channels = None

        # axis along which to flip
        self.flip_axis = reformat_to_list(flip_axis)

        # wether to swap labels, and corresponding label list
        self.swap_labels = reformat_to_list(swap_labels)
        self.label_list = label_list
        self.n_neutral_labels = n_neutral_labels
        self.swap_lut = None

        self.prob = prob

        super(RandomFlip, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["flip_axis"] = self.flip_axis
        config["swap_labels"] = self.swap_labels
        config["label_list"] = self.label_list
        config["n_neutral_labels"] = self.n_neutral_labels
        config["prob"] = self.prob
        return config

    def build(self, input_shape):

        if not isinstance(input_shape, list):
            self.several_inputs = False
            inputshape = [input_shape]
        else:
            inputshape = input_shape
        self.n_dims = len(inputshape[0][1:-1])
        self.list_n_channels = [i[-1] for i in inputshape]
        self.swap_labels = reformat_to_list(self.swap_labels, length=len(inputshape))

        # create label list with swapped labels
        if any(self.swap_labels):
            assert (self.label_list is not None) & (self.n_neutral_labels is not None), \
                'please provide a label_list, and n_neutral_labels when swapping' \
                ' the values of at least one input'
            n_labels = len(self.label_list)
            if self.n_neutral_labels == n_labels:
                self.swap_labels = [False] * len(self.swap_labels)
            else:
                rl_split = np.split(
                    self.label_list,
                    [self.n_neutral_labels,
                     self.n_neutral_labels + int((n_labels-self.n_neutral_labels)/2)]
                )
                label_list_swap = np.concatenate((rl_split[0], rl_split[2], rl_split[1]))
                swap_lut = get_mapping_lut(self.label_list, label_list_swap)
                self.swap_lut = tf.convert_to_tensor(swap_lut, dtype='int32')

        self.built = True
        super(RandomFlip, self).build(input_shape)

    def call(self, inputs, **kwargs):

        # convert inputs to list, and get each input type
        if not self.several_inputs:
            inputs = [inputs]
        types = [v.dtype for v in inputs]

        # sample boolean for each element of the batch
        batchsize = tf.split(tf.shape(inputs[0]), [1, self.n_dims + 1])[0]
        rand_flip = K.less(
            tf.random.uniform(tf.concat([batchsize, tf.ones(1, dtype='int32')], axis=0), 0, 1),
            self.prob
        )

        # swap r/l labels if necessary
        swapped_inputs = list()
        for i in range(len(inputs)):
            if self.swap_labels[i]:
                swapped_inputs.append(
                    tf.map_fn(
                        self._single_swap, [inputs[i], rand_flip], fn_output_signature=types[i]
                    )
                )
            else:
                swapped_inputs.append(inputs[i])

        # flip inputs and convert them back to their original type
        inputs = tf.concat([tf.cast(v, 'float32') for v in swapped_inputs], axis=-1)
        inputs = tf.map_fn(self._single_flip, [inputs, rand_flip], fn_output_signature=tf.float32)
        inputs = tf.split(inputs, self.list_n_channels, axis=-1)

        return [tf.cast(v, t) for (t, v) in zip(types, inputs)]

    def _single_swap(self, inputs):
        return K.switch(inputs[1], tf.gather(self.swap_lut, inputs[0]), inputs[0])

    def _single_flip(self, inputs):
        if self.flip_axis is None:
            flip_axis = tf.random.uniform([1], 0, self.n_dims, dtype='int32')
        else:
            idx = tf.squeeze(tf.random.uniform([1], 0, len(self.flip_axis), dtype='int32'))
            flip_axis = tf.expand_dims(
                tf.convert_to_tensor(self.flip_axis, dtype='int32')[idx], axis=0)
        return K.switch(inputs[1], K.reverse(inputs[0], axes=flip_axis), inputs[0])


class GaussianBlur(keras.layers.Layer):
    """Applies gaussian blur to an input image."""

    def __init__(self, sigma, random_blur_range=None, use_mask=False, **kwargs):
        self.sigma = reformat_to_list(sigma)
        assert np.all(np.array(self.sigma) >= 0), 'sigma should be superior or equal to 0'
        self.use_mask = use_mask

        self.n_dims = None
        self.n_channels = None
        self.blur_range = random_blur_range
        self.stride = None
        self.separable = None
        self.kernels = None
        self.convnd = None
        super(GaussianBlur, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["sigma"] = self.sigma
        config["random_blur_range"] = self.blur_range
        config["use_mask"] = self.use_mask
        return config

    def build(self, input_shape):

        # get shapes
        if self.use_mask:
            assert len(input_shape) == 2, \
                'please provide a mask as second layer input when use_mask=True'
            self.n_dims = len(input_shape[0]) - 2
            self.n_channels = input_shape[0][-1]
        else:
            self.n_dims = len(input_shape) - 2
            self.n_channels = input_shape[-1]

        # prepare blurring kernel
        self.stride = [1]*(self.n_dims+2)
        self.sigma = reformat_to_list(self.sigma, length=self.n_dims)
        self.separable = np.linalg.norm(np.array(self.sigma)) > 5
        if self.blur_range is None:  # fixed kernels
            self.kernels = gaussian_kernel(self.sigma, separable=self.separable)
        else:
            self.kernels = None

        # prepare convolution
        self.convnd = getattr(tf.nn, 'conv%dd' % self.n_dims)

        self.built = True
        super(GaussianBlur, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if self.use_mask:
            image = inputs[0]
            mask = tf.cast(inputs[1], 'bool')
        else:
            image = inputs
            mask = None

        # redefine the kernels at each new step when blur_range is activated
        if self.blur_range is not None:
            self.kernels = gaussian_kernel(
                self.sigma, blur_range=self.blur_range, separable=self.separable)

        if self.separable:
            for k in self.kernels:
                if k is not None:
                    image = tf.concat(
                        [self.convnd(tf.expand_dims(image[..., n], -1), k, self.stride, 'SAME')
                            for n in range(self.n_channels)], -1
                    )
                    if self.use_mask:
                        maskb = tf.cast(mask, 'float32')
                        maskb = tf.concat(
                            [self.convnd(tf.expand_dims(maskb[..., n], -1), k, self.stride, 'SAME')
                                for n in range(self.n_channels)], -1
                        )
                        image = image / (maskb + keras.backend.epsilon())
                        image = tf.where(mask, image, tf.zeros_like(image))
        else:
            if any(self.sigma):
                image = tf.concat(
                    [self.convnd(
                        tf.expand_dims(image[..., n], -1), self.kernels, self.stride, 'SAME'
                    ) for n in range(self.n_channels)], -1
                )
                if self.use_mask:
                    maskb = tf.cast(mask, 'float32')
                    maskb = tf.concat(
                        [self.convnd(
                            tf.expand_dims(maskb[..., n], -1), self.kernels, self.stride, 'SAME'
                        )for n in range(self.n_channels)], -1
                    )
                    image = image / (maskb + keras.backend.epsilon())
                    image = tf.where(mask, image, tf.zeros_like(image))

        return image


class ConvertLabels(keras.layers.Layer):
    """Convert all labels in a tensor by the corresponding given set of values.
    labels_converted = ConvertLabels(source_values, dest_values)(labels).
    labels must be an int32 tensor, and labels_converted will also be int32.

    :param source_values: list of all the possible values in labels.
        Must be a list or a 1D numpy array.
    :param dest_values: list of all the target label values.
        Must be ordered the same as source values:
    labels[labels == source_values[i]] = dest_values[i].
    If None (default), dest_values is equal to [0, ..., N-1],
        where N is the total number of values in source_values,
    which enables to remap label maps to [0, ..., N-1].
    """

    def __init__(self, source_values, dest_values=None, **kwargs):
        self.source_values = source_values
        self.dest_values = dest_values
        self.lut = None
        super(ConvertLabels, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["source_values"] = self.source_values
        config["dest_values"] = self.dest_values
        return config

    def build(self, input_shape):
        self.lut = tf.convert_to_tensor(
            get_mapping_lut(self.source_values, dest=self.dest_values), dtype='int32')
        self.built = True
        super(ConvertLabels, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.gather(self.lut, tf.cast(inputs, dtype='int32'))


# ---------------------------------------------------- edit tensors -------------------------------


def gaussian_kernel(sigma, max_sigma=None, blur_range=None, separable=True):
    """Build gaussian kernels of the specified standard deviation.
        The outputs are given as tensorflow tensors.
    :param sigma: standard deviation of the tensors. Can be given as a list/numpy array
        or as tensors. In each case, sigma must have the same length as the number of
        dimensions of the volume that will be blurred with the output
        tensors (e.g. sigma must have 3 values for 3D volumes).
    :param max_sigma:
    :param blur_range:
    :param separable:
    :return:
    """
    # convert sigma into a tensor
    if not tf.is_tensor(sigma):
        sigma_tens = tf.convert_to_tensor(reformat_to_list(sigma), dtype='float32')
    else:
        assert max_sigma is not None, 'max_sigma must be provided when sigma is given as a tensor'
        sigma_tens = sigma
    shape = sigma_tens.get_shape().as_list()

    # get n_dims and batchsize
    if shape[0] is not None:
        n_dims = shape[0]
        batchsize = None
    else:
        n_dims = shape[1]
        batchsize = tf.split(tf.shape(sigma_tens), [1, -1])[0]

    # reformat max_sigma
    if max_sigma is not None:  # dynamic blurring
        max_sigma = np.array(reformat_to_list(max_sigma, length=n_dims))
    else:  # sigma is fixed
        max_sigma = np.array(reformat_to_list(sigma, length=n_dims))

    # randomise the burring std dev and/or split it between dimensions
    if blur_range is not None:
        if blur_range != 1:
            sigma_tens = sigma_tens * tf.random.uniform(
                tf.shape(sigma_tens), minval=1 / blur_range, maxval=blur_range
            )

    # get size of blurring kernels
    windowsize = np.int32(np.ceil(2.5 * max_sigma) / 2) * 2 + 1

    if separable:

        split_sigma = tf.split(sigma_tens, [1] * n_dims, axis=-1)

        kernels = list()
        comb = np.array(list(combinations(list(range(n_dims)), n_dims - 1))[::-1])
        for (i, wsize) in enumerate(windowsize):

            if wsize > 1:

                # build meshgrid and replicate it along batch dim if dynamic blurring
                locations = tf.cast(tf.range(0, wsize), 'float32') - (wsize - 1) / 2
                if batchsize is not None:
                    locations = tf.tile(
                        tf.expand_dims(locations, axis=0),
                        tf.concat(
                            [batchsize, tf.ones(tf.shape(tf.shape(locations)), dtype='int32')],
                            axis=0
                        )
                    )
                    comb[i] += 1

                # compute gaussians
                exp_term = -K.square(locations) / (2 * split_sigma[i] ** 2)
                g = tf.exp(exp_term - tf.math.log(np.sqrt(2 * np.pi) * split_sigma[i]))
                g = g / tf.reduce_sum(g)

                for axis in comb[i]:
                    g = tf.expand_dims(g, axis=axis)
                kernels.append(tf.expand_dims(tf.expand_dims(g, -1), -1))

            else:
                kernels.append(None)

    else:

        # build meshgrid
        mesh = [tf.cast(f, 'float32') for f in volshape_to_meshgrid(windowsize, indexing='ij')]
        diff = tf.stack(
            [mesh[f] - (windowsize[f] - 1) / 2 for f in range(len(windowsize))], axis=-1)

        # replicate meshgrid to batch size and reshape sigma_tens
        if batchsize is not None:
            diff = tf.tile(
                tf.expand_dims(diff, axis=0),
                tf.concat([batchsize, tf.ones(tf.shape(tf.shape(diff)), dtype='int32')], axis=0)
            )
            for i in range(n_dims):
                sigma_tens = tf.expand_dims(sigma_tens, axis=1)
        else:
            for i in range(n_dims):
                sigma_tens = tf.expand_dims(sigma_tens, axis=0)

        # compute gaussians
        sigma_is_0 = tf.equal(sigma_tens, 0)
        exp_term = -K.square(diff) / (
            2 * tf.where(sigma_is_0, tf.ones_like(sigma_tens), sigma_tens)**2)
        norms = exp_term - tf.math.log(
            tf.where(sigma_is_0, tf.ones_like(sigma_tens), np.sqrt(2 * np.pi) * sigma_tens))
        kernels = K.sum(norms, -1)
        kernels = tf.exp(kernels)
        kernels /= tf.reduce_sum(kernels)
        kernels = tf.expand_dims(tf.expand_dims(kernels, -1), -1)

    return kernels


def volshape_to_meshgrid(volshape, **kwargs):
    """
    compute Tensor meshgrid from a volume size
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return meshgrid(*linvec, **kwargs)


def meshgrid(*args, **kwargs):
    """
    meshgrid code that builds on (copies) tensorflow's meshgrid but dramatically
    """

    indexing = kwargs.pop("indexing", "xy")
    if kwargs:
        key = list(kwargs.keys())[0]
        raise TypeError("'{}' is an invalid keyword argument "
                        "for this function".format(key))

    if indexing not in ("xy", "ij"):
        raise ValueError("indexing parameter must be either 'xy' or 'ij'")

    # with ops.name_scope(name, "meshgrid", args) as name:
    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
        output.append(tf.reshape(tf.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    # Create parameters for broadcasting each tensor to the full size
    shapes = [tf.size(x) for x in args]
    sz = [x.get_shape().as_list()[0] for x in args]

    # output_dtype = tf.convert_to_tensor(args[0]).dtype.base_dtype
    if indexing == "xy" and ndim > 1:
        output[0] = tf.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = tf.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]

    # This is the part of the implementation from tf that is slow.
    # We replace it below to get a ~6x speedup (essentially using tile instead of * tf.ones())
    # mult_fact = tf.ones(shapes, output_dtype)
    # return [x * mult_fact for x in output]
    for i in range(len(output)):
        stack_sz = [*sz[:i], 1, *sz[(i + 1):]]
        if indexing == 'xy' and ndim > 1 and i < 2:
            stack_sz[0], stack_sz[1] = stack_sz[1], stack_sz[0]
        output[i] = tf.tile(output[i], tf.stack(stack_sz))
    return output
