# -*-coding:utf-8 -*-
'''
Code based on https://github.com/freesurfer/freesurfer
@Author  :   Freesurfer
'''

import logging
import os
import numpy as np
import tensorflow as tf
from keras import backend
from scipy.ndimage import gaussian_filter
import preprocessing.freesurfer_utils as fs

backend.set_image_data_format('channels_last')

# ================================================================================================
#                                 Prediction and Processing Utilities
# ================================================================================================


def predict(path_images,
            path_predictions,
            path_model,
            ct_mode,
            disable_sharpening,
            disable_flipping,
            threads=1,
            cpu=False):
    '''
    Prediction pipeline.
    '''

    if path_model is None:
        raise Exception("A model file is necessary")

    # prepare input/output filepaths
    path_images, path_predictions = prepare_output_files(path_images, path_predictions)

    if threads == 1:
        logging.debug('using 1 thread')
    else:
        logging.debug('using %s threads' % threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)

    # build network
    net = build_model(path_model)

    # perform SR/synthesis
    loop_info = fs.LoopInfo(len(path_images), 10, 'predicting', True)
    for idx, (path_image, path_prediction) in enumerate(zip(path_images, path_predictions)):
        loop_info.update(idx)

        # preprocessing
        try:
            image, aff, h, pad_idx = preprocess(path_image, ct_mode)
        except Exception as e:
            logging.warning(
                '\nthe following problem occured when preprocessing image %s :' % path_image)
            logging.warning(e)
            logging.warning('resuming program execution\n')
            continue

        # prediction
        try:
            device = '/cpu:0' if cpu else '/gpu:0'
            with tf.device(device):
                if disable_flipping:
                    pred = np.clip(255 * np.squeeze(net.predict(image)), 0, 128)
                else:
                    # logging.debug('Prediction without flipping')
                    pred1 = np.squeeze(net.predict(image))
                    # logging.debug('Prediction with flipping')
                    pred2 = np.flip(np.squeeze(net.predict(np.flip(image, axis=1))), axis=0)
                pred = 0.5 * np.clip(255 * pred1, 0, 128) + 0.5 * np.clip(255 * pred2, 0, 128)
        except Exception as e:
            logging.warning(
                '\nthe following problem occured when predicting output for image %s :'
                % path_image
            )
            logging.warning(e)
            logging.warning('\nresuming program execution')
            continue

        # postprocessing
        try:
            pred = postprocess(pred, pad_idx, aff, disable_sharpening)
        except Exception as e:
            logging.warning(
                '\nthe following problem occured when postprocessing prediction for image %s :'
                % path_image
            )
            logging.warning(e)
            logging.warning('\nresuming program execution')
            continue

        # write results to disk
        try:
            fs.save_volume(pred, aff, h, path_prediction)
        except Exception as e:
            logging.warning(
                '\nthe following problem occured when saving the result for image %s :'
                % path_image
            )
            logging.warning(e)
            logging.warning('\nresuming program execution')
            continue

    # print output info
    if len(path_predictions) == 1:
        logging.debug('\nprediction  saved in: ' + path_predictions[0])
    else:
        logging.debug('\npredictions saved in: ' + os.path.dirname(path_predictions[0]))

    logging.debug(
        '\nIf you use this tool in a publication, please cite:\n\n'
        'Joint super-resolution and synthesis of 1 mm isotropic MP-RAGE volumes from clinical '
        'MRI exams with scans of different orientation, resolution and contrast'
        'JE Iglesias, B Billot, Y Balbastre, A Tabari, J Conklin, RG Gonzalez, DC Alexander,'
        'P Golland, BL Edlow, B Fischl, for the ADNI NeuroImage, 118206 (2021)\n\n'
        'SynthSR: a public AI tool to turn heterogeneous clinical brain scans into '
        'high-resolution T1-weighted images for 3D morphometry'
        'JE Iglesias, B Billot, Y Balbastre, C Magdamo, S Arnold, S Das, B Edlow, D Alexander,'
        'P Golland, B Fischl Science Advances, 9(5), eadd3607 (2023)\n\n'
        'If you use the low-field (Hyperfine) version, please cite also:\n\n'
        'Quantitative Brain Morphometry of Portable Low-Field-Strength MRI Using Super-Resolution '
        'Machine Learning JE Iglesias, R Schleicher, S Laguna, B Billot, P Schaefer, '
        'B McKaig, JN Goldstein, KN Sheth, MS Rosen, WT Kimberly Radiology, 220522 (2022)\n\n'
    )


def prepare_output_files(path_images, path_predictions):
    '''
    Prepare output files.
    '''

    # check inputs
    if path_images is None:
        raise Exception('please specify an input file/folder (--i)')
    if path_predictions is None:
        raise Exception('please specify an output file/folder (--o)')

    # convert path to absolute paths
    path_images = os.path.abspath(path_images)
    basename = os.path.basename(path_images)
    path_predictions = os.path.abspath(path_predictions)

    if basename[-4:] == '.txt':

        # input images
        if not os.path.isfile(path_images):
            raise Exception(
                'provided text file containing paths of input images does not exist' % path_images
            )
        with open(path_images, 'r') as f:
            path_images = [line.replace('\n', '') for line in f.readlines() if line != '\n']

        # predictions
        if path_predictions[-4:] != '.txt':
            raise Exception('if path_images given as text file, so must be the output predictions')
        with open(path_predictions, 'r') as f:
            path_predictions = [line.replace('\n', '') for line in f.readlines() if line != '\n']

    # path_images is a folder
    elif (
        ('.nii.gz' not in basename) & ('.nii' not in basename) &
        ('.mgz' not in basename) & ('.npz' not in basename)
    ):

        # input images
        if os.path.isfile(path_images):
            raise Exception(
                'Extension not supported for %s, only use: .nii.gz, .nii, .mgz, or .npz'
                % path_images
            )
        path_images = fs.list_images_in_folder(path_images)

        # predictions
        if path_predictions[-4:] == '.txt':
            raise Exception('path_predictions can only be given as text file when path_images is.')
        if (path_predictions[-7:] == '.nii.gz') | (path_predictions[-4:] == '.nii') | \
                (path_predictions[-4:] == '.mgz') | (path_predictions[-4:] == '.npz'):
            raise Exception(
                'Output folders cannot have extensions: .nii.gz, .nii, .mgz, or .npz, had %s'
                % path_predictions
            )
        fs.mkdir(path_predictions)
        path_predictions = [
            os.path.join(path_predictions, os.path.basename(image)).replace('.nii', '_synthsr.nii')
            for image in path_images
        ]
        path_predictions = [
            path_pred.replace('.mgz', '_synthsr.mgz') for path_pred in path_predictions]
        path_predictions = [
            path_pred.replace('.npz', '_synthsr.npz') for path_pred in path_predictions]

    # path_images is an image
    else:
        # input images
        if not os.path.isfile(path_images):
            raise Exception(
                "file does not exist: %s \nplease make sure the path and the extension are correct"
                % path_images
            )
        path_images = [path_images]

        # predictions
        if path_predictions[-4:] == '.txt':
            raise Exception('path_predictions can only be given as text file when path_images is.')
        if ('.nii.gz' not in path_predictions) & ('.nii' not in path_predictions) & \
                ('.mgz' not in path_predictions) & ('.npz' not in path_predictions):
            fs.mkdir(path_predictions)
            filename = os.path.basename(path_images[0]).replace('.nii', '_synthsr.nii')
            filename = filename.replace('.mgz', '_synthsr.mgz')
            filename = filename.replace('.npz', '_synthsr.npz')
            path_predictions = os.path.join(path_predictions, filename)
        else:
            fs.mkdir(os.path.dirname(path_predictions))
        path_predictions = [path_predictions]

    return path_images, path_predictions


def preprocess(path_image, ct_mode, n_levels=5):

    # read image and corresponding info
    im, _, aff, n_dims, n_channels, h, _ = fs.get_volume_info(path_image, True)
    if n_dims < 3:
        raise Exception('input should have 3 dimensions, had %s' % n_dims)
    elif n_dims == 4 and n_channels == 1:
        im = im[..., 0]
    elif n_dims > 3:
        raise Exception('input should have 3 dimensions, had %s' % n_dims)
    elif n_channels > 1:
        logging.warning('WARNING: detected more than 1 channel, only keeping the first channel.')
        im = im[..., 0]

    # resample and align image
    im, aff = fs.resample_volume(im, aff, [1.0, 1.0, 1.0])
    im = fs.align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=3)

    # pad image to shape divisible by 32
    padding_shape = (np.ceil(np.array(im.shape[:n_dims]) / 2**n_levels) * 2**n_levels).astype('int')
    im, pad_idx = fs.pad_volume(im, padding_shape, return_pad_idx=True)

    # normalise image
    if ct_mode:
        im = np.clip(im, 0, 80)
    im = im - np.min(im)
    im = im / np.max(im)

    # add batch and channel axes
    im = fs.add_axis(im, axis=[0, -1])

    return im, aff, h, pad_idx


def build_model(model_file):
    '''
    Builds keras unet model.
    '''
    if not os.path.isfile(model_file):
        raise Exception("The provided model path does not exist.")

    # build UNet
    net = fs.unet(
        nb_features=24,
        input_shape=[None, None, None, 1],
        nb_levels=5,
        conv_size=3,
        nb_labels=1,
        feat_mult=2,
        nb_conv_per_level=2,
        final_pred_activation='linear',
        batch_norm=-1
    )
    net.load_weights(model_file, by_name=True)

    return net


def postprocess(pred, pad_idx, aff, disable_sharpening):

    pred = fs.crop_volume_with_idx(pred, pad_idx, n_dims=3)

    # unsharp masking
    amount_usm = 1.0
    sigma_usm = 1.5
    if (sigma_usm > 0) and (amount_usm > 0) and (not disable_sharpening):
        pred = pred + (pred - gaussian_filter(pred, sigma_usm * np.ones(3))) * amount_usm

    # align prediction back to first orientation
    pred = fs.align_volume_to_ref(pred, aff=np.eye(4), aff_ref=aff, n_dims=3)

    return pred
