# -*-coding:utf-8 -*-
'''
Code based on https://github.com/freesurfer/freesurfer
@Author  :   Freesurfer
'''

import csv
import logging
import os
import sys
import traceback

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.layers as KL
from keras import backend as K

import preprocessing.freesurfer_utils as fs_utils

K.set_image_data_format('channels_last')

# ================================================================================================
#                                 Prediction and Processing Utilities
# ================================================================================================


def predict(path_images,
            path_segmentations,
            path_model_segmentation,
            labels_segmentation,
            robust,
            fast,
            v1,
            n_neutral_labels,
            names_segmentation,
            labels_denoiser,
            path_posteriors,
            path_resampled,
            path_volumes,
            do_parcellation,
            path_model_parcellation,
            labels_parcellation,
            names_parcellation,
            path_qc_scores,
            path_model_qc,
            labels_qc,
            names_qc,
            cropping,
            topology_classes,
            cpu,
            threads
            ):
    '''
    Prediction pipeline.
    '''
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)

    # prepare input/output filepaths
    outputs = prepare_output_files(path_images, path_segmentations, path_posteriors, path_resampled,
                                   path_volumes, path_qc_scores)
    path_images = outputs[0]
    path_segmentations = outputs[1]
    path_posteriors = outputs[2]
    path_resampled = outputs[3]
    path_volumes = outputs[4]
    unique_vol_file = outputs[5]
    path_qc_scores = outputs[6]
    unique_qc_file = outputs[7]

    # get label lists
    labels_segmentation, _ = fs_utils.get_list_labels(label_list=labels_segmentation)
    if (n_neutral_labels is not None) & (not fast) & (not robust):
        labels_segmentation, flip_indices, unique_idx = get_flip_indices(
            labels_segmentation, n_neutral_labels)
    else:
        labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)
        flip_indices = None

    # prepare other labels list
    names_segmentation = fs_utils.load_array_if_path(names_segmentation)[unique_idx]
    topology_classes = fs_utils.load_array_if_path(topology_classes, load_as_numpy=True)[unique_idx]
    labels_denoiser = np.unique(fs_utils.get_list_labels(labels_denoiser)[0])
    if do_parcellation:
        labels_parcellation, unique_i_parc = np.unique(
            fs_utils.get_list_labels(labels_parcellation)[0], return_index=True)
        labels_volumes = np.concatenate([labels_segmentation, labels_parcellation[1:]])
        names_parcellation = fs_utils.load_array_if_path(names_parcellation)[unique_i_parc][1:]
        names_volumes = np.concatenate([names_segmentation, names_parcellation])
    else:
        labels_volumes = labels_segmentation
        names_volumes = names_segmentation
    if not v1:
        labels_volumes = np.concatenate([labels_volumes, np.array([np.max(labels_volumes + 1)])])
        names_volumes = np.concatenate([names_volumes, np.array(['total intracranial'])])
    do_qc = True if path_qc_scores[0] is not None else False
    if do_qc:
        labels_qc = fs_utils.get_list_labels(labels_qc)[0][unique_idx]
        names_qc = fs_utils.load_array_if_path(names_qc)[unique_idx]

    # set cropping/padding
    if cropping is not None:
        cropping = fs_utils.reformat_to_list(cropping, length=3, dtype='int')
        min_pad = cropping
    else:
        min_pad = 128

    # prepare volume/QC files if necessary
    if unique_vol_file & (path_volumes[0] is not None):
        write_csv(
            path_volumes[0], None, True, labels_volumes, names_volumes, last_first=(not v1)
        )
    if unique_qc_file & do_qc:
        write_csv(path_qc_scores[0], None, True, labels_qc, names_qc)

    # build network
    net = build_model(model_file_segmentation=path_model_segmentation,
                      model_file_parcellation=path_model_parcellation,
                      model_file_qc=path_model_qc,
                      labels_segmentation=labels_segmentation,
                      labels_denoiser=labels_denoiser,
                      labels_parcellation=labels_parcellation,
                      labels_qc=labels_qc,
                      flip_indices=flip_indices,
                      robust=robust,
                      do_parcellation=do_parcellation,
                      do_qc=do_qc)

    # perform segmentation
    if len(path_images) <= 10:
        loop_info = fs_utils.LoopInfo(len(path_images), 1, 'predicting', True)
    else:
        loop_info = fs_utils.LoopInfo(len(path_images), 10, 'predicting', True)
    list_errors = list()
    for i in range(len(path_images)):
        loop_info.update(i)

        try:
            device = '/cpu:0' if cpu else '/gpu:0'
            with tf.device(device):
                # preprocessing
                image, aff, h, im_res, shape, pad_idx, crop_idx = preprocess(
                    path_image=path_images[i],
                    crop=cropping,
                    min_pad=min_pad,
                    path_resample=path_resampled[i]
                )

                # prediction
                shape_input = fs_utils.add_axis(np.array(image.shape[1:-1]))
                if do_parcellation & do_qc:
                    post_patch_segmentation, post_patch_parcellation, qc_score = \
                        net.predict([image, shape_input])
                elif do_parcellation & (not do_qc):
                    post_patch_segmentation, post_patch_parcellation = net.predict(image)
                    qc_score = None
                elif (not do_parcellation) & do_qc:
                    post_patch_segmentation, qc_score = net.predict([image, shape_input])
                    post_patch_parcellation = None
                else:
                    post_patch_segmentation = net.predict(image)
                    post_patch_parcellation = qc_score = None

                # postprocessing
                seg, posteriors, volumes = postprocess(
                    post_patch_seg=post_patch_segmentation,
                    post_patch_parc=post_patch_parcellation,
                    shape=shape,
                    pad_idx=pad_idx,
                    crop_idx=crop_idx,
                    labels_segmentation=labels_segmentation,
                    labels_parcellation=labels_parcellation,
                    aff=aff,
                    im_res=im_res,
                    fast=fast,
                    topology_classes=topology_classes,
                    v1=v1
                )

                # write predictions to disc
                fs_utils.save_volume(seg, aff, h, path_segmentations[i], dtype='int32')
                if path_posteriors[i] is not None:
                    fs_utils.save_volume(posteriors, aff, h, path_posteriors[i], dtype='float32')

                # write volumes to disc if necessary
                if path_volumes[i] is not None:
                    row = [os.path.basename(path_images[i]).replace('.nii.gz', '')]
                    row = row + [str(vol) for vol in volumes]
                    write_csv(
                        path_volumes[i], row, unique_vol_file, labels_volumes,
                        names_volumes, last_first=(not v1)
                    )

                # write QC scores to disc if necessary
                if path_qc_scores[i] is not None:
                    qc_score = np.around(np.clip(np.squeeze(qc_score)[1:], 0, 1), 4)
                    row = [os.path.basename(path_images[i]).replace('.nii.gz', '')]
                    row = row + ['%.4f' % q for q in qc_score]
                    write_csv(path_qc_scores[i], row, unique_qc_file, labels_qc, names_qc)

        except Exception as e:
            e = e
            list_errors.append(path_images[i])
            logging.debug(
                f'\nThe following problem occured with image {path_images[i]}:\n'
                f'{traceback.format_exc()}\n. Resuming program execution.\n'
            )
            continue

    # print output info
    # only one image is processed with no error
    if (len(path_segmentations) == 1) & (len(list_errors) == 0):
        logging.debug('\nsegmentation  saved in:    ' + path_segmentations[0])
        if path_posteriors[0] is not None:
            logging.debug('posteriors saved in:       ' + path_posteriors[0])
        if path_resampled[0] is not None:
            logging.debug('resampled image saved in:  ' + path_resampled[0])
        if path_volumes[0] is not None:
            logging.debug('volumes saved in:          ' + path_volumes[0])
        if path_qc_scores[0] is not None:
            logging.debug('QC scores saved in:        ' + path_qc_scores[0])
    # at least 1 image with no error
    elif (len(path_segmentations) > 1) & (len(list_errors) < len(path_segmentations)):
        # only print info if all segmentations are in the same folder,
        # and we have unique vol/QC files
        seg_paths = set([os.path.dirname(path) for path in path_segmentations])
        if len(seg_paths) <= 1:
            logging.debug('\nsegmentations saved in:    ' + os.path.dirname(path_segmentations[0]))
            if path_posteriors[0] is not None:
                logging.debug('posteriors saved in:       ' + os.path.dirname(path_posteriors[0]))
            if path_resampled[0] is not None:
                logging.debug('resampled images saved in: ' + os.path.dirname(path_resampled[0]))
            if path_volumes[0] is not None:
                logging.debug('volumes saved in:          ' + path_volumes[0])
            if path_qc_scores[0] is not None:
                logging.debug('QC scores saved in:        ' + path_qc_scores[0])

    if robust:
        logging.debug(
            '\nIf you use the new robust version of SynthSeg in a publication, please cite:'
            'Robust Segmentation of Brain MRI in the Wild with Hierarchical CNNs and no Retraining'
            'B. Billot, M. Collin, S.E. Arnold, S. Das, J.E. Iglesias'
        )
    else:
        logging.debug(
            '\nIf you use this tool in a publication, please cite: SynthSeg: domain randomisation '
            'for segmentation of brain MRI scans of any contrast and resolution B. Billot, D.N. '
            'Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias'
        )

    if len(list_errors) > 0:
        logging.debug(
            '\nERROR: some problems occured for the following inputs '
            '(see corresponding errors above):'
        )
        for path_error_image in list_errors:
            logging.debug(path_error_image)
        sys.exit(1)


def prepare_output_files(path_images, out_seg, out_posteriors, out_resampled, out_volumes, out_qc):
    '''
    Prepare output files.
    '''

    # check inputs
    if path_images is None:
        raise Exception('please specify an input file/folder (--i)')
    if out_seg is None:
        raise Exception('please specify an output file/folder (--o)')

    # convert path to absolute paths
    path_images = os.path.abspath(path_images)
    basename = os.path.basename(path_images)
    out_seg = os.path.abspath(out_seg)
    out_posteriors = os.path.abspath(out_posteriors) \
        if (out_posteriors is not None) else out_posteriors
    out_resampled = os.path.abspath(out_resampled) \
        if (out_resampled is not None) else out_resampled
    out_volumes = os.path.abspath(out_volumes) \
        if (out_volumes is not None) else out_volumes
    out_qc = os.path.abspath(out_qc) \
        if (out_qc is not None) else out_qc

    # path_images is a text file
    condition_p1 = ('.nii.gz' not in basename) & ('.nii' not in basename)
    condition_p2 = ('.mgz' not in basename) & ('.npz' not in basename)
    if basename[-4:] == '.txt':
        # input images
        if not os.path.isfile(path_images):
            raise Exception(
                'provided text file containing paths of input images does not exist' % path_images
            )
        with open(path_images, 'r') as f:
            path_images = [line.replace('\n', '') for line in f.readlines() if line != '\n']

        # define helper to deal with outputs
        def text_helper(path, name):
            if path is not None:
                if path[-4:] != '.txt':
                    raise Exception('if path_images given as text file, so must be %s' % name)
                with open(path, 'r') as ff:
                    path = [line.replace('\n', '') for line in ff.readlines() if line != '\n']
                recompute_files = [not os.path.isfile(p) for p in path]
            else:
                path = [None] * len(path_images)
                recompute_files = [False] * len(path_images)
            unique_file = False
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_seg, recompute_seg, _ = text_helper(out_seg, 'path_segmentations')
        out_posteriors, recompute_post, _ = text_helper(out_posteriors, 'path_posteriors')
        out_resampled, recompute_resampled, _ = text_helper(out_resampled, 'path_resampled')
        out_volumes, recompute_volume, unique_volume_file = text_helper(out_volumes, 'path_volume')
        out_qc, recompute_qc, unique_qc_file = text_helper(out_qc, 'path_qc_scores')

    # path_images is a folder
    elif (condition_p1 & condition_p2):
        # input images
        if os.path.isfile(path_images):
            raise Exception(
                f'Extension not supported for {path_images}, only use: .nii.gz, .nii, .mgz, or .npz'
            )
        path_images = fs_utils.list_images_in_folder(path_images)

        # define helper to deal with outputs
        def helper_dir(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                if path[-4:] == '.txt':
                    raise Exception('%s can only be given as text file when path_images is.' % name)
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        logging.debug(
                            '%s provided without csv extension. Adding csv extension.' % name
                        )
                        path += '.csv'
                    path = [path] * len(path_images)
                    recompute_files = [True] * len(path_images)
                    unique_file = True
                else:
                    condition_pA = (path[-7:] == '.nii.gz') | (path[-4:] == '.nii')
                    condition_pB = (path[-4:] == '.mgz') | (path[-4:] == '.npz')
                    if condition_pA | condition_pB:
                        raise Exception('Output FOLDER had a FILE extension' % path)
                    path = [os.path.join(path, os.path.basename(p)) for p in path_images]
                    path = [p.replace('.nii', '_%s.nii' % suffix) for p in path]
                    path = [p.replace('.mgz', '_%s.mgz' % suffix) for p in path]
                    path = [p.replace('.npz', '_%s.npz' % suffix) for p in path]
                    recompute_files = [not os.path.isfile(p) for p in path]
                fs_utils.mkdir(os.path.dirname(path[0]))
            else:
                path = [None] * len(path_images)
                recompute_files = [False] * len(path_images)
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_seg, _, _ = helper_dir(out_seg, 'path_segmentations', '', 'synthseg')
        out_posteriors, _, _ = helper_dir(
            out_posteriors, 'path_posteriors', '', 'posteriors')
        out_resampled, _, _ = helper_dir(
            out_resampled, 'path_resampled', '', 'resampled')
        out_volumes, _, unique_volume_file = helper_dir(
            out_volumes, 'path_volumes', 'csv', '')
        out_qc, _, unique_qc_file = helper_dir(out_qc, 'path_qc_scores', 'csv', '')

    # path_images is an image
    else:
        # input images
        if not os.path.isfile(path_images):
            raise Exception(
                'file does not exist: %s \n'
                'please make sure the path and the extension are correct' % path_images
            )
        path_images = [path_images]

        # define helper to deal with outputs
        def helper_im(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                if path[-4:] == '.txt':
                    raise Exception(
                        '%s can only be given as text file when path_images is.' % name)
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        # print('%s provided without csv extension. Adding csv extension.' % name)
                        path += '.csv'
                    recompute_files = [True]
                    unique_file = True
                else:
                    if (('.nii.gz' not in path) & ('.nii' not in path) &
                            ('.mgz' not in path) & ('.npz' not in path)):
                        file_name = os.path.basename(
                            path_images[0]).replace('.nii', '_%s.nii' % suffix)
                        file_name = file_name.replace('.mgz', '_%s.mgz' % suffix)
                        file_name = file_name.replace('.npz', '_%s.npz' % suffix)
                        path = os.path.join(path, file_name)
                    recompute_files = [not os.path.isfile(path)]
                fs_utils.mkdir(os.path.dirname(path))
            else:
                recompute_files = [False]
            path = [path]
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_seg, _, _ = helper_im(out_seg, 'path_segmentations', '', 'synthseg')
        out_posteriors, _, _ = helper_im(
            out_posteriors, 'path_posteriors', '', 'posteriors')
        out_resampled, _, _ = helper_im(
            out_resampled, 'path_resampled', '', 'resampled')
        out_volumes, _, unique_volume_file = helper_im(
            out_volumes, 'path_volumes', 'csv', '')
        out_qc, _, unique_qc_file = helper_im(out_qc, 'path_qc_scores', 'csv', '')

    return path_images, out_seg, out_posteriors, out_resampled, out_volumes, \
        unique_volume_file, out_qc, unique_qc_file


def preprocess(path_image, n_levels=5, crop=None, min_pad=None, path_resample=None):

    # read image and corresponding info
    im, _, aff, n_dims, n_channels, h, im_res = fs_utils.get_volume_info(path_image, True)
    if n_dims < 3:
        raise Exception('input should have 3 dimensions, had %s' % n_dims)
    elif n_dims == 4 and n_channels == 1:
        n_dims = 3
        im = im[..., 0]
    elif n_dims > 3:
        raise Exception('input should have 3 dimensions, had %s' % n_dims)
    elif n_channels > 1:
        # print('WARNING: detected more than 1 channel, only keeping the first channel.')
        im = im[..., 0]

    # resample image if necessary
    if np.any((im_res > 1.05) | (im_res < 0.95)):
        im_res = np.array([1.]*3)
        im, aff = fs_utils.resample_volume(im, aff, im_res)
        if path_resample is not None:
            fs_utils.save_volume(im, aff, h, path_resample)

    # align image
    im = fs_utils.align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)
    shape = list(im.shape[:n_dims])

    # crop image if necessary
    if crop is not None:
        crop = fs_utils.reformat_to_list(crop, length=n_dims, dtype='int')
        crop_shape = [
            fs_utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in crop]
        im, crop_idx = fs_utils.crop_volume(im, cropping_shape=crop_shape, return_crop_idx=True)
    else:
        crop_idx = None

    # normalise image
    im = fs_utils.rescale_volume(im, new_min=0, new_max=1, min_percentile=0.5, max_percentile=99.5)

    # pad image
    input_shape = im.shape[:n_dims]
    pad_shape = [
        fs_utils.find_closest_number_divisible_by_m(
            s, 2 ** n_levels, 'higher') for s in input_shape
    ]
    min_pad = fs_utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
    min_pad = [
        fs_utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in min_pad]
    pad_shape = np.maximum(pad_shape, min_pad)
    im, pad_idx = fs_utils.pad_volume(im, padding_shape=pad_shape, return_pad_idx=True)

    # add batch and channel axes
    im = fs_utils.add_axis(im, axis=[0, -1])

    return im, aff, h, im_res, shape, pad_idx, crop_idx


def build_model(model_file_segmentation,
                model_file_parcellation,
                model_file_qc,
                labels_segmentation,
                labels_denoiser,
                labels_parcellation,
                labels_qc,
                flip_indices,
                robust,
                do_parcellation,
                do_qc):

    if not os.path.isfile(model_file_segmentation):
        raise Exception("The provided model path does not exist.")

    # get labels
    n_labels_seg = len(labels_segmentation)

    if robust:
        n_groups = len(labels_denoiser)

        # build first UNet
        net = fs_utils.unet(
            nb_features=24,
            input_shape=[None, None, None, 1],
            nb_levels=5,
            conv_size=3,
            nb_labels=n_groups,
            feat_mult=2,
            activation='elu',
            nb_conv_per_level=2,
            batch_norm=-1,
            name='unet'
        )

        # transition between the two networks: one_hot -> argmax -> one_hot
        #  (it simulates how the network was trained)
        last_tensor = net.output
        last_tensor = KL.Lambda(lambda x: tf.argmax(x, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(
            lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=n_groups, axis=-1))(last_tensor)
        net = keras.Model(inputs=net.inputs, outputs=last_tensor)

        # build denoiser
        net = fs_utils.unet(
            nb_features=16,
            input_shape=[None, None, None, 1],
            nb_levels=5,
            conv_size=5,
            nb_labels=n_groups,
            feat_mult=2,
            nb_conv_per_level=2,
            batch_norm=-1,
            activation='elu',
            skip_n_concatenations=2,
            input_model=net,
            name='l2l'
        )

        # transition between the two networks: one_hot -> argmax -> one_hot,
        # and concatenate input image and labels
        input_image = net.inputs[0]
        last_tensor = net.output
        last_tensor = KL.Lambda(lambda x: tf.argmax(x, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(
            lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=n_groups, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(
            lambda x: tf.cast(tf.concat(x, axis=-1), 'float32'))([input_image, last_tensor])
        net = keras.Model(inputs=net.inputs, outputs=last_tensor)

        # build 2nd network
        net = fs_utils.unet(
            nb_features=24,
            input_shape=[None, None, None, 2],
            nb_levels=5,
            conv_size=3,
            nb_labels=n_labels_seg,
            feat_mult=2,
            activation='elu',
            nb_conv_per_level=2,
            batch_norm=-1,
            input_model=net,
            name='unet2'
        )
        net.load_weights(model_file_segmentation, by_name=True)
        name_segm_prediction_layer = 'unet2_prediction'

    else:

        # build UNet
        net = fs_utils.unet(
            nb_features=24,
            input_shape=[None, None, None, 1],
            nb_levels=5,
            conv_size=3,
            nb_labels=n_labels_seg,
            feat_mult=2,
            activation='elu',
            nb_conv_per_level=2,
            batch_norm=-1,
            name='unet'
        )
        net.load_weights(model_file_segmentation, by_name=True)
        input_image = net.inputs[0]
        name_segm_prediction_layer = 'unet_prediction'

        # smooth posteriors
        last_tensor = net.output
        last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
        last_tensor = fs_utils.GaussianBlur(sigma=0.5)(last_tensor)
        net = keras.Model(inputs=net.inputs, outputs=last_tensor)

        if flip_indices is not None:

            # segment flipped image
            seg = net.output
            image_flipped = fs_utils.RandomFlip(flip_axis=0, prob=1)(input_image)
            last_tensor = net(image_flipped)

            # flip back and re-order channels
            last_tensor = fs_utils.RandomFlip(flip_axis=0, prob=1)(last_tensor)
            last_tensor = KL.Lambda(
                lambda x: tf.split(x, [1] * n_labels_seg, axis=-1), name='split')(last_tensor)
            reordered_channels = [last_tensor[flip_indices[i]] for i in range(n_labels_seg)]
            # we need the [0, ...] below because in this vresion of TF,
            #  tf.split adds a dim at the beginning...
            last_tensor = KL.Lambda(
                lambda x: tf.concat(x, -1)[0, ...], name='concat')(reordered_channels)

            # average two segmentations and build model
            name_segm_prediction_layer = 'average_lr'
            last_tensor = KL.Lambda(
                lambda x: 0.5 * (x[0] + x[1]), name=name_segm_prediction_layer)([seg, last_tensor])
            net = keras.Model(inputs=net.inputs, outputs=last_tensor)

    # add aparc segmenter if needed
    if do_parcellation:
        n_labels_parcellation = len(labels_parcellation)

        last_tensor = net.output
        last_tensor = KL.Lambda(lambda x: tf.cast(tf.argmax(x, axis=-1), 'int32'))(last_tensor)
        last_tensor = fs_utils.ConvertLabels(
            np.arange(n_labels_seg), labels_segmentation)(last_tensor)
        parcellation_masking_values = np.array(
            [1 if ((ll == 3) | (ll == 42)) else 0 for ll in labels_segmentation])
        last_tensor = fs_utils.ConvertLabels(
            labels_segmentation, parcellation_masking_values)(last_tensor)
        last_tensor = KL.Lambda(
            lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=2, axis=-1))(last_tensor)
        last_tensor = KL.Lambda(
            lambda x: tf.cast(tf.concat(x, axis=-1), 'float32'))([input_image, last_tensor])
        net = keras.Model(inputs=net.inputs, outputs=last_tensor)

        # build UNet
        net = fs_utils.unet(
            nb_features=24,
            input_shape=[None, None, None, 3],
            nb_levels=5,
            conv_size=3,
            nb_labels=n_labels_parcellation,
            feat_mult=2,
            activation='elu',
            nb_conv_per_level=2,
            batch_norm=-1,
            name='unet_parc',
            input_model=net
        )
        net.load_weights(model_file_parcellation, by_name=True)

        # smooth predictions
        last_tensor = net.output
        last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
        last_tensor = fs_utils.GaussianBlur(sigma=0.5)(last_tensor)
        net = keras.Model(
            inputs=net.inputs,
            outputs=[net.get_layer(name_segm_prediction_layer).output, last_tensor]
        )

    # add CNN regressor for automated QC if needed
    if do_qc:
        n_labels_qc = len(np.unique(labels_qc))

        # transition between the two networks: one_hot -> argmax -> qc_labels -> one_hot
        shape_prediction = KL.Input([3], dtype='int32')
        if do_parcellation:
            last_tensor = KL.Lambda(
                lambda x: tf.cast(tf.argmax(x[0], axis=-1), 'int32'))(net.outputs)
        else:
            last_tensor = KL.Lambda(lambda x: tf.cast(tf.argmax(x, axis=-1), 'int32'))(net.output)
        last_tensor = MakeShape(224)([last_tensor, shape_prediction])
        last_tensor = fs_utils.ConvertLabels(
            np.arange(n_labels_seg), labels_segmentation)(last_tensor)
        last_tensor = fs_utils.ConvertLabels(labels_segmentation, labels_qc)(last_tensor)
        last_tensor = KL.Lambda(
            lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=n_labels_qc, axis=-1))(last_tensor)
        net = keras.Model(inputs=[*net.inputs, shape_prediction], outputs=last_tensor)

        # build QC regressor network
        net = fs_utils.conv_enc(
            nb_features=24,
            input_shape=[None, None, None, n_labels_qc],
            nb_levels=4,
            conv_size=5,
            name='qc',
            feat_mult=2,
            activation='relu',
            use_residuals=True,
            nb_conv_per_level=2,
            batch_norm=-1,
            input_model=net
        )
        last_tensor = net.outputs[0]
        conv_kwargs = {'padding': 'same', 'activation': 'relu', 'data_format': 'channels_last'}
        last_tensor = KL.MaxPool3D(
            pool_size=(2, 2, 2), name='qc_maxpool_3', padding='same')(last_tensor)
        last_tensor = KL.Conv3D(
            n_labels_qc, kernel_size=5, **conv_kwargs, name='qc_final_conv_0')(last_tensor)
        last_tensor = KL.Conv3D(
            n_labels_qc, kernel_size=5, **conv_kwargs, name='qc_final_conv_1')(last_tensor)
        last_tensor = KL.Lambda(
            lambda x: tf.reduce_mean(x, axis=[1, 2, 3]), name='qc_final_pred')(last_tensor)

        # build model
        if do_parcellation:
            outputs = [net.get_layer(name_segm_prediction_layer).output,
                       net.get_layer('unet_parc_prediction').output,
                       last_tensor]
        else:
            outputs = [net.get_layer(name_segm_prediction_layer).output, last_tensor]
        net = keras.Model(inputs=net.inputs, outputs=outputs)
        net.load_weights(model_file_qc, by_name=True)

    return net


def postprocess(post_patch_seg, post_patch_parc, shape, pad_idx, crop_idx,
                labels_segmentation, labels_parcellation, aff, im_res, fast, topology_classes, v1):

    # get posteriors
    post_patch_seg = np.squeeze(post_patch_seg)
    if fast:
        post_patch_seg = fs_utils.crop_volume_with_idx(
            post_patch_seg, pad_idx, n_dims=3, return_copy=False)

    # keep biggest connected component
    tmp_post_patch_seg = post_patch_seg[..., 1:]
    post_patch_seg_mask = np.sum(tmp_post_patch_seg, axis=-1) > 0.25
    post_patch_seg_mask = fs_utils.get_largest_connected_component(post_patch_seg_mask)
    post_patch_seg_mask = np.stack([post_patch_seg_mask]*tmp_post_patch_seg.shape[-1], axis=-1)
    tmp_post_patch_seg = fs_utils.mask_volume(
        tmp_post_patch_seg, mask=post_patch_seg_mask, return_copy=False)
    post_patch_seg[..., 1:] = tmp_post_patch_seg

    # reset posteriors to zero outside the largest connected component of each topological class
    if not fast:
        post_patch_seg_mask = post_patch_seg > 0.25
        for topology_class in np.unique(topology_classes)[1:]:
            tmp_topology_indices = np.where(topology_classes == topology_class)[0]
            tmp_mask = np.any(post_patch_seg_mask[..., tmp_topology_indices], axis=-1)
            tmp_mask = fs_utils.get_largest_connected_component(tmp_mask)
            for idx in tmp_topology_indices:
                post_patch_seg[..., idx] *= tmp_mask
        post_patch_seg = fs_utils.crop_volume_with_idx(
            post_patch_seg, pad_idx, n_dims=3, return_copy=False)
    else:
        post_patch_seg_mask = post_patch_seg > 0.2
        post_patch_seg[..., 1:] *= post_patch_seg_mask[..., 1:]

    # get hard segmentation
    post_patch_seg /= np.sum(post_patch_seg, axis=-1)[..., np.newaxis]
    seg_patch = labels_segmentation[post_patch_seg.argmax(-1).astype('int32')].astype('int32')

    # postprocess parcellation
    if post_patch_parc is not None:
        post_patch_parc = np.squeeze(post_patch_parc)
        post_patch_parc = fs_utils.crop_volume_with_idx(
            post_patch_parc, pad_idx, n_dims=3, return_copy=False)
        mask = (seg_patch == 3) | (seg_patch == 42)
        post_patch_parc[..., 0] = np.ones_like(post_patch_parc[..., 0])
        post_patch_parc[..., 0] = fs_utils.mask_volume(
            post_patch_parc[..., 0], mask=mask < 0.1, return_copy=False)
        post_patch_parc /= np.sum(post_patch_parc, axis=-1)[..., np.newaxis]
        parc_patch = labels_parcellation[post_patch_parc.argmax(-1).astype('int32')].astype('int32')
        seg_patch[mask] = parc_patch[mask]

    # paste patches back to matrix of original image size
    if crop_idx is not None:
        # we need to go through this because of the posteriors of the background,
        #  otherwise pad_volume would work
        seg = np.zeros(shape=shape, dtype='int32')
        posteriors = np.zeros(shape=[*shape, labels_segmentation.shape[0]])
        posteriors[..., 0] = np.ones(shape)  # place background around patch
        seg[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = seg_patch
        posteriors[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], :] = \
            post_patch_seg
    else:
        seg = seg_patch
        posteriors = post_patch_seg

    # align prediction back to first orientation
    seg = fs_utils.align_volume_to_ref(
        seg, aff=np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)
    posteriors = fs_utils.align_volume_to_ref(
        posteriors, np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)

    # compute volumes
    volumes = np.sum(posteriors[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1)))
    if not v1:
        volumes = np.concatenate([np.array([np.sum(volumes)]), volumes])
    if post_patch_parc is not None:
        volumes_parc = np.sum(
            post_patch_parc[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1)))
        total_volume_cortex = np.sum(
            volumes[np.where((labels_segmentation == 3) | (labels_segmentation == 42))[0] - 1])
        volumes_parc = volumes_parc / np.sum(volumes_parc) * total_volume_cortex
        volumes = np.concatenate([volumes, volumes_parc])
    volumes = np.around(volumes * np.prod(im_res), 3)

    return seg, posteriors, volumes


class MakeShape(KL.Layer):
    """Expects one-hot encoding of the two input label maps."""

    def __init__(self, target_shape, **kwargs):
        self.n_dims = None
        self.target_shape = target_shape
        self.cropping_shape = None
        super(MakeShape, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["target_shape"] = self.target_shape
        return config

    def build(self, input_shape):
        self.n_dims = input_shape[1][1]
        self.cropping_shape = np.array(
            fs_utils.reformat_to_list(self.target_shape, length=self.n_dims))
        self.built = True
        super(MakeShape, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.map_fn(self._single_process, inputs, fn_output_signature=tf.int32)

    def _single_process(self, inputs):

        x = inputs[0]
        shape = inputs[1]

        # find cropping indices
        mask = tf.logical_and(tf.not_equal(x, 0), tf.not_equal(x, 24))
        indices = tf.cast(tf.where(mask), 'int32')

        min_idx = K.switch(tf.equal(tf.shape(indices)[0], 0),
                           tf.zeros(self.n_dims, dtype='int32'),
                           tf.maximum(tf.reduce_min(indices, axis=0), 0))
        max_idx = K.switch(tf.equal(tf.shape(indices)[0], 0),
                           tf.minimum(shape, self.cropping_shape),
                           tf.minimum(tf.reduce_max(indices, axis=0) + 1, shape))

        # expand/retract (depending on the desired shape) the cropping region around the centre
        intermediate_vol_shape = max_idx - min_idx
        min_idx = min_idx - tf.cast(
            tf.math.ceil((self.cropping_shape - intermediate_vol_shape) / 2), 'int32')
        max_idx = max_idx + tf.cast(
            tf.math.floor((self.cropping_shape - intermediate_vol_shape) / 2), 'int32')
        tmp_min_idx = tf.maximum(min_idx, 0)
        tmp_max_idx = tf.minimum(max_idx, shape)
        x = tf.slice(x, begin=tmp_min_idx, size=tf.minimum(tmp_max_idx - tmp_min_idx, shape))

        # pad if necessary
        min_padding = tf.abs(tf.minimum(min_idx, 0))
        max_padding = tf.maximum(max_idx - shape, 0)
        x = K.switch(
            tf.reduce_any(tf.logical_or(tf.greater(min_padding, 0), tf.greater(max_padding, 0))),
            tf.pad(x, tf.stack([min_padding, max_padding], axis=1)), x
        )

        return x


def get_flip_indices(labels_segmentation, n_neutral_labels):

    # get position labels
    n_sided_labels = int((len(labels_segmentation) - n_neutral_labels) / 2)
    neutral_labels = labels_segmentation[:n_neutral_labels]
    left = labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels]

    # get correspondance between labels
    lr_corresp = np.stack([labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels],
                           labels_segmentation[n_neutral_labels + n_sided_labels:]])
    lr_corresp_unique, lr_corresp_indices = np.unique(lr_corresp[0, :], return_index=True)
    lr_corresp_unique = np.stack([lr_corresp_unique, lr_corresp[1, lr_corresp_indices]])
    lr_corresp_unique = lr_corresp_unique[:, 1:] \
        if not np.all(lr_corresp_unique[:, 0]) else lr_corresp_unique

    # get unique labels
    labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)

    # get indices of corresponding labels
    lr_indices = np.zeros_like(lr_corresp_unique)
    for i in range(lr_corresp_unique.shape[0]):
        for j, lab in enumerate(lr_corresp_unique[i]):
            lr_indices[i, j] = np.where(labels_segmentation == lab)[0]

    # build 1d vector to swap LR corresponding labels taking into account neutral labels
    flip_indices = np.zeros_like(labels_segmentation)
    for i in range(len(flip_indices)):
        if labels_segmentation[i] in neutral_labels:
            flip_indices[i] = i
        elif labels_segmentation[i] in left:
            flip_indices[i] = \
                lr_indices[1, np.where(lr_corresp_unique[0, :] == labels_segmentation[i])]
        else:
            flip_indices[i] = \
                lr_indices[0, np.where(lr_corresp_unique[1, :] == labels_segmentation[i])]

    return labels_segmentation, flip_indices, unique_idx


def write_csv(path_csv, data, unique_file, labels, names, skip_first=True, last_first=False):

    # initialisation
    fs_utils.mkdir(os.path.dirname(path_csv))
    labels, unique_idx = np.unique(labels, return_index=True)
    if skip_first:
        labels = labels[1:]
    if names is not None:
        names = names[unique_idx].tolist()
        if skip_first:
            names = names[1:]
        header = names
    else:
        header = [str(lab) for lab in labels]
    if last_first:
        header = [header[-1]] + header[:-1]
    if (not unique_file) & (data is None):
        raise ValueError('data can only be None when initialising a unique volume file')

    # modify data
    if unique_file:
        if data is None:
            type_open = 'w'
            data = ['subject'] + header
        else:
            type_open = 'a'
        data = [data]
    else:
        type_open = 'w'
        header = [''] + header
        data = [header, data]

    # write csv
    with open(path_csv, type_open) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
