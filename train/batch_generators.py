import numpy as np
import cv2

from random import choice
from os.path import join
import os
from time import time

import sys
sys.path.append('..')

from computer_vision_utils.io_helper import read_image, normalize
from computer_vision_utils.stitching import stitch_together
from computer_vision_utils.tensor_manipulation import resize_tensor, crop_tensor

from config import DREYEVE_DIR, DATA_MODE, FRAME_SIZE_BEFORE_CROP, CROP_TYPE, FRAMES_PER_SEQ


def sample_signature(sequence_ids, image_size, allow_mirror):
    """Function to create a unique batch signature for the Dreyeve dataset."""
    h, w = image_size
    h_c = h // 4
    w_c = w // 4
    h_before_crop, w_before_crop = FRAME_SIZE_BEFORE_CROP

    # get random sequence
    sequence_id = choice(sequence_ids)

    # get random start of sequence
    video_frames = [name for name in os.listdir(os.path.join(DREYEVE_DIR, sequence_id, 'images_4hz'))]
    start_frame_index = np.random.randint(8, len(video_frames) - FRAMES_PER_SEQ - 1)

    # get random crop
    if CROP_TYPE == 'central':
        hc1 = h_before_crop // 4
        wc1 = w_before_crop // 4
    elif CROP_TYPE == 'random':
        hc1 = np.random.randint(0, h_before_crop - h_c)
        wc1 = np.random.randint(0, w_before_crop - w_c)
    else:
        raise ValueError
    
    hc2 = hc1 + h_c
    wc2 = wc1 + w_c

    do_mirror = choice([True, False]) if allow_mirror else False

    return tuple((sequence_id, start_frame_index, hc1, hc2, wc1, wc2, do_mirror))


def load_batch_data(signatures, nb_frames, image_size, batch_type):
    """
    Function to load a data batch. This is common for `image`, `optical_flow` and `semseg`.

    :param signatures: sample signatures, previously evaluated. List of tuples like
                    (num_run, start, hc1, hc2, wc1, wc2). The list is batchsize signatures long.
    :param nb_frames: number of temporal frames in each sample.
    :param image_size: tuple in the form (h,w). This refers to the fullframe image.
    :param batch_type: choose among [`image`, `optical_flow`, `semseg`].
    :return: a tuple holding the fullframe, the small and the cropped batch.
    """
    assert batch_type == 'image', 'Unknown batch type: {}'.format(batch_type)

    batchsize = len(signatures)
    h, w = image_size
    h_s = h_c = h // 4
    w_s = w_c = w // 4

    B_ff = np.zeros(shape=(batchsize, 3, 1, h, w), dtype=np.float32)
    B_s = np.zeros(shape=(batchsize, 3, nb_frames, h_s, w_s), dtype=np.float32)
    B_c = np.zeros(shape=(batchsize, 3, nb_frames, h_c, w_c), dtype=np.float32)
    # mean frame image, to subtract mean
    # ----------------------------- Are we subtracting the mean frame?  -----------------------------
    # mean_image = read_image(join(DREYEVE_DIR, 'dreyeve_mean_frame.png'), channels_first=True, resize_dim=image_size)

    for b in range(batchsize):
        # retrieve the signature
        sequence_id, start, hc1, hc2, wc1, wc2, do_mirror = signatures[b]

        video_frames = [name for name in os.listdir(os.path.join(DREYEVE_DIR, sequence_id, 'images_4hz'))]
        video_frames.sort()
        video_frames = video_frames[start:start+FRAMES_PER_SEQ]

        for offset in range(0, nb_frames):
            x = read_image(join(DREYEVE_DIR, sequence_id, 'images_4hz', video_frames[offset]), channels_first=True, resize_dim=image_size)

            # resize to (256, 256) before cropping
            x_before_crop = resize_tensor(x, new_size=FRAME_SIZE_BEFORE_CROP)

            B_s[b, :, offset, :, :] = resize_tensor(x, new_size=(h_s, w_s))
            B_c[b, :, offset, :, :] = crop_tensor(x_before_crop, indexes=(hc1, hc2, wc1, wc2))
        
        B_ff[b, :, 0, :, :] = x

        if do_mirror:
            B_ff = B_ff[:, :, :, :, ::-1]
            B_s = B_s[:, :, :, :, ::-1]
            B_c = B_c[:, :, :, :, ::-1]

    return [B_ff, B_s, B_c]


def load_saliency_data(signatures, nb_frames, image_size, gt_type):
    """
    Function to load a saliency batch.

    :param signatures: sample signatures, previously evaluated. List of tuples like
                    (num_run, start, hc1, hc2, wc1, wc2). The list is batchsize signatures long.
    :param nb_frames: number of temporal frames in each sample.
    :param image_size: tuple in the form (h,w). This refers to the fullframe image.
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth).
    :return: a tuple holding the fullframe and the cropped saliency.
    """

    batchsize = len(signatures)
    h, w = image_size
    h_c = h // 4
    w_c = w // 4

    Y = np.zeros(shape=(batchsize, 1, h, w), dtype=np.float32)
    Y_c = np.zeros(shape=(batchsize, 1, h_c, w_c), dtype=np.float32)

    for b in range(0, batchsize):
        # retrieve the signature
        sequence_id, start, hc1, hc2, wc1, wc2, do_mirror = signatures[b]

        video_frames = [name for name in os.listdir(os.path.join(DREYEVE_DIR, sequence_id, 'heatmap_4hz_20_eye_tracker'))]
        video_frames.sort()
        video_frames = video_frames[start:start+FRAMES_PER_SEQ]

        # saliency, choose between webcam and eye tracker as ground truth
        if gt_type == "webcam":
            y = read_image(join(DREYEVE_DIR, sequence_id, 'heatmap_4hz_20_gaze_recorder', video_frames[nb_frames - 1]), channels_first=True, color=False, resize_dim=image_size)
        elif gt_type == "eye_tracker":
            y = read_image(join(DREYEVE_DIR, sequence_id, 'heatmap_4hz_20_eye_tracker', video_frames[nb_frames - 1]), channels_first=True, color=False, resize_dim=image_size)

        # resize to (256, 256) before cropping
        y_before_crop = resize_tensor(np.expand_dims(y, axis=0), new_size=FRAME_SIZE_BEFORE_CROP)

        Y[b, 0, :, :] = y
        Y_c[b, 0, :, :] = crop_tensor(y_before_crop, indexes=(hc1, hc2, wc1, wc2))[0]

        if do_mirror:
            Y = Y[:, :, :, ::-1]
            Y_c = Y_c[:, :, :, ::-1]

    return [Y, Y_c]


def dreyeve_I_batch(batchsize, nb_frames, image_size, mode, gt_type, sequence_ids):
    """
    Function to load a Dreyeve batch of only images.

    :param batchsize: batchsize.
    :param nb_frames: number of temporal frames in each sample.
    :param image_size: tuple in the form (h,w). This refers to the fullframe image.
    :param mode: choose among [`train`, `val`, `test`].
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth).
    :return: an image batch and the relative saliency in the form (X,Y).
    """
    assert mode in ['train', 'val', 'test'], 'Unknown mode {} for dreyeve batch loader'.format(mode)
    assert gt_type in ['webcam', 'eye_tracker'], 'Unknown gt_type {} for dreyeve batch loader'.format(gt_type)

    if mode == 'train':
        allow_mirror = True
    elif mode == 'val':
        allow_mirror = False
    elif mode == 'test':
        allow_mirror = False

    # generate batch signatures
    signatures = []
    for b in range(0, batchsize):
        signatures.append(sample_signature(sequence_ids=sequence_ids, image_size=image_size, allow_mirror=allow_mirror))

    # get an image batch
    I = load_batch_data(signatures=signatures, nb_frames=nb_frames, image_size=image_size, batch_type='image')
    Y = load_saliency_data(signatures=signatures, nb_frames=nb_frames, image_size=image_size, gt_type=gt_type)
    return I, Y


def generate_dreyeve_I_batch(batchsize, nb_frames, image_size, mode, gt_type, sequence_ids):
    """
    Function to generate a batch from the dreyeve dataset. The batch will only contain images.

    :param batchsize: batchsize.
    :param nb_frames: number of frames for each batch.
    :param image_size: dimension of tensors.
    :param mode: `train` or `test`.
    :param gt_type: choose among `sal` (old groundtruth) and `fix` (new groundtruth).
    :return: a tuple like: ([X, X_s, X_c], [Y, Y_c]).
    """
    while True:
        yield dreyeve_I_batch(batchsize=batchsize, nb_frames=nb_frames,
                              image_size=image_size, mode=mode, gt_type=gt_type, sequence_ids=sequence_ids)
