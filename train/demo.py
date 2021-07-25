# Usage: THEANO_FLAGS=device=cuda python demo.py

import numpy as np
import cv2

import os
from os.path import join

import sys
sys.path.append('..')
from models import SaliencyBranch
from computer_vision_utils.io_helper import read_image, normalize
from computer_vision_utils.tensor_manipulation import resize_tensor

from config import DREYEVE_DIR as DATASET_PATH
from config import DATA_MODE

from utils import setup_dataset


def blend_map(img, map, factor, colormap=cv2.COLORMAP_JET):

    assert 0 < factor < 1, 'factor must satisfy 0 < factor < 1'

    map = np.float32(map)
    map /= map.max()
    map *= 255
    map = map.astype(np.uint8)

    blend = cv2.addWeighted(src1=img, alpha=factor,
                            src2=cv2.applyColorMap(map, colormap), beta=(1-factor),
                            gamma=0)

    return blend

def normalize_map(s_map):
	# normalize the salience map (as done in MIT code)
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return 255.0*norm_s_map


def load_dreyeve_sample(sequence_id, sample, frames_per_seq=16, h=448, w=448):

    # get video frame files
    video_frames = [name for name in os.listdir(os.path.join(DATASET_PATH, sequence_id, 'images_4hz'))]
    start_frame_index = np.random.randint(8, len(video_frames) - frames_per_seq - 1)
    video_frames.sort()
    video_frames = video_frames[start_frame_index:start_frame_index+frames_per_seq]
    # print("last image is ", video_frames[frames_per_seq-1])

    h_c = h_s = h // 4
    w_c = w_s = h // 4

    I_ff = np.zeros(shape=(1, 3, 1, h, w), dtype='float32')
    I_s = np.zeros(shape=(1, 3, frames_per_seq, h_s, w_s), dtype='float32')
    I_c = np.zeros(shape=(1, 3, frames_per_seq, h_c, w_c), dtype='float32')

    for fr in xrange(0, frames_per_seq):
        # read image
        # x = get_image(os.path.join(DATASET_PATH, sequence_id, 'images_4hz', video_frames[fr]), h, w, normalized=True)
        # - mean dreyeve image?
        x = read_image(join(DATASET_PATH, sequence_id, 'images_4hz', video_frames[fr]), channels_first=True, resize_dim=(h, w))

        I_s[0, :, fr, :, :] = resize_tensor(x, new_size=(h_s, w_s))


    I_ff[0, :, 0, :, :] = x

    return [I_ff, I_s, I_c], join(DATASET_PATH, sequence_id, 'images_4hz', video_frames[frames_per_seq-1])


if __name__ == '__main__':
    frames_per_seq, h, w = 16, 448, 448
    verbose = True

    sequence_ids = setup_dataset()
    print("Sequence ids : ", sequence_ids)
    print("Sequence_ids set up")

    model_path_bdda = os.path.join("bdda_image_branch.h5")

    image_branch_bdda = SaliencyBranch(input_shape=(3, frames_per_seq, h, w), c3d_pretrained=True, branch='image')
    image_branch_bdda.compile(optimizer='adam', loss='kld')
    image_branch_bdda.load_weights(model_path_bdda)  # load weights

    print("Model weights loaded")

    idx = 0
    sequence_id = sequence_ids[idx]
    X, demo_img= load_dreyeve_sample(sequence_id=sequence_id, sample=64, frames_per_seq=frames_per_seq, h=h, w=w)

    Y_image_bdda = image_branch_bdda.predict(X[:3])[0]  # predict on image

    print("Output predicted")

    im = read_image(demo_img, channels_first=True)

    h, w = im.shape[1], im.shape[2]

    bdda_pred = Y_image_bdda[0]

    bdda_pred = np.expand_dims(cv2.resize(bdda_pred[0], dsize=(w, h)), axis=0)

    im = normalize_map(im)
    bdda_pred = normalize_map(bdda_pred)

    im = im.astype(np.uint8)
    bdda_pred = bdda_pred.astype(np.uint8)

    im = np.transpose(im, (1, 2, 0))
    bdda_pred = np.transpose(bdda_pred, (1, 2, 0))

    heatmap_bdda = blend_map(im, bdda_pred, factor=0.5)

    cv2.imwrite(join('heatmap_bdda.jpg'), heatmap_bdda)

    print("Formatted and Done.")
