import argparse
import uuid
import os

# use Theano backend instead of Tensorflow backend for keras
os.environ['KERAS_BACKEND'] = 'theano'

from keras.optimizers import Adam
from batch_generators import generate_dreyeve_I_batch
from models import SaliencyBranch
from loss_functions import saliency_loss
from callbacks import get_callbacks
from tqdm import tqdm
import random
import pandas as pd

# training constants
DREYEVE_DIR = "../shared/hca_grp/hca_attention/"
DATA_MODE = "manual"

BATCH_SIZE = 16
FRAMES_PER_SEQ = 16
H = 448
W = 448
OPT = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
TRAIN_SAMPLES_PER_EPOCH = 512 * BATCH_SIZE
VAL_SAMPLES_PER_EPOCH = 64 * BATCH_SIZE
NB_EPOCHS = 20

FULL_FRAME_LOSS = 'kld'
CROP_LOSS = 'kld'
W_LOSS_FINE = 1.0
W_LOSS_CROPPED = 1.0

def setup_dataset():
    sequence_ids = []
    for participant_id in tqdm(os.listdir(os.path.join(DREYEVE_DIR))):
        if os.path.exists(os.path.join(DREYEVE_DIR, participant_id, 'alignment.csv')):
            alignment = pd.read_csv(os.path.join(DREYEVE_DIR, participant_id, 'alignment.csv'))

            for session_name in alignment['session_name']:
                input_path = os.path.join(participant_id, session_name)
                if DATA_MODE in input_path:
                    sequence_ids.append(input_path)
    seed = 0
    random.Random(seed).shuffle(sequence_ids)
    return sequence_ids


def train_image_branch():
    """
    Function to train a SaliencyBranch model on images.
    """

    experiment_id = 'COLOR_{}'.format(uuid.uuid4())
    sequence_ids = setup_dataset()

    model = SaliencyBranch(input_shape=(3, FRAMES_PER_SEQ, H, W), c3d_pretrained=True, branch='image')
    model.compile(optimizer=OPT,
                  loss={'prediction_fine': saliency_loss(name=FULL_FRAME_LOSS),
                        'prediction_crop': saliency_loss(name=CROP_LOSS)},
                  loss_weights={'prediction_fine': W_LOSS_FINE,
                                'prediction_crop': W_LOSS_CROPPED})
    model.summary()

    model.fit_generator(generator=generate_dreyeve_I_batch(batchsize=BATCH_SIZE, nb_frames=FRAMES_PER_SEQ,
                                                           image_size=(H, W), mode='train', gt_type='fix', sequence_ids=sequence_ids),
                        validation_data=generate_dreyeve_I_batch(batchsize=BATCH_SIZE, nb_frames=FRAMES_PER_SEQ,
                                                                 image_size=(H, W), mode='val', gt_type='fix', sequence_ids=sequence_ids),
                        nb_val_samples=TRAIN_SAMPLES_PER_EPOCH,
                        samples_per_epoch=VAL_SAMPLES_PER_EPOCH,
                        nb_epoch=NB_EPOCHS,
                        callbacks=get_callbacks(experiment_id=experiment_id))


# training entry point
if __name__ == '__main__':
    train_image_branch()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--which_branch")

    # args = parser.parse_args()
    # assert args.which_branch in ['finetuning', 'image', 'flow', 'seg']

    # if args.which_branch == 'image':
    #     train_image_branch()
    # elif args.which_branch == 'flow':
    #     train_flow_branch()
    # elif args.which_branch == 'seg':
    #     train_seg_branch()
    # else:
    #     fine_tuning()





# def fine_tuning():
#     """
#     Function to launch training on DreyeveNet. It is called `fine_tuning` since supposes
#     the three branches to be pretrained. Should also work from scratch.
#     """

#     experiment_id = 'DREYEVE_{}'.format(uuid.uuid4())

#     model = DreyeveNet(frames_per_seq=frames_per_seq, h=h, w=w)
#     model.compile(optimizer=opt,
#                   loss={'prediction_fine': saliency_loss(name=full_frame_loss),
#                         'prediction_crop': saliency_loss(name=crop_loss)},
#                   loss_weights={'prediction_fine': w_loss_fine,
#                                 'prediction_crop': w_loss_cropped})
#     model.summary()

#     model.fit_generator(generator=generate_dreyeve_batch(batchsize=batchsize, nb_frames=frames_per_seq,
#                                                          image_size=(h, w), mode='train'),
#                         validation_data=generate_dreyeve_batch(batchsize=batchsize, nb_frames=frames_per_seq,
#                                                                image_size=(h, w), mode='val'),
#                         nb_val_samples=val_samples_per_epoch,
#                         samples_per_epoch=train_samples_per_epoch,
#                         nb_epoch=nb_epochs,
#                         callbacks=get_callbacks(experiment_id=experiment_id))
                        
# def train_flow_branch():
#     """
#     Function to train a SaliencyBranch model on optical flow.
#     """
#     experiment_id = 'FLOW_{}'.format(uuid.uuid4())

#     model = SaliencyBranch(input_shape=(3, frames_per_seq, h, w), c3d_pretrained=True, branch='flow')
#     model.compile(optimizer=opt,
#                   loss={'prediction_fine': saliency_loss(name=full_frame_loss),
#                         'prediction_crop': saliency_loss(name=crop_loss)},
#                   loss_weights={'prediction_fine': w_loss_fine,
#                                 'prediction_crop': w_loss_cropped})
#     model.summary()

#     model.fit_generator(generator=generate_dreyeve_OF_batch(batchsize=batchsize, nb_frames=frames_per_seq,
#                                                             image_size=(h, w), mode='train'),
#                         validation_data=generate_dreyeve_OF_batch(batchsize=batchsize, nb_frames=frames_per_seq,
#                                                                   image_size=(h, w), mode='val'),
#                         nb_val_samples=val_samples_per_epoch,
#                         samples_per_epoch=train_samples_per_epoch,
#                         nb_epoch=nb_epochs,
#                         callbacks=get_callbacks(experiment_id=experiment_id))


# def train_seg_branch():
#     """
#     Function to train a SaliencyBranch model on semantic segmentation.
#     """

#     experiment_id = 'SEGM_{}'.format(uuid.uuid4())

#     model = SaliencyBranch(input_shape=(19, frames_per_seq, h, w), c3d_pretrained=False, branch='semseg')
#     model.compile(optimizer=opt,
#                   loss={'prediction_fine': saliency_loss(name=full_frame_loss),
#                         'prediction_crop': saliency_loss(name=crop_loss)},
#                   loss_weights={'prediction_fine': w_loss_fine,
#                                 'prediction_crop': w_loss_cropped})
#     model.summary()

#     model.fit_generator(generator=generate_dreyeve_SEG_batch(batchsize=batchsize, nb_frames=frames_per_seq,
#                                                              image_size=(h, w), mode='train'),
#                         validation_data=generate_dreyeve_SEG_batch(batchsize=batchsize, nb_frames=frames_per_seq,
#                                                                    image_size=(h, w), mode='val'),
#                         nb_val_samples=val_samples_per_epoch,
#                         samples_per_epoch=train_samples_per_epoch,
#                         nb_epoch=nb_epochs,
#                         callbacks=get_callbacks(experiment_id=experiment_id))