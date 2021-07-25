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

from utils import setup_dataset

from config import DREYEVE_DIR, DATA_MODE, GT_TYPE
from config import BATCH_SIZE, FRAMES_PER_SEQ, H, W, OPT
from config import TRAIN_SAMPLES_PER_EPOCH, VAL_SAMPLES_PER_EPOCH, NB_EPOCHS
from config import FULL_FRAME_LOSS, CROP_LOSS, W_LOSS_FINE, W_LOSS_CROPPED

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
                                                           image_size=(H, W), mode='train', gt_type=GT_TYPE, sequence_ids=sequence_ids),
                        validation_data=generate_dreyeve_I_batch(batchsize=BATCH_SIZE, nb_frames=FRAMES_PER_SEQ,
                                                                 image_size=(H, W), mode='val', gt_type=GT_TYPE, sequence_ids=sequence_ids),
                        nb_val_samples=TRAIN_SAMPLES_PER_EPOCH,
                        samples_per_epoch=VAL_SAMPLES_PER_EPOCH,
                        nb_epoch=NB_EPOCHS,
                        callbacks=get_callbacks(experiment_id=experiment_id))


# training entry point
if __name__ == '__main__':
    train_image_branch()
