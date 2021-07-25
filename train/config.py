import os

# use Theano backend instead of Tensorflow backend for keras
os.environ['KERAS_BACKEND'] = 'theano'

from keras.optimizers import Adam

# --- GLOBAL --- #
DREYEVE_DIR = "../shared/hca_grp/hca_attention/"
DATA_MODE = "manual"
GT_TYPE = "eye_tracker"

# output directories
LOG_DIR = "./logs"
CKP_DIR = "./checkpoints"
PRD_DIR = "./predictions"

# --- TRAIN --- #
BATCH_SIZE = 16
FRAMES_PER_SEQ = 16
H = 448
W = 448
FRAME_SIZE_BEFORE_CROP = (256, 256)
CROP_TYPE = 'central'  # choose among [`central`, `random`]

TRAIN_SAMPLES_PER_EPOCH = 512 * BATCH_SIZE
VAL_SAMPLES_PER_EPOCH = 64 * BATCH_SIZE
NB_EPOCHS = 20


# optimizer
FULL_FRAME_LOSS = 'kld'
CROP_LOSS = 'kld'
W_LOSS_FINE = 1.0
W_LOSS_CROPPED = 1.0
mse_beta = 0.1
OPT = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)\

# callbacks
CALLBACK_BATCHSIZE = 8