# dreyeve-modified
Modified DrEyeVe network with training and inference script to run our gaze data

# Dependencies
Run `pip install -r requirements.txt` in the `train` folder. Make sure you're using Python2 (python 2.7) - most of the requirements are deprecated or unavailable with Python 3.

You will have to download `computer_vision_utils` from SageNet Google Drive found at the link https://drive.google.com/drive/folders/1LCuyO69kBmGnhVOmkM1jpFS8Ph3VvkIq for one of the helper functions used for the script.

Since these scripts use Theano backend, to use GPU, run `THEANO_FLAGS=device=cuda0 python {script}`

# Parameters
`config.py` contains all the hyperparameters and constants used for training and inference. This file can be modified accordingly.

# Training
Run `python train.py` - you will have to modify the constants in `config.py` before you run so that the script is pointed to the correct data folder path etc.

# Demo
Run `python demo.py` - you will have to modify the constants in `config.py` before you run so that the script is pointed to the correct data folder path etc.

# Acknowledgement & Links
This repo is a modified version of the original repo for the DR (eye) ve network found at https://github.com/ndrplz/dreyeve
