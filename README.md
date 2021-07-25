# dreyeve-modified
Modified DrEyeVe network with inference script to run our gaze data

# How to run
Run `pip install -r requirements.txt` in the `train` folder. Make sure you're using Python2 (python 2.7) - most of the requirements are deprecated or unavailable with Python 3.

You will have to download `computer_vision_utils` from SageNet Google Drive found at the link https://drive.google.com/drive/folders/1LCuyO69kBmGnhVOmkM1jpFS8Ph3VvkIq for one of the helper functions used for the script.

Then, run `python demo.py` - you will have to modify the constants in `demo.py` before you run so that the script is pointed to the correct data folder path etc.

In summary, 
```
cd train
pip install -r requirements.txt
python demo.py
```

# How to train 
Follow all the instructions above but instead of `demo.py`, run `train.py`. 