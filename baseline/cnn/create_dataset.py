from data_processing.dataset import Dataset
import warnings
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.python.client import device_lib
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

warnings.filterwarnings(action='ignore')

def create():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)

    tf.device('/gpu:0')
    path_to_scene_files = "../../../clarity_CEC2_data_train/clarity_data/train/scenes/"
  
    # get training and validation file names
    scene_filenames = glob.glob(os.path.join(path_to_scene_files, '*_hr.wav'))[:1000]
    scene_train_set, scene_test_set  = train_test_split(scene_filenames, test_size=0.2) 

    windowLength = 256
    config = {'windowLength': windowLength,
            'overlap': round(0.25 * windowLength),
            'fs': 8000,
        }

    ## Create Train Set
    train_dataset = Dataset(scene_train_set, use_mel_spec = True, **config)
    train_dataset.create_tf_record(prefix='train_mel', subset_size=500, parallel=False)

    ## Create Test Set
    val_dataset = Dataset(scene_test_set, use_mel_spec = True, **config)
    val_dataset.create_tf_record(prefix='val_mel', subset_size=500, parallel=False)

if __name__ == "__main__":
    create()
