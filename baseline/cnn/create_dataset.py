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

    path_to_target_files = "../../../clarity_CEC2_data/clarity_data/train/targets"
    path_to_interferers_files = "../../../clarity_CEC2_data/clarity_data/train/interferers"
       
    # get training and validation tf record file names
    target_filenames = glob.glob(os.path.join(path_to_target_files, '*.wav'))[:100]

    music_interferers_filenames = glob.glob(os.path.join(path_to_interferers_files, 'music', '**', '*.mp3'), recursive=True)[:100]
    noise_interferers_filenames = glob.glob(os.path.join(path_to_interferers_files, 'noise', '*'))
    speech_interferers_filenames = glob.glob(os.path.join(path_to_interferers_files, 'speech', '*'))

    
    target_len = len(target_filenames)

    music_len = len(music_interferers_filenames)
    noise_len = len(noise_interferers_filenames)
    speech_len = len(speech_interferers_filenames)

    noise_filenames = music_interferers_filenames + noise_interferers_filenames + speech_interferers_filenames

    clean_train_set, clean_test_set  = train_test_split(target_filenames, test_size=0.2) 
    noise_train_set, noise_test_set = train_test_split(noise_filenames, test_size=0.2) 



    windowLength = 256
    config = {'windowLength': windowLength,
            'overlap': round(0.25 * windowLength),
            'fs': 16000,
            'audio_max_duration': 6}

    ## Create Train Set
    train_dataset = Dataset(clean_train_set, noise_train_set, **config)
    train_dataset.create_tf_record(prefix='train_6', subset_size=200, parallel=False)

    ## Create Test Set
    val_dataset = Dataset(clean_test_set, noise_test_set, **config)
    val_dataset.create_tf_record(prefix='val_6', subset_size=200, parallel=False)

if __name__ == "__main__":
    create()
