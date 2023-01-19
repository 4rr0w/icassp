from data_processing.dataset import Dataset
import warnings
import numpy as np
import glob
import os

warnings.filterwarnings(action='ignore')

path_to_target_files = "../../data/clarity_CEC2_data/clarity_data/train/targets/"
path_to_interferers_files = "../../data/clarity_CEC2_data/clarity_data/train/interferers/"

# get training and validation tf record file names
target_filenames = glob.glob(os.path.join(path_to_target_files, '*.wav'))

music_interferers_filenames = glob.glob(os.path.join(path_to_interferers_files, 'music/**/*.mp3'), recursive=True)
noise_interferers_filenames = glob.glob(os.path.join(path_to_interferers_files, 'noise/*'))
speech_interferers_filenames = glob.glob(os.path.join(path_to_interferers_files, 'speech/*'))

# print(music_interferers_filenames)
target_len = len(target_filenames)
music_len = len(music_interferers_filenames)
noise_len = len(noise_interferers_filenames)
speech_len = len(speech_interferers_filenames)
print(target_len, music_len, noise_len, speech_len)


# train dataset files
clean_train_set = target_filenames[:target_len//2]
noise_train_set = music_interferers_filenames[:music_len//2] + noise_interferers_filenames[:noise_len//2] + speech_interferers_filenames[:speech_len//2]
#validation dataset files
clean_val_set = target_filenames[target_len//2:]
noise_val_set = music_interferers_filenames[music_len//2:] + noise_interferers_filenames[noise_len//2:] + speech_interferers_filenames[speech_len//2:]

windowLength = 256
config = {'windowLength': windowLength,
          'overlap': round(0.25 * windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}

# ## Create Train Set
# train_dataset = Dataset(clean_train_set, noise_train_set, **config)
# train_dataset.create_tf_record(prefix='train_1000', subset_size=100)

## Create Test Set
val_dataset = Dataset(clean_val_set, noise_val_set, **config)
val_dataset.create_tf_record(prefix='val_1000', subset_size=100, parallel=False)
