from data_processing.dataset import Dataset
import warnings
import numpy as np
import glob
import os

warnings.filterwarnings(action='ignore')

path_to_dataset = "../../data/clarity_CEC2_data/clarity_data/dev/scenes/"

# get training and validation tf record file names
limit = 1000
mix_filenames = glob.glob(os.path.join(path_to_dataset, '*_mix_CH1*'))[:limit]
clean_filenames = glob.glob(os.path.join(path_to_dataset, '*_target_CH1*'))[:limit]
noise_filenames = glob.glob(os.path.join(path_to_dataset, '*_interferer_CH1*'))[:limit]


windowLength = 256
config = {'windowLength': windowLength,
          'overlap': round(0.25 * windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}

train_dataset = Dataset(clean_filenames[:limit//2], mix_filenames[:limit//2], **config)
train_dataset.create_tf_record(prefix='train_mix', subset_size=500)

# ## Create Test Set
test_dataset = Dataset(clean_filenames[limit//2:], mix_filenames[:limit//2], **config)
test_dataset.create_tf_record(prefix='test_mix', subset_size=500, parallel=False)
