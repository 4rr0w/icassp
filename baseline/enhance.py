import math
import numpy as np
import datetime
import os, random, sys

import librosa
import scipy
import tensorflow as tf
import json
import logging
import pathlib

import hydra
import numpy as np
from evaluate import make_scene_listener_list
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm
from cnn.data_processing.feature_extractor import FeatureExtractor
import math
import matplotlib.pyplot as plt
import noisereduce as nr

windowLength = 256
overlap      = round(0.25 * windowLength) # overlap of 75%
ffTLength    = windowLength
inputFs      = 44.1e3
fs           = 16e3
numFeatures  = ffTLength//2 + 1
numSegments  = 8   

def plot_wav(sr, data):
    for i, d in enumerate(data):
        signal = d[:]
        Time = np.linspace(0, len(signal) / sr, num=len(signal))
        plt.subplot(len(data), 1, i + 1)
        plt.plot(Time, signal, lw=0.15)
   
    plt.show()

@hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:
    model_name = "../sigmoid_cnn_new.h5"
    model = tf.keras.models.load_model(f"{model_name}", compile = False)

    enhanced_folder = pathlib.Path("enhanced_signals")
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    with open(cfg.path.scenes_listeners_file, "r", encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)

    with open(cfg.path.listeners_file, "r", encoding="utf-8") as fp:
        listener_audiograms = json.load(fp)  # noqa: F841

    # Make list of all scene listener pairs that will be run
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, cfg.evaluate.small_test
    )

    for scene, listener in tqdm(scene_listener_pairs):
        ## Audiograms can read like this, but they are not needed for the baseline
        #
        # cfs = np.array(listener_audiograms[listener]["audiogram_cfs"])
        #
        # audiogram_left = np.array(
        #    listener_audiograms[listener]["audiogram_levels_l"]
        # )
        # audiogram_right = np.array(
        #    listener_audiograms[listener]["audiogram_levels_r"]
        # )

        fs, noise = wavfile.read(
        pathlib.Path(cfg.path.scenes_folder) / f"{scene}_interferer_CH1.wav"
        )

        fs, ch1_data = wavfile.read(
        pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
        )
        _, ch2_data = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH2.wav"
        )
        _, ch3_data = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH3.wav"
        )

        ## Convert to 32-bit floating point scaled between -1 and 1
        noise = (noise / 32768.0).astype(np.float32)
        ch1_data = (ch1_data / 32768.0).astype(np.float32)
        ch2_data = (ch2_data / 32768.0).astype(np.float32)
        ch3_data = (ch3_data / 32768.0).astype(np.float32)    

        reduced_mix_left = nr.reduce_noise(y=ch1_data[:,0], sr=fs)
        reduced_mix_right = nr.reduce_noise(y=ch1_data[:,1], sr=fs,)
        denoised_signal = np.column_stack((reduced_mix_left, reduced_mix_right))
         
        wavfile.write(
            enhanced_folder / f"{scene}_{listener}_enhanced.wav", fs, denoised_signal
        )

if __name__ == "__main__":
    enhance()
