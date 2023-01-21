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

windowLength = 256
overlap      = round(0.25 * windowLength) # overlap of 75%
ffTLength    = windowLength
inputFs      = 44.1e3
fs           = 16e3
numFeatures  = ffTLength//2 + 1
numSegments  = 8

def revert_features_to_audio(features, phase, noiseAudioFeatureExtractor, cleanMean=None, cleanStd=None):
    # scale the outpus back to the original range
    if cleanMean and cleanStd:
        features = cleanStd * features + cleanMean

    phase = np.transpose(phase, (1, 0))
    features = np.squeeze(features)

    # features = librosa.db_to_power(features)
    features = features * np.exp(1j * phase)  # that fixes the abs() ope previously done

    features = np.transpose(features, (1, 0))
    return noiseAudioFeatureExtractor.get_audio_from_stft_spectrogram(features)


def prepare_input_features(stft_features):
    # Phase Aware Scaling: To avoid extreme differences (more than
    # 45 degree) between the noisy and clean phase, the clean spectral magnitude was encoded as similar to [21]:
    noisySTFT = np.concatenate([stft_features[:,0:numSegments-1], stft_features], axis=1)
    stftSegments = np.zeros((numFeatures, numSegments , noisySTFT.shape[1] - numSegments + 1))

    for index in range(noisySTFT.shape[1] - numSegments + 1):
        stftSegments[:,:,index] = noisySTFT[:,index:index + numSegments]
    return stftSegments


def denoise_with_cnn_model(signal, model):
    noiseAudioFeatureExtractor = FeatureExtractor(signal, windowLength=windowLength, overlap=overlap, sample_rate=fs)
    noise_stft_features = noiseAudioFeatureExtractor.get_stft_spectrogram()

    # Paper: Besides, spectral phase was not used in the training phase.
    # At reconstruction, noisy spectral phase was used instead to
    # perform in-verse STFT and recover human speech.
    noisyPhase = np.angle(noise_stft_features)
    
    noise_stft_features = np.abs(noise_stft_features)

    mean = np.mean(noise_stft_features)
    std = np.std(noise_stft_features)
    noise_stft_features = (noise_stft_features - mean) / std

    predictors = prepare_input_features(noise_stft_features)
    predictors = np.reshape(predictors, (predictors.shape[0], predictors.shape[1], 1, predictors.shape[2]))
    predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)

    
    STFTFullyConvolutional = model.predict(predictors)
    
    denoisedAudioFullyConvolutional = revert_features_to_audio(STFTFullyConvolutional, noisyPhase,noiseAudioFeatureExtractor, mean, std)
    return denoisedAudioFullyConvolutional
   

def plot_wav(sr, data):
    for i, d in enumerate(data):
        signal = d[:]
        Time = np.linspace(0, len(signal) / sr, num=len(signal))
        plt.subplot(len(data), 1, i + 1)
        plt.plot(Time, signal, lw=0.15)
   
    plt.show()



@hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:

    model_name = "../denoiser_cnn_mix.h5"
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
        ch1_data = (ch1_data / 32768.0).astype(np.float32)
        ch2_data = (ch2_data / 32768.0).astype(np.float32)
        ch3_data = (ch3_data / 32768.0).astype(np.float32)    

        denoised_left = denoise_with_cnn_model(ch1_data[:, 0], model)
        denoised_rigth = denoise_with_cnn_model(ch1_data[:, 1], model)
    
        # print(denoised_left, denoised_rigth)
        denoised_signal = np.column_stack((denoised_left, denoised_rigth))

        # data = [
        # ch1_data, 
        # ch2_data, 
        # ch3_data, 
        # denoisedsignal,
        # # denoised_ch1, 
        # # denoised_ch2, 
        # # denoised_ch3, 
        # # combined_signal
        # ]
        # plot_wav(fs, data)
         
        wavfile.write(
            enhanced_folder / f"{scene}_{listener}_enhanced.wav", fs, denoised_signal
        )


if __name__ == "__main__":
    enhance()
