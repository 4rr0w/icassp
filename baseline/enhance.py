import math
import numpy as np
import datetime
import os, random, sys

import librosa.core as audio
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

import json
import logging
import pathlib

import hydra
import numpy as np
from evaluate import make_scene_listener_list
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm



def genStartVal(vLen,nLen):
    startVal = math.floor(abs(np.random.randn()*0.02*vLen))
    if startVal+vLen < nLen:
        return startVal
    else:
        genStartVal(vLen,nLen)

def spectrumSequence(time_series_data,nfft,fs_):
    nFiles = len(time_series_data)
    sequence = []
    lengths = []
    for idx in range(nFiles):
        thisData = time_series_data[idx].T.squeeze()
        spectrum =  audio.stft(thisData,n_fft=nfft,hop_length=int(nfft/2),center=False)
        Mag = np.abs(spectrum).T
        sequence.append(Mag)
        lengths.append(len(Mag))
    return sequence,lengths

def pad_seq(allData,maxlen):
    paddedData = pad_sequences(allData,maxlen=maxlen,dtype='float32',value=0.0)
    return paddedData

def back_to_wav(pred):
    pred = pred.squeeze()
    scaler2.inverse_transform(pred)
    scaler1.inverse_transform(pred)
    pred = pred.T
    wav = audio.istft(pred)
    return wav.T.squeeze()


def softMasking(y):
    input = y[0]
    y1_hat = y[1]
    y2_hat = y[2]
    s1,s2 = computeSoftMask(y1_hat,y2_hat)
    y1_tilde = tf.multiply(s1,input)
    y2_tilde = tf.multiply(s2,input)
    return [y1_tilde, y2_tilde]

def maskedOutShape(shape):
    shape_0 = list(shape[0])
    shape_1 = list(shape[1])
    return [tuple(shape_0),tuple(shape_1)]

def computeSoftMask(y1,y2):
    y1 = K.abs(y1)
    y2 = K.abs(y2)
    m1 = tf.divide(y1,tf.add(y1,y2))
    m2 = tf.divide(y2,tf.add(y1,y2))
    # m2 = 1 - m1
    return [m1,m2]


logger = logging.getLogger(__name__)

@hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:

    model_name = "model.hdf5"
    model = tf.keras.models.load_model(f"Models/{model_name}", compile = False)

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

    for scene, listener in tqdm(scene_listener_pairs[:50]):

        # # Audiograms can read like this, but they are not needed for the baseline
        #
        # cfs = np.array(listener_audiograms[listener]["audiogram_cfs"])
        #
        # audiogram_left = np.array(
        #    listener_audiograms[listener]["audiogram_levels_l"]
        # )
        # audiogram_right = np.array(
        #    listener_audiograms[listener]["audiogram_levels_r"]
        # )


        s_fs, signal1 = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
        )
        _, signal2 = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH2.wav"
        )
        _, signal3 = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH3.wav"
        )

        t_fs, target1 = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_target_CH1.wav"
        )
        _, target2 = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_target_CH2.wav"
        )
        _, target3 = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_target_CH3.wav"
        )


        i_fs, interferer1 = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_interferer_CH1.wav"
        )
        _, interferer2 = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_interferer_CH2.wav"
        )
        _, interferer3 = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_interferer_CH3.wav"
        )
      

        # # Convert to 32-bit floating point scaled between -1 and 1
        signal1 = (signal1 / 32768.0).astype(np.float32)
        signal2 = (signal2 / 32768.0).astype(np.float32)
        signal3 = (signal3 / 32768.0).astype(np.float32)

        target1 = (target1 / 32768.0).astype(np.float32)
        target2 = (target2 / 32768.0).astype(np.float32)
        target3 = (target3 / 32768.0).astype(np.float32)

        interferer1 = (interferer1 / 32768.0).astype(np.float32)
        interferer2 = (interferer2 / 32768.0).astype(np.float32)
        interferer3 = (interferer3 / 32768.0).astype(np.float32)

        # # Audiograms can read like this, but they are not needed for the baseline
        #
        # cfs = np.array(listener_audiograms[listener]["audiogram_cfs"])
        #
        # audiogram_left = np.array(
        #    listener_audiograms[listener]["audiogram_levels_l"]
        # )
        # audiogram_right = np.array(
        #    listener_audiograms[listener]["audiogram_levels_r"]
        # )

        # Baseline just reads the signal from the front microphone pair
        # and write it out as the enhanced signal
        #
        
        s_pred_voice, s_pred_noise = model.predict(signal1)
        print(s_pred_voice.shape)


        wavfile.write(
            enhanced_folder / f"{scene}_{listener}_enhanced.wav", s_fs, s_pred_voice
        )


if __name__ == "__main__":
    enhance()
