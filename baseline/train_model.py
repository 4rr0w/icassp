import math
import numpy as np
import datetime
# import scipy.signal as signal
# import scipy.io.wavfile as wav
# import scipy.io.wavfile as wav
import os, random, sys
# from pylab import plot,show, figure, imshow
# import matplotlib.pyplot as plt
import librosa.core as audio
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed,Dense,LSTM,Input,Lambda,Dropout #,CuDNNLSTM, CuDNNGRU,,BatchNormalization,
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers as reg
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import normalize, MinMaxScaler, MaxAbsScaler
import tensorflow.keras.backend as K
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

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
def train(cfg: DictConfig) -> None:
    with open(cfg.path.scenes_listeners_file, "r", encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)

    with open(cfg.path.listeners_file, "r", encoding="utf-8") as fp:
        listener_audiograms = json.load(fp)  # noqa: F841

    # Make list of all scene listener pairs that will be run
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, cfg.evaluate.small_test
    )
<<<<<<< HEAD

=======
        
>>>>>>> 02b7f3f (cnn model and clean code)
    mixed = []
    target = []
    noise = []

    for scene, listener in tqdm(scene_listener_pairs[:50]):
<<<<<<< HEAD
=======
        print(scene, listener)
>>>>>>> 02b7f3f (cnn model and clean code)

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

        mixed.append(signal1[:,0])
        target.append(target1[:,0])
        noise.append(interferer1[:,0])

    mir_music = noise
    mir_voice = target

    musicData = []
    voiceData = []
    musicAddedData = []

    for each,thisVoice in enumerate(mir_voice):
        thisMusic = mir_music[each]
        voiceBit = normalize(thisVoice.reshape(1,-1),norm='max')
        musicBit = normalize(thisMusic.reshape(1,-1),norm='max')
        #musicAdd = 0.5*np.add(voiceBit,musicBit)
        musicAdd = mixed[each]
        voiceData.append(voiceBit)
        musicData.append(musicBit)
        musicAddedData.append(musicAdd)


    nfft = 1024
<<<<<<< HEAD
    fs = 16000
=======
    fs = 41000
>>>>>>> 02b7f3f (cnn model and clean code)

    x_data,l1  = spectrumSequence(musicAddedData,nfft,fs)
    y1_data,l2 = spectrumSequence(voiceData,nfft,fs)
    y2_data,l3 = spectrumSequence(musicData,nfft,fs)

    assert len(x_data) == len(y1_data) == len(y2_data)

    # ----------------------------------------------
    # Normalize Spectra to the Input
    # ----------------------------------------------

    scaler1 = MaxAbsScaler(copy=False)
    scaler2 = MinMaxScaler(feature_range=(0.0,1.0),copy=False)

    for idx in range(len(x_data)):
        scaler1.fit_transform(x_data[idx])
        # scaler1.fit_transform(x_data[idx])
        scaler1.fit_transform(y1_data[idx])
        scaler1.fit_transform(y2_data[idx])
        # scaler2.fit(x_data[idx])
        scaler2.fit_transform(x_data[idx])
        scaler2.fit_transform(y1_data[idx])
        scaler2.fit_transform(y2_data[idx])



    l1 = max(l1)
    l2 = max(l2)
    l3 = max(l3)
    maxL = max(l1,l2,l3)

<<<<<<< HEAD
    del mir_music, mir_voice #, combinedDataFrames
    #
=======
    del mir_music, mir_voice #combinedDataFrames
>>>>>>> 02b7f3f (cnn model and clean code)
    train_x = pad_seq(x_data,maxL)
    y1      = pad_seq(y1_data,maxL)
    y2      = pad_seq(y2_data,maxL)


    del x_data, y1_data, y2_data

    batch_size = 10
    learning_rate = 1e-4
    decay_ = 1e-3
    epochs = 2
    n_units = 600 #int(2*nfft/1)

    shape = train_x.shape[1:]
<<<<<<< HEAD
    n_outs = train_x.shape[2] # Note: Not train_x.shape[1:], which returns shape for input_shape, instead of int.

=======
>>>>>>> 02b7f3f (cnn model and clean code)

    # # CPU Version :: Functional API
    regularizer = reg.l2(0.05)
    input_1 = Input(shape=shape)
    # input_mask = Masking(mask_value=0.,input_shape=shape)(input)
    hid1 = LSTM(n_units,return_sequences=True, activation='relu')(input_1)
    dp1  = Dropout(0.2)(hid1)
    hid2 = LSTM(n_units,return_sequences=True, activation='relu')(dp1)
    dp2  = Dropout(0.2)(hid2)
    hid3 = LSTM(n_units,return_sequences=True, activation='relu')(dp2)
    y1_hat = TimeDistributed(Dense(train_x.shape[2], activation='softmax', input_shape=train_x.shape[1:]), name='y1_hat')(hid3)
    y2_hat = TimeDistributed(Dense(train_x.shape[2], activation='softmax', input_shape=train_x.shape[1:]), name='y2_hat')(hid3)
    out1,out2 = Lambda(softMasking,maskedOutShape,name='softMask')([input_1,y1_hat,y2_hat])

    model = Model(inputs=input_1,outputs=[out1,out2])
    model.summary()
    #
    #
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='kullback_leibler_divergence',optimizer=opt, metrics=['acc','mse']) #kullback_leibler_divergence

    curdir = os.getcwd()+"/logs/"

    if not os.path.exists('Checkpoints'):
        os.makedirs('Checkpoints')

    chkpoint_path = os.getcwd()+"/Checkpoints/ModelChkpoint_epoch{epoch:02d}_vLoss{val_loss:.2f}.hdf5"

    tensorboard = TensorBoard(log_dir=curdir)

    checkpt = ModelCheckpoint(filepath=chkpoint_path,monitor='val_softMask_acc',save_best_only=True,save_weights_only=False)
    earlystop = EarlyStopping(monitor='val_softMask_acc', min_delta=1e-3, patience=10)
    history = model.fit(train_x,[y1,y2],batch_size=batch_size,epochs=epochs,validation_split=0.825,callbacks=[tensorboard,checkpt,earlystop])

    if not os.path.exists('Models'):
        os.makedirs('Models')

    date_time = datetime.datetime.now()
    model_path = os.getcwd()+f"/Models/Model_{date_time}.hdf5"
    print(model_path)
    model.save(model_path)

if __name__ == "__main__":
    train()
