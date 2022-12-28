""" Run the dummy enhancement. """
import json
import logging
import pathlib

import hydra
import numpy as np
from evaluate import make_scene_listener_list
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm
import noisereduce as nr

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.io import wavfile

def plot_wav(sr, data):
    for i, d in enumerate(data):
        signal = d[:]
        Time = np.linspace(0, len(signal) / sr, num=len(signal))
        plt.subplot(len(data), 1, i + 1)
        plt.plot(Time, signal, lw=0.15)
   
    plt.show()
    

@hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:
    """Run the dummy enhancement."""

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
         
    scene, listener = scene_listener_pairs[10]
    
    
    # Audiograms can read like this, but they are not needed for the baseline  
    cfs = np.array(listener_audiograms[listener]["audiogram_cfs"])
    
    audiogram_left = np.array(
        listener_audiograms[listener]["audiogram_levels_l"]
    )
    audiogram_right = np.array(
        listener_audiograms[listener]["audiogram_levels_r"]
    )
    print(scene, listener)
 
    fs, ch1_data = wavfile.read(
        pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
    )
    _, ch2_data = wavfile.read(
        pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH2.wav"
    )
    _, ch3_data = wavfile.read(
        pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH3.wav"
    )
     
    # ch1_data = (ch1_data / 32768.0).astype(np.float64)
    # ch2_data = (ch2_data / 32768.0).astype(np.float64)
    # ch3_data = (ch3_data / 32768.0).astype(np.float64)
    
    denoised_ch1 = nr.reduce_noise(y=ch1_data, sr=fs)
    denoised_ch2 = nr.reduce_noise(y=ch2_data, sr=fs)
    denoised_ch3 = nr.reduce_noise(y=ch3_data, sr=fs)
    
   
   
    # Combine the denoised signals
    combined_signal = denoised_ch1 + denoised_ch2 + denoised_ch3
    
    # combined_signal= (combined_signal * 10).astype(np.float32)

    # # Convert to 32-bit floating point scaled between -1 and 1
    # signal0 = (ch0_data / 32768.0).astype(np.float32)
    # signal1 = (ch1_data / 32768.0).astype(np.float32)
    # signal2 = (ch2_data / 32768.0).astype(np.float32)
    # signal3 = (ch3_data / 32768.0).astype(np.float32)
    

    # # Audiograms can read like this, but they are not needed for the baseline
    
    # cfs = np.array(listener_audiograms[listener]["audiogram_cfs"])
    
    # audiogram_left = np.array(
    #    listener_audiograms[listener]["audiogram_levels_l"]
    # )
    # audiogram_right = np.array(
    #    listener_audiograms[listener]["audiogram_levels_r"]
    # )
    
    # # Baseline just reads the signal from the front microphone pair
    # # and write it out as the enhanced signal
    # #

    # wavfile.write(
    #     f"../../{scene}_{listener}_enhanced.wav", fs, combined_signal
    # )
    
    # data = [
    #     ch1_data, 
    #     ch2_data, 
    #     ch3_data, 
    #     denoised_ch1, 
    #     denoised_ch2, 
    #     denoised_ch3, 
    #     combined_signal
    #     ]
    # plot_wav(fs, data)


if __name__ == "__main__":
    enhance()


import json
import logging
import pathlib

import hydra
import numpy as np
from evaluate import make_scene_listener_list
from omegaconf import DictConfig
from scipy.io import wavfile
from tqdm import tqdm

from scipy.signal import stft, istft

logger = logging.getLogger(__name__)


