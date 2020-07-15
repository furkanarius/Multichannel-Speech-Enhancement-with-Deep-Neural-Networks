from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy
import os, random
import librosa
import pyloudnorm

def sanity_check(sound):
    return not('flac' in sound.lower() or 'wav' in sound.lower())

## Sound Input
folder_input = "DATASET_PATH_1"
f_folder_input = 'DATASET_PATH_2'
t_folder_input = 'DATASET_PATH_3'
lj_folder_input = 'DATASET_PATH_4'
noise_folder = 'NOISE_DATASET_1'

# Important to change
folder = 'OUTPUT_PATH'

# change based on needs
fs = 16000
m = 0; ls = 0; lj = 0 ; f = 0; timit = 1
counter = 0
for i in range(5000):
    counter = counter + 1

    # Sound Input
    if ls == 1:
        source_random = random.choice(os.listdir(folder_input))
        if sanity_check(source_random):
            continue
        source_random = folder_input + source_random

    elif f == 1:
        source_random = random.choice(os.listdir(f_folder_input))
        if sanity_check(source_random):
            continue
        source_random = f_folder_input + source_random

    elif timit == 1:
        source_random = random.choice(os.listdir(t_folder_input))
        if sanity_check(source_random):
            continue
        source_random = t_folder_input + source_random

    elif lj == 1:
        source_random = random.choice(os.listdir(lj_folder_input))
        if sanity_check(source_random):
            continue
        source_random = lj_folder_input + source_random

    # Source Read
    audio, _ = librosa.load(source_random, mono=True, sr=fs)
    temp_audio = pyloudnorm.normalize.peak(audio, -1.0)
    wavfile.write(folder + 'timit' + str(counter) +'.wav', rate=fs, data=temp_audio)

    # Noise Read

    ## Noise Input
    noise_random = random.choice(os.listdir(noise_folder))
    noise_random = noise_folder + noise_random
    if sanity_check(noise_random):
        continue

    ## Random Noise Way
    # noise, _ = librosa.load(noise_random, mono=True, sr=fs)
    # Removing Randomness

    noise, _ = librosa.load(noise_random, mono=True, sr=fs)
    amplitude_levels = [-1,-4,-7,-10]
    noise_level = random.choice(amplitude_levels)
    temp_noise = pyloudnorm.normalize.peak(noise, noise_level)

    # Repeat noise to the same length as speech.
    if len(temp_noise) < len(temp_audio):
        n_repeat = int(np.ceil(float(len(temp_audio)) / float(len(temp_noise))))
        noise_audio_ex = np.tile(temp_noise, n_repeat)
        temp_noise = noise_audio_ex[0: len(temp_audio)]
    elif len(temp_audio) < len(temp_noise):
        temp_noise = temp_noise[0: len(temp_audio)]
        # Truncate noise to the same length as speech.
    else:
        # noise_audio = noise_audio[noise_onset : noise_offset]
        temp_noise = temp_noise
        temp_audio = temp_audio

    # Mixing
    mixed_audio = temp_audio + temp_noise
    mixed_audio = mixed_audio - np.mean(mixed_audio)

    name = 'timit' + str(counter) + '_n' + str(amplitude_levels.index(noise_level)) + '.wav'
    final_file = folder + name
    wavfile.write(final_file, rate=fs, data=mixed_audio)
