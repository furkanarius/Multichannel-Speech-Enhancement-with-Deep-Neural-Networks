'''
This code generates wav files with 4 channels.
It contains noise and audio with randomly selected locations
for room, noise and speech source.
'''
import time
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import scipy
import os, random
import librosa
import pyloudnorm

# Arranging the files to be processed
fs = 8000
speech_signal_filename = '/Users/furkan/Downloads/ThesisData/speech_commands_v0.01/sheila/1a073312_nohash_0.wav'
speech_signal, _ = librosa.load(speech_signal_filename, mono=True, sr=fs)
speech_signal = pyloudnorm.normalize.peak(speech_signal, -1.0)

noise_signal_filename = 'noise.wav'
noise_signal, _ = librosa.load(noise_signal_filename, mono=True, sr=fs)
noise_signal_trimmed = noise_signal[:len(speech_signal)]
for i in range(0,50):
    lowSNR  = 1.5
    highSNR = 15.0
    SNR = lowSNR + (highSNR - lowSNR) * (np.random.randint(30) - 1) / (30 - 1.)
    noise_signal_trimmed = pyloudnorm.normalize.peak(noise_signal_trimmed, -SNR)

    # Converting to int16
    speech_signal_int  = speech_signal * 32767.0
    noise_signal_int   = noise_signal_trimmed  * 32767.0
    noise_signal_int   = noise_signal_int.astype(np.int16)
    speech_signal_int  = speech_signal_int.astype(np.int16)

    # Arranging the room
    # Random room and source generation
    N = 50
    x_low  = 3
    x_high = 7
    x = x_low + (x_high - x_low) * (np.random.randint(N) - 1) / (N - 1.)
    y = x_low + (x_high - x_low) * (np.random.randint(N) - 1) / (N - 1.)
    room = pra.ShoeBox([x, y, 3.], fs=8000, absorption=0.2, max_order=17)

    # Adding sources
    xy_source_low = 0.5
    x = x - 0.1
    y = y - 0.1
    s_xloc = xy_source_low + (x - xy_source_low) * (np.random.randint(N) - 1) / (N - 1.)
    n_xloc = xy_source_low + (x - xy_source_low) * (np.random.randint(N) - 1) / (N - 1.)

    # Adding noises
    s_yloc = xy_source_low + (y - xy_source_low) * (np.random.randint(N) - 1) / (N - 1.)
    n_yloc = xy_source_low + (y - xy_source_low) * (np.random.randint(N) - 1) / (N - 1.)

    s_zloc = abs(2.5 * (np.random.randint(N) - 1) / (N - 1.))
    n_zloc = abs(2.5 * (np.random.randint(N) - 1) / (N - 1.))
    room.add_source([s_xloc, s_yloc, s_zloc], signal=speech_signal_int)
    room.add_source([n_xloc, n_yloc, n_zloc], signal=noise_signal_int)

    s_xloc = xy_source_low + (x - xy_source_low) * (np.random.randint(N) - 1) / (N - 1.)
    s_yloc = xy_source_low + (y - xy_source_low) * (np.random.randint(N) - 1) / (N - 1.)
    s_zloc = abs(2.5 * (np.random.randint(N) - 1) / (N - 1.))
    d = 0.06
    R = np.array([[s_xloc, s_xloc + d, s_xloc, s_xloc + d], [s_yloc, s_yloc, s_yloc + d, s_yloc + d],
                  [s_zloc, s_zloc, s_zloc, s_zloc]])  # [[x], [y], [z]]

    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    # compute image sources
    room.image_source_model(use_libroom=True)
    room.simulate()
    print(room.mic_array.signals.shape)
    final_signal = np.transpose(room.mic_array.signals)
    main_path = '/Users/furkan/Downloads/ThesisData/ALL_DATA/Test_4Channel/'
    mixture_filename = main_path + 'mixture_' + str(i) + '_n' + str(int(SNR)) + '.wav'
    wavfile.write(mixture_filename, rate=fs, data=np.int16(final_signal))
