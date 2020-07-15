'''
A simple example of using pyroomacoustics to simulate
sound propagation in a shoebox room and record the result
to a wav file.
'''
from __future__ import print_function
import time
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy
import os, random
import librosa
import librosa.display

fs = 16000
FrameSize = 512# 512 under 16KHz time=32ms(normally 20~30ms )
                # 512 under 25KHz time=20ms
                # 1024 under 25KHz time=40ms
Overlap = FrameSize//2 # half of Framesize(return the integer part)
FFTSize = FrameSize #FFT window size=FRAMESIZE
Rate = fs
FrameWidth = 2
FrequencyBin = FrameSize//2+1 #stft_matrix:np.ndarray [shape=(1 + n_fft/2, t)]
_, audio = wavfile.read('/Users/furkan/PycharmProjects/thesis/FurkanTest/m1.wav')
room = pra.ShoeBox([7, 7, 2.4], absorption=0.25, fs=fs, max_order=15)
# place the source in the room
room.add_source([4.5, 4.73, 1.76], signal=audio)

d = 0.06
x = 0; y = 0
R = np.c_[
    [x, y, 1],  # mic 0
    [x+d, y, 1],  # mic 1
    [x, y+d, 1],  # mic 2
    [x+d, y+d, 1],  # mic 3
    ]
# the fs of the microphones is the same as the room
mic_array = pra.MicrophoneArray(R, room.fs)

# finally place the array in the room
room.add_microphone_array(mic_array)
# Simulation
room.simulate()
# plot signal at microphone 1
# sound, = plt.plot(audio, label='Sound')
# mic0, = plt.plot(room.mic_array.signals[0,:], label='Mic 0')
# mic1, = plt.plot(room.mic_array.signals[1,:], label='Mic 1')
# mic2, = plt.plot(room.mic_array.signals[2,:], label='Mic 2')
# mic3, = plt.plot(room.mic_array.signals[3,:], label='Mic 3')
# plt.legend(handles=[sound,mic0,mic1,mic2,mic3])
# plt.show()
#
wavfile.write('mic0-out.wav',rate=fs, data=np.int16(np.transpose(room.mic_array.signals)))
# STFT Operations - MAIN AUDIO
y, sr = librosa.load('m1.wav', mono=False, sr=fs)  # if sr=None to read raw sample_rate
D = librosa.stft(y, n_fft=FrameSize, hop_length=Overlap, win_length=FFTSize,
                 window=scipy.signal.hamming)

y1, _ = librosa.load('mic0-out.wav', mono=False, sr=fs)  # if sr=None to read raw sample_rate
D1 = librosa.stft(y1[0,:], n_fft=FrameSize, hop_length=Overlap, win_length=FFTSize,
                 window=scipy.signal.hamming)
D_in = D.shape[1]
D_out = D1.shape[1]
D_final = x = np.pad(D,((0,0),(0,D_out-D_in)), 'constant')
audioOut = librosa.istft(D_final, hop_length=Overlap, win_length=FFTSize,
                 window=scipy.signal.hamming)

audioOut1 = librosa.istft(D1, hop_length=Overlap, win_length=FFTSize,
                 window=scipy.signal.hamming)

wavfile.write('m1-istft.wav',rate=fs, data=audioOut)
wavfile.write('mic0-istft.wav',rate=fs, data=audioOut1)

# librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max),y_axis='log', x_axis='time')
# plt.title('Power spectrogram')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()
