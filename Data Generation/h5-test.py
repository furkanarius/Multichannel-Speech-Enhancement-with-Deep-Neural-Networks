from __future__ import print_function
import numpy as np
from scipy.io import wavfile
import scipy
import os
import h5py
import librosa
import matplotlib.pyplot as plt
import librosa.display
# furkan = np.random.randint(20,size=(3,2,5))
# with h5py.File('timit_nodelay.hdf5', 'w') as hf:
#     hf.create_dataset('furkan', data=furkan, compression="gzip", compression_opts=9,
#                       chunks=True, maxshape=(3, 2, None))
#
# yusuf = np.random.randint(20,size=(3,2,8))
# with h5py.File('timit_nodelay.hdf5', 'a') as hf:
#     print(hf["furkan"].shape[2])
#     hf["furkan"].resize((hf["furkan"].shape[2] + yusuf.shape[2]), axis=2)
#     hf["furkan"][:,:,-yusuf.shape[2]:] = yusuf

with h5py.File('/Users/furkan/PycharmProjects/thesis/FurkanTest/fullmixture_5k_2channels_large_phase.hdf5', 'r') as hf:
    clear_train = hf.get('clear_timit_train')
    mixture_train = hf.get('mixture_timit_train')

    clear_train = np.array(clear_train)
    mixture_train = np.array(mixture_train)
print(clear_train.shape)
print(mixture_train.shape)

# with h5py.File('fullmixture_5k_4channels_large_p1.hdf5', 'w') as hf:
#     hf.create_dataset('clear_timit_train', data=clear_train[0:193000,:], compression="gzip", chunks=True,
#                       maxshape=(None, 129))
#     hf.create_dataset('mixture_timit_train', data=mixture_train[0:193000,:,:,:], compression="gzip", chunks=True,
#                       maxshape=(None, 129, 8, 4))
# #
# with h5py.File('fullmixture_5k_4channels_large_p2.hdf5', 'w') as hf:
#     hf.create_dataset('clear_timit_train', data=clear_train[193000:,:], compression="gzip", chunks=True,
#                       maxshape=(None, 129))
#     hf.create_dataset('mixture_timit_train', data=mixture_train[193000:,:,:,:], compression="gzip", chunks=True,
#                       maxshape=(None, 129, 8, 4))

a = librosa.amplitude_to_db(np.transpose(mixture_train[1:200,:,0,0]),ref=np.max)
b = librosa.amplitude_to_db(np.transpose(clear_train[1:200, :]), ref=np.max)
plt.subplot(2, 2, 1)
librosa.display.specshow(a, hop_length=192,sr=8000, y_axis='log', x_axis='time')
plt.subplot(2, 2, 2)
plt.colorbar(format='%+2.0f dB')
librosa.display.specshow(b, hop_length=192,sr=8000, y_axis='log', x_axis='time')
plt.show()

# x = librosa.amplitude_to_db(mixture_train[7,:,:,0],ref=np.max)
# y = librosa.amplitude_to_db(np.transpose(clear_train[8:51, :]), ref=np.max)
# plt.subplot(2, 2, 1)
# librosa.display.specshow(x, hop_length=192,sr=8000, y_axis='log', x_axis='time')
# plt.subplot(2, 2, 2)
# plt.colorbar(format='%+2.0f dB')
# librosa.display.specshow(y, hop_length=192,sr=8000, y_axis='log', x_axis='time')
# plt.show()
