'''
4 channel, single speech, single noise STFT generator code
'''
from __future__ import print_function
import numpy as np
from scipy.io import wavfile
import scipy
import os
import librosa
import h5py
import time
from tqdm import tqdm

def stft_generator(filename,noisyAudio,fs=8000):
    FrameSize = 256  # 512 under 16KHz time=32ms(normally 20~30ms )
    Overlap = round(0.75 * FrameSize)
    FFTSize = FrameSize  # FFT window size=FRAMESIZE
    FrequencyBins = FrameSize // 2 + 1  # stft_matrix:np.ndarray [shape=(1 + n_fft/2, t)]
    NumSegments = 8

    # Reading the audio data
    audio_data, sr = librosa.load(filename, mono=False, sr=fs)  # if sr=None to read raw sample_rate

    if noisyAudio == False:
        stft_segments= librosa.stft(audio_data, n_fft=FrameSize, hop_length=Overlap, win_length=FFTSize,
                                              window=scipy.signal.hamming(FrameSize,sym=False))
        # Phase
        audio_phase = np.angle(stft_segments)
        # Magnitude matrix
        stft_segments = np.transpose(abs(stft_segments))
    else:
        counter = 0
        for audio in audio_data:
            channelNumber = 2
            audio = np.nan_to_num(audio)
            noisySTFT = librosa.stft(audio, n_fft=FrameSize, hop_length=Overlap, win_length=FFTSize,
                                         window=scipy.signal.hamming(FrameSize,sym=False))
            audio_phase = np.angle(noisySTFT)
            # Magnitude matrix
            noisySTFT =np.abs(noisySTFT)
            new_noisy_STFT = np.concatenate((noisySTFT[:,0:NumSegments-1], noisySTFT), axis=1)
            # initialize STFT 4 channel structure
            if counter == 0:
                stft_segments = np.zeros((new_noisy_STFT.shape[1] - NumSegments + 1, FrequencyBins, NumSegments,channelNumber))

            for index in range(0, new_noisy_STFT.shape[1] - NumSegments + 1):
                stft_segments[index,:,:,counter] = new_noisy_STFT[:, index:index+NumSegments]
            counter = counter + 1

    return {'stft': stft_segments, 'phase': audio_phase}

def convert_to_audio(outname,stft_magnitude,stft_phase,fs=8000,):
    # Converting the data back to human language
    cleanStd  = 0.8258655
    cleanMean = 0.2438803
    noisyMean = 0.075
    noisyStd  = 0.551
    FrameSize = 256  # 512 under 16KHz time=32ms(normally 20~30ms )
    Overlap   = round(0.75 * FrameSize)
    FFTSize   = FrameSize  # FFT window size=FRAMESIZE
    FrequencyBins = FrameSize // 2 + 1  # stft_matrix:np.ndarray [shape=(1 + n_fft/2, t)]
    NumSegments = 8
    stft_magnitude_nonNormal = cleanStd*stft_magnitude+cleanMean
    final_STFT = stft_magnitude_nonNormal*np.exp(1j*stft_phase)
    # converted_audio = []
    # for i in range(stft_segments.shape[2]):
    #     converted_audio.extend(librosa.istft(stft_segments[:, :, i], hop_length=Overlap, win_length=FFTSize,
    #                                          window=scipy.signal.hamming))
    # Audio Converter
    converted_audio = librosa.istft(final_STFT, hop_length=Overlap, win_length=FFTSize,
                  window=scipy.signal.hamming(FrameSize,sym=False))
    wavfile.write(outname, rate=fs, data=converted_audio)

# Sound Input
folder = '/Users/furkan/Downloads/ThesisData/ALL_DATA/FullMixture_5k_4Channel_Fixed/'
all_sound_files = os.listdir(folder)
mixture_files = [i for i in all_sound_files if "_" in i]
clear_files = [i for i in all_sound_files if not "_" in i]

counter = 0
for clear_audio in tqdm(clear_files[0:10000]):
    mixture_number = '_' + clear_audio.split('.')[0][5:] + '_'
    if any(mixture_number in x  for x in mixture_files):
        mixture_fullname = folder + [item for i, item in enumerate(mixture_files) if mixture_number in item][0]
    else:
        continue
    clear_audio_fullname = folder + clear_audio

    if counter == 0:
        # 1st STFT arrays have been generated
        all_mixture_files_array = stft_generator(mixture_fullname, noisyAudio=True)['stft']
        # convert_to_audio('noise.wav',all_mixture_files_array,0.2) # STFT to audio
        all_clear_files_array = stft_generator(clear_audio_fullname, noisyAudio=False)['stft']
        # convert_to_audio('clean.wav',all_clear_files_array,0.2) # STFT to audio
        with h5py.File('fullmixture_5k_2channels_large.hdf5', 'w') as hf:
            hf.create_dataset('clear_timit_train', data=all_clear_files_array, compression="gzip", chunks=True,
                              maxshape=(None, 129))
            hf.create_dataset('mixture_timit_train', data=all_mixture_files_array, compression="gzip", chunks=True,
                              maxshape=(None, 129, 8, 4))
    # TRAINING DATASET CREATION
    elif counter > 0 and counter < 10000:
        # Concatenate operation
        if counter%200 == 0 and counter > 200:
            print(str(counter) + ' - ' + clear_audio_fullname)
            # ARRAY APPEND
            with h5py.File('fullmixture_5k_2channels_large.hdf5', 'a') as hf:
                # Append the array to clear training dataset
                hf["clear_timit_train"].resize((hf["clear_timit_train"].shape[0] + all_clear_files_array.shape[0]),axis=0)
                hf["clear_timit_train"][-all_clear_files_array.shape[0]:, :] = all_clear_files_array

                # Append the array to mixture training dataset
                hf["mixture_timit_train"].resize((hf["mixture_timit_train"].shape[0] + all_mixture_files_array.shape[0]), axis=0)
                hf["mixture_timit_train"][-all_mixture_files_array.shape[0]:, :, :,:] = all_mixture_files_array
            # Reset the array
            all_mixture_files_array = stft_generator(mixture_fullname, noisyAudio=True)['stft']
            all_clear_files_array = stft_generator(clear_audio_fullname, noisyAudio=False)['stft']

        elif counter == 200:
            with h5py.File('fullmixture_5k_2channels_large.hdf5', 'w') as hf:
                hf.create_dataset('clear_timit_train', data=all_clear_files_array, compression="gzip", chunks=True, maxshape=(None,129))
                hf.create_dataset('mixture_timit_train', data=all_mixture_files_array, compression="gzip", chunks=True, maxshape=(None,129, 8, 4))
        else:
            clean_stft = stft_generator(clear_audio_fullname, noisyAudio=False)['stft']
            noisy_stft = stft_generator(mixture_fullname, noisyAudio=True)['stft']
            if clean_stft.shape[0] != noisy_stft.shape[0]:
                print("OOPSIE")
                continue
            all_clear_files_array = np.concatenate([all_clear_files_array, clean_stft],axis=0)
            all_mixture_files_array = np.concatenate([all_mixture_files_array, noisy_stft],axis=0)
        if all_mixture_files_array.shape[0] != all_clear_files_array.shape[0]:
            print("ERROR")
            break
    counter += 1