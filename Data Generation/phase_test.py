from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import librosa

counter = 0
FrameSize = 256  # 512 under 16KHz time=32ms(normally 20~30ms )
Overlap = round(0.75 * FrameSize)
FFTSize = FrameSize  # FFT window size=FRAMESIZE
FrequencyBins = FrameSize // 2 + 1  # stft_matrix:np.ndarray [shape=(1 + n_fft/2, t)]
NumSegments = 8
# noisyMean = 0.34470737
# noisyStd  = 0.8982022
cleanStd  = 0.62
cleanMean = 0.25

def convert_to_audio(outname,stft_magnitude,stft_phase,noisyAudio,fs=8000,):
    # Normal Case:
    # noisyMean = 0.34470737
    # noisyStd  = 0.8982022
    cleanStd  = 0.62
    cleanMean = 0.25

    # Converting the data back to human language
    FrameSize = 256  # 512 under 16KHz time=32ms(normally 20~30ms )
    Overlap   = round(0.75 * FrameSize)
    FFTSize   = FrameSize  # FFT window size=FRAMESIZE
    FrequencyBins = FrameSize // 2 + 1  # stft_matrix:np.ndarray [shape=(1 + n_fft/2, t)]
    NumSegments = 8
    if noisyAudio == True:
      stft_magnitude_nonNormal = noisyStd*stft_magnitude+noisyMean
    else:
      stft_magnitude_nonNormal = cleanStd*stft_magnitude+cleanMean
    final_STFT = stft_magnitude_nonNormal*np.exp(1j*stft_phase)
    # Audio Converter
    converted_audio = librosa.istft(final_STFT, hop_length=Overlap, win_length=FFTSize,
                  window=scipy.signal.hamming(FrameSize,sym=False))
    wavfile.write(outname, rate=fs, data=converted_audio)

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
        noisySTFT = librosa.stft(audio_data, n_fft=FrameSize, hop_length=Overlap, win_length=FFTSize,
                                     window=scipy.signal.hamming(FrameSize,sym=False))
        audio_phase = np.angle(noisySTFT)
        # Magnitude matrix
        noisySTFT =np.abs(noisySTFT)
        new_noisy_STFT = np.concatenate((noisySTFT[:,0:NumSegments-1], noisySTFT), axis=1)
        stft_segments = np.zeros((new_noisy_STFT.shape[1] - NumSegments + 1, FrequencyBins, NumSegments))
        for index in range(0, new_noisy_STFT.shape[1] - NumSegments + 1):
            stft_segments[index,:,:] = new_noisy_STFT[:, index:index+NumSegments]

    return {'stft': stft_segments, 'phase': audio_phase}
