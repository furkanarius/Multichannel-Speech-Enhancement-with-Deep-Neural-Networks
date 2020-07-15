from scipy.io import wavfile
from pesq import pesq
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import scipy
import os

EPS = np.finfo("float").eps
def _resample_window_oct(p, q):
    """Port of Octave code to Python"""

    gcd = np.gcd(p, q)
    if gcd > 1:
        p /= gcd
        q /= gcd

    # Properties of the antialiasing filter
    log10_rejection = -3.0
    stopband_cutoff_f = 1. / (2 * max(p, q))
    roll_off_width = stopband_cutoff_f / 10

    # Determine filter length
    rejection_dB = -20 * log10_rejection
    L = np.ceil((rejection_dB - 8) / (28.714 * roll_off_width))

    # Ideal sinc filter
    t = np.arange(-L, L + 1)
    ideal_filter = 2 * p * stopband_cutoff_f \
        * np.sinc(2 * stopband_cutoff_f * t)

    # Determine parameter of Kaiser window
    if (rejection_dB >= 21) and (rejection_dB <= 50):
        beta = 0.5842 * (rejection_dB - 21)**0.4 \
            + 0.07886 * (rejection_dB - 21)
    elif rejection_dB > 50:
        beta = 0.1102 * (rejection_dB - 8.7)
    else:
        beta = 0.0

    # Apodize ideal filter response
    h = np.kaiser(2 * L + 1, beta) * ideal_filter

    return h
def resample_oct(x, p, q):
    """Resampler that is compatible with Octave"""
    h = _resample_window_oct(p, q)
    window = h / np.sum(h)
    return resample_poly(x, p, q, window=window)
def thirdoct(fs, nfft, num_bands, min_freq):
    """ Returns the 1/3 octave band matrix and its center frequencies
    # Arguments :
        fs : sampling rate
        nfft : FFT size
        num_bands : number of 1/3 octave bands
        min_freq : center frequency of the lowest 1/3 octave band
    # Returns :
        obm : Octave Band Matrix
        cf : center frequencies
    """
    f = np.linspace(0, fs, nfft + 1)
    f = f[:int(nfft/2) + 1]
    k = np.array(range(num_bands)).astype(float)
    cf = np.power(2. ** (1. / 3), k) * min_freq
    freq_low = min_freq * np.power(2., (2 * k -1 ) / 6)
    freq_high = min_freq * np.power(2., (2 * k + 1) / 6)
    obm = np.zeros((num_bands, len(f))) # a verifier

    for i in range(len(cf)):
        # Match 1/3 oct band freq with fft frequency bin
        f_bin = np.argmin(np.square(f - freq_low[i]))
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = np.argmin(np.square(f - freq_high[i]))
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        # Assign to the octave band matrix
        obm[i, fl_ii:fh_ii] = 1
    return obm, cf
def stft(x, win_size, fft_size, overlap=4):
    """ Short-time Fourier transform for real 1-D inputs
    # Arguments
        x : 1D array, the waveform
        win_size : integer, the size of the window and the signal frames
        fft_size : integer, the size of the fft in samples (zero-padding or not)
        overlap: integer, number of steps to make in fftsize
    # Returns
        stft_out : 2D complex array, the STFT of x.
    """
    hop = int(win_size / overlap)
    w = scipy.hanning(win_size + 2)[1: -1]  # = matlab.hanning(win_size)
    stft_out = np.array([np.fft.rfft(w * x[i:i + win_size], n=fft_size)
                        for i in range(0, len(x) - win_size, hop)])
    return stft_out
def remove_silent_frames(x, y, dyn_range, framelen, hop):
    """ Remove silent frames of x and y based on x
    A frame is excluded if its energy is lower than max(energy) - dyn_range
    The frame exclusion is based solely on x, the clean speech signal
    # Arguments :
        x : array, original speech wav file
        y : array, denoised speech wav file
        dyn_range : Energy range to determine which frame is silent
        framelen : Window size for energy evaluation
        hop : Hop size for energy evaluation
    # Returns :
        x without the silent frames
        y without the silent frames (aligned to x)
    """
    # Compute Mask
    w = scipy.hanning(framelen + 2)[1:-1]

    x_frames = np.array(
        [w * x[i:i + framelen] for i in range(0, len(x) - framelen, hop)])
    y_frames = np.array(
        [w * y[i:i + framelen] for i in range(0, len(x) - framelen, hop)])

    # Compute energies in dB
    x_energies = 20 * np.log10(np.linalg.norm(x_frames, axis=1) + EPS)

    # Find boolean mask of energies lower than dynamic_range dB
    # with respect to maximum clean speech energy frame
    mask = (np.max(x_energies) - dyn_range - x_energies) < 0

    # Remove silent frames by masking
    x_frames = x_frames[mask]
    y_frames = y_frames[mask]

    # init zero arrays to hold x, y with silent frames removed
    n_sil = (len(x_frames) - 1) * hop + framelen
    x_sil = np.zeros(n_sil)
    y_sil = np.zeros(n_sil)

    for i in range(x_frames.shape[0]):
        x_sil[range(i * hop, i * hop + framelen)] += x_frames[i, :]
        y_sil[range(i * hop, i * hop + framelen)] += y_frames[i, :]

    return x_sil, y_sil
def vect_two_norm(x, axis=-1):
    """ Returns an array of vectors of norms of the rows of matrices from 3D array """
    return np.sum(np.square(x), axis=axis, keepdims=True)
def row_col_normalize(x):
    """ Row and column mean and variance normalize an array of 2D segments """
    # Row mean and variance normalization
    x_normed = x + EPS * np.random.standard_normal(x.shape)
    x_normed -= np.mean(x_normed, axis=-1, keepdims=True)
    x_inv = 1. / np.sqrt(vect_two_norm(x_normed))
    x_diags = np.array(
        [np.diag(x_inv[i].reshape(-1)) for i in range(x_inv.shape[0])])
    x_normed = np.matmul(x_diags, x_normed)
    # Column mean and variance normalization
    x_normed += + EPS * np.random.standard_normal(x_normed.shape)
    x_normed -= np.mean(x_normed, axis=1, keepdims=True)
    x_inv = 1. / np.sqrt(vect_two_norm(x_normed, axis=1))
    x_diags = np.array(
        [np.diag(x_inv[i].reshape(-1)) for i in range(x_inv.shape[0])])
    x_normed = np.matmul(x_normed, x_diags)
    return x_normed
# Constant definition
FS = 10000                          # Sampling frequency
N_FRAME = 256                       # Window support
NFFT = 512                          # FFT Size
NUMBAND = 15                        # Number of 13 octave band
MINFREQ = 150                       # Center frequency of 1st octave band (Hz)
OBM, CF = thirdoct(FS, NFFT, NUMBAND, MINFREQ)  # Get 1/3 octave band matrix
N = 30                              # N. frames for intermediate intelligibility
BETA = -15.                         # Lower SDR bound
DYN_RANGE = 40                      # Speech dynamic range
def stoi(x, y, fs_sig, extended=False):
    """ Short term objective intelligibility
    Computes the STOI (See [1][2]) of a denoised signal compared to a
    clean signal, The output is expected to have a monotonic
    relation with the subjective speech-intelligibility, where a higher d
    denotes better speech intelligibility
    # Arguments
        x : clean original speech
        y : denoised speech
        fs_sig : sampling rate of x and y
        extended : Boolean, whether to use the extended STOI described in [3]
    # Returns
        Short time objective intelligibility measure between clean and denoised
        speech
    # Raises
        AssertionError : if x and y have different lengths
    # Reference
        [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
            Objective Intelligibility Measure for Time-Frequency Weighted Noisy
            Speech', ICASSP 2010, Texas, Dallas.
        [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
            Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
            IEEE Transactions on Audio, Speech, and Language Processing, 2011.
        [3] Jesper Jensen and Cees H. Taal, 'An Algorithm for Predicting the
            Intelligibility of Speech Masked by Modulated Noise Maskers',
            IEEE Transactions on Audio, Speech and Language Processing, 2016.
    """
    if x.shape != y.shape:
        raise Exception('x and y should have the same length,' +
                        'found {} and {}'.format(x.shape, y.shape))

    # Resample is fs_sig is different than fs
    if fs_sig != FS:
        x = resample_oct(x, FS, fs_sig)
        y = resample_oct(y, FS, fs_sig)

    # Remove silent frames
    x, y = remove_silent_frames(x, y, DYN_RANGE, N_FRAME, int(N_FRAME/2))

    # Take STFT
    x_spec = stft(x, N_FRAME, NFFT, overlap=2).transpose()
    y_spec = stft(y, N_FRAME, NFFT, overlap=2).transpose()

    # Apply OB matrix to the spectrograms as in Eq. (1)
    x_tob = np.sqrt(np.matmul(OBM, np.square(np.abs(x_spec))))
    y_tob = np.sqrt(np.matmul(OBM, np.square(np.abs(y_spec))))

    # Take segments of x_tob, y_tob
    x_segments = np.array(
        [x_tob[:, m - N:m] for m in range(N, x_tob.shape[1] + 1)])
    y_segments = np.array(
        [y_tob[:, m - N:m] for m in range(N, x_tob.shape[1] + 1)])

    if extended:
        x_n = row_col_normalize(x_segments)
        y_n = row_col_normalize(y_segments)
        return np.sum(x_n * y_n / N) / x_n.shape[0]

    else:
        # Find normalization constants and normalize
        normalization_consts = (
            np.linalg.norm(x_segments, axis=2, keepdims=True) /
            (np.linalg.norm(y_segments, axis=2, keepdims=True) + EPS))
        y_segments_normalized = y_segments * normalization_consts

        # Clip as described in [1]
        clip_value = 10 ** (-BETA / 20)
        y_primes = np.minimum(
            y_segments_normalized, x_segments * (1 + clip_value))

        # Subtract mean vectors
        y_primes = y_primes - np.mean(y_primes, axis=2, keepdims=True)
        x_segments = x_segments - np.mean(x_segments, axis=2, keepdims=True)

        # Divide by their norms
        y_primes /= (np.linalg.norm(y_primes, axis=2, keepdims=True) + EPS)
        x_segments /= (np.linalg.norm(x_segments, axis=2, keepdims=True) + EPS)
        # Find a matrix with entries summing to sum of correlations of vectors
        correlations_components = y_primes * x_segments

        # J, M as in [1], eq.6
        J = x_segments.shape[0]
        M = x_segments.shape[1]

        # Find the mean of all correlations
        d = np.sum(correlations_components) / (J * M)
        return d
#
def evaluate_pesq(cleaned, noisy, clean, fs=8000,):
  noisy_score   = pesq(fs, clean, noisy, 'nb')
  cleaned_score = pesq(fs, clean[:len(cleaned)], cleaned, 'nb')
  return [noisy_score, cleaned_score]

def evaluate_stoi(cleaned, noisy, clean, fs=8000,):
  noisy_score   = stoi(clean, noisy, fs, extended=False)
  cleaned_score = stoi(clean[:len(cleaned)], cleaned, fs, extended=False)
  return [noisy_score, cleaned_score]

import pandas as pd
data_folder = '/Users/furkan/Downloads/ThesisData/ALL_DATA/'
dataset_folder = data_folder + 'FullMixture_5k_4Channel_Fixed_TEST/'
cleaned_folder = data_folder + '5k_4Channel_GSC_TEST/'
all_sound_files = os.listdir(dataset_folder)
mixture_files = [i for i in all_sound_files if "_" in i and "wav" in i]
clean_files   = [i for i in all_sound_files if not "_" in i]
noisy_stoi_score = []; noisy_pesq_score = []; mixture_name = []
cleaned_stoi_score = []; cleaned_pesq_score = []
for clean_audio in clean_files:
    mixture_number = '_' + clean_audio.split('.')[0][5:] + '_'
    if any(mixture_number in x  for x in mixture_files):
        noisy_name = [item for i, item in enumerate(mixture_files) if mixture_number in item][0]
    else:
        continue
    mixture_name.append(noisy_name)
    # Evaluating the signals
    path_clean   = dataset_folder +  clean_audio
    path_noisy   = dataset_folder +  noisy_name
    path_cleaned = cleaned_folder + 'cleaned_' +noisy_name
    clean, fs    = librosa.load(path_clean,   mono=True, sr=8000)
    noisy, fs    = librosa.load(path_noisy,   mono=True, sr=8000)
    cleaned, fs  = librosa.load(path_cleaned, mono=True, sr=8000)
    # Score Evaluation
    pesq_score   = evaluate_pesq(cleaned, noisy, clean)
    stoi_score   = evaluate_stoi(cleaned, noisy, clean)
    # Append Noisy Scores
    noisy_stoi_score.append(stoi_score[0])
    noisy_pesq_score.append(pesq_score[0])
    # Append Cleaned Scores
    cleaned_stoi_score.append(stoi_score[1])
    cleaned_pesq_score.append(pesq_score[1])

scores = {'Mixture Name': mixture_name,
         'Noisy STOI Score': noisy_stoi_score,
         'Cleaned STOI Score': cleaned_stoi_score,
         'Noisy PESQ Score': noisy_pesq_score,
         'Cleaned PESQ Score': cleaned_pesq_score
        }
df = pd.DataFrame(scores, columns= ['Mixture Name', 'Noisy STOI Score','Cleaned STOI Score','Noisy PESQ Score','Cleaned PESQ Score'])

df.to_csv(data_folder+'scores_GSC.csv', index = False, header=True)

