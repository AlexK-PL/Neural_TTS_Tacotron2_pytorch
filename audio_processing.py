import torch
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util
import librosa
import scipy.io.wavfile

from hyper_parameters import tacotron_params
import matplotlib.pyplot as plt


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles).double())
    signal = stft_fn.inverse(magnitudes.double(), angles.double()).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


# NEW FUNCTIONS FOR MODEL IMPROVEMENTS:
def melspectrogram(hparams, y):
    D = _stft(hparams, y)
    S = _amp_to_db(hparams, _linear_to_mel(hparams,np.abs(D)))

    return _normalize(hparams, S)
    # return S


def _stft(hparams, y):
    return librosa.stft(y=y, n_fft=hparams["fft_size"],
                        hop_length=hparams["hop_length"],
                        win_length=hparams["win_length"],
                        pad_mode=hparams["stft_pad_mode"], )


def _amp_to_db(hparams, x):
    return hparams["spec_gain"] * np.log10(np.maximum(1e-5, x))


def _linear_to_mel(hparams, spectrogram):
    mel_basis = _build_mel_basis(hparams)
    return np.dot(mel_basis, spectrogram)


def _normalize(hparams, S):
    S = S.copy()
    if hparams["signal_norm"]:
        '''
        if hasattr(hparams, "mel_scaler"):
            if S.shape[0] == hparams["num_mels"]:

                return self.mel_scaler.transform(S.T).T
            elif S.shape[0] == hparams["fft_size"] / 2:

                return self.linear_scaler.transform(S.T).T
            else:
                raise RuntimeError(" [!] Mean - Var stats do not match the given feature dimensions. ")
        '''
        # DESCARTO ESTA RESTA PORQUE ENTONCES EL VALOR MÁS BAJO YA NO ES 100 SINO 120...
        # S -= hparams["ref_level_db"]  # discard certain range of DB assuming it is air noise
        S_norm = ((S - hparams["min_level_db"]) / (-hparams["min_level_db"]))
        if hparams["symmetric_norm"]:
            S_norm = ((2 * hparams["max_norm"]) * S_norm) - hparams["max_norm"]
            if hparams["clip_norm"]:
                S_norm = np.clip(S_norm, -hparams["max_norm"], hparams["max_norm"])

                return S_norm
        else:
            S_norm = hparams["max_norm"] * S_norm
            if hparams["clip_norm"]:
                S_norm = np.clip(S_norm, 0, hparams["max_norm"])

            return S_norm
    else:
        return S


def _denormalize(hparams, S):
    """denormalize values"""
    # pylint: disable=no-else-return
    S_denorm = S.copy()
    if hparams["signal_norm"]:
        # mean-var scaling
        # if hasattr(self, 'mel_scaler'):
        #     if S_denorm.shape[0] == self.num_mels:
        #         return self.mel_scaler.inverse_transform(S_denorm.T).T
        #     elif S_denorm.shape[0] == self.fft_size / 2:
        #         return self.linear_scaler.inverse_transform(S_denorm.T).T
        #     else:
        #         raise RuntimeError(' [!] Mean-Var stats does not match the given feature dimensions.')
        if hparams["symmetric_norm"]:
            if hparams["clip_norm"]:
                S_denorm = np.clip(S_denorm, -hparams["max_norm"],
                                   hparams["max_norm"])  # pylint: disable=invalid-unary-operand-type
            # + hparams["min_level_db"]
            S_denorm = ((S_denorm + hparams["max_norm"]) * -hparams["min_level_db"] / (2 * hparams["max_norm"]))/(-100)
            return S_denorm   # + hparams["ref_level_db"]
        else:
            if hparams["clip_norm"]:
                S_denorm = np.clip(S_denorm, 0, hparams["max_norm"])
            S_denorm = (S_denorm * -hparams["min_level_db"] /
                        hparams["max_norm"]) + hparams["min_level_db"]
            return S_denorm + hparams["ref_level_db"]
    else:
        return S_denorm


#  To construct self.mel_basis
def _build_mel_basis(hparams, ):
    if hparams["mel_fmax"] is not None:
        assert hparams["mel_fmax"] <= hparams["sampling_rate"] // 2

    return librosa.filters.mel(hparams["sampling_rate"],
                               hparams["fft_size"],
                               n_mels=hparams["n_mel_channels"],
                               fmin=hparams["mel_fmin"],
                               fmax=hparams["mel_fmax"])

@staticmethod
def sound_norm(x):
    return x / abs(x).max() * 0.9


# Save and load wav
def load_wav(hparams, filename, sr=None):
    if sr is None:
        # x, sr = sf.read(filename)
        x, sr = librosa.load(filename)
    else:
        x, sr = librosa.load(filename, sr=sr)
    assert hparams["sampling_rate"] == sr, "%s vs %s" % (hparams["sample_rate"], sr)
    if hparams["do_sound_norm"]:
        x = sound_norm(x)

    return x


def save_wav(hparams, wav, path):
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    scipy.io.wavfile.write(path, hparams["sampling_rate"], wav_norm.astype(np.in16))


if __name__ == "__main__":

    wav_path = "C:/Users/AlexPL/Desktop/LJ001-0001.wav"
    hparams = tacotron_params

    wav = load_wav(hparams, wav_path)
    # print(wav)
    S = melspectrogram(hparams, wav)

    print(np.min(S))
    print(np.max(S))

    S_denorm = _denormalize(hparams, S)

    print(np.min(S_denorm))
    print(np.max(S_denorm))

    S = np.flipud(S)
    plt.imshow(S)
    plt.show()

