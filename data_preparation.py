import random

import numpy as np
import torch
import torch.utils.data

import nn_layers
from scipy.io.wavfile import read
from scipy.interpolate import interp1d
from text import text_to_sequence
import audio_processing as ap
from hyper_parameters import tacotron_params

import tkinter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import math


class DataPreparation(torch.utils.data.Dataset):

    def __init__(self, audiopaths_and_text, tacotron_hyperparams):
        self.audiopaths_and_text = audiopaths_and_text
        self.audio_text_parameters = tacotron_hyperparams
        self.stft = nn_layers.TacotronSTFT(tacotron_hyperparams['filter_length'], tacotron_hyperparams['hop_length'],
                                           tacotron_hyperparams['win_length'], tacotron_hyperparams['n_mel_channels'],
                                           self.audio_text_parameters['sampling_rate'],
                                           tacotron_hyperparams['mel_fmin'], tacotron_hyperparams['mel_fmax'])
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def load_audiowav_torch(self, audiopath, samp_rate):
        # data = ap.load_wav(self.audio_text_parameters, audiopath, sr=samp_rate)
        sr, data = read(audiopath)
        assert samp_rate == sr, "Sample rate does not match with the configuration"

        return torch.FloatTensor(data.astype(np.float32))
        # return data

    def melspec_textSequence_pair(self, audiopath_and_text):
        wav_path, sentence = audiopath_and_text[0], audiopath_and_text[1]

        # wav to torch tensor
        wav_torch = self.load_audiowav_torch(wav_path, self.audio_text_parameters['sampling_rate'])
        wav_torch_norm = wav_torch / self.audio_text_parameters['max_wav_value']
        wav_torch_norm = wav_torch_norm.unsqueeze(0)
        wav_torch_norm = torch.autograd.Variable(wav_torch_norm, requires_grad=False)
        mel_spec = self.stft.mel_spectrogram(wav_torch_norm)
        mel_spec = torch.squeeze(mel_spec, 0)
        # mel_spec = torch.FloatTensor(ap.melspectrogram(self.audio_text_parameters, wav)).contiguous()
        # text to torch integer tensor sequence
        sentence_sequence = torch.IntTensor(text_to_sequence(sentence, self.audio_text_parameters['text_cleaners']))

        return sentence_sequence, mel_spec

    def __getitem__(self, index):
        return self.melspec_textSequence_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class DataCollate:

    def __init__(self, number_frames_step):
        self.number_frames_step = number_frames_step

    def __call__(self, batch):
        inp_lengths, sorted_decreasing = torch.sort(torch.LongTensor([len(x[0]) for x in batch]),
                                                   dim=0, descending=True)
        max_length_in = inp_lengths[0]

        # padding sentences sequences for a fixed-length tensor size
        sentences_padded = torch.LongTensor(len(batch), max_length_in)
        sentences_padded.zero_()
        for i in range(len(sorted_decreasing)):
            int_seq_sentence = batch[sorted_decreasing[i]][0]
            # all slots of a line until the end of the sentence. The rest, 0's
            sentences_padded[i, :int_seq_sentence.size(0)] = int_seq_sentence

        # length of the mel filterbank used
        num_melfilters = batch[0][1].size(0)

        # longest recorded spectrogram representation + 1 space to mark the end
        max_length_target = max([x[1].size(1) for x in batch])  # THERE IS A CHANGE FROM THE ORIGINAL CODE!!!
        # add extra space if the number of frames per step is higher than 1
        if max_length_target % self.number_frames_step != 0:
            max_length_target += self.number_frames_step - max_length_target % self.number_frames_step
            assert max_length_target % self.number_frames_step == 0

        # padding mel spectrogram representations. The output is a 3D tensor
        melspec_padded = torch.FloatTensor(len(batch), num_melfilters, max_length_target)
        melspec_padded.zero_()

        gate_padded = torch.FloatTensor(len(batch), max_length_target)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        for j in range(len(sorted_decreasing)):
            melspec = batch[sorted_decreasing[j]][1]
            melspec_padded[j, :, :melspec.size(1)] = melspec

            gate_padded[j, melspec.size(1) - 1:] = 1
            output_lengths[j] = melspec.size(1)

        return sentences_padded, inp_lengths, melspec_padded, gate_padded, output_lengths


if __name__ == "__main__":

    '''
    wav_path = "C:/Users/AlexPL/Desktop/LJ001-0001.wav"
    hparams = tacotron_params
    data_processing = DataPreparation([], hparams)
    stft = nn_layers.TacotronSTFT(hparams['filter_length'], hparams['hop_length'],
                                  hparams['win_length'], hparams['n_mel_channels'],
                                  hparams['sampling_rate'],
                                  hparams['mel_fmin'], hparams['mel_fmax'])

    wav = data_processing.load_audiowav_torch(wav_path, 22050)
    wav_torch_norm = wav / hparams['max_wav_value']
    wav_torch_norm = wav_torch_norm.unsqueeze(0)
    wav_torch_norm = torch.autograd.Variable(wav_torch_norm, requires_grad=False)
    mel_spec = stft.mel_spectrogram(wav_torch_norm)
    mel_spec = torch.squeeze(mel_spec, 0)
    '''

    # m = interp1d([-10, 0], [-4, 4])
    # mel_spec = m(mel_spec)

    # print(np.isnan(mel_spec).any())
    N = 153
    T = 750
    k = 0.05
    A = np.zeros([N, T])

    for i in range(N):
        for j in range(T):
            x = (math.exp(-((i/N - j/T)**2 / (2*k)**2)))
            A[i, j] = x

    # A = np.clip(A, 0, 2)
    # print(A[120, 670])
    # print(A[N-1, T-1])
    A = np.flipud(A)
    plt.imshow(A)
    plt.show()
