from math import sqrt

import torch
from torch import nn

from Encoder import Encoder
from Decoder import Decoder, DoubleDecoderConsistency
from Postnet import Postnet

from utils import to_gpu, get_mask_from_lengths
from fp16_optimizer import fp32_to_fp16, fp16_to_fp32


class tacotron_2(nn.Module):
    def __init__(self, tacotron_hyperparams):
        super(tacotron_2, self).__init__()
        self.mask_padding = tacotron_hyperparams['mask_padding']
        self.fp16_run = tacotron_hyperparams['fp16_run']
        self.n_mel_channels = tacotron_hyperparams['n_mel_channels']
        self.n_frames_per_step = tacotron_hyperparams['number_frames_step']
        self.embedding = nn.Embedding(
            tacotron_hyperparams['n_symbols'], tacotron_hyperparams['symbols_embedding_length'])
        # CHECK THIS OUT!!!
        std = sqrt(2.0 / (tacotron_hyperparams['n_symbols'] + tacotron_hyperparams['symbols_embedding_length']))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(tacotron_hyperparams)
        self.decoder = Decoder(tacotron_hyperparams)
        self.coarse_decoder = DoubleDecoderConsistency(tacotron_hyperparams)
        self.postnet = Postnet(tacotron_hyperparams)

    def parse_batch(self, batch):
        # GST I add the new tensor from prosody features to train GST tokens:
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        max_len = int(torch.max(input_lengths.data).item())  # With item() you get the pure value (not in a tensor)
        input_lengths = to_gpu(input_lengths).long()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_input(self, inputs):
        inputs = fp32_to_fp16(inputs) if self.fp16_run else inputs
        return inputs

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        outputs = fp16_to_fp32(outputs) if self.fp16_run else outputs

        return outputs

    def forward(self, inputs):
        inputs, input_lengths, targets, max_len, output_lengths = self.parse_input(inputs)
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, input_lengths)

        # print("Encoder outputs size is:")
        # print(encoder_outputs.size())
        # print("The input lengths are:")
        # print(input_lengths)

        T = targets.shape[2]
        print("T is: {}".format(T))
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths)
        mel_outputs_coarse, _, alignments_coarse = self.coarse_decoder(
            encoder_outputs.detach(), targets, memory_lengths=input_lengths)  # LOOK AT HERE!!!

        # print("Size of gate outputs:")
        # print(gate_outputs.size())
        # print("Size of targets:")
        # print(targets.size())
        gate_outputs = gate_outputs.unsqueeze(0)
        gate_outputs = torch.nn.functional.interpolate(
            gate_outputs, size=targets.shape[2], mode='nearest')
        gate_outputs = gate_outputs.squeeze(0)
        # print("Gate outputs size is: ")
        # print(gate_outputs.size())

        alignments_coarse = torch.nn.functional.interpolate(
            alignments_coarse.transpose(1, 2),
            size=T,
            mode='nearest').transpose(1, 2)
        alignments = torch.nn.functional.interpolate(
            alignments.transpose(1, 2),
            size=T,
            mode='nearest').transpose(1, 2)
        mel_outputs_coarse = mel_outputs_coarse.transpose(1, 2)
        mel_outputs_coarse = mel_outputs_coarse[:, :T, :]
        mel_outputs_coarse = mel_outputs_coarse.transpose(1, 2)
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs = mel_outputs[:, :T, :]
        mel_outputs = mel_outputs.transpose(1, 2)
        # print("Mel outputs coarse size after treatment: ")
        # print(mel_outputs_coarse.size())
        # print("Mel outputs size after treatment: ")
        # print(mel_outputs.size())        # return decoder_outputs_backward, alignments_backward

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        # print("Size of alignments:")
        # print(alignments.size())
        # print("Size of coarse alignments")
        # print(alignments_coarse.size())

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, mel_outputs_coarse,
             alignments_coarse, output_lengths],
            output_lengths)

    def inference(self, inputs):
        inputs = self.parse_input(inputs)
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
