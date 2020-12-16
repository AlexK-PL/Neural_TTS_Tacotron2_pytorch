import torch
import numpy as np
from scipy.io.wavfile import write
# from scipy.interpolate import interp1d

import time
from melgan.model.generator import Generator

# from multi_band_melgan.model.generator import Generator
# from multi_band_melgan.utils.hparams import load_hparam_str
# from multi_band_melgan.utils.pqmf import PQMF

from hyper_parameters import tacotron_params
from training import load_model
from text import text_to_sequence
from audio_processing import _denormalize
from audio_processing import griffin_lim
from nn_layers import TacotronSTFT

import tkinter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

torch.manual_seed(1234)

predicted_melspec_folder = 'C:/Users/AlexPL/Tacotron2_SpecNorm_DDC_BNPrenet/predicted_melspec/'
audio_path = 'C:/Users/AlexPL/Tacotron2_SpecNorm_DDC_BNPrenet/synthesis_wavs/169k_steps_melgan_INGENIOUS_MOS/20.wav'

MAX_WAV_VALUE = 32768.0

# load trained tacotron 2 model:
checkpoint_path = "tacotron2_model/checkpoint_169000"
temp_model = torch.load(checkpoint_path, map_location='cpu')
state_dict = temp_model['state_dict']
hparams = tacotron_params
tacotron2 = load_model(hparams)
tacotron2.load_state_dict(state_dict)
_ = tacotron2.eval()


# load pre trained MelGAN model for mel2audio:
vocoder_model_path = 'melgan_model/nvidia_tacotron2_LJ11_epoch6400.pt'
temp_model = torch.load(vocoder_model_path, map_location='cpu')
melgan = Generator(80)  # Number of mel channels
melgan.load_state_dict(temp_model['model_g'])
melgan.eval(inference=False)


'''
# load pre-trained Multi-Band MelGAN for mel2audio:
checkpoint = torch.load('C:/Users/AlexPL/Tacotron2_SpecNorm_DDC_BNPrenet/melgan_model/mb_melgan_901be72_0600.pt',
                        map_location='cpu')
hp = load_hparam_str(checkpoint['hp_str'])

vocoder = Generator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                    ratios=hp.model.generator_ratio, mult=hp.model.mult,
                    out_band=hp.model.out_channels)
vocoder.load_state_dict(checkpoint['model_g'])
vocoder.eval(inference=False)
'''

# preparing inputs for synthesis:
# test_text = "That is not only my accusation."  # 1
# test_text = "The car engine was totally broken."  # 2
# test_text = "This was all the police wanted to know."  # 3
# test_text = "And there may be only nine."  # 4
# test_text = "He had here completed his ascent."  # 5
# test_text = "From defection to return to Fort Worth."  # 6
# test_text = "Yet the law was seldom if ever enforced."  # 7
# test_text = "The latter too was to be laid before the House of Commons."  # 8
# test_text = "Palmer speedily found imitators."  # 9
# test_text = "There were others less successful."  # 10
# test_text = "You can find her in the office 55, third floor, number 25."
# test_text = "This is not even fair! Bitch!"
# test_text = "Alexa, switch on the lights!"
# test_text = "The president of the united states was chosen by the majority of the population."

# test_text = "From defection to return to Fort Worth."
# test_text = "inspectors of prisons should be appointed, who should visit all the prisons from time to time and report to the Secretary of State."
# test_text = "the forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves."

# INGENIOUS TEST SENTENCES
# ---------------------------------------------------------------------------------------------------------------------
test_text = "We need an ambulance in Bilbao."  # 1
# test_text = "We have a case of domestic violence in Barcelona."  # 2
# test_text = "There is a gas leak at the nuclear power plant."  # 3
# test_text = "The robbery ended with a shooting."  # 4
# test_text = "The fire spreads to the west."  # 5
# test_text = "We have an emergency."  # 6
# test_text = "We request a patrol car, code Tee Key 12."  # 7
# test_text = "We need reinforcements, code Tee Key 12."  # 8
# test_text = "We have a Tee Key 12 code, we request reinforcements."  # 9
# test_text = "Register a Tee Key 12 in Zarauz."  # 10
# test_text = "We have two civilian casualties here."  # 11 (created)
# test_text = "Firefighters have entered through the back door."  # 12 (created)
# test_text = "The suspect has fled through the roof."  # 13 (created)
# test_text = "Two people have entered the private home."  # 14 (created)
# test_text = "What is the status?"  # 15 (created)
# test_text = "The exit is blocked by debris."  # 16 (created)
# test_text = "Debris blocks the exit."  # 17 (created)
# test_text = "Civilians are trapped in the rubble."  # 18 (created)
# test_text = "We request more supplies."  # 19 (created)
# test_text = "We have the fire under control."  # 20 (created)

# test_text = "On December twenty-six, nineteen sixty-three, the FBI circulated additional instructions to all its agents,"
# test_text = "a purple pig and a green donkey flew a kite in the middle of the night and ended up sunburn the contained error poses as a logical target the divorce attacks near a missing doom the opera fines the daily examiner into a murderer."
# test_text = "Examination of the cartridge cases found on the sixth floor of the Depository Building"
# test_text = "someone i know recently combined maple syrup and buttered popcorn thinking it would taste like caramel popcorn it didn't and they don't recommend anyone else do it either the gentleman marches around the principal the divorce attacks near a missing doom the color misprints a circular worry across the controversy."
# test_text = "The Waystone was his, just as the third silence was his. This was appropriate, as it was the greatest silence of the three, wrapping the others inside itself. It was deep and wide as autumn’s ending. It was heavy as a great river-smooth stone. It was the patient, cut-flower sound of a man who is waiting to die."

sequence = np.array(text_to_sequence(test_text, ['english_cleaners']))[None, :]
sequence = torch.from_numpy(sequence).to(device='cpu', dtype=torch.int64)

taco_t1 = time.time()

# text2mel:
tacotron2.to('cpu')
with torch.no_grad():
    mel_outputs, mel_outputs_postnet, _, alignments = tacotron2.inference(sequence)

# mel_outputs_postnet = mel_outputs_postnet.data.cpu().detach().numpy()

taco_t2 = round(time.time() - taco_t1, 2)
print("Tacotron2 took {} seconds.".format(taco_t2))
print("text2mel prediction successfully performed...")

# S = mel_outputs_postnet.squeeze(0)
# S = np.flipud(S)
# plt.imshow(S)
# plt.show()

# mel_outputs_postnet = _denormalize(hparams, mel_outputs_postnet)

# mel_outputs_postnet = np.clip(mel_outputs_postnet, -hparams["max_norm"], hparams["max_norm"])
# m = interp1d([-4, 4], [-10, 0])
# mel_outputs_postnet = m(mel_outputs_postnet)

# mel_outputs_postnet = torch.from_numpy(mel_outputs_postnet).to(device='cpu')

# save the predicted outputs from tacotron2:
mel_outputs_path = predicted_melspec_folder + "output.pt"
mel_outputs_postnet_path = predicted_melspec_folder + "output_postnet.pt"
alignments_path = predicted_melspec_folder + "alignment.pt"
torch.save(mel_outputs, mel_outputs_path)
torch.save(mel_outputs_postnet, mel_outputs_postnet_path)
torch.save(alignments, alignments_path)

'''
mb_melgan_t1 = time.time()

# Multi-Band MelGAN synthesis:
with torch.no_grad():
    mel = mel_outputs_postnet.detach()
    if len(mel.shape) == 2:
        mel = mel.unsqueeze(0)
    # mel = mel.cuda()
    audio = vocoder.inference(mel)
    # For multi-band inference
    if hp.model.out_channels > 1:
        pqmf = PQMF()
        audio = pqmf.synthesis(audio).view(-1)

audio = audio.squeeze()  # collapse all dimension except time axis
audio = audio[:-(hp.audio.hop_length * 10)]
audio = MAX_WAV_VALUE * audio
audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
audio = audio.short()
audio = audio.cpu().detach().numpy()

write(audio_path, 22050, audio)

mb_melgan_t2 = round(time.time() - mb_melgan_t1, 2)

print("mel2wav successfully performed...")
print("Multi-Band MelGAN took {} seconds.".format(mb_melgan_t2))
'''


melgan_t1 = time.time()

# mel2wav MelGAN inference:
with torch.no_grad():
    result = melgan.inference(mel_outputs_postnet)

audio = result.data.cpu().detach().numpy()
write(audio_path, 22050, audio)

melgan_t2 = time.time() - melgan_t1

print("mel2wav successfully performed...")
print("MelGAN took {} seconds.".format(melgan_t2))


'''
# Griffin Lim vocoder synthesis:
griffin_iters = 100
taco_stft = TacotronSTFT(hparams['filter_length'], hparams['hop_length'], hparams['win_length'], sampling_rate=hparams['sampling_rate'])

t2 = time.time()
mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
mel_decompress = mel_decompress.transpose(1, 2).data.cpu()

spec_from_mel_scaling = 60
spec_from_mel = torch.mm(mel_decompress[0].double(), taco_stft.mel_basis.double())
spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
spec_from_mel = spec_from_mel * spec_from_mel_scaling

audio = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1].double()), taco_stft.stft_fn, griffin_iters)

audio = audio.squeeze()
audio_numpy = audio.cpu().numpy()

elapsed_griffinLim = time.time() - t2
print('Time elapsed in transforming mel to wav has been {} seconds.'.format(elapsed_griffinLim))

write(audio_path, 22050, audio_numpy)
'''

'''
with torch.no_grad():
    audio = MAX_WAV_VALUE*melgan.infer(mel_outputs_postnet, sigma=0.666)[0]
audio = audio.cpu().numpy()
audio = audio.astype('int16')
elapsed = time.time() - t
write(save_path, 22050, audio)

print("mel2audio synthesis successfully performed.")
print("Elapsed time during inference is" + str(elapsed) + "seconds.")
'''

