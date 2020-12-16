######################################################################################################
# The main script where the data preparation, training and evaluation happens.
######################################################################################################

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hyper_parameters import tacotron_params
from data_preparation import DataPreparation, DataCollate
from training import train


if __name__ == '__main__':
    # run()
    # ---------------------------------------- DEFINING INPUT ARGUMENTS ---------------------------------------------- #

    training_files = 'filelists/ljs_audio_text_train_filelist.txt'
    validation_files = 'filelists/ljs_audio_text_val_filelist.txt'

    output_directory = 'C:/Users/AlexPL/Tacotron2_SpecNorm_DDC_BNPrenet/outputs'
    log_directory = 'C:/Users/AlexPL/Tacotron2_SpecNorm_DDC_BNPrenet/loggs'
    # log_directory = '/tmp/loggs_tacotron2_original/'
    # checkpoint_path = '/homedtic/apeiro/TACOTRON2_ORIGINAL/outputs/checkpoint_78000'
    checkpoint_path = None
    warm_start = False
    n_gpus = 1  # NEED TO ASK TO TIC DEPARTMENT ABOUT CLUSTER
    rank = 0

    torch.backends.cudnn.enabled = tacotron_params['cudnn_enabled']
    torch.backends.cudnn.benchmark = tacotron_params['cudnn_benchmark']

    print("FP16 Run:", tacotron_params['fp16_run'])
    print("Dynamic Loss Scaling:", tacotron_params['dynamic_loss_scaling'])
    print("Distributed Run:", tacotron_params['distributed_run'])
    print("CUDNN Enabled:", tacotron_params['cudnn_enabled'])
    print("CUDNN Benchmark:", tacotron_params['cudnn_benchmark'])

    # --------------------------------------------- PREPARING DATA --------------------------------------------------- #

    # Read the training files
    with open(training_files, encoding='utf-8') as f:
        training_audiopaths_and_text = [line.strip().split("|") for line in f]

    L_t = len(training_audiopaths_and_text)
    training_audiopaths_and_text_cut = [['', '']]

    t_cool_counter = 0
    for t in range(L_t):
        text_seq = training_audiopaths_and_text[t][1]
        l_t = len(text_seq)
        if l_t > 6 and l_t < 153:
            t_cool_counter += 1
            training_audiopaths_and_text_cut.append(training_audiopaths_and_text[t])

    training_audiopaths_and_text_cut.pop(0)
    print(t_cool_counter)
    print(len(training_audiopaths_and_text_cut))
    print(training_audiopaths_and_text_cut[0])

    # Read the validation files
    with open(validation_files, encoding='utf-8') as f:
        validation_audiopaths_and_text = [line.strip().split("|") for line in f]

    L_v = len(validation_audiopaths_and_text)
    validation_audiopaths_and_text_cut = [['', '']]

    v_cool_counter = 0
    for v in range(L_v):
        text_seq = validation_audiopaths_and_text[v][1]
        l_t = len(text_seq)
        if l_t > 6 and l_t < 153:
            v_cool_counter += 1
            validation_audiopaths_and_text_cut.append(validation_audiopaths_and_text[v])

    validation_audiopaths_and_text_cut.pop(0)
    print(v_cool_counter)
    print(len(validation_audiopaths_and_text_cut))
    print(validation_audiopaths_and_text_cut[0])

    '''
    # prepare the data
    # GST adaptation to put prosody features path as an input argument:
    train_data = DataPreparation(training_audiopaths_and_text, tacotron_params)
    validation_data = DataPreparation(validation_audiopaths_and_text, tacotron_params)
    collate_fn = DataCollate(tacotron_params['number_frames_step'])

    # DataLoader prepares a loader for a set of data including a function that processes every
    # batch as we wish (collate_fn). This creates an object with which we can list the batches created.
    # DataLoader and Dataset (IMPORTANT FOR FURTHER DESIGNS WITH OTHER DATABASES)
    # https://jdhao.github.io/2017/10/23/pytorch-load-data-and-make-batch/

    train_sampler = DistributedSampler(train_data) if tacotron_params['distributed_run'] else None
    val_sampler = DistributedSampler(validation_data) if tacotron_params['distributed_run'] else None

    train_loader = DataLoader(train_data, num_workers=0, shuffle=False, sampler=train_sampler,
                              batch_size=tacotron_params['batch_size'], pin_memory=False, drop_last=True,
                              collate_fn=collate_fn)

    validate_loader = DataLoader(validation_data, num_workers=0, shuffle=False, sampler=val_sampler,
                                 batch_size=tacotron_params['batch_size_validation'], pin_memory=False, drop_last=True,
                                 collate_fn=collate_fn)
    '''

    # ------------------------------------------------- TRAIN -------------------------------------------------------- #

    train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus, rank, hyper_params=tacotron_params,
          val_paths=validation_audiopaths_and_text, train_paths=training_audiopaths_and_text, group_name="group_name")

    print("Training completed")
