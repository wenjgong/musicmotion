#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Perform preprocessing and raw feature extraction."""

import os
import librosa
import numpy as np
import torch
import pickle
from sklearn.preprocessing import StandardScaler,MinMaxScaler

data_dirs = '../datasets/AIST_DATASET/ANNOTATIONS_DIR/splits'
music_dir = '../datasets/AIST_DATASET/VIDEO_DIR'
dump_dir = '../data/AIST_segment_single'
skeletons_dir = '../datasets/AIST_DATASET/ANNOTATIONS_DIR/keypoints3d'
fps = 60
sampling_rate = 30700 #12800     # Sampling rate.
fft_size = 1024           # FFT size.
hop_size = 512            # Hop size.
win_length = 1024         # Window length.
                         # If set to null, it will be the same as fft_size.
window = "hann"           # Window function.
num_mels = 96             # Number of mel basis.
fmin = 80                 # Minimum freq in mel basis calculation.
fmax = 7600

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_dict = {'gBR': 0, 'gPO': 1, 'gLO': 2, 'gMH': 3, 'gLH': 4, 'gHO': 5, 'gWA': 6, 'gKR': 7, 'gJS': 8,
              'gJB': 9}
min_length = 425 # 先选出每一段音乐的最小值，将数据集中的每一个曲子切割成相同长度的。

def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

def extract_skeletons(file_path, use_optim):
    assert os.path.exists(file_path), f'File {file_path} does not exist!'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    if use_optim:
        return data['keypoints3d_optim'].reshape(data['keypoints3d_optim'].shape[0],-1)  # (N, 17, 3)
    else:
        return data['keypoints3d']  # (N, 17, 3)

def save_features(data_list, saves):
    motion = []
    music = []
    label_list = []
    name_list = []
    for i in data_list:
        mp3 = os.path.join(music_dir, i + '.mp4')
        mp3 = mp3.replace("cAll", "c01")
        if not os.path.exists(mp3):
            continue
        skeleton_path = os.path.join(skeletons_dir, i + '.pkl')
        skeleton = extract_skeletons(skeleton_path, True)
        audio, _ = librosa.core.load(mp3, sr=sampling_rate)
        mel = logmelfilterbank(audio,
                               sampling_rate=sampling_rate,
                               hop_size=hop_size,
                               fft_size=fft_size,
                               win_length=win_length,
                               num_mels=num_mels,
                               fmin=fmin,
                               fmax=fmax)
        mel = MinMaxScaler().fit_transform(mel)
        skeleton = MinMaxScaler().fit_transform(skeleton)
        # save
        mins = min(audio.shape[0] // hop_size, mel.shape[0], skeleton.shape[0])
        mel = mel[:mins, ]
        skeleton = skeleton[:mins, ]
        audio = audio[:mins * hop_size, ]
        skeleton = skeleton.reshape(skeleton.shape[0], 17, 3)
        counts = mins // min_length
        label = i.split('_')[0]
        for _ in range(counts):
            label_list.append(label_dict[label])
            name_list.append(label)
        for i in range(counts):
            music.append(mel[i * min_length:(i + 1) * min_length])
            motion.append(skeleton[i * min_length:(i + 1) * min_length])
    motion = np.array(motion)
    motion = motion.transpose(0, 3, 1, 2)
    music = np.array(music)
    np.save('{}/{}_joint.npy'.format(dump_dir, saves), motion)
    np.save('{}/{}_mfcc.npy'.format(dump_dir, saves), music)
    with open('{}/{}_label.pkl'.format(dump_dir, saves), 'wb') as f:
        pickle.dump((name_list, list(label_list)), f)  # 保存文件名称和对应的缩写（作为标签）形成pkl文件。

if __name__ == "__main__":
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    train_list = np.loadtxt(data_dirs + '/crossmodal_train.txt', dtype=str)
    val_list = np.loadtxt(data_dirs + '/crossmodal_val.txt', dtype=str)
    test_list = np.loadtxt(data_dirs + '/crossmodal_test.txt', dtype=str)
    save_features(train_list,'train')
    save_features(val_list,'val')
    save_features(test_list,'test')
