import os
import pickle
import json
import math
import numpy as np
from utils import compute_spectrogram
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import librosa
Datadir = '../datasets/MusicToDance'
out_path = '../data/MusicToDance'
audio_path = '../audioData'
fps = 25
num_joint = 23
min_duration = 229
min_num = 34
num_mfcc = 96

def load_skeleton(skeleton_json, config_json):
    with open(skeleton_json, 'r') as f, open(config_json, 'r') as cfg:
        data = json.load(f)
        cfg_data = json.load(cfg)
    return data['length'],data['center'],data['skeletons'], cfg_data['start_position'] #length是节点的帧数，以第一个文件为例，4300，center是每一帧有一个中心位置，是一个二维列表，4300x3,skeletons是所有帧的节点，4300x23x3,

def rotate_one_skeleton_by_axis(skeleton, axis, angle):
    delta_x = skeleton[0] - axis[0]
    delta_z = skeleton[2] - axis[2]
    skeleton_new = skeleton
    skeleton_new[0] = delta_x * math.cos(angle) + delta_z * math.sin(angle)
    skeleton_new[2] = -delta_x * math.sin(angle) + delta_z * math.cos(angle)
    return skeleton_new

def rotate_skeleton(frames):
    frames = np.asarray(frames) # 4300 23 3
    for i in range(len(frames)):
        this_frame = frames[i]
        waist_lf = this_frame[16]
        waist_rt = this_frame[7]
        axis = this_frame[2]
        lf = waist_lf - axis
        rt = waist_rt - axis
        mid = lf+rt
        theta = math.atan2(mid[2], mid[0]) # from x+ axis
        for j in range(len(this_frame)):
            frames[i][j] =  rotate_one_skeleton_by_axis(this_frame[j], axis, theta)
            frames[i][j] =  rotate_one_skeleton_by_axis(this_frame[j], axis, -math.pi/2) # turn to y- axis
    return frames

def motion_feature_extract(data_path, with_rotate, with_centering):
    skeleton_path = os.path.join(data_path, "skeletons.json")
    config_path = os.path.join(data_path, 'config.json')
    duration, center, frames, start = load_skeleton(skeleton_json=skeleton_path, config_json = config_path)
    center = np.asarray(center)
    frames = np.asarray(frames)
    if with_centering:
        for i in range(len(frames)):
            for j in range(len(frames[i])):
                frames[i][j] -= center[i]
    if with_rotate:
        frames = rotate_skeleton(frames)
    return frames, duration, start

def genda(data_dirs):
    train_dir = []
    test_dir = []
    val_dir = []
    seg = min_num // 3
    train_motion = np.zeros((seg, 3, min_duration, num_joint), dtype=np.float32)#(3, 3, 1494, 23, 1)
    val_motion = np.zeros((seg, 3, min_duration, num_joint), dtype=np.float32)
    test_motion = np.zeros((min_num - 2 * seg, 3, min_duration, num_joint), dtype=np.float32)

    train_music = np.zeros((seg, min_duration, num_mfcc), dtype=np.float32)  # (3, 3, 1494, 23, 1)
    val_music = np.zeros((seg, min_duration, num_mfcc), dtype=np.float32)
    test_music = np.zeros((min_num - 2 * seg, min_duration, num_mfcc), dtype=np.float32)

    temp_num = 0
    train_i = 0
    val_i = 0
    test_i = 0

    for i, one in enumerate(data_dirs):
        data_path = os.path.join(Datadir,one)
        skeletons_path = os.path.join(data_path, 'skeletons.json')
        with open(skeletons_path, 'r') as fin:
            data = json.load(fin)
        skeletons = np.array(data['skeletons'])

        motion_features_pre, duration, start = motion_feature_extract(data_path, with_centering=True, with_rotate=False)
        mp3 = os.path.join(Datadir, one, 'audio.mp3')
        audio, sr = librosa.load(mp3, sr=12800, offset=start / fps, duration=(duration - 1) / fps,
                                 res_type='kaiser_fast')
        music = compute_spectrogram(mp3, start, duration)
        motion_features = motion_features_pre.reshape(motion_features_pre.shape[0],-1)
        # min_max_scaler
        min_max_scaler = preprocessing.MinMaxScaler().fit_transform(motion_features)
        motion_features = min_max_scaler.reshape(min_max_scaler.shape[0],23,3)
        music_features = preprocessing.MinMaxScaler().fit_transform(music)

        # standard_scaler = StandardScaler().fit_transform(motion_features)
        # motion_features = standard_scaler.reshape(standard_scaler.shape[0], 23, 3)
        # music_features = StandardScaler().fit_transform(music)

        t = motion_features.shape[0] // min_duration
        
        for j in range(t):
            
            temp_num += 1

            temp_music = music_features[min_duration * j:min_duration * (j + 1), :]
            temp_features = motion_features[min_duration * j:min_duration * (j + 1), :, :]
            temp_features = np.transpose(temp_features, [2, 0, 1])
            temp_skeleton = skeletons[min_duration * j:min_duration * (j + 1), :, :]
            temp_audio = audio[min_duration * j * 512:min_duration * (j + 1)*512,]
            if temp_num <= seg:
                #train
                train_motion[train_i,:,:,:] = temp_features
                train_music[train_i,:,:] = temp_music
                train_dir.append(one)
                train_i += 1
                librosa.output.write_wav(os.path.join(audio_path,'train', str(one) + '-' + str(j) + '.wav'), temp_audio, sr)
                np.save(os.path.join(audio_path,'train', str(one) + '-' + str(j) + '.npy'),
                        temp_skeleton.astype(np.float32), allow_pickle=False)
            elif temp_num > seg and temp_num<= seg * 2:
                #val
                val_motion[val_i,:,:,:] = temp_features
                val_music[val_i,:,:] = temp_music
                val_dir.append(one)
                val_i += 1
                librosa.output.write_wav(os.path.join(audio_path, 'val', str(one) + '-' + str(j) + '.wav'),
                                         temp_audio, sr)
                np.save(os.path.join(audio_path, 'val', str(one) + '-' + str(j) + '.npy'),
                        temp_skeleton.astype(np.float32), allow_pickle=False)
            elif temp_num > seg * 2 and temp_num <= seg*3 + 1:
               #test
                test_motion[test_i,:,:,:] = temp_features
                test_music[test_i,:,:] = temp_music
                test_dir.append(one)
                test_i += 1
                librosa.output.write_wav(os.path.join(audio_path, 'test', str(one) + '-' + str(j) + '.wav'), temp_audio,
                                         sr)
                np.save(os.path.join(audio_path, 'test', str(one) + '-' + str(j) + '.npy'),
                        temp_skeleton.astype(np.float32), allow_pickle=False)
            else:
                break
    
    return train_motion,train_music, train_dir,val_motion,val_music,val_dir,test_motion,test_music,test_dir

if __name__ == '__main__':
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)
    if not os.path.exists(os.path.join(audio_path, 'train')):
        os.makedirs(os.path.join(audio_path, 'train'))
    if not os.path.exists(os.path.join(audio_path, 'val')):
        os.makedirs(os.path.join(audio_path, 'val'))
    if not os.path.exists(os.path.join(audio_path, 'test')):
        os.makedirs(os.path.join(audio_path, 'test'))
    All_dirs = os.listdir(Datadir)
    All_dirs.sort()
    C_dirs = []
    R_dirs = []
    T_dirs = []
    W_dirs = []
    for one in All_dirs:
        if one.split('_')[1] == 'C':
            C_dirs.append(one)
        elif one.split('_')[1] == 'R':
            R_dirs.append(one)
        elif one.split('_')[1] == 'T':
            T_dirs.append(one)
        else:
            W_dirs.append(one)
    C_train_motion,C_train_music,C_train_dir,C_val_motion,C_val_music,C_val_dir,C_test_motion,C_test_music,C_test_dir = genda(C_dirs)
    R_train_motion,R_train_music,R_train_dir,R_val_motion,R_val_music,R_val_dir,R_test_motion,R_test_music,R_test_dir = genda(R_dirs)
    T_train_motion,T_train_music,T_train_dir,T_val_motion,T_val_music,T_val_dir,T_test_motion,T_test_music,T_test_dir = genda(T_dirs)
    W_train_motion,W_train_music,W_train_dir,W_val_motion,W_val_music,W_val_dir,W_test_motion,W_test_music,W_test_dir = genda(W_dirs)

    train_motion = np.vstack((C_train_motion,R_train_motion,T_train_motion,W_train_motion))
    val_motion = np.vstack((C_val_motion,R_val_motion,T_val_motion,W_val_motion))
    test_motion = np.vstack((C_test_motion,R_test_motion,T_test_motion,W_test_motion))
    train_music = np.vstack((C_train_music, R_train_music, T_train_music, W_train_music))
    val_music = np.vstack((C_val_music, R_val_music, T_val_music, W_val_music))
    test_music = np.vstack((C_test_music, R_test_music, T_test_music, W_test_music))
    train_dir = C_train_dir + R_train_dir + T_train_dir + W_train_dir
    test_dir = C_test_dir + R_test_dir + T_test_dir + W_test_dir
    val_dir = C_val_dir + R_val_dir + T_val_dir + W_val_dir

    seg = min_num // 3

    train_label = [0] * seg + [1] * seg + [2] * seg + [3] * seg
    val_label = [0] * seg + [1] * seg + [2] * seg + [3] * seg
    test_label = [0] * (seg + 1) + [1] * (seg + 1) + [2] * (seg + 1) + [3] * (seg + 1)

    np.save('{}/{}_joint.npy'.format(out_path, 'train'), train_motion)
    np.save('{}/{}_joint.npy'.format(out_path,'val'), val_motion)
    np.save('{}/{}_joint.npy'.format(out_path,'test'), test_motion)

    np.save('{}/{}_mfcc.npy'.format(out_path, 'train'), train_music)
    np.save('{}/{}_mfcc.npy'.format(out_path, 'val'), val_music)
    np.save('{}/{}_mfcc.npy'.format(out_path, 'test'), test_music)

    with open('{}/{}_label.pkl'.format(out_path,'train' ), 'wb') as f:
        pickle.dump((train_dir, list(train_label)), f)#save file name and labler
    with open('{}/{}_label.pkl'.format(out_path,'val'), 'wb') as f:
        pickle.dump((val_dir, list(val_label)), f)
    with open('{}/{}_label.pkl'.format(out_path,'test'), 'wb') as f:
        pickle.dump((test_dir, list(test_label)), f)