#Opported bone data and joint data data
import pickle
import numpy as np
import os
out_path = '../data/AIST_segment_single'#'../data/MusicToDance'#

train_joint = np.load(out_path + '/train_joint.npy')
val_joint = np.load(out_path + '/val_joint.npy')
test_joint = np.load(out_path + '/test_joint.npy')
#(61, 3, 300, 23, 1)
train_bone = np.load(out_path + '/train_bone.npy')
val_bone = np.load(out_path + '/val_bone.npy')
test_bone = np.load(out_path + '/test_bone.npy')
#(61, 6, 300, 23, 1)
train_data = np.hstack((train_joint,train_bone))
val_data = np.hstack((val_joint,val_bone))
test_data = np.hstack((test_joint,test_bone))
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)
np.save('{}/{}_data.npy'.format(out_path, 'train'), train_data)
np.save('{}/{}_data.npy'.format(out_path, 'val'), val_data)
np.save('{}/{}_data.npy'.format(out_path, 'test'), test_data)
