#To ensemble the results of joints and bones,将joints和bone的试验结果结合起来，这种方式是将最后的结果结合起来，而不是将两种特征叠加起来输入到网络中
import argparse
import pickle

import numpy as np
from tqdm import tqdm
true_label_path = './result/AIST_true_label.pkl'
pred_label_path = './result'
alpha = 1

label = open(true_label_path, 'rb')
label = np.array(pickle.load(label))
r1 = open(pred_label_path + '/AIST_Joint_FC_1000.pkl', 'rb')
r1 = pickle.load(r1)
r2 = open(pred_label_path + '/AIST_BONE_FC_1000.pkl', 'rb')
r2 = pickle.load(r2)
right_num = total_num = 0
text_path = pred_label_path + '/AIST_FC_ensemble_1000.txt'
f_w = open(text_path,'w')
for i in tqdm(range(len(label))):
    l = label[i]
    r11 = r1[i]
    r22 = r2[i]
    r = r11 + r22 * alpha
    r = np.argmax(r)
    f_w.write(str(int(l))+','+str(r) + '\n')#真，预测
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
print(acc)
