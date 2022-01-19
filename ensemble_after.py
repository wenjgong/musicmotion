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
r1 = open(pred_label_path + '/AIST_JOINT_DROPOUT_500.pkl', 'rb')
r1 = pickle.load(r1)
r2 = open(pred_label_path + '/AIST_BONE_DROPOUT_500.pkl', 'rb')
r2 = pickle.load(r2)
text_path = pred_label_path + '/AIST_DROPOUT_ensemble_after_500.txt'
f_w = open(text_path,'w')
i = 0
count = 0
total_num = 0
while i < len(label):
    temp = []
    j = 0
    l = label[i]
    while j < 17:
        r11 = r1[i]
        r22 = r2[i]
        r = r11 + r22 * alpha
        r = np.argmax(r)
        temp.append(r)
        i += 1
        j += 1
    if int(l) == max(temp, key=temp.count):
        count += 1
    f_w.write(str(int(l))+','+str(max(temp, key=temp.count)) + '\n')#真，预测
    total_num += 1
acc = count / total_num
print(acc)
