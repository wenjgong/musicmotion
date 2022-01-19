import numpy as np
import cv2
import json
import os
import random
from moviepy.editor import *
import pickle
import shutil
CANVAS_SIZE = (900,600,3)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
fps = 25
scale = 3
outputs_path = './outvideo/'
data_dirs = './audioData'

def draw_skeleton_number(cvs, frame):
    for j in range(21):
        cv2.putText(cvs, str(j), (int(frame[j][0]), (CANVAS_SIZE[1] - int(frame[j][1]))), cv2.FONT_ITALIC, 0.6,
                    (0, 0, 255), 1)

def draw(frames,video_path):
    frames[:,:,0] += CANVAS_SIZE[0]//2#x
    frames[:,:,1] += CANVAS_SIZE[1]//2#y
    video = cv2.VideoWriter(video_path, fourcc, fps, (CANVAS_SIZE[0],CANVAS_SIZE[1]), 1)
    for i in range(len(frames)):
        cvs = np.ones((CANVAS_SIZE[1], CANVAS_SIZE[0], CANVAS_SIZE[2]))
        cvs[:,:,:] = 255
        color = (0,0,0)
        hlcolor = (255,0,0)
        dlcolor = (0,0,255)
        for points in frames[i]:
            cv2.circle(cvs,(int(points[0]),int(points[1])),radius=4,thickness=-1,color=hlcolor)
        frame = frames[i]
        cv2.line(cvs, (int(frame[0][0]), int(frame[0][1])), (int(frame[1][0]), int(frame[1][1])), color, 2)
        cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[3][0]), int(frame[3][1])), color, 2)
        cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])), (int(frame[4][0]), int(frame[4][1])), color, 2)
        cv2.line(cvs, (int(frame[4][0]), int(frame[4][1])), (int(frame[5][0]), int(frame[5][1])), color, 2)
        cv2.line(cvs, (int(frame[5][0]), int(frame[5][1])), (int(frame[6][0]), int(frame[6][1])), color, 2)
        cv2.line(cvs, (int(frame[7][0]), int(frame[7][1])), (int(frame[8][0]), int(frame[8][1])), color, 2)
        cv2.line(cvs, (int(frame[8][0]), int(frame[8][1])), (int(frame[9][0]), int(frame[9][1])), color, 2)
        cv2.line(cvs, (int(frame[9][0]), int(frame[9][1])), (int(frame[10][0]), int(frame[10][1])), color, 2)
        cv2.line(cvs, (int(frame[10][0]), int(frame[10][1])), (int(frame[11][0]), int(frame[11][1])), color, 2)
        cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])), (int(frame[13][0]), int(frame[13][1])), color, 2)
        cv2.line(cvs, (int(frame[13][0]), int(frame[13][1])), (int(frame[14][0]), int(frame[14][1])), color, 2)
        cv2.line(cvs, (int(frame[14][0]), int(frame[14][1])), (int(frame[15][0]), int(frame[15][1])), color, 2)
        cv2.line(cvs, (int(frame[16][0]), int(frame[16][1])), (int(frame[17][0]), int(frame[17][1])), color, 2)
        cv2.line(cvs, (int(frame[17][0]), int(frame[17][1])), (int(frame[18][0]), int(frame[18][1])), color, 2)
        cv2.line(cvs, (int(frame[18][0]), int(frame[18][1])), (int(frame[19][0]), int(frame[19][1])), color, 2)
        cv2.line(cvs, (int(frame[19][0]), int(frame[19][1])), (int(frame[20][0]), int(frame[20][1])), color, 2)
        cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[7][0]), int(frame[7][1])), color, 2)
        cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[12][0]), int(frame[12][1])), color, 2)
        cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[16][0]), int(frame[16][1])), color, 2)
        cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])), (int(frame[7][0]), int(frame[7][1])), color, 2)
        cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])), (int(frame[12][0]), int(frame[12][1])), color, 2)
        cv2.line(cvs, (int(frame[7][0]), int(frame[7][1])), (int(frame[16][0]), int(frame[16][1])), color, 2)
        cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])), (int(frame[16][0]), int(frame[16][1])), color, 2)

        '''
        for j in range(23):
            cv2.putText(cvs,str(j),(int(frame[j][0]),int(frame[j][1])),cv2.FONT_HERSHEY_SIMPLEX,.4, (155, 0, 255), 1, False)
            '''
        #cv2.imshow('canvas',np.flip(cvs,0))
        #cv2.waitKey(0)
        ncvs = np.flip(cvs, 0).copy()
        # draw_skeleton_number(ncvs, frame)
        video.write(np.uint8(ncvs))
    video.release()
    pass

def load_start_end_frame_num(config_fp):
    with open(config_fp, 'r') as f:
        data = json.load(f)
        start = data["start_position"]
        end = data["end_position"]
        return start,end
    pass
def load_skeleton(skeleton_json):
    with open(skeleton_json, 'r') as f:
        data = json.load(f)
        return data['length'],data['center'],data['skeletons']
    pass
def load_data_dict(data_list, name):
    i = 0
    while i < len(data_list):
        if data_list[i].split('_')[1] == 'C':
            if '0' not in data_dict:
                data_dict['0'] = [name + '/' + data_list[i]]
            else:
                data_dict['0'].append(name + '/' + data_list[i])
        elif data_list[i].split('_')[1] == 'R':
            if '1' not in data_dict:
                data_dict['1'] = [name + '/' + data_list[i]]
            else:
                data_dict['1'].append(name + '/' + data_list[i])
        elif data_list[i].split('_')[1] == 'T':
            if '2' not in data_dict:
                data_dict['2'] = [name + '/' + data_list[i]]
            else:
                data_dict['2'].append(name + '/' + data_list[i])
        else:
            if '3' not in data_dict:
                data_dict['3'] = [name + '/' + data_list[i]]
            else:
                data_dict['3'].append(name + '/' + data_list[i])
        i += 2

if __name__ == '__main__':
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    train_list = os.listdir(os.path.join(data_dirs, 'train'))
    val_list = os.listdir(os.path.join(data_dirs, 'val'))
    test_list = os.listdir(os.path.join(data_dirs, 'test'))
    test_list.sort()
    data_dict = {}
    load_data_dict(train_list, 'train')
    load_data_dict(val_list, 'val')
    load_data_dict(test_list, 'test')
    ensemble = open('./result/ensemble_3000.txt')
    e = ensemble.readline().rstrip()
    e_line_tre = e[0]
    e_line_pre = e[2]
    result_list = []
    i = 0
    while i < len(test_list):
        true_skeletons = os.path.join(data_dirs, 'test', test_list[i])
        true_data = np.load(true_skeletons)
        vp = os.path.join(outputs_path,test_list[i].split('.')[0])
        if not os.path.exists(vp):
            os.makedirs(vp)

        video_path = os.path.join(vp, 'gt.mp4')
        draw(true_data, video_path)
        gt_audio = os.path.join(data_dirs, 'test', test_list[i + 1])
        # 将真实的节点值与音乐保存到输出文件夹中
        shutil.copyfile(true_skeletons, os.path.join(vp, 'true_ske.npy'))
        shutil.copyfile(gt_audio, os.path.join(vp, 'true_audio.wav'))

        out_gt_path = os.path.join(vp, 'groundtruth.mp4')
        audio = AudioFileClip(gt_audio)  # 这里可以debug一下，看一下参数，返回值
        print('Analyzed the audio, found a period of %.02f seconds' % audio.duration)
        video = VideoFileClip(video_path, audio=False)
        video = video.set_audio(audio)
        video.write_videofile(out_gt_path)

        temp_list = data_dict[e_line_pre][:]

        recomd_list = random.sample(temp_list,3)
        for j in range(3):
            pre_audio = os.path.join(data_dirs, recomd_list[j].split('.')[0] + '.wav')
            shutil.copyfile(pre_audio, os.path.join(vp, 'predict_audio' + str(j) + '.wav'))
            out_pre_path = os.path.join(vp, str(j) + '_'+ str(e_line_tre) + '_predict_' + str(e_line_pre) + '.mp4')
            audio = AudioFileClip(pre_audio)  # 这里可以debug一下，看一下参数，返回值
            sub = audio
            # sub = audio.subclip(start / fps, end / fps)
            print('Analyzed the audio, found a period of %.02f seconds' % sub.duration)
            video = VideoFileClip(video_path, audio=False)
            video = video.set_audio(sub)
            video.write_videofile(out_pre_path)
        i += 2
        e = ensemble.readline().rstrip()
        e_line_tre = e[0]
        e_line_pre = e[2]