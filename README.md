# Multi-modal collaborative learning encoder for improving the accuracy of music recommendation

This is the repository of the method presented in the manuscript "Multi-modal collaborative learning encoder for improving the accuracy of
music recommendation" by Wenjuan Gong, Qingshuang Yu, etc. The manuscript is submitted to a peer review journal. 

![avatar](./model.png)

## Data preparation
1. Start by downloading the [MusicToDance](https://github.com/Music-to-dance-motion-synthesis/dataset) datasets and [AIST++](https://google.github.io/aistplusplus_dataset/download.html) datasets and put them under the /datasets folder.
2. Run the data feature extraction code.\
 `python3 ./data_pre/m2d_music_motion.py` Extract the joint features of the MusicToDance dataset and divide the train,valid,test.\
  `python3 ./data_pre/aist_moiton_music.py`Extract the joint features of the AIST++ dataset and divide the train,valid,test.\
  `python3 ./data_pre/gen_bone.py`Extract the bone features of the MusicToDance and AIST++ dataset and divide the train,valid,test.\
  `python3 ./data_pre/emerged_bone_joint.py`Emerged the joint features and the bone features of the MusicToDance and AIST++ dataset.\
3. The extracted features of the MusicToDance and AIST++ dataset held in the folder /data
4. We provide the data in [Baidu Netdisk](https://pan.baidu.com/s/1fTV7uZs4oQZwfyY1bWFaTQ?pwd=prmi)  Extraction code ：prmi
## Train the model
`python3 train_dual_ae_DWM.py --config_file ./configs/dual_ae_c.json`\
‘train’ parameter in the train_dual_ae_DWM.py determine whether to train or test.\
‘data_dirs’ parameter in the train_dual_ae_DWM.py determine which experiment to run.\
ensemble.py combines experimental result from joint and bone.\
./saved_models saved the experimental model.\
We provide the pre-trained model in [Baidu Netdisk](https://pan.baidu.com/s/1vILpFAsCVsob6LMxWxy_tw?pwd=prmi ) Extraction code: prmi

## Others
./confusion_matrix.py Get the confusion matrix of the experimental results.\
./draw_video.py Generate the music video.
