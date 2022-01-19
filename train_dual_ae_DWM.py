import json
import argparse
from dual_ae_DWM_trainer import DualAETrainer
#测试集更改为false
train = False
# train = False
#数据集更改需要改变文件目录
data_dirs = './data/AIST_segment_single'

def main(config_file):
    params = json.load(open(config_file, 'rb'))
    print("Training Dual AutoEncoder with params:")
    print(json.dumps(params, separators=("\n", ": "), indent=4))
    trainer = DualAETrainer(params)
    if train:
        trainer.train(data_dirs)
    else:
        trainer.test(data_dirs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Dual Auto Encoder')
    parser.add_argument('--config_file', type=str,
                        help='configuration file for the training')         
    args = parser.parse_args()

    main(args.config_file)
