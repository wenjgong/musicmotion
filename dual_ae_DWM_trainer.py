"""
motion encoder网络和music autoencoder网络先分开训练再结合起来
This file regroups the procedures for training the neural networks.
A training uses a configuration json file (e.g. configs/dual_ae_c.json).
"""
from pathlib import Path
from itertools import chain
import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import os
import json
import numpy as np
from models.motion_aagcn import Model
#更改修改文件的名称
# from models.audio_encoder_dropout import *
from models.audio_encoder_fc import *
from utils import kullback_leibler, contrastive_loss,XSigmoidLoss
from matplotlib import  pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass

class DualAETrainer():
    def __init__(self, params):
        self.params = params
        self.audio_encoder = None
        self.audio_decoder = None
        self.motion_model = None
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.learning_rate = params['learning_rate']
        self.device = torch.device(params['device'])
        self.experiment_name = params['experiment_name']
        self.log_interval = params['log_interval']
        self.save_model_every = params['save_model_every']
        self.iteration_idx = 0
        self.last_epoch = 0
        self.motion_loss_fun = nn.CrossEntropyLoss().to(self.device)
        self.music_loss = nn.MSELoss(reduce=True, size_average=True).to(self.device)
        self.pair_loss = nn.MSELoss(reduce=True, size_average=True).to(self.device)
        self.fc_audio = Sequential(
            Linear(256, 512, bias=False),
            Linear(512, 512, bias=False),
            BatchNorm1d(512),
            ReLU(),
            Linear(512, 1024, bias=False),
            Linear(1024, 1024, bias=False),
            Linear(1024, 2048, bias=False),
            Linear(2048, 2048, bias=False),
            Linear(2048, 4096, bias=False),
            Linear(4096, 4096, bias=False),
            Linear(4096, 2048, bias=False),
            Linear(2048, 2048, bias=False),
            Linear(2048, 1024, bias=False),
            Linear(1024, 512, bias=False),
            BatchNorm1d(512),
            Linear(512, 256, bias=False),
            BatchNorm1d(256),
            ReLU(),
        )
        self.fc_audio = self.fc_audio.to(self.device)

    def init_models(self):
        self.audio_encoder = AudioEncoder()
        self.audio_decoder = AudioDecoder()
        self.motion_model = Model()

    def load_model_checkpoints(self):
        saved_models_folder = Path('saved_models', self.experiment_name)
        try:
            last_epoch = 4900
            self.audio_encoder.load_state_dict(torch.load(str(Path(f'saved_models', self.experiment_name, f'audio_encoder_epoch_{last_epoch}.pt'))))
            self.audio_decoder.load_state_dict(torch.load(str(Path(f'saved_models', self.experiment_name, f'audio_decoder_epoch_{last_epoch}.pt'))))
            self.motion_model.load_state_dict(torch.load(str(Path(f'saved_models', self.experiment_name, f'motion_model_epoch_{last_epoch}.pt'))))
            print(f'Model checkpoints from epoch {last_epoch} loaded...')
        except FileNotFoundError:
            last_epoch = 0
            print('No model loaded, training from scratch...')

        self.iteration_idx = last_epoch * int(self.length_test_dataset / self.batch_size)
        self.last_epoch = last_epoch


    def train(self,data_dirs):
        """ Train the dual Auto Encoder
        """
        train_music = np.load(os.path.join(data_dirs, 'train_mfcc.npy'))
        #更改训练的模型
        train_motion = np.load(os.path.join(data_dirs, 'train_data.npy'))
        # train_motion = np.load(os.path.join(data_dirs, 'train_bone.npy'))
        # train_motion = np.load(os.path.join(data_dirs, 'train_data.npy'))
        with open(os.path.join(data_dirs,'train_label.pkl'), 'rb') as f:
            train_label = pickle.load(f)
        train_label = train_label[1]
        val_music = np.load(os.path.join(data_dirs, 'val_mfcc.npy'))
        #更改验证的模型
        val_motion = np.load(os.path.join(data_dirs, 'val_data.npy'))
        # val_motion = np.load(os.path.join(data_dirs, 'val_bone.npy'))
        # val_motion = np.load(os.path.join(data_dirs, 'val_data.npy'))
        with open(os.path.join(data_dirs,'val_label.pkl'), 'rb') as f:
            val_label = pickle.load(f)
        val_label = val_label[1]
        loader_params = {
            'batch_size': self.batch_size, 
            'shuffle': True, 
            'num_workers': 1,
            'drop_last': True,
        }

        train_music_features = np.empty([0, 1, 96, 25])
        #更改数据维度 data数据 3->6  AIST数据集 23->17
        train_motion_features = np.empty([0, 25, 17, 6])
        train_motion_label = []

        for i in range(len(train_music)):
            music = train_music[i]
            motion = train_motion[i]
            motion = np.transpose(motion, [1,2,0])
            label = train_label[i]
            x_chunks = np.array([music[i * 25:(i + 1) * 25].T for i in range(music.shape[0] // 25)])
            x_chunks = torch.unsqueeze(torch.tensor(x_chunks), 1)
            train_music_features = np.append(train_music_features, x_chunks, axis=0)
            motion_chunks = np.array([motion[i * 25:(i + 1) * 25] for i in range(x_chunks.shape[0])])
            for i in range(x_chunks.shape[0]):
                train_motion_label.append(label)
            train_motion_features = np.append(train_motion_features, motion_chunks, axis=0)

        train_motion_label = torch.Tensor(train_motion_label)
        train_motion_features = torch.from_numpy(train_motion_features)
        train_music_features = torch.from_numpy(train_music_features)
        assert len(train_motion_features) == len(train_music_features)
        train_dataset = TensorDataset(train_music_features, train_motion_features, train_motion_label)
        self.train_loader = DataLoader(dataset=train_dataset, **loader_params)
        '''
        先划分数据集
        '''
        val_music_features = np.empty([0, 1, 96, 25])
        # 更改数据维度 data数据 3->6  AIST数据集 23->17
        val_motion_features = np.empty([0, 25, 17, 6])
        val_motion_label = []

        for i in range(len(val_music)):
            music = val_music[i]
            motion = val_motion[i]
            motion = np.transpose(motion, [1, 2, 0])
            label = val_label[i]
            x_chunks = np.array([music[i * 25:(i + 1) * 25].T for i in range(music.shape[0] // 25)])
            x_chunks = torch.unsqueeze(torch.tensor(x_chunks), 1)
            val_music_features = np.append(val_music_features, x_chunks, axis=0)
            motion_chunks = np.array([motion[i * 25:(i + 1) * 25] for i in range(x_chunks.shape[0])])
            for i in range(x_chunks.shape[0]):
                val_motion_label.append(label)
            val_motion_features = np.append(val_motion_features, motion_chunks, axis=0)

        val_motion_label = torch.Tensor(val_motion_label)
        val_motion_features = torch.from_numpy(val_motion_features)
        val_music_features = torch.from_numpy(val_music_features)
        assert len(val_motion_features) == len(val_music_features)
        val_dataset = TensorDataset(val_music_features, val_motion_features, val_motion_label)
        self.val_loader = DataLoader(dataset=val_dataset, **loader_params)
        self.length_train_dataset = len(self.train_loader.dataset)
        self.length_val_dataset = len(self.val_loader.dataset)
        print('Train dataset length: ', self.length_train_dataset)
        print('Val dataset length: ', self.length_val_dataset)
        # folder for model checkpoints
        model_checkpoints_folder = Path('saved_models', self.experiment_name)
        if not model_checkpoints_folder.exists():
            model_checkpoints_folder.mkdir()

        # models
        self.init_models()

        self.audio_encoder.to(self.device)
        self.audio_decoder.to(self.device)
        self.motion_model.to(self.device)

        # optimizers
        self.audio_dae_opt = optim.Adam(chain(self.audio_encoder.parameters(), self.audio_decoder.parameters()), lr=self.learning_rate)
        self.motion_opt = optim.Adam(self.motion_model.parameters(), lr=self.learning_rate)
        self.fc_opt = optim.Adam(self.fc_audio.parameters(), lr= self.learning_rate)
        # tensorboard
        with SummaryWriter(log_dir=str(Path('runs', self.experiment_name)), max_queue=100) as self.tb:

            # Training loop
            for epoch in range(0, self.epochs):
                self.train_one_epoch_dual_AE(epoch)
                self.val_dual_AE(epoch)

    def train_one_epoch_dual_AE(self, epoch):
        """ Train one epoch

        """
        self.audio_decoder.train()
        self.motion_model.train()
        self.audio_encoder.train()
        self.fc_audio.train()
        # losses
        train_loss = 0
        train_audio_recon_loss = 0
        train_motion_loss = 0
        train_pairwise_loss = 0
        train_motion_acc = 0

        for batch_idx, (music, motion, label) in enumerate(self.train_loader):
            self.audio_dae_opt.zero_grad()
            self.motion_opt.zero_grad()
            self.fc_opt.zero_grad()
            self.iteration_idx += 1
            x = music.view(-1, 1, 96, 25).to(self.device)
            y = motion.to(self.device)
            label = label.to(self.device)
            # 更改数据维度 data数据 3->6  AIST数据集 23->17
            y = y.reshape(y.shape[0], y.shape[1], 17, 6)
            y = y.permute(0, 3, 1, 2).unsqueeze(-1)


            motion_encoder, output = self.motion_model(y)
            z_audio = self.audio_encoder(x.float())
            motion_pre_loss = self.motion_loss_fun(output, label.long())
            value, predict_label = torch.max(output.detach(), 1)
            acc = torch.mean((predict_label == label.data).float())
            motion_pre_loss.backward(retain_graph=True)
            pairwise_loss = self.pair_loss(motion_encoder, z_audio)

            if epoch < 500:
                x_recon = self.audio_decoder(z_audio)
            else:
                #motion_encoder -> FC MODEL-> z_audio
                #增加一个全连接层，打印出来数据维度更改参数
                motion_encoder = self.fc_audio(motion_encoder)
                x_recon = self.audio_decoder(motion_encoder)
                pairwise_loss.backward(retain_graph = True)
            audio_recon_loss = self.music_loss(x_recon.float(), x.float())
            audio_recon_loss.backward(retain_graph=True)

            loss = audio_recon_loss + motion_pre_loss + pairwise_loss
            # Optimize models
            self.audio_dae_opt.step()
            self.motion_opt.step()
            self.fc_opt.step()
            train_audio_recon_loss += audio_recon_loss.item()
            train_motion_loss += motion_pre_loss.item()
            train_pairwise_loss += pairwise_loss.item()
            train_motion_acc += acc
            train_loss += loss.item()
            # write to tensorboard
            if False:
                self.tb.add_scalar("iter/audio_recon_loss", audio_recon_loss.item(), self.iteration_idx)
                self.tb.add_scalar("iter/motion_pre_loss", motion_pre_loss.item(), self.iteration_idx)
                self.tb.add_scalar("iter/contrastive_pairwise_loss", pairwise_loss.item(), self.iteration_idx)
                self.tb.add_scalar("iter/total_loss", loss.item(), self.iteration_idx)
                self.tb.add_scalar("iter/motion_acc", acc.item(), self.iteration_idx)


        # epoch logs
        train_loss = train_loss / self.length_train_dataset * self.batch_size
        train_audio_recon_loss = train_audio_recon_loss / self.length_train_dataset * self.batch_size
        train_motion_loss = train_motion_loss / self.length_train_dataset * self.batch_size
        train_pairwise_loss = train_pairwise_loss / self.length_train_dataset * self.batch_size
        train_motion_acc = train_motion_acc / self.length_train_dataset * self.batch_size
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
        print('recon loss audio: {:.4f}'.format(train_audio_recon_loss))
        print('recon loss motion: {:.4f}'.format(train_motion_loss))
        print('motion pre acc: {:.4f}'.format(train_motion_acc))
        print('pairwise loss: {:.8f}'.format(train_pairwise_loss))
        print('\n')

        # tensorboard
        self.tb.add_scalar("audio_recon_loss/train", train_audio_recon_loss, epoch)
        self.tb.add_scalar("motion_loss/train", train_motion_loss, epoch)
        self.tb.add_scalar("contrastive_pairwise_loss/train", train_pairwise_loss, epoch)
        self.tb.add_scalar("total_loss/train", train_loss, epoch)
        self.tb.add_scalar("total_acc/train", train_motion_acc.item(), epoch)

        if epoch%self.save_model_every == 0:
            torch.save(self.audio_encoder.state_dict(), str(Path(f'saved_models', self.experiment_name, f'audio_encoder_epoch_{epoch}.pt')))
            torch.save(self.audio_decoder.state_dict(), str(Path(f'saved_models', self.experiment_name, f'audio_decoder_epoch_{epoch}.pt')))
            torch.save(self.motion_model.state_dict(), str(Path(f'saved_models', self.experiment_name, f'motion_model_epoch_{epoch}.pt')))
            torch.save(self.fc_audio.state_dict(),
                       str(Path(f'saved_models', self.experiment_name, f'fc_model_epoch_{epoch}.pt')))


    def val_dual_AE(self, epoch):
        """ Validation dual autoencoder

        """
        self.audio_encoder.eval()
        self.audio_decoder.eval()
        self.motion_model.eval()
        self.fc_audio.eval()
        val_audio_recon_loss = 0
        val_motion_loss = 0
        val_loss = 0
        val_pairwise_loss = 0
        val_motion_acc = 0

        with torch.no_grad():
            for batch_idx, (music, motion, motion_label) in enumerate(self.val_loader):
                x = music.view(-1, 1, 96, 25).to(self.device)
                y = motion.to(self.device)
                motion_label = motion_label.to(self.device)
                # 更改数据维度 data数据 3->6  AIST数据集 23->17
                y = y.reshape(y.shape[0], y.shape[1], 17, 6)
                y = y.permute(0, 3, 1, 2).unsqueeze(-1)

                motion_encoder, output = self.motion_model(y)
                z_audio = self.audio_encoder(x.float())
                motion_pre_loss = self.motion_loss_fun(output, motion_label.long())
                value, predict_label = torch.max(output.detach(), 1)
                acc = torch.mean((predict_label == motion_label.data).float())
                pairwise_loss = self.pair_loss(motion_encoder, z_audio)
                #
                if epoch < 500:
                    x_recon = self.audio_decoder(z_audio)
                else:
                    motion_encoder = self.fc_audio(motion_encoder)
                    x_recon = self.audio_decoder(motion_encoder)

                audio_recon_loss = self.music_loss(x_recon.float(), x.float())

                loss = audio_recon_loss + motion_pre_loss + pairwise_loss

                val_audio_recon_loss += audio_recon_loss.item()
                val_motion_loss += motion_pre_loss.item()
                val_pairwise_loss += pairwise_loss.item()
                val_motion_acc += acc
                val_loss += loss.item()

        val_motion_acc = val_motion_acc / self.length_val_dataset * self.batch_size
        val_loss = val_loss / self.length_val_dataset * self.batch_size
        val_audio_recon_loss = val_audio_recon_loss / self.length_val_dataset * self.batch_size
        val_motion_loss = val_motion_loss / self.length_val_dataset * self.batch_size
        val_pairwise_loss = val_pairwise_loss / self.length_val_dataset * self.batch_size

        print('====> Val average loss: {:.4f}'.format(val_loss))
        print('recon loss audio: {:.4f}'.format(val_audio_recon_loss))
        print('recon loss motion: {:.4f}'.format(val_motion_loss))
        print('motion pre acc: {:.4f}'.format(val_motion_acc))
        print('pairwise loss: {:.4f}'.format(val_pairwise_loss))
        print('\n\n')

        # tensorboard
        self.tb.add_scalar("audio_recon_loss/val", val_audio_recon_loss, epoch)
        self.tb.add_scalar("motion_loss/val", val_motion_loss, epoch)
        self.tb.add_scalar("contrastive_pairwise_loss/val", val_pairwise_loss, epoch)
        self.tb.add_scalar("total_loss/val", val_loss, epoch)
        self.tb.add_scalar("total_acc/val", val_motion_acc, epoch)


    def test(self, data_dirs):
        """ Train the dual Auto Encoder
        """
        # Data loaders

        test_music = np.load(os.path.join(data_dirs, 'test_mfcc.npy'))
        # 更改数据名称
        test_motion = np.load(os.path.join(data_dirs, 'test_data.npy'))
        # test_motion = np.load(os.path.join(data_dirs, 'test_bone.npy'))
        # test_motion = np.load(os.path.join(data_dirs, 'test_data.npy'))
        with open(os.path.join(data_dirs,'test_label.pkl'), 'rb') as f:
            test_label = pickle.load(f)
        test_label = test_label[1]
        loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 1,
            'drop_last': True,
        }

        test_music_features = np.empty([0, 1, 96, 25])
        # 更改数据维度 data数据 3->6  AIST数据集 23->17
        test_motion_features = np.empty([0, 25, 17, 6])
        test_motion_label = []

        for i in range(len(test_music)):
            music = test_music[i]
            motion = test_motion[i]
            motion = np.transpose(motion, [1, 2, 0])
            label = test_label[i]
            x_chunks = np.array([music[i * 25:(i + 1) * 25].T for i in range(music.shape[0] // 25)])
            x_chunks = torch.unsqueeze(torch.tensor(x_chunks), 1)
            test_music_features = np.append(test_music_features, x_chunks, axis=0)
            motion_chunks = np.array([motion[i * 25:(i + 1) * 25] for i in range(x_chunks.shape[0])])
            for i in range(x_chunks.shape[0]):
                test_motion_label.append(label)
            test_motion_features = np.append(test_motion_features, motion_chunks, axis=0)

        test_motion_label = torch.Tensor(test_motion_label)
        test_motion_features = torch.from_numpy(test_motion_features)
        test_music_features = torch.from_numpy(test_music_features)
        assert len(test_motion_features) == len(test_music_features)
        train_dataset = TensorDataset(test_music_features, test_motion_features, test_motion_label)
        self.test_loader = DataLoader(dataset=train_dataset, **loader_params)


        self.length_test_dataset = len(self.test_loader.dataset)
        print('Test dataset length: ', self.length_test_dataset)

        # models
        self.init_models()
        # 加载预先训练好的music-ae和motion子网络
        self.motion_model.load_state_dict(
        #更改训练的epoch 1400  2900 以及文件夹名称
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_FC', 'motion_model_epoch_2900.pt'))))
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'motion_model_epoch_2900.pt'))))
            torch.load(str(Path(f'saved_models', 'AIST_DATA_FC', 'motion_model_epoch_900.pt'))))
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'motion_model_epoch_2900.pt'))))
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'motion_model_epoch_1400.pt'))))
        self.audio_encoder.load_state_dict(
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_FC', 'audio_encoder_epoch_2900.pt'))))
            torch.load(str(Path(f'saved_models', 'AIST_DATA_FC', 'audio_encoder_epoch_900.pt'))))
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'audio_encoder_epoch_2900.pt'))))
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'audio_encoder_epoch_2900.pt'))))
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'audio_encoder_epoch_1400.pt'))))
        self.audio_decoder.load_state_dict(
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_FC', 'audio_decoder_epoch_2900.pt'))))
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'audio_decoder_epoch_2900.pt'))))
            torch.load(str(Path(f'saved_models', 'AIST_DATA_FC', 'audio_decoder_epoch_900.pt'))))
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'audio_decoder_epoch_2900.pt'))))
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'audio_decoder_epoch_1400.pt'))))
        self.fc_audio.load_state_dict(
            torch.load(str(Path(f'saved_models', 'AIST_DATA_FC', 'fc_model_epoch_900.pt'))))
        self.audio_encoder.to(self.device)
        self.audio_decoder.to(self.device)
        self.motion_model.to(self.device)
        # loss for tag autoencoder

        self.audio_encoder.eval()
        self.audio_decoder.eval()
        self.motion_model.eval()
        self.fc_audio.eval()
        test_audio_recon_loss = 0
        test_motion_loss = 0
        test_loss = 0
        test_pairwise_loss = 0
        test_motion_acc = 0
        predict_list = []
        true_list = []
        with torch.no_grad():
            for batch_idx, (music, motion, motion_label) in enumerate(self.test_loader):
                x = music.view(-1, 1, 96, 25).to(self.device)
                y = motion.to(self.device)
                motion_label = motion_label.to(self.device)
                # encode
                z_audio = self.audio_encoder(x.float())
                #更改数据维度 data数据 3->6  AIST数据集 23->17
                y = y.reshape(y.shape[0], y.shape[1], 17, 6)
                y = y.permute(0, 3, 1, 2).unsqueeze(-1)

                motion_encoder, output = self.motion_model(y)

                motion_pre_loss = self.motion_loss_fun(output, motion_label.long())
                value, predict_label = torch.max(output.detach(), 1)
                acc = torch.mean((predict_label == motion_label.data).float())
                # audio reconstruction
                predict = output.cpu().numpy()
                true = motion_label.data.cpu().numpy()
                true_list.append(true)
                predict_list.append(predict)
                x_recon = self.audio_decoder(z_audio)
                audio_recon_loss = self.music_loss(x_recon, x)
                # contrastive loss
                pairwise_loss = self.pair_loss(motion_encoder, z_audio)

                # total loss
                loss = audio_recon_loss + motion_pre_loss + pairwise_loss

                test_audio_recon_loss += audio_recon_loss.item()
                test_motion_loss += motion_pre_loss.item()
                test_loss += loss.item()
                test_pairwise_loss += pairwise_loss.item()
                test_motion_acc += acc
        #更改对应的文件名
        # with open('./result/M2D_result_joint_dropout_1500.pkl', 'wb') as f:
        # with open('./result/M2D_result_joint_fc_1500.pkl', 'wb') as f:
        # with open('./result/M2D_result_bone_dropout_1500.pkl', 'wb') as f:
        # with open('./result/M2D_result_bone_dropout_3000.pkl', 'wb') as f:
        # with open('./result/M2D_result_data_dropout_3000.pkl', 'wb') as f:
        # with open('./result/M2D_result_joint_dropout_3000.pkl', 'wb') as f:
        # with open('./result/M2D_result_joint_dropout_3000.pkl', 'wb') as f:
        with open('./result/AIST_DATA_FC_1000.pkl', 'wb') as f:
            pickle.dump(predict_list, f)
        with open('./result/AIST_true_label.pkl', 'wb') as f:
            pickle.dump(true_list, f)
        test_motion_acc = test_motion_acc / self.length_test_dataset
        test_loss = test_loss / self.length_test_dataset
        test_audio_recon_loss = test_audio_recon_loss / self.length_test_dataset
        test_motion_loss = test_motion_loss / self.length_test_dataset
        test_pairwise_loss = test_pairwise_loss / self.length_test_dataset

        print('====> Test average loss: {:.4f}'.format(test_loss))
        print('recon loss audio: {:.4f}'.format(test_audio_recon_loss))
        print('motion pre acc: {:.4f}'.format(test_motion_acc))
        print('recon loss motion: {:.4f}'.format(test_motion_loss))
        print('pairwise loss: {:.4f}'.format(test_pairwise_loss))
        print('\n\n')

    def test1(self, data_dirs):
        """ Train the dual Auto Encoder
        """
        # Data loaders

        test_music = np.load(os.path.join(data_dirs, 'test_mfcc.npy'))
        test_motion = np.load(os.path.join(data_dirs, 'test_data.npy'))
        with open(os.path.join(data_dirs, 'test_label.pkl'), 'rb') as f:
            test_label = pickle.load(f)
        test_label = test_label[1]
        loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 1,
            'drop_last': True,
        }
        # models
        self.init_models()
        # 加载预先训练好的music-ae和motion子网络
        self.motion_model.load_state_dict(
            # 更改训练的epoch 1400  2900 以及文件夹名称
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_FC', 'motion_model_epoch_2900.pt'))))
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'motion_model_epoch_2900.pt'))))
            torch.load(str(Path(f'saved_models', 'AIST_DATA_FC', 'motion_model_epoch_900.pt'))))
        # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'motion_model_epoch_2900.pt'))))
        # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'motion_model_epoch_1400.pt'))))
        self.audio_encoder.load_state_dict(
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_FC', 'audio_encoder_epoch_2900.pt'))))
            torch.load(str(Path(f'saved_models', 'AIST_DATA_FC', 'audio_encoder_epoch_900.pt'))))
        # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'audio_encoder_epoch_2900.pt'))))
        # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'audio_encoder_epoch_2900.pt'))))
        # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'audio_encoder_epoch_1400.pt'))))
        self.audio_decoder.load_state_dict(
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_FC', 'audio_decoder_epoch_2900.pt'))))
            # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'audio_decoder_epoch_2900.pt'))))
            torch.load(str(Path(f'saved_models', 'AIST_DATA_FC', 'audio_decoder_epoch_900.pt'))))
        # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'audio_decoder_epoch_2900.pt'))))
        # torch.load(str(Path(f'saved_models', 'M2D_JOINT_DROPOUT', 'audio_decoder_epoch_1400.pt'))))
        self.fc_audio.load_state_dict(
            torch.load(str(Path(f'saved_models', 'AIST_DATA_FC', 'fc_model_epoch_900.pt'))))

        self.audio_encoder.to(self.device)
        self.audio_decoder.to(self.device)
        self.motion_model.to(self.device)
        # self.fc_audio.to(self.device)
        # loss for tag autoencoder

        self.audio_encoder.eval()
        self.audio_decoder.eval()
        self.motion_model.eval()
        self.fc_audio.eval()
        text_path = 'result/AIST_DATA_FC_ALL_1000.txt'
        f_w = open(text_path, 'w')
        count = 0
        for i in range(len(test_music)):
            test_music_features = np.empty([0, 1, 96, 25])
            test_motion_features = np.empty([0, 25, 17, 6])
            test_motion_label = []

            music = test_music[i]
            motion = test_motion[i]
            motion = np.transpose(motion, [1, 2, 0])
            label = test_label[i]
            x_chunks = np.array([music[i * 25:(i + 1) * 25].T for i in range(music.shape[0] // 25)])
            x_chunks = torch.unsqueeze(torch.tensor(x_chunks), 1)
            test_music_features = np.append(test_music_features, x_chunks, axis=0)
            motion_chunks = np.array([motion[i * 25:(i + 1) * 25] for i in range(x_chunks.shape[0])])
            for i in range(x_chunks.shape[0]):
                test_motion_label.append(label)
            test_motion_features = np.append(test_motion_features, motion_chunks, axis=0)
            test_motion_label = torch.Tensor(test_motion_label)
            test_motion_features = torch.from_numpy(test_motion_features)
            test_music_features = torch.from_numpy(test_music_features)
            assert len(test_motion_features) == len(test_music_features)
            train_dataset = TensorDataset(test_music_features, test_motion_features, test_motion_label)
            self.test_loader = DataLoader(dataset=train_dataset, **loader_params)

            self.length_test_dataset = len(self.test_loader.dataset)
            print('Test dataset length: ', self.length_test_dataset)
            test_audio_recon_loss = 0
            test_motion_loss = 0
            test_loss = 0
            test_pairwise_loss = 0
            test_motion_acc = 0
            # predict_list = []
            # true_list = []
            predict_list = []
            with torch.no_grad():
                for batch_idx, (music, motion, motion_label) in enumerate(self.test_loader):
                    x = music.view(-1, 1, 96, 25).to(self.device)
                    y = motion.to(self.device)
                    motion_label = motion_label.to(self.device)
                    # encode
                    z_audio = self.audio_encoder(x.float())
                    y = y.reshape(y.shape[0], y.shape[1], 17, 6)
                    y = y.permute(0, 3, 1, 2).unsqueeze(-1)
                    motion_encoder, output = self.motion_model(y)
                    motion_pre_loss = self.motion_loss_fun(output, motion_label.long())
                    value, predict_label = torch.max(output.detach(), 1)
                    acc = torch.mean((predict_label == motion_label.data).float())
                    # audio reconstruction
                    predict = output.cpu().numpy()
                    true = motion_label.data.cpu().numpy()
                    # true_list.append(true)
                    # predict_list.append(predict)
                    predict_list.append(predict_label.cpu().numpy())
                    x_recon = self.audio_decoder(z_audio)
                    audio_recon_loss = self.music_loss(x_recon, x)
                    # contrastive loss
                    pairwise_loss = self.pair_loss(motion_encoder, z_audio)

                    # total loss
                    loss = audio_recon_loss + motion_pre_loss + pairwise_loss

                    test_audio_recon_loss += audio_recon_loss.item()
                    test_motion_loss += motion_pre_loss.item()
                    test_loss += loss.item()
                    test_pairwise_loss += pairwise_loss.item()
                    test_motion_acc += acc
                print(max(predict_list, key=predict_list.count))
                if int(test_motion_label[0]) == max(predict_list, key=predict_list.count)[0]:
                    count += 1
                f_w.write(str(int(test_motion_label[0])) + ',' + str(
                    max(predict_list, key=predict_list.count)[0]) + '\n')  # 真，预测
            # with open('./result/aist_result_data_500.pkl', 'wb') as f:
            #     pickle.dump(predict_list, f)
            # with open('./result/aist_true_label.pkl', 'wb') as f:
            #     pickle.dump(true_list, f)
            test_motion_acc = test_motion_acc / self.length_test_dataset
            test_loss = test_loss / self.length_test_dataset
            test_audio_recon_loss = test_audio_recon_loss / self.length_test_dataset
            test_motion_loss = test_motion_loss / self.length_test_dataset
            test_pairwise_loss = test_pairwise_loss / self.length_test_dataset

            print('====> Test average loss: {:.4f}'.format(test_loss))
            print('recon loss audio: {:.4f}'.format(test_audio_recon_loss))
            print('motion pre acc: {:.4f}'.format(test_motion_acc))
            print('recon loss motion: {:.4f}'.format(test_motion_loss))
            print('pairwise loss: {:.4f}'.format(test_pairwise_loss))
            print('\n\n')
        print(count / len(test_music))
