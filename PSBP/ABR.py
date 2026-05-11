import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model 
from MPC import mpc
from statistics import harmonic_mean

import torch

from models import PatchTST

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='PatchTST',  # DLinear PatchTST FEDformer
                    help='model name, options: [Autoformer, Informer, Transformer, FEDformer]')

# data loader
parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')  # ETTm1 custom bandwidth
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='xy1.csv', help='data file')
parser.add_argument('--features', type=str, default='MS',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='avg', help='target feature in S or MS task')  # label
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=3, help='input sequence length')  # 96  4
parser.add_argument('--label_len', type=int, default=1, help='start token length')  # 48
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')  # 96

# DLinear
#  parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')  # 0.05
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=4, help='patch length')  # 5
parser.add_argument('--stride', type=int, default=2, help='stride')  # 8
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers
parser.add_argument('--embed_type', type=int, default=0,
                    help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=32,
                    help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=32, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')  # 512 768
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')  # 8
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')  # 2048  3072
parser.add_argument('--moving_avg', type=int, default=50, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')  # 0.05
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=70, help='train epochs')  # 100
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')  # 128
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')  # 20
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
args = parser.parse_args()
# random seed
fix_seed = args.random_seed
#random.seed(fix_seed)
#torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
#args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
print('Args in experiment:')
print(args)
S_INFO = 7
S_LEN = 16
A_DIM = 8
ACTOR_LR_RATE = 1e-4
CRITIC_LR_RATE = 1e-3
last_bit_rate = 0
#"./bandwidth_new_seg.npz.h5"
class Algorithm:
    def __init__(self,model_path="./bandwidth_seg.npz.h5",use_gru=False):#"./bandwidth_seg.npz.h5"./bandwidth_norway.npz.h5
        # fill your init vars
        self.buffer_size = 0
        #if use_gru:
        self.model=load_model(model_path)
        self.args=args
        self.model_dict = {
           # 'Autoformer': Autoformer,
            #'Transformer': Transformer,
            #'Informer': Informer,
            #'DLinear': DLinear,
            #'NLinear': NLinear,
            #'Linear': Linear,
           'PatchTST': PatchTST,
        }
        self.last_bit_rate = 0
        self.model_avg = self.model_dict[self.args.model].Model(self.args).float()

        self.model_avg.load_state_dict(torch.load("utils/checkpoint.pth"))#saved_models/checkpoint.pth QoE_data/checkpoint.pth
        self.model_avg.eval()
    # if you use Machine Learning ,intial your parameter
    def Initial(self):
        # Initail your session or something
        IntialVars = []
        return IntialVars

    def predict_throughput_ewma1(self,bandwidth_history, alpha=0.5):
        """
        使用EWMA预测带宽。

        参数:
            bandwidth_history (list or array-like): 历史带宽值，按时间顺序排列。
            alpha (float): 平滑系数，0 < alpha <= 1，越大越偏向最新值。

        返回:
            float: 下一时刻的带宽预测值。
        """
        if not bandwidth_history:
            raise ValueError("bandwidth_history 不能为空")

        ewma = bandwidth_history[0]
        for bw in bandwidth_history[1:]:
            ewma = alpha * bw + (1 - alpha) * ewma
        return ewma
    def predict_throughput_ewma2(self,bandwidth_history, alpha=0.03):
        """
        使用EWMA预测带宽。

        参数:
            bandwidth_history (list or array-like): 历史带宽值，按时间顺序排列。
            alpha (float): 平滑系数，0 < alpha <= 1，越大越偏向最新值。

        返回:
            float: 下一时刻的带宽预测值。
        """
        if not bandwidth_history:
            raise ValueError("bandwidth_history 不能为空")

        ewma = bandwidth_history[0]
        for bw in bandwidth_history[1:]:
            ewma = alpha * bw + (1 - alpha) * ewma
        return ewma
    def predict_throughput_mean(self, bw):
        if len(bw) <8:
            return sum(bw)/len(bw)
        else:
            return sum(bw[-8:])/8.0

    # Define your al
    def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,
            S_end_delay, S_decision_flag, S_buffer_flag, S_cdn_flag, end_of_video, cdn_newest_id, download_id,
            cdn_has_frame, IntialVars, bandwidth_ls, alg_option="TR_horm",video_chunk_size=[0,0,0,0]):

        # If you choose the machine learning
        '''actor = IntialVars[0]
        critic = IntialVars[1]
        state = []

        state[0] = ...
        state[1] = ...
        state[2] = ...
        state[3] = ...
        state[4] = ...

        decision = actor.predict(state).argmax()
        bit_rate, target_buffer = decison//4, decison % 4 .....
        return bit_rate, target_buffer'''
        #print(alg_option)
        # If you choose BBA  (RESEVOIR' =RESEVOIR, CUSHION’ = 2 * CUSHION - RESEVOIR)
        #print("S_time_interval",S_time_interval[-1])
        #print("S_play_time_len",S_play_time_len[-1])
        RESEVOIR = 0.4
        CUSHION = 1
        bit_rate = 1
        target_buffer = 1
        #print("time",time)
        #print("size",video_chunk_size)
        #print("bandwidth",bandwidth_ls[-1])
        #print("buffer", S_buffer_size[-1])

        #with open("bandwidth.txt", "a") as f1:
            #f1.write(f"{time}\t{str(bandwidth_ls[-1])}\n")
        # 打开第二个文件，写入2
       # with open("buffer.txt", "a") as f2:
            #f2.write(f"{time}\t{str(S_buffer_size[-1])}\n")
        if alg_option == "BBA":
            if S_buffer_size[-1] < RESEVOIR:
                bit_rate = 0
            elif S_buffer_size[-1] >= RESEVOIR + CUSHION:
                bit_rate = 2
            elif S_buffer_size[-1] >= CUSHION + CUSHION:
                bit_rate = 3
            else:
                bit_rate = 1
            # target_buffer = 1
            return bit_rate, target_buffer
        elif alg_option == "TR_GRU":
            #pre_bw=self.predict_throughput_ewma(bandwidth)
            #pre_bw = self.predict_throughput_mean(bandwidth_ls)
            #hom_bw=harmonic_mean(bandwidth_ls)
            pre_bw = self.predict_throughput_ewma2(bandwidth_ls[-7:] )
            #pre_bw2 = self.predict_throughput_ewma2(bandwidth_ls)
            #if(pre_bw1>pre_bw2):
              #  pre_bw=pre_bw2
            #else :
               # pre_bw=pre_bw1
            #pre_bw = hom_bw
            #print("ca",pre_bw,hom_bw)
            #print("cc",hom_bw)
           #print("ban",bandwidth_ls[-1:],pre_bw)
            BITRATE_OPTIONS = [500.0, 850.0, 1200.0, 1850.0]
            selected_bitrate_index = 0

            #if len(bandwidth_ls) > 6:
               # inp = np.array(bandwidth_ls[-6:]).reshape(1, 6, 1)
                # try:
               # pre_bw = self.model.predict(inp, verbose=0)[0]
                # print(pre_bw)
                # except:
                #   print(inp)
                # exit(0)
            #else:
               # pre_bw = sum(bandwidth_ls) / len(bandwidth_ls)


            for i in range(len(BITRATE_OPTIONS) - 1, -1, -1):
                #print("ccc",BITRATE_OPTIONS[i]/1000)
                if pre_bw >= BITRATE_OPTIONS[i]/1000:
                    selected_bitrate_index = i
                    break
            #print("bitrate",selected_bitrate_index,pre_bw,bandwidth_ls[-1])
            return selected_bitrate_index, 1,pre_bw
        elif alg_option == "MPC":
            pre_bw = 1
            #pre_bw=self.predict_throughput_ewma(bandwidth_ls)
            #print("gru",pre_bw)

            if len(bandwidth_ls) > 6:
               inp = torch.from_numpy(np.array(bandwidth_ls[-6:]).reshape(1, 1, 6)).float()
               buffer_r=torch.from_numpy(np.array(S_buffer_size[-1]).reshape(1, 1)).float()
                #print("okok")
               try:
                   pre_bw=self.model_avg(inp,buffer_r)[0]
               except:
               # #print("re",bandwidth_ls[-1:])
                #print("inp",inp)
                  #  print("sh",pre_bw)
                    exit(0)
            else:
                pre_bw=sum(bandwidth_ls)/len(bandwidth_ls)
            
            #print("harm",harmonic_mean(bandwidth_ls))
            #print("ewam",self.predict_throughput_ewma(bandwidth_ls))
            #print("real",bandwidth_ls[-1])

            """
            if len(bandwidth_ls) > 5:
                    inp = np.array(bandwidth_ls[-5:]).reshape(1, 5, 1)
                    try:
                        pre_bw = self.model.predict(inp, verbose=0)[0]
                    # print(pre_bw)
                    except:
                       print(inp)
                    # exit(0)
            else:
                    pre_bw = sum(bandwidth_ls) / len(bandwidth_ls)
            """
            selected_bitrate_index,i=mpc(S_time_interval, S_send_data_size, download_id, self.last_bit_rate, S_buffer_size,bandwidth_ls, pre_bw,S_rebuf)

            self.last_bit_rate = selected_bitrate_index
            #print("bitrate",selected_bitrate_index)
            return selected_bitrate_index, 1,pre_bw


        elif alg_option=="TR_with_SSS-PBP":
            #print("sxsxs")
            if len(bandwidth_ls) > 6:
               inp = torch.from_numpy(np.array(bandwidth_ls[-6:]).reshape(1, 1, 6)).float()
               buffer_r=torch.from_numpy(np.array(S_buffer_size[-1]).reshape(1, 1)).float()
                #print("okok")
               try:
                   pre_bw=self.model_avg(inp,buffer_r)[0]
               except:
               # #print("re",bandwidth_ls[-1:])
                #print("inp",inp)
                  #  print("sh",pre_bw)
                    exit(0)
            else:
                pre_bw=sum(bandwidth_ls)/len(bandwidth_ls)
            BITRATE_OPTIONS = [500.0, 850.0, 1200.0, 1850.0]
            selected_bitrate_index = 0

            for i in range(len(BITRATE_OPTIONS) - 1, -1, -1):
                if pre_bw >= (BITRATE_OPTIONS[i]/1000):
                    selected_bitrate_index = i
                    break
            #print("selected_bitrate", selected_bitrate_index)
            return selected_bitrate_index, 1,pre_bw

        elif alg_option=="TR_with_SSS-PBP_2":



            if len(bandwidth_ls)>5:
                inp=np.array(bandwidth_ls[-5:]).reshape(1,5,1)
                #try:
                pre_bw=self.model.predict(inp,verbose=0)[0]
                #print(pre_bw)
                #except:
                 #   print(inp)
                   # exit(0)
            else:
                pre_bw=sum(bandwidth_ls)/len(bandwidth_ls)
            BITRATE_OPTIONS = [500.0, 850.0, 1200.0, 1850.0]
            selected_bitrate_index = 0
            #print("pre",pre_bw)
            for i in range(len(BITRATE_OPTIONS) - 1, -1, -1):
                if pre_bw >= BITRATE_OPTIONS[i]/1000:
                    selected_bitrate_index = i
                    break
            #print("selected_bitrate",selected_bitrate_index)

            return selected_bitrate_index, 1,pre_bw


        # If you choose other
        # ......

    def get_params(self):
        # get your params
        your_params = []
        return your_params
