import numpy as np
import sys

import LiveStreamingEnv.load_trace as load_trace
from statistics import harmonic_mean
import video_env
sys.path.append("..")
_INFO = 7
S_LEN = 16
A_DIM = 4
ACTOR_LR_RATE = 1e-4
CRITIC_LR_RATE = 1e-3
frame_time_len = 0.04
reward_all_sum = 0
BIT_RATE = [500.0,850.0,1200.0,1850.0] # kpbs码率
TARGET_BUFFER = [2.0,3.0]   # seconds定义了两个不同的目标缓冲区大小，单位是秒（seconds）
frame_time_len = 0.04
# QOE setting
reward_frame = 0#当前帧的质量评价（QoE）奖励值
reward_all = 0
SMOOTH_PENALTY= 0.02
REBUF_PENALTY = 1.5#1.5
LANTENCY_PENALTY = 0.005
B_IN_MB=1000000.0
e=1850.0/500.0
MILLISECONDS_IN_SECOND = 1000.0

total_bandwidth = 0
choose_bit_rate = 0
bandwidth_ests= [0]
past_errors=[0]
TRAIN_TRACES = './network_trace/'  # +str("trace/")#+trace_name   #train trace path setting,
video_size_file = './video_trace/AsianCup_China_Uzbekistan/frame_trace_'  # video trace path setting,
LogFile_Path = "./log/"  # log file trace path setting,
# Debug Mode: if True, You can see the debug info in the logfile
#             if False, no log ,but the training speed is high
DEBUG = True
# load the trace
all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
def mpc(S_time_interval, S_send_data_size, download_id, last_bit_rate, S_buffer_size,bandwidth_ls,pere,rebuf):  # 估计网络情况
    curr_error = 0
    video_size_file = './video_trace/AsianCup_China_Uzbekistan/frame_trace_'  # video trace path setting,

    all_future_chunks_size=[0]
    past_bandwidths = [0]
    #for i in range(50):
       # time_interval = S_time_interval[-i]
       # data_size = S_send_data_size[-i]
        # 检查是否为非零元素，如果是，则添加到新的列表中
       # if time_interval != 0 and data_size != 0:
        #    past_bandwidth = data_size/time_interval/B_IN_MB
         #   past_bandwidths.append(past_bandwidth)
    #past_bandwidths=bandwidth_ls
    if (len(bandwidth_ests) > 0):
        # 获取 bandwidth_ests[-1]
        last_prediction = bandwidth_ests[-1]
        # 计算与 past_bandwidth 中每个值的差值的绝对值的最大值，将预测的误差的最大值作为本轮为误差
        curr_error = max([abs(last_prediction - bw) for bw in past_bandwidths])
        # 将最大值存入 past_errors
        past_errors.append(curr_error)
   #求调和平均值
    """
    def calculate_future_bandwidth(past_bandwidths):
        # 计算调和平均
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            if past_val != 0:
                bandwidth_sum += (1 / float(past_val))
        harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))
        # 计算未来带宽 未来带宽为调和带宽除以（1 + 过去五个误差的最大值）。这是为了稳定预测带宽
        #若有5个，则考虑过去5个块的最大值，若无则有多少考虑多少。
        max_error = 0
        error_pos = -5
        if (len(past_errors) < 5):
            error_pos = -len(past_errors)
        max_error = float(max(past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth / (1 + max_error)  # robustMPC here
        return future_bandwidth
    """
    future_bandwidth = harmonic_mean(bandwidth_ls)#pere#harmonic_mean(bandwidth_ls)#pere#harmonic_mean(bandwidth_ls)#pere#harmonic_mean(bandwidth_ls)#pere#harmonic_mean(bandwidth_ls)#harmonic_mean(bandwidth_ls)#pere#harmonic_mean(bandwidth_ls)#pere#harmonic_mean(bandwidth_ls)#pere#harmonic_mean(bandwidth_ls)
    #future_bandwidth = calculate_future_bandwidth(past_bandwidths)
    #print(future_bandwidth)
    '''
    choose_bit_rate = last_bit_rate
    bandwidth_ests.append(future_bandwidth)
    if future_bandwidth >= e*BIT_RATE[last_bit_rate]/B_IN_MB and last_bit_rate +1< 3:
        choose_bit_rate = last_bit_rate +1
    target_buffer = 1
    '''
    max_reward = float('-inf')
    start_buffer = S_buffer_size[-1]
    #print("buffer",start_buffer)
    video = video_env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw,VIDEO_SIZE_FILE=video_size_file)  # 视频数据传到video_env文件中解析
    for bit_rate in range(4):
        curr_rebuffer_time = 0
        curr_buffer = start_buffer  # ms
        bitrate_sum = 0
        rebuffer_time = 0
        smoothness_diffs = 0
        bitrate=0
        latency=0
        rebuf=0
        download_time=0
        for position in range(0, 25):#50
           frame_size,laten ,rebu = video.get_video_frame(bit_rate, download_id + position,1)#video.get_video_frame_size(bit_rate, download_id + position)
           #print("frame_size",frame_size)
           latency+=laten
           rebuf+=rebu
           bitrate += 0.04 * BIT_RATE[bit_rate] / 1000
          # download_time1 = (frame_size * 8) / (future_bandwidth * 1e6)
          # print()
           download_time   +=   ((frame_size)/1000000.0)/(future_bandwidth) # this is MB/MB/s --> seconds
           #print("down",download_time )
           curr_buffer += frame_time_len
          # print("cur",curr_buffer)
          # print("buff", curr_buffer, "dw", download_time)
           if (curr_buffer < download_time):
                curr_rebuffer_time = (download_time - curr_buffer)
                curr_buffer = 0
           else:
                curr_buffer -= download_time
        #print("bitr",bitrate)
        smoothness_diffs += abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate])
        #print("reb",rebuf, curr_rebuffer_time)
        #print("latency",0.005*latency)
        #print(bitrate,(BIT_RATE[bit_rate]/1000.),REBUF_PENALTY*rebuf,REBUF_PENALTY*curr_rebuffer_time/1000,smoothness_diffs)
         # reward = (BIT_RATE[bit_rate]/1000.) - (REBUF_PENALTY*sum(rebuf[-50:])/1000.) - (smoothness_diffs/1000.)
        reward = BIT_RATE[bit_rate]/1000 - (curr_rebuffer_time) - SMOOTH_PENALTY*(smoothness_diffs/1000.)#-LANTENCY_PENALTY *latency #(BIT_RATE[bit_rate] / 1000.) - (REBUF_PENALTY * sum(rebuf[-50:]) / 1.) - (SMOOTH_PENALTY*smoothness_diffs / 1000.)
        #print("MPC",reward, BIT_RATE[bit_rate]/1000, ( curr_rebuffer_time ),
             # SMOOTH_PENALTY * (  smoothness_diffs / 1000.), LANTENCY_PENALTY * latency)

        #(bit_rate/1000.) - (REBUF_PENALTY*curr_rebuffer_time/1000.) - (smoothness_diffs/1000.)
        #print("re",reward,max_reward)
        #判断这个码率是否为达到最大qoe的码率
        if ( reward >= max_reward ):
            choose_bit_rate=bit_rate
            max_reward = reward
    # choose_bit_rate为本轮所决定的码率
    target_buffer=1
    return choose_bit_rate, target_buffer


def simple_mpc(S_buffer_size, last_bit_rate, future_bandwidth, download_id, get_video_frame_size_fn):
    """
    简化的MPC算法，选择具有最高QoE奖励的码率。

    参数:
        S_buffer_size (float): 当前缓冲区大小（秒）
        last_bit_rate (int): 上一次选择的码率索引
        future_bandwidth (float): 未来带宽预测值（单位：Mbps）
        download_id (int): 当前下载帧的起始ID
        get_video_frame_size_fn (function): 获取指定帧大小的函数(bit_rate, frame_id) -> size(Bytes)

    返回:
        int: 选择的码率索引
        float: 目标缓冲值（此处固定为1.0秒）
    """
    max_reward = float('-inf')
    best_bit_rate = 0

    for bit_rate in range(len(BIT_RATE)):
        curr_buffer = S_buffer_size
        rebuffer_time = 0
        smoothness_diff = abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate])

        for pos in range(50):  # 预估接下来50帧
            frame_size = get_video_frame_size_fn(bit_rate, download_id + pos)  # 单位：Bytes
            download_time = (frame_size * 8) / (future_bandwidth * 1e6)  # Bytes -> bits，再除Mbps -> 秒

            if curr_buffer < download_time:
                rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0
            else:
                curr_buffer -= download_time

            curr_buffer += frame_time_len

        reward = (BIT_RATE[bit_rate] / 1000.0) - (REBUF_PENALTY * rebuffer_time) - (SMOOTH_PENALTY * smoothness_diff / 1000.0)

        if reward > max_reward:
            max_reward = reward
            best_bit_rate = bit_rate

    return best_bit_rate, 1.0  # 固定目标缓冲为1.0秒