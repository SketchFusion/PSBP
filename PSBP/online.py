''' Demo SDK for LiveStreaming
    Author Dan Yang
    Time 2018-10-15
    For LiveStreaming Game'''
import os

# import the env from pip

from plots import save_qoe_data
import LiveStreamingEnv.fixed_env as fixed_env
#import fixed_env
import LiveStreamingEnv.load_trace as load_trace
#import matplotlib.pyplot as plt
import time
import numpy as np
import ABR
# path setting
def test(user_id,trace_name='fcc'):
    print("trace_name",trace_name)
    #TRAIN_TRACES = '/home/game/test_sim_traces/'   #train trace path setting,
    #video_size_file = '/home/game/video_size_'      #video trace path setting,
    #LogFile_Path = "/home/game/log/"                #log file trace path setting,
    print("ok")
    # TRAIN_TRACES = './network_trace/'   #train trace path setting,
    TRAIN_TRACES = './network_trace/'#+str("trace/")#+trace_name   #train trace path setting,
    video_size_file = './video_trace/AsianCup_China_Uzbekistan/frame_trace_'      #video trace path setting,
    LogFile_Path = "./log/"                #log file trace path setting,
    # Debug Mode: if True, You can see the debug info in the logfile
    #             if False, no log ,but the training speed is high
    DEBUG = True
    # load the trace
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
    #print("time",all_cooked_time)
    print("bandwidth trace",all_file_names)
    #random_seed 
    random_seed = 2
    count = 0
    video_count = 0
    FPS = 25
    frame_time_len = 0.04
    reward_all_sum = 0
    #init 
    #setting one:
    #     1,all_cooked_time : timestamp
    #     2,all_cooked_bw   : throughput
    #     3,all_cooked_rtt  : rtt
    #     4,agent_id        : random_seed
    #     5,logfile_path    : logfile_path
    #     6,VIDEO_SIZE_FILE : Video Size File Path
    #     7,Debug Setting   : Debug
    net_env = fixed_env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw,
                                  random_seed=random_seed,
                                  logfile_path=LogFile_Path,
                                  VIDEO_SIZE_FILE=video_size_file,
                                  Debug = DEBUG)
    
    abr = ABR.Algorithm()
    abr_init = abr.Initial()

    BIT_RATE      = [500.0,850.0,1200.0,1850.0] # kpbs
    TARGET_BUFFER = [2.0,3.0]   # seconds
    # ABR setting
    RESEVOIR = 0.5
    CUSHION  = 2

    cnt = 0
    # defalut setting
    last_bit_rate = 0
    bit_rate = 0
    target_buffer = 0

    # QOE setting
    reward_frame = 0
    reward_all = 0
    SMOOTH_PENALTY= 0.02
    REBUF_PENALTY = 1.5
    LANTENCY_PENALTY = 0.005
    # past_info setting
    past_frame_num  = 7500
    S_time_interval = [0] * past_frame_num
    S_send_data_size = [0] * past_frame_num
    S_chunk_len = [0] * past_frame_num
    S_rebuf = [0] * past_frame_num
    S_buffer_size = [0] * past_frame_num
    S_end_delay = [0] * past_frame_num
    S_chunk_size = [0] * past_frame_num
    S_play_time_len = [0] * past_frame_num
    S_decision_flag = [0] * past_frame_num
    S_buffer_flag = [0] * past_frame_num
    S_cdn_flag = [0] * past_frame_num
    # params setting
    chunk_lo = 0.0
    rebuf_r=0
    buf=0
    laten=0
    #bandwidth_file = open("./bandwidth_data/frame_bandwidth_{0}.txt".format(trace_name), "w+")
    bandwidth_file = open("./bandwidth_data/frame_bandwidth_systhesis.txt", "w+")
    frame_bandwidth_list=[]
    tt = 0
    while True:
        #reward_frame = 0

        #reward_chunk = 0
        # input the train steps
        # if cnt > 5000:
        #     plt.ioff()
        #     break
        # actions bit_rate  target_buffer
        # every steps to call the environment
        # time           : physical time 
        # time_interval  : time duration in this step
        # send_data_size : download frame data size in this step
        # chunk_len      : frame time len
        # rebuf          : rebuf time in this step          
        # buffer_size    : current client buffer_size in this step
        # play_time_len  : played time len  in this step          
        # end_delay      : end to end latency which means the (upload end timestamp - play end timestamp)
        # decision_flag  : Only in decision_flag is True ,you can choose the new actions, other time can't Becasuse the Gop is consist by the I frame and P frame. Only in I frame you can skip your frame
        # buffer_flag    : If the True which means the video is rebuffing , client buffer is rebuffing, no play the video
        # cdn_flag       : If the True cdn has no frame to get 
        # end_of_video   : If the True ,which means the video is over.
        time,time_interval, send_data_size, chunk_len,\
               rebuf, buffer_size, play_time_len,end_delay,\
                cdn_newest_id, download_id, cdn_has_frame, decision_flag,\
                buffer_flag, cdn_flag, end_of_video= net_env.get_video_frame(bit_rate,target_buffer)


        # S_info is sequential order
        S_time_interval.pop(0)
        S_send_data_size.pop(0)
        S_chunk_len.pop(0)
        S_buffer_size.pop(0)
        S_rebuf.pop(0)
        S_end_delay.pop(0)
        S_play_time_len.pop(0)
        S_decision_flag.pop(0)
        S_buffer_flag.pop(0)
        S_cdn_flag.pop(0)

        S_time_interval.append(time_interval)
        S_send_data_size.append(send_data_size)
        S_chunk_len.append(chunk_len)
        S_buffer_size.append(buffer_size)
        S_rebuf.append(rebuf)
        S_end_delay.append(end_delay)
        S_play_time_len.append(play_time_len)
        S_decision_flag.append(decision_flag)
        S_buffer_flag.append(buffer_flag)
        S_cdn_flag.append(cdn_flag)

        try:

            bw=(send_data_size/time_interval)/(1e6)
            print("bw",bw)
            #chunk_lo += float(send_data_size / bw)
            if bw > abs(1e-3):
                bandwidth_file.write("{0} {1} {2}\n".format(time,bw,decision_flag))
                frame_bandwidth_list.append(bw)

        except:
            pass

        # QOE setting
        #print("cdn_flag",cdn_flag)
        if not cdn_flag:
            #if (rebuf): print("reg1", rebuf)
            #print("lat",LANTENCY_PENALTY  * end_delay)
            rebuf_r +=  - REBUF_PENALTY * rebuf
            laten += - LANTENCY_PENALTY  * end_delay
            chunk_lo += time_interval
            buf+=buffer_size
            #reward_frame = frame_time_len * float(BIT_RATE[bit_rate]) / 1000  - REBUF_PENALTY * rebuf #- LANTENCY_PENALTY  * end_delay
            #reward_frame =  play_time_len*float(BIT_RATE[bit_rate]) / 1000 - REBUF_PENALTY * rebuf #- LANTENCY_PENALTY  * end_delay
            #print("reward_frame", BIT_RATE[bit_rate],rebuf,end_delay)
        else:
            #if(rebuf):print("reg2",rebuf)
            rebuf_r += - REBUF_PENALTY * rebuf
            chunk_lo += time_interval
            buf += buffer_size
            #reward_frame = -(REBUF_PENALTY * rebuf)
            #print("reward_frame", reward_frame)
        #reward_chunk = 0
        if decision_flag or end_of_video:
            #tt += 1
            # reward formate = play_time * BIT_RATE - 4.3 * rebuf - 1.2 * end_delay
            #if (rebuf): print("reg3", rebuf)
            #print("ref",rebuf_r,float(BIT_RATE[bit_rate]) / 1000 )

            reward_frame +=float(BIT_RATE[bit_rate]) / 1000 +rebuf_r-1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            #if end_of_video:
            #print("REWA", reward_frame, float(BIT_RATE[bit_rate]) / 1000,SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000), rebuf_r, laten)
            bitr=float(BIT_RATE[bit_rate]) / 1000
            lata=laten
            smot=SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            reb=rebuf_r
            #print(reward_frame)
            laten = 0
            rebuf_r = 0
            #print("play",time,time_interval,play_time_len)
            #reward_frame +=  -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            tt += 1
            #print("reward_frame2", reward_all)
           # reward_chunk += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            # last_bit_rate
            last_bit_rate = bit_rate
            # ----------------------  Your Althgrithom------------------------------------------
            # -------------------------------------------Your Althgrithom -------------------------------------------
            # which part is the althgrothm part ,the buffer based,
            # if the buffer is enough ,choose the high quality
            # if the buffer is danger, choose the low  quality
            # if there is no rebuf ,choose the low target_buffer
            bit_rate, target_buffer,pre_bw = abr.run(time,S_time_interval,S_send_data_size,S_chunk_len,S_rebuf,S_buffer_size, S_play_time_len,
                                              S_end_delay,S_decision_flag,S_buffer_flag,S_cdn_flag, end_of_video, cdn_newest_id,
                                              download_id,cdn_has_frame,abr_init,
                                              bandwidth_ls=frame_bandwidth_list,
                                              alg_option="MPC",#"TR_with_SSS-PBP_2"#"TR_with_SSS-PBP",#"TR_horm",
                                              #video_chunk_size=video_chunk_size
                                              )#MPC9
            #print(time,frame_bandwidth_list[-1], float(pre_bw), lata,chunk_lo)   #buffer size
            save_qoe_data("MPC_HM_oboe77.txt", time,reward_frame, bitr,  reb, smot,  lata, frame_bandwidth_list[-1], float(pre_bw),chunk_lo)
            chunk_lo = 0
            buf=0
            # ------------------------------------------- End  -------------------------------------------
        #reward_all += reward_frame
        if end_of_video:

            #save_qoe_data(reward_all,)
            print("video count", video_count, reward_all,tt, float(reward_all)/float(tt))
            tt=0
            reward_all_sum += reward_all / 1000

            video_count += 1
            if video_count >= len(all_file_names):
                    break
            cnt = 0
            last_bit_rate = 0
            reward_all = 0
            bit_rate = 0
            target_buffer = 0

            S_time_interval = [0] * past_frame_num
            S_send_data_size = [0] * past_frame_num
            S_chunk_len = [0] * past_frame_num
            S_rebuf = [0] * past_frame_num
            S_buffer_size = [0] * past_frame_num
            S_end_delay = [0] * past_frame_num
            S_chunk_size = [0] * past_frame_num
            S_play_time_len = [0] * past_frame_num
            S_decision_flag = [0] * past_frame_num
            S_decision_flag = [0] * past_frame_num
            S_buffer_flag = [0] * past_frame_num
            S_cdn_flag = [0] * past_frame_num
            
        reward_all += reward_frame
        reward_frame=0
        #print("reward_all",reward_all)
        #print("ccc", reward_chunk,tt)
    bandwidth_file.close()
    return reward_all_sum,


if __name__ == '__main__':
    network_trace_folder=os.listdir("./network_trace")
   # print("ok1")
    for tracefile in network_trace_folder:
            #print("ok1")
        #if tracefile == "0":
            a = test("aaa",trace_name=tracefile)
            print(a)
    # test("test_0","./network_trace/interpolated_0.csv")
