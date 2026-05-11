import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

class GaussianUnknownMean:

    def __init__(self, mean0, var0, varx):
        """Initialize model.
        p(meanx) ~ N(mean0, var0)
        p(x) ~ N(meanx, varx)
        """
        self.mean0 = mean0
        self.var0 = var0
        self.varx = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1 / var0])

    def log_pred_prob(self, t, x):
        post_means = self.mean_params[:t]
        post_stds = np.sqrt(self.var_params[:t])
        return norm(post_means, post_stds).logpdf(x)

    def update_params(self, t, x):
        new_prec_params = self.prec_params + (1 / self.varx)
        self.prec_params = np.append([1 / self.var0], new_prec_params)
        new_mean_params = (self.mean_params * self.prec_params[:-1] + \
                           (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1. / self.prec_params + self.varx
    

def Identify_segmented_stable_length(data,model,hazard=0.1):
    #批处理一系列数据
    T=len(data)
    #每一行对应一个时间步下，Gt取各种值的概率。
    #Gt=j截至t，稳定段长度为j，由于Gt只和Gt-1相关，为了降低空间复杂度，使用滚动数组存储
    log_R = -np.inf * np.ones((2, T + 1))#概率0取对数是负无穷
    
    log_R[0, 0] = 0  
    pmean = np.empty(T)  
    pvar = np.empty(T)  
    log_message = np.array([0])  
    log_H = np.log(hazard)
    log_1mH = np.log(1 - hazard)
    
    Gt_list=[]#存储每一个时间步下Gt概率取最大值对应的j
    for t in range(1, T + 1):
        x = data[t - 1]
        
        
        pmean[t - 1] = np.sum(np.exp(log_R[(t - 1)%2, :t]) * model.mean_params[:t])
        pvar[t - 1] = np.sum(np.exp(log_R[(t - 1)%2, :t]) * model.var_params[:t])

        
        log_pis = model.log_pred_prob(t, x)
        log_growth_probs = log_pis + log_message + log_1mH
        log_cp_prob = logsumexp(log_pis + log_message + log_H)
        
        new_log_joint = np.append(log_cp_prob, log_growth_probs)
        
        # 存储新的Gt各种取值的概率
        log_R[t%2, :t + 1] = new_log_joint
        log_R[t%2, :t + 1] -= logsumexp(new_log_joint)#归一化
        
        model.update_params(t, x)
        log_message = new_log_joint 
        Gt_list.append(np.argmax(      np.exp(log_R[t%2])[1:]   ))
    return Gt_list#pmean, pvar



class SegmentInference():
    def __init__(self,model,hazard):
        self.model=model
        self.hazard=hazard
        T=100000
        self.log_R = -np.inf * np.ones((2, T))

        self.log_R[0, 0] = 0
        self.pmean = np.empty(T)
        self.pvar = np.empty(T)
        self.log_message = np.array([0])  
        self.log_H = np.log(hazard)
        self.log_1mH = np.log(1 - hazard)
        self.Gt_list = []  # 存储每一个时间步下Gt概率取最大值对应的j
    def ObserveAData(self,x,t):
        # if t>=len(self.log_message):#扩容
        #     new_column_log_R = np.full((self.log_R.shape[0], 1), 0)  # 创建一个空列
        #     self.log_R=np.concatenate((self.log_R, new_column_log_R), axis=1)
        #     self.pmean=np.append(self.pmean,0)
        #     self.pvar=np.append(self.pvar,0)
        #     self.log_message=np.append(self.log_message,0)

        self.pmean[t - 1] = np.sum(np.exp(self.log_R[(t - 1)%2, :t]) * self.model.mean_params[:t])
        self.pvar[t - 1] = np.sum(np.exp(self.log_R[(t - 1)%2, :t]) * self.model.var_params[:t])
        log_pis = self.model.log_pred_prob(t, x)

        log_growth_probs = log_pis + self.log_message + self.log_1mH
        log_cp_prob = logsumexp(log_pis + self.log_message + self.log_H)

        
        new_log_joint = np.append(log_cp_prob, log_growth_probs)
        self.log_R[t % 2, :t + 1] = new_log_joint
        self.log_R[t % 2, :t + 1] -= logsumexp(new_log_joint)  # 归一化

        self.model.update_params(t, x)
        self.log_message = new_log_joint
        self.Gt_list.append(np.argmax(np.exp(self.log_R[t % 2])[1:]))



if __name__ == "__main__":

    #for bandwidth_file in os.listdir("./bandwidth_data"):
        #print(bandwidth_file)
        bandwidth_series=[]
        time_series=[]

        #with open("./bandwidth_data/{0}".format(bandwidth_file),"r") as f:
        #with open("./bandwidth_data/{0}".format(bandwidth_file), "r") as f:
        #with open("./bandwidth_data/frame_bandwidth_1.txt", "r") as f:
        with open("E:\LiveStreamingDemo-master-2\\bandwidth_oboe.txt", "r") as f:
        #with open("./bandwidth_new.txt", "r") as f:
            while True:
                try:
                    t,c=map(float,(f.readline().split())[:2])
                    if time_series and   abs(c-bandwidth_series[-1])<1e-3:
                        pass
                    else:
                        time_series.append(t)
                        bandwidth_series.append(c)
                except:
                    break
        print(len(bandwidth_series))


        start=0
        end=int(len(bandwidth_series))
        T = end-start
        mean, std = np.mean(bandwidth_series[start:end]), (np.var(bandwidth_series[start:end])) ** 0.5
        thres = 2 * std
        hazard = np.sum(np.sum((bandwidth_series[start:end] > mean + thres) | (bandwidth_series[start:end] > mean + thres))) / T
        print("T,hazard",T,hazard)

        mean0 = mean
        var0 = 2 * std
        varx = std

        model = GaussianUnknownMean(mean0, var0, varx)

        pre = SegmentInference(model, hazard)

        for i in range(1, end):
            pre.ObserveAData(bandwidth_series[i], i)

        m0=int(np.percentile(pre.Gt_list,20))
        avg_length=1#+40
        print("m0",len(pre.Gt_list),m0)
        
        idx = []

        last=0
        X=[]
        y=[]
        y_avg=[]



        for i in range(1, len(pre.Gt_list)):
            if (pre.Gt_list[i] - pre.Gt_list[i - 1]) < 0:

                if len(bandwidth_series[last:i])>=m0:
                    s=last
                    print("i",i)
                    while s+m0<=i-1:
                        #if s+m0+avg_length-1 <=i-1 :
                            X.append(bandwidth_series[s:s+m0])
                            avg=sum(bandwidth_series[s+m0:s+m0+avg_length])/avg_length
                            #print("sum",bandwidth_series[s+m0:s+m0+avg_length])
                           # print("yy",bandwidth_series[s + m0])
                            y_avg.append(avg)
                            y.append(bandwidth_series[s + m0])
                            s+=1
                    last=i
        assert len(X) == len(y) and len(X) == len(y_avg)
        print(len(X))

        X_array = np.array(X)
        y_array = np.array(y)
        y_avg_array=np.array(y_avg)
        #name=bandwidth_file.split(".")[0]
        np.savez('./bandwidth_frame_seg_oboe1.npz' , X=X_array, y=y_array, y_avg=y_avg_array)
        #np.savez('./seg_bandwidth_data/{0}_seg.npz'.format(name), X=X_array, y=y_array,y_avg=y_avg_array)
