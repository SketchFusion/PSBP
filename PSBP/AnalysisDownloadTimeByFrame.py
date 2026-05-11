import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

frame_bandwidth_folder = "./bandwidth_data/"


def GetFrameDownloadTime(filename, folder=frame_bandwidth_folder):
    ls = []
    with open(folder + filename, "r") as f:
        start_time = float(f.readline().split()[0])
        while True:
            try:
                t = float(f.readline().split()[0])
                ls.append(t - start_time)
                start_time = t
            except:
                break
    return np.array(ls)


def GetChunkDownloadTime(filename, folder=frame_bandwidth_folder):
    intervals = []
    chunk_vs_frame = []
    cnt = 0
    with open(folder + filename, "r") as f:
        s = None
        while True:
            try:
                t, c, flag = f.readline().split()
                t = float(t)
                if flag == "True":
                    if s == None:
                        s = t
                    else:
                        intervals.append(float(t) - s)
                        s = float(t)
                    chunk_vs_frame.append(cnt)
                    cnt = 0
                else:
                    cnt += 1
            except:
                break
    return np.array(intervals), chunk_vs_frame

def GetChunkInterval(filename, folder=frame_bandwidth_folder):
    intervals = []
    with open(folder + filename, "r") as f:
        true_flag = None
        while True:
            try:
                t, c, flag = f.readline().split()
                t = float(t)
                if flag == "False":
                    if true_flag!=None:
                        intervals.append(t - true_flag)
                        true_flag=None
                    else:
                        pass
                else:
                    true_flag=t
            except:
                break
    return np.array(intervals)

def ComputeCDF(data):
    series = pd.Series(data)
    cdf = series.value_counts().sort_index().cumsum() / len(series)
    return cdf


def ComputePercentile(data):
    tmp = np.array(data)
    percentiles = np.arange(0, 110, 10)  # 创建从0到100的11个等间距数字
    percentile_values = np.percentile(data, percentiles)
    for percentile, val in zip(percentiles, percentile_values):
        print(f"The {percentile}th percentile is {val}")


def CDFPlot(ls, percentile=90,title="frame download time"):
    # 计算 CDF
    cdf = ComputeCDF(ls)
    print("okok")
    # 确定百分位数对应的值和概率
    value_at_percentile = cdf.index[int(len(cdf) * (percentile / 100))]
    cumulative_prob = cdf.values[int(len(cdf) * (percentile / 100))]

    # 绘制 CDF 曲线
    sns.lineplot(x=cdf.index, y=cdf.values)
    plt.title(title)
    plt.xlabel('download time (s)')
    plt.ylabel('CDF')

    # 添加灰色网格
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # 标注百分位数关键点
    plt.scatter(value_at_percentile, cumulative_prob, color='red', zorder=5)
    plt.axvline(value_at_percentile, color='red', linestyle='--', linewidth=0.8)
    plt.axhline(cumulative_prob, color='red', linestyle='--', linewidth=0.8)
    plt.ylim(0,1)
    # 设置横坐标从0开始，步长为0.1
    x_min = max(0, min(cdf.index))  # 确保横坐标从0开始
    x_max = max(cdf.index)  # 横坐标最大值
    plt.xticks([round(x, 1) for x in list(plt.xticks()[0]) if x >= x_min and x <= x_max])

    plt.yticks([round(y, 2) for y in np.arange(0, 1.05, 0.1)])

    # 在图中添加箭头和文本标注横坐标关键点的值
    plt.annotate(
        f'{value_at_percentile:.3f}',  # 格式化显示值，保留3位小数
        xy=(value_at_percentile, 0),  # 箭头指向横轴上的关键点
        xytext=(value_at_percentile + 0.15, 0.05),  # 文本显示在关键点右上方
        arrowprops=dict(arrowstyle='->', color='red'),  # 设置箭头样式
        color='red', fontsize=10, ha='center', va='bottom'  # 文本样式
    )



    plt.show()


import matplotlib.font_manager as fm



if __name__ == "__main__":
    for font in fm.findSystemFonts():
        if "SimSun" in font or "YaHei" in font or "Song" in font:
            print(font)

    print(os.listdir("./bandwidth_data"))

    while True:
        file = input("input filename:(q for quit)")
        if file == 'q':
            break

        if os.path.exists("./download_time_data/{0}.npz".format(file)):
            with np.load("./download_time_data/{0}.npz".format(file)) as data:
                download_time_frame_ls, download_time_chunk_ls=data['download_time_frame_ls'],data['download_time_chunk_ls']
        else:
            download_time_frame_ls = GetFrameDownloadTime(file)
            download_time_chunk_ls, frams_in_chunk_ls = GetChunkDownloadTime(file)
            np.savez("./download_time_data/{0}.npz".format(file),
                 download_time_frame_ls=download_time_frame_ls,
                 download_time_chunk_ls=download_time_chunk_ls)
        chunk_interval_ls=GetChunkInterval(file)
        ComputePercentile(download_time_chunk_ls)
        ComputePercentile(download_time_frame_ls)
        ComputePercentile(chunk_interval_ls)

        CDFPlot(download_time_frame_ls)
        CDFPlot(download_time_chunk_ls,10,"chunk download time")
        CDFPlot(chunk_interval_ls)



