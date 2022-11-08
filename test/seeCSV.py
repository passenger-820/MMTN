

import os
import torch
import torch.nn as nn
import numpy as np
import math

from utils import drawAxisPicturesWithData


"""可视化全图+提前图"""
# file_root = r"D:\BaiduNetdiskWorkspace\研究生\写论文\跌倒参考2022\FPS-conference\辅助图表"
# file = "jlm_10383_2.csv"
# file = os.path.join(file_root, file)
# save_root = r'D:\BaiduNetdiskWorkspace\研究生\写论文\跌倒参考2022\FPS-conference\辅助图表\sensor_data'
#
# data = torch.from_numpy(np.loadtxt(fname=file, delimiter=',')).T
# print(data.shape)
# draw1 = drawAxisPicturesWithData(batch_data=data, picture_title="all", save_root=save_root)
# draw1.pltTripleAxis()
#
# part_data = data[:, 25:101]
# print(part_data.shape)
# draw2 = drawAxisPicturesWithData(batch_data=part_data, picture_title="part", save_root=save_root)
# draw2.pltTripleAxis()

"""可视化ME"""
file_root = r"D:\BaiduNetdiskWorkspace\研究生\写论文\跌倒参考2022\FPS-conference\辅助图表"
file = "jlm_10383_2.csv"
file = os.path.join(file_root, file)
save_root = r'D:\BaiduNetdiskWorkspace\研究生\写论文\跌倒参考2022\FPS-conference\辅助图表\sensor_data'
data = torch.from_numpy(np.loadtxt(fname=file, delimiter=',')).T
part_data = data[:, 25:101].unsqueeze(0)
print(part_data.shape)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        # 假设我的d_model是512，2i以步长2从0取到了512，那么i对应取值就是0,1,2...255
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) # 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，步长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term) # 这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，步长为2，其实代表的就是奇数位置
        # 下面这行代码之后，我们得到的pe形状是：[max_len * 1 * d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe) # 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        # print(x)
        # print(self.pe)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

part_data_PE = PositionalEncoding(d_model=76)(part_data)
print(part_data_PE.shape)

part_data_PE_a = part_data_PE[0, :3, :]
print(part_data_PE_a.shape)
draw3 = drawAxisPicturesWithData(batch_data=part_data_PE_a, picture_title="part_PE_a", save_root=save_root)
draw3.pltTripleAxis()


