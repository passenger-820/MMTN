from paras import get_parameters
import torch
import torch.nn as nn
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

args = get_parameters()

# AverageMeter可以记录当前的输出，累加到某个变量之中，然后根据需要可以打印出历史上的平均
class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# 进度条
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# 从Pth数据中加载,返回数据的shape均为 batch,axis,points
class GetPthData():
    def __init__(self, pth_data_dir, file_name):
        self.dir = pth_data_dir
        self.file_name = file_name

    """
    得到原始数据，传入文件上级目录和文件名
    """
    def get_raw(self):
        full_dir = os.path.join(self.dir, self.file_name)
        raw_data = torch.load(full_dir)
        return torch.transpose(raw_data,1,2)

    # """
    # 下采样至40Hz，并保留前3维（只要加速度）
    # """
    # def down_sample_to_40Hz_and_select_first_three_dimension(data):
    #     return data[:, ::5, 0:3]

    """
    下采样至40Hz（因为这是从200Hz下采样，所以间隔200/40=5）。
    这个函数可以泛化，传递原始Hz，目标Hz，和所需轴数，返回下采样数据。
    可以用step参数改变下采样的频率
    """
    def down_sample(self, step=args.step_down):
        data = self.get_raw()
        return data[:, :, ::step]

    """
    找到指定窗口的数据
    """
    def three_second_window_data(self):
        return None



# 低通、高通、带通的巴特沃斯滤波器
class ButterWorth():
    """
    采样频率：25Hz
    位置：腰部
        我们暂且认为某动作的频率为10Hz以下，变绿后续计算截止频率Wn
            Wn：归一化截止频率。计算公式Wn=2*截止频率/采样频率
            注意：根据采样定理，采样频率要大于两倍的信号本身最大的频率，才能还原信号。截止频率一定小于信号本身最大的频率，所以Wn一定在0和1之间
            当构造带通滤波器或者带阻滤波器时，Wn为长度为2的列表

    本方法不限制轴数，即可以过滤3轴，6轴，9轴，只需保证传入的数据shape为[axis，points]的二维形式即可
    目前还需要修改绘图部分的代码！！！！！！！！！！！！！！！！！！！！！
    还得要适应batch
    """
    def __init__(self, data): # 传入的data是 axis,points 这个顺序
        # csv地址
        self.root = r".\static\qs_10383_1.csv"
        # 读取csv文件，转置，进而将维度0设置为轴
        self.data = data
        # 绘图的x轴，从1开始，到总记录数
        self.axisX = np.arange(1, self.data.shape[1] + 1)
        # x轴标签
        self.xLabel = 'Samples'
        # y轴标签
        self.yLabel = 'Accelerator (g)'
        # 线legend名称
        self.lineLegend = ['Accelerator_X', 'Accelerator_Y', 'Accelerator_Z', 'Filtered_X', 'Filtered_Y', 'Filtered_Z']
        # 线条颜色
        self.lineColor = ['blue','teal','blueviolet', 'greenyellow','cyan','darkorange']


    """
    截止频率为5Hz的5阶数字低通滤波器
    """
    def lowPass(self):
        # 截止频率
        cut = 5.0
        # 采样频率
        fs = 25.0
        # 3dB带宽点
        wn = 2*cut/fs
        b, a = signal.butter(5, wn, 'lowpass')
        filtered = np.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            filtered[i] = np.array(signal.filtfilt(b, a, self.data[i]))
        return filtered

    """
    截止频率为8Hz的5阶数字高通滤波器
    """
    def highPass(self):
        # 截止频率
        cut = 8.0
        # 采样频率
        fs = 25.0
        # 3dB带宽点
        wn = 2 * cut / fs
        b, a = signal.butter(5, wn, 'highpass')
        filtered = np.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            filtered[i] = np.array(signal.filtfilt(b, a, self.data[i]))
        return filtered

    """
    截止频率为[5Hz,8Hz]的5阶数字带通滤波器
    """
    def bandPass(self):
        # 低截止频率,高截止频率
        lowCut = 5.0
        highCut = 8.0
        # 采样频率
        fs = 25.0
        # 3dB带宽点
        wn_low = 2 * lowCut / fs
        wn_high = 2 * highCut / fs
        b, a = signal.butter(5, [wn_low,wn_high], 'bandpass')
        filtered = np.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            filtered[i] = np.array(signal.filtfilt(b, a, self.data[i]))
        return filtered

    def pltFigure(self, data, figSaveRoot, figName):
        # 开启一张空白图，设置尺寸和分辨率
        fig = plt.figure(figsize=(9, 4), dpi=200)
        # 开启坐标轴
        ax = plt.axes()
        # 先将3轴数据加入，再一次绘制出这个3轴图
        for i in range(data.shape[0]):
            axis_data = np.squeeze(data[i])
            if(i<3):
                ax.plot(self.axisX, axis_data, color=self.lineColor[i], label=self.lineLegend[i])
            else:
                ax.plot(self.axisX, axis_data, color=self.lineColor[i], label=self.lineLegend[i])

        # 将图标绘入
        ax.set(xlabel=self.xLabel, ylabel=self.yLabel)
        ax.legend()
        # 展示图片
        # plt.show()
        """
        下面这部分用于保存图片
        """
        figName = figName + '.png'
        figRoot = os.path.join(figSaveRoot, figName)
        plt.savefig(figRoot)
        plt.close()

# 将数据绘制成波形图
class drawAxisPicturesWithData():
    def __init__(self, batch_data, picture_title='test', save_root='D:/'): # batch, axis, points
        # 数据
        self.data = batch_data
        # 根据数据确定batch和axis数
        if self.data.ndim == 3:
            self.batch_size = self.data.shape[0]
            self.axis = self.data.shape[1]
        elif self.data.ndim == 2:
            self.axis = self.data.shape[0]
        else:
            print("The shape of data should be like [batch, axis, points] or [axis, points].")

        # 绘图的x轴，从1开始，到总记录数
        self.range_x = np.arange(1, self.data.shape[-1] + 1)
        # 传感器数量
        self.sensor_number = int(self.data.shape[-2] / 3)
        # x轴标签
        self.x_label = 'samples'
        # 单轴标签，legend名称
        self.single_axis_legend = ['a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z', 'm_x', 'm_y', 'm_z']
        # 三轴标签
        self.triple_axis_label = ['Accelerator', 'Gyroscope', 'Magnetometer']
        # 线条颜色
        self.line_color = ['red', 'blue', 'cyan', 'chartreuse', 'darkgoldenrod', 'slateblue', 'black', 'springgreen',
                          'lightskyblue']
        """
            Accelerator_X:      red
            Accelerator_Y:      blue
            Accelerator_Z:      cyan
            Gyroscope_X:        chartreuse
            Gyroscope_Y:        darkgoldenrod
            Gyroscope_Z:        slateblue
            Magnetometer_X:     black
            Magnetometer_Y:     springgreen
            Magnetometer_Z:     lightskyblue
        """
        # 线条类型
        self.line_style = ['-', '--', '-.', ':']
        """
            '-'       solid line style
            '--'      dashed line style
            '-.'      dash-dot line style
            ':'       dotted line style
        """

        # 图片标题
        self.picture_title = picture_title
        # 保存图片的地址
        self.save_root = save_root

    """
    打印某一种传感器的3轴
        如果输入为3轴，即一个传感器，则就绘制一张三轴图
        如果输入为n*3轴，即n个传感器，则就绘制n张三轴图
    """
    def pltTripleAxis(self):
        # 为了节省figure量，就开一张图，每次画完就清空
        plt.figure(figsize=(9, 4), dpi=200)
        ax = plt.axes()

        # 根据传感器数量，每次绘制一张3轴的图
        if self.data.ndim == 2:
            for sensor_number in range(self.sensor_number):
                # 先将3轴数据加入，再一次绘制出这个3轴图
                for i in range(3):
                    axis_data = np.squeeze(self.data[i + sensor_number * 3])
                    ax.plot(self.range_x, axis_data, color=self.line_color[i + sensor_number * 3], label=self.single_axis_legend[i + sensor_number * 3])
                    ax.set(xlabel=self.x_label, ylabel=self.triple_axis_label[sensor_number], title=self.picture_title)
                ax.legend()
                # plt.show()
                fig_name = self.picture_title + '_' + 'sensor' + str(sensor_number) + '.png'
                fig_root = os.path.join(self.save_root, fig_name)
                # 如果已经存过了，就跳过
                if (os.path.exists(fig_root)):
                    continue
                plt.savefig(fig_root)
                plt.cla()
            plt.close()
            return None

        if self.data.ndim == 3:
            for batch in range(self.data.shape[0]):
                for sensor_number in range(self.sensor_number):
                    # 先将3轴数据加入，再一次绘制出这个3轴图
                    for i in range(3):
                        axis_data = np.squeeze(self.data[batch][i + sensor_number * 3])
                        ax.plot(self.range_x, axis_data, color=self.line_color[i + sensor_number * 3], label=self.single_axis_legend[i + sensor_number * 3])
                        ax.set(xlabel=self.x_label, ylabel=self.triple_axis_label[sensor_number], title=self.picture_title)
                    ax.legend()
                    # plt.show()
                    fig_name = self.picture_title + '_batch'+ str(batch) + '_sensor'+ str(sensor_number) + '.png'
                    fig_root = os.path.join(self.save_root, fig_name)
                    # 如果已经存过了，就跳过
                    if (os.path.exists(fig_root)):
                        continue
                    plt.savefig(fig_root)
                    plt.cla()
            plt.close()
            return None


    """打印所有轴"""
    def pltAllAxis(self):

        # 一张图足矣
        plt.figure(figsize=(9, 4), dpi=200)
        ax = plt.axes()

        # 将所有轴读入后，绘制在一张图中
        if self.data.ndim == 2:
            for i in range(self.data.shape[0]):
                axis_data = np.squeeze(self.data[i])
                ax.plot(self.range_x, axis_data,color=self.line_color[i],label=self.single_axis_legend[i])
            ax.legend()
            ax.set(xlabel=self.x_label, ylabel="Data of All Sensors", title=self.picture_title)
            # plt.show()
            fig_name = self.picture_title + '.png'
            fig_root = os.path.join(self.save_root, fig_name)
            # 如果已经存过了，就跳过
            if (os.path.exists(fig_root)==False):
                plt.savefig(fig_root)
            plt.close()
            return None

        if self.data.ndim == 3:
            for batch in range(self.data.shape[0]):
                for i in range(self.data.shape[1]):
                    axis_data = np.squeeze(self.data[batch][i])
                    ax.plot(self.range_x, axis_data,color=self.line_color[i],label=self.single_axis_legend[i])
                ax.legend()
                ax.set(xlabel=self.x_label, ylabel="Data of All Sensors", title=self.picture_title)
                # plt.show()
                fig_name = self.picture_title + '_batch' + str(batch) +'.png'
                fig_root = os.path.join(self.save_root, fig_name)
                # 如果已经存过了，就跳过
                if (os.path.exists(fig_root)==False):
                    plt.savefig(fig_root)
                    plt.cla()
            plt.close()
            return None

"""
由于我们的数据是严格的时许数据，所有position embedding可以使用 正余弦不可学习PE（可学习的应该也可以，学出来效果可能是一样的严格时序）
参考Transformer PE - sin-cos 1d：https://blog.csdn.net/qq_19841133/article/details/126245602

这是Transformer原文里用的

PE矩阵可以看作是两个矩阵相乘，一个矩阵是pos（/左边），另一个矩阵是i（/右边），奇数列和偶数列再分别乘sin和cos
可以实现任意位置通过已有的位置编码线性组合表示，不要求偶数列是sin，奇数列是cos，也可以前一半是sin，后一半是cos
"""
class PositionalEncoding(nn.Module):
    """
    参考他的理解 https://blog.csdn.net/weixin_41790863/article/details/123480570
    注意：在Positional Encoding的时候，在forward中已经将Embedding加了进去，因此，PositionalEncoding类返回的结果直接进EncoderLayer。

    我试了下这样输入
        raw = torch.randint(0,5,(3,3,16))
        pos_emd = PositionalEncoding(d_model=16)(raw)  返回结果是[3,3,16]

    Positional Encoding需要d_model为偶数，即最后一维得是偶数，接着为进行modalEmbedding，
        这样最后一维会+1变为奇数，即d_model为奇数。而TransformerEncoder需要 d_m = d_v = d_model / nhead
        为此，让这个偶数为62，+1为63，nhead取7或9
    """
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

# 模态嵌入
def modalEmbedding(modal_num, batch_data):
    """
    先PositionalEncoding之后再modalEmbedding，shape变为[batch，axis，points+1]
    raw = torch.randn((2,9,4))
    print(raw)
    ml = modalEmbedding(3, raw)
    print(ml)
    """
    modal_types = torch.zeros(3)
    for i in range(modal_num - 1):
        modal_types = torch.cat([modal_types, torch.ones(3) + i])
    modal_embedded = torch.cat([modal_types.view(1, modal_num*3, 1).expand(batch_data.shape[0], -1, -1), batch_data], dim=2)
    return modal_embedded

# 模态嵌入（消融试验）：全-1
def modalEmbeddingAblation(modal_num, batch_data):
    """
    为了保持数据量一致，我们不是不用 ME，
    而是让所有 modal type 全为 -1
    不用全0是因为咱们ME都是≥0的，用个负数，应该有明显区别，不过没细致研究
    """
    modal_types = torch.zeros(3)-1
    for i in range(modal_num - 1):
        modal_types = torch.cat([modal_types, torch.zeros(3)-1])
    modal_embedded = torch.cat([modal_types.view(1, modal_num*3, 1).expand(batch_data.shape[0], -1, -1), batch_data], dim=2)
    return modal_embedded

# 一些查看网络内容的方法
# 查看网络参数
# p = sum(map(lambda p:p.numel(), pre_model.parameters()))
# print('parameters size:', p)

# for name, t in net.named_parameters():
#     print('parameters:', name, t.shape)
#
# for name, m in net.named_children():
#     print('children:', name, m)
#
# for name, m in net.named_modules():
#     print('modules:', name, m)

if __name__ == '__main__':
    """
    看看拿原始数据和降采样的数据
    """
    # getData = GetPthData(pth_data_dir=r"F:\DataSets\MachineLearning\FallDetection\SisFall\ori_pth",
    #                      file_name="SisFall_ADL_12.pth")
    # rawData = getData.get_raw()
    # print(rawData.shape)
    # rawData_40Hz = getData.down_sample_to_40Hz()
    # print(rawData_40Hz.shape)


    """
    看原始图和滤波后的图
    """
    # getData = GetPthData(pth_data_dir=r"F:\DataSets\MachineLearning\FallDetection\SisFall\ori_pth",
    #                      file_name="SisFall_ADL_12.pth")
    getData = GetPthData(pth_data_dir=r"d:\datasets\MachineLearning\FallDetection\SisFall\ori_pth",
                         file_name="SisFall_Fall_15.pth")
    rawData_40Hz = getData.down_sample()
    a = rawData_40Hz[0:2,:,:]
    draw = drawAxisPicturesWithData(batch_data=a,picture_title="test",save_root='static/figures/')
    # draw.pltTripleAxis()
    draw.pltAllAxis()
    # print(a.shape)
    # butterWorthFilter = ButterWorth(a)
    # a_lowPass = butterWorthFilter.lowPass()
    # print(a_lowPass.shape)
    # # 开启一张空白图，设置尺寸和分辨率
    # fig = plt.figure(figsize=(9, 4), dpi=200)
    # # 开启坐标轴
    # ax = plt.axes()
    # # 先将3轴数据加入，再一次绘制出这个3轴图
    # for i in range(a.shape[0]):
    #     axis_data = np.squeeze(a[i])
    #     ax.plot(axis_data)
    # # 将图标绘入
    # ax.set()
    # ax.legend()
    # # 展示图片
    # plt.show()
    # # 开启一张空白图，设置尺寸和分辨率
    # fig = plt.figure(figsize=(9, 4), dpi=200)
    # # 开启坐标轴
    # ax = plt.axes()
    # # 先将3轴数据加入，再一次绘制出这个3轴图
    # for i in range(a_lowPass.shape[0]):
    #     axis_data = np.squeeze(a_lowPass[i])
    #     ax.plot(axis_data)
    # # 将图标绘入
    # ax.set()
    # ax.legend()
    # # 展示图片
    # plt.show()

