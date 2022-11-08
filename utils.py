import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
    上采样再参照他另写一个。
    """
    def down_sample_to_40Hz(self):
        data = self.get_raw()
        return data[:, :, ::5]

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
    rawData_40Hz = getData.down_sample_to_40Hz()
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

