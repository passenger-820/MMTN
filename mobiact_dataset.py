import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset

from features import GetFeaWinFromDataset
from utils import GetPthData, ButterWorth

"""
部分数据集的传感器数据是ADC数据（例如sisfall），需要借助公式转换为我们所认知的重力加速度，角速度和磁力
传入：初始数据，数据集名称（全部小写，无间隔）
"""
# class AD2Raw():
#     def __init__(self, ori_data, ori_name):
#         self.ori_data = ori_data
#         # sisfall
#         if ori_name == 'sisfall':
#             self.sisfall_acc1_range = torch.tensor(16)
#             self.sisfall_acc1_resolution = torch.tensor(13)
#             self.sisfall_gyr_range = torch.tensor(2000)
#             self.sisfall_gyr_resolution = torch.tensor(16)
#             self.sisfall_acc2_range = torch.tensor(8)
#             self.sisfall_acc2_resolution = torch.tensor(14)
#         else:
#             self.default_acc_range = torch.tensor(16)
#             self.default_acc_resolution = torch.tensor(13)
#             self.default_gyr_range = torch.tensor(2000)
#             self.default_gyr_resolution = torch.tensor(16)
#             self.default_mag_range = torch.tensor(8)
#             self.defaultl_mag_resolution = torch.tensor(14)
#
#     def fromSisfall(self):
#         self.ori_data[:,0:3,:] = ((2 * self.sisfall_acc1_range)/torch.pow(2, self.sisfall_acc1_resolution)) * self.ori_data[:,0:3,:]
#         self.ori_data[:,3:6,:] = ((2 * self.sisfall_gyr_range)/torch.pow(2, self.sisfall_gyr_resolution)) * self.ori_data[:,3:6,:]
#         self.ori_data[:,6:9,:] = ((2 * self.sisfall_acc2_range)/torch.pow(2, self.sisfall_acc2_resolution)) * self.ori_data[:,6:9,:]
#         return self.ori_data



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

class MyMobiactData(Dataset):

    def __init__(self):
        super(MyMobiactData, self).__init__()
        """
        读取的数据就是batch，axis，points
        timestamp	rel_time	[acc_x	acc_y	acc_z	gyro_x	gyro_y	gyro_z]	azimuth	pitch	roll	label
        """
        BSC_Fall_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="BSC_Fall.pth").down_sample_to_40Hz()
        BSC_Fall_raw_40Hz = BSC_Fall_raw_40Hz[:,2:8,:]
        CHU_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="CHU_ADL.pth").down_sample_to_40Hz()
        CHU_ADL_raw_40Hz = CHU_ADL_raw_40Hz[:,2:8,:]
        CSI_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="CSI_ADL.pth").down_sample_to_40Hz()
        CSI_ADL_raw_40Hz = CSI_ADL_raw_40Hz[:,2:8,:]
        CSO_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="CSO_ADL.pth").down_sample_to_40Hz()
        CSO_ADL_raw_40Hz = CSO_ADL_raw_40Hz[:,2:8,:]
        FKL_Fall_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="FKL_Fall.pth").down_sample_to_40Hz()
        FKL_Fall_raw_40Hz = FKL_Fall_raw_40Hz[:,2:8,:]
        FOL_Fall_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="FOL_Fall.pth").down_sample_to_40Hz()
        FOL_Fall_raw_40Hz = FOL_Fall_raw_40Hz[:,2:8,:]
        JOG_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="JOG_ADL.pth").down_sample_to_40Hz()
        JOG_ADL_raw_40Hz = JOG_ADL_raw_40Hz[:,2:8,:]
        JUM_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="JUM_ADL.pth").down_sample_to_40Hz()
        JUM_ADL_raw_40Hz = JUM_ADL_raw_40Hz[:,2:8,:]
        SBE_Flow_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="SBE_Flow_ADL.pth").down_sample_to_40Hz()
        SBE_Flow_ADL_raw_40Hz = SBE_Flow_ADL_raw_40Hz[:,2:8,:]
        SBW_Flow_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="SBW_Flow_ADL.pth").down_sample_to_40Hz()
        SBW_Flow_ADL_raw_40Hz = SBW_Flow_ADL_raw_40Hz[:,2:8,:]
        SCH_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="SCH_ADL.pth").down_sample_to_40Hz()
        SCH_ADL_raw_40Hz = SCH_ADL_raw_40Hz[:,2:8,:]
        SDL_Fall_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="SDL_Fall.pth").down_sample_to_40Hz()
        SDL_Fall_raw_40Hz = SDL_Fall_raw_40Hz[:,2:8,:]
        SIT_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="SIT_ADL.pth").down_sample_to_40Hz()
        SIT_ADL_raw_40Hz = SIT_ADL_raw_40Hz[:,2:8,:]
        SLH_Flow_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="SLH_Flow_ADL.pth").down_sample_to_40Hz()
        SLH_Flow_ADL_raw_40Hz = SLH_Flow_ADL_raw_40Hz[:,2:8,:]
        SLW_Flow_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="SLW_Flow_ADL.pth").down_sample_to_40Hz()
        SLW_Flow_ADL_raw_40Hz = SLW_Flow_ADL_raw_40Hz[:,2:8,:]
        SRH_Flow_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="SRH_Flow_ADL.pth").down_sample_to_40Hz()
        SRH_Flow_ADL_raw_40Hz = SRH_Flow_ADL_raw_40Hz[:,2:8,:]
        STD_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="STD_ADL.pth").down_sample_to_40Hz()
        STD_ADL_raw_40Hz = STD_ADL_raw_40Hz[:,2:8,:]
        STN_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="STN_ADL.pth").down_sample_to_40Hz()
        STN_ADL_raw_40Hz = STN_ADL_raw_40Hz[:,2:8,:]
        STU_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="STU_ADL.pth").down_sample_to_40Hz()
        STU_ADL_raw_40Hz = STU_ADL_raw_40Hz[:,2:8,:]
        WAL_ADL_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="WAL_ADL.pth").down_sample_to_40Hz()
        WAL_ADL_raw_40Hz = WAL_ADL_raw_40Hz[:,2:8,:]

        """
        feature window 以及 label
        """
        BSC_Fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=BSC_Fall_raw_40Hz).getWindow()
        self.label = torch.ones(BSC_Fall_fea_window.shape[0], dtype=torch.long)

        CHU_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=CHU_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(CHU_ADL_fea_window.shape[0], dtype=torch.long)])

        CSI_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=CSI_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(CSI_ADL_fea_window.shape[0], dtype=torch.long)])

        CSO_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=CSO_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(CSO_ADL_fea_window.shape[0], dtype=torch.long)])

        FKL_Fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=FKL_Fall_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.ones(FKL_Fall_fea_window.shape[0], dtype=torch.long)])

        FOL_Fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=FOL_Fall_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.ones(FOL_Fall_fea_window.shape[0], dtype=torch.long)])

        JOG_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=JOG_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(JOG_ADL_fea_window.shape[0], dtype=torch.long)])

        JUM_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=JUM_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(JUM_ADL_fea_window.shape[0], dtype=torch.long)])

        SBE_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SBE_Flow_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SBE_Flow_ADL_fea_window.shape[0], dtype=torch.long)])

        SBW_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SBW_Flow_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SBW_Flow_ADL_fea_window.shape[0], dtype=torch.long)])

        SCH_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SCH_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SCH_ADL_fea_window.shape[0], dtype=torch.long)])

        SDL_Fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SDL_Fall_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.ones(SDL_Fall_fea_window.shape[0], dtype=torch.long)])

        SIT_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SIT_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SIT_ADL_fea_window.shape[0], dtype=torch.long)])

        SLH_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SLH_Flow_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SLH_Flow_ADL_fea_window.shape[0], dtype=torch.long)])

        SLW_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SLW_Flow_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SLW_Flow_ADL_fea_window.shape[0], dtype=torch.long)])

        SRH_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SRH_Flow_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SRH_Flow_ADL_fea_window.shape[0], dtype=torch.long)])

        STD_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=STD_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(STD_ADL_fea_window.shape[0], dtype=torch.long)])

        STN_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=STN_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(STN_ADL_fea_window.shape[0], dtype=torch.long)])

        STU_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=STU_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(STU_ADL_fea_window.shape[0], dtype=torch.long)])

        WAL_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=WAL_ADL_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(WAL_ADL_fea_window.shape[0], dtype=torch.long)])





        """
        原始数据和滤波后的数据分别全部concatenate在一起
        """
        self.signals = torch.cat([BSC_Fall_fea_window,CHU_ADL_fea_window,CSI_ADL_fea_window,CSO_ADL_fea_window,FKL_Fall_fea_window,FOL_Fall_fea_window,JOG_ADL_fea_window,JUM_ADL_fea_window,
                                  SBE_Flow_ADL_fea_window,SBW_Flow_ADL_fea_window,SCH_ADL_fea_window,SDL_Fall_fea_window,SIT_ADL_fea_window,SLH_Flow_ADL_fea_window,
                                  SLW_Flow_ADL_fea_window,SRH_Flow_ADL_fea_window,STD_ADL_fea_window,STN_ADL_fea_window,STU_ADL_fea_window,WAL_ADL_fea_window], dim=0)

        butterWorthFilter = ButterWorth(self.signals)
        self.signals_lowPass = torch.from_numpy(butterWorthFilter.lowPass()).to(torch.float32)

        """
        AD2Raw: 将传感器最初adc读数按照一定换算公式转换为raw数据，即重力加速度、角速度等
        sisfall的加速度和陀螺仪数据需要转换
            In order to convert the acceleration data (AD) given in bits into gravity, use this equation: 
                Acceleration [g]: [(2*Range)/(2^Resolution)]*AD
            In order to convert the rotation data (RD) given in bits into angular velocity, use this equation:
                Angular velocity [?s]: [(2*Range)/(2^Resolution)]*RD
        """
        # self.signals = AD2Raw(ori_data=self.signals, ori_name='sisfall').fromSisfall()
        # self.signals_lowPass = AD2Raw(ori_data=self.signals_lowPass, ori_name='sisfall').fromSisfall() # 可以使用转换后的数据滤波，节省时间

        """
        pos和modal embedding
        """
        self.signals = PositionalEncoding(d_model=120)(self.signals)
        self.signals = modalEmbedding(modal_num=2, batch_data=self.signals)

        self.signals_lowPass = PositionalEncoding(d_model=120)(self.signals_lowPass)
        self.signals_lowPass = modalEmbedding(modal_num=2, batch_data=self.signals_lowPass)


    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        signal_filtered = self.signals_lowPass[idx]
        label = self.label[idx]

        return signal, signal_filtered, label



if __name__ == '__main__':
    # print('')
    #
    mydata = MyMobiactData()
    print(mydata.signals.shape)
    print('')

    # # 可视化看下数据有没有问题
    # BSC_Fall_raw_40Hz = GetPthData(pth_data_dir=r"F:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz", file_name="BSC_Fall.pth").get_raw()
    # BSC_Fall_raw_40Hz = BSC_Fall_raw_40Hz[:, 2:8, ]
    # print(BSC_Fall_raw_40Hz.shape)
    # from utils import drawAxisPicturesWithData
    # draw = drawAxisPicturesWithData(BSC_Fall_raw_40Hz, picture_title="mobiact", save_root='static/figures/')
    # draw.pltAllAxis()

