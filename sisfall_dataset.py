import torch
from torch.utils.data import Dataset

from features import GetFeaWinFromDataset
from utils import GetPthData, ButterWorth, PositionalEncoding, modalEmbedding, modalEmbeddingAblation

"""
部分数据集的传感器数据是ADC数据（例如sisfall），需要借助公式转换为我们所认知的重力加速度，角速度和磁力
传入：初始数据，数据集名称（全部小写，无间隔）
"""
class AD2Raw():
    def __init__(self, ori_data, ori_name):
        self.ori_data = ori_data
        # sisfall
        if ori_name == 'sisfall':
            self.sisfall_acc1_range = torch.tensor(16)
            self.sisfall_acc1_resolution = torch.tensor(13)
            self.sisfall_gyr_range = torch.tensor(2000)
            self.sisfall_gyr_resolution = torch.tensor(16)
            self.sisfall_acc2_range = torch.tensor(8)
            self.sisfall_acc2_resolution = torch.tensor(14)
        else:
            self.default_acc_range = torch.tensor(16)
            self.default_acc_resolution = torch.tensor(13)
            self.default_gyr_range = torch.tensor(2000)
            self.default_gyr_resolution = torch.tensor(16)
            self.default_mag_range = torch.tensor(8)
            self.defaultl_mag_resolution = torch.tensor(14)

    def fromSisfall(self):
        self.ori_data[:,0:3,:] = ((2 * self.sisfall_acc1_range)/torch.pow(2, self.sisfall_acc1_resolution)) * self.ori_data[:,0:3,:]
        self.ori_data[:,3:6,:] = ((2 * self.sisfall_gyr_range)/torch.pow(2, self.sisfall_gyr_resolution)) * self.ori_data[:,3:6,:]
        self.ori_data[:,6:9,:] = ((2 * self.sisfall_acc2_range)/torch.pow(2, self.sisfall_acc2_resolution)) * self.ori_data[:,6:9,:]
        return self.ori_data


class MySisfallData(Dataset):

    def __init__(self):
        super(MySisfallData, self).__init__()
        """
        读取的数据就是batch，axis，points
        """
        adl_12_raw_40Hz = GetPthData(pth_data_dir=r"d:\datasets\MachineLearning\FallDetection\SisFall\ori_pth",
                                     file_name="SisFall_ADL_12.pth").down_sample_to_40Hz()
        adl_25_raw_40Hz = GetPthData(pth_data_dir=r"d:\datasets\MachineLearning\FallDetection\SisFall\ori_pth",
                                     file_name="SisFall_ADL_25.pth").down_sample_to_40Hz()
        adl_100_raw_40Hz = GetPthData(pth_data_dir=r"d:\datasets\MachineLearning\FallDetection\SisFall\ori_pth",
                                      file_name="SisFall_ADL_100.pth").down_sample_to_40Hz()
        fall_15_raw_40Hz = GetPthData(pth_data_dir=r"d:\datasets\MachineLearning\FallDetection\SisFall\ori_pth",
                                      file_name="SisFall_Fall_15.pth").down_sample_to_40Hz()
        # adl_12_raw_40Hz = GetPthData(pth_data_dir=r"f:\datasets\MachineLearning\FallDetection\SisFall\ori_pth",
        #                              file_name="SisFall_ADL_12.pth").down_sample_to_40Hz()
        # adl_25_raw_40Hz = GetPthData(pth_data_dir=r"f:\datasets\MachineLearning\FallDetection\SisFall\ori_pth",
        #                              file_name="SisFall_ADL_25.pth").down_sample_to_40Hz()
        # adl_100_raw_40Hz = GetPthData(pth_data_dir=r"f:\datasets\MachineLearning\FallDetection\SisFall\ori_pth",
        #                               file_name="SisFall_ADL_100.pth").down_sample_to_40Hz()
        # fall_15_raw_40Hz = GetPthData(pth_data_dir=r"f:\datasets\MachineLearning\FallDetection\SisFall\ori_pth",
        #                               file_name="SisFall_Fall_15.pth").down_sample_to_40Hz()

        """
        feature window 以及 label
        """
        fall_15_fea_window = GetFeaWinFromDataset(multi_axis_dataset=fall_15_raw_40Hz).getWindow()
        self.label = torch.ones(fall_15_fea_window.shape[0], dtype=torch.long)

        adl_12_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_12_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(adl_12_fea_window.shape[0], dtype=torch.long)])

        adl_25_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_25_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(adl_25_fea_window.shape[0], dtype=torch.long)])

        adl_100_fea_window = GetFeaWinFromDataset(multi_axis_dataset=adl_100_raw_40Hz).getWindow()
        self.label = torch.cat([self.label, torch.zeros(adl_100_fea_window.shape[0], dtype=torch.long)])

        """
        原始数据和滤波后的数据分别全部concatenate在一起
        """
        self.signals = torch.cat([fall_15_fea_window, adl_12_fea_window, adl_25_fea_window, adl_100_fea_window], dim=0)
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
        self.signals = AD2Raw(ori_data=self.signals, ori_name='sisfall').fromSisfall()
        self.signals_lowPass = AD2Raw(ori_data=self.signals_lowPass, ori_name='sisfall').fromSisfall() # 可以使用转换后的数据滤波，节省时间

        """
        pos和modal embedding
        """
        self.signals = PositionalEncoding(d_model=120)(self.signals)
        self.signals = modalEmbedding(modal_num=2, batch_data=self.signals) # a,g
        # self.signals = modalEmbedding(modal_num=3, batch_data=self.signals) # a,g,a
        # self.signals = modalEmbeddingAblation(modal_num=2, batch_data=self.signals) # a,g
        # self.signals = modalEmbeddingAblation(modal_num=3, batch_data=self.signals) # a,g,a

        self.signals_lowPass = PositionalEncoding(d_model=120)(self.signals_lowPass)
        self.signals_lowPass = modalEmbedding(modal_num=3, batch_data=self.signals_lowPass)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        signal_filtered = self.signals_lowPass[idx]
        label = self.label[idx]

        return signal, signal_filtered, label

if __name__ == '__main__':
    print('')