import torch
from dataset import MyData
# adc = torch.rand(1,9,6)
# print(adc)
from utils import drawAxisPicturesWithData, GetPthData


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
        self.default_acc_range = 16
        self.default_acc_resolution = 13
        self.default_gyr_range = 2000
        self.default_gyr_resolution = 16
        self.default_mag_range = 8
        self.defaultl_gyr_resolution = 14

    def fromSisfall(self):
        self.ori_data[:,0:3,:] = ((2 * self.sisfall_acc1_range)/torch.pow(2, self.sisfall_acc1_resolution)) * self.ori_data[:,0:3,:]
        self.ori_data[:,3:6,:] = ((2 * self.sisfall_gyr_range)/torch.pow(2, self.sisfall_gyr_resolution)) * self.ori_data[:,3:6,:]
        self.ori_data[:,6:9,:] = ((2 * self.sisfall_acc2_range)/torch.pow(2, self.sisfall_acc2_resolution)) * self.ori_data[:,6:9,:]
        return self.ori_data


# ad2raw = AD2Raw(adc,'sisfall')
# raw = ad2raw.fromSisfall()
# print(raw)


# data = MyData().signals
#
# test1 = data[0:2,...]
# print(test1.shape)
# print(test1)
#

fall_15_raw_40Hz = GetPthData(pth_data_dir=r"f:\DataSets\MachineLearning\FallDetection\SisFall\ori_pth",
                                  file_name="SisFall_Fall_15.pth").down_sample_to_40Hz()

test1 = fall_15_raw_40Hz[0:2,...]

draw = drawAxisPicturesWithData(batch_data=test1, picture_title='test1', save_root='../static/figures/')
draw.pltTripleAxis()

ad2raw = AD2Raw(test1,'sisfall')
raw = ad2raw.fromSisfall()
draw = drawAxisPicturesWithData(batch_data=raw, picture_title='raw', save_root='../static/figures/')
draw.pltTripleAxis()