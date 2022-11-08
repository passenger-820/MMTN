import numpy as np
import torch
import os
# -v7格式
import scipy.io as sio
# sio.loadmat
# matfn = 'C:/Users/Administrator/Desktop/F_04835861-01-2013-10-16-13-19-35.mat'
# data = sio.loadmat(matfn)
# print(data)

# -v7.3格式
# import h5py
# nyu = h5py.File('C:/Users/Administrator/Desktop/F_04835861-01-2013-10-16-13-19-35.mat')
# # # imgs = nyu['images']
# # # dpts = nyu['depths']
# # # sems = nyu['labels']
# a = nyu['tmp'][:]
# """
# 1 Relative time in seconds
# 2 Absolute time in MATLAB time format (see below)
# 3 Acceleration signal along the x-axis [m/s2]
# 4 Acceleration signal along the y-axis [m/s2]
# 5 Acceleration signal along the z-axis [m/s2]
# 6 Gyroscope signal along the x-axis [°/s]
# 7 Gyroscope signal along the y-axis [°/s]
# 8 Gyroscope signal along the z-axis [°/s]
# 9 Magnetometer signal along the x-axis [μT]
# 10 Magnetometer signal along the y-axis [μT]
# 11 Magnetometer signal along the z-axis [μT]
# 12 Fall indicator (see below)
# """
# print(a.shape)

class LoadFile():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def getit(self):
        return









