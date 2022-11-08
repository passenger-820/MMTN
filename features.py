import torch

"""
Sum Magnitude Vector 
    Parameters:
        batch_data: 批数据，batch可为0. 数据形状 [batch, axis, points] 或 [axis, points]
        axis_dim: 计算sum的维度，默认在[batch, axis, points]的axis维度计算。若为[axis, points]，则也在axis维度计算。
    Return:
        idx: smv_max所在索引
        smv: 所有的smv
        smv_sum: smv没开根号的数据，不开根号可以减少计算时间
"""
def SMV(batch_data, axis_dim=1): # batch, axis, points
    if batch_data.ndim == 2:
        axis_dim = 0
    elif batch_data.ndim == 3:
        axis_dim = 1
    else:
        return "Input shape error."
    # print(batch_data)
    smv_pow2 = torch.pow(batch_data, 2)
    # print(smv_pow2)
    smv_sum = torch.sum(smv_pow2, dim=axis_dim) # 不开根号，也就是不用sqrt，理论上是减少计算量的
    # print(smv_sum)
    smv = torch.sqrt(smv_sum)
    # print(smv)
    idx = torch.argmax(smv, dim=axis_dim)
    # print(idx)
    # for i in range(len(idx)):
    #     if idx[i] < 120:
    #         print(i, "\t", idx[i])
    return idx, smv, smv_sum


"""
根据SMV—MAX来选择数据窗口
调用getWindow()方法来得到窗口， 窗口大小在初始化GetFeaWinFromDataset类时指定
"""
class GetFeaWinFromDataset():
    """
    注意区分idx和len的含义，idx从0开始，len就是所以理解的长度。
        举个例子，假设维度为[...,128]，所以idx为0~127，长度最长128
        如果target_idx为22，其实其所在长度已经为23，则其前面其实最大长度（front_len）只有22，所以如果 front_len（22） > target_idx（22），则不够长。
        同理，其后面最大长度只有105，其对应idx为127。所以，如果 target_idx（22） + rear_len（105） + 1 > 128，则不够长。
        由于python左开右闭，所以取右边时+1（不然算上target_idx点，总point会少一个），才能保证rear取的长度一致

        此外，说明一下：我是将目标窗口数据覆盖到最前面，然后切出这些前面的数据。这样可以保证我返回的数据长度绝对没问题

        Parameters:
            batch_data: 批数据，batch可为0. 数据形状 [batch, axis, points] 或 [axis, points]
            target_idx: 目标点，用于向前后取数据点（包含目标点）
            front_len: 目标点前n个数据
            rear_len: 目标点后n个数据
        Return:
            batch_data: 根据指定idx和指定前后截取长度，所截取的window数据
        Tips:
            如果有长度错误，本次数据将全部变为0
                只有ndim为3时，我们才删除全是0的那些记录，
                因为ndim为2时，说明我们只是传来一个记录，删了可能有bug，而且即便不删也能很轻易看到全是0
    """
    def __init__(self, multi_axis_dataset, pre_len=24, front_len=119, rear_len=0): # 40Hz 提前0.6s，窗口2s
        acc = multi_axis_dataset[:, 0:3, :]
        smv_max_idx, smv_max, _ = SMV(acc)

        self.batch_data = multi_axis_dataset
        self.target_idx = smv_max_idx - pre_len # 提前预测时间
        self.front_len = front_len
        self.rear_len = rear_len

    # 取3块，第一块target_idx所在点的前front_len个，第二块target_idx所在点，第三块target_idx所在点后rear_len个
    def getFeaWinByTgtIdx(self):
        # 传入的不是batch数据，
        if self.batch_data.ndim == 2:
            # 长度无误，将目标窗口数据搬到最前面
            if (self.front_len >= 0) and (self.rear_len >= 0) and (self.front_len <= self.target_idx) and (
                    self.target_idx + self.rear_len + 1 <= self.batch_data.shape[-1]):
                self.batch_data[..., 0:(self.front_len + self.rear_len + 1)] = self.batch_data[...,
                                                                (self.target_idx - self.front_len):(self.target_idx + self.rear_len + 1)]
            else:
                # 长度有错就全变0，返回
                # print("The front_len or rear_len could not be negative. Or the front data length is not enough. Or The rear data length is not enough.\n")
                self.batch_data = torch.zeros_like(self.batch_data)
            # 返回这么长的数据
            return self.batch_data[..., 0:(self.front_len + self.rear_len + 1)]
            # return batch_data[..., 0:(front_len + rear_len)] # 少要1个point

        if self.batch_data.ndim == 3:
            # 长度无误，将目标窗口数据搬到最前面
            for i in range(self.batch_data.shape[0]):
                if (self.front_len >= 0) and (self.rear_len >= 0) and (self.front_len <= self.target_idx[i]) and (
                        self.target_idx[i] + self.rear_len + 1 <= self.batch_data[i].shape[-1]):
                    self.batch_data[i, :, 0:(self.front_len + self.rear_len + 1)] = self.batch_data[i, :, (self.target_idx[i] - self.front_len):(
                                self.target_idx[i] + self.rear_len + 1)]
                else:
                    # 长度有错就全变0，返回
                    # print("The data of index " + str(i) + ": The front_len or rear_len could not be negative. Or the front data length is not enough. Or The rear data length is not enough.")
                    self.batch_data[i, ...] = torch.zeros_like(self.batch_data[i, ...])

            # 得到非全0 ‘索引’
            non_zero_rows = torch.abs(self.batch_data[:, 0, :]).sum(dim=-1) > 0
            # 只要非全0的记录
            self.batch_data = self.batch_data[non_zero_rows]

            return self.batch_data[..., 0:(self.front_len + self.rear_len + 1)]
            # return batch_data[..., 0:(front_len + rear_len)] # 少要1个point

    # 方便调用
    def getWindow(self):
        fea_window = self.getFeaWinByTgtIdx()
        return fea_window



if __name__ == '__main__':
    print(' ')









