import math
import time

import torch
from torch import nn
from visdom import Visdom

from sisfall_dataset import MySisfallData
from mobiact_dataset import MyMobiactData
from softfall_dataset import MySoftfallData
import torch.nn.functional as F
import torch.optim as optim

from utils import AverageMeter, ProgressMeter


# 调整学习率
def adjust_learning_rate(optimizer, init_lr, epoch, total_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr



# my_dataset = MySisfallData()
# my_dataset = MyMobiactData()
my_dataset = MySoftfallData()
"""
划分训练集和测试集，参考 https://www.lmlphp.com/user/16517/article/item/512864/
此方法可继续增加lengths个数，来更细致地划分数据集，如加个验证集
也可以在dataset中先打乱维度 tensor = tensor[torch.randperm(tensor.size(0))]  # 打乱第一个维度
"""
train_size = int(0.6 * len(my_dataset))
test_size = len(my_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset=my_dataset, lengths=[train_size, test_size])



train_sampler = None
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True)

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True)







# D:\BaiduNetdiskWorkspace\研究生\读论文\对比学习\SimSiam\code\官方\myTest\ckpt\checkpoint_0099.pth.tar
# ckpt = torch.load("../ckpt/checkpoint_0050.pth.tar")

# pre_model = simsiam_builder.SimSiam()


# pre_model.load_state_dict(ckpt['state_dict'], strict=False)



# new_model = nn.Sequential(*list(pre_model.children()))
# new_model = nn.Sequential(*list(pre_model.children())[:-1])
# new_model = nn.Sequential(*list(pre_model.children())[:-2])


# for _, param in new_model.named_parameters():
#         param.requires_grad = False

class backbone(nn.Module):
    """
    This is a self-defined SimSiam encoder.
    """
    def __init__(self, d_model, nhead):
        super().__init__()
        """
        输入和输出都是m*n，只需传入n
        """
        super(backbone, self).__init__()
        # src = batch,seq_len,fea_len [xx, 9, 120] 按理说是这个形式，最后一维是每轴长度
        # d_m = d_v = d_model / h
        # d_model是本encoder的输出大小，其应当与下一层的输入相同，即 等于proj_in_dim！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！1
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)


    def forward(self, x):
        x = self.transformer_encoder(x)
        return x

class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()

    def forward(self, x):
        return x.mean(dim=-2)
class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        # self.lstm = nn.LSTM(input_size=121,hidden_size=128, batch_first=True)
        self.lstm = nn.LSTM(input_size=77,hidden_size=128, batch_first=True)

    def forward(self, x):
        x = self.lstm(x)
        return x[0]

new_model = nn.Sequential(
    simsiam_builder.backbone(d_model=77, nhead=11),
    nn.Linear(77, 8),
    # simsiam_builder.backbone(d_model=121, nhead=11),
    # nn.Linear(121, 8),
    # lstm(),
    # nn.Linear(128, 8),
    nn.LayerNorm(8),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.1),
    GAP(),
    nn.Linear(8,2)
)

# new_model = nn.Sequential(
#     *list(new_model.children()),
#     nn.Linear(81,8),
#     nn.ReLU(inplace=True),
#     nn.Flatten(),
#     nn.Linear(8*9,2)
# )



print(new_model)

epochs=2000

torch.cuda.set_device('cuda:0')
device = torch.device('cuda:0')

net = new_model.to(device)

# init_lr = 0.05 * 64 / 256
init_lr = 1e-3
optimizer = torch.optim.SGD(net.parameters(), init_lr,
                                momentum=0.9,
                                weight_decay=1e-4)

# optimizer = torch.optim.SGD(net.parameters(), init_lr,
#                                 momentum=0.9,)
criteon = nn.CrossEntropyLoss().to(device)


viz = Visdom()
viz.line([0.], [0.], win='train', opts=dict(title='train loss'))
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss & acc.',
                                                   legend=['loss', 'acc.']))
global_loss = 0

for epoch in range(epochs):

    adjust_learning_rate(optimizer, init_lr, epoch, epochs)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # 切换到训练模式
    net.train()
    # 用于记录运行时间
    end = time.time()

    for step, (signal, _, label) in enumerate(train_loader):
        signal = signal[:, 0:6, ...].to(device)
        # signal = signal.to(device)
        label = label.to(device)


        logits = net(signal)
        loss = criteon(logits, label)

        losses.update(loss.item(), signal.size(0))
        global_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # 计算 elapsed 时间
        batch_time.update(time.time() - end)
        end = time.time()  # 如果单卡，好像不用这一步，反正就一张卡，时间都是一致的


        if step % 10 == 0:
            progress.display(step)

    viz.line([global_loss], [epoch], win='train', update='append')

    # test
    net.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        for (signal, _, label) in test_loader:
            signal = signal[:, 0:6, ...].to(device)
            # signal = signal.to(device)
            label = label.to(device)

            logits = net(signal)
            test_loss += criteon(logits, label)

            pred = logits.argmax(dim=1)
            correct += pred.eq(label).float().sum().item()
        test_loss /= len(test_loader.dataset) # 这个平均loss肯定小啊，train中的是每个step下的，这里是epoch的平均，所以肯定小，下面vis时，得移到里面，再说吧！！！！！！！！！！！！！！！！！！！！！！

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        viz.line([[test_loss.item(), correct / len(test_loader.dataset)]],
                 [epoch], win='test', update='append')