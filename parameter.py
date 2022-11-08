import argparse
import torchvision.models as models

"""
拿到torchvision.models 里的所有模型名称，并从按照字母升序排列
alexnet,densenet121,...,wide_resnet50_2，我这个版本共37个模型
"""
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))


"""
这一部分是参数配置
"""
def get_parameters():
    # 创建一个解析对象，然后按需添加参数
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # 数据集地址
    # parser.add_argument('data', metavar='DIR', help='path to dataset')
    # backbone模型结构 default='ViLT', choices=['Transformer', 'ViT', 'ViLT']
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names, help='model architecture')
    # 多线程数
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 32)')
    # 总epochs
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    # 继续训练的起始epoch
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    # batch size
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    # 学习率
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', help='initial (base) learning rate', dest='lr')
    # 动量，这是优化器的动量，不是模型参数更新的动量
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    # 权重衰减
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    # 打印频率
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    # 保存的最新的ckpt地址
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    # 种子
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    # 指定GPU
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    # simsiam specific configs:
    # encoder 输入和输出的维度
    parser.add_argument('--dim', default=256, type=int, help='feature dimension (default: 256)')
    # predictor head 输出维度
    parser.add_argument('--pred-dim', default=64, type=int, help='hidden dimension of the predictor (default: 512)')
    # predictor head 固定学习率
    parser.add_argument('--fix-pred-lr', action='store_true',  help='Fix learning rate for the predictor')

    return parser.parse_args()