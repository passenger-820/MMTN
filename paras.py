import argparse


"""
这一部分是参数配置
"""
def get_parameters():
    # 创建一个解析对象，然后按需添加参数
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # 数据集地址
    parser.add_argument('--mobiact-dir', default=r"d:\datasets\MachineLearning\FallDetection\MobiAct\pth-200Hz")
    parser.add_argument('--sisfall-dir', default=r"d:\datasets\MachineLearning\FallDetection\SisFall\ori_pth")
    parser.add_argument('--softfall-dir', default=r"d:\datasets\MachineLearning\Softfall\pth_raw")

    # 下采样后的频率 ( 25Hz = 200 / down)
    parser.add_argument('--step-down', default=8, type=int)
    parser.add_argument('--mobiact-down', default=8, type=int)
    parser.add_argument('--sisfall-down', default=8, type=int)

    # 特征窗口总长 (front_len + 1 + rear_len) 以25Hz为基准，提前0.6s的3s窗口
    parser.add_argument('--pre-len', default=15, type=int, help='lead time')
    parser.add_argument('--front-len', default=75, type=int, help='before SMV')
    parser.add_argument('--rear-len', default=0, type=int, help='after SMV')

    # 模态数
    parser.add_argument('--modal-two', default=2, type=int)
    parser.add_argument('--modal-three', default=3, type=int)

    # Positonal Embedding 配置
    parser.add_argument('--d-pe', default=76, type=int)

    # Transformer配置
    parser.add_argument('--d-model', default=77, type=int)
    parser.add_argument('--n-head', default=11, type=int)

    # Linear Projector 配置
    parser.add_argument('--projected-dim', default=8, type=int)

    # Binary Predictor 配置
    parser.add_argument('--predicted-dim', default=2, type=int)

    # 一些训练的配置
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float)

    # 随机种子
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_parameters()
    print(args.sisfall_dir)
