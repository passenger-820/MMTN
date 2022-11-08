# 数据
batch，axis，points

# 关于d_model
最后一维是points即d_model,Positional Encoding需要d_model为偶数，即最后一维得是偶数，接着进行modalEmbedding.这样最后一维会+1变为奇数，即d_model为奇数。而TransformerEncoder需要 d_m = d_v = d_model / nhead  为此，让这个偶数为62，+1为63，nhead可取7
需要设置d_model的地方有：class PositionalEncoding(nn.Module)，这里的需要是偶数。class backbone(nn.Module)（也可以在main_simsiam的创建模型中指定），这里需要时奇数。

不使用PositionalEncoding,就可以更方便指定d_model，目前是80 points + 1 modal = 81 

# 数据集
想要改窗口长度，就去class GetFeaWinFromDataset()里面改默认参数，最好别在调用时指定
