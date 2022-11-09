import torch
from sklearn import metrics
from sklearn.metrics import confusion_matrix

correct = 0
pred = torch.tensor([1,1,1,0,0,1,0,1,1,1,1,0,0,1,0,0,0,1,1,1,1])
label = torch.tensor([1,1,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,1,1,0])
# print(pred.shape)
correct += pred.eq(label).float().sum().item()
print(correct)

# confusion_matrix(label, pred)

# 打印分类报告，有查全率，查准率等等
print(f"{metrics.classification_report(label, pred)}\n")
aa= metrics.classification_report(label, pred)


# 绘制多分类混淆矩阵
# probabilities为所有测试结果，yv为验证集Validation samples (images, labels)
cnf_matrix = confusion_matrix(label, pred).reshape(4)
print(cnf_matrix)