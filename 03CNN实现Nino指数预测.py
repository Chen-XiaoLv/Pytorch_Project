import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

data=pd.read_excel(r"C:\Users\lenovo\Desktop\data1.xlsx")
data=np.array(data.drop("val",axis=1).values).ravel()
# 划分数据集
test_size=36
train_set=data[:-test_size]
test_set=data[-test_size:]

from sklearn.preprocessing import MinMaxScaler
# 归一化处理
scaler = MinMaxScaler(feature_range=(-1, 1))
train_norm = scaler.fit_transform(train_set.reshape(-1, 1))

# 转换成 tensor
train_norm = torch.FloatTensor(train_norm).view(-1)
window_size = 12
#将数据按window_size一组分段，每次输入一段后，会输出一个预测的值y_pred
#y_pred与每段之后的window_size+1个数据作为对比值，用于计算损失函数
#例如前5个数据为(1,2,3,4,5),取前4个进行CNN预测,得出的值与(5)比较计算loss
#这里使用每组13个数据,最后一个数据作评估值,即window_size=12
def input_data(seq,ws):
    out = []
    L = len(seq)
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws]
        out.append((window, label))
    return out
train_data = input_data(train_norm,window_size)

class CNNnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(1,64,kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        self.Linear1= nn.Linear(64*11,50)
        self.Linear2= nn.Linear(50,1)
    def forward(self,x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)
        return x

import time
torch.manual_seed(101)
device=torch.device("cuda:0")
model =CNNnetwork()
# 设置损失函数,这里使用的是均方误差损失
criterion = nn.MSELoss()
# 设置优化函数和学习率lr
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 设置训练周期
epochs = 100
model.train()
start_time = time.time()
model=model.to(device)
criterion=criterion.to(device)
print(next(model.parameters()).device)
for epoch in range(epochs):
    for seq, y_train in train_data:
        # 每次更新参数前都梯度归零和初始化
        optimizer.zero_grad()
        seq,y_train=seq.to(device),y_train.to(device)
        # 注意这里要对样本进行reshape，
        # 转换成conv1d的input size（batch size, channel, series length）
        y_pred = model(seq.reshape(1,1,-1))
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}')
print(f'\nDuration: {time.time() - start_time:.0f} seconds')

future = 12
# 选取序列最后12个值开始预测
preds = train_norm[-window_size:].tolist()
# 设置成eval模式
model.eval()
# 循环的每一步表示向时间序列向后滑动一格
for i in range(future):
    seq = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
        preds.append(model(seq.reshape(1,1,-1)).item())
# 逆归一化还原真实值
true_predictions = scaler.inverse_transform(np.array(preds[window_size:]).reshape(-1, 1))
# 对比真实值和预测值
plt.figure(figsize=(12,4))
plt.grid(True)
plt.plot(data)
x = np.arange('2018-02-01', '2019-02-01', dtype='datetime64[M]').astype('datetime64[D]')
plt.plot(x,true_predictions)
plt.show()
