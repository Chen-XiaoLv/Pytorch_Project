import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

data=pd.read_excel(r"C:\Users\lenovo\Desktop\data1.xlsx")
data=np.array(data.drop("val",axis=1).values).ravel()
plt.figure(figsize=(12,4))
plt.grid(True)
plt.plot(data)
plt.show()

# 划分数据集
test_size=12
train_set=data[:-test_size]
test_set=data[-test_size:]
train_set=data

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
# 打印一组数据集

class CNN(nn.Module):
    def __init__(self, output_dim=1):
        super(CNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(1, 64, 2)
        self.maxpool1 = nn.AdaptiveAvgPool1d(output_size=32)
        self.conv2 = nn.Conv1d(64, 128, 2)
        self.maxpool2 = nn.AdaptiveAvgPool1d(output_size=32)

        self.flatten = nn.Flatten()
        self.lstm1 = nn.LSTM(128 * 32, 1024)
        self.lstm2 = nn.LSTM(1024, 256)
        self.fc = nn.Linear(1024, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        # 输入形状：32, 2205, 50 批次，序列长度，特征长度
        N, T = x.shape[0], x.shape[1]
        x = x.transpose(1, 2)  # 32, 50, 2205 批次，特征长度，序列长度

        x = self.conv1(x)  # torch.Size([32, 100, 2203])
        x = self.maxpool1(x)  # torch.Size([32, 100, 500])
        x = self.conv2(x)  # torch.Size([32, 200, 498])
        x = self.maxpool2(x)  # torch.Size([32, 200, 300])
        # x = self.conv3(x)  # torch.Size([32, 300, 298])
        # x = self.maxpool3(x)  # torch.Size([32, 300, 100])
        x = self.flatten(x)
        # 注意Flatten层后输出为(N×T,C_new)，需要转换成(N,T,C_new)
        _, C_new = x.shape
        x = x.view(N, T, C_new)

        # LSTM部分
        x, h = self.lstm1(x)
        x, h = self.lstm2(x)
        # 注意这里只使用隐层的输出
        x, _ = h

        x = self.fc(x.squeeze())
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

class CNN_Net1(nn.Module):
    def __init__(self, output_dim=1):
        super(CNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(12, 64, 1)
        self.maxpool1 = nn.AdaptiveMaxPool1d(output_size=32)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(output_size=32)
        self.conv3=nn.Conv1d(128,128,1)

        self.flatten = nn.Flatten()
        self.lstm1 = nn.LSTM(128 * 32, 1024)
        self.lstm2 = nn.LSTM(1024, 256)
        self.fc = nn.Linear(256, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        # 输入形状：32, 2205, 50 批次，序列长度，特征长度
        N, T = x.shape[0], x.shape[1]
        x = x.transpose(1, 2)  # 32, 50, 2205 批次，特征长度，序列长度

        x = self.conv1(x)  # torch.Size([32, 100, 2203])
        x = self.maxpool1(x)  # torch.Size([32, 100, 500])
        x = self.conv2(x)  # torch.Size([32, 200, 498])
        x = self.maxpool2(x)  # torch.Size([32, 200, 300])
        x = self.conv3(x)  # torch.Size([32, 300, 298])
        x = self.maxpool2(x)  # torch.Size([32, 300, 100])
        x = self.flatten(x)
        # 注意Flatten层后输出为(N×T,C_new)，需要转换成(N,T,C_new)
        _, C_new = x.shape
        x = x.view(N, T, C_new)

        # LSTM部分
        x, h = self.lstm1(x)
        x, h = self.lstm2(x)
        # 注意这里只使用隐层的输出
        x, _ = h

        x = self.fc(x.squeeze())
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

class CNN_Net2(nn.Module):
    def __init__(self):
        super(CNN_Net2, self).__init__()
        self.hidden=nn.Sequential(
            nn.Conv1d(12,64,1),
            nn.MaxPool1d(32),
            nn.Conv1d(64,128,2),
            nn.MaxPool1d(64),
            nn.Conv1d(128,256,3),
            nn.AvgPool1d(32),
            nn.Conv1d(256,256,1),

            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(256*32,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,1)
        )

    def forward(self,x):
        return self.hidden(x)

