import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#讀取資料集
df = pd.read_csv(r"dateset\15m_data.csv", sep = "\t")

df.head(100)

df.columns

df.info()

#定義InputX InputY
y = df['Close'].astype('Float32').values

df['DateTime'] = pd.to_datetime(df['DateTime']) #原本時間是物件 改成pandas時間格式

df["Timestamp"] = df["DateTime"].astype("int64")//10**9 #將時間改成秒數

x = df["Timestamp"].values

delta = 1e-8

x_mean = x.mean(axis=0)
y_mean = y.mean(axis=0)
x_std = x.std(axis=0)
y_std = y.std(axis=0)

x_norm = (x - x_mean)/(x_std + delta)

y_norm = (y - y_mean)/(y_std + delta)
 

X_time = torch.tensor(x_norm , dtype=torch.float32 )

Y_data = torch.tensor(y_norm, dtype= torch.float32)


#分割資料集


train_ratio = 0.8

train_size = int(len(X_time)*0.8)
test_size = int(len(X_time) - train_size)

X_train = X_time[:train_size]
X_test = X_time[-test_size: ]
Y_train = Y_data[:train_size]
Y_test = Y_data[-test_size:]

#繼承pytorch裡面的物件dataloader，並修改裡面的dunder method

from torch.utils.data import Dataset  # 注意 D 大寫

class TimeSeriesWindowSet(Dataset):
    def __init__(self, X_time, Y_data, window_size):
        self.X_time = X_time
        self.Y_data = Y_data
        self.window_size = window_size

    def __len__(self):
        return len(self.X_time) - self.window_size

    def __getitem__(self, idx):
        # 防呆：確保不會超出 index
        if idx + self.window_size >= len(self.X_time):
            raise IndexError("Index out of range.")
        x = self.X_time[idx: idx + self.window_size]
        y = self.Y_data[idx + self.window_size]
        return x, y

# 測試
window_size = 5
timeslicer = TimeSeriesWindowSet(X_train, Y_train, window_size)
print("總共可以切出多少組：", len(timeslicer))
# x, y = timeslicer[155]
# print(x, y)

#將訓練資料做分割訓練

dataloader = DataLoader(timeslicer, batch_size=75, shuffle= False)

#定義梯度優化器

learning_rate = 0.01


#定義model
model =nn.Sequential(
    nn.Linear(5,15),
    nn.ReLU(),
    nn.Linear(15,4),
    nn.ReLU(),
    nn.Linear(4,1)
)


optimizer = optim.SGD(model.parameters(), lr = learning_rate)

loss_list= []

batch_list = []

criterion = nn.MSELoss()

epochs = 20


for epoch in range(epochs):

    for batch_idx, (x, y) in enumerate(dataloader):

        optimizer.zero_grad()

        pred_Y = model(x)

        loss = criterion(pred_Y, y)

        loss.backward()

        optimizer.step()

        loss_list.append(loss.item())


        batch_list.append(batch_idx)

    print(f"Epoch {epoch+1}/{epochs}: {loss.item()}")


plt.figure(figsize=(15, 5))
plt.plot(loss_list, label='Loss')
plt.xlabel('Batch Index')
plt.ylabel('Loss Value')
plt.title('Training Loss per Batch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#現在我要把我訓練好的模型把test_data丟進去看結果

timeslicer =  TimeSeriesWindowSet(X_test, Y_test, window_size)

x_test, y_test = timeslicer[155]
# print(timeslicer[155])

test_pred_list = []
test_true_list = []

for x_test, y_test in timeslicer:
     
    x_test = x_test.unsqueeze(0)  # 增加 batch 維度
    y_pred = model(x_test)

    loss = criterion(y_pred, y_test)

    test_pred_list.append(y_pred.item())
    test_true_list.append(y_test.item())

    print(f"預測值: {y_pred.item()}, 真實值: {y_test.item()}, 損失: {loss.item()}")


#反標準化

test_pred_actual = [v * y_std + y_mean for v in test_pred_list]
test_true_actual = [v * y_std + y_mean for v in test_true_list]


# 取得測試集起始位置的 index
test_start_idx = len(df) - len(X_test)
# 取得對應的 DateTime 欄位
test_datetime = df['DateTime'].iloc[test_start_idx + window_size : ].values[:len(test_pred_actual)]
# 畫圖
plt.figure(figsize=(15, 5))
plt.plot(test_datetime, test_true_actual, label='Actual (True)', color='blue')
plt.plot(test_datetime, test_pred_actual, label='Predicted', color='orange')
plt.xlabel('DateTime')
plt.ylabel('Close Price')
plt.title('NASDAQ-100 Price Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
