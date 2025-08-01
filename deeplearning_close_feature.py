import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 讀資料
df = pd.read_csv(r"dateset\15m_data.csv", sep = "\t")
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values('DateTime')  # 先排序，保險！

# 只用 close price 做時序資料
y = df['Close'].astype('float32').values
y_mean = y.mean()
y_std = y.std()
y_norm = (y - y_mean) / (y_std + 1e-8)

# Torch tensor
X_seq = torch.tensor(y_norm, dtype=torch.float32)
Y_seq = torch.tensor(y_norm, dtype=torch.float32)

# 分割訓練/測試
train_ratio = 0.8
train_size = int(len(X_seq) * train_ratio)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
Y_train, Y_test = Y_seq[:train_size], Y_seq[train_size:]




window_size = 5

class TimeSeriesWindowSet(Dataset):
    def __init__(self, X, Y, window_size):
        self.X = X
        self.Y = Y
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.window_size]
        y = self.Y[idx + self.window_size]
        return x, y

train_set = TimeSeriesWindowSet(X_train, Y_train, window_size)
test_set  = TimeSeriesWindowSet(X_test, Y_test, window_size)

batch_size = 75
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=1, shuffle=False)

# MLP 模型
model = nn.Sequential(
    nn.Linear(window_size, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
epochs = 20
loss_list = []

# 訓練
for epoch in range(epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y = y.unsqueeze(1)
        pred_y = model(x)
        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# 畫 loss 曲線
plt.figure(figsize=(10, 4))
plt.plot(loss_list)
plt.title("Training Loss")
plt.show()

# 測試預測
model.eval()
pred_list = []
true_list = []
with torch.no_grad():
    for x, y in test_set:
        x = x.unsqueeze(0)
        pred = model(x)
        pred_list.append(pred.item())
        true_list.append(y.item())

# 反標準化
pred_actual = np.array(pred_list) * y_std + y_mean
true_actual = np.array(true_list) * y_std + y_mean

# 取得對應時間
time_for_pred = df['DateTime'].iloc[train_size+window_size:].values

# 畫圖
plt.figure(figsize=(15, 5))
plt.plot(time_for_pred, true_actual, label="Actual", color='blue')
plt.plot(time_for_pred, pred_actual, label="Predicted", color='orange')
plt.xlabel("DateTime")
plt.ylabel("Close Price")
plt.title("NASDAQ-100 Price Prediction")
plt.legend()
plt.tight_layout()
plt.show()
