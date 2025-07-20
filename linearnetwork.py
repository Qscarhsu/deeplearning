import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import sklearn

# 引入torch模組裡面神經元的函數
from torch.nn import Sequential, Linear, Sigmoid, Softmax, ReLU

# 引路torch的優化器
from torch.optim import SGD

# torch裡面資料處裡的工具包
# DataLoader 資料加載器，負責把資料分成一個一個 batch，並支援多種取樣方式（打亂、順序、分批）。

# 張量資料集，把多個 tensor（通常是 features 和 labels）包裝在一起，讓 DataLoader 可以讀取。

from torch.utils.data import DataLoader, TensorDataset


###### Step 1 建立一個toy dataset ######

X = np.random.rand(100, 3)  # 隨機產生100筆資料，每筆有3個特徵

y = np.random.randint(0, 2, size=(100,))  # 隨機產生100筆資料的標籤，0或


#####Step 2 將資料轉換為 PyTorch 張量 ######

X_new = torch.tensor(X, dtype=torch.float32)  # 將特徵轉換為 float32 類型的張量
y_new = torch.tensor(y, dtype=torch.int64)  # 將標籤
