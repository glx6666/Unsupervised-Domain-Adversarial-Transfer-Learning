import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from scipy.io import loadmat





class CustomCWRUDataset(Dataset):
    def __init__(self, data_pd, transform=None):
        """
        CWRU 自定义数据集
        :param data_pd: pandas DataFrame (包含 'data' 和 'label' 列)
        :param transform: 数据预处理方法
        """
        self.data_pd = data_pd
        self.transform = transform

    def __len__(self):
        return len(self.data_pd)

    def __getitem__(self, idx):
        file_path = self.data_pd.iloc[idx]['data']
        label = int(self.data_pd.iloc[idx]['label'])

        # 读取 .mat 文件
        mat_data = loadmat(file_path)
        data = mat_data['data'].transpose()  # 变换维度 (N, 1)
        data = data.reshape(1, -1)[:,:1024]  # 保持数据形状为 (1, L)
        data = torch.tensor(data).float()  # 转换为 Float Tensor

        if self.transform:
            data = self.transform(data)

        return data, label
