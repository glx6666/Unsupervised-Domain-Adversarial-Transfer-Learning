import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from PIL import Image  # 加载图片
import torchvision.transforms as transforms  # 预处理用

class CustomImageDataset(Dataset):
    def __init__(self, data_pd, transform=True):
        """
        自定义图片数据集
        :param data_pd: pandas DataFrame (包含 'data' 和 'label' 列)
        :param transform: 数据预处理方法（比如ToTensor/Resize/Normalize等）
        """
        self.data_pd = data_pd
        self.transform = transform
        self.transforms = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(size=64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.data_pd)

    def __getitem__(self, idx):
        file_path = self.data_pd.iloc[idx]['data']
        label = int(self.data_pd.iloc[idx]['label'])

        # 打开图片
        image = Image.open(file_path).convert('L')  # 转成灰度图模式

        # 应用transform（如有）
        if self.transform:
            image = self.transforms(image)
        else:
            # 如果没有提供 transform，默认转成Tensor
            image = transforms.ToTensor()(image)

        return image, label
