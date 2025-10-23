import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from scipy.io import loadmat
from ulit.cwru_datasets import *
from ulit.dataset_image import *

class CWRU(object):
    def __init__(self, source_dir, target_dir, test_dir, transform=None, image = False):
        """
        CWRU 数据集加载器，支持源域、目标域和测试集数据的加载

        :param source_dir: 源域数据所在目录
        :param target_dir: 目标域数据所在目录
        :param test_dir: 测试集数据所在目录
        :param transform: 数据预处理方法
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.test_dir = test_dir
        self.transform = transform
        self.image = image

        # 加载数据
        self.source_data = self.load_cwru_data(self.source_dir)
        self.target_data = self.load_cwru_data(self.target_dir)
        self.test_data = self.load_cwru_data(self.test_dir)

    def load_cwru_data(self, root_dir):
        """加载指定目录下的 CWRU 数据"""
        data = {'data': [], 'label': []}
        for class_label, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    data['data'].append(file_path)
                    data['label'].append(class_label)
        return pd.DataFrame(data)

    def train_test_split_order(self):
        """
        按类别划分训练/测试集
        :param test_size: 测试集占比
        :return: source_train, target_train, test_set
        """
        source_train = self.source_data
        target_train = self.target_data
        test_set = self.test_data
        if self.image:
            source_dataset = CustomImageDataset(source_train, transform=self.transform)
            target_dataset = CustomImageDataset(target_train, transform=self.transform)
            test_dataset = CustomImageDataset(test_set, transform=self.transform)
        else:# 转换为 Dataset
            source_dataset = CustomCWRUDataset(source_train, transform=self.transform)
            target_dataset = CustomCWRUDataset(target_train, transform=self.transform)
            test_dataset = CustomCWRUDataset(test_set, transform=self.transform)

        return source_dataset, target_dataset, test_dataset