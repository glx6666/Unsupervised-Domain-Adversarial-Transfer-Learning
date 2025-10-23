import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from ulit.init_seed import *
from ulit.CWRU import *
from ulit.acc import *
from model.feature_extractor import *
from model.classification_extractor import *
from model.mthod1 import *

parser = argparse.ArgumentParser(description='PyTorch PN_Data Training')
parser.add_argument('--source_data', default=r'F:\数据\dataset\shanke_chilun\source_data\0overlap_1024\1500r_0A', help='path to source dataset')
parser.add_argument('--target_data', default=r'F:\数据\dataset\shanke_chilun\target_data\0overlap_1024\1500r_0.2A', help='path to target dataset')
parser.add_argument('--val_data', default=r'F:\数据\dataset\shanke_chilun\test_data\1500r_0A_0.2A', help='path to val dataset')
parser.add_argument('-j', '--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size')
parser.add_argument('-seed', default=123, type=int)
parser.add_argument('-use_model', default='resnet', help='model')
parser.add_argument('--save_or_not', default=True, type=bool)
parser.add_argument('--result_dir', default=r'F:\数据\域对抗迁移学习\result\shanke_chilun\resnet\resnet_MMD_1500r_0A_0.2A_50epoch_0overlap_1024_shanke_wavelet transform_ASA_weigh', type=str)
parser.add_argument('--save_dir', default=r'F:\数据\域对抗迁移学习\result\shanke_chilun\resnet\resnet_MMD_1500r_0A_0.2A_50epoch_0overlap_1024_shanke_wavelet transform_ASA_weigh', type=str)

def test_plot(loader, feature_extractor, classification_extractor, criterion, device):
    raw_feat_list, fc_feat_list, label_list = [], [], []
    feature_extractor.eval()
    classification_extractor.eval()
    losses = AverageMeter('Loss', ':.4f')

    with torch.no_grad():
        for data, label in loader:
            data_raw = data.reshape(data.shape[0], -1)  # 原始数据展平
            data = data.to(device)
            label = label.to(device)

            features = feature_extractor(data)
            outputs = classification_extractor(features)

            loss = criterion(outputs, label)
            losses.update(loss.item(), label.size(0))

            raw_feat_list.append(data_raw.cpu().numpy())
            fc_feat_list.append(features.cpu().numpy())
            label_list.append(label.cpu().numpy())

    print(f'Test Loss: {losses.avg:.3f}')
    return raw_feat_list, fc_feat_list, label_list

def plot():
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    init_seed(args.seed)

    if args.use_model == 'ResNet18':
        feature_extractor = ResNet18_source().to(device)
        classification_extractor = ClassificationExtractor().to(device)
    elif args.use_model == 'resnet':
        feature_extractor = ResNet18().to(device)
        classification_extractor = ClassificationExtractor().to(device)

    # Load pretrained weights
    feature_extractor.load_state_dict(torch.load(os.path.join(args.save_dir, 'feature_extractor_50.pth')))
    classification_extractor.load_state_dict(torch.load(os.path.join(args.save_dir, 'classification_extractor_50.pth')))

    criterion = nn.CrossEntropyLoss().to(device)
    cwru_data = CWRU(args.source_data, args.target_data, args.val_data)
    source_dataset, target_dataset, val_dataset = cwru_data.train_test_split_order()

    source_loader = data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True, drop_last=True)

    # Extract features
    val_raw, val_fc, val_label = test_plot(val_loader, feature_extractor, classification_extractor, criterion, device)
    src_raw, src_fc, src_label = test_plot(source_loader, feature_extractor, classification_extractor, criterion, device)

    def stack_features(lst):
        return np.vstack(lst)

    def stack_labels(lst):
        return np.concatenate(lst)

    val_raw_np, val_fc_np = stack_features(val_raw), stack_features(val_fc)
    val_label_np = stack_labels(val_label)

    src_raw_np, src_fc_np = stack_features(src_raw), stack_features(src_fc)
    src_label_np = stack_labels(src_label)

    if args.save_or_not:
        np.savetxt(os.path.join(args.save_dir, 'val_raw_feat.txt'), val_raw_np, delimiter=' ')
        np.savetxt(os.path.join(args.save_dir, 'val_fc_feat.txt'), val_fc_np, delimiter=' ')
        np.savetxt(os.path.join(args.save_dir, 'val_labels.txt'), val_label_np, delimiter=' ')
        np.savetxt(os.path.join(args.save_dir, 'src_fc_feat.txt'), src_fc_np, delimiter=' ')
        np.savetxt(os.path.join(args.save_dir, 'src_labels.txt'), src_label_np, delimiter=' ')

    # 合并特征和标签，统一做TSNE
    all_fc = np.concatenate([src_fc_np, val_fc_np], axis=0)
    all_labels = np.concatenate([src_label_np, val_label_np], axis=0)
    domain_flag = np.array([0] * len(src_label_np) + [1] * len(val_label_np))  # 0: 源域, 1: 验证域

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    all_tsne = tsne.fit_transform(all_fc)

    label_mapping = {0: 'Normal', 1: 'PC', 2: 'PW', 3: 'SC', 4: 'SD', 5: 'PF', 6:'pw'}

    plt.rcParams['font.sans-serif'] = ['times new roman']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['r', 'orange', 'y', 'g', 'c', 'b']  # 红橙黄绿青蓝
    markers = {0: 'o', 1: 'x'}  # 0: 源域圆圈，1: 验证域叉叉

    for class_label in np.unique(all_labels):
        color = colors[class_label % len(colors)]
        for domain in [0, 1]:
            indices = (all_labels == class_label) & (domain_flag == domain)
            if np.sum(indices) == 0:
                continue
            ax.scatter(
                all_tsne[indices, 0],
                all_tsne[indices, 1],
                marker=markers[domain],
                color=color,
                alpha=0.7,
                s=40,
                label=f"{label_mapping[class_label]} - {'源域' if domain == 0 else '验证域'}"
            )

    # 避免图例重复
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize='small', loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'tsne_features.png'))
    plt.show()

if __name__ == '__main__':
    plot()
