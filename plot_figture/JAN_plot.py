import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['times new roman']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号val
import os
import seaborn as sns
import torch
from ulit.init_seed import *
from ulit.CWRU import *
from sklearn.manifold import TSNE


def plot_JAN(source_acc_list, source_losses_list,test_acc_list,losses_list, cm, save_dir, fault_classes):
    print('plot_acc_loss_JAN')
    plt.figure(1, figsize=(10, 4), dpi=600)
    plt.rcParams['font.sans-serif'] = ['times new roman']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.subplot(1, 2, 1)
    # 绘制曲线
    plt.plot(source_acc_list, label='source_train_acc', linestyle='-', color='blue')
    plt.plot(test_acc_list, label='val_acc', linestyle='--', color='red')
    # 添加标签和标题
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('train_val_acc')
    plt.legend(loc='lower right')  # 左下显示
    plt.subplot(1, 2, 2)
    plt.plot(source_losses_list, label='source_train_loss', linestyle='-', color='blue')
    plt.plot(losses_list, label='val_loss', linestyle='-', color='purple')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('all loss')
    plt.legend(loc='upper right')

    # 保存损失曲线图
    fig_path = os.path.join(save_dir, 'all_loss.pdf')
    fig_path1 = os.path.join(save_dir, 'all_loss.png')
    plt.savefig(fig_path)
    plt.savefig(fig_path1)
    # plt.show()
    plt.close()
    # 可视化混淆矩阵
    plt.figure(figsize=(6, 6), dpi=600)
    plt.rcParams['font.sans-serif'] = ['times new roman']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=fault_classes, yticklabels=fault_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('confusion_matrix')

    fig_path = os.path.join(save_dir, 'confusion_matrix.pdf')
    fig_path1 = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(fig_path)
    plt.savefig(fig_path1)





def test_plot(loader, model, device):
    raw_feat_list, fc_feat_list, label_list = [], [], []
    model.eval()
    with torch.no_grad():
        for data, label in loader:
            data_raw = data.reshape(data.shape[0], -1)  # 原始数据展平
            data = data.to(device)
            label = label.to(device)

            outputs, _ = model(data)

            raw_feat_list.append(data_raw.cpu().numpy())
            fc_feat_list.append(outputs.cpu().numpy())
            label_list.append(label.cpu().numpy())

    return raw_feat_list, fc_feat_list, label_list

def plot_tsne_JAN(args, save_dir, JAN_model, JAN_model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print('plot_tsne_JAN')
    init_seed(args.seed)
    JAN_model.to(device)

    # Load pretrained weights

    JAN_model.load_state_dict(torch.load(os.path.join(save_dir, JAN_model_name)))

    cwru_data = CWRU(args.source_data, args.target_data, args.test_data)
    source_dataset, _, val_dataset = cwru_data.train_test_split_order()

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, pin_memory=True, drop_last=False)

    # Extract features
    val_raw, val_fc, val_label = test_plot(val_loader, JAN_model,  device)
    src_raw, src_fc, src_label = test_plot(source_loader, JAN_model,  device)

    def stack_features(lst):
        return np.vstack(lst)

    def stack_labels(lst):
        return np.concatenate(lst)

    val_raw_np, val_fc_np = stack_features(val_raw), stack_features(val_fc)
    val_label_np = stack_labels(val_label)

    src_raw_np, src_fc_np = stack_features(src_raw), stack_features(src_fc)
    src_label_np = stack_labels(src_label)

    if args.save_or_not:
        np.savetxt(os.path.join(save_dir, 'val_raw_feat.txt'), val_raw_np, delimiter=' ')
        np.savetxt(os.path.join(save_dir, 'val_fc_feat.txt'), val_fc_np, delimiter=' ')
        np.savetxt(os.path.join(save_dir, 'val_labels.txt'), val_label_np, delimiter=' ')
        np.savetxt(os.path.join(save_dir, 'src_fc_feat.txt'), src_fc_np, delimiter=' ')
        np.savetxt(os.path.join(save_dir, 'src_labels.txt'), src_label_np, delimiter=' ')

    # 合并特征和标签，统一做TSNE
    all_fc = np.concatenate([src_fc_np, val_fc_np], axis=0)
    all_labels = np.concatenate([src_label_np, val_label_np], axis=0)
    domain_flag = np.array([0] * len(src_label_np) + [1] * len(val_label_np))  # 0: 源域, 1: 验证域

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    all_tsne = tsne.fit_transform(all_fc)

    label_mapping = {0: 'Normal', 1: 'PC', 2: 'PW', 3: 'SC', 4:'1'}  #,7:'11', 8:'22', 9:'33'

    plt.rcParams['font.sans-serif'] = ['times new roman']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

    colors = ['r', 'orange', 'y', 'g','magenta'] # 红橙黄绿青蓝, 'magenta', 'pink', 'brown'
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
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), fontsize='small', loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_features.pdf'))
    plt.savefig(os.path.join(save_dir, 'tsne_features.png'))
    #plt.show()
    plt.close()