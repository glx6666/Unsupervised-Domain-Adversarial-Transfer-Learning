import torch
import torch.nn as nn
import torch.nn.functional as F

class MMDLoss(object):
    def __init__(self, bandwidths=[0.1, 0.5, 1.0], temperature = 0.5):
        """
        MMD (Maximum Mean Discrepancy) 损失类
        :param bandwidths: List[float] - 多个高斯核的带宽
        """
        super(MMDLoss, self).__init__()
        self.bandwidths = bandwidths
        self.temperature = temperature


    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        计算源样本和目标样本之间的高斯核
        """
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        # 扩展total矩阵以便计算所有样本之间的对
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

        # 计算L2距离
        L2_distance = ((total0 - total1) ** 2).sum(2)

        # 如果提供了fix_sigma，使用固定的带宽
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            # 计算带宽，默认值
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)  # 调整带宽
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]  # 生成不同带宽的核

        # 计算多个带宽下的高斯核
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        # 返回多个核值的和
        return sum(kernel_val)

    def adaptive_sigma(self, source, target):
        """
        计算自适应的sigma值（可以根据源和目标数据的L2距离来计算）
        """
        n_samples = source.size(0)

        # 计算源和目标样本之间的L2距离
        L2_distance = torch.sum((source.unsqueeze(0) - target.unsqueeze(1)) ** 2, dim=2)

        # 根据L2距离计算sigma
        sigma = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)
        return sigma

    def maximum_mean_discrepancy(self, source, target, kernel_mul=2.0, kernel_num=5):
        """最大均值差异损失"""
        mmd_loss = 0.0

        # 如果没有指定sigma值，使用自适应sigma
        if self.bandwidths is None:
            sigma = self.adaptive_sigma(source, target)
            self.bandwidths = [sigma]

        for sigma in self.bandwidths:
            # 计算源样本的高斯核
            kernel_ss = self.gaussian_kernel(source, source, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                             fix_sigma=sigma)
            # 计算目标样本的高斯核
            kernel_tt = self.gaussian_kernel(target, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                             fix_sigma=sigma)
            # 计算源和目标样本之间的高斯核
            kernel_st = self.gaussian_kernel(source, target, fix_sigma=sigma, kernel_mul=kernel_mul,
                                             kernel_num=kernel_num)
            # 计算目标域和源域之间的高斯核
            kernel_ts = self.gaussian_kernel(target, source, fix_sigma=sigma, kernel_mul=kernel_mul,
                                             kernel_num=kernel_num)

            # 对应sigma的MMD损失
            k_ss = kernel_ss[source.shape[0]:, :source.shape[0]]
            k_tt = kernel_tt[:target.shape[0], target.shape[0]:]
            k_st = kernel_st[:source.shape[0], target.shape[0]:]
            k_ts = kernel_ts[source.shape[0]:, :target.shape[0]]
            mmd_loss += torch.mean(k_ss + k_tt - k_st - k_ts)
            # mmd_loss += torch.mean(kernel_ss) + torch.mean(kernel_tt) -  torch.mean(kernel_st) - torch.mean(kernel_ts)

        return mmd_loss

    def CORAL(self, source, target):
        d = source.data.shape[1]  # 获取特征的维度，即特征数

        # 计算源域的协方差矩阵
        xm = torch.mean(source, 0, keepdim=True) - source  # 计算源域每个样本与均值的差
        xc = xm.t() @ xm  # 协方差矩阵计算：X'X

        # 计算目标域的协方差矩阵
        xmt = torch.mean(target, 0, keepdim=True) - target  # 计算目标域每个样本与均值的差
        xct = xmt.t() @ xmt  # 协方差矩阵计算：Y'Y

        # 计算源域和目标域的协方差矩阵之间的 Frobenius 范数
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))  # Frobenius 范数的平方
        loss = loss / (4 * d * d)  # 正则化，按特征维度 d 进行归一化

        return loss

    def cosine_similarity_loss(self, source, target):
        """计算余弦相似度损失"""
        source_flat = source.view(source.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        cos_loss = 1 - F.cosine_similarity(source_flat, target_flat, dim=1).mean()

        return cos_loss

    def wasserstein_distance_loss(self, source, target):
        """计算Wasserstein距离损失"""

        # 计算源样本和目标样本之间的L2距离
        # 这里使用L2距离作为Wasserstein距离的一种度量方式
        n_source = source.size(0)
        n_target = target.size(0)

        # 计算所有源样本和目标样本之间的距离
        source_expanded = source.unsqueeze(1).expand(n_source, n_target, source.size(1))  # 扩展源样本
        target_expanded = target.unsqueeze(0).expand(n_source, n_target, target.size(1))  # 扩展目标样本

        # 计算L2距离
        l2_distance = torch.norm(source_expanded - target_expanded, p=2, dim=2)

        # 计算Wasserstein距离（取平均L2距离）
        wasserstein_loss = torch.mean(l2_distance)

        return wasserstein_loss



    def pseudo_labels(self, source, target, source_labels):
        """
        生成目标域伪标签
        :param source: 源域特征
        :param target: 目标域特征
        :param source_labels: 源域标签
        """
        # 归一化
        source_norm = F.normalize(source, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)

        # 计算余弦相似度
        sim_matrix = torch.mm(target_norm, source_norm.T)  # [N_target, N_source]

        # 最近邻标签
        _, indices = torch.max(sim_matrix, dim=1)
        pseudo_labels = source_labels[indices]

        return pseudo_labels
    def NT_Xent_Loss(self, source, target, source_labels):
        """
        基于标签的对比学习损失
        """
        device = source.device
        batch_size = source.shape[0] + target.shape[0]

        # 生成伪标签
        pseudo_labels = self.pseudo_labels(source, target, source_labels)

        # 合并特征
        combined_feature = torch.cat([source, target], dim=0)
        combined_labels = torch.cat([source_labels, pseudo_labels], dim=0)

        # 归一化特征
        combined_feature = F.normalize(combined_feature, p=2, dim=1)

        # 余弦相似度矩阵
        cosine_similarity = torch.mm(combined_feature, combined_feature.T)

        # 正样本掩码
        combined_labels = combined_labels.view(-1, 1)
        positive_mask = torch.eq(combined_labels, combined_labels.T).float().to(device)

        # 排除自身
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        cosine_similarity = cosine_similarity.masked_fill(mask, -float("inf"))

        # 计算损失
        exp_sim = torch.exp(cosine_similarity / self.temperature)
        positive_sim = torch.sum(exp_sim * positive_mask, dim=1)
        total_sim = torch.sum(exp_sim, dim=1)

        contrastive_loss = -torch.log(positive_sim / (total_sim + 1e-8)).mean()
        return contrastive_loss

    def gaussian_kernel1(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        计算多核Gaussian核矩阵，作为MMD的核函数
        """
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)  # 多核加权求和

    def lmmd(self, source_features, target_features, source_labels, target_preds, num_classes):
        """
        计算LMMD
        source_features: (n_source, feature_dim)
        target_features: (n_target, feature_dim)
        source_labels: (n_source,) int tensor
        target_preds: (n_target, num_classes) 概率预测，用于软标签
        num_classes: 类别数量
        """

        batch_size_source = source_features.size(0)
        batch_size_target = target_features.size(0)

        kernels = self.gaussian_kernel1(source_features, target_features)

        loss = 0.0
        for i in range(num_classes):
            # source子域掩码
            source_mask = (source_labels == i).float()
            # target子域掩码 (用预测概率作为软权重)
            target_mask = target_preds[:, i]

            # 归一化权重
            source_sum = source_mask.sum()
            target_sum = target_mask.sum()

            if source_sum.item() == 0 or target_sum.item() == 0:
                continue

            source_mask = source_mask / source_sum
            target_mask = target_mask / target_sum

            # 计算对应子域的MMD值
            XX = torch.matmul(source_mask.unsqueeze(0),
                              torch.matmul(kernels[:batch_size_source, :batch_size_source],
                                           source_mask.unsqueeze(1)))
            YY = torch.matmul(target_mask.unsqueeze(0),
                              torch.matmul(kernels[batch_size_source:, batch_size_source:],
                                           target_mask.unsqueeze(1)))
            XY = torch.matmul(source_mask.unsqueeze(0),
                              torch.matmul(kernels[:batch_size_source, batch_size_source:],
                                           target_mask.unsqueeze(1)))

            loss += XX + YY - 2 * XY

        return loss.squeeze()