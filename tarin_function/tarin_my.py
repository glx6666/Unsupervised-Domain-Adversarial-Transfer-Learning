from ulit.acc import *
import torch
import torch.nn.functional as F

def train(logger, source_loader, target_loader ,feature_extractor, classification_extractor, domain_extractor,
          criterion, domain_losses, optimizer, epoch, lr_scheduler, device, MMD_Loss):
    source_total_losses = AverageMeter('source_total_losses', ':.4f')
    class_losses = AverageMeter('class_loss',':.4f')
    source_train_acc = AverageMeter('source_train_acc', ':.4f')
    target_train_acc = AverageMeter('target_train_acc', ':.4f')
    entropy_min_losses = AverageMeter('entropy_min_loss', ':.4f')
    entropy_max_losses = AverageMeter('entropy_max_loss', ':.4f')
    weighted_domain_losses = AverageMeter('weighted_domain_loss', ':.4f')

    # switch to train mode
    feature_extractor.train()

    classification_extractor.train()
    domain_extractor.train()

    for (source_data, source_label), (target_data, target_label) in zip(source_loader, target_loader):
        feature_extractor.zero_grad()

        classification_extractor.zero_grad()
        domain_extractor.zero_grad()
        optimizer.zero_grad()
        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)
        target_label = target_label.to(device)
        # 计算特征
        sourcer_features = feature_extractor(source_data)
        target_features = feature_extractor(target_data)
        combines_features = torch.cat([sourcer_features, target_features], dim=0)
        #计算域对抗损失
        source_domain_label = torch.zeros(sourcer_features.size(0), 1).to(device)  # 源域 label=0
        target_domain_label = torch.ones(target_features.size(0), 1).to(device)  # 目标域 label=1
        domain_source_features = domain_extractor(sourcer_features)
        domain_target_features = domain_extractor(target_features)
        domain_loss = domain_losses(domain_source_features, source_domain_label.float()) + domain_losses(domain_target_features, target_domain_label.float())

        #计算分类损失
        source_class_pred = classification_extractor(sourcer_features)
        target_class_pred = classification_extractor(target_features)
        class_loss = criterion(source_class_pred, source_label.long()) 
        
        class_losses.update(class_loss.item(), source_label.size(0))
        #最小熵
        target_class_pred = classification_extractor(target_features.detach())
        target_probs = F.softmax(target_class_pred, dim=1)  # 转换为概率
        entropy_min_loss = entropy_loss(target_probs)  # 计算熵损失
        entropy_min_losses.update(entropy_min_loss.item(), target_class_pred.size(0))
        # #最大熵
        combines_preds = classification_extractor(combines_features)
        combines_preds = F.softmax(combines_preds, dim=1)
        entropy_max_loss = entropy_loss(combines_preds)
        entropy_max_losses.update(entropy_max_loss, combines_preds.size(0))
        #计算加权域损失
        entropy_weights = 1 + torch.exp(-entropy_loss(target_probs))  # 计算权重
        weighted_domain_loss = (entropy_weights * domain_loss).mean()  # 计算加权域损失
        weighted_domain_losses.update(weighted_domain_loss.item(), source_domain_label.size(0))
        #计算特征MMD
        MMDloss = MMD_Loss(sourcer_features, target_features)
        #cosLoss = MMD_Loss.cosine_similarity_loss(sourcer_features, target_features)
        #wasserstein_distance_loss = MMD_Loss.wasserstein_distance_loss(sourcer_features, target_features)

        #计算总损失

        source_total_loss = class_loss +  entropy_min_loss -  entropy_max_loss + weighted_domain_loss + MMDloss

        
        _, source_predicted = torch.max(source_class_pred, 1)
        _, target_predicted = torch.max(target_class_pred, 1)
        source_accuracy = (source_predicted == source_label).sum().item() / source_label.size(0)
        target_accuracy = (target_predicted == target_label).sum().item() / target_label.size(0)
        source_train_acc.update(source_accuracy, source_label.size(0))
        target_train_acc.update(target_accuracy, target_label.size(0))
        source_total_loss.backward()
        optimizer.step()
    lr_scheduler.step()
    logger.info(f'Epoch:[{epoch}] train_Acc:{source_train_acc.avg:.4f} train_Loss:{source_total_losses.avg:.4f} target_acc:{target_train_acc.avg:.4f} class_loss:{class_losses.avg:.4f} weighted_domain_loss:{weighted_domain_losses.avg:.4f} ' \
          f'entropy_min_loss: {entropy_min_losses.avg:.4f}  entropy_max_loss: {entropy_max_losses.avg:.4f} ' )
    return source_train_acc.avg,source_total_losses.avg, class_losses.avg, weighted_domain_losses.avg






def test(logger, test_loader, feature_extractor, classification_extractor, domain_extractor, criterion, epoch, device):
    losses = AverageMeter('Loss', ':.4f')
    test_acc = AverageMeter('Acc', ':.4f')
    # switch to train mode
    feature_extractor.eval()
    classification_extractor.eval()
    domain_extractor.eval()
    all_preds = []  # 预测标签
    all_labels = []  # 真实标签
    with torch.no_grad():
        for i, (test_data, test_label) in enumerate(test_loader):
            test_data = test_data.to(device)
            test_label = test_label.to(device)

            test_features = feature_extractor(test_data)
            class_pred = classification_extractor(test_features)
            loss = criterion(class_pred, test_label.long())
            losses.update(loss.item(), test_label.size(0))
            # Compute accuracy
            _, predicted = torch.max(class_pred, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(test_label.cpu().numpy())
            accuracy = (predicted == test_label).sum().item() / test_label.size(0)
            test_acc.update(accuracy, test_label.size(0))
        logger.info(f'Epoch:[{epoch}] test_Acc:{test_acc.avg:.4f} test_Loss:{losses.avg:.4f}')
    return test_acc.avg, losses.avg, all_labels, all_preds




def entropy_loss(predictions):
    epsilon = 1e-5  # 避免 log(0)
    entropy = -torch.sum(predictions * torch.log(predictions + epsilon), dim=1)  # 按类别求和

    return entropy.mean()  # 取 batch 平均
