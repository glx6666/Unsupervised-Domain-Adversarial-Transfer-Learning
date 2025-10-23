from ulit.acc import *
import torch
import torch.nn.functional as F


def classifier_discrepancy(p1, p2):
    #print("p1 shape:", p1.shape)
    #print("p2 shape:", p2.shape)
    # p1 = p1.float()
    # p2 = p2.float()
    return torch.mean(torch.sum(torch.abs(p1 - p2)))  # batch mean of L1
def tarin_MCD(logger,feature_extractor, classifier1, classifier2,source_loader, target_loader,optimizer_g, optimizer_f, criterion, device, epoch,lr_scheduler_f, lr_scheduler_c,n_step_B=4):
    """
    n_step_B: 在 Step B（固定 G，训练 F1/F2 最大化 discrepancy）中迭代的 steps（通常 >1）
    """
    source_losses = AverageMeter('source_loss', ':.4f')
    source_acc = AverageMeter('source_acc', ':.4f')
    losses_dis = AverageMeter('target_loss', ':.4f')
    losses_g = AverageMeter('source_acc', ':.4f')
    feature_extractor.train()
    classifier1.train()
    classifier2.train()
    for i, (source_data, source_label) in enumerate(source_loader):
        feature_extractor.zero_grad()
        classifier1.zero_grad()
        classifier2.zero_grad()
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        source_data, source_label = source_data.to(device), source_label.to(device)
        source_features = feature_extractor(source_data)
        output_source_pred1 = classifier1(source_features)
        output_source_pred2 = classifier2(source_features)
        _, pred_source_class = torch.max(output_source_pred1, 1)

        loss1 = criterion(output_source_pred1, source_label)
        loss2 = criterion(output_source_pred2, source_label)
        loss = loss1 + loss2
        loss.backward()
        optimizer_f.step()
        optimizer_g.step()
        source_losses.update(loss.item(), source_label.size(0))
        acc = (pred_source_class == source_label).sum().item() / source_label.size(0)
        source_acc.update(acc, source_label.size(0))
    lr_scheduler_f.step()
    lr_scheduler_c.step()
    feature_extractor.eval()
    for _ in range(n_step_B):
        for i, (target_data, _) in enumerate(target_loader):
            optimizer_g.zero_grad()
            target_data = target_data.to(device)
            target_features = feature_extractor(target_data)
            output_target_pred1 = classifier1(target_features)
            output_target_pred2 = classifier2(target_features)
            target_pred1 = F.softmax(output_target_pred1, dim=1)
            target_pred2 = F.softmax(output_target_pred2, dim=1)
            loss_dis = classifier_discrepancy(target_pred1, target_pred2)
            losses_dis.update(loss_dis.item(), source_label.size(0))
            loss_dis.backward()
            optimizer_g.step()
    lr_scheduler_c.step()
    classifier2.eval()
    classifier1.eval()
    feature_extractor.train()
    for i, (target_data, target_label) in enumerate(target_loader):
        target_data, target_label = target_data.to(device), target_label.to(device)
        optimizer_f.zero_grad()
        target_features = feature_extractor(target_data)
        target_pred1 = F.softmax(classifier1(target_features), dim=1)
        target_pred2 = F.softmax(classifier2(target_features), dim=1)

        loss_g = classifier_discrepancy(target_pred1, target_pred2)
        loss_g.backward()
        optimizer_f.step()
        losses_g.update(loss_g.item(), target_label.size(0))
    lr_scheduler_f.step()
    logger.info(f'Epoch:[{epoch}] source_acc:{source_acc.avg:.4f} source_losses:{source_losses.avg:.4f} losses_dis:{losses_dis.avg:.4f} losses_g:{losses_g.avg:.4f}')
    return source_acc.avg, source_losses.avg, losses_dis.avg, losses_g.avg

def test_MCD(epoch,feature_extractor, classifier1, classifier2, val_loader, device, criterion, logger):
    val_losses = AverageMeter('source_loss', ':.4f')
    val_acc = AverageMeter('source_acc', ':.4f')
    feature_extractor.eval()
    classifier1.eval()
    classifier2.eval()
    all_preds = []  # 预测标签
    all_labels = []  # 真实标签
    with torch.no_grad():
        for i, (val_data, val_label) in enumerate(val_loader):
            val_data, val_label = val_data.to(device), val_label.to(device)
            val_features = feature_extractor(val_data)
            val_pred1 = classifier1(val_features)
            val_pred2 = classifier2(val_features)
            loss = criterion(val_pred1, val_label) + criterion(val_pred2, val_label)
            val_losses.update(loss.item(), val_label.size(0))
            val_pred = (val_pred1 + val_pred2) / 2
            _, pred = torch.max(val_pred, 1)
            acc = (pred == val_label).sum().item() / val_label.size(0)
            val_acc.update(acc, val_label.size(0))
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(val_label.cpu().numpy())  # 真实标签
        logger.info(f'Epoch:[{epoch}] val_acc:{val_acc.avg:.4f} val_losses:{val_losses.avg:.4f}')
        return val_acc.avg, val_losses.avg, all_preds, all_labels