import torch
from ulit.acc import *
from sklearn.cluster import KMeans
import torch.nn.functional as F

def train_CAT(logger,source_loader, target_loader, teacher_model, student_model,epoch,
              optimizer_t, optimizer_s, lr_scheduler_t, lr_scheduler_s, criterion, device, num_classes, lambda_cluster=0.5):
    source_teacher_losses = AverageMeter('source_loss', ':.4f')
    source_teacher_acc = AverageMeter('source_acc', ':.4f')
    target_student_losses = AverageMeter('target_loss', ':.4f')
    target_teacher_acc = AverageMeter('source_acc', ':.4f')

    teacher_model.train()
    for i,(source_data, source_label) in enumerate(source_loader):
        teacher_model.zero_grad()
        source_data, source_label = source_data.to(device), source_label.to(device)

        optimizer_t.zero_grad()
        out_s, _= teacher_model(source_data)
        loss = criterion(out_s, source_label)

        _, pred_s = torch.max(out_s, 1)
        source_acc = (pred_s == source_label).sum().item() / source_label.size(0)

        source_teacher_losses.update(loss.item(), source_label.size(0))
        source_teacher_acc.update(source_acc, source_label.size(0))

        loss.backward()
        optimizer_t.step()

    lr_scheduler_t.step()

    teacher_model.eval()
    student_model.train()

    # 收集目标域教师特征做 epoch 级 KMeans
    all_teacher_features = []
    with torch.no_grad():
        for i,(source_data, _  ) in enumerate(target_loader):
            source_data = source_data.to(device)
            _, feat_T = teacher_model(source_data)
            all_teacher_features.append(feat_T.cpu())
    all_teacher_features = torch.cat(all_teacher_features, dim=0).numpy()
    if all_teacher_features.ndim == 3:
        all_teacher_features = all_teacher_features.reshape(all_teacher_features.shape[0], -1)

    # KMeans 聚类教师特征
    kmeans_T = KMeans(n_clusters=num_classes, random_state=0).fit(all_teacher_features)
    centers_T = torch.tensor(kmeans_T.cluster_centers_, dtype=torch.float32).to(device)
    centers_T = F.normalize(centers_T, dim=1)
    #print(centers_T.shape)

    for i,(target_data, target_label) in enumerate(target_loader):
        student_model.zero_grad()
        target_data = target_data.to(device)
        target_label = target_label.to(device)
        optimizer_s.zero_grad()

        # 教师软标签
        with torch.no_grad():
            out_T, _ = teacher_model(target_data)
            p_T = F.softmax(out_T, dim=1)

        # 学生预测
        out_S, feat_S = student_model(target_data)
        _, p_s = torch.max(out_S, dim=1)
        p_S = F.normalize(out_S, dim=1)
        #print(p_S.shape)
        #print(target_label.shape)

        # KL散度软标签对齐
        loss_soft = F.kl_div(p_S.log(), p_T, reduction='batchmean')

        # 聚类对齐
        feat_S_norm = F.normalize(feat_S, dim=1)
        feat_S_norm = feat_S_norm.squeeze(-1)
        #print(feat_S_norm.shape)
        dist = torch.cdist(feat_S_norm, centers_T, p=2)  # (batch, num_classes)
        loss_cluster = dist.min(dim=1)[0].mean()

        # 总损失
        loss_total = loss_soft + lambda_cluster * loss_cluster
        loss_total.backward()
        optimizer_s.step()

        target_acc = (p_s == target_label).sum().item() / target_label.size(0)
        target_student_losses.update(loss_total.item(), target_data.size(0))
        target_teacher_acc.update(target_acc, target_label.size(0))
    lr_scheduler_s.step()
    logger.info(f'Epoch:[{epoch}] source_acc:{source_teacher_acc.avg:.4f} source_losses:{source_teacher_losses.avg:.4f} target_acc:{target_teacher_acc.avg:.4f} target_losses:{target_student_losses.avg:.4f}')
    return source_teacher_acc.avg, source_teacher_losses.avg, target_teacher_acc.avg, target_student_losses.avg

def test_CAT(logger, epoch,val_loader, student_model,device, criterion):
    losses = AverageMeter('Loss', ':.4f')
    test_acc = AverageMeter('Acc', ':.4f')
    student_model.eval()
    all_preds = []  # 预测标签
    all_labels = []  # 真实标签
    with torch.no_grad():
        for i, (val_data, val_label) in enumerate(val_loader):
            val_data = val_data.to(device)
            val_label = val_label.to(device)
            val_pred, _ = student_model(val_data)
            loss = criterion(val_pred, val_label)
            _, pred_class = torch.max(val_pred, 1)
            losses.update(loss.item(), val_label.size(0))
            acc = (pred_class == val_label).sum().item() / val_label.size(0)
            test_acc.update(acc, val_label.size(0))
            all_preds.extend(pred_class.cpu().numpy())
            all_labels.extend(val_label.cpu().numpy())
        logger.info(f'Epoch:[{epoch}] test_Acc:{test_acc.avg:.4f} test_Loss:{losses.avg:.4f}')
    return test_acc.avg, losses.avg, all_labels, all_preds