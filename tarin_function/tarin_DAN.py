from ulit.acc import *
import torch


def tarin_DAN(logger, epoch,feature_extractor, classification_extractor, source_loader, target_loader, criterion,optimizer, lr_scheduler, device, MMD_Loss):
    source_losses = AverageMeter('source_total_losses', ':.4f')
    source_acc = AverageMeter('source_total_acc', ':.4f')
    feature_extractor.train()
    classification_extractor.train()

    for (source_data, source_label), (target_data, _) in zip(source_loader, target_loader):
        feature_extractor.zero_grad()
        classification_extractor.zero_grad()
        optimizer.zero_grad()
        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)

        source_features = feature_extractor(source_data)
        target_features = feature_extractor(target_data)

        source_pred = classification_extractor(source_features)

        loss_class = criterion(source_pred, source_label)
        mmd = MMD_Loss(source_features, target_features)
        loss = loss_class + mmd
        source_losses.update(loss.item(), source_label.size(0))
        _, pred = torch.max(source_pred, 1)
        acc = (pred == source_label).sum().item() / source_label.size(0)
        source_acc.update(acc, source_label.size(0))
        loss.backward()
        optimizer.step()
    lr_scheduler.step()
    logger.info(f'Epoch:[{epoch}] source_acc:{source_acc.avg:.4f} source_losses:{source_losses.avg:.4f}')
    return source_acc.avg, source_losses.avg

def test_DAN(logger, val_loader, feature_extractor, classification_extractor, criterion, epoch, device):
    val_acc = AverageMeter('val_acc', ':.4f')
    val_losses = AverageMeter('val_losses', ':.4f')
    feature_extractor.eval()
    classification_extractor.eval()
    all_pred = []
    all_label = []
    with torch.no_grad():
        for i, (val_data, val_label) in enumerate(val_loader):
            val_data, val_label = val_data.to(device), val_label.to(device)
            val_features = feature_extractor(val_data)
            val_pred = classification_extractor(val_features)
            loss = criterion(val_pred, val_label)
            _, pred = torch.max(val_pred, 1)
            acc = (pred == val_label).sum().item() / val_label.size(0)
            val_acc.update(acc, val_label.size(0))
            val_losses.update(loss.item(), val_label.size(0))
            all_label.extend(val_label.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
    logger.info(f'Epoch:[{epoch}] val_acc:{val_acc.avg:.4f} val_losses:{val_losses.avg:.4f}')
    return val_acc.avg, val_losses.avg, all_pred, all_label