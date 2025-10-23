from ulit.acc import *
import torch

def tarin_DANN(logger, source_loader, target_loader ,feature_extractor, classification_extractor, domain_extractor,
          criterion, domain_losses, optimizer, epoch, lr_scheduler, device, MMD_Loss):
    source_total_losses = AverageMeter('source_total_losses', ':.4f')
    source_total_acc = AverageMeter('source_total_acc', ':.4f')
    weight_domain_losses = AverageMeter('domain_loss', ':.4f')

    feature_extractor.train()
    classification_extractor.train()
    domain_extractor.train()

    for (source_data, source_label), (target_data, _) in zip(source_loader, target_loader):
        feature_extractor.zero_grad()
        classification_extractor.zero_grad()
        domain_extractor.zero_grad()
        optimizer.zero_grad()
        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)

        source_features = feature_extractor(source_data)
        target_features = feature_extractor(target_data)

        source_pred_class = classification_extractor(source_features)
        #target_pred_class = classification_extractor(target_features)

        domain_source = domain_extractor(source_features)
        domain_target = domain_extractor(target_features)

        source_domain_label = torch.zeros(source_features.size(0), 1).to(device)  # 源域 label=0
        target_domain_label = torch.ones(target_features.size(0), 1).to(device)  # 目标域 label=1

        loss_class = criterion(source_pred_class, source_label)
        weight_domain_loss = domain_losses(domain_source, source_domain_label) + \
            domain_losses(domain_target, target_domain_label)
        weight_domain_losses.update(weight_domain_loss.item(), source_domain_label.size(0) + target_domain_label.size(0))
        mmd = MMD_Loss(source_features, target_features)

        loss = loss_class + weight_domain_loss + mmd
        source_total_losses.update(loss.item(), source_label.size(0))
        _, pred = torch.max(source_pred_class, 1)
        acc = (pred == source_label).sum().item() / source_label.size(0)
        source_total_acc.update(acc, source_label.size(0))
        loss.backward()
        optimizer.step()
    lr_scheduler.step()
    logger.info(f'Epoch:[{epoch}] source_acc:{source_total_acc.avg:.4f} source_losses:{source_total_losses.avg:.4f}')
    return source_total_acc.avg, source_total_losses.avg

def test_DANN(logger, val_loader, feature_extractor, classification_extractor, criterion, epoch, device):
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
            val_losses.update(loss.item(), val_label.size(0))
            _, pred = torch.max(val_pred, 1)
            acc = (pred == val_label).sum().item() / val_label.size(0)
            val_acc.update(acc, val_label.size(0))
            all_label.extend(val_label.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
        logger.info(f'Epoch:[{epoch}] val_acc:{val_acc.avg:.4f} val_losses:{val_losses.avg:.4f}')
        return val_acc.avg, val_losses.avg, all_pred, all_label