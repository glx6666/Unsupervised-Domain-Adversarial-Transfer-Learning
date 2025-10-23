from ulit.acc import *
import torch


def tarin_JAN(source_loader, target_loader, model, optimizer, device,
              epoch, jmmd, criterion, logger, lr_scheduler):
    source_losses = AverageMeter('source_loss', ':.4f')
    source_acc = AverageMeter('source_acc', ':.4f')
    model.train()
    #target_iter = iter(target_loader)
    for (source_data, source_label), (target_data, _) in zip(source_loader, target_loader):
        optimizer.zero_grad()
        source_data, source_label = source_data.to(device), source_label.to(device)
        target_data = target_data.to(device)
        #target_data = target_data.to(device)
        source_pred, source_features = model(source_data)
        target_pred, target_features = model(target_data)
        loss_class = criterion(source_pred, source_label)
        source_losses.update(loss_class.item(), source_label.size(0))
        loss_jmmd = jmmd(source_features, target_features)
        loss = loss_class + loss_jmmd
        source_losses.update(loss.item(), source_label.size(0))
        _, pred = torch.max(source_pred, 1)
        acc = (pred == source_label).sum().item() / source_label.size(0)
        source_acc.update(acc, source_label.size(0))
        loss.backward()
        optimizer.zero_grad()
    lr_scheduler.step()
    logger.info(f'Epoch:[{epoch}] source_acc:{source_acc.avg:.4f} source_losses:{source_losses.avg:.4f}')
    return source_acc.avg, source_losses.avg

def test_JAN(model, val_loader, device, criterion, logger, epoch):
    model.eval()
    losses = AverageMeter('Loss', ':.4f')
    test_acc = AverageMeter('Acc', ':.4f')
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for i, (val_data, val_label) in enumerate(val_loader):
            val_data, val_label = val_data.to(device), val_label.to(device)
            val_pred, _ = model(val_data)
            loss = criterion(val_pred, val_label)
            losses.update(loss.item(), val_label.size(0))
            _, pred = torch.max(val_pred, 1)
            acc = (pred == val_label).sum().item() / val_label.size(0)
            test_acc.update(acc, val_label.size(0))
            all_labels.extend(val_label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
    logger.info(f'Epoch:[{epoch}] val_acc:{test_acc.avg:.4f} val_losses:{losses.avg:.4f}')
    return test_acc.avg, losses.avg, all_labels, all_preds
