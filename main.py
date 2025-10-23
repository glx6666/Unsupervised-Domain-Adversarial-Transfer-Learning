import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os
os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import time
import torch.optim
import torch.utils.data
from ulit.init_seed import *
from ulit.acc import *
from ulit.CWRU import *
from model.CAT import *
from model.CDAN import *
from model.JAN import *
from model.DAN import *
from model.DANN import *
from model.MCD import *
#import tarin function
from tarin_function.tarin_MCD import *
from tarin_function.tarin_my import *
from tarin_function.tarin_CAT import *
from tarin_function.tarin_DAN import *
from tarin_function.tarin_DANN import *
from tarin_function.tarin_JAN import *
#plot_function
from plot_figture.CAT_plot import *
from plot_figture.DAN_plot import *
from plot_figture.DANN_plot import *
from plot_figture.JAN_plot import *
from plot_figture.MCD_plot import *
from plot_figture.my_method_plot import *
from model.domain_extractor import *
from model.feature_extractor import *
from model.classification_extractor import *
#loss function
from loss import *
import matplotlib.pyplot as plt
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from sklearn.manifold import TSNE


import numpy
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['times new roman']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号val

import os
import logging
from datetime import datetime

def setup_logger(log_dir, log_filename):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler（如果多次调用 setup_logger）
    if not logger.handlers:
        # 控制台日志输出
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)

        # 文件日志输出
        file = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file.setLevel(logging.INFO)
        file.setFormatter(formatter)

        logger.addHandler(console)
        logger.addHandler(file)

    return logger

# def main(v, f1, f2, model):
def main(v, f1, f2, model):
    parser = argparse.ArgumentParser(description='PyTorch PN_Data Training')
    parser.add_argument('--source_data', metavar='DIR', default=\\source_data\\{v}\\{f1}', help='path to dataset')
    parser.add_argument('--target_data', metavar='DIR', default=\\target_data\\{v}\\{f2}', help='path to dataset')
    parser.add_argument('--test_data', metavar='DIR', default=\\val_data\\{v}\\{f2}', help='path to dataset')
    parser.add_argument('--classes', default=5, type=int, metavar='N', help='number of classification')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default = 64, type=int,metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('-gamma', default=0.1, type=float)
    parser.add_argument('-stepsize', default=10, type=int)
    parser.add_argument('-seed', default=123, type=int)
    parser.add_argument('-use_model', default= model, help='use model')
    # save
    parser.add_argument('--save_or_not', default=True, type=bool)
    parser.add_argument('--save_model', default=True, type=bool)
    parser.add_argument('--save_dir', default='.\result', type=str, help='save_root')
    parser.add_argument('--save_acc_loss_dir', default='.\result', type=str)

    # 判断是否含有gpu，否则加载到cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("use gpu")
    else:
        device = torch.device('cpu')
        print("use cpu")
    args = parser.parse_args()
    init_seed(args.seed)  # 初始化随机种子参数
    criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵损失
    domain_losses = nn.BCELoss().to(device)
    # 构建模型
    if args.use_model == 'CAT':
        print('use CAT')
        student_model = CAT(args.classes).to(device)
        teacher_model = CAT(args.classes).to(device)
        optimizer_s = torch.optim.Adam(
            list(student_model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        optimizer_t = torch.optim.Adam(
            list(teacher_model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        lr_scheduler_t = StepLR(optimizer_t, gamma=args.gamma, step_size=args.stepsize)
        lr_scheduler_s = StepLR(optimizer_s, gamma=args.gamma, step_size=args.stepsize)

    elif args.use_model == 'DAN':
        source_acc_list_DAN, source_losses_list_DAN, val_acc_list_DAN, val_losses_list_DAN = [], [], [], []
        print('use DAN')
        feature_extractor_DAN = DAN_F().to(device)
        classification_extractor_DAN = DAN_C(args.classes).to(device)
        optimizer_DAN = torch.optim.Adam(
            list(feature_extractor_DAN.parameters()) +
            list(classification_extractor_DAN.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        lr_scheduler_DAN = StepLR(optimizer_DAN, gamma=args.gamma, step_size=args.stepsize)


    elif args.use_model == 'DANN':
        source_total_acc_list_DANN, source_total_losses_list_DANN, val_acc_list_DANN, val_losses_list_DANN = [], [], [], []
        print('use DANN')
        feature_extractor_DANN = DANN_F().to(device)
        classification_extractor_DANN = DANN_C(args.classes).to(device)
        domain_extractor_DANN = DANN_D().to(device)
        optimizer_DANN = torch.optim.Adam(
            list(feature_extractor_DANN.parameters()) +
            list(classification_extractor_DANN.parameters()) +
            list(domain_extractor_DANN.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        lr_scheduler_DANN = StepLR(optimizer_DANN, gamma=args.gamma, step_size=args.stepsize)
    elif args.use_model == 'JAN':
        print('use JAN')
        JAN_model = JAN(args.classes).to(device)
        optimizer_JAN = torch.optim.Adam(
            list(JAN_model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        lr_scheduler_JAN = StepLR(optimizer_JAN, gamma=args.gamma, step_size=args.stepsize)
    elif args.use_model == 'MCD':
        print('use MCD')
        feature_extractor_MCD = MCD_F().to(device)
        classification_extractor1 = MCD_C1(args.classes).to(device)
        classification_extractor2 = MCD_C2(args.classes).to(device)
        optimizer_f = torch.optim.Adam(
            list(feature_extractor_MCD.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        optimizer_g = torch.optim.Adam(
            list(classification_extractor1.parameters()) +
            list(classification_extractor2.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        lr_scheduler_f = StepLR(optimizer_f, gamma=args.gamma, step_size=args.stepsize)
        lr_scheduler_g = StepLR(optimizer_g, gamma=args.gamma, step_size=args.stepsize)
    elif args.use_model == 'my_method':
        feature_extractor = ResNet18_source().to(device)
        classification_extractor = ClassificationExtractor(args.classes).to(device)
        domain_extractor = DomainExtractor().to(device)
        optimizer = torch.optim.Adam(
            list(feature_extractor.parameters()) +
            list(classification_extractor.parameters()) +
            list(domain_extractor.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        lr_scheduler = StepLR(optimizer, gamma=args.gamma, step_size=args.stepsize)

    save_dir1 = os.path.join(args.save_dir, f'{model}_')
    save_dir = os.path.join(save_dir1, args.use_model + f'{v}_{f1}_{v}_{f2}')
    print(f'数据保存到{save_dir}')
    logger = setup_logger(save_dir, log_filename='train.log')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 如果文件夹不存在，则创建文件夹
    args.save_acc_loss_dir = os.path.join(save_dir, 'train_test_result.csv')
    cwru_data = CWRU(args.source_data, args.target_data, args.test_data)
    source_dataset, target_dataset, test_dataset = cwru_data.train_test_split_order()
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    # 训练、测试模型
    #source_train_acc_list,source_total_loss_list, class_loss_list, domain_loss_list = [], [], [], []
    #test_acc_list, test_loss_list = [], []
    source_train_acc_list, source_total_losses_list, class_losses_list, weighted_domain_losses_list, test_acc_list, losses_list = [], [], [], [], [], []
    fault_classes = ['0','1','2','3','4']
    start_time = time.time()
    logger.info(f"Start training...--{model}")
    for epoch in range(args.start_epoch, args.epochs):
        if args.use_model == 'CAT':
            
            source_teacher_acc, source_teacher_losses, target_teacher_acc, target_student_losses = train_CAT(logger, source_loader, target_loader, teacher_model, student_model, epoch,
                      optimizer_t, optimizer_s, lr_scheduler_t, lr_scheduler_s, criterion, device, args.classes,
                      lambda_cluster=0.5)
            test_acc, losses, all_labels, all_preds = test_CAT(logger, epoch, test_loader, student_model, device, criterion)
            source_teacher_acc_list.append(round(source_teacher_acc,4))
            source_teacher_losses_list.append(round(source_teacher_losses,4))
            target_teacher_acc_list.append(round(target_teacher_acc,4))
            target_student_losses_list.append(round(target_student_losses,4))
            test_acc_list.append(round(test_acc,4))
            losses_list.append(round(losses,4))
            if args.save_model and epoch == args.epochs -1:

                student_model_name = 'student_model' + '_' + str(epoch + 1) + '.pth'
                teacher_model_name = 'teacher_model' + '_' + str(epoch + 1) + '.pth'
                torch.save(student_model.state_dict(), os.path.join(save_dir, student_model_name))
                torch.save(teacher_model.state_dict(), os.path.join(save_dir, teacher_model_name))
                test_acc, losses, all_labels, all_preds = test_CAT(logger, epoch, test_loader, student_model, device,
                                                                  criterion)
                logger.info(f'真实类别:{set(all_labels)} 预测类别:{set(all_preds)} ')

                cm = confusion_matrix(all_labels, all_preds)
                np.savetxt(os.path.join(save_dir, 'confusion_matrix.txt'), cm, fmt='%d')
                try:
                    report = classification_report(all_labels, all_preds,
                                                   target_names=fault_classes)  # [str(label) for label in sorted(set(true_labels))]
                    print(report)
                    end_time = time.time()
                    print(f'训练时间:{end_time - start_time :.4f}')
                    print('test_acc:', test_acc, 'test_loss:', losses)
                except Exception as e:
                    print(f'错误原因：{str(e)}')
                    pass
                with open(args.save_acc_loss_dir, 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(
                        ['Epoch', 'source_teacher_acc', 'source_teacher_losses', 'target_teacher_acc', 'target_student_losses', 'test_acc',
                         'losses_list'])  # 写入表头
                    for epoch in range(len(source_teacher_acc_list)):
                        writer.writerow([epoch + 1, source_teacher_acc_list[epoch], source_teacher_losses_list[epoch],
                                         target_teacher_acc_list[epoch], target_student_losses_list[epoch], test_acc_list[epoch],
                                         losses_list[epoch]])
                plot_CAT(source_teacher_acc_list, source_teacher_losses_list, target_teacher_acc_list, target_student_losses_list, test_acc_list, losses_list, cm, save_dir, fault_classes)
                plot_tsne_CAT(args, save_dir, student_model, teacher_model, student_model_name, teacher_model_name)
        if args.use_model == 'DAN':
           
            source_acc, source_losses = tarin_DAN(logger, epoch, feature_extractor_DAN, classification_extractor_DAN, source_loader, target_loader,
                      criterion, optimizer_DAN, lr_scheduler_DAN, device, maximum_mean_discrepancy)
            val_acc, val_losses, all_pred, all_label = test_DAN(logger, test_loader, feature_extractor_DAN, classification_extractor_DAN, criterion, epoch, device)
            source_acc_list_DAN.append(round(source_acc, 4))
            source_losses_list_DAN.append(round(source_losses, 4))
            val_acc_list_DAN.append(round(val_acc, 4))
            val_losses_list_DAN.append(round(val_losses, 4))
            if args.save_model and epoch == args.epochs -1:
                feature_extractor_name = 'feature_extractor_DAN' + '_' + str(epoch + 1) + '.pth'
                classification_extractor_name = 'classification_extractor_DAN' + '_' + str(epoch + 1) + '.pth'
                torch.save(feature_extractor_DAN.state_dict(), os.path.join(save_dir, feature_extractor_name))
                torch.save(classification_extractor_DAN.state_dict(), os.path.join(save_dir, classification_extractor_name))
                val_acc, val_losses, all_pred, all_label = test_DAN(logger, test_loader, feature_extractor_DAN,
                                                                    classification_extractor_DAN, criterion, epoch, device)
                logger.info(f'真实类别:{set(all_label)} 预测类别:{set(all_pred)} ')
                cm = confusion_matrix(all_label, all_pred)
                np.savetxt(os.path.join(save_dir, 'confusion_matrix.txt'), cm, fmt='%d')
                try:
                    report = classification_report(all_label, all_pred,
                                               target_names=fault_classes)
                    print(report)

                    end_time = time.time()
                    print(f'训练时间:{end_time - start_time :.4f}')
                except Exception as e:
                    print(f'错误原因：{str(e)}')
                    pass
                with open(args.save_acc_loss_dir, 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(
                        ['Epoch', 'source_acc', 'source_losses', 'val_acc',
                         'val_losses'])  # 写入表头
                    for epoch in range(len(source_acc_list_DAN)):
                        writer.writerow([epoch + 1, source_acc_list_DAN[epoch], source_losses_list_DAN[epoch],
                                         val_acc_list_DAN[epoch], val_losses_list_DAN[epoch],])
                plot_DAN(source_acc_list_DAN, source_losses_list_DAN, val_acc_list_DAN, val_losses_list_DAN, cm, save_dir, fault_classes)
                plot_tsne_DAN(args, save_dir, feature_extractor_DAN, classification_extractor_DAN,
                              feature_extractor_name, classification_extractor_name)
        if args.use_model == 'DANN':
            
            source_total_acc, source_total_losses = tarin_DANN( logger, source_loader, target_loader, feature_extractor_DANN, classification_extractor_DANN,
                       domain_extractor_DANN, criterion, domain_losses, optimizer_DANN, epoch, lr_scheduler_DANN, device, maximum_mean_discrepancy)
            val_acc, val_losses, all_pred, all_label= test_DANN(logger, test_loader, feature_extractor_DANN, classification_extractor_DANN, criterion, epoch, device)
            source_total_acc_list_DANN.append(round(source_total_acc, 4))
            source_total_losses_list_DANN.append(round(source_total_losses, 4))
            val_acc_list_DANN.append(round(val_acc, 4))
            val_losses_list_DANN.append(round(val_losses, 4))
            if args.save_model and epoch == args.epochs -1:
                feature_extractor_name = 'feature_extractor_DANN' + '_' + str(epoch + 1) + '.pth'
                classification_extractor_name = 'classification_extractor_DANN' + '_' + str(epoch + 1) + '.pth'
                torch.save(feature_extractor_DANN.state_dict(), os.path.join(save_dir, feature_extractor_name))
                torch.save(classification_extractor_DANN.state_dict(), os.path.join(save_dir, classification_extractor_name))
                val_acc, val_losses, all_pred, all_label = test_DANN(logger, test_loader, feature_extractor_DANN, classification_extractor_DANN, criterion, epoch, device)
                logger.info(f'真实类别:{set(all_label)} 预测类别:{set(all_pred)} ')
                cm = confusion_matrix(all_label, all_pred)
                np.savetxt(os.path.join(save_dir, 'confusion_matrix.txt'), cm, fmt='%d')
                try:
                    report = classification_report(all_label, all_pred,
                                                   target_names=fault_classes)
                    print(report)
                    end_time = time.time()
                    print(f'训练时间:{end_time - start_time :.4f}')
                except Exception as e:
                    print(f'错误原因：{str(e)}')
                    pass
                with open(args.save_acc_loss_dir, 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(
                        ['Epoch', 'source_total_acc', 'source_total_losses', 'val_acc',
                         'val_losses'])
                    for epoch in range(len(source_total_acc_list_DANN)):
                        writer.writerow([epoch + 1, source_total_acc_list_DANN[epoch], source_total_losses_list_DANN[epoch], val_acc_list_DANN[epoch], val_losses_list_DANN[ epoch]])
                plot_DANN(source_total_acc_list_DANN, source_total_losses_list_DANN, val_acc_list_DANN, val_losses_list_DANN, cm, save_dir, fault_classes)
                plot_tsne_DANN(args, save_dir, feature_extractor_DANN, classification_extractor_DANN,
                               feature_extractor_name, classification_extractor_name)
        if args.use_model == 'JAN':
            
            source_acc, source_losses=tarin_JAN(source_loader, target_loader, JAN_model, optimizer_JAN, device,
                      epoch, jmmd_loss, criterion, logger, lr_scheduler_JAN)
            test_acc, losses, all_labels, all_preds=test_JAN(JAN_model, test_loader, device, criterion, logger, epoch)
            source_acc_list.append(round(source_acc, 4))
            source_losses_list.append(round(source_losses, 4))
            test_acc_list.append(round(test_acc, 4))
            losses_list.append(round(losses, 4))
            if args.save_model and epoch == args.epochs -1:
                JAN_model_name = 'JAN_model' + '_' + str(epoch + 1) + '.pth'
                torch.save(JAN_model.state_dict(), os.path.join(save_dir, JAN_model_name))
                val_acc, val_losses, all_pred, all_label = test_JAN(JAN_model, test_loader, device, criterion, logger, epoch)
                logger.info(f'真实类别:{set(all_label)} 预测类别:{set(all_pred)} ')
                cm = confusion_matrix(all_label, all_pred)
                np.savetxt(os.path.join(save_dir, 'confusion_matrix.txt'), cm, fmt='%d')
                try :
                    report = classification_report(all_label, all_pred, target_names=fault_classes)
                    print(report)
                    end_time = time.time()
                    print(f'训练时间:{end_time - start_time :.4f}')
                except Exception as e:
                    print(f'错误原因：{str(e)}')
                    pass
                with open(args.save_acc_loss_dir, 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(
                        ['Epoch', 'source_acc', 'source_losses', 'test_acc',
                         'test_losses']
                    )
                    for epoch in range(len(source_acc_list)):
                        writer.writerow([epoch + 1, source_acc_list[epoch], source_losses_list[epoch], test_acc_list[epoch], losses_list[epoch]])
                plot_JAN(source_acc_list, source_losses_list,test_acc_list,losses_list, cm,save_dir, fault_classes)
                plot_tsne_JAN(args, save_dir, JAN_model,  JAN_model_name)
        if args.use_model == 'MCD':
            
            source_acc, source_losses, losses_dis, losses_g = tarin_MCD(logger, feature_extractor_MCD, classification_extractor1, classification_extractor2, source_loader, target_loader, optimizer_g,
                      optimizer_f, criterion, device, epoch, lr_scheduler_f, lr_scheduler_g, n_step_B=4)
            val_acc, val_losses, all_preds, all_labels = test_MCD(epoch, feature_extractor_MCD, classification_extractor1, classification_extractor2, test_loader, device, criterion, logger)
            source_acc_list.append(round(source_acc, 4))
            source_losses_list.append(round(source_losses, 4))
            losses_dis_list.append(round(losses_dis, 4))
            losses_g_list.append(round(losses_g, 4))
            val_acc_list.append(round(val_acc, 4))
            val_losses_list.append(round(val_losses, 4))
            if args.save_model and epoch == args.epochs -1:
                feature_extractor_MCD_name = 'feature_extractor_MCD' + '_' + str(epoch + 1) + '.pth'
                torch.save(feature_extractor_MCD.state_dict(), os.path.join(save_dir, feature_extractor_MCD_name))
                classification_extractor1_name = 'classification_extractor1' + '_' + str(epoch + 1) + '.pth'
                torch.save(classification_extractor1.state_dict(), os.path.join(save_dir, classification_extractor1_name))
                classification_extractor2_name = 'classification_extractor2' + '_' + str(epoch + 1) + '.pth'
                torch.save(classification_extractor2.state_dict(), os.path.join(save_dir, classification_extractor2_name))
                val_acc, val_losses, all_preds, all_labels = test_MCD(epoch, feature_extractor_MCD,
                                                                      classification_extractor1,
                                                                      classification_extractor2, test_loader, device,
                                                                      criterion, logger)
                logger.info(f'真实类别:{set(all_labels)} 预测类别:{set(all_preds)} ')
                cm = confusion_matrix(all_labels, all_preds)
                np.savetxt(os.path.join(save_dir, 'confusion_matrix.txt'), cm, fmt='%d')
                try:
                    report = classification_report(all_labels, all_preds,target_names=fault_classes)
                    print(report)
                    end_time = time.time()
                    print(f'训练时间:{end_time - start_time :.4f}')
                except Exception as e:
                    print(f'错误原因：{str(e)}')
                    pass
                with open(args.save_acc_loss_dir, 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(
                        ['Epoch', 'source_acc', 'source_losses', 'losses_dis', 'losses_g', 'val_acc', 'val_losses']
                    )
                    for epoch in range(len(source_acc_list)):
                        writer.writerow([epoch + 1, source_acc_list[epoch], source_losses_list[epoch], losses_dis_list[epoch], losses_g_list[epoch], val_acc_list[epoch], val_losses_list[epoch]])
                plot_MCD(source_acc_list, source_losses_list, losses_dis_list, losses_g_list, val_acc_list, val_losses_list, cm, save_dir, fault_classes)
                plot_tsne_MCD(args, save_dir, feature_extractor_MCD, classification_extractor1, classification_extractor2,
                              feature_extractor_MCD_name, classification_extractor1_name, classification_extractor2_name)
        if args.use_model == 'my_method':
            
            source_train_acc,source_total_losses, class_losses, weighted_domain_losses = train(logger, source_loader, target_loader, feature_extractor, classification_extractor, domain_extractor,
                  criterion, domain_losses, optimizer, epoch, lr_scheduler, device, maximum_mean_discrepancy)
            test_acc, losses, all_labels, all_preds = test(logger, test_loader, feature_extractor, classification_extractor, domain_extractor, criterion, epoch,
                 device)
            source_train_acc_list.append(round(source_train_acc, 4))
            source_total_losses_list.append(round(source_total_losses, 4))
            class_losses_list.append(round(class_losses, 4))
            weighted_domain_losses_list.append(round(weighted_domain_losses, 4))
            test_acc_list.append(round(test_acc, 4))
            losses_list.append(round(losses, 4))
            if args.save_model and epoch == args.epochs -1:
                #logger_a.info(f'a的权重为：{a} 准确率为：{test_acc}')
                #logging.disable(logging.CRITICAL)
                feature_extractor_name = 'feature_extractor' + '_' + str(epoch + 1) + '.pth'
                torch.save(feature_extractor.state_dict(), os.path.join(save_dir, feature_extractor_name))
                classification_extractor_name = 'classification_extractor' + '_' + str(epoch + 1) + '.pth'
                torch.save(classification_extractor.state_dict(), os.path.join(save_dir, classification_extractor_name))
                logger.info(f'真实类别:{set(all_labels)} 预测类别:{set(all_preds)} ')
                cm = confusion_matrix(all_labels, all_preds)
                np.savetxt(os.path.join(save_dir, 'confusion_matrix.txt'), cm, fmt='%d')
                try:
                    report = classification_report(all_labels, all_preds, target_names=fault_classes)
                    print(report)
                    end_time = time.time()
                    print(f'训练时间:{end_time - start_time :.4f}')
                except Exception as e:
                    print(f'错误原因：{str(e)}')
                    pass
                with open(args.save_acc_loss_dir, 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(
                        ['Epoch', 'source_train_acc', 'source_total_losses', 'class_losses', 'weighted_domain_losses', 'test_acc', 'losses']
                    )
                    for epoch in range(len(source_train_acc_list)):
                        writer.writerow([epoch + 1, source_train_acc_list[epoch], source_total_losses_list[epoch], class_losses_list[epoch], weighted_domain_losses_list[epoch], test_acc_list[epoch], losses_list[epoch]])
                plot_my(source_train_acc_list, source_total_losses_list, class_losses_list, weighted_domain_losses_list, test_acc_list, losses_list, cm, save_dir, fault_classes)
                plot_tsne_my(args, save_dir, feature_extractor, classification_extractor, feature_extractor_name, classification_extractor_name)




if __name__ == '__main__':
    v2 = [ '100', '300', '500', '700', '900', '1200','1500'] #'100', '300', '500', '700', '900', '1200','100', '300', '500', '700', '900', '1200',
    fnn = ['0V', '1V', '2V', '3V', '4V']
    dic = ['DAN','DANN', 'JAN', 'MCD', 'my_method']  # 'CAT', 'DAN','DANN', 'JAN', 'MCD', 'my_method' #'cos', 'cos_entropy', 'cos_mmd','no_entropy',
    for model in dic:
    #for i in v1:
        for v in v2:
            for f1 in fnn:
                for f2 in fnn:
                    if f1 != f2:
                        source_path = \\source_data\\{v}\\{f1}'
                        target_path = \\target_data\\{v}\\{f2}'
                        test_path = \\val_data\\{v}\\{f2}'
                        if os.path.exists(source_path) and  os.path.exists(target_path) and  os.path.exists(test_path):
                            print(f'转速{v},负载{f1}向转速{v}负载{f2}迁移')
                            main(v, f1, f2, model)
                        else:
                            pass
    


