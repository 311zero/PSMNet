"""  to train a PSMNet on Scene Flow
python main.py --maxdisp 192 最大视差值
               --model stackhourglass 堆叠沙漏
               --datapath (your scene flow data folder) scene flow数据集路径
               --epochs 10 \
               --loadmodel (optional) 加载预训练模型
               --savemodel (path for saving model)
"""
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader
from models.stackhourglass import PSMNet_STACK
from models.basic import PSMNet_BASIC
from tqdm import tqdm
from utils.show import plot_hist

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='dataset/Scene_Flow_Datasets',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--trainbatch', type=int, default=8,
                    help='mini-batch to train')
parser.add_argument('--testbatch', type=int, default=6,
                    help='mini-batch to test or val')
parser.add_argument('--loadmodel', default=None,  # 'pretrained_model/pretrained_sceneflow.tar',
                    help='load model')
parser.add_argument('--savemodel', default='save_model',
                    help='path to save train and test model information')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if not os.path.isdir(args.savemodel):
    os.mkdir(args.savemodel)

# set gpu id used
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp, test_imgname = lt.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
    SecenFlowLoader.myImageFloder(all_left_img, all_right_img, all_left_disp, training=True),
    batch_size=args.trainbatch, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    SecenFlowLoader.myImageFloder(test_left_img, test_right_img, test_left_disp, training=False),
    batch_size=args.testbatch, shuffle=False, num_workers=4, drop_last=False)

if args.model == 'stackhourglass':
    model = PSMNet_STACK(args.maxdisp)
elif args.model == 'basic':
    model = PSMNet_BASIC(args.maxdisp)
else:
    print('no model')

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()



print('[+] Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
print(f'[+] Number of train_left_images: {len(TrainImgLoader)}')
print(f'[+] Number of val(test)_left_images: {len(TestImgLoader)}')


optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    mask = disp_true < args.maxdisp
    mask.detach_()

    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], reduction='mean') + \
               0.7 * F.smooth_l1_loss(output2[mask], disp_true[mask], reduction='mean') + \
               F.smooth_l1_loss(output3[mask], disp_true[mask], reduction='mean')
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], reduction='mean')

    loss.backward()
    optimizer.step()

    return loss.data.item()


def val(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    mask = disp_true < 192

    with torch.no_grad():
        output3 = model(imgL, imgR)

    output = torch.squeeze(output3.data.cpu(), 1)[:, 4:, :]

    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output[mask] - disp_true[mask]))  # end-point-error

    return loss.data.item()


def test():
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss = test(imgL, imgR, disp_L)
        print('Iter %d test loss = %.3f' % (batch_idx, test_loss))
        total_test_loss += test_loss

    print('total test loss = %.3f' % (total_test_loss / len(TestImgLoader)))

    # SAVE test information
    savefilename = args.savemodel + 'testinformation.tar'
    torch.save({
        'test_loss': total_test_loss / len(TestImgLoader),
    }, savefilename)


# def adjust_learning_rate(optimizer, epoch, Lr):
# if patience == 2:
#     patience = 0
#     # 加载模型参数
#     model.load_state_dict(torch.load('best_val_weight.pth'))
#     lr = lr / 10
#     print(f'[+] set lr={lr}')
# if epoch == 1:
#     lr = 0.002
#     print(f'[+] set lr={lr}')
#     # 定义优化方法,方法=Adam,网络参数=model.fresh_params(),学习率=lr
#     optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
# else:
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0002)
# if epoch == 1:
#     lr = 0.002
#     optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
# else:
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0002)
# for param_group in optimizer.param_groups:
#     param_group['lr'] = lr


def main():
    start_full_time = time.time()
    patience = 0
    min_loss = float('inf')
    lr = 0
    hist = {}
    hist['train_loss'] = []
    hist['val_loss'] = []
    for epoch in range(1, args.epochs + 1):
        print('[+] This is %d-th epoch' % (epoch))
        total_train_loss = 0
        # adjust_learning_rate(optimizer, epoch, patience, lr)

        # ---------------------- training ---------------------
        pbar = tqdm(TrainImgLoader, total=len(TrainImgLoader))
        batch_idx = 0
        for imgL_crop, imgR_crop, disp_crop_L, imgname in pbar:
            start_time = time.time()
            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            # pbar.set_description(f'    Iter {batch_idx} training loss = {loss:.3f} , '
            #                      f'time = {time.time() - start_time:.2f}')
            print('\t epoch_%d Iter %d training loss = %.3f , time = %.2f'
                  % (epoch, batch_idx, loss, time.time() - start_time))
            batch_idx = batch_idx + 1
            total_train_loss += loss
        mean_loss = total_train_loss / len(TrainImgLoader)
        hist['train_loss'].append(mean_loss)
        print('epoch %d total training loss = %.3f' % (epoch, mean_loss))

        # SAVE
        savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename)

        #   -------------------- validation ---------------------------
        pbar = tqdm(TestImgLoader, total=len(TestImgLoader))
        batch_idx = 0
        epoch_total_val_loss = 0
        for imgL, imgR, disp_L, imgname in pbar:
            val_loss = val(imgL, imgR, disp_L)
            print('\t   val epoch_%d Iter %d val loss = %.3f' % (epoch, batch_idx, val_loss))
            batch_idx += 1
            epoch_total_val_loss += val_loss
        hist['val_loss'].append(epoch_total_val_loss)
        epoch_mean_val_loss = epoch_total_val_loss / len(TestImgLoader)
        print('val epoch %d mean val loss = %.3f' % (epoch, epoch_mean_val_loss))
        histpath = './'  # 当前目录下
        plot_hist(hist, histpath)
    print('full training and val time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    """
        if epoch_mean_val_loss < min_loss:
            savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss/len(TrainImgLoader),
            }, savefilename)
            print(f'[+] val score improved from {min_loss:.5f} to {epoch_mean_val_loss:.5f}. Saved!')
            print(f'and saved in {savefilename}')
            patience = 0
        else:
            patience += 1
    """






if __name__ == '__main__':
    main()