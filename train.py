'''Train CIFAR10 with PyTorch.'''
import torch
import random
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.nn import init

import os
import argparse

from models import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')  #optional arguments
parser.add_argument('--layer_num',default=101,type=int,help='the number of resnet')
parser.add_argument('--rank',type=int,help='the id of net')
args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def weigth_init(m):
    #权重 正态分布(0,1)  偏置   全0
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight,a=0,mode='fan_in', nonlinearity='relu') 
        if m.bias is not None:
            init.constant_(m.bias,0)
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), #先扩充到36*36随机裁剪32*32
    transforms.RandomHorizontalFlip(), #以p概率水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='/bdata/gg/cifar10/pytorch-cifar-master/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='/bdata/gg/cifar10/pytorch-cifar-master/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
it = random.randint(1,4)

'''
net = torch.nn.DataParallel(ResNet101(),device_ids=[0,1,2,3])
net = net.cuda()
'''
#保证同时训练
if args.layer_num == 18:
    net = ResNet18()
    if args.rank == 0:
        it = 2
    if args.rank == 1:
        it = 3
if args.layer_num == 34:
    net = ResNet34()
    if args.rank == 0:
        it = 0
    if args.rank == 1:
        it = 1
if args.layer_num == 50:
    net = ResNet50()
    if args.rank == 0:
        it = 0
    if args.rank == 1:
        it = 1
if args.layer_num == 101:
    net = ResNet101()
    if args.rank == 0:
        it = 0
    if args.rank == 1:
        it = 1
if args.layer_num == 152:
    net = ResNet152()
    if args.rank == 0:
        it = 2
    if args.rank == 1:
        it = 3
device = torch.device('cuda:{}'.format(it))
print(device)
net.apply(weigth_init)
net = net.to(device)
criterion = nn.CrossEntropyLoss() # classify 损失函数
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
#Momentum算法借用了物理中的动量概念，它模拟的是物体运动时的惯性，即更新的时候在一定程度上保留之前更新的方向，同时利用当前batch的梯度微调最终的更新方向
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#PyTorch学习率调整策略通过torch.optim.lr_scheduler接口实现:有序调整：余弦退火CosineAnnealing

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0) #训练集总数
        correct += predicted.eq(targets).sum().item()#训练集正确总数
    print("train_loss: {} ,acc: {}%".format(train_loss,100.*correct/total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc :#and epoch>150:#节省时间
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, '../../src/model_data/he_normal_resnet{}_{}.pth'.format(args.layer_num,args.rank))
        best_acc = acc

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
    #print(optimizer.param_groups[0])
    print(optimizer.param_groups[0]['lr'])
