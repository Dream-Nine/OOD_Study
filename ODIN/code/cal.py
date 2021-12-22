# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import calMetric as m
import calData as d
#CUDA_DEVICE = 0

start = time.time()
#loading data sets

transform = transforms.Compose([
    transforms.ToTensor(),
    # `Normalize` is the class which normalize data channelwise by the given mean and std.
    # In this case, the number of input channels is 3 because the lenght of the iterable object is 3.
    # Ref: https://pytorch.org/vision/stable/transforms.html
    # Ref of various image normalizing: https://stackoverflow.com/questions/33610825/normalization-in-image-processing
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])




# loading neural network

# Name of neural networks
# Densenet trained on CIFAR-10:         densenet10
# Densenet trained on CIFAR-100:        densenet100
# Densenet trained on WideResNet-10:    wideresnet10
# Densenet trained on WideResNet-100:   wideresnet100
#nnName = "densenet10"

#imName = "Imagenet"



criterion = nn.CrossEntropyLoss()



def test(nnName, dataName, CUDA_DEVICE, epsilon, temperature):
    
    net1 = torch.load("../models/{}.pth".format(nnName))
    optimizer1 = optim.SGD(net1.parameters(), lr = 0, momentum = 0)
    net1.cuda(CUDA_DEVICE)
    
    # We suppose the out-of distribution data is `Uniform` or `Gaussian`. 
    # If you do not, we call the data as `testsetout`.
    if dataName != "Uniform" and dataName != "Gaussian":
        testsetout = torchvision.datasets.ImageFolder("/home/dreamnine/data/{}".format(dataName), transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1,
                                         shuffle=False, num_workers=2)
        
    # OOD 성능을 평가할 때는 고정되어야 하는 것들이 있다.
    # 적어도 같은 모델이어야 하고 같은 데이터이어야 한다(in dist & out-of dist 모두). (이것도 하이퍼 파라미터라고 볼 수 있을까?)
    # 그래야 OOD 방법론에 대한 정량적인 평가가 가능하기때문.
    if nnName == "densenet10" or nnName == "wideresnet10": 
        testset = torchvision.datasets.CIFAR10(root="/home/dreamnine/data", train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)
        
    if nnName == "densenet100" or nnName == "wideresnet100": 
        testset = torchvision.datasets.CIFAR100(root="/home/dreamnine/data", train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)
    
    if nnName == "densenet10_svhn":
        testset = torchvision.datasets.SVHN(root="/home/dreamnine/data", split='test', download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)
    
    if dataName == "Gaussian":
        d.testGaussian(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, nnName, dataName, epsilon, temperature)
        m.metric(nnName, dataName)

    elif dataName == "Uniform":
        d.testUni(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, nnName, dataName, epsilon, temperature)
        m.metric(nnName, dataName)
        
    else:
        d.testData(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderOut, nnName, dataName, epsilon, temperature) 
        m.metric(nnName, dataName)








