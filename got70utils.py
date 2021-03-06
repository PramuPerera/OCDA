import os
import cv2
import copy
import sys
import time
import torch
import random
import scipy.io
from dataset_file import *
from converter import *
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import torch.functional as func
import torchvision.datasets as dset
import torch.nn.functional as nnfunc
import torchvision
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import torchvision.models as models
import scipy.spatial.distance as spd
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from sklearn import metrics
from torch.autograd import Variable
from itertools import cycle, islice
from sklearn import cluster, datasets
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
sys.path.append('../models/')
from models import *
from evt_utils import *
from sklearn.svm import OneClassSVM
import numpy as np
import pickle as cPickle
import models
import wrn
import pdb
from utils2 import load_cifar10, load_cats_vs_dogs, load_fashion_mnist, load_cifar100
from utils2 import save_roc_pr_curve_data, get_class_name_from_index, get_channels_axis
from transformations import Transformer

class RotateDataset(torch.utils.data.Dataset):
    def __init__(self, dset, transform=None):
        self.dset = dset
        self.transform = transform

    def __getitem__(self, index):
        rot_id = np.random.randint(4, size=1)[0]
        x_id = np.random.randint(3, size=1)[0]
        y_id = np.random.randint(3, size=1)[0]
        flip_id = np.random.randint(2, size=1)[0]
        dset = RotateDatasetDef(self.dset, rot_id, x_id, y_id, flip_id)
        return(dset[index][0], rot_id+4*x_id+12*y_id+36*flip_id)
        '''tf_rot = [torchvision.transforms.RandomRotation((0,0)), torchvision.transforms.RandomRotation((90,90)), torchvision.transforms.RandomRotation((180,180)), torchvision.transforms.RandomRotation((270,270))]
        tf_xid = [torchvision.transforms.RandomAffine((0,0), translate=(0.75,0)), torchvision.transforms.RandomAffine((0,0), translate=(0,0)), torchvision.transforms.RandomAffine((0,0), translate=(0.25,0))]
        tf_yid = [torchvision.transforms.RandomAffine((0, 0), translate=(0, 0.75)),
                  torchvision.transforms.RandomAffine((0, 0), translate=(0, 0)),
                  torchvision.transforms.RandomAffine((0, 0), translate=(0, 0.25))]
        tf_flip = [torchvision.transforms.RandomHorizontalFlip(p=1.0), torchvision.transforms.RandomHorizontalFlip(p=0.0)]
        tf = transforms.Compose([torchvision.transforms.ToPILImage(), tf_flip[flip_id ]])
        return torchvision.transforms.functional.to_tensor(torchvision.transforms.functional.affine(tf(self.dset[index][0]), 90*rot_id, (4*x_id, 4*y_id), 1.0, 0)), rot_id+4*x_id+12*y_id+36*flip_id'''

    def __len__(self):
        return len(self.dset)

class RotateDatasetDef(torch.utils.data.Dataset):
    def __init__(self, dset, rot, x, y, flip, transform=None):
        self.dset = dset
        self.rot = rot
        self.x = x
        self.y = y
        self.flip = flip
        self.transform = transform


    def __getitem__(self, index):
        rot_id = self.rot
        x_id = self.x
        y_id = self.y
        flip_id = self.flip
        tf_rot = [torchvision.transforms.RandomRotation((0,0)), torchvision.transforms.RandomRotation((90,90)), torchvision.transforms.RandomRotation((180,180)), torchvision.transforms.RandomRotation((270,270))]
        tf_xid = [torchvision.transforms.RandomAffine((0,0), translate=(0.75,0)), torchvision.transforms.RandomAffine((0,0), translate=(0,0)), torchvision.transforms.RandomAffine((0,0), translate=(0.25,0))]
        tf_yid = [torchvision.transforms.RandomAffine((0, 0), translate=(0, 0.75)),
                  torchvision.transforms.RandomAffine((0, 0), translate=(0, 0)),
                  torchvision.transforms.RandomAffine((0, 0), translate=(0, 0.25))]
        tf_flip = [torchvision.transforms.RandomHorizontalFlip(p=1.0), torchvision.transforms.RandomHorizontalFlip(p=0.0)]
        tf = transforms.Compose([torchvision.transforms.ToPILImage(), tf_flip[flip_id ]])
        return torchvision.transforms.functional.to_tensor(torchvision.transforms.functional.affine(tf(self.dset[index][0]), 90*rot_id, (4*x_id, 4*y_id), 1.0, 0)), self.dset[index][1]

    def __len__(self):
        return len(self.dset)


class RotateDatasetOld(torch.utils.data.Dataset):
    def __init__(self, dset, transform=None):
        self.dset = dset
        self.transform = transform

    def __getitem__(self, index):
        rot_id = np.random.randint(4, size=1)[0]
        x_id = np.random.randint(3, size=1)[0]
        y_id = np.random.randint(3, size=1)[0]
        flip_id = np.random.randint(2, size=1)[0]
        tf_rot = [torchvision.transforms.RandomRotation((0,0)), torchvision.transforms.RandomRotation((90,90)), torchvision.transforms.RandomRotation((180,180)), torchvision.transforms.RandomRotation((270,270))]
        tf_xid = [torchvision.transforms.RandomAffine((0,0), translate=(0.75,0)), torchvision.transforms.RandomAffine((0,0), translate=(0,0)), torchvision.transforms.RandomAffine((0,0), translate=(0.25,0))]
        tf_yid = [torchvision.transforms.RandomAffine((0, 0), translate=(0, 0.75)),
                  torchvision.transforms.RandomAffine((0, 0), translate=(0, 0)),
                  torchvision.transforms.RandomAffine((0, 0), translate=(0, 0.25))]
        tf_flip = [torchvision.transforms.RandomHorizontalFlip(p=1.0), torchvision.transforms.RandomHorizontalFlip(p=0.0)]
        if self.transform is not None:
            tf = transforms.Compose([torchvision.transforms.ToPILImage(), self.transform, tf_flip[flip_id ], tf_xid[x_id], tf_yid[y_id], tf_rot[rot_id] ,torchvision.transforms.ToTensor()])
        else:
            tf = transforms.Compose([torchvision.transforms.ToPILImage(), tf_flip[flip_id ], tf_xid[x_id], tf_yid[y_id], tf_rot[rot_id] ,torchvision.transforms.ToTensor()])
        return tf(self.dset[index][0]), rot_id+4*x_id+12*y_id+36*flip_id

    def __len__(self):
        return len(self.dset)

class ECCVFile(Dataset):
    def __init__(self, filename, tf, fold='train'):
        self.fold = fold
        self.dset = DatasetFile(filename, example_count=None)
        self.tf = tf
        self.img_conv = ImageConverter(self.dset)
        self.lab_conv = LabelConverter(self.dset)
        self.batcher = self.dset.get_all_batches(fold=fold)

    def getLabels(self):
        return self.lab_conv.getLabels()

    def __len__(self):
        return len(self.batcher)

    def __getitem__(self, idx):
        images = self.img_conv.to_array(self.batcher[idx])
        labels = self.lab_conv.to_array(self.batcher[idx])

        return self.tf(images), labels


class MyClass(object):
    class_var = 1

    def __init__(self):
        self.i_var = 1


aa = MyClass()

cfg = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


from torch.utils.data import Dataset, TensorDataset

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)




def octrain(hyper_para):
    for single_class_ind in range(10):
        hyper_para.C = wrn.WideResNet(28, num_classes=72, dropout_rate=0, widen_factor=10)
        (x_train, y_train), (x_test, y_test) = load_cifar10()
        transformer = Transformer(8, 8)
        x_train_task = x_train[y_train.flatten() == single_class_ind]
        transformations_inds = np.tile(np.arange(transformer.n_transforms), len(x_train_task))
        x_train_task_transformed = transformer.transform_batch(np.repeat(x_train_task, transformer.n_transforms, axis=0),
                                                       transformations_inds)
        dset = CustomDataset(x_train_task_transformed,transformations_inds )
        trainLoader = DataLoader(dset, batch_size=32, shuffle=True)
        # define networks
        C = hyper_para.C
        # define loss functions
        ce_criterion = nn.CrossEntropyLoss()

        # define optimizer
        optimizer_c = optim.Adam(C.parameters(), lr=hyper_para.lr, betas=(0.9, 0.999))

        # turn on the train mode
        C.train(mode=True)

        # initialization of auxilary variables
        running_tl = 0.0
        running_cc = 0.0
        running_rc = 0.0
        running_ri = 0.0

        # if gpu use cuda


        for i in [1]:
            if hyper_para.gpu:
                C.cuda()
            for idx, (inputs, labels) in enumerate(trainLoader):
                inputs = inputs.permute(0,3,2,1)
                t1 = time.time()
                if hyper_para.gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                act,_ = C(inputs)
                loss_cc = ce_criterion(act, labels)
                optimizer_c.zero_grad()
                loss_cc.backward()
                optimizer_c.step()
                running_cc += loss_cc.data
                t2 = time.time()
                if idx%10==0:
                    line = hyper_para.BLUE + '[' + str(format(i + 1, '8d')) + '/' + str(
                    format(int(hyper_para.iterations), '8d')) + ']' + hyper_para.ENDC + \
                       hyper_para.GREEN + ' loss_cc: ' + hyper_para.ENDC + str(
                    format(running_cc / hyper_para.stats_frequency, '1.8f')) + \
                       hyper_para.YELLOW + ' time (min): ' + hyper_para.ENDC + str(int((t2 - t1) * 20.0))
                    print(line)
                running_cc = 0.0

        torch.save(C.state_dict(), hyper_para.experiment_name +'_'+str(single_class_ind)+ '.pth')


def os_train(hyper_para):
    C = wrn.WideResNet(28, num_classes=10, dropout_rate=0, widen_factor=10).cuda()
    single_class_ind = hyper_para.inclass[0]
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    transformer = Transformer(8, 8)
    x_train_task = x_train[[i in hyper_para.inclass for i in y_train.flatten()]]
    y_train_task = y_train[[i in hyper_para.inclass for i in y_train.flatten()]]
    print(np.shape(y_train_task))
    print(np.shape(y_train))
    y_train_task = y_train_task.astype(int)
    #x_train_task =     x_train_task[0:15][:][:][:] 

    dset = CustomDataset(x_train_task, y_train_task )
    trainLoader = DataLoader(dset, batch_size=128, shuffle=True)
    # define networks
    # define loss functions
    ce_criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer_c = optim.Adam(C.parameters(), lr=hyper_para.lr, betas=(0.9, 0.999))

    # turn on the train mode
    C.train(mode=True)

    # initialization of auxilary variables
    running_tl = 0.0
    running_cc = 0.0
    running_rc = 0.0
    running_ri = 0.0

    # if gpu use cuda


    for i in range(int(20)):
        if hyper_para.gpu:
            C.cuda()
        for iii, (inputs, labels) in enumerate(trainLoader):
            inputs = inputs.permute(0,3,2,1)
            t1 = time.time()
            if hyper_para.gpu:
                inputs = inputs.cuda()
                labels = labels[:,0].cuda()
            act,_ = C(inputs)
            loss_cc = ce_criterion(act, labels)
            optimizer_c.zero_grad()
            loss_cc.backward()
            optimizer_c.step()
            running_cc += loss_cc.data
            t2 = time.time()
            if iii%50 == 0:
                    line = hyper_para.BLUE + '[' + str(format(i + 1, '8d')) + '/' + str(
                        format(int(hyper_para.iterations), '8d')) + ']' + hyper_para.ENDC + \
                           hyper_para.GREEN + ' loss_cc: ' + hyper_para.ENDC + str(
                        format(running_cc / hyper_para.stats_frequency, '1.8f')) + \
                           hyper_para.YELLOW + ' time (min): ' + hyper_para.ENDC + str(int((t2 - t1) * 20.0))
                    print(line)
                    running_cc = 0.0

        torch.save(C.state_dict(), hyper_para.experiment_name + '.pth')


def os_test( hyper_para):
    C = wrn.WideResNet(28, num_classes=10, dropout_rate=0, widen_factor=10).cuda()
    C.load_state_dict(torch.load(hyper_para.experiment_name + '.pth'))
    single_class_ind = hyper_para.inclass[0]
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    C.eval()
    score = []
    lbl = []
    features = []
    correct = 0
    total = 0
    dset = CustomDataset(x_test, y_test )
    testLoader = DataLoader(dset, batch_size=128)
    for i, (inputs, labels) in enumerate(testLoader):
            inputs = inputs.permute(0,3,2,1)
            if hyper_para.gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            act, f =  C(inputs)
            features+= f.detach().cpu().tolist()
            val , ind = torch.max(act,dim=1)
            score += val.detach().cpu().tolist()
            val , ind, labels = (val.detach().cpu().tolist(),ind.detach().cpu().tolist(),labels.detach().cpu().tolist())
            for ii,gt in zip(ind,labels):
                    gt = gt[0]
                    if gt in hyper_para.inclass:
                        total+=1
                        if ii == gt:
                            correct+=1
                        lbl.append(1)
                    else:
                        lbl.append(0)
    fpr, tpr, thresholds = metrics.roc_curve(lbl,score)
    AUC = metrics.auc(fpr, tpr)
    ACC = float(correct)/total
    print('AUROC: ' + str(AUC) + '\t Accuracy: ' + str(ACC))
    #return (ACC, AUC)



def osoc_test( hyper_para):
    transformer = Transformer(8, 8)
    C = wrn.WideResNet(28, num_classes=10, dropout_rate=0, widen_factor=10).cuda()
    C.load_state_dict(torch.load(hyper_para.experiment_name + '.pth'))
    single_class_ind = hyper_para.inclass[0]
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    C.eval()
    score = []
    lbl = []
    features = []
    correct = 0
    total = 0
    dset = CustomDataset(x_test, y_test )
    testLoader = DataLoader(dset, batch_size=128)
    for i, (inputs0, labels) in enumerate(testLoader):
            inputs = inputs0.permute(0,3,2,1)
            if hyper_para.gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            act, f =  C(inputs)
            features+= f.detach().cpu().tolist()
            val , ind = torch.max(act,dim=1)
            #score += val.detach().cpu().tolist()
            CC = wrn.WideResNet(28, num_classes=72, dropout_rate=0, widen_factor=10).cuda()
            val , ind, labels = (val.detach().cpu().tolist(),ind.detach().cpu().tolist(),labels.detach().cpu().tolist())
            for ind,(ii,gt) in enumerate(zip(ind,labels)):
                    gt = gt[0]
                    CC.load_state_dict(torch.load('saved/'+hyper_para.experiment_name +'_'+str(gt)+ '.pth'))

                    x_test0 = transformer.transform_batch(np.expand_dims(inputs0[ind,:,:,:].detach().cpu().numpy(),0), [0] )
                    x_test0 = torch.tensor(x_test0).permute(0,3,2,1).cuda()	
                    act, f =  CC(x_test0)
                    act = act.detach().cpu().tolist()
                    score+= [act[0][0]]			

                    if gt in hyper_para.inclass:
                        total+=1
                        if ii == gt:
                            correct+=1
                        lbl.append(1)
                    else:
                        lbl.append(0)
            break
    fpr, tpr, thresholds = metrics.roc_curve(lbl,score)
    AUC = metrics.auc(fpr, tpr)
    ACC = float(correct)/total
    print('AUROC: ' + str(AUC) + '\t Accuracy: ' + str(ACC))
    #return (ACC, AUC)


def octest(hyper_para):
    CC = wrn.WideResNet(28, num_classes=10, dropout_rate=0, widen_factor=10).cuda()
    CC.load_state_dict(torch.load(hyper_para.experiment_name + '.pth'))
    C = wrn.WideResNet(28, num_classes=72, dropout_rate=0, widen_factor=10).cuda()
    C.cuda().eval()
    CC.cuda().eval()

    muc = {}
    stdc = {}
    mu = {}
    std = {}
    for cname in hyper_para.inclass:
         single_class_ind = cname
         C.load_state_dict(torch.load('saved/'+hyper_para.experiment_name +'_'+str(cname)+ '.pth'))
         (x_train, y_train), (x_test, y_test) = load_cifar10()
         transformer = Transformer(8, 8)
         x_train_task = x_train[[i in hyper_para.inclass for i in y_train.flatten()]]
         y_train_task = y_train[[i in hyper_para.inclass for i in y_train.flatten()]]
         #x_train_task = transformer.transform_batch(torch.unsqueeze(x_train_task,0).detach().cpu().numpy(), [0]*len(y_train_task))




         dset = CustomDataset(x_train_task, y_train_task )
         trainLoader = DataLoader(dset, batch_size=128, shuffle=True)
         features = []
         for  inputs0, labels in trainLoader:
                   inputs = inputs0.permute(0,3,2,1).cuda()
                   act, f =  C(inputs)
                   features+= act[:,0].detach().cpu().tolist()
         mu[str(cname)] = np.mean(features)
         std[str(cname)]= np.sqrt(np.var(features))





    features = {}

    if True:
         (x_train, y_train), (x_test, y_test) = load_cifar10()
         x_train_task = x_train[[i in hyper_para.inclass for i in y_train.flatten()]]
         y_train_task = y_train[[i in hyper_para.inclass for i in y_train.flatten()]]
         dset = CustomDataset(x_train_task, y_train_task )
         trainLoader = DataLoader(dset, batch_size=128, shuffle=True)
         for  inputs0, labels0 in trainLoader:
                   inputs = inputs0.permute(0,3,2,1).cuda()
                   act, f =  CC(inputs)
                   val , ind = torch.max(act,dim=1)
                   val , ind, labels0 = (val.detach().cpu().tolist(),ind.detach().cpu().tolist(),labels0.detach().cpu().tolist())
                   for idx, (ii,gt) in enumerate(zip(ind,labels0)):
                        gt = gt[0]
                        if ii==gt and gt in hyper_para.inclass:
                              if str(ii) not in features.keys():
                                   features[str(ii)]= [act[idx,ii].detach().cpu().tolist()]                             
                              else:
                                   features[str(ii)]+= [act[idx,ii].detach().cpu().tolist()  ]
                              print(np.shape(features[str(ii)]))                   
         for k in features.keys():
                   muc[str(k)] = np.mean(features[str(k)])
                   stdc[str(k)]= np.sqrt(np.var(features[str(k)]))





    (x_train, y_train), (x_test, y_test) = load_cifar10()
    transformer = Transformer(8, 8)

    scores = np.array([[]])
    features = []
    lbl  = []
    correct = 0
    total = 0
    preds = np.zeros((len(x_test), transformer.n_transforms))
    dset = CustomDataset(x_test, y_test )
    testLoader = DataLoader(dset, batch_size=128)
    for t in [0]:#range(72):
         score = []
         for i, (inputs0, labels) in enumerate(testLoader):	
                   inputs = inputs0.permute(0,3,2,1)		
                   if True:
                                inputs = inputs.cuda()
                                labels = labels.cuda()




                   act0, f =  CC(inputs)
                   features+= f.detach().cpu().tolist()
                   val , ind = torch.max(act0,dim=1)
                   val , ind, labels = (val.detach().cpu().tolist(),ind.detach().cpu().tolist(),labels.detach().cpu().tolist())
                   #act = torch.nn.functional.softmax(act, dim=1)

                   #score += val

                   for idx, (ii,gt) in enumerate(zip(ind,labels)):

                    C.load_state_dict(torch.load('saved/'+hyper_para.experiment_name +'_'+str(ii)+ '.pth'))
                    gt = gt[0]
                    x_test0 = transformer.transform_batch(torch.unsqueeze(inputs0[idx,:,:,:],0).detach().cpu().numpy(), [0])
                    inputs = torch.tensor(x_test0).permute(0,3,2,1).cuda()
                    act,_ = C(inputs)
                    act = act[:,0]
                    act = act.detach().cpu().tolist()
                    #score+= [val[idx]]
                    score+= [(act[0]-mu[str(ii)])/(std[str(ii)])]#  + (val[idx]-muc[str(ii)])/(stdc[str(ii)])]


                    if gt in hyper_para.inclass:
                        total+=1
                        if ii == gt:
                            correct+=1
                        lbl.append(1)
                    else:
                        lbl.append(0)	                 	

                   

    fpr, tpr, thresholds = metrics.roc_curve(lbl, score)
    AUC = metrics.auc(fpr, tpr)
    print('AUROC: ' + str(AUC) )
    return([0,0])



def octest0(hyper_para):
    C = wrn.WideResNet(28, num_classes=72, dropout_rate=0, widen_factor=10).cuda()
    C.load_state_dict(torch.load('saved/'+hyper_para.experiment_name +'_'+str(hyper_para.inclass[0])+ '.pth'))
    single_class_ind = hyper_para.inclass[0]
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    transformer = Transformer(8, 8)
    glabels = y_test.flatten() == single_class_ind
    C.cuda().eval()
    scores = np.array([[]])
    features = []

    correct = 0
    total = 0
    preds = np.zeros((len(x_test), transformer.n_transforms))
    for t in [0]:#range(72):
         score = []
         x_test0 = transformer.transform_batch(x_test, [t] * len(x_test))
         dset = CustomDataset(x_test0, [t] * len(x_test) )
         testLoader = DataLoader(dset, batch_size=128, shuffle=False)
         for i, (inputs, labels) in enumerate(testLoader):	
                   inputs = inputs.permute(0,3,2,1)		
                   if True:
                                inputs = inputs.cuda()
                                labels = labels.cuda()
                   act, f =  C(inputs)
                   features+= f.detach().cpu().tolist()
                   #act = torch.nn.functional.softmax(act, dim=1)
                   score+=  act[:,t].detach().cpu().tolist() 
         preds[:,t] = list(score) 
         fpr, tpr, thresholds = metrics.roc_curve(glabels, score)
         AUC = metrics.auc(fpr, tpr)
         print('AUROC: ' + str(AUC) )

    scores = np.sum(((preds)), 1)     
    fpr, tpr, thresholds = metrics.roc_curve(glabels, scores)
    AUC = metrics.auc(fpr, tpr)
    print('AUROC: ' + str(AUC) )
    return([0,0])




def os_test_ens(testLoader, hyper_para, C, isauc):
    C = wrn.WideResNet(28, num_classes=72, dropout_rate=0, widen_factor=10).cuda()
    C.load_state_dict(torch.load('saved/'+hyper_para.experiment_name +'_'+str(hyper_para.inclass[0])+ '.pth'))
    single_class_ind = hyper_para.inclass[0]
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    transformer = Transformer(8, 8)
    glabels = y_test.flatten() == single_class_ind
    C.cuda().eval()
    scores = np.array([[]])
    features = []

    correct = 0
    total = 0
    preds = np.zeros((len(x_test), transformer.n_transforms))
    for t in [0]:#range(72):
         score = []
         x_test0 = transformer.transform_batch(x_test, [t] * len(x_test))
         dset = CustomDataset(x_test0, [t] * len(x_test) )
         testLoader = DataLoader(dset, batch_size=128, shuffle=False)
         for i, (inputs, labels) in enumerate(testLoader):	
                   inputs = inputs.permute(0,3,2,1)		
                   if True:
                                inputs = inputs.cuda()
                                labels = labels.cuda()
                   act, f =  C(inputs)
                   features+= f.detach().cpu().tolist()
                   #act = torch.nn.functional.softmax(act, dim=1)
                   score+=  act[:,t].detach().cpu().tolist() 
         preds[:,t] = list(score) 
         fpr, tpr, thresholds = metrics.roc_curve(glabels, score)
         AUC = metrics.auc(fpr, tpr)
         print('AUROC: ' + str(AUC) )

    scores = np.sum(((preds)), 1)     
    fpr, tpr, thresholds = metrics.roc_curve(glabels, scores)
    AUC = metrics.auc(fpr, tpr)
    print('AUROC: ' + str(AUC) )
    return([0,0])
