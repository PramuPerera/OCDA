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
from utils2 import *
from utils2 import save_roc_pr_curve_data, get_class_name_from_index, get_channels_axis
from transformations import Transformer


from torch.utils.data import Dataset, TensorDataset

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


class RotateDataset(Dataset):
    def __init__(self, x_tensor, lbl):
        self.x = x_tensor
        self.y = lbl
        self.perm = np.random.permutation(len(x_tensor))
        self.transformer = Transformer(8, 8)

    def __getitem__(self, index):
        index = self.perm[index]
        trans_id = np.random.randint(self.transformer.n_transforms, size=1)[0]
        #return(self.transformer.transform_sample(self.x[index], trans_id), trans_id)
        return((self.transformer.transform_batch(np.expand_dims(self.x[index],0), [trans_id]))[0], trans_id, self.y[index])
    def __len__(self):
        return len(self.x)



class FCDiscriminator(nn.Module):
    def __init__(self):
        super(FCDiscriminator, self).__init__()

        self.net = [
            GradientReversal(),
            nn.Linear(640,50),
            nn.ReLU(),
            nn.Linear(50,20),
            nn.ReLU(),
            nn.Linear(20,1),]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)




def octrain(hyper_para):
    for single_class_ind in hyper_para.inclass:
        hyper_para.C = wrn.WideResNet(28, num_classes=72, dropout_rate=0, widen_factor=10)
        if hyper_para.source == 'mnist':
               (x_train, y_train), (x_test, y_test) = load_mnist()
               (x_train2, y_train2), (x_test2, y_test2) = load_svhn()
        elif hyper_para.source == 'svhn':
               (x_train, y_train), (x_test, y_test) = load_svhn()
               (x_train2, y_train2), (x_test2, y_test2) = load_mnist()
        elif hyper_para.source == 'amazon':
               (x_train, y_train), (x_test, y_test) = load_amazon()
               (x_train2, y_train2), (x_test2, y_test2) = load_dslr()
        elif hyper_para.source == 'dslr':
               (x_train, y_train), (x_test, y_test) = load_dslr()
               (x_train2, y_train2), (x_test2, y_test2) = load_amazon()
        if hyper_para.method == 'justsource':
            x_train_task = x_train[y_train.flatten() == single_class_ind]

            x_test = x_train_task[int(len(x_train_task)*0.8):]
            x_train_task = x_train_task[0:int(len(x_train_task)*0.8)]
            domain_lbl = [0]*len(x_train_task)
            tst_lbl = [0]*len(x_test)
            dset2 = RotateDataset(x_test, tst_lbl )
            testLoader = DataLoader(dset2, batch_size=16, shuffle=True)

        else:#if hyper_para.method == 'balancedsourcetarget':
               x_train_task = x_train[y_train.flatten() == single_class_ind]
               x_train_task2 = x_train2[y_train2.flatten() == single_class_ind]
               x_train_task2 = x_train_task2[0: hyper_para.target_n]
               #x_train_task = x_train_task[0: 20]#hyper_para.target_n]
               x_test = x_train_task[int(len(x_train_task)*0.8):]
               x_train_task = x_train_task[0:int(len(x_train_task)*0.8)]
               x_test2 = x_train_task2[int(len(x_train_task2)*0.8):]
               x_train_task2 = x_train_task2[0:int(len(x_train_task2)*0.8)]

               domain_lbl = [0]*len(x_train_task) + [1]*(len(x_train_task2)*int(len(x_train_task)/len(x_train_task2)))
               x_train_task2 = np.tile(x_train_task2, (int(len(x_train_task)/len(x_train_task2)),1,1,1) )
               x_train_task = np.concatenate((x_train_task, x_train_task2))
               tst_lbl = [0]*len(x_test) + [1]*(len(x_test2)*int(len(x_test)/len(x_test2)))
               x_test2 = np.tile(x_test2, (int(len(x_test)/len(x_test2)),1,1,1) )
               x_test = np.concatenate((x_test, x_test2))


               dset2 = RotateDataset(x_test, tst_lbl )
               testLoader = DataLoader(dset2, batch_size=16, shuffle=True)
               transformer = Transformer(8, 8)

        dset = RotateDataset(x_train_task, domain_lbl )
        trainLoader = DataLoader(dset, batch_size=512, shuffle=True)





        '''transformations_inds = np.tile(np.arange(transformer.n_transforms), len(x_train_task))
        x_train_task_transformed = transformer.transform_batch(np.repeat(x_train_task, transformer.n_transforms, axis=0),
                                                       transformations_inds)
        dset = CustomDataset(x_train_task_transformed,transformations_inds )
        trainLoader = DataLoader(dset, batch_size=32, shuffle=True)'''

        if hyper_para.method == 'dd':
            Dnet = FCDiscriminator().cuda()
            optimizer_D = optim.Adam(Dnet.parameters(), lr=hyper_para.lr, betas=(0.9, 0.999))
            Dnet.train()
        # define networks
        C = hyper_para.C
        C = torch.nn.DataParallel(C)
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

        best_acc = 0
        for i in range(500):#int(100*hyper_para.source_n/len(x_train_task))):
            acct = 0
            nelt = 0
            acc0t = 0
            nel0t = 0
            acc1t = 0
            nel1t = 0
            if hyper_para.gpu:
                C.cuda()
            for idx, (inputs, labels, dlbls) in enumerate(trainLoader):
                inputs = inputs.permute(0,3,2,1)
                t1 = time.time()
                if hyper_para.gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    dlbls = dlbls.cuda()
                act,fea = C(inputs)
                _, ind = torch.max(act,1)

                if hyper_para.method == 'dd':
                    # Train Discriminator
                    Dnet.train()
                    d = Dnet(fea)
                    _, mind = torch.max(d,1)
                    acc = torch.mean(torch.eq(mind,dlbls).cpu().float())
                    optimizer_D.zero_grad()
                    optimizer_c.zero_grad()
                    loss_d = nnfunc.binary_cross_entropy_with_logits(d.squeeze(), dlbls.float())  
                    loss_cc = ce_criterion(act, labels)
                    loss = loss_cc + loss_d
                    loss.backward()
                    optimizer_c.step()
                    optimizer_D.step()
                else:
                       loss_cc = ce_criterion(act, labels)
                       optimizer_c.zero_grad()
                       loss = loss_cc 
                       loss.backward()
                       optimizer_c.step()
                running_cc += loss_cc.data
                t2 = time.time()
                acct += torch.sum(torch.eq(ind, labels)).cpu().numpy()
                nelt += ind.shape[0]
                acc0t += torch.sum(torch.eq(ind[dlbls==0], labels[dlbls==0])).cpu().numpy()
                nel0t += (dlbls==0).shape[0]
                acc1t += torch.sum(torch.eq(ind[dlbls==1], labels[dlbls==1])).cpu().numpy()
                nel1t += (dlbls==1).shape[0]
                if idx%10==0:
                    line = hyper_para.BLUE + '[' + str(format(i + 1, '8d')) + '/' + str(
                    format(int(hyper_para.iterations), '8d')) + ']' + hyper_para.ENDC + \
                       hyper_para.GREEN + ' loss_cc: ' + hyper_para.ENDC + str(
                    format(running_cc / hyper_para.stats_frequency, '1.8f')) + \
                       hyper_para.YELLOW + ' time (min): ' + hyper_para.ENDC + str(int((t2 - t1) * 20.0))
                    print(line)
                    acc = 0
                    nel = 0
                    acc0 = 0
                    nel0 = 0
                    acc1 = 0
                    nel1 = 0
                    for idx, (inputs, labels, dlbls) in enumerate(testLoader):
                       inputs = inputs.permute(0,3,2,1)
                       t1 = time.time()
                       if hyper_para.gpu:
                            inputs = inputs.cuda()
                            labels = labels.cuda()
                            dlbls = dlbls.cuda()
                       act,_ = C(inputs)
                       _, ind = torch.max(act,1)
                       acc += torch.sum(torch.eq(ind, labels)).cpu().numpy()
                       nel += ind.shape[0]
                       acc0 += torch.sum(torch.eq(ind[dlbls==0], labels[dlbls==0])).cpu().numpy()
                       nel0 += (dlbls==0).shape[0]
                       acc1 += torch.sum(torch.eq(ind[dlbls==1], labels[dlbls==1])).cpu().numpy()
                       nel1 += (dlbls==1).shape[0]
                #(['Val', acc/nel, acc0/nel0  , acc1/nel1 , nel0, nel1  ])
                if acc/nel >= best_acc:
                       torch.save(C.module.state_dict(), hyper_para.experiment_name +'_'+str(single_class_ind)+ '.pth')
                       best_acc = acc/nel
                running_cc = 0.0
            #print(['Train', acct/(nel0t+nel1t), acc0t/nel0t  , acc1t/nel1t , nel0t, nel1t  ])




def oceval(hyper_para):
    import sklearn
    for single_class_ind in hyper_para.inclass:
        C = wrn.WideResNet(28, num_classes=72, dropout_rate=0, widen_factor=10)
        C.load_state_dict(torch.load(hyper_para.experiment_name +'_'+str(hyper_para.inclass[0])+ '.pth'))
        C.cuda()
        if hyper_para.source == 'mnist':
               (x_train, y_train), (x_test, y_test) = load_mnist()
               (x_train2, y_train2), (x_test2, y_test2) = load_svhn()
        elif hyper_para.source == 'svhn':
               (x_train, y_train), (x_test, y_test) = load_svhn()
               (x_train2, y_train2), (x_test2, y_test2) = load_mnist()
        elif hyper_para.source == 'amazon':
               (x_train, y_train), (x_test, y_test) = load_amazon()
               (x_train2, y_train2), (x_test2, y_test2) = load_dslr()
        elif hyper_para.source == 'dslr':
               (x_train, y_train), (x_test, y_test) = load_dslr()
               (x_train2, y_train2), (x_test2, y_test2) = load_amazon()
        if hyper_para.method == 'justsource':
            x_train_task = x_train[y_train.flatten() == single_class_ind]

            x_test = x_train_task[int(len(x_train_task)*0.8):]
            x_train_task = x_train_task[0:int(len(x_train_task)*0.8)]
            domain_lbl = [0]*len(x_train_task)
            tst_lbl = [0]*len(x_test)
            dset2 = RotateDataset(x_test, tst_lbl )
            testLoader = DataLoader(dset2, batch_size=512, shuffle=True)

        else:#if hyper_para.method == 'balancedsourcetarget':
               x_train_task = x_train[y_train.flatten() == single_class_ind]
               x_train_task2 = x_train2[y_train2.flatten() == single_class_ind]
               x_train_task2 = x_train_task2[0: hyper_para.target_n]

               x_test = x_train_task[int(len(x_train_task)*0.8):]
               x_train_task = x_train_task[0:int(len(x_train_task)*0.8)]
               x_test2 = x_train_task2[int(len(x_train_task2)*0.8):]
               x_train_task2 = x_train_task2[0:int(len(x_train_task2)*0.8)]

               domain_lbl = [0]*len(x_train_task) + [1]*(len(x_train_task2)*int(len(x_train_task)/len(x_train_task2)))
               x_train_task2 = np.tile(x_train_task2, (int(len(x_train_task)/len(x_train_task2)),1,1,1) )
               x_train_task = np.concatenate((x_train_task, x_train_task2))
               tst_lbl = [0]*len(x_test) + [1]*(len(x_test2)*int(len(x_test)/len(x_test2)))
               x_test2 = np.tile(x_test2, (int(len(x_test)/len(x_test2)),1,1,1) )
               x_test = np.concatenate((x_test, x_test2))


               dset2 = RotateDataset(x_test, tst_lbl )
               testLoader = DataLoader(dset2, batch_size=512, shuffle=True)
               transformer = Transformer(8, 8)

        dset = RotateDataset(x_train_task, domain_lbl )
        trainLoader = DataLoader(dset, batch_size=512, shuffle=True)
    np.set_printoptions(threshold=sys.maxsize)
    correct = 0
    total = 0
    preds = []
    target = []
    for t in range(10):
         for i, (inputs, labels, dlbls) in enumerate(testLoader):
                   inputs = inputs.permute(0,3,2,1)		
                   if True:
                                inputs = inputs.cuda()
                                labels = labels.cuda()
                   act, f =  C(inputs)
                   _, ind = torch.max(act,1)
                   target+= labels.detach().cpu().tolist()
                   preds+= ind.detach().cpu().tolist()

    import matplotlib
    import matplotlib.pyplot as plt

    cm = sklearn.metrics.confusion_matrix(target, preds, normalize='true')
    plt.imshow(cm)
    plt.show()







        




def octest0(hyper_para):
    C = wrn.WideResNet(28, num_classes=72, dropout_rate=0, widen_factor=10).cuda()
    C.load_state_dict(torch.load(hyper_para.experiment_name +'_'+str(hyper_para.inclass[0])+ '.pth'))
    single_class_ind = hyper_para.inclass[0]
    if hyper_para.source == 'mnist':
               (x_train, y_train), (x_test, y_test) = load_mnist()
    elif hyper_para.source == 'svhn':
               (x_train, y_train), (x_test, y_test) = load_svhn()
    elif hyper_para.source == 'amazon':
               (x_train, y_train), (x_test, y_test) = load_amazon()
    elif hyper_para.source == 'dslr':
               (x_train, y_train), (x_test, y_test) = load_dslr()
    transformer = Transformer(8, 8)

    glabels = y_test.flatten() == single_class_ind


    print(hyper_para.source)
    print(len(x_test))
    print(np.sum(glabels))
    C.cuda().eval()
    scores = np.array([[]])
    features = []

    correct = 0
    total = 0
    preds = np.zeros((len(x_test), transformer.n_transforms))
    for t in range(1):
         score = []

         dset = CustomDataset(x_test, y_test )
         testLoader = DataLoader(dset, batch_size=128, shuffle=False)
         transformer = Transformer(8, 8)

         for i, (inputs, labels) in enumerate(testLoader):
                   inputs = torch.tensor(transformer.transform_batch(inputs.detach().cpu().numpy(), [t] * len(inputs)))	
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
         #print('AUROC: ' + str(AUC) )

    scores = np.sum(((preds)), 1)     
    fpr, tpr, thresholds = metrics.roc_curve(glabels, scores)
    AUC = metrics.auc(fpr, tpr)
    print('AUROC: ' + str(AUC) )
    file1 = open(hyper_para.source+ "_"+hyper_para.source+ "_"+ hyper_para.method +"txt","a")#append mode 
    file1.write(hyper_para.source+ "\t"+ hyper_para.method +"\t"+ str(hyper_para.inclass[0])+"\t"+ hyper_para.experiment_name+"\t"+str(AUC)+"\n") 
    file1.close() 


    C = wrn.WideResNet(28, num_classes=72, dropout_rate=0, widen_factor=10).cuda()
    C.load_state_dict(torch.load(hyper_para.experiment_name +'_'+str(hyper_para.inclass[0])+ '.pth'))
    single_class_ind = hyper_para.inclass[0]
    if hyper_para.target == 'mnist':
               (x_train, y_train), (x_test, y_test) = load_mnist()
    elif hyper_para.target == 'svhn':
               (x_train, y_train), (x_test, y_test) = load_svhn()
    elif hyper_para.target == 'amazon':
               (x_train, y_train), (sx_test, y_test) = load_amazon()
    elif hyper_para.target == 'dslr':
               (x_train, y_train), (x_test, y_test) = load_dslr()
    transformer = Transformer(8, 8)


    C.cuda().eval()
    scores = np.array([[]])
    features = []

    correct = 0
    total = 0
    preds = np.zeros((len(x_test)+len(x_train), transformer.n_transforms))
    for t in range(1):
         score = []

         dset = CustomDataset(np.concatenate((x_test,x_train),0), np.concatenate((y_test,y_train),0) )
         testLoader = DataLoader(dset, batch_size=128, shuffle=False)
         transformer = Transformer(8, 8)

         for i, (inputs, labels) in enumerate(testLoader):
                   inputs = torch.tensor(transformer.transform_batch(inputs.detach().cpu().numpy(), [t] * len(inputs)))	
                   inputs = inputs.permute(0,3,2,1)		
                   if True:
                                inputs = inputs.cuda()
                                labels = labels.cuda()
                   act, f =  C(inputs)
                   features+= f.detach().cpu().tolist()
                   #act = torch.nn.functional.softmax(act, dim=1)
                   score+=  act[:,t].detach().cpu().tolist() 
         preds[:,t] = list(score) 
         #print('AUROC: ' + str(AUC) )
    glabels = (np.concatenate((y_test,y_train))).flatten() == single_class_ind


    scores = np.sum(((preds)), 1)     
    fpr, tpr, thresholds = metrics.roc_curve(glabels, scores)
    AUC = metrics.auc(fpr, tpr)
    print('AUROC: ' + str(AUC) )
    file1 = open(hyper_para.target+ "_"+hyper_para.source+ "_"+ hyper_para.method +"txt","a")#append mode 
    file1.write(hyper_para.target+ "\t"+ hyper_para.method +"\t"+ str(hyper_para.inclass[0])+"\t"+ hyper_para.experiment_name+"\t"+str(AUC)+"\n") 
    file1.close() 



