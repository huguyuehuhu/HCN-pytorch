# encoding: utf-8

"""
@author: huguyuehuhu
@time: 18-4-16 下午6:51
Permission is given to modify the code, any problem please contact huguyuehuhu@gmail.com
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import utils
import torchvision
import os

class HCN(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=3,
                 num_joint=25,
                 num_person=2,
                 out_channel=64,
                 window_size=32,
                 num_class = 60,
                 fc7_channel=256,
                 bypass_conv4=True,
                 ):
        super(HCN, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        if(not bypass_conv4):
            self.conv4 = nn.Sequential(
                nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2))
        else:
            self.conv4=None
            # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        if(not bypass_conv4):
            self.conv4m = nn.Sequential(
                nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2))
        else:
            self.conv4m=None
        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2 if not bypass_conv4 else out_channel, out_channels=out_channel*2 if not bypass_conv4 else out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2 if not bypass_conv4 else out_channel, out_channels=out_channel*4 if not bypass_conv4 else out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7= nn.Sequential(
            nn.Linear((out_channel * 4)*(window_size//16)*(window_size//16) if not bypass_conv4 else (out_channel * 2)*(window_size//8)*(window_size//8),fc7_channel), # 4*4 for window=64; 8*8 for window=128
            
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(fc7_channel,num_class)

        # initial weight
        utils.initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')


    def forward(self, x,target=None):
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)

        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])

            out = self.conv2(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            out = self.conv3(out)
            if(self.conv4):
                out_p = self.conv4(out)
            else:
                out_p=out

            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            out = self.conv2m(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3m(out)
            if(self.conv4m):
                out_m = self.conv4m(out)
            else:
                out_m=out
            # concat
            out = torch.cat((out_p,out_m),dim=1)
            out = self.conv5(out)
            out = self.conv6(out)

            logits.append(out)

        # max out logits
        out = torch.max(logits[0],logits[1])
        out = out.view(out.size(0), -1)
        
        out = self.fc7(out)
        out = self.fc8(out)

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        return out


def loss_fn(outputs,labels,current_epoch=None,params=None):
    """
    Compute the cross entropy loss given outputs and labels.

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    if params.loss_args["type"] == 'CE':
        CE = nn.CrossEntropyLoss()(outputs, labels)
        loss_all = CE
        loss_bag = {'ls_all': loss_all, 'ls_CE': CE}
    #elif: other losses

    return loss_bag


def accuracytop1(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def accuracytop2(output, target, topk=(2,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def accuracytop3(output, target, topk=(3,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def accuracytop5(output, target, topk=(5,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracytop1': accuracytop1,
    'accuracytop5': accuracytop5,
    # could add more metrics such as accuracy for each token type
}

if __name__ == '__main__':
    model = HCN()
    children = list(model.children())
    print(children)
