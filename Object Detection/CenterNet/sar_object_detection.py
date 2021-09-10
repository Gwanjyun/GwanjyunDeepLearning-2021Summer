import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.functional import F


import torchvision
from torchvision.ops import nms
from torchvision.ops import RoIPool

from torchvision.ops import boxes as box_ops

import pandas as pd
import os
from tqdm import trange, tqdm
import cv2

import xml.etree.ElementTree as ET
from model import CenterNet

use_cuda = True
device = torch.device('cuda:4' if use_cuda else 'cpu')
x = torch.Tensor([0]).cuda(device)

def parsexml(gt_path, filename):
    gt_filename = gt_path + filename
    gt_sample = ET.parse(gt_filename)

    img_filename = gt_sample.find('source').find('filename').text

    Objects = gt_sample.find('objects').findall('object')
    objects_name = []
    bboxes = []
    for object in Objects:
        object_name = object.find('possibleresult').find('name').text
        objects_name.append(object_name)
        points = object.find('points').findall('point')
        xmin,ymin = points[0].text.split(',')
        xmax,ymax = points[2].text.split(',')
        bbox = [int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))]
        bboxes.append(torch.Tensor(bbox))
    cls_dict = { 'A220':0,
                 'A330':1, 
                 'A320/321':2, 
                 'Boeing737-800':3,
                 'Boeing787':4,
                 'ARJ21':5, 
                 'other':6}
    target = {
                'filename':img_filename,
                'labels':[cls_dict[i] for i in objects_name],
                'boxes':bboxes
            }
    return target

path = 'data/SAR_Airplane_Recognition_trainData/trainData/'
image_path = path + 'Images/'
gt_path = path + 'gt/'

class SAR_dset(torch.utils.data.Dataset):
    def __init__(self, path = 'data/SAR_Airplane_Recognition_trainData/trainData/'):
        super(SAR_dset, self).__init__()
        self.image_path = path + 'Images/'
        self.gt_path = path + 'gt/'
        self.gt_dir = sorted([i for i in os.listdir(self.gt_path) if i[-3:] == 'xml'])
        self.targets = []
        for filename in  self.gt_dir:
            self.targets.append(self.parsexml(gt_path,filename))
        
        self.imgs = []
        for target in tqdm(self.targets, desc = 'Load data'):
            filename = target['filename']
            img = cv2.imread(self.image_path + filename)
            img = torch.from_numpy(img).permute(2,0,1)/255
            self.imgs.append(img)
        
        self.train_img = self.imgs[:1500]
        self.train_targets = self.targets[:1500]
        
        self.test_img = self.imgs[1500:]
        self.test_targets = self.targets[1500:]
        self.mode = 'train'
        
        
        
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.train_img[index],self.train_targets[index]
        else:
            return self.test_img[index],self.test_targets[index]
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_img)
        else:
            len(self.test_img)
    

        
    def parsexml(self, gt_path, filename):
        gt_filename = gt_path + filename
        gt_sample = ET.parse(gt_filename)

        img_filename = gt_sample.find('source').find('filename').text

        Objects = gt_sample.find('objects').findall('object')
        objects_name = []
        bboxes = []
        for object in Objects:
            object_name = object.find('possibleresult').find('name').text
            objects_name.append(object_name)
            points = object.find('points').findall('point')
            xmin,ymin = points[0].text.split(',')
            xmax,ymax = points[2].text.split(',')
            bbox = [int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))]
            bboxes.append(torch.Tensor(bbox))
        cls_dict = { 'A220':0,
                     'A330':1, 
                     'A320/321':2, 
                     'Boeing737-800':3,
                     'Boeing787':4,
                     'ARJ21':5, 
                     'other':6}
        target = {
                    'filename':img_filename,
                    'classes':[cls_dict[i] for i in objects_name],
                    'bboxes':bboxes
                }
        return target


def collect(batch):
    img = torch.unsqueeze(batch[0][0],0)
    target = [batch[0][1]]
    for i in range(len(target)):
        if isinstance(target[i]['bboxes'],list):
            target[i]['bboxes'] = torch.stack(target[i]['bboxes'],dim = 0)
            target[i]['classes'] = torch.from_numpy(np.array(target[i]['classes'],dtype = np.int64))
    return img, target

sar_dset = SAR_dset()
sar_dataloader = torch.utils.data.DataLoader(sar_dset, batch_size = 1, shuffle = True,collate_fn = collect)
for x,y in tqdm(sar_dataloader):
    pass

def getTarget(y, device):
    for i in range(len(y)):
        y[i]['bboxes'] = y[i]['bboxes'].cuda(device)
        y[i]['classes'] = y[i]['classes'].cuda(device)
    return y

try:
    centerNet = torch.load('CenterNet/model/centerNet_sar.pt')
    print('!!!!load model success!!!!6')
except:
    centerNet = CenterNet(7)
centerNet.to(device)
centerNet.mode = 'train'

optimizer = optim.AdamW(centerNet.parameters(), 5e-4)

centerNet.train()
centerNet.mode = 'train'
min_loss = 1e9
epoch_range = trange(80)
for epoch in epoch_range:
    myiter = tqdm(sar_dataloader,colour = '#0066FF')
    myiter.set_description_str('car dataloader')
    all_loss = 0
    for x,y in myiter:
        x = x.cuda(device)
        target = getTarget(y,device)
        result, losses = centerNet(x,y)
        optimizer.zero_grad()
        loss = losses['point_focal_loss'] + losses['size_loss'] + losses['offset_loss']
        loss.backward()
        optimizer.step()
        all_loss += float(loss)
        myiter.set_postfix(l = float(loss),fl = float(losses['point_focal_loss']),
                           szl = float(losses['size_loss']),
                           osl = float(losses['offset_loss']),
                           )
    epoch_range.set_postfix(allLoss = float(all_loss),minLoss = float(min_loss))
    
    if all_loss < min_loss:
        min_loss = all_loss
        torch.save(centerNet, 'CenterNet/model/centerNet_sar.pt')
torch.save(centerNet, 'CenterNet/model/centerNet_sar_final.pt')










