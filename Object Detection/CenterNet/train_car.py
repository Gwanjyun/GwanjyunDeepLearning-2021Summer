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

from model import CenterNet

use_cuda = True
device = torch.device('cuda:4' if use_cuda else 'cpu')
x = torch.Tensor([0]).cuda(device)

# dataset的路径
path = 'data/car-object-detection/data/'

'''
    image file name: array([bbox1,bbox2])
'''
train_bbox_pd = pd.read_csv(path + 'train_solution_bounding_boxes.csv')
train_bbox_np = train_bbox_pd.to_numpy()
train_bbox = {}
for d in train_bbox_np:
    if d[0] not in train_bbox:
        train_bbox.update({d[0]:[d[1:]]})
    else:
        train_bbox[d[0]].append(d[1:])
train_bbox = {d:np.array(train_bbox[d]) for d in train_bbox}

# dataset的路径
path = 'data/car-object-detection/data/'
train_imagefile = [i for i in os.listdir(path + 'training_images') if i[-3:] == 'jpg']
test_imagefile = [i for i in os.listdir(path + 'testing_images') if i[-3:] == 'jpg']

class cardset(torch.utils.data.Dataset):
    def __init__(self, path = 'data/car-object-detection/data/'):
        super(cardset, self).__init__()
        self.path = path
        # 读取图像文件名
        self.train_imagefile = [i for i in os.listdir(path + 'training_images') if i[-3:] == 'jpg']
        self.test_imagefile = [i for i in os.listdir(path + 'testing_images') if i[-3:] == 'jpg']
        self.train_img = []
        self.test_img = []
        # 读取训练集的bbox
        train_bbox_pd = pd.read_csv(self.path + 'train_solution_bounding_boxes.csv')
        self.train_bbox_np = train_bbox_pd.to_numpy()
        self.train_bbox = {}
        self.idx2file = {}
        self.file2idx = {}
        i = 0
        for d in self.train_bbox_np:
            if d[0] not in self.train_bbox:
                self.train_bbox.update({d[0]:[d[1:]]})
                self.idx2file.update({i:d[0]})
                self.file2idx.update({d[0]:i})
                i += 1
            else:
                self.train_bbox[d[0]].append(d[1:])
        self.train_bbox = {d:np.array(self.train_bbox[d],dtype = np.float32) for d in self.train_bbox}
        # 读取数据到内存
        for filename in tqdm(self.train_imagefile,desc = 'Reading train data'):
            try:
                img = Image.open(path + 'training_images/' + filename)
            except:
                continue
            self.train_img.append([filename,img])
            if filename not in self.train_bbox:
                self.train_bbox.update({filename:[]})
                self.idx2file.update({i:filename})
                self.file2idx.update({filename:i})
                i += 1
                
        for filename in tqdm(self.test_imagefile,desc = 'Reading test data'):
            try:
                img = Image.open(path + 'testing_images/' + filename)
            except:
                continue
            self.test_img.append([filename,img])
            
    def __getitem__(self, index):
        if isinstance(self.train_img[index][1],(Image.Image)):
            self.train_img[index][1] = torchvision.transforms.functional.pil_to_tensor(self.train_img[index][1])/255
        img = self.train_img[index][1]
#         print(self.train_bbox[self.train_img[index][0]])
        label_num = self.file2idx[self.train_img[index][0]]
        return img, label_num
    
    def __len__(self):
        return len(self.train_img)

car = cardset()
def collect(batch):
    img,label_num = [i for i in zip(*batch)]
    img = torch.stack(img,0)
    label_num = torch.Tensor(label_num)
    return img,label_num
car_dataloader = torch.utils.data.DataLoader(car, batch_size = 8, shuffle = True,collate_fn = collect,drop_last = True)

def getTarget(x,y):
    target = []
    for i in y:
        t = car.train_bbox[car.idx2file[int(i)]]
        shape = len(t)
        label = 1
        if len(t) == 0:
            t = [[0,0,x.shape[-1]-1,x.shape[-2]-1]]
            shape = 1
            label = 0
        target.append({'bboxes':torch.Tensor(t).cuda(device), 'classes': torch.from_numpy(np.zeros(shape, dtype = np.int64) + label).cuda(device)})
    return target
try:
    centerNet = torch.load('CenterNet/model/centerNet3.pt')
except:
    centerNet = CenterNet(2)
centerNet.to(device)
optimizer = optim.Adam(centerNet.parameters(),5e-4)

centerNet.train()
centerNet.mode = 'train'
min_loss = 1e9
epoch_range = trange(100)

for epoch in epoch_range:
    myiter = tqdm(car_dataloader,colour = '#0066FF')
    myiter.set_description_str('car dataloader')
    all_loss = 0
    for x,y in myiter:
        x = x.cuda(device)
        target = getTarget(x,y)
        result, losses = centerNet(x,target)

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
        torch.save(centerNet, 'CenterNet/model/centerNet3.pt')
torch.save(centerNet, 'CenterNet/model/centerNet3.pt')
    



