import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import math

import numpy as np

class CenterNet_ResNet_backbone(nn.Module):
    def __init__(self, downsample = 4):
        super(CenterNet_ResNet_backbone, self).__init__()
        # pre-process
        self.preprocess = None
        # resnet backbone, down sample
        self.resnetBackbone = self._make_backbone()
        # up sample
        num_layer = 5 - int(math.log2(downsample))
        self.upsample = self._make_deconv_layer(num_layer)
        
        self._normal_weight(self.upsample)
        
    def forward(self, x):
        x_pre = x
        if self.preprocess is not None:
            x_pre = self.preprocess(x)
        
        x = self.resnetBackbone(x_pre)
        x = self.upsample(x)
        return x, x_pre
    
    def _make_deconv_layer(self, num_layer):
        inplanes = 2048
        outplanes = 1024
        layers = []
        for i in range(num_layer):
            fc = [
                        nn.Conv2d(inplanes, outplanes, 3, 1, 1),
                        nn.BatchNorm2d(outplanes),
                        nn.ReLU(inplace = True)
            ]
            up = [
                        nn.ConvTranspose2d(outplanes, outplanes, 4, 2, 1),
                        nn.BatchNorm2d(outplanes),
                        nn.ReLU(inplace = True)
            ]
            layers.extend(fc)
            layers.extend(up)
            inplanes = outplanes
            outplanes = int(outplanes/2)
        return nn.Sequential(*layers)
    
    def _make_backbone(self):
        resnet = torchvision.models.resnet50(pretrained=True)
        backboneList = [
                        resnet.conv1, resnet.bn1, 
                        resnet.relu, resnet.maxpool,
                        resnet.layer1, resnet.layer2,
                        resnet.layer3, resnet.layer4
                        ]
        return nn.Sequential(*backboneList)
    
    def _normal_weight(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std = 0.001)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std = 0.001)
    
class CenterNet_Network(nn.Module):
    def __init__(self, num_classes = 1, downsample = 4):
        super(CenterNet_Network, self).__init__()
        self.downsample = downsample
        self.centerNet_ResNet_backbone = CenterNet_ResNet_backbone(downsample)
        outplanes = 2**(10 - int(math.log2(downsample)))
        self.heatmap_network = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, 1, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes, num_classes, 1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )
        self.size_network = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, 1, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.offset_network = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, 1, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )

        self._normal_weight(self.heatmap_network)
        self._normal_weight(self.size_network)
        self._normal_weight(self.offset_network)
        
    def forward(self, x):
        feature_map, x_pre = self.centerNet_ResNet_backbone(x)
        hm = self.heatmap_network(feature_map)
        sz = self.size_network(feature_map)
        os = self.offset_network(feature_map)
        return x_pre, hm, sz, os

    def _normal_weight(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std = 0.001)
    
class focal_loss(nn.Module):
    def __init__(self, alpha = 2, beta = 4):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred_hm, gt_hm):
        focalLoss = 0
        pred_hm = torch.clamp(pred_hm, 1e-6, 1-1e-6)
        pos_idx = (gt_hm == 1)
        neg_idx = (gt_hm != 1)
        # positive focal loss
        focalLoss += torch.sum((1 - pred_hm[pos_idx]).pow(self.alpha)*torch.log(pred_hm[pos_idx]))
        # negative
        focalLoss += torch.sum((1 - gt_hm[neg_idx]).pow(self.beta)*pred_hm[neg_idx].pow(self.alpha)*torch.log(1 - pred_hm[neg_idx]))

        return -focalLoss
        
        
class CenterNet(nn.Module):
    def __init__(self, num_classes = 1, downsample = 4, mode = 'train'):
        super(CenterNet, self).__init__()
        self.num_classes = num_classes
        self.downsample = downsample
        self.mode = mode
        # centerNet neural network
        self.centerNet = CenterNet_Network(num_classes, downsample)
        
    def forward(self, x, target = None):
        batch, C, H, W = x.shape
        # x_pre, heatmap, size, offset
        x_pre,hm,sz,os = self.centerNet(x)
        hm_peaks = self._nms(hm)
        hm_peaks_permute = hm_peaks.permute(0,2,3,1)
        sz_permute = sz.permute(0,2,3,1)
        os_permute = os.permute(0,2,3,1)
        result = []
        losses = []

        for img_index in range(batch):
            img_hm_peaks = hm_peaks_permute[img_index].detach() # single image heatmap
            img_szs = sz_permute[img_index].detach() # single image size
            img_oss = os_permute[img_index].detach() # single image offset
            values, idxs = img_hm_peaks.max(dim = 2)
            keep = values>0

            img_points =torch.nonzero(values)[:,[1,0]] # center point
            img_scores = values[keep] # scores
            img_idxs = idxs[keep] # classes
            img_sz = img_szs[keep] # size
            img_os = img_oss[keep] # offset

            img_sz[:,0] = img_sz[:,0]*W
            img_sz[:,1] = img_sz[:,1]*H

            xymin = img_points*self.downsample + img_os - img_sz/2
            xymax = img_points*self.downsample + img_os + img_sz/2
            img_bboxes = torch.hstack((xymin, xymax))
            img_bboxes = self._bbox_clamp(img_bboxes, (H,W))

            result.append(
                dict(classes = img_idxs, 
                     scores = img_scores,
                     bboxes = img_bboxes,
                     )
#                      heatmap_peaks = hm_peaks,
#                      size = sz,
#                      offset = os,
#                      img_cpoints = img_points,
#                      img_size = img_sz,
#                      img_offset = img_os)
            )
            
        if self.mode == 'train':
            # training mode
            assert target is not None
            # initialize the ground truth
            gt_hm = torch.zeros_like(hm, device = hm.device)
            hm_H,hm_W = hm.shape[-2:]
            hm_index = torch.stack(torch.meshgrid((torch.arange(hm_H),torch.arange(hm_W))),-1)
            if hm.is_cuda:
                hm_index = hm_index.cuda(hm.device)
            # gt_sz = torch.zeros_like(sz_permute, device = sz_permute.device)
            # gt_os = torch.zeros_like(os_permute, device = os_permute.device)

            size_loss = 0
            offset_loss = 0


            N = 0
            for img_index in range(batch):
                img_target = target[img_index]
                gt_hm_cpoitns, gt_sizes, gt_offsets = self.bbox2point_size_offset(img_target, self.downsample)
                
                gt_sizes = gt_sizes.detach()
                gt_offsets = gt_offsets.detach()
                
                gt_sizes[:,0] = gt_sizes[:,0]/W
                gt_sizes[:,1] = gt_sizes[:,1]/H
                
                for i in range(len(img_target['classes'])):
                    N += 1
                    img_cls = int(img_target['classes'][i])
                    x_idx,y_idx = int(gt_hm_cpoitns[i][0]),int(gt_hm_cpoitns[i][1])

                    # gt_sz[img_index][x_idx,y_idx] = gt_sizes[i] # The ground truth of image size
                    
                    # gt_os[img_index][x_idx,y_idx] = gt_offsets[i] # The ground truth of image local offset
                    
                    size_loss += F.smooth_l1_loss(sz_permute[img_index][x_idx, y_idx], gt_sizes[i])
                    offset_loss += F.smooth_l1_loss(os_permute[img_index][x_idx, y_idx], gt_offsets[i])


                    sigma = float(torch.max(gt_sizes[i]))/self.downsample/3
                    gauss_kernel = self.gauss2D(hm_index, gt_hm_cpoitns[i], sigma).detach()
                    if gt_hm.is_cuda:
                        gauss_kernel = gauss_kernel.cuda(gt_hm.device)
                    gt_hm[img_index][img_cls] = torch.max(gt_hm[img_index][img_cls], gauss_kernel)
            # supervision acts only at keypoints. 
            # keep_train = torch.sum(gt_hm.permute(0,2,3,1) == 1,-1) > 0
            # pred_sz = sz_permute[keep_train]
            # pred_os = os_permute[keep_train] 
            # N = len(gt_sz[keep_train])

            # N = 
            
            # size_loss = F.smooth_l1_loss(pred_sz, gt_sz[keep_train])
            # offset_loss = F.smooth_l1_loss(pred_os, gt_os[keep_train])
            
            size_loss = size_loss/N

            offset_loss = offset_loss/N

            focalLoss = focal_loss(2,4)
            point_focal_loss = focalLoss(hm, gt_hm)/N

            
            
            losses = dict(
                point_focal_loss = point_focal_loss,
                size_loss = size_loss,
                offset_loss = offset_loss,
                heatmap = hm,)
                # pred_sz = pred_sz,
                # pred_os = pred_os,
                # gt_sz = gt_sz[keep_train],
                # gt_os = gt_os[keep_train],
                # gt_sizes = gt_sizes,
                # gt_hm = gt_hm,
#                 gt_sz = gt_sz,
#                 gt_os = gt_os)
        return result, losses
            
            
    def _nms(self, hm):
        # serve as nms, choose peak of heatmap(keypoints)
        pool = nn.MaxPool2d(3,1,1)
        hmax = pool(hm)
        keep = (hm == hmax).float()
        return hm * keep
                                         
    def _bbox_clamp(self, bbox, imsize):
        bbox[:,[0,2]] = torch.clamp(bbox[:,[0,2]], 0, imsize[1])
        bbox[:,[1,3]] = torch.clamp(bbox[:,[1,3]], 0, imsize[0])
        return bbox    
    
    def gauss2D(self, hm_index, gt_hm_cpoint, sigma = 10):
        gauss_kernel = (-(hm_index - gt_hm_cpoint).pow(2).sum(-1)/(2*sigma)).exp()
        return gauss_kernel
    
    def bbox2point_size_offset(self, img_target, downsample = 4):
        img_bboxes = img_target['bboxes']
        xmin, ymin = img_bboxes[:, 0:1], img_bboxes[:, 1:2]
        xmax, ymax = img_bboxes[:, 2:3], img_bboxes[:, 3:4]
        gt_img_cpoints = torch.hstack([(ymin+ymax)/2, (xmin+xmax)/2])
        gt_hm_cpoitns = torch.floor(gt_img_cpoints/downsample)
        gt_sizes = torch.hstack([xmax-xmin, ymax-ymin])
        gt_offsets =  gt_img_cpoints/downsample - gt_hm_cpoitns
        return gt_hm_cpoitns, gt_sizes, gt_offsets