import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.utils.net_utils import *

class match_block(nn.Module):
    def __init__(self, inplanes):
        super(match_block, self).__init__()

        self.sub_sample = False

        self.in_channels = inplanes
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.Q = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.Q[1].weight, 0)
        nn.init.constant_(self.Q[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )
        
        self.ChannelGate = ChannelGate(self.in_channels)
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)


        
    def forward(self, detect, aim):

        

        batch_size, channels, height_a, width_a = aim.shape
        batch_size, channels, height_d, width_d = detect.shape


        #####################################find aim image similar object ####################################################

        d_x = self.g(detect).view(batch_size, self.inter_channels, -1)
        d_x = d_x.permute(0, 2, 1).contiguous()

        a_x = self.g(aim).view(batch_size, self.inter_channels, -1)
        a_x = a_x.permute(0, 2, 1).contiguous()

        theta_x = self.theta(aim).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(detect).view(batch_size, self.inter_channels, -1)

        

        f = torch.matmul(theta_x, phi_x)

        N = f.size(-1)
        f_div_C = f / N

        f = f.permute(0, 2, 1).contiguous()
        N = f.size(-1)
        fi_div_C = f / N

        non_aim = torch.matmul(f_div_C, d_x)
        non_aim = non_aim.permute(0, 2, 1).contiguous()
        non_aim = non_aim.view(batch_size, self.inter_channels, height_a, width_a)
        non_aim = self.W(non_aim)
        non_aim = non_aim + aim

        non_det = torch.matmul(fi_div_C, a_x)
        non_det = non_det.permute(0, 2, 1).contiguous()
        non_det = non_det.view(batch_size, self.inter_channels, height_d, width_d)
        non_det = self.Q(non_det)
        non_det = non_det + detect

        ##################################### Response in chaneel weight ####################################################

        c_weight = self.ChannelGate(non_aim)
        act_aim = non_aim * c_weight
        act_det = non_det * c_weight

        return non_det, act_det, act_aim, c_weight

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic

        
        self.match_net = match_block(self.dout_base_model)


        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        self.triplet_loss = torch.nn.MarginRankingLoss(margin = cfg.TRAIN.MARGIN)

    def forward(self, im_data, query, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        detect_feat = self.RCNN_base(im_data)
        query_feat = self.RCNN_base(query)

        rpn_feat, act_feat, act_aim, c_weight = self.match_net(detect_feat, query_feat)


        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feat, im_info, gt_boxes, num_boxes)


        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            margin_loss = 0
            rpn_loss_bbox = 0
            score_label = None

        rois = Variable(rois)
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(act_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(act_feat, rois.view(-1,5))


        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        query_feat  = self._head_to_tail(act_aim)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)


        pooled_feat = pooled_feat.view(batch_size, rois.size(1), -1)
        query_feat = query_feat.unsqueeze(1).repeat(1,rois.size(1),1)


        pooled_feat = torch.cat((pooled_feat,query_feat), dim=2).view(-1, 4096)


        # compute object classification probability
        score = self.RCNN_cls_score(pooled_feat)

        score_prob = F.softmax(score, 1)[:,1]


        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        

        if self.training:
            # classification loss
            
            score_label = rois_label.view(batch_size, -1).float()
            gt_map = torch.abs(score_label.unsqueeze(1)-score_label.unsqueeze(-1))

            score_prob = score_prob.view(batch_size, -1)
            pr_map = torch.abs(score_prob.unsqueeze(1)-score_prob.unsqueeze(-1))
            target = -((gt_map-1)**2) + gt_map
            
            RCNN_loss_cls = F.cross_entropy(score, rois_label)

            margin_loss = 3 * self.triplet_loss(pr_map, gt_map, target)

            # RCNN_loss_cls = similarity + margin_loss
    
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = score_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, margin_loss, RCNN_loss_bbox, rois_label, c_weight

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score[0], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score[1], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
