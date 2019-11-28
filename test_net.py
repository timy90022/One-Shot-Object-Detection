
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def save_weight(weight, time, seen):
  time = np.where(time==0, 1, time)
  weight = weight/time[:,np.newaxis]
  result_map = np.zeros((len(weight), len(weight)))
  for i in range(len(weight)):
    for j in range(len(weight)):
      v1 = weight[i]
      v2 = weight[j]
      # v1_ = np.linalg.norm(v1)
      # v2_ = np.linalg.norm(v2)
      # v12 = np.sum(v1*v2)
      # print(v12)
      # print(v1_)
      # print(v2_)
      distance = np.linalg.norm(v1-v2)
      if np.sum(v1*v2)== 0 :
        result_map[i][j] = 0
      else:
        result_map[i][j] = distance
      

  df = pd.DataFrame (result_map)

  ## save to xlsx file

  filepath = 'similarity_%d.xlsx'%(seen)

  df.to_excel(filepath, index=False)

  weight = weight*255


  cv2.imwrite('./weight_%d.png'%(seen), weight)

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='coco', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      default=True)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      default=True)
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--s', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=10, type=int)
  parser.add_argument('--p', dest='checkpoint',
                      help='checkpoint to load network',
                      default=1663, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--seen', dest='seen',
                       help='Reserved: 1 training, 2 testing, 3 both', default=2, type=int)
  parser.add_argument('--a', dest='average', help='average the top_k candidate samples', default=1, type=int)
  parser.add_argument('--g', dest='group',
                      help='which group want to training/testing',
                      default=0, type=int) 
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2017_train"
      args.imdbval_name = "coco_2017_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_{}.yml".format(args.net, args.group) if args.group != 0 else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)


  # Load dataset
  cfg.TRAIN.USE_FLIPPED = False
  imdb_vu, roidb_vu, ratio_list_vu, ratio_index_vu, query_vu = combined_roidb(args.imdbval_name, False, seen=args.seen)
  imdb_vu.competition_mode(on=True)
  dataset_vu = roibatchLoader(roidb_vu, ratio_list_vu, ratio_index_vu, query_vu, 1, imdb_vu.num_classes, training=False, seen=args.seen)


  
  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb_vu.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb_vu.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb_vu.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb_vu.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()
  fasterRCNN.create_architecture()

  # Load checkpoint
  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)

  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  # initilize the tensor holder here.
  print('load model successfully!')
  im_data = torch.FloatTensor(1)
  query   = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  catgory = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    cfg.CUDA = True
    fasterRCNN.cuda()
    im_data = im_data.cuda()
    query = query.cuda()
    im_info = im_info.cuda()
    catgory = catgory.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  query = Variable(query)
  im_info = Variable(im_info)
  catgory = Variable(catgory)
  gt_boxes = Variable(gt_boxes)
    
  # record time
  start = time.time()

  # visiualization
  vis = args.vis
  if vis:
    thresh = 0.05
  else:
    thresh = 0.0
  max_per_image = 100

  # create output Directory
  output_dir_vu = get_output_dir(imdb_vu, 'faster_rcnn_unseen')

  fasterRCNN.eval()
  for avg in range(args.average):
    dataset_vu.query_position = avg
    dataloader_vu = torch.utils.data.DataLoader(dataset_vu, batch_size=1,shuffle=False, num_workers=0,pin_memory=True)

    data_iter_vu = iter(dataloader_vu)

    # total quantity of testing images, each images include multiple detect class
    num_images_vu = len(imdb_vu.image_index)
    num_detect = len(ratio_index_vu[0])

    all_boxes = [[[] for _ in xrange(num_images_vu)]
                for _ in xrange(imdb_vu.num_classes)]

    
    _t = {'im_detect': time.time(), 'misc': time.time()}
    if args.group != 0:
      det_file = os.path.join(output_dir_vu, 'sess%d_g%d_seen%d_%d.pkl'%(args.checksession, args.group, args.seen, avg))
    else:
      det_file = os.path.join(output_dir_vu, 'sess%d_seen%d_%d.pkl'%(args.checksession, args.seen, avg))
    print(det_file)

    if os.path.exists(det_file):
      with open(det_file, 'rb') as fid:
        all_boxes = pickle.load(fid)
    else:
      for i,index in enumerate(ratio_index_vu[0]):
        data = next(data_iter_vu)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        query.data.resize_(data[1].size()).copy_(data[1])
        im_info.data.resize_(data[2].size()).copy_(data[2])
        gt_boxes.data.resize_(data[3].size()).copy_(data[3])
        catgory.data.resize_(data[4].size()).copy_(data[4])


        # Run Testing
        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, _, RCNN_loss_bbox, \
        rois_label, weight = fasterRCNN(im_data, query, im_info, gt_boxes, catgory)


        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        
        # Apply bounding-box regression 
        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
              if args.class_agnostic:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  box_deltas = box_deltas.view(1, -1, 4)
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))


        # Resize to original ratio
        pred_boxes /= data[2][0][2].item()

        # Remove batch_size dimension
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        # Record time
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        # Post processing
        inds = torch.nonzero(scores>thresh).view(-1)
        if inds.numel() > 0:
          # remove useless indices
          cls_scores = scores[inds]
          cls_boxes = pred_boxes[inds, :]
          cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

          # rearrange order
          _, order = torch.sort(cls_scores, 0, True)
          cls_dets = cls_dets[order]

          # NMS
          keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
          cls_dets = cls_dets[keep.view(-1).long()]
          all_boxes[catgory][index] = cls_dets.cpu().numpy()

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
          try:
            image_scores = all_boxes[catgory][index][:,-1]
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]

                keep = np.where(all_boxes[catgory][index][:,-1] >= image_thresh)[0]
                all_boxes[catgory][index] = all_boxes[catgory][index][keep, :]
          except:
            pass

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            .format(i + 1, num_detect, detect_time, nms_time))
        sys.stdout.flush()

        # save test image
        if vis and i%1==0:
          im2show = cv2.imread(dataset_vu._roidb[dataset_vu.ratio_index[i]]['image'])
          im2show = vis_detections(im2show, 'shot', cls_dets.cpu().numpy(), 0.8)

          o_query = data[1][0].permute(1, 2,0).contiguous().cpu().numpy()
          o_query *= [0.229, 0.224, 0.225]
          o_query += [0.485, 0.456, 0.406]
          o_query *= 255
          o_query = o_query[:,:,::-1]

          (h,w,c) = im2show.shape
          o_query = cv2.resize(o_query, (h, h),interpolation=cv2.INTER_LINEAR)
          im2show = np.concatenate((im2show, o_query), axis=1)

          cv2.imwrite('./test_img/%d_d.png'%(i), im2show)
      
    
      with open(det_file, 'wb') as f:
          pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
      
    print('Evaluating detections')
    imdb_vu.evaluate_detections(all_boxes, output_dir_vu) 


    end = time.time()
    print("test time: %0.4fs" % (end - start))