
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch
from collections import Counter

from scipy.misc import imread
from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch, get_minibatch
from model.utils.blob import prep_im_for_blob, im_list_to_blob, crop
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import cv2
import random
import time
import pdb

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, query, batch_size, num_classes, training=True, normalize=None, seen=True):
    self._roidb = roidb
    self._query = query
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.query_position = 0

    if training:
        self.ratio_index = ratio_index
    else:
        self.cat_list = ratio_index[1]
        self.ratio_index = ratio_index[0]

    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)

    # given the ratio_list, we want to make the ratio same for each batch.
    self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
    num_batch = int(np.ceil(len(ratio_index) / batch_size))
    if self.training:
        for i in range(num_batch):
            left_idx = i*batch_size
            right_idx = min((i+1)*batch_size-1, self.data_size-1)

            if ratio_list[right_idx] < 1:
                # for ratio < 1, we preserve the leftmost in each batch.
                target_ratio = ratio_list[left_idx]
            elif ratio_list[left_idx] > 1:
                # for ratio > 1, we preserve the rightmost in each batch.
                target_ratio = ratio_list[right_idx]
            else:
                # for ratio cross 1, we make it to be 1.
                target_ratio = 1

            self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio

    self._cat_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
            82, 84, 85, 86, 87, 88, 89, 90
        ]
    self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
    self._classes_inv = {
            value: key for key, value in self._classes.items()
        }
    
    self.filter(seen)
    self.probability()


  def __getitem__(self, index):
    index_ratio = int(self.ratio_index[index])

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index_ratio]]

    blobs = get_minibatch(minibatch_db , self._num_classes)

    blobs['gt_boxes'] = [x for x in blobs['gt_boxes'] if x[-1] in self.list_ind]
    blobs['gt_boxes'] = np.array(blobs['gt_boxes'])


    if self.training:
        # Random choice query catgory
        catgory = blobs['gt_boxes'][:,-1]
        cand = np.unique(catgory)
        if len(cand)==1:
            choice = cand[0]
        else:
            p = []
            for i in cand:
                p.append(self.show_time[i])
            p = np.array(p)
            p /= p.sum()
            choice  = np.random.choice(cand,1,p=p)[0]

        # Delete useless gt_boxes
        blobs['gt_boxes'][:,-1] = np.where(blobs['gt_boxes'][:,-1]==choice,1,0)
        # Get query image
        query = self.load_query(choice)
    else:
        query = self.load_query(index, minibatch_db[0]['img_id'])

    data = torch.from_numpy(blobs['data'])
    query = torch.from_numpy(query)
    query = query.permute(0, 3, 1, 2).contiguous().squeeze(0)
    im_info = torch.from_numpy(blobs['im_info'])
    
    # we need to random shuffle the bounding box.
    data_height, data_width = data.size(1), data.size(2)
    if self.training:
        np.random.shuffle(blobs['gt_boxes'])
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])

        ########################################################
        # padding the input image to fixed size for each group #
        ########################################################

        # NOTE1: need to cope with the case where a group cover both conditions. (done)
        # NOTE2: need to consider the situation for the tail samples. (no worry)
        # NOTE3: need to implement a parallel data loader. (no worry)
        # get the index range

        # if the image need to crop, crop to the target size.
        ratio = self.ratio_list_batch[index]

        if self._roidb[index_ratio]['need_crop']:
            if ratio < 1:
                # this means that data_width << data_height, we need to crop the
                # data_height
                min_y = int(torch.min(gt_boxes[:,1]))
                max_y = int(torch.max(gt_boxes[:,3]))
                trim_size = int(np.floor(data_width / ratio))
                if trim_size > data_height:
                    trim_size = data_height                
                box_region = max_y - min_y + 1
                if min_y == 0:
                    y_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        y_s_min = max(max_y-trim_size, 0)
                        y_s_max = min(min_y, data_height-trim_size)
                        if y_s_min == y_s_max:
                            y_s = y_s_min
                        else:
                            y_s = np.random.choice(range(y_s_min, y_s_max))
                    else:
                        y_s_add = int((box_region-trim_size)/2)
                        if y_s_add == 0:
                            y_s = min_y
                        else:
                            y_s = np.random.choice(range(min_y, min_y+y_s_add))
                # crop the image
                data = data[:, y_s:(y_s + trim_size), :, :]

                # shift y coordiante of gt_boxes
                gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                # update gt bounding box according the trip
                gt_boxes[:, 1].clamp_(0, trim_size - 1)
                gt_boxes[:, 3].clamp_(0, trim_size - 1)

            else:
                # this means that data_width >> data_height, we need to crop the
                # data_width
                min_x = int(torch.min(gt_boxes[:,0]))
                max_x = int(torch.max(gt_boxes[:,2]))
                trim_size = int(np.ceil(data_height * ratio))
                if trim_size > data_width:
                    trim_size = data_width                
                box_region = max_x - min_x + 1
                if min_x == 0:
                    x_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        x_s_min = max(max_x-trim_size, 0)
                        x_s_max = min(min_x, data_width-trim_size)
                        if x_s_min == x_s_max:
                            x_s = x_s_min
                        else:
                            x_s = np.random.choice(range(x_s_min, x_s_max))
                    else:
                        x_s_add = int((box_region-trim_size)/2)
                        if x_s_add == 0:
                            x_s = min_x
                        else:
                            x_s = np.random.choice(range(min_x, min_x+x_s_add))
                # crop the image
                data = data[:, :, x_s:(x_s + trim_size), :]

                # shift x coordiante of gt_boxes
                gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                # update gt bounding box according the trip
                gt_boxes[:, 0].clamp_(0, trim_size - 1)
                gt_boxes[:, 2].clamp_(0, trim_size - 1)

        # based on the ratio, padding the image.
        if ratio < 1:
            # this means that data_width < data_height
            trim_size = int(np.floor(data_width / ratio))

            padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                             data_width, 3).zero_()

            padding_data[:data_height, :, :] = data[0]
            # update im_info
            im_info[0, 0] = padding_data.size(0)
            # print("height %d %d \n" %(index, anchor_idx))
        elif ratio > 1:
            # this means that data_width > data_height
            # if the image need to crop.
            padding_data = torch.FloatTensor(data_height, \
                                             int(np.ceil(data_height * ratio)), 3).zero_()
            padding_data[:, :data_width, :] = data[0]
            im_info[0, 1] = padding_data.size(1)
        else:
            trim_size = min(data_height, data_width)
            padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
            padding_data = data[0][:trim_size, :trim_size, :]
            # gt_boxes.clamp_(0, trim_size)
            gt_boxes[:, :4].clamp_(0, trim_size)
            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size


        # check the bounding box:
        not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
        # not_keep = (gt_boxes[:,2] - gt_boxes[:,0]) < 10 
        # print(not_keep)
        # not_keep = (gt_boxes[:,2] - gt_boxes[:,0]) < torch.FloatTensor([10]) | (gt_boxes[:,3] - gt_boxes[:,1]) < torch.FloatTensor([10])

        keep = torch.nonzero(not_keep == 0).view(-1)

        gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
        if keep.numel() != 0 :
            gt_boxes = gt_boxes[keep]
            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
        else:
            num_boxes = 0

            # permute trim_data to adapt to downstream processing
        padding_data = padding_data.permute(2, 0, 1).contiguous()
        im_info = im_info.view(3)

        return padding_data, query, im_info, gt_boxes_padding, num_boxes
    else:
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)

        # gt_boxes = torch.FloatTensor([1,1,1,1,1])
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])
        choice = self.cat_list[index]

        return data, query, im_info, gt_boxes, choice

  def load_query(self, choice, id=0):
    
    if self.training:
        # Random choice query catgory image
        all_data = self._query[choice]
        data     = random.choice(all_data)
    else:
        # Take out the purpose category for testing
        catgory = self.cat_list[choice]
        # list all the candidate image 
        all_data = self._query[catgory]

        # Use image_id to determine the random seed
        # The list l is candidate sequence, which random by image_id
        random.seed(id)
        l = list(range(len(all_data)))
        random.shuffle(l)

        # choose the candidate sequence and take out the data information
        position=l[self.query_position%len(l)]
        data     = all_data[position]

    # Get image
    path       = data['image_path']
    im = imread(path)
    

    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)

    im = crop(im, data['boxes'], cfg.TRAIN.query_size)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    # im = im[:,:,::-1]
    if random.randint(0,99)/100 > 0.5 and self.training:
      im = im[:, ::-1, :]


    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.TRAIN.query_size,
                    cfg.TRAIN.MAX_SIZE)
    
    query = im_list_to_blob([im])

    return query

  def __len__(self):
    return len(self.ratio_index)
  
  def filter(self, seen):
    if seen==1:
      self.list = cfg.train_categories
      # Group number to class
      if len(self.list)==1:
        self.list = [self._classes[cat] for cat in range(1,81) if cat%4 != self.list[0]]

    elif seen==2:
      self.list = cfg.test_categories
      # Group number to class
      if len(self.list)==1:
        self.list = [self._classes[cat] for cat in range(1,81) if cat%4 == self.list[0]]
    
    elif seen==3:
      self.list = cfg.train_categories + cfg.test_categories
      # Group number to class
      if len(self.list)==2:
        self.list = [self._classes[cat] for cat in range(1,81)]

    self.list_ind = [self._classes_inv[x] for x in self.list]
  
  def probability(self):
    show_time = {}
    for i in self.list_ind:
        show_time[i] = 0
    for roi in self._roidb:
        result = Counter(roi['gt_classes'])
        for t in result:
            if t in self.list_ind:
                show_time[t] += result[t]

    for i in self.list_ind:
        show_time[i] = 1/show_time[i]

    sum_prob = sum(show_time.values())

    for i in self.list_ind:
        show_time[i] = show_time[i]/sum_prob
    
    self.show_time = show_time
