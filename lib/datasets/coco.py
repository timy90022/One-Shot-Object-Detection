# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.utils.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid
# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

class coco(imdb):
  def __init__(self, image_set, year):
    imdb.__init__(self, 'coco_' + year + '_' + image_set)
    # COCO specific config options
    self.config = {'use_salt': True,
                   'cleanup': True}
    # name, paths
    self._year = year
    self._image_set = image_set
    self._data_path = osp.join(cfg.DATA_DIR, 'coco')

    # load COCO API, classes, class <-> id mappings
    self._COCO = COCO(self._get_ann_file())
    cats = self._COCO.loadCats(self._COCO.getCatIds())
    # class name
    self._classes = tuple(['__background__'] + [c['name'] for c in cats])
    # class name to ind    (0~80) 0= __background__
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    # class name to cat_id (1~90) 1= person
    self._class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats],
                                               self._COCO.getCatIds())))
    # Lookup table to map from COCO category ids to our internal class
    # indices
    # 1~90 : 1~80
    self.coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                      self._class_to_ind[cls])
                                     for cls in self._classes[1:]])
    # 1~80 : 1~90
    self.coco_class_ind_to_cat_id = dict([(self._class_to_ind[cls],
                                      self._class_to_coco_cat_id[cls])
                                     for cls in self._classes[1:]])

    self._image_index = self._load_image_set_index()


    

    # Default to roidb handler
    self.set_proposal_method('gt')
    self.competition_mode(False)

    # Some image sets are "views" (i.e. subsets) into others.
    # For example, minival2014 is a random 5000 image subset of val2014.
    # This mapping tells us where the view's images and proposals come from.
    self._view_map = {
      'minival2014': 'val2014',  # 5k val2014 subset
      'valminusminival2014': 'val2014',  # val2014 \setminus minival2014
      'test-dev2015': 'test2015',
      'valminuscapval2014': 'val2014',
      'capval2014': 'val2014',
      'captest2014': 'val2014'
    }
    coco_name = image_set + year  # e.g., "val2014"
    self._data_name = (self._view_map[coco_name]
                       if coco_name in self._view_map
                       else coco_name)
    # Dataset splits that have ground-truth annotations (test splits
    # do not have gt annotations)
    self._gt_splits = ('train', 'val', 'minival')

    # set reference file
    self._reference_dir  = os.path.join(cfg.DATA_DIR, "coco_reference_image")
    self._reference_file = os.path.join(self._reference_dir, "coco_{}_e2e_mask_rcnn_R_101_FPN_1x_caffe2.pkl".format(self._data_name))
    if not os.path.exists(self._reference_file):
        print('No reference file.')
        assert False
    else:
        with open(self._reference_file, "rb") as f:
            self.reference_image = pickle.load(f)

    self.cat_data = {}

    for i in self._class_to_ind.values():
      # i = 1~80
      self.cat_data[i] = []

  def _get_ann_file(self):
    prefix = 'instances' if self._image_set.find('test') == -1 \
      else 'image_info'
    return osp.join(self._data_path, 'annotations',
                    prefix + '_' + self._image_set + self._year + '.json')

  def _load_image_set_index(self):
    """
    Load image ids.
    """
    image_ids = self._COCO.getImgIds()
    return image_ids

  def _get_widths(self):
    anns = self._COCO.loadImgs(self._image_index)
    widths = [ann['width'] for ann in anns]
    return widths

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_id_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self._image_index[i]

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    # Example image path for index=119993:
    #   images/train2014/COCO_train2014_000000119993.jpg
    if self._data_name=='train2014':
      file_name = ('COCO_' + self._data_name + '_' +
                  str(index).zfill(12) + '.jpg')
    else:
      file_name = (str(index).zfill(12) + '.jpg')
      
    image_path = osp.join(self._data_path, 'images',
                          self._data_name, file_name)
    assert osp.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    
    if osp.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        [roidb, self.cat_data] = pickle.load(fid)
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb
    

    gt_roidb = [self._load_coco_annotation(index)
                for index in self._image_index]

    with open(cache_file, 'wb') as fid:
      pickle.dump([gt_roidb,self.cat_data], fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _load_coco_annotation(self, index):
    """
    Loads COCO bounding-box instance annotations. Crowd instances are
    handled by marking their overlaps (with all categories) to -1. This
    overlap value means that crowd "instances" are excluded from training.
    """
    im_ann = self._COCO.loadImgs(index)[0]
    im_path = self.image_path_from_index(index)
    width = im_ann['width']
    height = im_ann['height']

    # Get the useful information
    reference  = self.reference_image[index]
    save_seq = reference.keys()

    annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
    objs = self._COCO.loadAnns(annIds)
    # Sanitize bboxes -- some are invalid
    valid_objs = []
    for i, obj in enumerate(objs):
      x1 = np.max((0, obj['bbox'][0]))
      y1 = np.max((0, obj['bbox'][1]))
      x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
      y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
      if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
        obj['clean_bbox'] = [x1, y1, x2, y2]
        valid_objs.append(obj)
      
        if i in save_seq:
          entry = {
               'boxes': obj['clean_bbox'],
               'image_path': im_path
               }
          
          self.cat_data[self.coco_cat_id_to_class_ind[obj['category_id']]].append(entry)
      
    objs = valid_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    

    for ix, obj in enumerate(objs):
      cls = self.coco_cat_id_to_class_ind[obj['category_id']]
      boxes[ix, :] = obj['clean_bbox']
      gt_classes[ix] = cls
      seg_areas[ix] = obj['area']
      if obj['iscrowd']:
        # Set overlap to -1 for all classes for crowd objects
        # so they will be excluded during training
        overlaps[ix, :] = -1.0
      else:
        overlaps[ix, cls] = 1.0

    ds_utils.validate_boxes(boxes, width=width, height=height)
    overlaps = scipy.sparse.csr_matrix(overlaps)
    return {'width': width,
            'height': height,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_widths(self):
    return [r['width'] for r in self.roidb]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()

    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}

      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def _get_box_file(self, index):
    # first 14 chars / first 22 chars / all chars + .mat
    # COCO_val2014_0/COCO_val2014_000000447/COCO_val2014_000000447991.mat
    file_name = ('COCO_' + self._data_name +
                 '_' + str(index).zfill(12) + '.mat')
    return osp.join(file_name[:14], file_name[:22], file_name)

  def _print_detection_eval_metrics(self, coco_eval):
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
      ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                     (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
      iou_thr = coco_eval.params.iouThrs[ind]
      assert np.isclose(iou_thr, thr)
      return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = \
      coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
           '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      # minus 1 because of __background__
      precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
      ap = np.mean(precision[precision > -1])
      print('{:.1f}'.format(100 * ap))

    print('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()

  def _do_detection_eval(self, res_file, output_dir):
    ann_type = 'bbox'

    tmp = [self.coco_cat_id_to_class_ind[i]-1 for i in self.list]

    coco_dt = self._COCO.loadRes(res_file)

    cocoEval = customCOCOeval(self._COCO, coco_dt, "bbox")
    cocoEval.params.imgIds = self._image_index
    cocoEval.evaluate()
    # print(cocoEval.ious)
    cocoEval.accumulate()
    cocoEval.summarize(class_index=tmp)


    eval_file = osp.join(output_dir, 'detection_results.pkl')
    with open(eval_file, 'wb') as fid:
      pickle.dump(cocoEval, fid, pickle.HIGHEST_PROTOCOL)
    print('Wrote COCO eval results to: {}'.format(eval_file))

  def _coco_results_one_category(self, boxes, cat_id):
    results = []
    for im_ind, index in enumerate(self.image_index):
      dets = boxes[im_ind]
      if dets == []:
        continue
      dets = np.array(dets).astype(np.float)
      scores = dets[:, -1]
      xs = dets[:, 0]
      ys = dets[:, 1]
      ws = dets[:, 2] - xs + 1
      hs = dets[:, 3] - ys + 1
      for k in range(len(dets)):
        results.extend(
          [{'image_id': index,
            'category_id': cat_id,
            'bbox': [xs[k], ys[k], ws[k], hs[k]],
            'score': scores[k]} ])
    return results

  def _write_coco_results_file(self, all_boxes, res_file):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                       self.num_classes - 1))
      coco_cat_id = self._class_to_coco_cat_id[cls]
      results.extend(self._coco_results_one_category(all_boxes[cls_ind], coco_cat_id))
    print('Writing results json to {}'.format(res_file))
    with open(res_file, 'w') as fid:
      json.dump(results, fid)

  def evaluate_detections(self, all_boxes, output_dir):


    res_file = osp.join(output_dir, ('detections_' +
                                     self._image_set +
                                     self._year +
                                     '_results'))
    if self.config['use_salt']:
      res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    self._write_coco_results_file(all_boxes, res_file)
    # Only do evaluation on non-test sets
    if self._image_set.find('test') == -1:
      self._do_detection_eval(res_file, output_dir)
    # Optionally cleanup results json file
    if self.config['cleanup']:
      os.remove(res_file)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True

  def filter(self, seen=1):

    # if want to use train_categories, seen = 1 
    # if want to use test_categories , seen = 2
    # if want to use both            , seen = 3

    if seen==1:
      self.list = cfg.train_categories
      # Group number to class
      if len(self.list)==1:
        self.list = [self.coco_class_ind_to_cat_id[cat] for cat in range(1,81) if cat%4 != self.list[0]]

    elif seen==2:
      self.list = cfg.test_categories
      # Group number to class
      if len(self.list)==1:
        self.list = [self.coco_class_ind_to_cat_id[cat] for cat in range(1,81) if cat%4 == self.list[0]]
    
    elif seen==3:
      self.list = cfg.train_categories + cfg.test_categories
      # Group number to class
      if len(self.list)==2:
        self.list = [self.coco_class_ind_to_cat_id[cat] for cat in range(1,81)]

    # Transfer categories id to class indices
    self.inverse_list = [self.coco_cat_id_to_class_ind[i] for i in self.list ]

    # Which index need to be remove
    all_index = list(range(len(self._image_index)))

    for index, info in enumerate(self.roidb):
      for cat in info['gt_classes']:
        if self.coco_class_ind_to_cat_id[cat] in self.list:
            all_index.remove(index)
            break

    # Remove index from the end to start
    all_index.reverse()
    for index in all_index:
      self._image_index.pop(index)
      self.roidb.pop(index)
  

class customCOCOeval(COCOeval):
    
    def summarize(self, class_index=None, verbose=1):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if not class_index is None:
                    s = s[:,:,class_index,aind,mind]
                else:
                    s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if not class_index is None:
                    s = s[:,class_index,aind,mind]
                else:
                    s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            if verbose > 0:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self, cass_index=None):
        self.summarize(class_index)