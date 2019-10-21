# One-Shot Object Detection with Co-Attention and Co-Excitation

## Introduction

This project is a pure pytorch implementation of One-Shot Objection.  
Code for reproducing results in the following paper:

![Image](images/method.pdf)

[**One-Shot Object Detection with Co-Attention and Co-Excitation**]()  
Ting-I Hsieh, Yi-Chen Lo, Hwann-Tzong Chen, Tyng-Luh Liu.  
Neural Information Processing Systems (NeurIPS), 2019

### What we are doing and going to do

- [x] Support tensorboardX.
- [x] Support pytorch-1.0 (this branch).
- [ ] Upload the ImageNet pre-trained model.
- [ ] Provide pre-trained model.
- [ ] Train PASCAL_VOC datasets


## Preparation


First of all, clone the code
```
git clone https://github.com/jwyang/faster-rcnn.pytorch.git
```

Then, create a folder:
```
cd faster-rcnn.pytorch && mkdir data
```

### prerequisites

* Python or 3.6
* Pytorch 1.0 

### Data Preparation

* **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, creat softlinks in the folder data/.

* **COCO**: Please also follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare the data.
e scripts provided in this repository.

### Pretrained Model

We used two pretrained models in our experiments, VGG and ResNet101. You can download these two models from:

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and put them into the data/pretrained_model/.

### Compilation

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
python setup.py build develop
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**

## Train

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.

In coco dataset, we split it to 4 group. It will train and test different category. Just to adjust "*--g*".

If you want to train part of dataset, try to modify "*--seen*". When training, use 1 to see train_categories.  When testing, use 2 to see test_categories. If you want to see both, use 3 to seen all categories.

To train a model with res50 on coco, simply run:

```
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset coco --net res50 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda --g $SPLIT --seen $SEEN
```

Above, BATCH_SIZE and WORKER_NUMBER can be set adaptively according to your GPU memory size. **On eight NVIDIA V100 GPUs with 32G memory in parallel , it can be up to 128**.

If you have multiple (say 8) Titan Xp GPUs, then just use them all! Try:

```
python trainval_net.py --dataset coco --net vgg16 \
                       --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   		--lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                  		--cuda --g $SPLIT --seen $SEEN --mGPUs

```




## Test

If you want to evlauate the detection performance of a pre-trained vgg16 model on pascal_voc test set, simply run
```
python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
```
Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416.


## Authorship

This project is equally contributed by [Jianwei Yang](https://github.com/jwyang) and [Jiasen Lu](https://github.com/jiasenlu), and many others (thanks to them!).

## Citation

    @article{jjfaster2rcnn,
        Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
        Title = {A Faster Pytorch Implementation of Faster R-CNN},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2017}
    }

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
