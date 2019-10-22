# One-Shot Object Detection with Co-Attention and Co-Excitation

## Introduction

![Image](images/method.png)

This project is a pure pytorch implementation of One-Shot Objection. A lot of code is modified from [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).  

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
git clone https://github.com/timy90022/One-Shot-Object-Detection.git
```


### prerequisites

* Python or 3.6
* Pytorch 1.0 

### Data Preparation

* **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, creat softlinks in the folder data/.

* **COCO**: Please also follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare the data.
e scripts provided in this repository.

### Pretrained Model

We used two pretrained models in our experiments, ResNet50. You can download these two models from:

* ResNet50: [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

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

Above, BATCH_SIZE and WORKER_NUMBER can be set adaptively according to your GPU memory size. **On NVIDIA V100 GPUs with 32G memory, it can be up to 16 batch size**.

If you have multiple (say 8) V100 GPUs, then just use them all! Try:

```
python trainval_net.py --dataset coco --net res50 \
                       --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                       --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                       --cuda --g $SPLIT --seen $SEEN --mGPUs

```




## Test

If you want to evlauate the detection performance of a pre-trained res50 model on coco test set, simply run
```
python test_net.py --dataset coco --net res50 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda --g $SPLIT
```
Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416.


## Acknowledgments

Code is modified from [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) and [AlexHex7/Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch). All credit is attributed to them.

