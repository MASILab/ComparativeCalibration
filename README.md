# Comparative Analyeses of Calibration

This repository includes the code for reproduce the results of our comparative study of confidence calibration in deep learning prediction. Our code are highly motivated by https://github.com/dvlab-research/MiSLAS. 


## Examples for Running

**Stage-1**:

To train a model for Stage-1 with *CEL* on CIFAR10-LT, run:

```
python train_stage1.py --cfg ./config/cifar10_trvaltt/cifar10_imb001_stage1.yaml
```

To train a model for Stage-1 with *CEL + mixup* on CIFAR10-LT, run:

```
python train_stage1.py --cfg ./config/cifar10_trvaltt/cifar10_imb001_stage1_mixup.yaml
```

To train a model for Stage-1 with *Focal* on CIFAR10-LT, run:

```
python train_stage1.py --cfg ./config/cifar10_trvaltt/cifar10_imb001_stage1_focal.yaml
```

**Stage-2**:

To train a model for Stage-2 with *LAS*, run (be careful about the *resume* is the pretrained stage 1 model):

```
python train_stage2.py --cfg ./config/cifar10_trvaltt/cifar10_imb001_stage2.yaml

```

To train a model for Stage-2 with *temperature scaling*:

```
python temperature_scaling.py --cfg ./cconfig/cifar10_trvaltt/cifar10_imb001_stage1.yaml  resume '/media/gaor2/Elements/saved_file/Calibration/MiSLAS/imb001/stage1_imb001/ckps/model_best.pth.tar'
```





