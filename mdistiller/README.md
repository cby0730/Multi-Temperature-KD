
# Classification and detection code for ER-KD, ER-DKD, and ER-MLD

This code is based on [DKD(mdistiller)](<https://github.com/megvii-research/mdistiller>), [MLD](<https://github.com/Jin-Ying/Multi-Level-Logit-Distillation.git>) and [ER-KD](https://arxiv.org/pdf/2311.13621).

## Installation

To install the package, run:

```
python3 setup.py develop
```

## Training on CIFAR100 / TinyImageNet

```
sh script_ER.sh
```

## Current status for MTKD

Only Finish on ER-DKD + Mulit-Temperature.
