################################
# See more options in configs/ #
################################

############ CIFAR100 ############


# KD
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/res32x4_res8x4.yaml 
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/res32x4_shuv2.yaml 
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/res50_mv2.yaml 
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/vgg13_mv2.yaml 
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/vgg13_vgg8.yaml 
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/wrn40_2_wrn_16_2.yaml 
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/wrn40_2_wrn_40_1.yaml 

# DKD
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml 
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/res32x4_shuv2.yaml 
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/res50_mv2.yaml 
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/vgg13_mv2.yaml 
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/vgg13_vgg8.yaml 
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/wrn40_2_wrn_16_2.yaml 
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/wrn40_2_wrn_40_1.yaml 

# ER-KD
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/res32x4_res8x4.yaml --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/res32x4_shuv2.yaml --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/res50_mv2.yaml --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/vgg13_mv2.yaml --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/vgg13_vgg8.yaml --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/wrn40_2_wrn_16_2.yaml --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/wrn40_2_wrn_40_1.yaml --er

# ER-DKD
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/res32x4_shuv2.yaml --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/res50_mv2.yaml --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/vgg13_mv2.yaml --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/vgg13_vgg8.yaml --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/wrn40_2_wrn_16_2.yaml --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/wrn40_2_wrn_40_1.yaml --er

# MT-ER-DKD
#pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/res32x4_res8x4.yaml --mt --er --dt
#pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/res32x4_shuv2.yaml --mt --er --dt
#pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/res50_mv2.yaml --mt --er --dt
#pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/vgg13_mv2.yaml --mt --er --dt
#pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/vgg13_vgg8.yaml --mt --er --dt
#pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/wrn40_2_wrn_16_2.yaml --mt --er --dt
#pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/wrn40_2_wrn_40_1.yaml --mt --er --dt

# MT-ER-KD
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/res32x4_res8x4.yaml --mt --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/res32x4_shuv2.yaml --mt --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/res50_mv2.yaml --mt --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/vgg13_mv2.yaml --mt --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/vgg13_vgg8.yaml --mt --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/wrn40_2_wrn_16_2.yaml --mt --er
# pipenv run python3 tools/train.py --cfg configs/cifar100/kd/wrn40_2_wrn_40_1.yaml --mt --er

# MT-ER-DKD
pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/res32x4_res8x4.yaml --mt --er --dt --ct
pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/res32x4_shuv2.yaml --mt --er --dt --ct

pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/res50_mv2.yaml --mt --er --dt --ct 

pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/vgg13_mv2.yaml --mt --er --dt --ct

pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/vgg13_vgg8.yaml --mt --er --dt --ct

pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/wrn40_2_wrn_16_2.yaml --mt --er --dt --ct

pipenv run python3 tools/train.py --cfg configs/cifar100/mtkd/wrn40_2_wrn_40_1.yaml --mt --er --dt --ct

############ CIFAR100 ############
# ER-KD
#pipenv run python3 tools/train.py --cfg configs/cifar100/kd/res32x4_res8x4.yaml --mt --er

# ER-DKD
#pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml --mt --er

# ER-MLD
#pipenv run python3 tools/train_mld.py --cfg configs/cifar100/mld/res32x4_res8x4.yaml --er 1

############ TinyImageNet ############
# ER-KD
#python3 tools/train.py --cfg configs/tinyimagenet200/kd/res32x4_res8x4.yaml --er 1
# ER-KD Transformer Teacher
#python3 tools/train.py --cfg configs/tinyimagenet200/kd/vit_ResNet18.yaml --er 1

# ER-MLD
#python3 tools/train_mld.py --cfg configs/tinyimagenet200/mld/res32x4_res8x4.yaml --er 1
