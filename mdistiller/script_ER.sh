################################
# See more options in configs/ #
################################

############ CIFAR100 ############
# ER-DKD
pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml --er

pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/res32x4_shuv2.yaml --er

pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/res50_mv2.yaml --er

pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/vgg13_mv2.yaml --er

pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/vgg13_vgg8.yaml --er

pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/wrn40_2_wrn_16_2.yaml --er

pipenv run python3 tools/train.py --cfg configs/cifar100/dkd/wrn40_2_wrn_40_1.yaml --er

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