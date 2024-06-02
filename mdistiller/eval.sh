python3 tools/eval.py -m resnet8x4 -c /root/Work/MTKD/mdistiller/output/cifar100_MTKD/entropy,dkd,res32x4,res8x4/student_best

python3 tools/eval.py -m ShuffleV2 -c /root/Work/MTKD/mdistiller/output/cifar100_MTKD/entropy,dkd,res32x4,shuv2/student_best

python3 tools/eval.py -m MobileNetV2 -c /root/Work/MTKD/mdistiller/output/cifar100_MTKD/entropy,dkd,res50,mv2/student_best 

python3 tools/eval.py -m MobileNetV2 -c /root/Work/MTKD/mdistiller/output/cifar100_MTKD/entropy,dkd,vgg13,mv2/student_best

python3 tools/eval.py -m vgg8 -c /root/Work/MTKD/mdistiller/output/cifar100_MTKD/entropy,dkd,vgg13,vgg8/student_best 

python3 tools/eval.py -m wrn_40_1 -c /root/Work/MTKD/mdistiller/output/cifar100_baselines/entropy,dkd,wrn_40_2,wrn_40_1/student_best

python3 tools/eval.py -m wrn_16_2 -c /root/Work/MTKD/mdistiller/output/cifar100_MTKD/entropy,dkd,wrn_40_2,wrn_16_2/student_best