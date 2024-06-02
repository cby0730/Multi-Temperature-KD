# Multi-Temperature Knowledge Distillation (MTKD)

This repository is an improvement based on the papers [Decouple Knowledge Distillation](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.pdf) and [Knowledge From the Dark Side: Entropy-Reweighted Knowledge Distillation for Balanced Knowledge Transfer](https://arxiv.org/pdf/2311.13621).

By providing **multiple temperatures**, the student model can learn a wider variety of outputs from the teacher model.

## Environments:

- Python 3.10
- PyTorch 2.3.0
- CUDA 12.1
- RTX 8000 GPU 

## Result
Result on the CIFAR-100 validation.
| Experiments | Top-1 (%) | Top-5 (%) |
| -------- | ----- | ----- |
| Original ER-KD experiment  | 75.01%  | 93.66%  |
| Original ER-DKD experiment    | 76.40%  | 93.87%  |
| ER-DKD + Multi-Temperature (4, 3, 2, 1)    | 77.02 %  | 94.06 %  |