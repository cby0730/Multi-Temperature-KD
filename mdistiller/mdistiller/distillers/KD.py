import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from ._base import Distiller


def kd_loss(logits_student, logits_teacher, temperature):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    loss = F.kl_div(p_s.log(), p_t, reduction="none").sum(1).mean() * temperature**2
    return loss

def mt_kd_loss(logits_student, logits_teacher, temperatures):
    all_loss = []
    for temperature in temperatures: 
        p_s = F.softmax(logits_student / temperature, dim=1)
        p_t = F.softmax(logits_teacher / temperature, dim=1)
        loss = F.kl_div(p_s.log(), p_t, reduction="none")* temperature**2
        all_loss.append(loss)

    return torch.stack(all_loss).mean()


def er_kd_loss(logits_student, logits_teacher, temperature, t=4):
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    loss = F.kl_div(p_s.log(), p_t, reduction="none")  

    _p_t = F.softmax(logits_teacher / t, dim=1)
    entropy = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1)

    loss = (loss * entropy.unsqueeze(1)).sum(1).mean() * temperature**2
    return loss

def mt_er_kd_loss(logits_student, logits_teacher, temperatures, t=4):
    all_loss = []
    for temperature in temperatures: 
        p_s = F.softmax(logits_student / temperature, dim=1)
        p_t = F.softmax(logits_teacher / temperature, dim=1)
        loss = F.kl_div(p_s.log(), p_t, reduction="none")

        _p_t = F.softmax(logits_teacher / t, dim=1)
        entropy = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1)

        loss = (loss * entropy.unsqueeze(1)).sum(1).mean() * temperature**2
        all_loss.append(loss)

    return torch.stack(all_loss).mean()


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg, t, er, mt):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.t = t
        self.er = er
        self.mt = mt
        initial_temperatures = torch.tensor([4.0, 3.0, 2.0, 1.0], requires_grad=True)
        self.temperatures = nn.Parameter(initial_temperatures)
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.tea_name = cfg.DISTILLER.TEACHER

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)

        with torch.no_grad():
            if self.tea_name in ['deit', 'vit', 'swin']:
                image = transforms.Resize(384, interpolation=InterpolationMode.BICUBIC)(image)
                logits_teacher = self.teacher(image)
            else:
                logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        if self.er:
            if self.mt:
                loss_kd = self.kd_loss_weight * mt_er_kd_loss(
                    logits_student, logits_teacher, self.temperatures
                )
            else:
                loss_kd = self.kd_loss_weight * er_kd_loss(
                    logits_student, logits_teacher, self.temperature
                )
        else:
            if self.mt:
                loss_kd = self.kd_loss_weight * mt_kd_loss(
                    logits_student, logits_teacher, self.temperatures
                )
            else:
                loss_kd = self.kd_loss_weight * kd_loss(
                    logits_student, logits_teacher, self.temperature
                )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict