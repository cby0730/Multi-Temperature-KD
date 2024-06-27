from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
from .loss import CrossEntropyLabelSmooth

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand, reduce=True):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd

def er_kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    # Calculate entropy for teacher's predicted probabilities
    _p_t = F.softmax(logits_teacher_in / temperature, dim=1)
    entropy = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1)

    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)

    # Use entropy values as re-weighting coefficients
    loss = F.kl_div(p_s.log(), p_t, reduction="none") * entropy.unsqueeze(1)  
    
    return loss.sum(1).mean() * temperature**2

def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_conf(x, y, lam, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = lam.reshape(-1,1,1,1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class MLD2(Distiller):
    def __init__(self, student, teacher, cfg, t, er, std):
        super(MLD2, self).__init__(student, teacher)
        self.temperatures = nn.Parameter(torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True))
        self.t = t
        self.er = er
        self.std = std
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

        batch_size, class_num = logits_student_strong.shape

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
        
        if self.er == True:
            loss_kd_weak = self.kd_loss_weight * sum(
                (er_kd_loss(logits_student_weak, logits_teacher_weak, temperature, self.std) * mask).mean()
                for temperature in self.temperatures
            )

            loss_kd_strong = self.kd_loss_weight * sum(
                (er_kd_loss(logits_student_strong, logits_teacher_strong, temperature, self.std) * mask).mean()
                for temperature in self.temperatures
            )

        else:
            loss_kd_weak = self.kd_loss_weight * sum(
                (kd_loss(logits_student_weak, logits_teacher_weak, temperature, self.std) * mask).mean()
                for temperature in self.temperatures
            )

            loss_kd_strong = self.kd_loss_weight * sum(
                (kd_loss(logits_student_strong, logits_teacher_strong, temperature, self.std) * mask).mean()
                for temperature in self.temperatures
            )


        loss_cc_weak = self.kd_loss_weight * sum(
            (cc_loss(logits_student_weak, logits_teacher_weak, temperature) * class_conf_mask).mean()
            for temperature in self.temperatures
        )

        loss_bc_weak = self.kd_loss_weight * sum(
            (bc_loss(logits_student_weak, logits_teacher_weak, temperature) * mask).mean()
            for temperature in self.temperatures
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd_weak + loss_kd_strong,
            "loss_cc": loss_cc_weak,
            "loss_bc": loss_bc_weak
        }
        return logits_student_weak, losses_dict

