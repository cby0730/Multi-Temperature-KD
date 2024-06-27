import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, kd_loss_weight, temperature, logit_stand, dt):
    # std
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    # DTKD
    
    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    loss = F.kl_div(p_s.log(), p_t, reduction="none").sum(1).mean() * temperature * temperature
    return kd_loss_weight * loss

def dkd_loss(logits_student_in, logits_teacher_in, target, tckd_loss_weight: float, nckd_loss_weight: float, temperature, logit_stand, dt):
    # std
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    '''
    # DTKD
    # teacher entropy
    _p_t = F.softmax(logits_teacher / temperature, dim=1)
    entropy_teacher = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1)
    sigmoid_entropy_teacher = torch.sigmoid(entropy_teacher)
    
    # student entropy
    _p_s = F.softmax(logits_student / temperature, dim=1)
    entropy_student = -torch.sum(_p_s * torch.log(_p_s.clamp(min=1e-10)), dim=1)
    sigmoid_entropy_student = torch.sigmoid(entropy_student)

    temperature_teacher = (1 * (sigmoid_entropy_teacher/(sigmoid_entropy_student + sigmoid_entropy_teacher))) * temperature if dt else temperature
    temperature_student = (1 * (sigmoid_entropy_student/(sigmoid_entropy_student + sigmoid_entropy_teacher))) * temperature if dt else temperature
    temperature_teacher = temperature_teacher.unsqueeze(1)
    temperature_student = temperature_student.unsqueeze(1)
    print("temperature_teacher: ", temperature_teacher)
    print("temperature_student: ", temperature_student)

    # mtkd & mtkd2是單一動態溫度，mtkd3 & mtkd 4是兩個動態溫度
    # 目前最強的是單一動態溫度

    '''
    # DTKD
    # teacher temperature
    _p_t = F.softmax(logits_teacher / temperature, dim=1)
    entropy_teacher = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1).unsqueeze(1)
    sigmoid_entropy_teacher = torch.sigmoid(entropy_teacher)
    temperature_student = (1 - (1 - sigmoid_entropy_teacher)) * temperature if dt else temperature
    temperature_teacher = (1 - (1 - sigmoid_entropy_teacher)) * temperature if dt else temperature
    

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature_student, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature_teacher, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='none').sum(1)
        * (temperature_student * temperature_teacher)
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature_teacher - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature_student - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none').sum(1)
        * (temperature_student * temperature_teacher)
    )
    return tckd_loss_weight * tckd_loss + nckd_loss_weight * nckd_loss

def contrastive_loss(logits_student, logits_teacher, target, temperature):
    student_softmax = F.softmax(logits_student / temperature, dim=1)
    teacher_softmax = F.softmax(logits_teacher / temperature, dim=1)

    student_pos_teacher_neg = (
        -F.kl_div(torch.log(student_softmax).clamp(min=1e-10), 1 - teacher_softmax, reduction='none').sum(1)
        * temperature ** 2
    )

    student_neg_teacher_pos = (
        -F.kl_div(torch.log(1 - student_softmax).clamp(min=1e-10), teacher_softmax, reduction='none').sum(1)
        * temperature ** 2
    )

    student_neg_teacher_neg = (
        F.kl_div(torch.log(1 - student_softmax).clamp(min=1e-10), 1 - teacher_softmax, reduction='none').sum(1)
        * temperature ** 2
    )

    return student_pos_teacher_neg + student_neg_teacher_pos + student_neg_teacher_neg

def cc_contrastive_loss(logits_student, logits_teacher, temperature):
    batch_size, class_num = logits_teacher.shape
    student_softmax = F.softmax(logits_student / temperature, dim=1)
    teacher_softmax = F.softmax(logits_teacher / temperature, dim=1)

    student_matrix_pos = torch.mm(student_softmax.transpose(1, 0), student_softmax)
    teacher_matrix_pos = torch.mm(teacher_softmax.transpose(1, 0), teacher_softmax)
    student_matrix_neg = 1 - torch.mm(student_softmax.transpose(1, 0), student_softmax)
    teacher_matrix_neg = 1 - torch.mm(teacher_softmax.transpose(1, 0), teacher_softmax)

    student_pos_teacher_neg = ((1 - (teacher_matrix_neg - student_matrix_pos)) ** 2).sum() / class_num
    student_neg_teacher_pos = ((1 - (student_matrix_neg - teacher_matrix_pos)) ** 2).sum() / class_num
    student_neg_teacher_neg = (((student_matrix_neg - teacher_matrix_neg)) ** 2).sum() / class_num

    return student_pos_teacher_neg + student_neg_teacher_pos + student_neg_teacher_neg

def bb_contrastive_loss(logits_student, logits_teacher, temperature):
    batch_size, class_num = logits_teacher.shape
    student_softmax = F.softmax(logits_student / temperature, dim=1)
    teacher_softmax = F.softmax(logits_teacher / temperature, dim=1)

    student_matrix_pos = torch.mm(student_softmax, student_softmax.transpose(1, 0))
    teacher_matrix_pos = torch.mm(teacher_softmax, teacher_softmax.transpose(1, 0))
    student_matrix_neg = 1 - torch.mm(student_softmax, student_softmax.transpose(1, 0))
    teacher_matrix_neg = 1 - torch.mm(teacher_softmax, teacher_softmax.transpose(1, 0))

    student_pos_teacher_neg = ((1 - (teacher_matrix_neg - student_matrix_pos)) ** 2).sum() / batch_size
    student_neg_teacher_pos = ((1 - (student_matrix_neg - teacher_matrix_pos)) ** 2).sum() / batch_size
    student_neg_teacher_neg = (((student_matrix_neg - teacher_matrix_neg)) ** 2).sum() / batch_size

    return student_pos_teacher_neg + student_neg_teacher_pos + student_neg_teacher_neg


def mtkd_loss(logits_student, logits_teacher, target, loss_weight_dict: float, temperature, multi_temperaturs: list, t, er, mt, dt, ct, bc, std, use_kd_loss=False):
    temperatures = multi_temperaturs if mt else [temperature]
    loss_list = []
    # for multi temperature
    for temperature in temperatures:
        if use_kd_loss:
            loss_value = kd_loss(logits_student, logits_teacher, loss_weight_dict["kd"], temperature, std, dt)
        else:
            loss_value = dkd_loss(logits_student, logits_teacher, target, loss_weight_dict["tckd"], loss_weight_dict["nckd"], temperature, std, dt)
        # for contrastive loss function
        if ct:
            ct_loss_value = contrastive_loss(logits_student, logits_teacher, target, temperature)
        else:
            ct_loss_value = 0
        # for class contrastive and batch contrastive loss function
        if bc:
            cc_ct_loss_value = cc_contrastive_loss(logits_student, logits_teacher, temperature)
            bb_ct_loss_value = bb_contrastive_loss(logits_student, logits_teacher, temperature)
        else:
            cc_ct_loss_value = 0
            bb_ct_loss_value = 0
        # for er loss function
        if er:
            _p_t = F.softmax(logits_teacher / temperature, dim=1)
            entropy = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1)
            loss_list.append((loss_value * entropy.unsqueeze(1) + ct_loss_value + cc_ct_loss_value + bb_ct_loss_value).mean())
        else:
            loss_list.append(loss_value + ct_loss_value + cc_ct_loss_value + bb_ct_loss_value)

    return torch.stack(loss_list).mean()
    

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class MTKD2(Distiller):

    def __init__(self, student, teacher, cfg, t, er, mt, dt, ct, bc, std):
        super(MTKD2, self).__init__(student, teacher)
        self.loss_weight_dict = {
            "ce": cfg.MTKD.LOSS.CE_WEIGHT,
            "kd": cfg.MTKD.LOSS.KD_WEIGHT,
            "dtkd": cfg.MTKD.LOSS.DTKD_WEIGHT,
            "tckd": cfg.MTKD.LOSS.TCKD_WEIGHT,
            "nckd": cfg.MTKD.LOSS.NCKD_WEIGHT,
        }
        self.temperature = cfg.MTKD.TEMPERATURE
        self.t = t
        self.er = er
        self.mt = mt
        self.dt = dt
        self.ct = ct
        self.bc = bc
        self.std = std
        self.use_kd_loss = cfg.MTKD.BASE.upper() == "KD"
        self.temperatures = nn.Parameter(torch.tensor(cfg.MTKD.INIT_TEMPERATURE, requires_grad=True))
        self.warmup = cfg.MTKD.WARMUP

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

        # losses
        loss_ce = self.loss_weight_dict["ce"] * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))

        loss_mtkd_weak = min(kwargs["epoch"] / self.warmup, 1.0) * mtkd_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.loss_weight_dict,
            self.temperature,
            self.temperatures,
            self.t,
            self.er,
            self.mt,
            self.dt,
            self.ct,
            self.bc,
            self.std,
            self.use_kd_loss
        )

        loss_mtkd_strong = min(kwargs["epoch"] / self.warmup, 1.0) * mtkd_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.loss_weight_dict,
            self.temperature,
            self.temperatures,
            self.t,
            self.er,
            self.mt,
            self.dt,
            self.ct,
            self.bc,
            self.std,
            self.use_kd_loss
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_mtkd_weak + loss_mtkd_strong,
        }

        return logits_student_weak, losses_dict
