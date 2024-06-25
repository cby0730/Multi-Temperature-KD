import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, kd_loss_weight, temperature, logit_stand=True):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    p_s = F.softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    loss = F.kl_div(p_s.log(), p_t, reduction="none").sum(1).mean() * temperature**2
    return kd_loss_weight * loss

def dkd_loss(logits_student_in, logits_teacher_in, target, tckd_loss_weight: float, nckd_loss_weight: float, temperature, logit_stand=True):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return tckd_loss_weight * tckd_loss + nckd_loss_weight * nckd_loss

def dtkd_loss(logits_student, logits_teacher, dtkd_loss_weight: float, referance_temperature):
    #DTKD
    logits_student_max, _ = logits_student.max(dim=1, keepdim=True)
    logits_teacher_max, _ = logits_teacher.max(dim=1, keepdim=True)
    student_temperatur = 2 * logits_student_max / (logits_teacher_max + logits_student_max) * referance_temperature 
    teacher_temperatur = 2 * logits_teacher_max / (logits_teacher_max + logits_student_max) * referance_temperature
    
    dtkd_loss = nn.KLDivLoss(reduction='none')(
        F.log_softmax(logits_student / student_temperatur, dim=1),
        F.softmax(logits_teacher / teacher_temperatur, dim=1)
    ) 
    dtkd_loss_value = (dtkd_loss.sum(1, keepdim=True) * student_temperatur * teacher_temperatur).mean()

    return dtkd_loss_weight * dtkd_loss_value

def contrastive_loss(logits_student, logits_teacher, target, temperature):
    student_softmax = F.softmax(logits_student / temperature, dim=1)
    teacher_softmax = F.softmax(logits_teacher / temperature, dim=1)

    student_pos_teacher_neg = (
        -F.kl_div(student_softmax, 1 - teacher_softmax, reduction='none')
        * temperature ** 2
        / target.shape[0]
    )

    student_neg_teacher_pos = (
        -F.kl_div(1 - student_softmax, teacher_softmax, reduction='none')
        * temperature ** 2
        / target.shape[0]
    )

    student_neg_teacher_neg = (
        F.kl_div(1 - student_softmax, 1 - teacher_softmax, reduction='none')
        * temperature ** 2
        / target.shape[0]
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

def cc_loss(logits_student, logits_teacher, temperature):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    
    consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    
    consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    return consistency_loss


def mtkd_loss(logits_student, logits_teacher, target, loss_weight_dict: float, temperature, multi_temperaturs: list, t, er, mt, dt, ct, bc, std, use_kd_loss=False):
    temperatures = multi_temperaturs if mt else [temperature]
    loss_list = []
    # for multi temperature
    for temperature in temperatures:
        if use_kd_loss:
            loss_value = kd_loss(logits_student, logits_teacher, loss_weight_dict["kd"], temperature, std)
        else:
            loss_value = dkd_loss(logits_student, logits_teacher, target, loss_weight_dict["tckd"], loss_weight_dict["nckd"], temperature, std)
        
        # for dynamic temperature loss function
        if dt:
            dtkd_loss_value = dtkd_loss(logits_student, logits_teacher, loss_weight_dict["dtkd"], temperature)
        else:
            dtkd_loss_value = 0
        
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
            _p_t = F.softmax(logits_teacher / t, dim=1)
            entropy = -torch.sum(_p_t * torch.log(_p_t.clamp(min=1e-10)), dim=1)
            loss_list.append((loss_value * entropy.unsqueeze(1) + dtkd_loss_value + ct_loss_value + cc_ct_loss_value + bb_ct_loss_value).mean())
        else:
            loss_list.append(loss_value + dtkd_loss_value + ct_loss_value + cc_ct_loss_value + bb_ct_loss_value)

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


class MTKD(Distiller):

    def __init__(self, student, teacher, cfg, t, er, mt, dt, ct, bc, std):
        super(MTKD, self).__init__(student, teacher)
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

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.loss_weight_dict["ce"] * F.cross_entropy(logits_student, target)

        loss_mtkd = min(kwargs["epoch"] / self.warmup, 1.0) * mtkd_loss(
            logits_student,
            logits_teacher,
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
            "loss_kd": loss_mtkd,
        }

        return logits_student, losses_dict
