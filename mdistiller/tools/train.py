import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict, tinyimagenet200_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset, get_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg, seed_everything
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict

from timm.models import create_model

def main(cfg, resume, opts, args):
    seed_everything()
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        if args.er == 1:
            experiment_name = 'entropy,' + cfg.EXPERIMENT.TAG
        else:
            experiment_name = cfg.EXPERIMENT.TAG
        if args.mt == 1:
            experiment_name = 'mt,' + experiment_name
        if args.dt == 1:
            experiment_name = 'dt,' + experiment_name
        if args.ct == 1:
            experiment_name = 'ct,' + experiment_name
        if args.bc == 1:
            experiment_name = 'bc,' + experiment_name
        if args.std == 1:
            experiment_name = 'std,' + experiment_name
        if args.t != 4:
            experiment_name += ',t=' + str(args.t)

    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    if cfg.DISTILLER.TYPE in ['MLD', 'MLD2', 'MTKD2']:
        train_loader, val_loader, num_data, num_classes = get_dataset_strong(cfg)
    else:
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        elif cfg.DATASET.TYPE.startswith("tinyimagenet200"):
            model_student = tinyimagenet200_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        elif cfg.DATASET.TYPE.startswith("tinyimagenet200"):
            if cfg.DISTILLER.TEACHER == 'deit':
                model_teacher = create_model('deit_base_distilled_patch16_384', pretrained=False, drop_path_rate=0.1)
                for param in model_teacher.parameters():
                    param.requires_grad = False
                model_teacher.reset_classifier(num_classes=200)
                model_teacher.load_state_dict(torch.load('../download_ckpts/tinyimagenet_teachers/deit_base_distilled_384.pth')['model_state_dict'])
                model_student = tinyimagenet200_model_dict[cfg.DISTILLER.STUDENT][0](
                    num_classes=num_classes
                )
            elif cfg.DISTILLER.TEACHER == 'vit':
                model_teacher = create_model('vit_large_patch16_384', pretrained=False, drop_path_rate=0.1)
                for param in model_teacher.parameters():
                    param.requires_grad = False
                model_teacher.reset_classifier(num_classes=200)
                model_teacher.load_state_dict(torch.load('../download_ckpts/tinyimagenet_teachers/vit_large_384.pth')['model_state_dict'])
                model_student = tinyimagenet200_model_dict[cfg.DISTILLER.STUDENT][0](
                    num_classes=num_classes
                )
            elif cfg.DISTILLER.TEACHER == 'swin':
                model_teacher = create_model('swin_large_patch4_window12_384', pretrained=False, drop_path_rate=0.1)
                for param in model_teacher.parameters():
                    param.requires_grad = False
                model_teacher.reset_classifier(num_classes=200)
                model_teacher.load_state_dict(torch.load('../download_ckpts/tinyimagenet_teachers/swin_large_384.pth')['model_state_dict'])
                model_student = tinyimagenet200_model_dict[cfg.DISTILLER.STUDENT][0](
                    num_classes=num_classes
                )
            else:
                net, pretrain_model_path = tinyimagenet200_model_dict[cfg.DISTILLER.TEACHER]
                assert (
                    pretrain_model_path is not None
                ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
                model_teacher = net(num_classes=num_classes)
                model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
                model_student = tinyimagenet200_model_dict[cfg.DISTILLER.STUDENT][0](
                    num_classes=num_classes
                )
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )

        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        elif cfg.DISTILLER.TYPE == "MLD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, args.t, args.er
            )
        elif cfg.DISTILLER.TYPE == "MLD2":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, args.t, args.er, args.std
        )
        elif cfg.DISTILLER.TYPE == "MTKD2":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, args.t, args.er, args.mt, args.dt, args.ct, args.bc, args.std
        )
        elif cfg.DISTILLER.TYPE == "MTKD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, args.t, args.er, args.mt, args.dt, args.ct, args.bc, args.std
            )
        elif cfg.DISTILLER.TYPE in ["KD", "DKD"]:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, args.t, args.er, args.mt
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="", help="Path to the configuration file")
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint")
    parser.add_argument("--t", type=float, default=4, help="T for ER-KD")
    parser.add_argument("--mt", action="store_true", help="Enable Multi-Temperature")
    parser.add_argument("--er", action="store_true", help="Enable Entropy Reweighted")
    parser.add_argument("--dt", action="store_true", help="Enable Dynamic Temperature")
    parser.add_argument("--ct", action="store_true", help="Enable Contrastive KD")
    parser.add_argument("--bc", action="store_true", help="Enable batch and class contrastive")
    parser.add_argument("--std", action="store_true", help="Enable normalize")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Additional options")

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts, args)
