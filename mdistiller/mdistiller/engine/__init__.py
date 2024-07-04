from .trainer import BaseTrainer, CRDTrainer, AugTrainer, AugDOTTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "MLD": AugTrainer,
    "MLDDOT": AugDOTTrainer
}
