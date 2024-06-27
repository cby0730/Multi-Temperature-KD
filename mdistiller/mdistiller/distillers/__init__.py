from ._base import Vanilla
from .KD import KD
from .MLD import MLD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .DIST import DIST
from .MTKD import MTKD
from .MLD2 import MLD2
from .MTKD2 import MTKD2

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "MLD": MLD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
    "DIST": DIST,
    "MTKD": MTKD,
    "MLD2": MLD2,
    "MTKD2": MTKD2
}
