from modeling.excl import ExCL
from modeling.tmlga import TMLGA
from modeling.dori import DORi

def build(cfg):

    if cfg.MODEL == "TMLGA":
        return TMLGA(cfg)
    elif cfg.MODEL == "EXCL":
        return ExCL(cfg)
    elif cfg.MODEL == "DORI":
        return DORi(cfg)
    else:
        raise RuntimeError("TVG Model not available: {}".format(cfg.MODEL))