import torch
from torch.optim.lr_scheduler import StepLR, LambdaLR

def get_bert_linear_warmup(warmup_steps):
    def bert_linear_decay(step):
        if step <= warmup_steps:
            return step / warmup_steps
        else:
            return 1

    return bert_linear_decay

def get_bert_linear_warmup_decay(warmup_steps, total_steps):
    def bert_linear_decay(step):
        if step <= warmup_steps:
            return step / warmup_steps
        else:
            r = 1 - ((step-warmup_steps) / (total_steps-warmup_steps))
            return r

    return bert_linear_decay 

def make_optimizer(cfg, model):
    params = []

    if cfg.LANGUAGE.TRAINING == "finetuning":
    
        transformer_params = {
            "params": [],
            "lr": float(cfg.SOLVER.LANGUAGE_LR),
        }
        regular_params_skip = []

        transformer_params["params"].extend(model.language_model.parameters())
        regular_params_skip.append("language_model")

        regular_params = {
            "params": [
                param
                for name, param in model.named_parameters()
                if not any([skip in name for skip in regular_params_skip])
            ],
        }

        params = [transformer_params, regular_params]
        lr = cfg.SOLVER.BASE_LR
    
    else:   
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.TYPE == "ADAM":
        optimizer = torch.optim.Adam(params, lr, eps=cfg.SOLVER.EPSILON, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.TYPE == "ADAMW":
        optimizer = torch.optim.AdamW(params, lr, eps=cfg.SOLVER.EPSILON, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    
    return optimizer

def make_scheduler(cfg, optimizer, total_batches):

    if cfg.LANGUAGE.TRAINING == "finetuning":
        num_training_steps = cfg.EPOCHS * total_batches
        warmup_steps = int(cfg.SOLVER.WARMUP_RATE * num_training_steps)
        print("Warm-up steps: {}".format(warmup_steps))
        bert_linear_decay = get_bert_linear_warmup_decay(
            warmup_steps, num_training_steps
        )

        scheduler = LambdaLR(optimizer, [bert_linear_decay, lambda epoch: 1])
        interval = "step"
        
        # per batch
        # num_training_steps = cfg.EPOCHS * total_samples
        # warm_up_ratio = 0.1
        # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio*num_training_steps, num_training_steps=num_training_steps)
        # per epoch
        ## scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    else:
        if cfg.SOLVER.SCHEDULER == "EPOCH_DECAY":
            scheduler = StepLR(
                optimizer,
                step_size=cfg.SOLVER.SCH_STEP_SIZE,
                gamma=cfg.SOLVER.SCH_GAMMA,
            )
            interval = "epoch"
        else:
            raise NotImplementedError

    return scheduler, interval
