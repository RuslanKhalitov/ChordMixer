import math
from torch.optim.lr_scheduler import LambdaLR

def min_max_scale(min_val, max_val, value):
    return value * (max_val - min_val) + min_val

def linear_warmup_cosine_decay(
    optimizer, 
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """
    https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/optimization.py#L104
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def linear_warmup_cosine_decay_linear_plateau(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    plateau_steps: int,
    plateau_value: float = 0.05,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps - plateau_steps:
            return float(plateau_value) * ((num_training_steps - current_step) / plateau_steps)
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - plateau_steps - num_warmup_steps))
            value=0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
            return min_max_scale(
                min_val=plateau_value,
                max_val=1.,
                value=value
            )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def linear_warmup_cosine_decay_constant_plateau(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    plateau_steps: int,
    plateau_value: float = 0.05,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):  
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps - plateau_steps:
            return float(plateau_value)
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - plateau_steps - num_warmup_steps))
            value=0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
            return min_max_scale(
                min_val=plateau_value,
                max_val=1.,
                value=value
            )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def linear_warmup_linear_decay(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    last_epoch=-1
):
    """
    https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/optimization.py#L75
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def linear_warmup_polynomial_decay(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    lr_end=1e-8,
    power=1.0,
    last_epoch=-1
):
    """
    https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/optimization.py#L173
    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def polynomial_warmup_polynomial_decay(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    lr_end=1e-8,
    power_warmup=1,
    power_decay=1,
    last_epoch=-1
):
    """
    Adapted from
    https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/optimization.py#L173
    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            lr_range = lr_init
            pct_remaining = current_step / num_warmup_steps
            decay = lr_range * pct_remaining**power_warmup
            return decay / lr_init
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power_decay + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init
        
    return LambdaLR(optimizer, lr_lambda, last_epoch)