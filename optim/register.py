from .schedulers import *

def map_scheduler(
    scheduler_name,
    optimizer,
    num_warmup_steps,
    num_training_steps
):
    if scheduler_name == 'linear_warmup_linear_decay':
        return linear_warmup_linear_decay(
            optimizer,
            num_warmup_steps,
            num_training_steps
        )
    elif scheduler_name == 'linear_warmup_cosine_decay':
        return linear_warmup_cosine_decay(
            optimizer,
            num_warmup_steps,
            num_training_steps
        )
    elif scheduler_name == 'linear_warmup_pow2_decay':
        return linear_warmup_polynomial_decay(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            power=2
        )
    elif scheduler_name == 'linear_warmup_pow4_decay':
        return linear_warmup_polynomial_decay(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            power=4
        )
    elif scheduler_name == 'pow2_warmup_pow2_decay':
        return polynomial_warmup_polynomial_decay(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            power_warmup=2,
            power_decay=2
        )
    elif scheduler_name == 'pow4_warmup_pow4_decay':
        return polynomial_warmup_polynomial_decay(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            power_warmup=4,
            power_decay=4
        )
    elif scheduler_name == 'pow2_warmup_linear_decay':
        return polynomial_warmup_polynomial_decay(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            power_warmup=2,
            power_decay=1
        )