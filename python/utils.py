from __future__ import division

import numpy as np


def soft_update(target_network, source_network, tau):
    """
    Updates the parameters of the target network
    from the parameters of the source network in a soft way
    using tau as update factor:
        p_target = p_target * (1 - tau) + p_source * tau
    """
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target_network, source_network):
    """
    Updates the parameters of the target network
    form the parameters of the source network in a hard way
    (same as using soft update with tau=1):
        p_target = p_source
    """
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)


def linear_decay(epsilon_init, step, total_steps):
    """
    Decays epsilon linearly wrt to the current step.
    """
    return (1 - step / total_steps) * epsilon_init


def exp_decay(epsilon_init, step, total_steps, k=2):
    """
    Decays epsilon exponentially wrt to the current step.
    k is a hyperparameter that defines how much you want to decay.
    E.g. if k == 1, you decay until e^-1 * epsilon_init,
    if k == 2, you decay until e^-2 * epsilon_init.
    """
    return np.exp(-k * step / total_steps) * epsilon_init