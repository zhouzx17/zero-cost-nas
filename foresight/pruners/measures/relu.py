import torch
import numpy as np

from . import measure

@measure('relu', bn=True)
def compute_relu(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    N = inputs.shape[0]

    bs = 1
    net.count = 0

    def counting_forward_hook(module, inputs, outputs):
        net.count = net.count + 1

    for name, module in net.named_modules():
        if 'ReLU' in str(type(module)):
            module.register_forward_hook(counting_forward_hook)

    x = torch.clone(inputs)
    x = x[:bs]
    x = x.to(device)

    #inputs = inputs.to(device)

    net(x)

    return int(net.count)
