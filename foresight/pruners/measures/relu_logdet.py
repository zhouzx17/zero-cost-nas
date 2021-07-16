import torch
import numpy as np


from . import measure

@measure('relu_logdet', bn=True)
def compute_relu_logdet(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    N = inputs.shape[0]

    #No need to backward
    bs = N // split_data
    net.K = np.zeros((bs, bs))

    def counting_forward_hook(module, inputs, outputs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        inputs = inputs.view(inputs.size(0), -1)
        x = (inputs > 0).float()
        K = x @ x.t()
        K2 = (1 - x) @ (1 - x.t())
        net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()

    for name, module in net.named_modules():
        if 'ReLU' in str(type(module)):
            module.register_forward_hook(counting_forward_hook)

    x = torch.clone(inputs)
    x = x[:bs]
    x = x.to(device)

    net(x)

    s, ld = np.linalg.slogdet(net.K)
    return ld
