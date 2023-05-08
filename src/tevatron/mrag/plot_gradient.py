import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def plot_grad_flow(named_parameters, path, step):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class / vanilla training loop after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters(), output_path, step)" to visualize the gradient flow
    and save a plot in the output path as a series of .png images.
    Adapted from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    '''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        p = p.to('cpu')
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=1e-8, top=20)  # zoom in on the lower gradient regions
    plt.yscale("log")
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.savefig(os.path.join(
        path, f"{step:04d}.png"), bbox_inches='tight', dpi=200)
    fig.clear()