
import math

import torch
import torch.nn.functional as F


LN_2 = math.log(2)
def fuzzy_relu(tensor):
    # A modified softplus that admits a d(activation)/dt (t=0) = 1
    # and activation(0) = 0.
    # An activation of relu is problematic in this case as frequent
    # architecture changes causes lots of dead neurons...
    return torch.log(1 + torch.exp(-2*torch.abs(tensor))) + F.relu(2*tensor) - LN_2


class FuzzyRelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        return fuzzy_relu(t)

