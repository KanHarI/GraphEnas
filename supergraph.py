
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fuzzy_relu import fuzzy_relu


class SoftRelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        return fuzzy_relu(t)


class Supergraph(nn.Module):
    def __init__(self, sgraph_size, channels_count, activations_list, layers_between_halvings, inp_channels):
        super().__init__()
        self.activations = dict()
        self.norms = dict()
        self.channels_count = channels_count
        self.sgraph_size = sgraph_size
        self.activations_count = len(activations_list)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.layers_between_halvings = layers_between_halvings
        for i in range(1, sgraph_size):
            self.activations[i] = []
            for activation in activations_list:
                self.activations[i].append(activation(channels_count * (2**(i//self.layers_between_halvings))))
            self.norms[i] = nn.BatchNorm2d(channels_count * (2**(i//self.layers_between_halvings)), momentum=0.2)
        self.links = dict()
        for i in range(sgraph_size-1):
            for j in range(i+1,sgraph_size):
                # 1x1 convolutions
                self.links[i,j] = nn.Sequential(nn.Conv2d(
                                    channels_count * (2**(i//self.layers_between_halvings)),
                                    channels_count * (2**(j//self.layers_between_halvings)),
                                    1))
                for poolings in range(j//self.layers_between_halvings - i//self.layers_between_halvings):
                    self.links[i,j].add_module(str(poolings+1), self.pool)
        self.img_norm = nn.BatchNorm2d(inp_channels, momentum=0.05)
        self.img_extender = nn.Conv2d(inp_channels, channels_count, 1)

    def cuda(self):
        for i in range(1, self.sgraph_size):
            for j in range(self.activations_count):
                self.activations[i][j] = self.activations[i][j].cuda()
            self.norms[i] = self.norms[i].cuda()
        for key in self.links.keys():
            self.links[key] = self.links[key].cuda()
        self.img_extender = self.img_extender.cuda()
        self.img_norm = self.img_norm.cuda()
        return self

    def create_subgraph(self, chosen_activations, adj):
        return Subgraph(self, chosen_activations, adj)


class Subgraph(nn.Module):
    def __init__(self, supergraph, chosen_activations, adj):
        super().__init__()
        self.supergraph = supergraph
        self.adj = adj
        self.chosen_activations = chosen_activations
        self.incomings = dict()
        self.relevant_nodes = self.find_relevant_nodes()

    def find_relevant_nodes(self):
        relevant_nodes = set()
        for i in range(self.supergraph.sgraph_size):
            self.incomings[i] = torch.nonzero(self.adj[:,i]).view(-1).tolist()
        scanned = set()
        locs = list()
        locs.append(self.supergraph.sgraph_size-1)
        relevant_nodes.add(self.supergraph.sgraph_size-1)
        while len(locs) > 0:
            ptr = locs.pop()
            if ptr in scanned:
                continue
            for src in self.incomings[ptr]:
                relevant_nodes.add(src)
                locs.append(src)
            scanned.add(ptr)
        return relevant_nodes

    def forward(self, input_img):
        outputs = dict()
        for i in range(self.supergraph.sgraph_size):
            if i not in self.relevant_nodes:
                continue
            if i == 0:
                outputs[i] = self.supergraph.img_extender(self.supergraph.img_norm(input_img))
            else:
                l_input = None
                for src in self.incomings[i]:
                    if l_input is None:
                        l_input = self.supergraph.links[src,i](outputs[src])
                    else:
                        l_input += self.supergraph.links[src,i](outputs[src])
                if l_input is None:
                    # Create zero'd input manually...
                    # These layers are not removed to allow the network to 
                    # keep biases originated from these unconnected layers
                    l_input = torch.zeros(
                        input_img.shape[0],
                        self.supergraph.channels_count * (2**(i//self.supergraph.layers_between_halvings)),
                        input_img.shape[2]//(2**(i//self.supergraph.layers_between_halvings)),
                        input_img.shape[3]//(2**(i//self.supergraph.layers_between_halvings)))
                    if torch.cuda.is_available():
                        l_input = l_input.cuda()
                l_input = self.supergraph.norms[i](l_input)
                l_active = self.supergraph.activations[i][self.chosen_activations[i]] if i < self.supergraph.sgraph_size-1 else lambda x: x
                outputs[i] = fuzzy_relu(l_active(l_input))

        return outputs[self.supergraph.sgraph_size-1]
