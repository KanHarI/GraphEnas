
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import copy
import math

IMAGE_CHANNELS = 3

class Supergraph(nn.Module):
    def __init__(self, sgraph_size, channels_count, activations_list, num_halvings):
        self.activations = dict()
        self.norms = dict()
        self.sgraph_size = sgraph_size
        self.activations_count = len(activations_list)
        self.pool = nn.MaxPool2d(2,2)
        self.num_halvings = num_halvings
        self.layers_per_size = sgraph_size//num_halvings
        for i in range(1, sgraph_size):
            self.activations[i] = []
            for activation in activations_list:
                self.activations[i].append(activation(channels_count * (2**(i//self.layers_per_size))))
            self.norms[i] = nn.BatchNorm2d(channels_count, momentum=0.2)
        self.links = dict()
        for i in range(sgraph_size):
            for j in range(i+1,sgraph_size):
                # 1x1 convolutions
                self.links[i,j] = nn.Linear(channels_count * (2**(i//self.layers_per_size)), channels_count * (2**(j//self.layers_per_size)))
                for poolings in range(j//self.layers_per_size - i//self.layers_per_size):
                    self.links[i,j] = lambda inp: self.pool(self.links[i,j])
        self.img_norm = nn.BatchNorm2d(IMAGE_CHANNELS, momentum=0.05)
        self.img_extender = nn.Linear(IMAGE_CHANNELS, channels_count)

    def cuda(self):
        for i in range(self.sgraph_size):
            for j in range(activations_count):
                self.activations[i][j] = self.activations[i][j].cuda()
            self.norms[i] = self.norms[i].cuda()
        for key in self.links.keys():
            self.links[key] = self.links[key].cuda()
        self.img_extender = self.img_extender.cuda()
        return self

    def create_subgraph(self, chosen_activations, adj):
        return Subgraph(self, chosen_activations, adj)
        

class Subgraph(nn.Module):
    def __init__(self, supergraph, chosen_activations, adj):
        self.supergraph = supergraph
        self.adj = adj
        self.chosen_activations = chosen_activations
        self.incomings = dict()
        self.relevant_nodes = self.find_relevant_nodes()

    def find_relevant_nodes(self):
        relevan_nodes = set()
        for i in range(self.supergraph.sgraph_size):
            self.incomings[i] = torch.nonzero(adj[:,i]).view(-1).tolist()
        scanned = set()
        locs = list()
        locs.append(self.supergraph.sgraph_size-1)
        while len(locs) > 0:
            ptr = locs.pop()
            if ptr in scanned:
                continue
            for src in self.incomings[ptr]:
                relevan_nodes.add(src)
                locs.append(src)
            scanned.append(ptr)

    def forward(self, input_img):
        outputs = dict()
        for i in range(self.supergraph.sgraph_size):
            if i not in self.relevan_nodes:
                continue
            if i == 0:
                outputs[i] = F.relu(self.supergraph.img_extender(self.supergraph.img_norm(input_img)))
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
                        channels_count,
                        input_img.shape[2]//(2**(i//self.supergraph.layers_per_size)),
                        input_img.shape[3]//(2**(i//self.supergraph.layers_per_size)))
                l_input = self.supergraph.norms[i](l_input)
                outputs[i] = F.relu(self.supergraph.activations[i][self.chosen_activations[i]])
