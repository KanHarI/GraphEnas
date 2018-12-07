
import torch.nn as nn

class Supergraph:
    def __init__(self, sgraph_size, channels_count, activations_list):
        self.activations = dict()
        self.norms = dict()
        self.sgraph_size = sgraph_size
        self.activations_count = len(activations_list)
        for i in sgraph_size:
            self.activations[i] = []
            for activation in activations_list:
                self.activations[i].append(activation())
            self.norms[i] = nn.BatchNorm2d(channels_count, momentum=0.2)
        self._1x1convs = dict()
        for i in range(sgraph_size):
            for j in range(i+1,sgraph_size):
                if i==0:
                    self._1x1convs[i,j] = nn.Linear(3, channels_count)
                else:
                    self._1x1convs[i,j] = nn.Linear(channels_count, channels_count)

    def cuda(self):
        for i in range(self.sgraph_size):
            for j in range(activations_count):
                self.activations[i][j] = self.activations[i][j].cuda()
            self.norms[i] = self.norms[i].cuda()
        for key in self._1x1convs.keys():
            self._1x1convs[key] = self._1x1convs[key].cuda()
        return self

    def create_subgraph(self, nodes, adj):
        exec_order = []
        pointer = self.sgraph_size
